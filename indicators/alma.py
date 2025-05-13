#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .price_source import PriceSource
from .kalman_filter import KalmanFilter


@jit(nopython=True, cache=True)
def calculate_alma_numba(prices: np.ndarray, period: int, offset: float, sigma: float) -> np.ndarray:
    """
    ALMAを計算する（Numba JIT、NaN対応改善）
    
    Args:
        prices: 価格の配列 (フィルタリング済みの場合あり)
        period: 期間
        offset: オフセット (0-1)。1に近いほど最新のデータを重視
        sigma: シグマ。大きいほど重みの差が大きくなる
    
    Returns:
        ALMA値の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)
    
    if period <= 0 or length == 0:
        return result
    
    # ウィンドウサイズが価格データより大きい場合は調整
    window_size = period
    
    # ウェイトの計算
    m = offset * (window_size - 1)
    s = window_size / sigma
    weights = np.zeros(window_size)
    weights_sum = 0.0
    
    for i in range(window_size):
        weight = np.exp(-((i - m) ** 2) / (2 * s * s))
        weights[i] = weight
        weights_sum += weight
    
    # 重みの正規化 (ゼロ除算防止)
    if weights_sum > 1e-9:
        weights = weights / weights_sum
    else:
        # 重み合計がゼロに近い場合 (シグマが非常に小さいなど)、均等な重みを使用
        weights = np.full(window_size, 1.0 / window_size)
    
    # ALMAの計算
    for i in range(length):
        # ウィンドウに必要なデータがあるか確認
        window_start_idx = i - window_size + 1
        if window_start_idx < 0:
            continue # 最初の期間はスキップ

        # ウィンドウ内のNaNを確認
        window_prices = prices[window_start_idx : i + 1]
        if np.any(np.isnan(window_prices)):
            continue # ウィンドウ内にNaNがあれば結果もNaN

        # ALMAの計算
        alma_value = 0.0
        for j in range(window_size):
            alma_value += window_prices[j] * weights[j]
        result[i] = alma_value
    
    return result


class ALMA(Indicator):
    """
    改良版 ALMA (Arnaud Legoux Moving Average) インジケーター
    
    特徴:
    - ノイズを低減しながら、価格変動に素早く反応する移動平均線
    - 計算に使用する価格ソースを選択可能 ('close', 'hlc3', etc.)
    - オプションで価格ソースにカルマンフィルターを適用可能
    """
    
    def __init__(self, 
                 period: int = 9, 
                 offset: float = 0.85, 
                 sigma: float = 6,
                 src_type: str = 'close',
                 use_kalman_filter: bool = False,
                 kalman_measurement_noise: float = 1.0,
                 kalman_process_noise: float = 0.01,
                 kalman_n_states: int = 5):
        """
        コンストラクタ
        
        Args:
            period: 期間
            offset: オフセット (0-1)。1に近いほど最新のデータを重視
            sigma: シグマ。大きいほど重みの差が大きくなる
            src_type: 価格ソース ('close', 'hlc3', etc.)
            use_kalman_filter: ソース価格にカルマンフィルターを適用するかどうか
            kalman_*: カルマンフィルターのパラメータ
        """
        kalman_str = f"_kalman={'Y' if use_kalman_filter else 'N'}" if use_kalman_filter else ""
        super().__init__(f"ALMA(p={period},src={src_type}{kalman_str},off={offset},sig={sigma})")
        
        self.period = period
        self.offset = offset
        self.sigma = sigma
        
        self.src_type = src_type
        self.use_kalman_filter = use_kalman_filter
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_process_noise = kalman_process_noise
        self.kalman_n_states = kalman_n_states
        
        self.price_source_extractor = PriceSource()
        self.kalman_filter = None
        if self.use_kalman_filter:
            self.kalman_filter = KalmanFilter(
                price_source=self.src_type,
                measurement_noise=self.kalman_measurement_noise,
                process_noise=self.kalman_process_noise,
                n_states=self.kalman_n_states
            )
        
        self._cache = {}
        self._result: Optional[np.ndarray] = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        # src_typeに基づいて必要なカラムを決定
        required_cols = set()
        if self.src_type == 'open':
            required_cols.add('open')
        elif self.src_type == 'high':
            required_cols.add('high')
        elif self.src_type == 'low':
            required_cols.add('low')
        elif self.src_type == 'close':
            required_cols.add('close')
        elif self.src_type == 'hl2':
            required_cols.update(['high', 'low'])
        elif self.src_type == 'hlc3':
            required_cols.update(['high', 'low', 'close'])
        elif self.src_type == 'ohlc4':
            required_cols.update(['open', 'high', 'low', 'close'])
        elif self.src_type == 'hlcc4':
            required_cols.update(['high', 'low', 'close'])
        elif self.src_type == 'weighted_close':
            required_cols.update(['high', 'low', 'close'])
        else:
            required_cols.add('close') # Default

        if isinstance(data, pd.DataFrame):
            relevant_cols = [col for col in data.columns if col.lower() in required_cols]
            present_cols = [col for col in relevant_cols if col in data.columns]
            if not present_cols:
                try:
                    shape_tuple = data.shape
                    first_row = tuple(data.iloc[0]) if len(data) > 0 else ()
                    last_row = tuple(data.iloc[-1]) if len(data) > 0 else ()
                    data_repr_tuple = (shape_tuple, first_row, last_row)
                    data_hash_val = hash(data_repr_tuple)
                except Exception:
                    data_hash_val = hash(str(data))
            else:
                data_values = data[present_cols].values
                data_hash_val = hash(data_values.tobytes())

        elif isinstance(data, np.ndarray):
            col_indices = []
            if 'open' in required_cols: col_indices.append(0)
            if 'high' in required_cols: col_indices.append(1)
            if 'low' in required_cols: col_indices.append(2)
            if 'close' in required_cols: col_indices.append(3)
            col_indices = sorted(list(set(col_indices)))
            if data.ndim == 2 and data.shape[1] > max(col_indices if col_indices else [-1]):
                data_values = data[:, col_indices]
                data_hash_val = hash(data_values.tobytes())
            else:
                data_hash_val = hash(data.tobytes())
        else:
            data_hash_val = hash(str(data))

        param_str = (
            f"p={self.period}_src={self.src_type}_off={self.offset}_sig={self.sigma}_"
            f"kalman={self.use_kalman_filter}_{self.kalman_measurement_noise}_{self.kalman_process_noise}_{self.kalman_n_states}"
        )
        return f"{data_hash_val}_{param_str}"

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ALMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）または直接価格の配列
        
        Returns:
            ALMA値の配列
        """
        try:
            # データチェック - 1次元配列が直接渡された場合はそのまま使用
            if isinstance(data, np.ndarray) and data.ndim == 1:
                src_prices = data
                data_hash = hash(data.tobytes()) # シンプルなハッシュ
                data_hash_key = f"{data_hash}_{self.period}_{self.offset}_{self.sigma}"
                
                if data_hash_key in self._cache and self._result is not None:
                    return self._result
                    
                # 直接1次元配列に対してALMAを計算
                alma_values = calculate_alma_numba(src_prices, self.period, self.offset, self.sigma)
                self._result = alma_values
                self._cache[data_hash_key] = self._result
                return self._result
            
            # 通常のハッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._cache and self._result is not None:
                return self._result

            # PriceSourceを使ってソース価格を取得
            src_prices = PriceSource.calculate_source(data, self.src_type)

            # --- Optional Kalman Filtering ---
            effective_src_prices = src_prices
            if self.use_kalman_filter and self.kalman_filter:
                filtered_prices = self.kalman_filter.calculate(data) # Pass original data to Kalman
                if filtered_prices is not None and len(filtered_prices) > 0:
                    effective_src_prices = filtered_prices
                else:
                    self.logger.warning("カルマンフィルター計算失敗、元のソース価格を使用します。")

            # データ長の検証
            data_length = len(effective_src_prices)
            if data_length == 0:
                self.logger.warning("価格データが空です。空の配列を返します。")
                self._result = np.array([])
                self._cache[data_hash] = self._result
                return self._result
            # _validate_period は Indicator 親クラスにあるか？なければ簡易チェック
            if self.period > data_length:
                self.logger.warning(f"期間 ({self.period}) がデータ長 ({data_length}) より大きいです。")
                # 計算は可能だが結果はほぼNaNになる

            # ALMAの計算（Numba版）
            alma_values = calculate_alma_numba(effective_src_prices, self.period, self.offset, self.sigma)

            self._result = alma_values
            self._cache[data_hash] = self._result
            return self._result

        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"{self.name} 計算中にエラー: {error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            self._result = None # エラー時は結果をクリア
            return np.full(data_len, np.nan)

    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}
        if self.kalman_filter and hasattr(self.kalman_filter, 'reset'):
            self.kalman_filter.reset() 