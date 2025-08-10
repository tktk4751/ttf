#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from numba import jit, float64

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult
from ..kalman.unified_kalman import UnifiedKalman


@jit(nopython=True)
def calculate_autocorrelation_periodogram_numba(
    price: np.ndarray,
    avg_length: float = 3.0,
    cycle_part: float = 0.5,
    max_output: int = 34,
    min_output: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    自己相関ペリオドグラムドミナントサイクル検出のNumba実装
    
    Args:
        price: 価格データの配列
        avg_length: 平均化長
        cycle_part: サイクル部分の倍率
        max_output: 最大出力値
        min_output: 最小出力値
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ドミナントサイクル, 生の周期, 平滑化周期)
    """
    n = len(price)
    pi = 2 * np.arcsin(1.0)
    
    # 初期化
    hp = np.zeros(n)
    filt = np.zeros(n)
    dom_cycle = np.zeros(n)
    dominant_cycle_values = np.zeros(n)
    max_power = np.zeros(n)
    
    # フィルター係数
    alpha1 = (np.cos(0.707 * 2 * pi / 48) + np.sin(0.707 * 2 * pi / 48) - 1) / np.cos(0.707 * 2 * pi / 48)
    
    # Super Smoother係数
    a1 = np.exp(-1.414 * pi / 10)
    b1 = 2 * a1 * np.cos(1.414 * pi / 10)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    for i in range(n):
        # ハイパスフィルター（48バー未満の周期成分を除去）
        if i >= 2:
            hp[i] = (1 - alpha1 / 2) * (1 - alpha1 / 2) * (price[i] - 2 * price[i-1] + price[i-2])
            if i > 2:
                hp[i] += 2 * (1 - alpha1) * hp[i-1] - (1 - alpha1) * (1 - alpha1) * hp[i-2]
        else:
            hp[i] = 0.0
        
        # Super Smootherフィルター
        if i >= 2:
            filt[i] = c1 * (hp[i] + hp[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]
        elif i >= 1:
            filt[i] = c1 * (hp[i] + hp[i-1]) / 2
        else:
            filt[i] = hp[i]
        
        # 自己相関とDFTの計算
        max_lag = min(48, i)
        if max_lag > 10:
            # 各ラグに対するピアソン相関を計算
            corr = np.zeros(49)
            
            for lag in range(max_lag + 1):
                # 平均化長を設定
                m = int(avg_length) if avg_length > 0 else lag
                m = max(1, min(m, lag + 1))
                
                if m > 0:
                    sx = 0.0
                    sy = 0.0
                    sxx = 0.0
                    syy = 0.0
                    sxy = 0.0
                    
                    # サンプルを進めてピアソン成分を合計
                    for count in range(m):
                        if i - count >= 0 and i - lag - count >= 0:
                            x = filt[i - count]
                            y = filt[i - lag - count]
                            sx += x
                            sy += y
                            sxx += x * x
                            syy += y * y
                            sxy += x * y
                    
                    # 各ラグ値に対する相関を計算
                    denominator = (m * sxx - sx * sx) * (m * syy - sy * sy)
                    if denominator > 0:
                        corr[lag] = (m * sxy - sx * sy) / np.sqrt(denominator)
            
            # DFT計算
            cosine_part = np.zeros(49)
            sine_part = np.zeros(49)
            sq_sum = np.zeros(49)
            
            for period in range(10, 49):
                for n_val in range(3, 49):
                    if n_val < len(corr):
                        angle = 370.0 / 180.0 * pi * n_val / period
                        cosine_part[period] += corr[n_val] * np.cos(angle)
                        sine_part[period] += corr[n_val] * np.sin(angle)
                
                sq_sum[period] = cosine_part[period] ** 2 + sine_part[period] ** 2
            
            # パワースペクトラムの平滑化
            r1 = np.zeros(49)
            for period in range(10, 49):
                r1[period] = 0.2 * (sq_sum[period] ** 2) + 0.8 * r1[period]
            
            # 正規化のための最大パワーレベルを見つける
            if i > 0:
                max_power[i] = 0.995 * max_power[i-1]
            else:
                max_power[i] = 0.0
            
            for period in range(10, 49):
                if r1[period] > max_power[i]:
                    max_power[i] = r1[period]
            
            # パワーの正規化
            pwr = np.zeros(49)
            if max_power[i] > 0:
                for period in range(3, 49):
                    pwr[period] = r1[period] / max_power[i]
            
            # スペクトラムの重心を使用してドミナントサイクルを計算
            spx = 0.0
            sp = 0.0
            for period in range(10, 49):
                if pwr[period] >= 0.5:
                    spx += period * pwr[period]
                    sp += pwr[period]
            
            if sp != 0:
                dominant_cycle_values[i] = spx / sp
            else:
                dominant_cycle_values[i] = dominant_cycle_values[i-1] if i > 0 else 15.0
            
            # 制限を適用
            if dominant_cycle_values[i] < 10:
                dominant_cycle_values[i] = 10
            elif dominant_cycle_values[i] > 48:
                dominant_cycle_values[i] = 48
            
            # 最終出力計算
            dc_output = int(np.ceil(dominant_cycle_values[i] * cycle_part))
            if dc_output > max_output:
                dom_cycle[i] = max_output
            elif dc_output < min_output:
                dom_cycle[i] = min_output
            else:
                dom_cycle[i] = dc_output
        else:
            # 初期値
            dominant_cycle_values[i] = 15.0
            dom_cycle[i] = int(np.ceil(15.0 * cycle_part))
    
    # 生の周期値と平滑化周期値
    raw_period = np.copy(dominant_cycle_values)
    smooth_period = np.copy(dom_cycle)
    
    return dom_cycle, raw_period, smooth_period


class EhlersAutocorrelationPeriodogram(EhlersDominantCycle):
    """
    エーラーズの自己相関ペリオドグラムドミナントサイクル検出器
    
    自己相関ペリオドグラムの構築は、最小3バーの平均化を使用した自己相関関数から始まります。
    自己相関結果の離散フーリエ変換（DFT）を使用してサイクル情報を抽出します。
    このアプローチは他のスペクトル推定技術に対して少なくとも4つの明確な利点があります。
    
    特徴:
    - 自己相関関数による周期性の検出
    - 離散フーリエ変換による周波数分析
    - パワースペクトラムの重心計算による正確なサイクル特定
    - 適応的な平均化による安定性向上
    """
    
    # 許可されるソースタイプのリスト
    SRC_TYPES = ['close', 'hlc3', 'hl2', 'ohlc4']
    
    def __init__(
        self,
        avg_length: float = 3.0,
        cycle_part: float = 0.5,
        max_output: int = 34,
        min_output: int = 1,
        src_type: str = 'close',
        use_kalman_filter: bool = False,
        kalman_filter_type: str = 'adaptive'
    ):
        """
        コンストラクタ
        
        Args:
            avg_length: 平均化長（デフォルト: 3.0）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 1）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値（デフォルト）
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
            use_kalman_filter: Kalmanフィルターを使用するか（デフォルト: False）
            kalman_filter_type: Kalmanフィルターのタイプ（デフォルト: 'adaptive'）
        """
        super().__init__(
            f"EhlersAutocorrelationPeriodogram({avg_length}, {cycle_part})",
            cycle_part,
            48,  # max_cycle
            10,  # min_cycle
            max_output,
            min_output
        )
        self.avg_length = avg_length
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        
        # ソースタイプを保存
        self.src_type = src_type.lower()
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # Kalmanフィルターの初期化
        self.kalman_filter = None
        if self.use_kalman_filter:
            self.kalman_filter = UnifiedKalman(
                filter_type=kalman_filter_type,
                src_type='close'  # 内部的にはcloseで使用
            )
    
    def calculate_source_values(self, data: Union[pd.DataFrame, np.ndarray], src_type: str) -> np.ndarray:
        """
        指定されたソースタイプに基づいて価格データを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
        
        Returns:
            計算された価格データの配列
        """
        if isinstance(data, pd.DataFrame):
            if src_type == 'close':
                if 'close' in data.columns:
                    return data['close'].values
                elif 'Close' in data.columns:
                    return data['Close'].values
                else:
                    raise ValueError("DataFrameには'close'または'Close'カラムが必要です")
            
            elif src_type == 'hlc3':
                if all(col in data.columns for col in ['high', 'low', 'close']):
                    return (data['high'] + data['low'] + data['close']).values / 3
                elif all(col in data.columns for col in ['High', 'Low', 'Close']):
                    return (data['High'] + data['Low'] + data['Close']).values / 3
                else:
                    raise ValueError("hlc3の計算には'high', 'low', 'close'カラムが必要です")
            
            elif src_type == 'hl2':
                if all(col in data.columns for col in ['high', 'low']):
                    return (data['high'] + data['low']).values / 2
                elif all(col in data.columns for col in ['High', 'Low']):
                    return (data['High'] + data['Low']).values / 2
                else:
                    raise ValueError("hl2の計算には'high', 'low'カラムが必要です")
            
            elif src_type == 'ohlc4':
                if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    return (data['open'] + data['high'] + data['low'] + data['close']).values / 4
                elif all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                    return (data['Open'] + data['High'] + data['Low'] + data['Close']).values / 4
                else:
                    raise ValueError("ohlc4の計算には'open', 'high', 'low', 'close'カラムが必要です")
            
            else:
                raise ValueError(f"無効なソースタイプです: {src_type}")
        
        else:  # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                if src_type == 'close':
                    return data[:, 3]  # close
                elif src_type == 'hlc3':
                    return (data[:, 1] + data[:, 2] + data[:, 3]) / 3  # high, low, close
                elif src_type == 'hl2':
                    return (data[:, 1] + data[:, 2]) / 2  # high, low
                elif src_type == 'ohlc4':
                    return (data[:, 0] + data[:, 1] + data[:, 2] + data[:, 3]) / 4  # open, high, low, close
                else:
                    raise ValueError(f"無効なソースタイプです: {src_type}")
            else:
                return data  # 1次元配列として扱う
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        自己相関ペリオドグラムアルゴリズムを使用してドミナントサイクルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、指定したソースタイプに必要なカラムが必要
        
        Returns:
            ドミナントサイクルの値
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # ソースタイプに基づいて価格データを取得
            price = self.calculate_source_values(data, self.src_type)
            
            # Numba関数を使用してドミナントサイクルを計算
            dom_cycle, raw_period, smooth_period = calculate_autocorrelation_periodogram_numba(
                price,
                self.avg_length,
                self.cycle_part,
                self.max_output,
                self.min_output
            )
            
            # 結果を保存
            self._result = DominantCycleResult(
                values=dom_cycle,
                raw_period=raw_period,
                smooth_period=smooth_period
            )
            
            self._values = dom_cycle
            return dom_cycle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"EhlersAutocorrelationPeriodogram計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([]) 