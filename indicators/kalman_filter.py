#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit

from .indicator import Indicator
from .price_source import PriceSource

# Numbaで使用するためのヘルパー関数 (kalman_hull_supertrend.pyからコピー)
@njit(fastmath=True)
def nz(value: float, replacement: float = 0.0) -> float:
    """
    NaNの場合はreplacementを返す（Numba互換）
    """
    if np.isnan(value):
        return replacement
    return value

# --- Kalman Filter Core Logic (Numba JIT) ---
@njit(fastmath=True)
def kalman_filter_step(
    price: float,
    measurement_noise: float,
    process_noise: float,
    state_estimate: np.ndarray, # size N
    error_covariance: np.ndarray # size N
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Kalman Filterの1ステップ計算 (Numba JIT)
    状態ベクトルと誤差共分散を更新し、フィルタリングされた価格を返す。
    """
    N = len(state_estimate)
    predicted_state_estimate = np.copy(state_estimate) # Simplified prediction
    predicted_error_covariance = error_covariance + process_noise

    kalman_gain = np.zeros(N, dtype=np.float64)
    new_state_estimate = np.zeros(N, dtype=np.float64)
    new_error_covariance = np.zeros(N, dtype=np.float64)

    for i in range(N):
        denominator = predicted_error_covariance[i] + measurement_noise
        if denominator == 0:
             kg = 0.0
        else:
             kg = predicted_error_covariance[i] / denominator

        kalman_gain[i] = kg
        new_state_estimate[i] = predicted_state_estimate[i] + kg * (price - predicted_state_estimate[i])
        new_error_covariance[i] = (1.0 - kg) * predicted_error_covariance[i]

    state_estimate[:] = new_state_estimate
    error_covariance[:] = new_error_covariance

    return state_estimate[0]

@njit(fastmath=True)
def apply_kalman_filter_numba(
    prices: np.ndarray,
    measurement_noise: float,
    process_noise: float,
    n_states: int
) -> np.ndarray:
    """
    価格系列全体にKalman Filterを適用する (Numba JIT)
    """
    length = len(prices)
    filtered_prices = np.full(length, np.nan, dtype=np.float64)

    if measurement_noise <= 0:
        measurement_noise = 1e-9 # 非常に小さい正の値

    state_estimate = np.full(n_states, np.nan, dtype=np.float64)
    error_covariance = np.full(n_states, 100.0, dtype=np.float64)

    initial_price_found = False
    initial_price_index = -1
    for i in range(length):
        if not np.isnan(prices[i]):
            if not initial_price_found:
                state_estimate[:] = prices[i]
                error_covariance[:] = 1.0
                initial_price_found = True
                initial_price_index = i

    if not initial_price_found:
        return filtered_prices

    start_filter_idx = initial_price_index

    for i in range(start_filter_idx, length):
         if not np.isnan(prices[i]):
              # 状態と共分散はkalman_filter_step内で更新される
              filtered_prices[i] = kalman_filter_step(
                  prices[i], measurement_noise, process_noise,
                  state_estimate, error_covariance
              )
         else:
              # NaNの場合、フィルタリング結果もNaNとする
              filtered_prices[i] = np.nan
              # 状態は維持される（次の有効な価格で使用）

    return filtered_prices

# --- Kalman Filter Indicator Class ---
class KalmanFilter(Indicator):
    """
    Kalman Filter インジケーター

    価格系列にKalman Filterを適用して平滑化する。
    """

    def __init__(
        self,
        price_source: str = 'close',
        measurement_noise: float = 1.0, # デフォルト値を設定 (KHMAとは異なる用途も想定)
        process_noise: float = 0.01,
        n_states: int = 5,
        warmup_periods: Optional[int] = None
    ):
        """
        コンストラクタ

        Args:
            price_source (str): フィルタリング対象の価格 ('close', 'hlc3', etc.)
            measurement_noise (float): Kalman Filterの測定ノイズ。0より大きい値。
            process_noise (float): Kalman Filterのプロセスノイズ。
            n_states (int): Kalman Filterの状態数。
            warmup_periods (Optional[int]): 計算に必要な最小期間。Noneの場合、n_statesから推定。
        """
        if measurement_noise <= 0:
             raise ValueError("measurement_noiseは0より大きい必要があります。")
        if n_states <= 0:
             raise ValueError("n_statesは0より大きい必要があります。")

        self.price_source = price_source
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.n_states = n_states

        # ウォームアップ期間 (状態数に依存)
        estimated_warmup = n_states + 1 # 最小限必要な期間
        self._warmup_periods = warmup_periods if warmup_periods is not None else estimated_warmup

        name = f"KalmanFilter(src={price_source}, measN={measurement_noise}, procN={process_noise}, states={n_states})"
        self.last_valid_state = None

        # 親クラスの初期化 (Indicator)
        super().__init__(name)

        self._data_hash: Optional[int] = None # キャッシュ用

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する。"""
        param_hash = hash((
            self.price_source,
            self.measurement_noise,
            self.process_noise,
            self.n_states
        ))

        # PriceSource.calculate_source を使用して関連データを取得しハッシュ化
        try:
            src_prices = PriceSource.calculate_source(data, self.price_source)
            if src_prices is not None:
                 data_hash_val = hash(src_prices.tobytes())
            else:
                 # ソースが取得できない場合は元のデータでフォールバック
                 if isinstance(data, pd.DataFrame):
                     data_hash_val = hash(data.values.tobytes())
                 elif isinstance(data, np.ndarray):
                     data_hash_val = hash(data.tobytes())
                 else:
                     data_hash_val = hash(str(data))
        except Exception as e:
            self.logger.warning(f"データハッシュ計算中にエラー: {e}. PriceSource.calculate_source が失敗した可能性あり。フォールバック使用。", exc_info=False)
            if isinstance(data, pd.DataFrame):
                 data_hash_val = hash(data.values.tobytes())
            elif isinstance(data, np.ndarray):
                 data_hash_val = hash(data.tobytes())
            else:
                 data_hash_val = hash(str(data))

        return f"{data_hash_val}_{param_hash}"

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Kalman Filterを適用した価格系列を計算する

        Args:
            data: 価格データ（DataFrameまたはNumPy配列）

        Returns:
            np.ndarray: フィルタリングされた価格系列
        """
        try:
            current_hash = self._get_data_hash(data)
            if self._values is not None and current_hash == self._data_hash:
                return self._values # キャッシュを返す

            prices = PriceSource.calculate_source(data, self.price_source)
            if prices is None or len(prices) == 0:
                self.logger.warning(f"価格ソース '{self.price_source}' の取得に失敗しました。")
                return np.full(len(data), np.nan)

            if len(prices) < self._warmup_periods:
                self.logger.warning(f"データ長 ({len(prices)}) がウォームアップ期間 ({self._warmup_periods}) より短いため、NaN配列を返します。")
                self._values = np.full(len(prices), np.nan)
                self._data_hash = current_hash
                return self._values

            # Numba関数を呼び出し
            filtered_prices = apply_kalman_filter_numba(
                prices,
                self.measurement_noise,
                self.process_noise,
                self.n_states
            )

            self._values = filtered_prices
            self._data_hash = current_hash
            return self._values

        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"{self.name} 計算中にエラー: {error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            self._values = np.full(data_len, np.nan)
            self._data_hash = None # エラー時はキャッシュ無効化
            return self._values

    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._data_hash = None
