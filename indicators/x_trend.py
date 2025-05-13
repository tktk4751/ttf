#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange, vectorize

from .indicator import Indicator
from .c_atr import CATR # Use CATR instead of ATR
from .price_source import PriceSource
from .kalman_filter import apply_kalman_filter_numba
from .cycle_efficiency_ratio import CycleEfficiencyRatio # For dynamic multiplier
from .z_channel import calculate_dynamic_multiplier_vec, calculate_dynamic_max_multiplier, calculate_dynamic_min_multiplier # Reuse dynamic multiplier logic

# Numbaで使用するためのヘルパー関数
@njit(fastmath=True)
def nz(value: float, replacement: float = 0.0) -> float:
    """
    NaNの場合はreplacementを返す（Numba互換）
    """
    if np.isnan(value):
        return replacement
    return value

@dataclass
class XTrendResult:
    """XTrendの計算結果"""
    supertrend: np.ndarray  # Supertrendラインの値
    direction: np.ndarray   # トレンド方向（1=上昇トレンド、-1=下降トレンド）
    kalman_hma: np.ndarray  # Kalman Hull Moving Averageの値
    upper_band: np.ndarray  # 計算された上限バンド
    lower_band: np.ndarray  # 計算された下限バンド
    c_atr: np.ndarray       # CATR値 (金額ベース)
    cer: np.ndarray         # Cycle Efficiency Ratio
    dynamic_multiplier: np.ndarray # 動的ATR乗数

# --- Kalman Hull Moving Average (KHMA) ---
# (kalman_hull_supertrend.pyから流用)
@njit(fastmath=True)
def calculate_khma(
    prices: np.ndarray,
    length: float, # measurementNoiseに対応
    process_noise: float,
    n_states: int = 5
) -> np.ndarray:
    """
    Kalman Hull Moving Average (KHMA) を計算する (Numba JIT)
    """
    if length <= 0:
        return np.full_like(prices, np.nan)

    length_half = length / 2.0
    if length_half <= 0:
         length_half = 1e-9

    sqrt_length = np.sqrt(max(0.0, length))
    sqrt_length_rounded = np.round(sqrt_length)
    if sqrt_length_rounded <= 0:
         sqrt_length_rounded = 1e-9

    kalman_len_half = apply_kalman_filter_numba(prices, length_half, process_noise, n_states)
    kalman_len = apply_kalman_filter_numba(prices, length, process_noise, n_states)
    intermediate_series = 2.0 * kalman_len_half - kalman_len
    khma = apply_kalman_filter_numba(intermediate_series, sqrt_length_rounded, process_noise, n_states)
    return khma

# --- Supertrend with Dynamic Multiplier ---
@njit(fastmath=True) # Removed parallel=True as it's sequential
def calculate_supertrend_bands_trend_dynamic(
    close: np.ndarray,
    src: np.ndarray, # 通常はKHMA
    atr: np.ndarray, # 金額ベースのCATR
    dynamic_multiplier: np.ndarray # 動的乗数の配列
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Supertrendのバンドとトレンド方向を動的乗数で計算する (Numba JIT)
    """
    length = len(close)
    upper_band = np.full(length, np.nan, dtype=np.float64)
    lower_band = np.full(length, np.nan, dtype=np.float64)
    supertrend_line = np.full(length, np.nan, dtype=np.float64)
    direction = np.zeros(length, dtype=np.int8) # 1 for long, -1 for short

    # 最初の有効なインデックスを見つける (src, atr, dynamic_multiplier が必要)
    start_idx = -1
    for i in range(length):
        if not np.isnan(close[i]) and not np.isnan(src[i]) and \
           not np.isnan(atr[i]) and atr[i] > 1e-9 and \
           not np.isnan(dynamic_multiplier[i]):
            start_idx = i
            break

    if start_idx == -1:
        return upper_band, lower_band, supertrend_line, direction

    # 初期値の設定
    initial_factor = dynamic_multiplier[start_idx]
    initial_up = src[start_idx] + initial_factor * atr[start_idx]
    initial_lo = src[start_idx] - initial_factor * atr[start_idx]
    upper_band[start_idx] = initial_up
    lower_band[start_idx] = initial_lo

    if close[start_idx] > initial_up:
        direction[start_idx] = 1
        supertrend_line[start_idx] = initial_lo
    elif close[start_idx] < initial_lo:
        direction[start_idx] = -1
        supertrend_line[start_idx] = initial_up
    else:
        direction[start_idx] = 1 # デフォルトは上昇
        supertrend_line[start_idx] = initial_lo

    # メインループ (逐次処理)
    for i in range(start_idx + 1, length):
        # 必要な値がNaNでないかチェック
        if np.isnan(close[i]) or np.isnan(src[i]) or \
           np.isnan(atr[i]) or atr[i] <= 1e-9 or \
           np.isnan(dynamic_multiplier[i]):
            # データ欠損時は前の値を引き継ぐ
            upper_band[i] = upper_band[i-1]
            lower_band[i] = lower_band[i-1]
            supertrend_line[i] = supertrend_line[i-1]
            direction[i] = direction[i-1]
            continue

        # 現在のバンド計算 (動的乗数を使用)
        current_factor = dynamic_multiplier[i]
        up = src[i] + current_factor * atr[i]
        lo = src[i] - current_factor * atr[i]

        # 前のバンドと終値を取得
        prev_lower_band = nz(lower_band[i-1])
        prev_upper_band = nz(upper_band[i-1])
        prev_close = nz(close[i-1])

        # バンドの更新ロジック
        current_lower_band = lo
        if not (lo > prev_lower_band or prev_close < prev_lower_band):
             current_lower_band = prev_lower_band
        lower_band[i] = current_lower_band

        current_upper_band = up
        if not (up < prev_upper_band or prev_close > prev_upper_band):
             current_upper_band = prev_upper_band
        upper_band[i] = current_upper_band

        # トレンド方向の決定ロジック
        current_direction = direction[i-1]
        if direction[i-1] == 1:
            if close[i] < lower_band[i]:
                current_direction = -1
        elif direction[i-1] == -1:
            if close[i] > upper_band[i]:
                current_direction = 1
        direction[i] = current_direction

        # Supertrendラインの決定
        if current_direction == 1:
            supertrend_line[i] = lower_band[i]
        else:
            supertrend_line[i] = upper_band[i]

    return upper_band, lower_band, supertrend_line, direction


class XTrend(Indicator):
    """
    XTrend インジケーター

    Kalman Hull Supertrendをベースに、以下の機能を追加:
    - ATRの代わりにCATR (Cycle Average True Range) を使用
    - Cycle Efficiency Ratio (CER) に基づいてATR乗数を動的に調整
    """

    def __init__(
        self,
        # KHMA Parameters
        price_source: str = 'close',
        measurement_noise: float = 3.0,
        process_noise: float = 0.01,
        kalman_n_states: int = 5,
        # CATR Parameters (pass through to CATR instance)
        catr_detector_type: str = 'hody',
        catr_cycle_part: float = 0.5,
        catr_lp_period: int = 5,
        catr_hp_period: int = 55,
        catr_max_cycle: int = 55,
        catr_min_cycle: int = 5,
        catr_max_output: int = 34,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'alma',
        # CER Parameters (pass through to CER instance)
        cer_detector_type: str = 'phac_e', # Use a different detector for CER often
        cer_lp_period: int = 5,
        cer_hp_period: int = 55,
        cer_cycle_part: float = 0.382,
        cer_max_cycle: int = 62,
        cer_min_cycle: int = 5,
        cer_max_output: int = 34,
        cer_min_output: int = 5,
        # Dynamic Multiplier Parameters (from ZChannel)
        max_max_multiplier: float = 8.0,
        min_max_multiplier: float = 3.0,
        max_min_multiplier: float = 1.5,
        min_min_multiplier: float = 0.5,
        # General
        warmup_periods: Optional[int] = None
    ):
        """
        コンストラクタ
        """
        # Parameter validation
        if measurement_noise <= 0: raise ValueError("measurement_noise > 0")
        if kalman_n_states <= 0: raise ValueError("kalman_n_states > 0")
        # Add validation for other params if needed

        self._price_source_key = price_source
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.kalman_n_states = kalman_n_states

        self.max_max_multiplier = max_max_multiplier
        self.min_max_multiplier = min_max_multiplier
        self.max_min_multiplier = max_min_multiplier
        self.min_min_multiplier = min_min_multiplier

        # Instantiate sub-indicators
        self.c_atr_calculator = CATR(
            detector_type=catr_detector_type,
            cycle_part=catr_cycle_part,
            lp_period=catr_lp_period,
            hp_period=catr_hp_period,
            max_cycle=catr_max_cycle,
            min_cycle=catr_min_cycle,
            max_output=catr_max_output,
            min_output=catr_min_output,
            smoother_type=catr_smoother_type
        )
        self.cer_calculator = CycleEfficiencyRatio(
            detector_type=cer_detector_type,
            lp_period=cer_lp_period,
            hp_period=cer_hp_period,
            cycle_part=cer_cycle_part,
            max_cycle=cer_max_cycle,
            min_cycle=cer_min_cycle,
            max_output=cer_max_output,
            min_output=cer_min_output,
            src_type=price_source # Use same price source for CER
        )
        self.price_source_extractor = PriceSource()

        # Estimate warmup periods (consider KHMA and CATR/CER warmups)
        # CATR warmup depends on its internal DC detector's max_cycle
        # CER warmup depends on its internal DC detector's max_cycle
        # KHMA warmup depends on kalman_n_states
        estimated_warmup = max(catr_max_cycle, cer_max_cycle, kalman_n_states) + 10 # Add buffer
        self._warmup_periods = warmup_periods if warmup_periods is not None else estimated_warmup

        # Indicator name
        name = (f"XTrend(src={price_source}, measN={measurement_noise}, procN={process_noise}, "
                f"kStates={kalman_n_states}, catr={catr_detector_type}, cer={cer_detector_type}, "
                f"dynMult=[{min_min_multiplier}-{max_min_multiplier},{min_max_multiplier}-{max_max_multiplier}])")
        super().__init__(name)

        # Result cache
        self._result: Optional[XTrendResult] = None
        self._data_hash: Optional[int] = None # Use int hash for simplicity here

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> int:
        """データのハッシュ値を計算してキャッシュに使用する (簡易版)"""
        # This is a simplified hash function for demonstration.
        # A more robust hash would consider all relevant parameters like in ZChannel.
        if isinstance(data, pd.DataFrame):
            cols_to_hash = ['open', 'high', 'low', 'close']
            relevant_cols = [col for col in cols_to_hash if col in data.columns]
            if not relevant_cols: raise ValueError("No OHLC columns found")
            data_tuple = tuple(map(tuple, (data[col].values for col in relevant_cols)))
        else: # Assuming NumPy array
             if data.ndim != 2 or data.shape[1] < 4:
                 raise ValueError("NumPy array needs OHLC columns")
             data_tuple = tuple(map(tuple, data.T[:4])) # Hash first 4 columns (OHLC)

        # Include relevant parameters in hash
        param_tuple = (
            self._price_source_key, self.measurement_noise, self.process_noise, self.kalman_n_states,
            # Add CATR, CER, and dynamic multiplier params
            self.c_atr_calculator.detector_type, self.c_atr_calculator.max_cycle, self.c_atr_calculator.smoother_type,
            self.cer_calculator.detector_type, self.cer_calculator.max_cycle,
            self.max_max_multiplier, self.min_max_multiplier, self.max_min_multiplier, self.min_min_multiplier
        )
        return hash((data_tuple, param_tuple))

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        XTrendを計算する

        Args:
            data: 価格データ（DataFrameまたはNumPy配列）

        Returns:
            np.ndarray: トレンド方向の配列（1=上昇、-1=下降、0=計算不可）
        """
        try:
            current_hash = self._get_data_hash(data)
            if self._result is not None and current_hash == self._data_hash:
                return self._result.direction

            # Calculate all price sources using the extractor's calculate method
            self.price_source_extractor.calculate(data)
            # Get the required price sources as numpy arrays
            src_prices = self.price_source_extractor.get_source(self._price_source_key)
            close_prices = self.price_source_extractor.get_source('close')

            if src_prices is None or close_prices is None or len(src_prices) == 0:
                 raise ValueError("価格ソースの取得に失敗しました")

            if len(src_prices) < self._warmup_periods:
                self.logger.warning(f"データ長 ({len(src_prices)}) < ウォームアップ期間 ({self._warmup_periods})")
                nan_array = np.full(len(src_prices), np.nan)
                int_zero_array = np.zeros(len(src_prices), dtype=np.int8)
                self._result = XTrendResult(
                    supertrend=nan_array, direction=int_zero_array, kalman_hma=nan_array,
                    upper_band=nan_array, lower_band=nan_array, c_atr=nan_array,
                    cer=nan_array, dynamic_multiplier=nan_array
                )
                self._values = self._result.direction
                self._data_hash = current_hash
                return self._values

            # 1. Calculate CER
            cer_values = self.cer_calculator.calculate(data)
            if cer_values is None or len(cer_values) == 0:
                 raise ValueError("CER calculation failed")
            cer_np = np.asarray(cer_values, dtype=np.float64)


            # 2. Calculate CATR (requires CER)
            # CATR calculate method needs external_er
            _ = self.c_atr_calculator.calculate(data, external_er=cer_np) # Calculate first
            c_atr_absolute = self.c_atr_calculator.get_absolute_atr() # Get absolute values
            if c_atr_absolute is None or len(c_atr_absolute) == 0:
                 raise ValueError("CATR calculation failed")
            c_atr_np = np.asarray(c_atr_absolute, dtype=np.float64)


            # 3. Calculate Kalman Hull Moving Average (KHMA)
            khma_values = calculate_khma(
                src_prices, self.measurement_noise, self.process_noise, self.kalman_n_states
            )
            khma_np = np.asarray(khma_values, dtype=np.float64)

            # 4. Calculate Dynamic Multiplier based on CER
            # Reuse functions from z_channel
            max_mult_values = calculate_dynamic_max_multiplier(
                cer_np, self.max_max_multiplier, self.min_max_multiplier
            )
            min_mult_values = calculate_dynamic_min_multiplier(
                cer_np, self.max_min_multiplier, self.min_min_multiplier
            )
            dynamic_multiplier = calculate_dynamic_multiplier_vec(
                cer_np, max_mult_values, min_mult_values
            )
            dynamic_multiplier_np = np.asarray(dynamic_multiplier, dtype=np.float64)

            # 5. Calculate Supertrend using KHMA, CATR (absolute), and Dynamic Multiplier
            upper_band, lower_band, supertrend_line, direction = calculate_supertrend_bands_trend_dynamic(
                close_prices, khma_np, c_atr_np, dynamic_multiplier_np
            )

            # Store results
            self._result = XTrendResult(
                supertrend=supertrend_line,
                direction=direction,
                kalman_hma=khma_np,
                upper_band=upper_band,
                lower_band=lower_band,
                c_atr=c_atr_np, # Store absolute CATR
                cer=cer_np,
                dynamic_multiplier=dynamic_multiplier_np
            )
            self._values = self._result.direction
            self._data_hash = current_hash

            return self._values

        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"{self.name} 計算中にエラー: {error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            nan_array = np.full(data_len, np.nan)
            int_zero_array = np.zeros(data_len, dtype=np.int8)
            if self._result is None:
                 self._result = XTrendResult(
                     supertrend=nan_array, direction=int_zero_array, kalman_hma=nan_array,
                     upper_band=nan_array, lower_band=nan_array, c_atr=nan_array,
                     cer=nan_array, dynamic_multiplier=nan_array
                 )
                 self._values = int_zero_array
            else:
                 # Preserve other results if possible, reset direction
                 self._result.direction[:] = 0
                 self._values = self._result.direction
            self._data_hash = None
            return self._values

    # --- Getter methods for results ---
    def get_supertrend(self) -> Optional[np.ndarray]:
        return self._result.supertrend if self._result else None

    def get_direction(self) -> Optional[np.ndarray]:
        return self._result.direction if self._result else None

    def get_kalman_hma(self) -> Optional[np.ndarray]:
        return self._result.kalman_hma if self._result else None

    def get_bands(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._result:
            return self._result.upper_band, self._result.lower_band
        return None, None

    def get_c_atr(self) -> Optional[np.ndarray]:
        """金額ベースのCATR値を取得"""
        return self._result.c_atr if self._result else None

    def get_cer(self) -> Optional[np.ndarray]:
        """Cycle Efficiency Ratioの値を取得"""
        return self._result.cer if self._result else None

    def get_dynamic_multiplier(self) -> Optional[np.ndarray]:
        """動的ATR乗数の値を取得"""
        return self._result.dynamic_multiplier if self._result else None

    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._result = None
        self._data_hash = None
        self.c_atr_calculator.reset()
        self.cer_calculator.reset()
        # price_source_extractorは状態を持たない想定
