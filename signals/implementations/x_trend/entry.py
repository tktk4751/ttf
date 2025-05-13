#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.x_trend import XTrend # Import the new XTrend indicator

@njit(fastmath=True, parallel=True)
def calculate_xtrend_signals(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    XTrendの方向転換に基づいてエントリーおよびエグジットシグナルを計算する (Numba JIT)

    Args:
        direction: XTrendの方向配列 (1=上昇, -1=下降)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (エントリーシグナル, エグジットシグナル) のタプル
        エントリーシグナル: 1 = ロングエントリー, -1 = ショートエントリー, 0 = なし
        エグジットシグナル: 1 = ロング決済, -1 = ショート決済, 0 = なし
    """
    length = len(direction)
    entry_signals = np.zeros(length, dtype=np.int8)
    exit_signals = np.zeros(length, dtype=np.int8)

    # NaNでない最初のインデックスを見つける
    start_idx = 0
    for i in range(length):
        if not np.isnan(direction[i]):
            start_idx = i
            break

    # 並列処理でシグナルを計算
    for i in prange(start_idx + 1, length):
        # 現在または前の方向がNaNの場合はスキップ
        if np.isnan(direction[i]) or np.isnan(direction[i-1]):
            continue

        prev_dir = direction[i-1]
        curr_dir = direction[i]

        # ロングエントリー: -1から1へ転換
        if prev_dir == -1 and curr_dir == 1:
            entry_signals[i] = 1
        # ショートエントリー: 1から-1へ転換
        elif prev_dir == 1 and curr_dir == -1:
            entry_signals[i] = -1

        # ロング決済: 1から-1へ転換
        if prev_dir == 1 and curr_dir == -1:
            exit_signals[i] = 1 # ロングポジションを決済
        # ショート決済: -1から1へ転換
        elif prev_dir == -1 and curr_dir == 1:
            exit_signals[i] = -1 # ショートポジションを決済

    return entry_signals, exit_signals


class XTrendEntrySignal(BaseSignal, IEntrySignal):
    """
    XTrendインジケーターの方向転換に基づくエントリーシグナル

    シグナル条件:
    - ロングエントリー (1): XTrend方向が-1から1に転換
    - ショートエントリー (-1): XTrend方向が1から-1に転換
    - ロング決済 (1): XTrend方向が1から-1に転換
    - ショート決済 (-1): XTrend方向が-1から1に転換
    """

    def __init__(
        self,
        # XTrend Parameters (pass through)
        price_source: str = 'close',
        measurement_noise: float = 3.0,
        process_noise: float = 0.01,
        kalman_n_states: int = 5,
        catr_detector_type: str = 'hody',
        catr_cycle_part: float = 0.5,
        catr_lp_period: int = 5,
        catr_hp_period: int = 55,
        catr_max_cycle: int = 55,
        catr_min_cycle: int = 5,
        catr_max_output: int = 34,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'alma',
        cer_detector_type: str = 'phac_e',
        cer_lp_period: int = 5,
        cer_hp_period: int = 55,
        cer_cycle_part: float = 0.382,
        cer_max_cycle: int = 62,
        cer_min_cycle: int = 5,
        cer_max_output: int = 34,
        cer_min_output: int = 5,
        max_max_multiplier: float = 8.0,
        min_max_multiplier: float = 3.0,
        max_min_multiplier: float = 1.5,
        min_min_multiplier: float = 0.5,
        warmup_periods: Optional[int] = None
    ):
        """
        初期化
        """
        # Generate a unique name based on key parameters
        signal_name = (f"XTrendEntry(src={price_source},kStates={kalman_n_states},"
                       f"catr={catr_detector_type},cer={cer_detector_type})")
        super().__init__(signal_name)

        # Store parameters for hashing and potential reuse, excluding 'self', '__class__', and 'signal_name'
        self._params = {k: v for k, v in locals().items() if k not in ('self', '__class__', 'signal_name')}

        # Initialize the XTrend indicator with passed parameters
        self.x_trend_indicator = XTrend(**self._params)

        # Cache for signals
        self._entry_signals_cache = {}
        self._exit_signals_cache = {}
        self._last_data_hash = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> int:
        """データのハッシュ値を計算"""
        if isinstance(data, pd.DataFrame):
            # Use relevant columns for hashing
            cols = ['open', 'high', 'low', 'close']
            relevant_cols = [col for col in cols if col in data.columns]
            if not relevant_cols: raise ValueError("OHLC columns needed")
            data_bytes = b"".join(data[col].values.tobytes() for col in relevant_cols)
        elif isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] < 4:
                 raise ValueError("NumPy array needs OHLC columns")
            data_bytes = data[:, :4].tobytes() # Hash OHLC columns
        else:
            raise TypeError("Data must be pd.DataFrame or np.ndarray")

        # Combine data hash with parameter hash
        param_hash = hash(tuple(sorted(self._params.items())))
        return hash((data_bytes, param_hash))

    def _calculate_signals_if_needed(self, data: Union[pd.DataFrame, np.ndarray]):
        """必要に応じてシグナルを計算しキャッシュする"""
        data_hash = self._get_data_hash(data)
        if data_hash != self._last_data_hash:
            try:
                # Calculate XTrend direction
                direction = self.x_trend_indicator.calculate(data)
                if direction is None or len(direction) == 0:
                    raise ValueError("XTrend indicator calculation failed or returned empty.")

                direction_np = np.asarray(direction, dtype=np.float64) # Ensure float for NaN checks

                # Calculate entry and exit signals using Numba function
                entry_signals, exit_signals = calculate_xtrend_signals(direction_np)

                self._entry_signals_cache[data_hash] = entry_signals
                self._exit_signals_cache[data_hash] = exit_signals
                self._last_data_hash = data_hash

            except Exception as e:
                self.logger.error(f"Error calculating XTrend signals: {e}")
                # Cache empty signals on error to avoid repeated calculation attempts
                empty_signals = np.zeros(len(data), dtype=np.int8)
                self._entry_signals_cache[data_hash] = empty_signals
                self._exit_signals_cache[data_hash] = empty_signals
                self._last_data_hash = data_hash # Mark as calculated (even if failed)

    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する

        Args:
            data: 価格データ

        Returns:
            np.ndarray: エントリーシグナル (1: ロング, -1: ショート, 0: なし)
        """
        data_hash = self._get_data_hash(data)
        self._calculate_signals_if_needed(data)
        return self._entry_signals_cache.get(data_hash, np.zeros(len(data), dtype=np.int8))

    def get_exit_signal(self, data: Union[pd.DataFrame, np.ndarray], index: int = -1) -> int:
        """
        指定されたインデックスでのエグジットシグナルを取得する

        Args:
            data: 価格データ
            index: データのインデックス (デフォルト: -1、最新のバー)

        Returns:
            int: エグジットシグナル (1: ロング決済, -1: ショート決済, 0: なし)
        """
        data_hash = self._get_data_hash(data)
        self._calculate_signals_if_needed(data)
        signals = self._exit_signals_cache.get(data_hash, np.zeros(len(data), dtype=np.int8))

        if index == -1:
            index = len(signals) - 1

        if 0 <= index < len(signals):
            return signals[index]
        else:
            self.logger.warning(f"Index {index} out of bounds for exit signals.")
            return 0 # Return no signal if index is invalid

    def get_indicator_values(self) -> Optional[Dict[str, np.ndarray]]:
        """XTrendインジケーターの内部値を取得（デバッグ/分析用）"""
        if self.x_trend_indicator._result:
            return {
                "supertrend": self.x_trend_indicator.get_supertrend(),
                "direction": self.x_trend_indicator.get_direction(),
                "kalman_hma": self.x_trend_indicator.get_kalman_hma(),
                "upper_band": self.x_trend_indicator.get_bands()[0] if self.x_trend_indicator.get_bands() else None,
                "lower_band": self.x_trend_indicator.get_bands()[1] if self.x_trend_indicator.get_bands() else None,
                "c_atr": self.x_trend_indicator.get_c_atr(),
                "cer": self.x_trend_indicator.get_cer(),
                "dynamic_multiplier": self.x_trend_indicator.get_dynamic_multiplier(),
            }
        return None

    def reset(self) -> None:
        """シグナルの状態をリセットする"""
        super().reset()
        self.x_trend_indicator.reset()
        self._entry_signals_cache = {}
        self._exit_signals_cache = {}
        self._last_data_hash = None
