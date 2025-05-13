#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple,Optional
import numpy as np
import pandas as pd

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.x_trend.entry import XTrendEntrySignal # Import the new XTrend signal

class XTrendSignalGenerator(BaseSignalGenerator):
    """
    XTrendインジケーターに基づくシグナル生成クラス（シンプル版）

    エントリー条件:
    - ロング: XTrend方向が-1から1に転換
    - ショート: XTrend方向が1から-1に転換

    エグジット条件:
    - ロング: XTrend方向が1から-1に転換
    - ショート: XTrend方向が-1から1に転換
    """

    def __init__(
        self,
        # Pass all XTrend parameters here
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
        """初期化"""
        super().__init__("XTrendSignalGenerator")

        # Store parameters, excluding 'self' and '__class__'
        self._params = {k: v for k, v in locals().items() if k != 'self' and k != '__class__'}

        # Initialize the XTrend entry signal class
        self.x_trend_signal = XTrendEntrySignal(**self._params)

        # Cache variables
        self._data_len = 0
        self._entry_signals = None
        self._exit_signals = None # Store pre-calculated exit signals
        self._last_data_hash = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> int:
        """データのハッシュ値を計算"""
        # Reusing the hash function from XTrendEntrySignal for consistency
        return self.x_trend_signal._get_data_hash(data)

    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（必要に応じて）"""
        current_len = len(data)
        data_hash = self._get_data_hash(data)

        # Recalculate if data length or hash changes
        if self._entry_signals is None or current_len != self._data_len or data_hash != self._last_data_hash:
            try:
                # Use the internal calculation method of XTrendEntrySignal
                self.x_trend_signal._calculate_signals_if_needed(data)
                # Retrieve cached signals
                self._entry_signals = self.x_trend_signal._entry_signals_cache.get(data_hash)
                self._exit_signals = self.x_trend_signal._exit_signals_cache.get(data_hash)

                if self._entry_signals is None or self._exit_signals is None:
                     # This should ideally not happen if _calculate_signals_if_needed worked
                     raise ValueError("Signal calculation within XTrendEntrySignal failed to populate cache.")

                self._data_len = current_len
                self._last_data_hash = data_hash
            except Exception as e:
                self.logger.error(f"Error during signal calculation: {e}")
                # Set empty signals on error
                self._entry_signals = np.zeros(current_len, dtype=np.int8)
                self._exit_signals = np.zeros(current_len, dtype=np.int8)
                self._data_len = current_len
                self._last_data_hash = data_hash # Mark as calculated (even if failed)


    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得"""
        self.calculate_signals(data)
        return self._entry_signals if self._entry_signals is not None else np.zeros(len(data), dtype=np.int8)

    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成"""
        self.calculate_signals(data)

        if self._exit_signals is None:
            return False # Calculation failed

        if index == -1:
            index = self._data_len - 1

        if 0 <= index < self._data_len:
            exit_signal_at_index = self._exit_signals[index]
            if position == 1:  # Long position
                return bool(exit_signal_at_index == 1) # Exit long if exit signal is 1
            elif position == -1:  # Short position
                return bool(exit_signal_at_index == -1) # Exit short if exit signal is -1
        else:
            self.logger.warning(f"Index {index} out of bounds for exit signals.")

        return False

    def get_indicator_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Optional[Dict[str, np.ndarray]]:
        """XTrendインジケーターの内部値を取得"""
        try:
            # Ensure signals (and thus indicator values) are calculated
            if data is not None:
                 self.calculate_signals(data)
            return self.x_trend_signal.get_indicator_values()
        except Exception as e:
            self.logger.error(f"Error getting indicator values: {e}")
            return None

    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self.x_trend_signal.reset()
        self._data_len = 0
        self._entry_signals = None
        self._exit_signals = None
        self._last_data_hash = None
