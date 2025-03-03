#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.candlestick.pinbar_entry import PinbarEntrySignal
from signals.implementations.divergence.stoch_rsi_divergence import StochRSIDivergenceSignal
from signals.implementations.divergence.rsi_divergence import RSIDivergenceSignal
from signals.implementations.divergence.roc_divergence import ROCDivergenceSignal
from signals.implementations.divergence.mfi_divergence import MFIDivergenceSignal
from signals.implementations.divergence.macd_divergence import MACDDivergenceSignal
from signals.implementations.rsi.filter import RSIFilterSignal
from signals.implementations.chop.filter import ChopFilterSignal
from signals.implementations.adx.filter import ADXFilterSignal
from signals.implementations.bollinger.entry import BollingerCounterTrendEntrySignal
from signals.implementations.rsi.entry import RSICounterTrendEntrySignal
from signals.implementations.bollinger.exit import BollingerBreakoutExitSignal
from signals.implementations.donchian.entry import DonchianBreakoutEntrySignal
from signals.implementations.alma.direction import ALMACirculationSignal


@jit(nopython=True)
def calculate_points(signals: np.ndarray, points: np.ndarray) -> np.ndarray:
    """ポイントを計算（高速化版）"""
    return np.sum(signals * points)


@jit(nopython=True)
def calculate_entry_signals(points: np.ndarray, threshold: float) -> np.ndarray:
    """エントリーシグナルを計算（高速化版）"""
    return np.where(points >= threshold, -1, 0).astype(np.int8)  # 買い版と逆にする


class SellCounterPredicterSignalGenerator(BaseSignalGenerator):
    """
    複数のシグナルをポイント制で評価するシグナル生成クラス（売り専用・高速化版）
    
    エントリー条件:
    - 各シグナルのポイントの合計がエントリーしきい値を超えた場合に売りシグナル
    
    エグジット条件:
    - 各シグナルのポイントの合計がエグジットしきい値を超えた場合に買いシグナル
    """
    
    def __init__(
        self,
        entry_threshold: float = 10.0,
        exit_threshold: float = 10.0,
        entry_points: np.ndarray = None,
        exit_points: np.ndarray = None
    ):
        """初期化"""
        super().__init__("SellCounterPredicterSignalGenerator")
        
        # しきい値の設定
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        # エントリー用シグナル生成器の初期化
        self.pinbar = PinbarEntrySignal()
        self.roc_divergence = ROCDivergenceSignal(period=21,lookback=60)
        self.macd_divergence = MACDDivergenceSignal(fast_period=13,slow_period=34,signal_period=8,lookback=60)
        self.rsi_filter = RSIFilterSignal(period=13,solid={'upper_threshold': 80, 'lower_threshold': 20})
        self.adx = ADXFilterSignal(period=13, threshold=35)
        self.bollinger_counter_trend = BollingerCounterTrendEntrySignal()
        self.rsi_counter_trend = RSICounterTrendEntrySignal(period=13)
        self.donchian_breakout_exit = DonchianBreakoutEntrySignal(period=233)
        self.alma_circulation = ALMACirculationSignal(
            sigma=6.0,
            offset=0.85,
            params={
                'short_period': 21,
                'middle_period': 89,
                'long_period': 233
            }
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._entry_signals = None
        self._exit_signals = None
        
        # ポイントの設定
        self.entry_points = entry_points if entry_points is not None else np.array([
            7,  # ROCDivergenceSignal
            4,  # MACDDivergenceSignal
            8,  # RSIFilterSignal
            4,  # ADX
            1,  # Bollinger Counter Trend
            2,  # RSICounterTrendEntrySignal
            4,  # DonchianBreakoutEntrySignal
            6,  # ALMACirculationSignal
        ])
        
        self.exit_points = exit_points if exit_points is not None else np.array([
            2,  # Pinbar
            2,  # ROCDivergenceSignal
            8,  # MACDDivergenceSignal
            7,  # BollingerBreakoutExitSignal
            5,  # RSICounterTrendEntrySignal
            8,  # DonchianBreakoutEntrySignal
            5,  # ALMACirculationSignal
        ])
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        if self._entry_signals is None or current_len != self._data_len:
            # データフレームの作成
            df = data[['open', 'high', 'low', 'close', 'volume']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # エントリーシグナルの計算
            entry_signals = np.array([
                self.roc_divergence.generate(df)== -1,
                self.macd_divergence.generate(df)== -1,
                self.rsi_filter.generate(df)== -1,
                self.adx.generate(df)== -1,
                self.bollinger_counter_trend.generate(df)== -1,
                self.rsi_counter_trend.generate(df)== -1,
                self.donchian_breakout_exit.generate(df)== -1,
                self.alma_circulation.generate(df)== -1,
            ])
            
            # エグジットシグナルの計算
            exit_signals = np.array([
                self.pinbar.generate(df),
                self.roc_divergence.generate(df),
                self.macd_divergence.generate(df),
                self.bollinger_counter_trend.generate(df),
                self.rsi_counter_trend.generate(df),
                self.donchian_breakout_exit.generate(df),
                self.alma_circulation.generate(df),
            ])
            
            # 各時点でのポイントの合計を計算
            entry_points = np.zeros(current_len)
            exit_points = np.zeros(current_len)
            
            for i in range(current_len):
                entry_points[i] = calculate_points(entry_signals[:, i], self.entry_points)
                exit_points[i] = calculate_points(exit_signals[:, i], self.exit_points)
            
            # エントリー・エグジットシグナルの生成
            self._entry_signals = calculate_entry_signals(entry_points, self.entry_threshold)
            self._exit_signals = exit_points >= self.exit_threshold
            
            self._data_len = current_len
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._entry_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._entry_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if position != -1:  # 売りポジションのみ
            return False
        
        if self._exit_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        return bool(self._exit_signals[index]) 