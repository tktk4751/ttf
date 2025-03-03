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


@jit(nopython=True)
def calculate_points(signals: np.ndarray, points: np.ndarray) -> np.ndarray:
    """ポイントを計算（高速化版）"""
    return np.sum(signals * points)


@jit(nopython=True)
def calculate_entry_signals(points: np.ndarray, threshold: float) -> np.ndarray:
    """エントリーシグナルを計算（高速化版）"""
    return np.where(points >= threshold, 1, 0).astype(np.int8)


class BuyCounterPredicterSignalGenerator(BaseSignalGenerator):
    """
    複数のシグナルをポイント制で評価する逆張り買い戦略のシグナル生成クラス
    
    エントリー条件:
    - 各シグナルのポイントの合計がエントリーしきい値を超えた場合に買いシグナル
    
    エグジット条件:
    - 各シグナルのポイントの合計がエグジットしきい値を超えた場合に売りシグナル
    """
    
    def __init__(
        self,
        entry_threshold: float = 10.0,
        exit_threshold: float = 10.0
    ):
        """初期化"""
        super().__init__("BuyCounterPredicterSignalGenerator")
        
        # しきい値の設定
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        # エントリー用シグナル生成器の初期化
        self.pinbar = PinbarEntrySignal()
        self.stoch_rsi_divergence = StochRSIDivergenceSignal()
        self.rsi_divergence = RSIDivergenceSignal(rsi_period=13,lookback=60)
        self.roc_divergence = ROCDivergenceSignal(period=55,lookback=60)
        self.mfi_divergence = MFIDivergenceSignal(period=21,lookback=60)
        self.macd_divergence = MACDDivergenceSignal(fast_period=13,slow_period=34,signal_period=8,lookback=60)
        self.rsi_filter = RSIFilterSignal(period=13)
        self.chop = ChopFilterSignal(period=34)
        self.adx = ADXFilterSignal(period=8, threshold=30)
        self.bollinger_counter_trend = BollingerCounterTrendEntrySignal()
        self.rsi_counter_trend = RSICounterTrendEntrySignal(period=13)
        
        # エグジット用シグナル生成器の初期化
        self.pinbar_exit = PinbarEntrySignal()
        self.stoch_rsi_divergence_exit = StochRSIDivergenceSignal()
        self.rsi_divergence_exit = RSIDivergenceSignal(rsi_period=13,lookback=60)
        self.roc_divergence_exit = ROCDivergenceSignal(period=55,lookback=60)
        self.mfi_divergence_exit = MFIDivergenceSignal(period=21,lookback=60)
        self.macd_divergence_exit = MACDDivergenceSignal(fast_period=13,slow_period=34,signal_period=8,lookback=60)
        self.bollinger_breakout_exit = BollingerBreakoutExitSignal(period=55,num_std=3.0)
        self.rsi_counter_trend_exit = RSICounterTrendEntrySignal(period=13)
        self.donchian_breakout_exit = DonchianBreakoutEntrySignal(period=89)
        
        # キャッシュ用の変数
        self._data_len = 0
        self._entry_signals = None
        self._exit_signals = None
        
        # ポイントの設定
        self.entry_points = np.array([
            3,  # Pinbar
            3,  # StochRSIDivergenceSignal
            3,  # RSIDivergenceSignal
            3,  # ROCDivergenceSignal
            3,  # MFIDivergenceSignal
            3,  # MACDDivergenceSignal
            3,  # RSIFilterSignal
            4,  # CHOP
            4,  # ADX
            3,  # Bollinger Counter Trend
            3,  # RSICounterTrendEntrySignal
        ])
        
        self.exit_points = np.array([
            7,  # Pinbar
            3,  # StochRSIDivergenceSignal
            3,  # RSIDivergenceSignal
            3,  # ROCDivergenceSignal
            3,  # MFIDivergenceSignal
            3,  # MACDDivergenceSignal
            10,  # BollingerBreakoutExitSignal
            7,  # RSICounterTrendEntrySignal
            7,  # DonchianBreakoutEntrySignal
        ])
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        if self._entry_signals is None or current_len != self._data_len:
            # データフレームの作成
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # エントリーシグナルの計算
            entry_signals = np.array([
                self.pinbar.generate(df),
                self.stoch_rsi_divergence.generate(df),
                self.rsi_divergence.generate(df),
                self.roc_divergence.generate(df),
                self.mfi_divergence.generate(df),
                self.macd_divergence.generate(df),
                self.rsi_filter.generate(df),
                self.chop.generate(df),
                self.adx.generate(df),
                self.bollinger_counter_trend.generate(df),
                self.rsi_counter_trend.generate(df)
            ])
            
            # エグジットシグナルの計算
            exit_signals = np.array([
                self.pinbar_exit.generate(df)== -1,
                self.stoch_rsi_divergence_exit.generate(df)== -1,
                self.rsi_divergence_exit.generate(df)== -1,
                self.roc_divergence_exit.generate(df)== -1,
                self.mfi_divergence_exit.generate(df)== -1,
                self.macd_divergence_exit.generate(df)== -1,
                self.bollinger_breakout_exit.generate(df),
                self.rsi_counter_trend_exit.generate(df)== -1,
                self.donchian_breakout_exit.generate(df)== -1
            ])
            
            # 各時点でのポイント合計を計算
            entry_points_sum = np.zeros(current_len)
            exit_points_sum = np.zeros(current_len)
            
            for i in range(current_len):
                entry_points_sum[i] = calculate_points(
                    entry_signals[:, i],
                    self.entry_points
                )
                exit_points_sum[i] = calculate_points(
                    exit_signals[:, i],
                    self.exit_points
                )
            
            # エントリー/エグジットシグナルの生成
            self._entry_signals = calculate_entry_signals(
                entry_points_sum,
                self.entry_threshold
            )
            self._exit_signals = calculate_entry_signals(
                exit_points_sum,
                self.exit_threshold
            )
            
            self._data_len = current_len
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._entry_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._entry_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if position != 1:  # 買いポジションのみ
            return False
        
        if self._exit_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        return bool(self._exit_signals[index]) 