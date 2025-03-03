#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.supertrend.direction import SupertrendDirectionSignal
from signals.implementations.kama_keltner.breakout_entry import KAMAKeltnerBreakoutEntrySignal
from signals.implementations.donchian.entry import DonchianBreakoutEntrySignal
from signals.implementations.chop.filter import ChopFilterSignal
from signals.implementations.adx.filter import ADXFilterSignal
from signals.implementations.kama.direction import KAMATrendFollowingStrategy
from signals.implementations.roc.entry import ROCEntrySignal
from signals.implementations.squeeze.entry import SqueezeMomentumEntrySignal
from signals.implementations.rsi.exit import RSIExit2Signal
from signals.implementations.kama.entry import KAMACrossoverEntrySignal
from signals.implementations.divergence.rsi_divergence import RSIDivergenceSignal
from signals.implementations.bollinger.entry import BollingerCounterTrendEntrySignal
from signals.implementations.rsi.exit import RSIExitSignal
from signals.implementations.rsi.entry import RSIEntrySignal
from signals.implementations.donchian_atr.entry import DonchianATRBreakoutEntrySignal
from signals.implementations.candlestick.pinbar_entry import PinbarEntrySignal
from signals.implementations.divergence.macd_divergence import MACDDivergenceSignal
from signals.implementations.divergence.roc_divergence import ROCDivergenceSignal

@jit(nopython=True)
def calculate_points(signals: np.ndarray, points: np.ndarray) -> np.ndarray:
    """ポイントの計算（高速化版）"""
    # signals: (n_signals, n_samples)
    # points: (n_signals,)
    # signals.T: (n_samples, n_signals)
    return np.sum(signals.T * points, axis=1)  # (n_samples,)


@jit(nopython=True)
def calculate_entry_signals(points: np.ndarray, threshold: float) -> np.ndarray:
    """エントリーシグナルの計算（高速化版）"""
    return np.where(points >= threshold, 1, 0).astype(np.int8)


class BuyTrendpredictorSignalGenerator(BaseSignalGenerator):
    """
    複数のシグナルをポイント制で評価するシグナル生成クラス
    """
    
    def __init__(
        self,
        entry_threshold: float = 10.0,
        exit_threshold: float = 10.0,
        entry_points: np.ndarray = None,
        exit_points: np.ndarray = None
    ):
        """初期化"""
        super().__init__("BuyTrendpredictorSignalGenerator")
        
        # しきい値の設定
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        # エントリー用シグナル生成器の初期化
        self.supertrend = SupertrendDirectionSignal(period=55, multiplier=5.5)
        self.supertrend2 = SupertrendDirectionSignal(period=21, multiplier=7)
        self.keltner = KAMAKeltnerBreakoutEntrySignal(kama_period=144, kama_fast=2, kama_slow=34,atr_period=89,upper_multiplier=2.1, lower_multiplier=1.3)
        self.keltner2 = KAMAKeltnerBreakoutEntrySignal(kama_period=233, kama_fast=2, kama_slow=144,atr_period=93,upper_multiplier=0.1, lower_multiplier=2.9)
        self.donchian = DonchianATRBreakoutEntrySignal(period=233, atr_period=55, upper_multiplier=2.1, lower_multiplier=1.5)
        self.chop = ChopFilterSignal(period=55)
        self.adx = ADXFilterSignal(period=13, threshold=30)
        self.kama_trend = KAMATrendFollowingStrategy(short_period=55, long_period=233)
        self.roc = ROCEntrySignal(period=144)
        self.squeeze_momentum = SqueezeMomentumEntrySignal(bb_length=55, bb_mult=1.0, kc_length=89, kc_mult=2.1)
        self.squeeze_momentum2 = SqueezeMomentumEntrySignal(bb_length=16, bb_mult=2.5, kc_length=39, kc_mult=3.0)
        self.kama_crossover = KAMACrossoverEntrySignal(short_period=13, long_period=55)
        self.kama_crossover2 = KAMACrossoverEntrySignal(short_period=21, long_period=233)
        self.rsi_entry = RSIEntrySignal(period=5)
        
        # エグジット用の追加シグナル生成器
        self.macd_divergence = MACDDivergenceSignal(fast_period=13,slow_period=34,signal_period=8,lookback=60)
        self.roc_divergence = ROCDivergenceSignal(period=21,lookback=60)
        self.bollinger_counter_trend = BollingerCounterTrendEntrySignal(period=21, num_std=3)
        self.rsi_exit = RSIExitSignal(period=13)
        
        # キャッシュ用の変数
        self._data_len = 0
        self._entry_signals = None
        self._exit_signals = None
        
        # ポイントの設定
        self.entry_points = entry_points if entry_points is not None else np.array([
            2,  # Supertrend
            2,  # Supertrend2
            2,  # Keltner
            2,  # Keltner2('kama_period': 233, kama_fast=2, kama_slow=34, 'atr_period': 93, 'upper_multiplier': 0.1, 'lower_multiplier': 2.9)
            2,  # Donchian
            2,  # Chop
            2,  # ADX
            2,  # kama Trend
            2,  # ROC
            2,  # Squeeze Momentum
            2,  # Squeeze Momentum2('bb_length': 16, 'bb_mult': 2.5, 'kc_length': 39, 'kc_mult': 3.0)
            2,  # kama Crossover
            2,  # kama Crossover2
            2,  # RSI Entry
        ])
        
        self.exit_points = exit_points if exit_points is not None else np.array([
            4,  # Supertrend
            4,  # Supertrend2
            4,  # Keltner
            4,  # Keltner2
            4,  # Donchian
            4,  # Chop
            4,  # ADX
            4,  # kama Trend
            4,  # ROC
            4,  # Squeeze Momentum
            4,  # Squeeze Momentum2
            4,  # ROC Divergence
            4,  # Bollinger Counter Trend
            4,  # RSI Exit
        ])
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        if self._entry_signals is None or current_len != self._data_len:
            # データフレームの作成
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算
            entry_signals = np.array([
                self.supertrend.generate(df),
                self.supertrend2.generate(df),
                self.keltner.generate(df),
                self.keltner2.generate(df),
                self.donchian.generate(df),
                self.chop.generate(df),
                self.adx.generate(df),
                self.kama_trend.generate(df),
                self.roc.generate(df),
                self.squeeze_momentum.generate(df),
                self.squeeze_momentum2.generate(df),
                self.kama_crossover.generate(df),
                self.kama_crossover2.generate(df),
                self.rsi_entry.generate(df),
            ])  # Shape: (n_signals, n_samples)
            
            # 各シグナルが1かどうかをチェック
            entry_signals = (entry_signals == 1).astype(np.int8)
            
            # ポイントの計算
            entry_points = calculate_points(entry_signals, self.entry_points)
            
            # エグジットシグナルの計算
            exit_signals = np.array([
                self.supertrend.generate(df) == -1,
                self.supertrend2.generate(df) == -1,
                self.keltner.generate(df) == -1,
                self.keltner2.generate(df) == -1,
                self.donchian.generate(df) == -1,
                self.chop.generate(df) == -1,
                self.adx.generate(df) == -1,
                self.kama_trend.generate(df) == -1,
                self.roc.generate(df) == -1,
                self.squeeze_momentum.generate(df) == -1,
                self.squeeze_momentum2.generate(df) == -1,
                self.roc_divergence.generate(df) == -1,
                self.bollinger_counter_trend.generate(df) == -1,
                self.rsi_exit.generate(df),
            ]).astype(np.int8)  # bool型からint8型に変換
            
            # エグジットポイントの計算
            exit_points = calculate_points(exit_signals, self.exit_points)
            
            # シグナルの生成
            self._entry_signals = calculate_entry_signals(entry_points, self.entry_threshold)
            self._exit_signals = calculate_entry_signals(exit_points, self.exit_threshold)
            
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