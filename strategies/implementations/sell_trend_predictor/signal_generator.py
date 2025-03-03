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
from signals.implementations.alma.direction import ALMADirectionSignal2
from signals.implementations.rsi.exit import RSIExit2Signal
from signals.implementations.kama.entry import KAMACrossoverEntrySignal
from signals.implementations.divergence.rsi_divergence import RSIDivergenceSignal
from signals.implementations.bollinger.entry import BollingerCounterTrendEntrySignal
from signals.implementations.rsi.exit import RSIExitSignal
from signals.implementations.rsi.entry import RSIEntrySignal
from signals.implementations.alma_keltner.breakout_entry import ALMAKeltnerBreakoutEntrySignal
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
    return np.where(points >= threshold, -1, 0).astype(np.int8)  # 売りシグナルは-1


class SellTrendpredictorSignalGenerator(BaseSignalGenerator):
    """
    複数のシグナルをポイント制で評価するシグナル生成クラス（売り専用）
    """
    
    def __init__(
        self,
        entry_threshold: float = 10.0,
        exit_threshold: float = 10.0,
        entry_points: np.ndarray = None,
        exit_points: np.ndarray = None
    ):
        """初期化"""
        super().__init__("SellTrendpredictorSignalGenerator")
        
        # しきい値の設定
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        # エントリー用シグナル生成器の初期化
        self.supertrend = SupertrendDirectionSignal(period=62, multiplier=5.5)
        self.keltner = KAMAKeltnerBreakoutEntrySignal(kama_period=34, kama_fast=3, kama_slow=144,atr_period=34,upper_multiplier=1.3, lower_multiplier=2.1)
        self.donchian = DonchianATRBreakoutEntrySignal(period=89, atr_period=21, upper_multiplier=0.5, lower_multiplier=3.4)
        self.chop = ChopFilterSignal(period=55)
        self.adx = ADXFilterSignal(period=13, threshold=30)
        self.kama_trend = KAMATrendFollowingStrategy(short_period=55, long_period=144)
        self.roc = ROCEntrySignal(period=89)
        self.squeeze_momentum = SqueezeMomentumEntrySignal(bb_length=55, bb_mult=1.0, kc_length=89, kc_mult=2.1)
        self.alma_crossover = KAMACrossoverEntrySignal(short_period=13, long_period=34)
        self.alma_crossover2 = KAMACrossoverEntrySignal(short_period=21, long_period=233)
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
            3,  # Supertrend
            4,  # Keltner
            2,  # Donchian
            4,  # Chop
            3,  # ADX
            2,  # ALMA Trend
            2,  # ROC
            3,  # Squeeze Momentum
            1,  # ALMA Crossover
            2,  # ALMA Crossover2
            2,  # RSI Entry
        ])
        
        self.exit_points = exit_points if exit_points is not None else np.array([
            3,  # Supertrend
            4,  # Keltner
            2,  # Donchian
            2,  # Chop
            2,  # ADX
            3,  # ALMA Trend
            2,  # ROC
            3,  # Squeeze Momentum
            4,  # MACD Divergence
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
            
            # 各シグナルの計算（売りシグナルを検出）
            entry_signals = np.array([
                self.supertrend.generate(df) == -1,  # 売りシグナルを検出
                self.keltner.generate(df) == -1,
                self.donchian.generate(df) == -1,
                self.chop.generate(df),  # フィルターはそのまま
                self.adx.generate(df),  # フィルターはそのまま
                self.kama_trend.generate(df) == -1,
                self.roc.generate(df) == -1,
                self.squeeze_momentum.generate(df) == -1,
                self.alma_crossover.generate(df) == -1,
                self.alma_crossover2.generate(df) == -1,
                self.rsi_entry.generate(df) == -1,
            ]).astype(np.int8)  # Shape: (n_signals, n_samples)
            
            # ポイントの計算
            entry_points = calculate_points(entry_signals, self.entry_points)
            
            # エグジットシグナルの計算（買いシグナルを検出）
            exit_signals = np.array([
                self.supertrend.generate(df) == 1,  # 買いシグナルを検出
                self.keltner.generate(df) == 1,
                self.donchian.generate(df) == 1,
                self.chop.generate(df) == -1,  # フィルターは反転
                self.adx.generate(df) == -1,  # フィルターは反転
                self.kama_trend.generate(df) == 1,
                self.roc.generate(df) == 1,
                self.squeeze_momentum.generate(df) == 1,
                self.macd_divergence.generate(df) == 1,
                self.roc_divergence.generate(df) == 1,
                self.bollinger_counter_trend.generate(df) == 1,
                self.rsi_exit.generate(df),  # エグジットはそのまま
            ]).astype(np.int8)
            
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
        if position != -1:  # 売りポジションのみ
            return False
        
        if self._exit_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        return bool(self._exit_signals[index]) 