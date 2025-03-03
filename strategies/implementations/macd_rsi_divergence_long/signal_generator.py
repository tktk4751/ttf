#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.divergence.macd_divergence import MACDDivergenceSignal
from signals.implementations.bollinger.exit import BollingerBreakoutExitSignal
from signals.implementations.adx.filter import ADXFilterSignal


@jit(nopython=True)
def calculate_entry_signals(macd_signals: np.ndarray, adx_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    return np.where(
        (macd_signals == 1) & (adx_signals == 1),
        1,
        0
    ).astype(np.int8)


class MACDDivergenceLongSignalGenerator(BaseSignalGenerator):
    """
    MACDダイバージェンスを使用したシグナル生成クラス（買い専用・高速化版）
    
    エントリー条件:
    - MACDダイバージェンスが買いシグナル(1)
    
    エグジット条件:
    - ボリンジャーバンドブレイクアウトが買いエグジット(1)
    - MACDダイバージェンスが売りシグナル(-1)
    """
    
    def __init__(
        self,
        macd_fast_period: int = 12,
        macd_slow_period: int = 26,
        macd_signal_period: int = 9,
        macd_lookback: int = 30,
        bb_period: int = 21,
        bb_num_std: float = 3.0,
    ):
        """初期化"""
        super().__init__("MACDRSIDivergenceLongSignalGenerator")
        
        # シグナル生成器の初期化
        self.macd_signal = MACDDivergenceSignal(
            fast_period=macd_fast_period,
            slow_period=macd_slow_period,
            signal_period=macd_signal_period,
            lookback=macd_lookback
        )
        
        self.bb_signal = BollingerBreakoutExitSignal(
            period=bb_period,
            num_std=bb_num_std
        )
        
        self.adx_signal = ADXFilterSignal(
            period=13,
            threshold=35.0
        )
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._signals = None
        self._bb_signals = None
        self._macd_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            macd_signals = self.macd_signal.generate(df)
            adx_signals = self.adx_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(macd_signals, adx_signals)
            
            # エグジット用のシグナルを事前計算
            self._bb_signals = self.bb_signal.generate(df)
            
            self._macd_signals = macd_signals
            
            self._data_len = current_len
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if position != 1:  # 買いポジションのみ
            return False
        
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # キャッシュされたシグナルを使用
        # ボリンジャーバンドブレイクアウトまたはMACDダイバージェンスの売りシグナル
        return bool(self._bb_signals[index] == 1 or self._macd_signals[index] == -1) 