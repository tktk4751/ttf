#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.kama_keltner.breakout_entry import KAMAKeltnerBreakoutEntrySignal
from signals.implementations.chop.filter import ChopFilterSignal
from signals.implementations.rsi.exit import RSIExitSignal


@jit(nopython=True)
def calculate_entry_signals(kama: np.ndarray, filter_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    return np.where(
        (kama == -1) & (filter_signal == 1),
        -1,
        0
    ).astype(np.int8)


class KAMAKeltnerChopRSIShortSignalGenerator(BaseSignalGenerator):
    """
    KAMAケルトナーチャネル+チョピネスフィルター+RSIエグジットのシグナル生成クラス（売り専用・高速化版）
    
    エントリー条件:
    - KAMAケルトナーチャネルのロワーブレイクアウトで売りシグナル
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    - KAMAケルトナーチャネルの買いシグナル
    - RSIエグジットシグナルが-1
    """
    
    def __init__(
        self,
        kama_period: int = 10,
        kama_fast: int = 3,
        kama_slow: int = 144,
        atr_period: int = 10,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 2.0,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
        rsi_period: int = 14,
        rsi_solid: Dict[str, Any] = None,
    ):
        """初期化"""
        super().__init__("KAMAKeltnerChopRSIShortSignalGenerator")
        
        # シグナル生成器の初期化
        self.signal = KAMAKeltnerBreakoutEntrySignal(
            kama_period=kama_period,
            kama_fast=kama_fast,
            kama_slow=kama_slow,
            atr_period=atr_period,
            upper_multiplier=upper_multiplier,
            lower_multiplier=lower_multiplier
        )

        self.filter_signal = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_threshold}
        )
        self.rsi_exit_signal = RSIExitSignal(
            period=rsi_period,
        )
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._signals = None
        self._kama_signals = None
        self._rsi_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            entry_signals = self.signal.generate(df)
            filter_signals = self.filter_signal.generate(df)
            exit_signals = self.signal.generate(df)
            rsi_signals = self.rsi_exit_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(entry_signals, filter_signals)
            
            # エグジット用のシグナルを事前計算
            self._kama_signals = exit_signals
            self._rsi_signals = rsi_signals
            
            self._data_len = current_len
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if position != -1:  # 売りポジションのみ
            return False
        
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # KAMAシグナルまたはRSIシグナルがエグジットを示す場合
        return bool(self._kama_signals[index] == 1 or self._rsi_signals[index] == -1) 