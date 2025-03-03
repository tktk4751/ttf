#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.kama.direction import KAMACirculationSignal
from signals.implementations.chop.filter import ChopFilterSignal
from signals.implementations.donchian.entry import DonchianBreakoutEntrySignal


@jit(nopython=True)
def calculate_entry_signals(kama_signals: np.ndarray, chop_signals: np.ndarray, donchian_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    return np.where(
        (kama_signals == -1) & (chop_signals == 1) & (donchian_signals == 1),
        -1,
        0
    ).astype(np.int8)


class KAMACirculationChopDonchianShortSignalGenerator(BaseSignalGenerator):
    """
    KAMAの大循環、チョピネスフィルター、ドンチャンブレイクアウトを組み合わせたシグナル生成クラス（売り専用・高速化版）
    
    エントリー条件:
    - KAMAの大循環が売りシグナル(-1)
    - チョピネスフィルターがトレンド相場(1)
    - ドンチャンブレイクアウトが買いシグナル(1)
    
    エグジット条件:
    - KAMAの大循環が買いシグナル(1)
    """
    
    def __init__(
        self,
        kama_short_period: int = 21,
        kama_middle_period: int = 89,
        kama_long_period: int = 233,
        chop_period: int = 14,
        chop_solid: float = 50.0,
        donchian_period: int = 20,
    ):
        """初期化"""
        super().__init__("KAMACirculationChopDonchianShortSignalGenerator")
        
        # シグナル生成器の初期化
        self.kama_signal = KAMACirculationSignal(
            params={
                'short_period': kama_short_period,
                'middle_period': kama_middle_period,
                'long_period': kama_long_period
            }
        )
        
        self.chop_signal = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_solid}
        )
        
        self.donchian_signal = DonchianBreakoutEntrySignal(
            period=donchian_period
        )
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._signals = None
        self._kama_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            kama_signals = self.kama_signal.generate(df)
            chop_signals = self.chop_signal.generate(df)
            donchian_signals = self.donchian_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(kama_signals, chop_signals, donchian_signals)
            
            # エグジット用のシグナルを事前計算
            self._kama_signals = kama_signals
            
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
        
        # キャッシュされたシグナルを使用
        return bool(self._kama_signals[index] == 1) 