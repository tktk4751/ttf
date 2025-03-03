#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.donchian.entry import DonchianBreakoutEntrySignal
from signals.implementations.squeeze.direction import SqueezeDirectionSignal


@jit(nopython=True)
def calculate_entry_signals(donchian: np.ndarray, squeeze: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    return np.where(
        (donchian == 1) & (squeeze == 1),
        1,
        0
    ).astype(np.int8)


class SqueezeDonchianLongSignalGenerator(BaseSignalGenerator):
    """
    スクイーズ+ドンチャンブレイクアウトのシグナル生成クラス（買い専用・高速化版）
    
    エントリー条件:
    - スクイーズオン状態
    - ドンチャンチャネルのアッパーブレイクアウトで買いシグナル
    
    エグジット条件:
    - ドンチャンチャネルの売りシグナル
    """
    
    def __init__(
        self,
        donchian_period: int = 20,
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
    ):
        """初期化"""
        super().__init__("SqueezeDonchianLongSignalGenerator")
        
        # シグナル生成器の初期化
        self.entry_signal = DonchianBreakoutEntrySignal(
            period=donchian_period
        )
        self.direction_signal = SqueezeDirectionSignal(
            bb_length=bb_length,
            bb_mult=bb_mult,
            kc_length=kc_length,
            kc_mult=kc_mult
        )
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._signals = None
        self._donchian_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            donchian_signals = self.entry_signal.generate(df).astype(np.int8)
            squeeze_signals = self.direction_signal.generate(df).astype(np.int8)
            
            # エントリーシグナルの計算（ベクトル化）
            self._signals = calculate_entry_signals(donchian_signals, squeeze_signals)
            
            # エグジット用のシグナルを事前計算
            self._donchian_signals = donchian_signals
            
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
        return bool(self._donchian_signals[index] == -1) 