#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.gvidya_keltner.breakout_entry import GVIDYAKeltnerBreakoutEntrySignal
from signals.implementations.chop.filter import ChopFilterSignal


@jit(nopython=True)
def calculate_entry_signals(gvidya_signals: np.ndarray, chop_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    return np.where(
        (gvidya_signals == -1) & (chop_signals == 1),
        -1,
        0
    ).astype(np.int8)


class GVIDYAKeltnerSingleShortSignalGenerator(BaseSignalGenerator):
    """
    G-VIDYAケルトナーチャネルのシグナル生成クラス（単一チャネル・売り専用・高速化版）
    
    エントリー条件:
    - G-VIDYAケルトナーチャネルのロワーブレイクアウトで売りシグナル
    - チョピネスフィルターがトレンド相場を示している
    
    エグジット条件:
    - G-VIDYAケルトナーチャネルの買いシグナル
    """
    
    def __init__(
        self,
        vidya_period: int = 46,
        sd_period: int = 28,
        gaussian_length: int = 4,
        gaussian_sigma: float = 2.0,
        atr_period: int = 14,
        upper_multiplier: float = 1.3,
        lower_multiplier: float = 1.3,
        chop_period: int = 14,
        chop_solid: float = 50.0,
    ):
        """初期化"""
        super().__init__("GVIDYAKeltnerSingleShortSignalGenerator")
        
        # シグナル生成器の初期化
        self.gvidya_signal = GVIDYAKeltnerBreakoutEntrySignal(
            vidya_period=vidya_period,
            sd_period=sd_period,
            gaussian_length=gaussian_length,
            gaussian_sigma=gaussian_sigma,
            atr_period=atr_period,
            upper_multiplier=upper_multiplier,
            lower_multiplier=lower_multiplier
        )
        
        self.chop_signal = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_solid}
        )
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._signals = None
        self._gvidya_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            gvidya_signals = self.gvidya_signal.generate(df)
            chop_signals = self.chop_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(gvidya_signals, chop_signals)
            
            # エグジット用のシグナルを事前計算
            self._gvidya_signals = gvidya_signals
            
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
        return bool(self._gvidya_signals[index] == 1) 