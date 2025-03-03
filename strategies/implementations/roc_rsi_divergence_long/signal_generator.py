#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.divergence.roc_divergence import ROCDivergenceSignal
from signals.implementations.rsi.filter import RSIFilterSignal
from signals.implementations.bollinger.exit import BollingerBreakoutExitSignal
from signals.implementations.adx.filter import ADXFilterSignal


@jit(nopython=True)
def calculate_entry_signals(roc_signals: np.ndarray, adx_signals: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    return np.where(
        (roc_signals == 1) & (adx_signals == 1),
        1,
        0
    ).astype(np.int8)


class ROCDivergenceLongSignalGenerator(BaseSignalGenerator):
    """
    ROCダイバージェンスシグナル生成クラス（買い専用・高速化版）
    
    エントリー条件:
    - ROCダイバージェンスが買いシグナル(1)
    - ADXフィルターがトレンド相場(1)
    
    エグジット条件:
    - ボリンジャーバンドブレイクアウトが買いエグジット(1)
    - ROCダイバージェンスが売りシグナル(-1)
    """
    
    def __init__(
        self,
        roc_period: int = 12,
        roc_lookback: int = 30,
        bb_period: int = 21,
        bb_num_std: float = 3.0,
    ):
        """初期化"""
        super().__init__("ROCDivergenceLongSignalGenerator")
        
        # シグナル生成器の初期化
        self.roc_signal = ROCDivergenceSignal(
            period=roc_period,
            lookback=roc_lookback
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
        self._roc_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # 各シグナルの計算（一度に実行）
            roc_signals = self.roc_signal.generate(df)
            adx_signals = self.adx_signal.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(roc_signals, adx_signals)
            
            # エグジット用のシグナルを事前計算
            self._roc_signals = roc_signals
            
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
        # ボリンジャーバンドブレイクアウトまたはROCダイバージェンスの売りシグナル
        return bool(self._roc_signals[index] == -1) 