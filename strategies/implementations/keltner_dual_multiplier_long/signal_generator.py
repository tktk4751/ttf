#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.keltner.breakout_entry import KeltnerBreakoutEntrySignal
from signals.implementations.chop.filter import ChopFilterSignal


@jit(nopython=True)
def calculate_entry_signals(keltner: np.ndarray, chop_signal: np.ndarray) -> np.ndarray:
    """エントリーシグナルを一度に計算（高速化版）"""
    return np.where(
        (keltner == 1) & (chop_signal == 1),
        1,
        0
    ).astype(np.int8)


class KeltnerDualMultiplierLongSignalGenerator(BaseSignalGenerator):
    """
    ケルトナーチャネル+チョピネスフィルターのシグナル生成クラス（売買別ATR乗数・買い専用・高速化版）
    
    エントリー条件:
    - アッパーブレイクアウトで買いシグナル（upper_multiplierを使用）
    - チョピネスインデックスがトレンド相場を示している
    
    エグジット条件:
    - 終値がロワーバンドを下回る
    """
    
    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 2.0,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
    ):
        """初期化"""
        super().__init__("KeltnerDualMultiplierLongSignalGenerator")
        
        # シグナル生成器の初期化
        self.signal = KeltnerBreakoutEntrySignal(
            period=ema_period,
            atr_period=atr_period,
            upper_multiplier=upper_multiplier,
            lower_multiplier=lower_multiplier
        )
        self.chop_filter = ChopFilterSignal(
            period=chop_period,
            solid={'chop_solid': chop_threshold}
        )
        
        # パラメータの保存
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
        
        # キャッシュ用の変数（最小限に抑える）
        self._data_len = 0
        self._signals = None
        self._lower = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        current_len = len(data)
        
        # データ長が変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len:
            # データフレームの作成（必要な列のみ）
            df = data[['open', 'high', 'low', 'close']] if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            
            # ケルトナーチャネルの計算
            keltner_signals = self.signal.generate(df)
            chop_signals = self.chop_filter.generate(df)
            
            # エントリーシグナルの計算（ベクトル化・高速化）
            self._signals = calculate_entry_signals(keltner_signals, chop_signals)
            
            # エグジット用のロワーバンドを保存
            result = self.signal._keltner.calculate(df)
            if result is None:
                self._lower = np.zeros(current_len)
            else:
                # ATRの取得
                atr = self.signal._keltner.atr.calculate(df)
                # ロワーバンドの計算
                self._lower = result.middle - (self.lower_multiplier * atr)
            
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
        
        # 終値の取得
        close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
        
        # 買いポジションのエグジット判定
        return close[index] < self._lower[index] 