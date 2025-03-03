#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.keltner import KeltnerChannel

class KeltnerHalfBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    ケルトナーチャネルのハーフラインブレイクアウトによるエントリーシグナル
    
    - 現在の終値が前回のアッパーハーフラインを上回った場合: ロングエントリー (1)
    - 現在の終値が前回のロワーハーフラインを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(self, period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        """
        コンストラクタ
        
        Args:
            period: EMAの期間（デフォルト: 20）
            atr_period: ATRの期間（デフォルト: 10）
            multiplier: ATRの乗数（デフォルト: 2.0）
        """
        params = {
            'period': period,
            'atr_period': atr_period,
            'multiplier': multiplier
        }
        super().__init__(f"KeltnerHalfBreakout({period}, {atr_period}, {multiplier})", params)
        self._keltner = KeltnerChannel(period, atr_period, multiplier)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # ケルトナーチャネルの計算
        result = self._keltner.calculate(data)
        
        # 終値の取得
        close = data['close'].values if isinstance(data, pd.DataFrame) else data
        
        # シグナルの初期化
        signals = np.zeros(len(close))
        
        # 最初の期間はシグナルなし
        period = self._params['period']
        signals[:period] = 0
        
        # ブレイクアウトの判定
        for i in range(period, len(close)):
            # ロングエントリー: 終値がアッパーハーフラインを上回る
            if close[i] > result.upper.iloc[i-1]:
                signals[i] = 1
            # ショートエントリー: 終値がロワーハーフラインを下回る
            elif close[i] < result.half_lower.iloc[i-1]:
                signals[i] = -1
        
        return signals