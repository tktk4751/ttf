#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.donchian_atr import DonchianATR


class DonchianATRBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    ドンチャンATRのブレイクアウトによるエントリーシグナル
    
    - 現在の終値がドンチャンATRのアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値がドンチャンATRのロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        period: int = 20,
        atr_period: int = 10,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 2.0
    ):
        """
        コンストラクタ
        
        Args:
            period: ドンチャンチャネルの期間（デフォルト: 20）
            atr_period: ATRの期間（デフォルト: 10）
            upper_multiplier: アッパーバンドのATR乗数（デフォルト: 2.0）
            lower_multiplier: ロワーバンドのATR乗数（デフォルト: 2.0）
        """
        params = {
            'period': period,
            'atr_period': atr_period,
            'upper_multiplier': upper_multiplier,
            'lower_multiplier': lower_multiplier
        }
        super().__init__(
            f"DonchianATRBreakout({period}, {atr_period}, {upper_multiplier}, {lower_multiplier})",
            params
        )
        self._donchian_atr = DonchianATR(
            period=period,
            atr_period=atr_period,
            upper_multiplier=upper_multiplier,
            lower_multiplier=lower_multiplier
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # ドンチャンATRの計算
        result = self._donchian_atr.calculate(data)
        if result is None:
            return np.zeros(len(data))
        
        # 終値の取得
        close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
        
        # シグナルの初期化
        signals = np.zeros(len(close))
        
        # 最初の期間はシグナルなし
        period = self._params['period']
        signals[:period] = 0
        
        # ブレイクアウトの判定
        for i in range(period, len(close)):
            # ロングエントリー: 終値がアッパーバンドを上回る
            if close[i] > result.upper[i-1]:
                signals[i] = 1
            # ショートエントリー: 終値がロワーバンドを下回る
            elif close[i] < result.lower[i-1]:
                signals[i] = -1
        
        return signals 