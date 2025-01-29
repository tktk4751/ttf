#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.donchian import DonchianChannel


class DonchianBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    ドンチャンチャネルのブレイクアウトによるエントリーシグナル
    
    - 現在の終値がn期間ドンチャンのアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値がn期間ドンチャンのロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(self, period: int = 20):
        """
        コンストラクタ
        
        Args:
            period: ドンチャンチャネルの期間（デフォルト: 20）
        """
        params = {'period': period}
        super().__init__(f"DonchianBreakout({period})", params)
        self._donchian = DonchianChannel(period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # ドンチャンチャネルの計算
        self._donchian.calculate(data)
        upper, lower, _ = self._donchian.get_bands()
        
        # 終値の取得
        close = data['close'].values if isinstance(data, pd.DataFrame) else data
        
        # シグナルの初期化
        signals = np.zeros(len(close))
        
        # 最初の期間はシグナルなし
        period = self._params['period']
        signals[:period] = 0
        
        # ブレイクアウトの判定
        for i in range(period, len(close)):
            # ロングエントリー: 終値がアッパーバンドを上回る
            if close[i] > upper[i-1]:
                signals[i] = 1
            # ショートエントリー: 終値がロワーバンドを下回る
            elif close[i] < lower[i-1]:
                signals[i] = -1
        
        return signals 