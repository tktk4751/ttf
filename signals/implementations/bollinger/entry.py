#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.bollinger_bands import BollingerBands


class BollingerCounterTrendEntrySignal(BaseSignal, IEntrySignal):
    """
    ボリンジャーバンド逆張りエントリーシグナル
    
    エントリー条件:
    - ロング: 前のローソク足が3シグマ上限を上回り、現在のローソク足が3シグマ上限を下回る
    - ショート: 前のローソク足が3シグマ下限を下回り、現在のローソク足が3シグマ下限を上回る
    """
    
    def __init__(
        self,
        period: int = 21,
        num_std: float = 3.0,
    ):
        """
        コンストラクタ
        
        Args:
            period: ボリンジャーバンドの期間
            num_std: 標準偏差の乗数（シグマ）
        """
        super().__init__("BollingerBreakout")
        
        # パラメータの設定
        self.period = period
        self.num_std = num_std
        
        # インジケーターの初期化
        self.bb = BollingerBands(period=period, num_std=num_std)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            シグナル配列（1: ロング, -1: ショート, 0: シグナルなし）
        """
        # データフレームの作成
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        close = df['close'].values
        
        # ボリンジャーバンドの計算
        bb_result = self.bb.calculate(close)
        
        # シグナルの初期化
        signals = np.zeros(len(data))
        
        # 2番目の要素からシグナルを計算
        for i in range(1, len(data)):
            # 前のローソク足が上限を上回り、現在のローソク足が上限を下回る → ショート
            if close[i-1] > bb_result.upper[i-1] and close[i] < bb_result.upper[i]:
                signals[i] = -1
            # 前のローソク足が下限を下回り、現在のローソク足が下限を上回る → ロング
            elif close[i-1] < bb_result.lower[i-1] and close[i] > bb_result.lower[i]:
                signals[i] = 1
        
        return signals 