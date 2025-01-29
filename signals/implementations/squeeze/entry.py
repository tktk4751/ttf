#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.squeeze_momentum import SqueezeMomentum


class SqueezeMomentumEntrySignal(BaseSignal, IEntrySignal):
    """
    スクイーズモメンタムによるエントリーシグナル
    
    - スクイーズオン状態で買いモメンタムが発生: ロングエントリー (1)
    - スクイーズオン状態で売りモメンタムが発生: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
    ):
        """
        コンストラクタ
        
        Args:
            bb_length: Bollinger Bands の期間
            bb_mult: Bollinger Bands の乗数
            kc_length: Keltner Channels の期間
            kc_mult: Keltner Channels の乗数
            momentum_threshold: モメンタムの閾値（デフォルト: 0.0）
        """
        params = {
            'bb_length': bb_length,
            'bb_mult': bb_mult,
            'kc_length': kc_length,
            'kc_mult': kc_mult,
        }
        super().__init__(f"SqueezeMomentum({bb_length},{kc_length})", params)
        self._squeeze = SqueezeMomentum(
            bb_length=bb_length,
            bb_mult=bb_mult,
            kc_length=kc_length,
            kc_mult=kc_mult
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # スクイーズモメンタムの計算
        momentum = self._squeeze.calculate(data)
        sqz_on, _, _ = self._squeeze.get_squeeze_states()
        
        # シグナルの初期化
        signals = np.zeros(len(momentum))
        
        # 最初の期間はシグナルなし
        period = self._params['kc_length']
        signals[:period] = 0
        
        # モメンタムの閾値
        threshold = 0.0
        
        # シグナルの生成
        for i in range(period, len(momentum)):
            if sqz_on[i]:  # スクイーズオン状態
                # 買いモメンタム
                if momentum[i] > threshold and momentum[i-1] <= threshold:
                    signals[i] = 1
                # 売りモメンタム
                elif momentum[i] < -threshold and momentum[i-1] >= -threshold:
                    signals[i] = -1
        
        return signals 