#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.squeeze_momentum import SqueezeMomentum


class SqueezeDirectionSignal(BaseSignal, IDirectionSignal):
    """
    スクイーズのオン/オフ状態によるディレクションシグナル
    
    - スクイーズオン状態: トレンド相場 (1)
    - スクイーズオフ状態: レンジ相場 (-1)
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
        """
        params = {
            'bb_length': bb_length,
            'bb_mult': bb_mult,
            'kc_length': kc_length,
            'kc_mult': kc_mult,
        }
        super().__init__(f"SqueezeDirection({bb_length},{kc_length})", params)
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
            シグナルの配列 (1: スクイーズオン, -1: スクイーズオフ)
        """
        # スクイーズ状態の取得
        _ = self._squeeze.calculate(data)  # モメンタムの計算（必須）
        sqz_on, _, _ = self._squeeze.get_squeeze_states()
        
        # シグナルの初期化
        signals = np.zeros(len(sqz_on))
        
        # 最初の期間はシグナルなし
        period = self._params['kc_length']
        signals[:period] = 0
        
        # スクイーズ状態に基づくシグナルの生成
        signals[period:] = np.where(sqz_on[period:], 1, -1)
        
        return signals 