#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.rsi import RSI
from .divergence_signal import DivergenceSignal


class RSIDivergenceSignal(BaseSignal, IEntrySignal):
    """
    RSIダイバージェンスシグナル
    
    価格とRSIの間のダイバージェンスを検出し、エントリーシグナルを生成します。
    
    - 強気ダイバージェンス（ロングエントリー）：
      価格が安値を切り下げているのに対し、RSIが安値を切り上げている状態。
      上昇転換の可能性を示唆。
    
    - 弱気ダイバージェンス（ショートエントリー）：
      価格が高値を切り上げているのに対し、RSIが高値を切り下げている状態。
      下落転換の可能性を示唆。
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        lookback: int = 30,
        params: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            rsi_period: RSIの計算期間
            lookback: ダイバージェンス検出のルックバック期間
            params: その他のパラメータ（オプション）
        """
        super().__init__("RSIDivergence")
        
        # パラメータの設定
        self.rsi_period = rsi_period
        self.lookback = lookback
        
        # インジケーターの初期化
        self.rsi = RSI(period=rsi_period)
        self.divergence = DivergenceSignal(lookback=lookback)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        RSIダイバージェンスシグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            シグナル配列（1: ロング, -1: ショート, 0: シグナルなし）
        """
        # データフレームの作成
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        
        # RSIの計算
        rsi_values = self.rsi.calculate(data)
        
        # ダイバージェンスの検出
        signals = self.divergence.generate(df, rsi_values)
        
        return signals 