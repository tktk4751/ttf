#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.roc import ROC
from .divergence_signal import DivergenceSignal


class ROCDivergenceSignal(BaseSignal, IEntrySignal):
    """
    ROC（Rate of Change）ダイバージェンスシグナル
    
    価格とROCの間のダイバージェンスを検出し、エントリーシグナルを生成します。
    
    - 強気ダイバージェンス（ロングエントリー）：
      価格が安値を切り下げているのに対し、ROCが安値を切り上げている状態。
      上昇転換の可能性を示唆。
    
    - 弱気ダイバージェンス（ショートエントリー）：
      価格が高値を切り上げているのに対し、ROCが高値を切り下げている状態。
      下落転換の可能性を示唆。
    """
    
    def __init__(
        self,
        period: int = 12,
        lookback: int = 30,
        params: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            period: ROCの計算期間
            lookback: ダイバージェンス検出のルックバック期間
            params: その他のパラメータ（オプション）
        """
        super().__init__("ROCDivergence")
        
        # パラメータの設定
        self.period = period
        self.lookback = lookback
        
        # インジケーターの初期化
        self.roc = ROC(period=period)
        self.divergence = DivergenceSignal(lookback=lookback)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ROCダイバージェンスシグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            シグナル配列（1: ロング, -1: ショート, 0: シグナルなし）
        """
        # データフレームの作成
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        
        # ROCの計算
        roc_values = self.roc.calculate(data)
        
        # ダイバージェンスの検出
        signals = self.divergence.generate(df, roc_values)
        
        return signals 