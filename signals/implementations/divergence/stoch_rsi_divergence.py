#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.stochastic_rsi import StochasticRSI
from .divergence_signal import DivergenceSignal


class StochRSIDivergenceSignal(BaseSignal, IEntrySignal):
    """
    ストキャスティクスRSIダイバージェンスシグナル
    
    価格とストキャスティクスRSIの間のダイバージェンスを検出し、エントリーシグナルを生成します。
    
    - 強気ダイバージェンス（ロングエントリー）：
      価格が安値を切り下げているのに対し、ストキャスティクスRSIが安値を切り上げている状態。
      上昇転換の可能性を示唆。
    
    - 弱気ダイバージェンス（ショートエントリー）：
      価格が高値を切り上げているのに対し、ストキャスティクスRSIが高値を切り下げている状態。
      下落転換の可能性を示唆。
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
        lookback: int = 30,
        params: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            rsi_period: RSIの計算期間
            stoch_period: ストキャスティクスの期間
            k_period: %Kの期間
            d_period: %Dの期間
            lookback: ダイバージェンス検出のルックバック期間
            params: その他のパラメータ（オプション）
        """
        super().__init__("StochRSIDivergence")
        
        # パラメータの設定
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.k_period = k_period
        self.d_period = d_period
        self.lookback = lookback
        
        # インジケーターの初期化
        self.stoch_rsi = StochasticRSI(
            rsi_period=rsi_period,
            stoch_period=stoch_period,
            k_period=k_period,
            d_period=d_period
        )
        self.divergence = DivergenceSignal(lookback=lookback)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ストキャスティクスRSIダイバージェンスシグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            シグナル配列（1: ロング, -1: ショート, 0: シグナルなし）
        """
        # データフレームの作成
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        
        # ストキャスティクスRSIの計算
        stoch_rsi_result = self.stoch_rsi.calculate(data)
        
        # ダイバージェンスの検出（%Kラインを使用）
        signals = self.divergence.generate(df, stoch_rsi_result.k)
        
        return signals 