#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.macd import MACD
from .divergence_signal import DivergenceSignal


class MACDDivergenceSignal(BaseSignal, IEntrySignal):
    """
    MACDダイバージェンスシグナル
    
    価格とMACDの間のダイバージェンスを検出し、エントリーシグナルを生成します。
    
    - 強気ダイバージェンス（ロングエントリー）：
      価格が安値を切り下げているのに対し、MACDが安値を切り上げている状態。
      上昇転換の可能性を示唆。
    
    - 弱気ダイバージェンス（ショートエントリー）：
      価格が高値を切り上げているのに対し、MACDが高値を切り下げている状態。
      下落転換の可能性を示唆。
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        lookback: int = 30,
        params: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            fast_period: 短期EMAの期間
            slow_period: 長期EMAの期間
            signal_period: シグナルEMAの期間
            lookback: ダイバージェンス検出のルックバック期間
            params: その他のパラメータ（オプション）
        """
        super().__init__("MACDDivergence")
        
        # パラメータの設定
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.lookback = lookback
        
        # インジケーターの初期化
        self.macd = MACD(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period
        )
        self.divergence = DivergenceSignal(lookback=lookback)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        MACDダイバージェンスシグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            シグナル配列（1: ロング, -1: ショート, 0: シグナルなし）
        """
        # データフレームの作成
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        
        # MACDの計算
        macd_result = self.macd.calculate(data)
        
        # ダイバージェンスの検出（MACDラインを使用）
        signals = self.divergence.generate(df, macd_result.macd)
        
        return signals 