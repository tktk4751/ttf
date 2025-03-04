#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.tfc import TFC


class TFCDirectionSignal(BaseSignal, IDirectionSignal):
    """
    TFCを使用した方向性シグナル
    - TFCのトレンドが1: ロング方向 (1)
    - TFCのトレンドが-1: ショート方向 (-1)
    """
    
    def __init__(
        self,
        kama_period: int = 10,
        kama_fast: int = 2,
        kama_slow: int = 30,
        atr_period: int = 10,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 2.0
    ):
        """
        コンストラクタ
        
        Args:
            kama_period: KAMAの効率比の計算期間（デフォルト: 10）
            kama_fast: KAMAの速い移動平均の期間（デフォルト: 2）
            kama_slow: KAMAの遅い移動平均の期間（デフォルト: 30）
            atr_period: ATRの期間（デフォルト: 10）
            upper_multiplier: アッパーバンドのATR乗数（デフォルト: 2.0）
            lower_multiplier: ロワーバンドのATR乗数（デフォルト: 2.0）
        """
        params = {
            'kama_period': kama_period,
            'kama_fast': kama_fast,
            'kama_slow': kama_slow,
            'atr_period': atr_period,
            'upper_multiplier': upper_multiplier,
            'lower_multiplier': lower_multiplier
        }
        super().__init__(
            f"KAMAKeltnerTFCDirection({kama_period}, {kama_fast}, {kama_slow}, {atr_period}, {upper_multiplier}, {lower_multiplier})",
            params
        )
        self._tfc = TFC(
            kama_period=kama_period,
            kama_fast=kama_fast,
            kama_slow=kama_slow,
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
            シグナルの配列 (1: ロング方向, -1: ショート方向)
        """
        # TFCの計算
        self._tfc.calculate(data)
        
        # トレンド方向をそのままシグナルとして使用
        return self._tfc.get_trend()
    
    def get_tfc(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        TFC値を取得する
        
        Args:
            data: 価格データ
        
        Returns:
            np.ndarray: TFC値の配列
        """
        # TFCの計算
        self._tfc.calculate(data)
        return self._tfc._values  # TFC値を返す 