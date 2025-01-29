#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from ..base.strategy import BaseStrategy
from signals.implementations.atr.exit import ATRExitSignal

class ATRExitStrategy(BaseStrategy):
    """ATRエグジットを使用する戦略の基底クラス"""
    
    def __init__(self, name: str, atr_period: int = 14, atr_multiplier: float = 2.0):
        """
        初期化
        
        Args:
            name: 戦略名
            atr_period: ATRの期間
            atr_multiplier: ATRの乗数
        """
        super().__init__(name)
        self._parameters.update({
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier
        })
        
        # ATRエグジットシグナルの初期化
        self._exit_signal = ATRExitSignal(
            period=atr_period,
            multiplier=atr_multiplier
        )
    
    def get_entry_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """
        エントリー価格を取得する
        
        Args:
            data: 価格データ
            position: ポジション方向 (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1 = 最新のデータ）
            
        Returns:
            float: エントリー価格
        """
        # デフォルトの実装を使用（現在の終値）
        return super().get_entry_price(data, position, index)
    
    def get_stop_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """
        ストップロス価格を取得する
        
        Args:
            data: 価格データ
            position: ポジション方向 (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1 = 最新のデータ）
            
        Returns:
            float: ストップロス価格
        """
        # エントリー価格を取得
        entry_price = self.get_entry_price(data, position, index)
        
        # ATRエグジットシグナルを使用してストップロス価格を計算
        return self._exit_signal.calculate_stop_price(data, entry_price, position) 