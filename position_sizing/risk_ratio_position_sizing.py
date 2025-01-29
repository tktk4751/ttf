#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import numpy as np
import pandas as pd

from .position_sizing import PositionSizing

class RiskRatioPositionSizing(PositionSizing):
    """
    口座残高に対する損失率を基にしたポジションサイジング
    
    指定された損失率（例：2%）に基づいて、ストップロス到達時の損失額が
    口座残高の指定された割合となるようにポジションサイズを計算します。
    """
    
    def __init__(self, risk_ratio: float = 0.02):
        """
        初期化
        
        Args:
            risk_ratio: 口座残高に対する損失率（デフォルト: 0.02 = 2%）
        """
        super().__init__("RiskRatioPositionSizing")
        self.risk_ratio = risk_ratio
    
    def calculate(
        self,
        entry_price: float,
        stop_price: float,
        balance: float,
        position: int = 1,
        data: Union[pd.DataFrame, np.ndarray] = None
    ) -> float:
        """
        ポジションサイズを計算
        
        Args:
            entry_price: エントリー価格
            stop_price: ストップロス価格
            balance: 現在の口座残高
            position: ポジション方向 (1: ロング, -1: ショート)
            data: 価格データ（このメソッドでは使用しない）
            
        Returns:
            float: ポジションサイズ（単位：BTC）
        """
        # エントリー価格とストップロス価格の差（絶対値）
        price_diff = abs(entry_price - stop_price)
        if price_diff == 0:
            return 0.0
        
        # 許容損失額（口座残高 × リスク率）
        max_loss = balance * self.risk_ratio
        
        # ポジションサイズの計算
        # 損失額 = (エントリー価格 - ストップロス価格) × ポジションサイズ = max_loss
        # したがって、ポジションサイズ = max_loss / price_diff
        position_size = max_loss / price_diff
        
        return position_size
    
    def calculate_stop_distance(
        self,
        entry_price: float,
        position_size: float,
        balance: float,
        position: int = 1
    ) -> float:
        """
        ストップロス価格までの距離を計算
        
        Args:
            entry_price: エントリー価格
            position_size: ポジションサイズ
            balance: 現在の口座残高
            position: ポジション方向 (1: ロング, -1: ショート)
            
        Returns:
            float: ストップロス価格までの距離（価格単位）
        """
        if position_size == 0:
            return 0.0
        
        # 許容損失額（口座残高 × リスク率）
        max_loss = balance * self.risk_ratio
        
        # ストップロス価格までの距離を計算
        # 損失額 = distance × position_size = max_loss
        # したがって、distance = max_loss / position_size
        distance = max_loss / position_size
        
        return distance
    
    def calculate_stop_price(
        self,
        entry_price: float,
        position_size: float,
        balance: float,
        position: int = 1
    ) -> float:
        """
        ストップロス価格を計算
        
        Args:
            entry_price: エントリー価格
            position_size: ポジションサイズ
            balance: 現在の口座残高
            position: ポジション方向 (1: ロング, -1: ショート)
            
        Returns:
            float: ストップロス価格
        """
        distance = self.calculate_stop_distance(entry_price, position_size, balance, position)
        
        # ポジション方向に応じてストップロス価格を計算
        if position == 1:  # ロング
            return entry_price - distance
        else:  # ショート
            return entry_price + distance 