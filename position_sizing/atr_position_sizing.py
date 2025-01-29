#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from .position_sizing import PositionSizing
from indicators.atr import ATR
from indicators.supertrend import Supertrend, SupertrendResult


class ATRPositionSizing(PositionSizing):
    """タートルズトレーディングで使用されるATRベースのポジションサイジング"""
    
    def __init__(
        self,
        risk_percent: float = 0.04,
        atr_period: int = 14,
        supertrend_period: int = 10,
        supertrend_multiplier: float = 3.0
    ):
        """
        初期化
        
        Args:
            risk_percent: 1トレードあたりの許容リスク（デフォルト: 4%）
            atr_period: ATRの計算期間（デフォルト: 14）
            supertrend_period: スーパートレンドの計算期間（デフォルト: 10）
            supertrend_multiplier: スーパートレンドの乗数（デフォルト: 3.0）
        """
        self.risk_percent = risk_percent
        self.atr = ATR(period=atr_period)
        self.supertrend = Supertrend(
            period=supertrend_period,
            multiplier=supertrend_multiplier
        )
    
    def calculate(self, capital: float, price: float, **kwargs) -> float:
        """
        ATRに基づいてポジションサイズを計算
        
        Args:
            capital: 総資金
            price: 現在価格
            **kwargs:
                required:
                    - data: 価格データ（OHLCV）
                    - index: 現在の価格データのインデックス
                optional:
                    - direction: トレード方向（1: ロング、-1: ショート）
        
        Returns:
            float: ポジションサイズ（数量）
        """
        # 必須パラメータの取得
        data = kwargs.get('data')
        index = kwargs.get('index')
        if data is None or index is None:
            raise ValueError("data と index は必須パラメータです")
            
        # トレード方向の取得（デフォルトはロング）
        direction = kwargs.get('direction', 1)
        
        # スーパートレンドの計算
        supertrend_result = self.supertrend.calculate(data)
        
        # ストップロス価格の取得
        stop_price = self.get_stop_loss(
            price=price,
            supertrend_result=supertrend_result,
            index=index,
            direction=direction
        )
        
        # ストップロス幅の計算
        stop_distance = abs(price - stop_price)
        if stop_distance == 0:
            return 0.0
            
        # リスク額の計算
        risk_amount = capital * self.risk_percent
        
        # 1単位あたりの損失額
        unit_loss = stop_distance
        
        # ポジションサイズの計算
        position_size = risk_amount / unit_loss
        
        # 端数処理（小数点以下4桁に丸める）
        position_size = round(position_size, 4)
        
        return position_size
    
    def get_stop_loss(
        self,
        price: float,
        supertrend_result: SupertrendResult,
        index: int,
        direction: int = 1
    ) -> float:
        """
        ストップロス価格を取得
        
        Args:
            price: 現在価格
            supertrend_result: スーパートレンドの計算結果
            index: 現在の価格データのインデックス
            direction: トレード方向（1: ロング、-1: ショート）
            
        Returns:
            float: ストップロス価格
        """
        if direction == 1:  # ロングポジション
            
            return supertrend_result.lower_band[index]
        else:  # ショートポジション
            return supertrend_result.upper_band[index] 