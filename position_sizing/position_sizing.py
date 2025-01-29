from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import numpy as np
import pandas as pd

class PositionSizing(ABC):
    """ポジションサイズ計算の基底クラス"""
    
    @abstractmethod
    def calculate(self, capital: float, price: float, **kwargs) -> float:
        """
        ポジションサイズを計算する
        
        Args:
            capital: 総資金
            price: 現在の価格
            **kwargs: 追加のパラメータ
        
        Returns:
            float: ポジションサイズ（数量）
        """
        pass

# class FixedRiskSizing(PositionSizing):
#     """固定リスクでのポジションサイズ計算（今後の実装用）"""
#     pass

# class KellyPositionSizing(PositionSizing):
#     """ケリー基準でのポジションサイズ計算（今後の実装用）"""
#     pass

class ATRPositionSizing(PositionSizing):
    """タートルズトレーディングで使用されるATRベースのポジションサイジング"""
    
    def __init__(self, risk_percent: float = 0.04):
        """
        初期化
        
        Args:
            risk_percent: 1トレードあたりの許容リスク（デフォルト: 4%）
        """
        self.risk_percent = risk_percent
    
    def calculate(self, capital: float, price: float, **kwargs) -> float:
        """
        ATRに基づいてポジションサイズを計算
        
        Args:
            capital: 総資金
            price: 現在の価格
            **kwargs:
                required:
                    - supertrend_band: スーパートレンドのバンド価格（ストップロス価格として使用）
                optional:
                    - atr: ATR値（指定がない場合はスーパートレンドのバンドを使用）
        
        Returns:
            float: ポジションサイズ（数量）
        """
        # スーパートレンドのバンドを取得（必須）
        supertrend_band = kwargs.get('supertrend_band')
        if supertrend_band is None:
            raise ValueError("supertrend_bandは必須パラメータです")
            
        # ストップロス幅の計算
        stop_distance = abs(price - supertrend_band)
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
    
    def get_stop_loss(self, price: float, supertrend_band: float) -> float:
        """
        ストップロス価格を取得
        
        Args:
            price: 現在の価格
            supertrend_band: スーパートレンドのバンド価格
            
        Returns:
            float: ストップロス価格
        """
        return supertrend_band
