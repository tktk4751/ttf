from typing import Dict, Any
from backtesting.backtester import IPositionManager


class FixedRatioSizing(IPositionManager):
    """固定比率のポジションサイジング"""
    
    def __init__(self, ratio: float, leverage: float = 1.0):
        """
        コンストラクタ
        
        Args:
            ratio: 資金に対する割合（0.0-1.0）
            leverage: レバレッジ（デフォルト: 1）
        """
        self.ratio = ratio
        self.leverage = leverage
    
    def can_enter(self) -> bool:
        """新規ポジションを取れるかどうか"""
        return True
    
    def calculate_position_size(self, price: float, capital: float) -> float:
        """
        ポジションサイズを計算する
        
        Args:
            price: 現在の価格
            capital: 現在の資金
        
        Returns:
            float: ポジションサイズ（USD建て）
        """
        # 利用可能な資金を計算
        available_capital = capital * self.ratio
        
        # レバレッジを適用してUSD建てのポジションサイズを計算
        position_size_usd = available_capital * self.leverage
        
        return position_size_usd  # USD建てのポジションサイズを返す