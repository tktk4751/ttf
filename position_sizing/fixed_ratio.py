from typing import Dict, Any
from backtesting.backtester import IPositionManager


class FixedRatioSizing(IPositionManager):
    """固定比率のポジションサイジング"""
    
    def __init__(self, params: Dict[str, Any]):
        """
        コンストラクタ
        
        Args:
            params: パラメータ辞書
                - ratio: 資金に対する割合（0.0-1.0）
                - min_position_size: 最小ポジションサイズ（任意）
                - max_position_size: 最大ポジションサイズ（任意）
                - leverage: レバレッジ（デフォルト: 1）
        """
        self.ratio = params.get('ratio', 1.0)
        self.min_position = params.get('min_position_size')
        self.max_position = params.get('max_position_size')
        self.leverage = params.get('leverage', 1)
    
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
            ポジションサイズ
        """
        # 基本のポジションサイズを計算（資金に対する比率）
        position_ratio = self.ratio
        
        # 最小ポジションサイズの制限（資金に対する比率）
        if self.min_position is not None:
            position_ratio = max(position_ratio, self.min_position)
        
        # 最大ポジションサイズの制限（資金に対する比率）
        if self.max_position is not None:
            position_ratio = min(position_ratio, self.max_position)
        
        # 最終的なポジションサイズを計算（レバレッジを適用）
        size = (capital * position_ratio * self.leverage) / price
        
        return size