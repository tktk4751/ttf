from position_sizing.position_sizing import PositionSizing
from typing import Dict, Any


class FixedRatioSizing(PositionSizing):
    """固定比率でのポジションサイズ計算"""
    
    def __init__(self, params: Dict[str, Any]):
        """
        コンストラクタ
        
        Args:
            params: パラメータ
                - ratio: 資金に対する比率（0.0 ~ 1.0）
                - min_position: 最小ポジションサイズ（USD）
                - max_position: 最大ポジションサイズ（USD）
                - leverage: レバレッジ倍率
        """
        self.ratio = params.get('ratio', 0.99)
        self.min_position = params.get('min_position')
        self.max_position = params.get('max_position')
        self.leverage = params.get('leverage', 1)
    
    def calculate(self, capital: float, price: float) -> float:
        """
        固定比率でのポジションサイズを計算する
        
        Args:
            capital: 総資金
            price: 現在の価格
        
        Returns:
            float: ポジションサイズ（USD）
        """
        # 投資可能金額の計算（証拠金）
        margin = capital * self.ratio
        
        # レバレッジを考慮したポジションサイズの計算
        position_size = margin * self.leverage
        
        # 最小ポジションサイズの適用
        if self.min_position is not None:
            position_size = max(position_size, self.min_position)
        
        # 最大ポジションサイズの適用
        if self.max_position is not None:
            position_size = min(position_size, self.max_position)
        
        # 必要証拠金を超えないように調整
        max_allowed_size = capital * self.leverage
        position_size = min(position_size, max_allowed_size)
        
        return position_size