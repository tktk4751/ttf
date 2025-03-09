from typing import Dict, Any
from .interfaces import IPositionManager


class FixedRatioSizing(IPositionManager):
    """固定比率のポジションサイジング"""
    
    def __init__(self, ratio: float, leverage: float = 1.0):
        """
        コンストラクタ
        
        Args:
            ratio: 資金に対する割合（0.0-1.0）
            leverage: レバレッジ（デフォルト: 1）
        """
        if not isinstance(ratio, (int, float)) or ratio <= 0 or ratio > 1:
            raise ValueError("ratioは0より大きく1以下の数値である必要があります")
        if not isinstance(leverage, (int, float)) or leverage <= 0:
            raise ValueError("leverageは0より大きい数値である必要があります")
        
        self.ratio = float(ratio)
        self.leverage = float(leverage)
    
    def can_enter(self) -> bool:
        """新規ポジションを取れるかどうか"""
        return True
    
    def calculate_position_size(self, price: float, capital: float) -> Dict[str, float]:
        """
        ポジションサイズを計算する
        
        Args:
            price: 現在の価格
            capital: 現在の資金
        
        Returns:
            Dict[str, float]: ポジションサイズ情報
                - position_size: USD建てのポジションサイズ
        """
        try:
            # パラメータの検証
            if not isinstance(price, (int, float)) or price <= 0:
                return {'position_size': 0.0}
            if not isinstance(capital, (int, float)) or capital <= 0:
                return {'position_size': 0.0}
            
            # 利用可能な資金を計算
            available_capital = float(capital) * self.ratio
            
            # レバレッジを適用してUSD建てのポジションサイズを計算
            position_size_usd = available_capital * self.leverage
            
            # 無効な値のチェック
            if not isinstance(position_size_usd, (int, float)) or position_size_usd <= 0:
                return {'position_size': 0.0}
            
            return {'position_size': float(position_size_usd)}
            
        except Exception as e:
            print(f"ポジションサイズの計算中にエラーが発生しました: {e}")
            return {'position_size': 0.0}