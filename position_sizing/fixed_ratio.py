from position_sizing.position_sizing import PositionSizing
from typing import Dict, Any, Union


class FixedRatioSizing(PositionSizing):
    """固定比率でのポジションサイズ計算"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        コンストラクタ
        
        Args:
            params: パラメータ
                - ratio: 資金に対する比率（0.0 ~ 1.0）
                - min_position: 最小ポジションサイズ
                - max_position: 最大ポジションサイズ
        """
        self.params = params or {
            'ratio': 0.02,  # デフォルトは2%
            'min_position': 0.001,  # 最小ポジションサイズ
            'max_position': None  # 最大ポジションサイズ（Noneの場合は制限なし）
        }
    
    def calculate(self, capital: float, price: float, **kwargs) -> float:
        """
        固定比率でのポジションサイズを計算する
        
        Args:
            capital: 総資金
            price: 現在の価格
            **kwargs: 追加のパラメータ
        
        Returns:
            float: ポジションサイズ（数量）
        """
        # パラメータの取得
        ratio = self.params['ratio']
        min_position = self.params['min_position']
        max_position = self.params['max_position']
        
        # 投資金額の計算
        position_value = capital * ratio
        
        # 数量の計算
        position_size = position_value / price
        
        # 最小ポジションサイズの適用
        position_size = max(position_size, min_position)
        
        # 最大ポジションサイズの適用（設定されている場合）
        if max_position is not None:
            position_size = min(position_size, max_position)
        
        return position_size