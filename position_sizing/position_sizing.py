

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
