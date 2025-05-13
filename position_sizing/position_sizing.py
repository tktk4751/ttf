from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class PositionSizingParams:
    """ポジションサイジングに必要なパラメータを保持するデータクラス"""
    entry_price: float  # エントリー価格
    stop_loss_price: float  # ストップロス価格
    capital: float  # 利用可能な資金
    leverage: float = 1.0  # レバレッジ倍率
    risk_per_trade: float = 0.01  # 1トレードあたりの最大リスク（資金に対する割合）
    position_risk: float = 0.02  # ポジションに対する最大リスク
    min_position_size: float = 0.0  # 最小ポジションサイズ
    max_position_size: float = float('inf')  # 最大ポジションサイズ
    historical_data: pd.DataFrame = None  # 過去の価格データ
    volatility_window: int = 20  # ボラティリティ計算用ウィンドウサイズ
    
class PositionSizing(ABC):
    """ポジションサイズ計算の基底クラス"""
    
    def __init__(self):
        self._last_calculation: Dict[str, Any] = {}  # 最後の計算結果を保存
        
    @abstractmethod
    def calculate(self, params: PositionSizingParams) -> Dict[str, Any]:
        """
        ポジションサイズを計算する

        Args:
            params: PositionSizingParamsインスタンス

        Returns:
            Dict[str, Any]: {
                'position_size': float,  # ポジションサイズ（数量）
                'risk_amount': float,    # リスク額
                'risk_ratio': float,     # リスク比率
                'leverage_used': float,   # 使用レバレッジ
                'additional_metrics': Dict[str, Any]  # その他の計算指標
            }
        """
        pass
    
    def _calculate_risk_amount(self, params: PositionSizingParams) -> float:
        """リスク額を計算"""
        return abs(params.entry_price - params.stop_loss_price) / params.entry_price
    
    def _calculate_position_value(self, size: float, price: float, leverage: float) -> float:
        """ポジション価値を計算"""
        return size * price * leverage
    
    def _calculate_volatility(self, data: pd.DataFrame, window: int) -> float:
        """ボラティリティを計算"""
        if data is None or len(data) < window:
            return None
        returns = np.log(data['close'] / data['close'].shift(1))
        return returns.rolling(window=window).std().iloc[-1]
    
    def get_last_calculation(self) -> Dict[str, Any]:
        """最後の計算結果を取得"""
        return self._last_calculation.copy()
    
    def validate_position_size(self, size: float, params: PositionSizingParams) -> float:
        """ポジションサイズが制約を満たしているか検証し、必要に応じて調整"""
        size = max(params.min_position_size, min(params.max_position_size, size))
        position_value = self._calculate_position_value(size, params.entry_price, params.leverage)
        
        if position_value > params.capital:
            size = (params.capital / params.entry_price) / params.leverage
            
        return size


# class FixedRiskSizing(PositionSizing):
#     """固定リスクでのポジションサイズ計算（今後の実装用）"""
#     pass

# class KellyPositionSizing(PositionSizing):
#     """ケリー基準でのポジションサイズ計算（今後の実装用）"""
#     pass

