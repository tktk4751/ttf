from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from indicators.atr import ATR  # ATRインジケーターをインポート

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

class ATRPositionSizing(PositionSizing):
    """ATRベースのポジションサイズ計算"""
    
    def __init__(self, atr_period: int = 14, atr_multiplier: float = 2.0):
        """
        初期化
        
        Args:
            atr_period: ATR計算期間
            atr_multiplier: ATRの乗数（ストップロス幅の計算用）
        """
        super().__init__()
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.atr_indicator = ATR(period=atr_period)
    
    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """
        ATR（Average True Range）を計算
        
        Args:
            data: OHLC価格データ
            
        Returns:
            float: ATR値
        """
        atr_values = self.atr_indicator.calculate(data)
        return atr_values[-1]
    
    def calculate(self, params: PositionSizingParams) -> Dict[str, Any]:
        """
        ATRベースでポジションサイズを計算
        
        Args:
            params: ポジションサイジングパラメータ
            
        Returns:
            Dict[str, Any]: 計算結果
        """
        if params.historical_data is None:
            raise ValueError("Historical data is required for ATR calculation")
        
        # ATRの計算
        atr = self._calculate_atr(params.historical_data)
        
        # ATRベースのストップロス価格を計算
        if params.entry_price > params.stop_loss_price:  # ロングポジション
            stop_distance = atr * self.atr_multiplier
            stop_price = params.entry_price - stop_distance
        else:  # ショートポジション
            stop_distance = atr * self.atr_multiplier
            stop_price = params.entry_price + stop_distance
        
        # リスク額の計算（口座残高 × リスク率）
        risk_amount = params.capital * params.risk_per_trade
        
        # 価格差に基づくポジションサイズの計算
        price_diff = abs(params.entry_price - stop_price)
        position_size = (risk_amount / price_diff) * params.entry_price
        
        # レバレッジの適用
        position_size = position_size * params.leverage
        
        # ポジションサイズの検証と調整
        position_size = self.validate_position_size(position_size, params)
        
        # 結果の保存
        result = {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_ratio': params.risk_per_trade,
            'leverage_used': params.leverage,
            'additional_metrics': {
                'atr': atr,
                'atr_stop_distance': stop_distance,
                'calculated_stop_price': stop_price
            }
        }
        
        self._last_calculation = result
        return result

# class FixedRiskSizing(PositionSizing):
#     """固定リスクでのポジションサイズ計算（今後の実装用）"""
#     pass

# class KellyPositionSizing(PositionSizing):
#     """ケリー基準でのポジションサイズ計算（今後の実装用）"""
#     pass

