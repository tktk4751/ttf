from typing import Dict, Any
import numpy as np
import pandas as pd
from position_sizing.position_sizing import PositionSizing, PositionSizingParams
from position_sizing.interfaces import IPositionManager

class VolatilityStdSizing(PositionSizing, IPositionManager):
    """標準偏差ベースのボラティリティに応じたポジションサイジング"""
    
    def __init__(
        self,
        volatility_period: int = 21,
        volatility_multiplier: float = 2.0,
        risk_percent: float = 0.01,
        leverage: float = 1.0
    ):
        """
        コンストラクタ
        
        Args:
            volatility_period: ボラティリティ計算期間
            volatility_multiplier: ボラティリティの乗数（価格変動の許容範囲）
            risk_percent: リスク許容度（資金に対する割合、例: 0.01 = 1%）
            leverage: レバレッジ倍率
        """
        super().__init__()
        self.volatility_period = volatility_period
        self.volatility_multiplier = volatility_multiplier
        self.risk_percent = risk_percent
        self.leverage = leverage
    
    def can_enter(self) -> bool:
        """新規ポジションを取れるかどうか"""
        return True
    
    def calculate_position_size(self, price: float, capital: float) -> float:
        """
        シンプルなポジションサイズ計算（IPositionManagerインターフェース用）
        
        Args:
            price: 現在の価格
            capital: 現在の資金
            
        Returns:
            float: ポジションサイズ（USD建て）
        """
        params = PositionSizingParams(
            entry_price=price,
            stop_loss_price=price * 0.95,  # デフォルトのストップロス（5%）
            capital=capital,
            leverage=self.leverage,
            risk_per_trade=self.risk_percent
        )
        
        result = self.calculate(params)
        return result['position_size']
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        ボラティリティ（標準偏差）を計算
        
        Args:
            data: OHLC価格データ
            
        Returns:
            float: ボラティリティ値
        """
        # 対数収益率を計算
        returns = np.log(data['close'] / data['close'].shift(1))
        
        # 標準偏差を計算（年率化なし）
        volatility = returns.rolling(window=self.volatility_period).std().iloc[-1]
        
        return volatility
    
    def calculate(self, params: PositionSizingParams) -> Dict[str, Any]:
        """
        標準偏差ベースのポジションサイズ計算
        
        Args:
            params: ポジションサイジングパラメータ
            
        Returns:
            Dict[str, Any]: 計算結果
        """
        if params.historical_data is None or len(params.historical_data) < self.volatility_period:
            raise ValueError("Historical data is required for volatility calculation")
        
        # ボラティリティの計算
        volatility = self._calculate_volatility(params.historical_data)
        
        # ボラティリティに基づくリスク量の計算（価格変動の許容範囲）
        # volatilityは対数収益率の標準偏差なので、価格変動に変換
        risk_distance = params.entry_price * (1 - np.exp(-volatility * self.volatility_multiplier))
        
        # 許容リスク額の計算（資金 × リスク許容度）
        risk_amount = params.capital * self.risk_percent
        
        # ポジションサイズの計算
        # リスク額 ÷ リスク距離 = 取引数量
        position_size = risk_amount / risk_distance
        
        # レバレッジの適用
        position_size = position_size * params.leverage
        
        # ポジションサイズの検証と調整
        position_size = self.validate_position_size(position_size, params)
        
        # 結果の作成
        result = {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_ratio': self.risk_percent,
            'leverage_used': params.leverage,
            'additional_metrics': {
                'volatility': volatility,
                'volatility_multiplier': self.volatility_multiplier,
                'risk_distance': risk_distance,
                'risk_amount_per_unit': risk_amount / position_size if position_size > 0 else 0
            }
        }
        
        self._last_calculation = result
        return result 