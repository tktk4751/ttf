from typing import Dict, Any
from position_sizing.position_sizing import PositionSizing, PositionSizingParams
from position_sizing.interfaces import IPositionManager
from indicators.atr import ATR

class ATRVolatilitySizing(PositionSizing, IPositionManager):
    """ATRベースのボラティリティに応じたポジションサイジング"""
    
    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        risk_percent: float = 0.01,
        leverage: float = 1.0
    ):
        """
        コンストラクタ
        
        Args:
            atr_period: ATR計算期間
            atr_multiplier: ATRの乗数（価格変動の許容範囲）
            risk_percent: リスク許容度（資金に対する割合、例: 0.01 = 1%）
            leverage: レバレッジ倍率
        """
        super().__init__()
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.risk_percent = risk_percent
        self.leverage = leverage
        self.atr_indicator = ATR(period=atr_period)
    
    def can_enter(self) -> bool:
        """新規ポジションを取れるかどうか"""
        return True
    

    
    def calculate(self, params: PositionSizingParams) -> Dict[str, Any]:
        """
        ATRベースのポジションサイズ計算
        
        Args:
            params: ポジションサイジングパラメータ
            
        Returns:
            Dict[str, Any]: 計算結果
        """
        if params.historical_data is None or len(params.historical_data) < self.atr_period:
            raise ValueError("Historical data is required for ATR calculation")
        
        # ATRの計算
        atr = self.atr_indicator.calculate(params.historical_data)[-1]
        
        # ATRに基づくリスク量の計算（価格変動の許容範囲）
        risk_distance = atr * self.atr_multiplier
        
        # 許容リスク額の計算（資金 × リスク許容度）
        risk_amount = params.capital * self.risk_percent
        
        # ポジションサイズの計算
        # リスク額 ÷ （ATR × 乗数）= 取引数量
        position_size = (risk_amount / risk_distance) * params.entry_price
        
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
                'atr': atr,
                'atr_multiplier': self.atr_multiplier,
                'risk_distance': risk_distance,
                'risk_amount_per_unit': risk_amount / position_size if position_size > 0 else 0
            }
        }
        
        self._last_calculation = result
        return result 