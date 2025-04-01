from typing import Dict, Any
import numpy as np
import pandas as pd
from position_sizing.position_sizing import PositionSizing, PositionSizingParams
from position_sizing.interfaces import IPositionManager
from indicators.alpha_volatility import AlphaVolatility

class AlphaVolatilitySizing(PositionSizing, IPositionManager):
    """アルファボラティリティに応じたポジションサイジング"""
    
    def __init__(
        self,
        er_period: int = 21,
        max_vol_period: int = 89,
        min_vol_period: int = 13,
        smoothing_period: int = 14,
        volatility_multiplier: float = 4.0,
        risk_percent: float = 0.01,
        leverage: float = 1.0,
        min_risk_distance_percent: float = 0.001,  # 最小リスク距離（価格の割合）
        max_position_percent: float = 0.5  # 資金に対する最大ポジションサイズの割合
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_vol_period: ボラティリティ期間の最大値（デフォルト: 89）
            min_vol_period: ボラティリティ期間の最小値（デフォルト: 13）
            smoothing_period: ハイパースムーサー期間（デフォルト: 14）
            volatility_multiplier: ボラティリティの乗数（価格変動の許容範囲）
            risk_percent: リスク許容度（資金に対する割合、例: 0.01 = 1%）
            leverage: レバレッジ倍率
            min_risk_distance_percent: 最小リスク距離（価格のパーセンテージ、例: 0.002 = 0.2%）
            max_position_percent: 資金に対する最大ポジションサイズの割合（例: 0.5 = 50%）
        """
        super().__init__()
        self.er_period = er_period
        self.max_vol_period = max_vol_period
        self.min_vol_period = min_vol_period
        self.smoothing_period = smoothing_period
        self.volatility_multiplier = volatility_multiplier
        self.risk_percent = risk_percent
        self.leverage = leverage
        self.min_risk_distance_percent = min_risk_distance_percent
        self.max_position_percent = max_position_percent
        self.alpha_volatility = AlphaVolatility(
            er_period=er_period,
            max_vol_period=max_vol_period,
            min_vol_period=min_vol_period,
            smoothing_period=smoothing_period
        )
    
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
            risk_per_trade=self.risk_percent  # risk_percentを正しく渡す
        )
        
        result = self.calculate(params)
        return result['position_size']
    
    def calculate(self, params: PositionSizingParams) -> Dict[str, Any]:
        """
        アルファボラティリティベースのポジションサイズ計算
        
        Args:
            params: ポジションサイジングパラメータ
            
        Returns:
            Dict[str, Any]: 計算結果（position_sizeはUSD建て金額）
        """
        # 履歴データの確認
        if params.historical_data is None or len(params.historical_data) < self.max_vol_period:
            # 履歴データがない場合や少ない場合は、安全な固定値を使用
            risk_distance_percent = 0.03  # 3%
            volatility = risk_distance_percent / self.volatility_multiplier
            efficiency_ratio = 0.5  # デフォルト値
        else:
            # アルファボラティリティの計算
            self.alpha_volatility.calculate(params.historical_data)
            
            # 最新のボラティリティ値を取得（%ベース、すでに100倍されている）
            volatility_percent = self.alpha_volatility.get_percent_volatility()[-1] / 100  # 100で割って小数に戻す
            volatility = volatility_percent
            
            # 効率比の取得
            efficiency_ratio = self.alpha_volatility.get_efficiency_ratio()[-1]
            
            # ボラティリティに基づくリスク距離の割合を計算
            risk_distance_percent = volatility * self.volatility_multiplier
        
        # 最小リスク距離を保証
        risk_distance_percent = max(risk_distance_percent, self.min_risk_distance_percent)
        
        # 許容リスク額の計算（資金 × リスク許容度）
        # 重要: params.risk_per_tradeを使用することで、外部から渡されたリスク設定が反映される
        risk_amount = params.capital * params.risk_per_trade
        
        # USD建てのポジションサイズを計算
        # リスク額 / リスク距離の割合 = ポジションサイズ
        position_size_usd = (risk_amount / risk_distance_percent) * params.leverage
        
        # 効率比に基づいたポジションサイズの調整
        # トレンドが強い（効率比が高い）場合はポジションサイズを増やし、
        # トレンドが弱い（効率比が低い）場合はポジションサイズを縮小
        if params.historical_data is not None and len(params.historical_data) >= self.max_vol_period:
            # 効率比による調整係数（0.3〜2.0の範囲）
            # 効率比が1.0（最大）の場合は2.0倍、0.0（最小）の場合は0.3倍
            er_factor = 0.3 + (efficiency_ratio * 1.7)
            position_size_usd *= er_factor
        
        # 最大ポジションサイズの制限を適用
        max_position_usd = params.capital * self.max_position_percent * params.leverage
        position_size_usd = min(position_size_usd, max_position_usd)
        
        # 資産の数量を計算（表示用）
        asset_quantity = position_size_usd / params.entry_price if params.entry_price > 0 else 0
        
        # 絶対的なリスク距離（価格単位）の計算
        risk_distance_absolute = params.entry_price * risk_distance_percent
        
        # 結果の作成
        result = {
            'position_size': position_size_usd,  # USDベースの金額
            'asset_quantity': asset_quantity,    # 資産数量（単位数）
            'risk_amount': risk_amount,
            'risk_ratio': self.risk_percent,
            'leverage_used': params.leverage,
            'additional_metrics': {
                'volatility': volatility,
                'volatility_multiplier': self.volatility_multiplier,
                'risk_distance_percent': risk_distance_percent,
                'risk_distance_absolute': risk_distance_absolute,
                'max_position_usd': max_position_usd,
                'risk_per_usd': risk_distance_percent / self.leverage,
                'efficiency_ratio': efficiency_ratio
            }
        }
        
        self._last_calculation = result
        return result 