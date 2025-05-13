from typing import Dict, Any
from position_sizing.position_sizing import PositionSizing, PositionSizingParams
from indicators.atr import ATR
import pandas as pd

class ATRBasedPositionSizing(PositionSizing):
    """ATRベースのポジションサイジング"""
    
    def __init__(self, atr_period: int = 14, risk_per_trade: float = 0.01, atr_multiplier: float = 2.0):
        """
        初期化
        
        Args:
            atr_period: ATR計算期間
            risk_per_trade: 1トレードあたりのリスク（初期資金に対する割合）
            atr_multiplier: ATRの乗数
        """
        super().__init__()
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.atr_indicator = ATR(period=atr_period)
        self._current_position = None
        self._df = None  # データフレームを保持
        self._current_index = 0  # 現在のインデックス
    
    def initialize(self, df: pd.DataFrame) -> None:
        """
        バックテスト用の初期化
        
        Args:
            df: バックテストに使用するデータフレーム
        """
        self._df = df
        self._current_index = 0
    
    def update(self, index: int) -> None:
        """
        現在のインデックスを更新
        
        Args:
            index: 新しいインデックス
        """
        self._current_index = index
    
    def can_enter(self) -> bool:
        """
        新しいポジションを開始できるかどうかを判定
        
        Returns:
            bool: 新しいポジションを開始できる場合はTrue、そうでない場合はFalse
        """
        return self._current_position is None
    
    def on_position_opened(self, trade) -> None:
        """
        ポジションが開始された時の処理
        
        Args:
            trade: 開始されたトレード情報
        """
        self._current_position = trade
    
    def on_position_closed(self, trade) -> None:
        """
        ポジションが終了した時の処理
        
        Args:
            trade: 終了したトレード情報
        """
        self._current_position = None
    
    def calculate_position_size(self, price: float, capital: float, historical_data=None) -> float:
        """
        バックテスター用のポジションサイズ計算メソッド
        
        Args:
            price: 現在の価格
            capital: 現在の資金
            historical_data: 使用しない（後方互換性のため残す）
            
        Returns:
            float: ポジションサイズ
        """
        if self._df is None or self._current_index < self.atr_period:
            # データが不十分な場合は、価格の1%をATRとして使用
            atr = price * 0.01
        else:
            # 現在のインデックスまでのデータでATRを計算
            lookback_data = self._df.iloc[max(0, self._current_index - self.atr_period):self._current_index + 1]
            if len(lookback_data) >= self.atr_period:
                atr = self.atr_indicator.calculate(lookback_data)[-1]
            else:
                atr = price * 0.01
        
        # リスク額の計算（初期資金 × リスク率）
        risk_amount = capital * self.risk_per_trade
        
        # ポジションサイズの計算（USD）
        # (初期資金 × リスク率) ÷ (ATR × ATR乗数) × エントリー価格
        position_size = (risk_amount / (atr * self.atr_multiplier)) * price
        
        # ポジションサイズの検証と調整
        params = PositionSizingParams(
            entry_price=price,
            stop_loss_price=price,  # 一時的な値
            capital=capital,
            historical_data=self._df
        )
        position_size = self.validate_position_size(position_size, params)
        
        return position_size
    
    def calculate(self, params: PositionSizingParams) -> Dict[str, Any]:
        """
        ポジションサイズを計算
        
        Args:
            params: ポジションサイジングパラメータ
            
        Returns:
            Dict[str, Any]: 計算結果
        """
        if params.historical_data is None:
            raise ValueError("ATR計算には過去のデータが必要です")
        
        # ATRの計算
        atr = self.atr_indicator.calculate(params.historical_data)[-1]
        
        # リスク額の計算（初期資金 × リスク率）
        risk_amount = params.capital * self.risk_per_trade
        
        # ポジションサイズの計算（USD）
        # (初期資金 × リスク率) ÷ (ATR × ATR乗数) × エントリー価格
        position_size = (risk_amount / (atr * self.atr_multiplier)) * params.entry_price
        
        # ポジションサイズの検証と調整
        position_size = self.validate_position_size(position_size, params)
        
        # 結果の保存
        result = {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_ratio': self.risk_per_trade,
            'leverage_used': params.leverage,
            'additional_metrics': {
                'atr': atr,
                'atr_multiplier': self.atr_multiplier,
                'atr_stop_distance': atr * self.atr_multiplier
            }
        }
        
        self._last_calculation = result
        return result 