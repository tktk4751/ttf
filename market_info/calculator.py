import numpy as np
from typing import Tuple
from .interfaces import IMarketInfoCalculator, VolatilityInfo, Timeframe

class MarketInfoCalculator(IMarketInfoCalculator):
    """市場情報計算の実装クラス"""
    
    def __init__(self, ema_period: int = 21, trading_days: int = 365, minutes_per_day: int = 1440):
        """
        初期化
        
        Args:
            ema_period: EMAの計算期間
            trading_days: 年間取引日数
            minutes_per_day: 1日の取引分数（デフォルト：1440分 = 24時間）
        """
        self.ema_period = ema_period
        self.trading_days = trading_days
        self.minutes_per_day = minutes_per_day
    
    def _calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """リターンを計算する
        
        Args:
            prices: 価格データの配列
            
        Returns:
            np.ndarray: リターンの配列
        """
        return np.log(prices[1:] / prices[:-1])
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """指数移動平均を計算する
        
        Args:
            data: データ配列
            period: 計算期間
            
        Returns:
            np.ndarray: EMAの配列
        """
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = data[i] * alpha + ema[i-1] * (1 - alpha)
        
        return ema
    
    def _adjust_volatility_for_timeframe(self, volatility: float, timeframe: Timeframe) -> float:
        """ボラティリティを日次ベースに調整する
        
        Args:
            volatility: 元のタイムフレームでのボラティリティ
            timeframe: データの時間枠
            
        Returns:
            float: 日次ベースに調整されたボラティリティ
        """
        # 1日あたりの期間数を計算
        periods_per_day = self.minutes_per_day / timeframe.value
        
        # ボラティリティを日次ベースに変換
        return volatility * np.sqrt(periods_per_day)
    
    def calculate_volatility(self, prices: np.ndarray, timeframe: Timeframe = Timeframe.DAY_1) -> VolatilityInfo:
        """ボラティリティを計算する
        
        Args:
            prices: 価格データ（終値）の配列
            timeframe: データの時間枠（デフォルト：日足）
            
        Returns:
            VolatilityInfo: ボラティリティ情報
        """
        # リターンの計算
        returns = self._calculate_returns(prices)
        
        # 現在のタイムフレームでのボラティリティ計算
        timeframe_vol = np.std(returns) * 100  # パーセント表示
        
        # 現在のタイムフレームでのボラティリティのEMAを計算
        timeframe_vol_series = np.array([np.std(returns[max(0, i-20):i+1]) * 100 
                                       for i in range(len(returns))])
        timeframe_vol_ema = self._calculate_ema(timeframe_vol_series, self.ema_period)[-1]
        
        # 日次ボラティリティに変換
        daily_vol = self._adjust_volatility_for_timeframe(timeframe_vol, timeframe)
        
        # 年次ボラティリティの計算
        annual_vol = daily_vol * np.sqrt(self.trading_days)
        
        # 日次ボラティリティのEMAを計算
        daily_vol_series = np.array([self._adjust_volatility_for_timeframe(v, timeframe) 
                                   for v in timeframe_vol_series])
        annual_vol_series = daily_vol_series * np.sqrt(self.trading_days)
        
        daily_vol_ema = self._calculate_ema(daily_vol_series, self.ema_period)[-1]
        annual_vol_ema = self._calculate_ema(annual_vol_series, self.ema_period)[-1]
        
        # 金額ベースのボラティリティ計算
        current_price = prices[-1]
        
        # 現在のタイムフレームでの金額ベースボラティリティ
        timeframe_vol_amount = current_price * (timeframe_vol / 100)
        timeframe_vol_amount_ema = current_price * (timeframe_vol_ema / 100)
        
        # 日次・年次の金額ベースボラティリティ
        daily_vol_amount = current_price * (daily_vol / 100)
        annual_vol_amount = current_price * (annual_vol / 100)
        daily_vol_amount_ema = current_price * (daily_vol_ema / 100)
        annual_vol_amount_ema = current_price * (annual_vol_ema / 100)
        
        return VolatilityInfo(
            # 現在のタイムフレームでのボラティリティ
            timeframe_volatility=timeframe_vol,
            timeframe_volatility_ema=timeframe_vol_ema,
            timeframe_volatility_amount=timeframe_vol_amount,
            timeframe_volatility_amount_ema=timeframe_vol_amount_ema,
            
            # 日次・年次ボラティリティ
            daily_volatility=daily_vol,
            annual_volatility=annual_vol,
            daily_volatility_ema=daily_vol_ema,
            annual_volatility_ema=annual_vol_ema,
            daily_volatility_amount=daily_vol_amount,
            annual_volatility_amount=annual_vol_amount,
            daily_volatility_amount_ema=daily_vol_amount_ema,
            annual_volatility_amount_ema=annual_vol_amount_ema
        )
    
    def calculate_efficiency_ratio(self, prices: np.ndarray, period: int = 10) -> float:
        """効率比（ER）を計算する
        
        Args:
            prices: 価格データ（終値）の配列
            period: 計算期間
            
        Returns:
            float: 効率比
        """
        if len(prices) < period:
            return 0.0
        
        # 期間内の価格変化の絶対値
        directional_movement = abs(prices[-1] - prices[-period])
        
        # 期間内の価格変化の合計
        volatility = sum(abs(prices[i] - prices[i-1]) for i in range(-period+1, 0))
        
        # 0除算を防ぐ
        if volatility == 0:
            return 0.0
        
        return directional_movement / volatility 