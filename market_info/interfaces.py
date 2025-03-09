from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from enum import Enum

class Timeframe(Enum):
    """時間枠を表す列挙型"""
    MINUTE_1 = 1  # 1分足
    MINUTE_5 = 5  # 5分足
    MINUTE_15 = 15  # 15分足
    MINUTE_30 = 30  # 30分足
    HOUR_1 = 60  # 1時間足
    HOUR_4 = 240  # 4時間足
    DAY_1 = 1440  # 日足
    WEEK_1 = 10080  # 週足

@dataclass
class VolatilityInfo:
    """ボラティリティ情報を保持するデータクラス"""
    # 現在のタイムフレームでのボラティリティ
    timeframe_volatility: float  # 現在のタイムフレームでのボラティリティ（%）
    timeframe_volatility_ema: float  # 現在のタイムフレームでのボラティリティのEMA（%）
    timeframe_volatility_amount: float  # 現在のタイムフレームでのボラティリティ（金額）
    timeframe_volatility_amount_ema: float  # 現在のタイムフレームでのボラティリティのEMA（金額）
    
    # 日次ベースのボラティリティ
    daily_volatility: float  # 日次ボラティリティ（%）
    annual_volatility: float  # 年次ボラティリティ（%）
    daily_volatility_ema: float  # 日次ボラティリティの21日EMA（%）
    annual_volatility_ema: float  # 年次ボラティリティの21日EMA（%）
    daily_volatility_amount: float  # 日次ボラティリティ（金額）
    annual_volatility_amount: float  # 年次ボラティリティ（金額）
    daily_volatility_amount_ema: float  # 日次ボラティリティの21日EMA（金額）
    annual_volatility_amount_ema: float  # 年次ボラティリティの21日EMA（金額）

class IMarketInfoCalculator(ABC):
    """市場情報計算のインターフェース"""
    
    @abstractmethod
    def calculate_volatility(self, prices: np.ndarray, timeframe: Timeframe = Timeframe.DAY_1) -> VolatilityInfo:
        """ボラティリティを計算する
        
        Args:
            prices: 価格データ（終値）の配列
            timeframe: データの時間枠（デフォルト：日足）
            
        Returns:
            VolatilityInfo: ボラティリティ情報
        """
        pass
    
    @abstractmethod
    def calculate_efficiency_ratio(self, prices: np.ndarray, period: int = 10) -> float:
        """効率比（ER）を計算する
        
        Args:
            prices: 価格データ（終値）の配列
            period: 計算期間
            
        Returns:
            float: 効率比
        """
        pass 