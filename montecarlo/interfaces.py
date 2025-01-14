from typing import Protocol, List, Dict
from backtesting.trade import Trade
import numpy as np

class ITradeSimulator(Protocol):
    """トレードシミュレータのインターフェース"""
    def simulate_trades(self, trades: List[Trade], initial_capital: float) -> List[Trade]:
        """トレードをシミュレートする"""
        ...

class IEquityCurveCalculator(Protocol):
    """エクイティカーブ計算のインターフェース"""
    def calculate(self, trades: List[Trade], initial_capital: float) -> List[float]:
        """エクイティカーブを計算する"""
        ...

class IStatisticsCalculator(Protocol):
    """統計計算のインターフェース"""
    def calculate(self, results: List[Dict], initial_capital: float, confidence_level: float) -> Dict:
        """統計を計算する"""
        ...

class IResultVisualizer(Protocol):
    """結果の可視化インターフェース"""
    def visualize(self, equity_curves: List[List[float]], confidence_level: float) -> None:
        """結果を可視化する"""
        ... 