from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Type, Callable
import pandas as pd
from strategies.strategy import Strategy

class IOptimizer(ABC):
    """最適化のインターフェース"""
    
    @abstractmethod
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """最適化を実行し、最適なパラメータとスコアを返す"""
        pass
    
    @abstractmethod
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """最適なパラメータを取得"""
        pass
    
    @abstractmethod
    def get_best_score(self) -> Optional[float]:
        """最高スコアを取得"""
        pass

class BaseOptimizer(IOptimizer, ABC):
    """最適化の基底クラス"""
    
    def __init__(
        self,
        strategy_class: Type[Strategy],
        param_generator: Callable,
        config: Dict[str, Any]
    ):
        """
        Args:
            strategy_class: 最適化対象の戦略クラス
            param_generator: 戦略パラメーターを生成する関数
            config: 設定
        """
        self.strategy_class = strategy_class
        self.param_generator = param_generator
        self.config = config
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.best_trades: Optional[List[Any]] = None
        self._data: Optional[pd.DataFrame] = None
        self._data_dict: Optional[Dict[str, pd.DataFrame]] = None
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """最適なパラメータを取得"""
        return self.best_params
    
    def get_best_score(self) -> Optional[float]:
        """最高スコアを取得"""
        return self.best_score
    
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """データを取得"""
        return self._data
    
    @data.setter
    def data(self, value: Optional[pd.DataFrame]) -> None:
        """データを設定し、data_dictも更新"""
        self._data = value
        if value is not None:
            symbol = self.config['data']['symbol']
            self._data_dict = {symbol: value}
        else:
            self._data_dict = None