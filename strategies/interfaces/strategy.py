#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Protocol, Union, Dict, Any
import numpy as np
import pandas as pd

class IStrategy(Protocol):
    """戦略のインターフェース"""
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを生成する"""
        ...
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成する"""
        ...
    
    def get_parameters(self) -> Dict[str, Any]:
        """現在のパラメータを取得する"""
        ...
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """パラメータを設定する"""
        ...

    def get_entry_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """エントリー価格を取得する"""
        ...

    def get_stop_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """ストップロス価格を取得する"""
        ...

    def get_exit_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """エグジット価格を取得する"""
        ... 