#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

class Strategy(ABC):
    """
    戦略の基底クラス
    全ての戦略クラスはこのクラスを継承する
    """
    
    def __init__(self, name: str):
        """
        コンストラクタ
        
        Args:
            name: 戦略の名前
        """
        self.name = name
    
    @abstractmethod
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: ニュートラル)
        """
        pass
    
    @abstractmethod
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション (1: ロング, -1: ショート, 0: ニュートラル)
        
        Returns:
            True: エグジット, False: ホールド
        """
        pass
