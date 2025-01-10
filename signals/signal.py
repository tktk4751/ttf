#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

class Signal(ABC):
    """
    シグナルの基底クラス
    全てのシグナルクラスはこのクラスを継承する
    """
    
    def __init__(self, name: str):
        """
        コンストラクタ
        
        Args:
            name: シグナルの名前
        """
        self.name = name
    
    @abstractmethod
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列
        """
        pass
