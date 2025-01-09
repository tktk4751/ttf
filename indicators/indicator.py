#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Optional, Union, List

import numpy as np
import pandas as pd

from logger import get_logger


class Indicator(ABC):
    """
    インディケーターの基底クラス
    すべてのインディケーターはこのクラスを継承する
    """
    
    def __init__(self, name: str):
        """
        コンストラクタ
        
        Args:
            name: インディケーターの名前
        """
        self.name = name
        self.logger = get_logger()
        self._values: Optional[np.ndarray] = None
    
    @abstractmethod
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        インディケーターを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            計算結果のNumPy配列
        """
        pass
    
    def _validate_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        入力データを検証し、適切な形式に変換する
        
        Args:
            data: 入力データ
        
        Returns:
            検証済みのNumPy配列
        """
        if isinstance(data, pd.DataFrame):
            if 'close' not in data.columns:
                raise ValueError("DataFrameには'close'カラムが必要です")
            return data['close'].values
        elif isinstance(data, np.ndarray):
            if data.ndim != 1:
                raise ValueError("NumPy配列は1次元である必要があります")
            return data
        else:
            raise ValueError("サポートされていないデータ型です")
    
    def _validate_period(self, period: int, data_length: int) -> None:
        """
        期間パラメータを検証する
        
        Args:
            period: 期間
            data_length: データの長さ
        """
        if period < 1:
            raise ValueError("期間は1以上である必要があります")
        if period > data_length:
            raise ValueError("期間がデータ長より大きいです")
    
    def get_values(self) -> np.ndarray:
        """
        計算済みの値を取得する
        
        Returns:
            計算結果のNumPy配列
        """
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._values
