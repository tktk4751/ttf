#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Tuple
import numpy as np
import pandas as pd

from .indicator import Indicator


class DonchianChannel(Indicator):
    """
    ドンチャンチャネルインディケーター
    
    指定期間の最高値、最安値、およびその中央値を計算する
    - 上限: n期間の最高値
    - 下限: n期間の最安値
    - 中央線: (上限 + 下限) / 2
    """
    
    def __init__(self, period: int = 20):
        """
        コンストラクタ
        
        Args:
            period: 期間（デフォルト: 20）
        """
        super().__init__(f"DonchianChannel({period})")
        self.period = period
        self._upper = None
        self._lower = None
        self._middle = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ドンチャンチャネルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'カラムが必要
        
        Returns:
            中央線の値を返す
        """
        # データの検証と変換
        if isinstance(data, pd.DataFrame):
            if 'high' not in data.columns or 'low' not in data.columns:
                raise ValueError("DataFrameには'high'と'low'カラムが必要です")
            high = data['high'].values
            low = data['low'].values
        else:
            raise ValueError("データはDataFrameである必要があります")
        
        # データ長の検証
        data_length = len(high)
        self._validate_period(self.period, data_length)
        
        # 初期化
        self._upper = np.zeros(data_length)
        self._lower = np.zeros(data_length)
        self._middle = np.zeros(data_length)
        
        # 最初のperiod-1個は計算不可能なのでnanで埋める
        self._upper[:self.period-1] = np.nan
        self._lower[:self.period-1] = np.nan
        self._middle[:self.period-1] = np.nan
        
        # 各時点でのperiod期間の最高値、最安値を計算
        for i in range(self.period-1, data_length):
            self._upper[i] = np.max(high[i-self.period+1:i+1])
            self._lower[i] = np.min(low[i-self.period+1:i+1])
            self._middle[i] = (self._upper[i] + self._lower[i]) / 2
        
        self._values = self._middle
        return self._middle
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        上限、下限、中央線の値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (上限, 下限, 中央線)の値
        """
        if self._upper is None or self._lower is None or self._middle is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._upper, self._lower, self._middle 