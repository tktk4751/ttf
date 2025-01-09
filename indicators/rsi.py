#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Tuple

import numpy as np
import pandas as pd

from .indicator import Indicator


class RSI(Indicator):
    """
    RSI (Relative Strength Index) インディケーター
    RSI = 100 - (100 / (1 + RS))
    RS = 上昇の平均 / 下降の平均
    """
    
    def __init__(self, period: int = 14):
        """
        コンストラクタ
        
        Args:
            period: 期間
        """
        super().__init__(f"RSI{period}")
        self.period = period
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        RSIを計算する
        
        Args:
            data: 価格データ
        
        Returns:
            RSIの配列
        """
        prices = self._validate_data(data)
        self._validate_period(self.period, len(prices))
        
        # 価格変化を計算
        changes = np.diff(prices)
        changes = np.insert(changes, 0, 0)  # 最初の要素を0で埋める
        
        # 上昇・下降を分離
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        
        # 上昇・下降の平均を計算
        avg_gains = self._calculate_average(gains)
        avg_losses = self._calculate_average(losses)
        
        # RSを計算（ゼロ除算を防ぐ）
        rs = np.divide(
            avg_gains,
            avg_losses,
            out=np.zeros_like(avg_gains),
            where=avg_losses != 0
        )
        
        # RSIを計算
        self._values = 100 - (100 / (1 + rs))
        
        return self._values
    
    def _calculate_average(self, values: np.ndarray) -> np.ndarray:
        """
        移動平均を計算する（Wilderの平滑化方式）
        
        Args:
            values: 計算対象の配列
        
        Returns:
            移動平均の配列
        """
        result = np.full_like(values, np.nan, dtype=np.float64)
        
        # 最初の平均を計算
        result[self.period] = np.mean(values[1:self.period+1])
        
        # Wilderの平滑化を適用
        for i in range(self.period + 1, len(values)):
            result[i] = (
                (result[i-1] * (self.period - 1) + values[i]) / self.period
            )
        
        return result
    
    def get_overbought_oversold(
        self,
        overbought: float = 70,
        oversold: float = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        買われすぎ・売られすぎのシグナルを取得
        
        Args:
            overbought: 買われすぎのレベル
            oversold: 売られすぎのレベル
        
        Returns:
            (買われすぎシグナル, 売られすぎシグナル)のタプル
            シグナルは、条件を満たす場合は1、そうでない場合は0
        """
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        overbought_signal = np.where(self._values >= overbought, 1, 0)
        oversold_signal = np.where(self._values <= oversold, 1, 0)
        
        return overbought_signal, oversold_signal
