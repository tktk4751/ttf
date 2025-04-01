#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Tuple, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator


@jit(nopython=True)
def calculate_rsx_numba(
    data: np.ndarray,
    length: int
) -> np.ndarray:
    """
    Jurik RSX (Relative Strength Index)をNumbaを使用して計算（高速化版）
    
    Args:
        data: 価格データの配列
        length: 計算期間
        
    Returns:
        RSXの配列（0-100の範囲）
    """
    size = len(data)
    f8 = data * 100.0  # 元の値を100倍
    rsx = np.zeros(size)
    
    # 初期値の設定
    f10 = np.zeros(size)
    v8 = np.zeros(size)
    f28 = np.zeros(size)
    f30 = np.zeros(size)
    vC = np.zeros(size)
    f38 = np.zeros(size)
    f40 = np.zeros(size)
    v10 = np.zeros(size)
    f48 = np.zeros(size)
    f50 = np.zeros(size)
    v14 = np.zeros(size)
    f58 = np.zeros(size)
    f60 = np.zeros(size)
    v18 = np.zeros(size)
    f68 = np.zeros(size)
    f70 = np.zeros(size)
    v1C = np.zeros(size)
    f78 = np.zeros(size)
    f80 = np.zeros(size)
    v20 = np.zeros(size)
    
    f88 = np.zeros(size)
    f90 = np.zeros(size)
    f0 = np.zeros(size)
    v4 = np.zeros(size)
    
    # パラメータの計算
    f18 = 3.0 / (length + 2.0)
    f20 = 1.0 - f18
    
    for i in range(1, size):
        # 価格変化の計算
        f10[i] = f8[i-1]
        v8[i] = f8[i] - f10[i]
        
        # フィルタリング（1段階目）
        f28[i] = f20 * f28[i-1] + f18 * v8[i]
        f30[i] = f18 * f28[i] + f20 * f30[i-1]
        vC[i] = f28[i] * 1.5 - f30[i] * 0.5
        
        # フィルタリング（2段階目）
        f38[i] = f20 * f38[i-1] + f18 * vC[i]
        f40[i] = f18 * f38[i] + f20 * f40[i-1]
        v10[i] = f38[i] * 1.5 - f40[i] * 0.5
        
        # フィルタリング（3段階目）
        f48[i] = f20 * f48[i-1] + f18 * v10[i]
        f50[i] = f18 * f48[i] + f20 * f50[i-1]
        v14[i] = f48[i] * 1.5 - f50[i] * 0.5
        
        # 絶対値のフィルタリング（1段階目）
        f58[i] = f20 * f58[i-1] + f18 * abs(v8[i])
        f60[i] = f18 * f58[i] + f20 * f60[i-1]
        v18[i] = f58[i] * 1.5 - f60[i] * 0.5
        
        # 絶対値のフィルタリング（2段階目）
        f68[i] = f20 * f68[i-1] + f18 * v18[i]
        f70[i] = f18 * f68[i] + f20 * f70[i-1]
        v1C[i] = f68[i] * 1.5 - f70[i] * 0.5
        
        # 絶対値のフィルタリング（3段階目）
        f78[i] = f20 * f78[i-1] + f18 * v1C[i]
        f80[i] = f18 * f78[i] + f20 * f80[i-1]
        v20[i] = f78[i] * 1.5 - f80[i] * 0.5
        
        # カウンタの計算
        if f90[i-1] == 0:
            if length - 1 >= 5:
                f88[i] = length - 1
            else:
                f88[i] = 5
        else:
            if f88[i-1] <= f90[i-1]:
                f88[i] = f88[i-1] + 1
            else:
                f88[i] = f90[i-1] + 1
        
        # フラグの計算
        if f88[i] >= f90[i-1] and f8[i] != f10[i]:
            f0[i] = 1
        else:
            f0[i] = 0
        
        if f88[i] == f90[i-1] and f0[i] == 0:
            f90[i] = 0
        else:
            f90[i] = f90[i-1]
        
        # RSXの計算
        if f88[i] < f90[i] and v20[i] > 0:
            v4[i] = (v14[i] / v20[i] + 1) * 50
        else:
            v4[i] = 50
        
        # 0-100の範囲にクリップ
        if v4[i] > 100:
            rsx[i] = 100
        elif v4[i] < 0:
            rsx[i] = 0
        else:
            rsx[i] = v4[i]
    
    return rsx


class RSX(Indicator):
    """
    Jurik RSX (Relative Strength eXtended) インディケーター
    
    RSIの改良版で、よりスムーズで反応が速い。
    マルチステージフィルタリングを使用して、価格変動から効率的にシグナルを抽出する。
    
    特徴:
    - RSIよりもスムーズなシグナル
    - より速い反応速度
    - 優れたノイズ除去能力
    - より安定したオーバーボート/オーバーソールドレベル
    
    使用方法:
    - 標準的なRSIと同様に70以上で買われすぎ、30以下で売られすぎと判断
    - クロスオーバー/クロスアンダーでのエントリー/エグジットポイントの確認
    - トレンドの方向とストレングスの判断
    """
    
    def __init__(
        self,
        period: int = 14,
        high_level: float = 70,
        low_level: float = 30
    ):
        """
        コンストラクタ
        
        Args:
            period: 計算期間（デフォルト: 14）
            high_level: 買われすぎレベル（デフォルト: 70）
            low_level: 売られすぎレベル（デフォルト: 30）
        """
        super().__init__(f"RSX({period})")
        self.period = period
        self.high_level = high_level
        self.low_level = low_level
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        RSXを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、HLC3が計算されます（High, Low, Closeの平均）
        
        Returns:
            RSXの配列（0-100の範囲）
        """
        # データの検証と変換
        if isinstance(data, pd.DataFrame):
            if set(['high', 'low', 'close']).issubset(data.columns):
                # HLC3の計算
                hlc3 = (data['high'] + data['low'] + data['close']) / 3
                prices = hlc3.values
            elif 'close' in data.columns:
                prices = data['close'].values
            else:
                raise ValueError("DataFrameには'close'または'high','low','close'カラムが必要です")
        else:
            if data.ndim == 2 and data.shape[1] >= 4:
                # OHLC配列からHLC3を計算
                high = data[:, 1]  # high
                low = data[:, 2]   # low
                close = data[:, 3] # close
                prices = (high + low + close) / 3
            elif data.ndim == 1:
                prices = data
            else:
                raise ValueError("サポートされていないデータ形式です")
        
        # データ長の検証
        data_length = len(prices)
        self._validate_period(self.period, data_length)
        
        # RSXの計算（Numba高速化版）
        self._values = calculate_rsx_numba(prices, self.period)
        
        return self._values
    
    def get_overbought_oversold(
        self,
        ob_level: float = None,
        os_level: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        買われすぎ・売られすぎのシグナルを取得
        
        Args:
            ob_level: 買われすぎのレベル（Noneの場合はコンストラクタで設定した値を使用）
            os_level: 売られすぎのレベル（Noneの場合はコンストラクタで設定した値を使用）
        
        Returns:
            (買われすぎシグナル, 売られすぎシグナル)のタプル
            シグナルは、条件を満たす場合は1、そうでない場合は0
        """
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        ob = self.ob_level if ob_level is None else ob_level
        os = self.os_level if os_level is None else os_level
        
        overbought_signal = np.where(self._values >= ob, 1, 0)
        oversold_signal = np.where(self._values <= os, 1, 0)
        
        return overbought_signal, oversold_signal
    
    def get_crossover_signals(self, level: float = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        クロスオーバー・クロスアンダーのシグナルを取得
        
        Args:
            level: クロスするレベル（デフォルト：50）
        
        Returns:
            (クロスオーバーシグナル, クロスアンダーシグナル)のタプル
            シグナルは、条件を満たす場合は1、そうでない場合は0
        """
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        # 1つ前の値を取得（最初の要素には前の値がないので同じ値を使用）
        prev_values = np.roll(self._values, 1)
        prev_values[0] = self._values[0]
        
        # クロスオーバー: 前の値がレベル未満で、現在の値がレベル以上
        crossover = np.where(
            (prev_values < level) & (self._values >= level),
            1, 0
        )
        
        # クロスアンダー: 前の値がレベル以上で、現在の値がレベル未満
        crossunder = np.where(
            (prev_values >= level) & (self._values < level),
            1, 0
        )
        
        return crossover, crossunder 