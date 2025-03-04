#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .kama_keltner import KAMAKeltnerChannel


@dataclass
class TFCResult:
    """KAMAケルトナーTFCの計算結果"""
    middle: np.ndarray      # 中心線（KAMA）
    upper: np.ndarray       # 上限線
    lower: np.ndarray       # 下限線
    tfc: np.ndarray        # TFC値
    trend: np.ndarray       # トレンド方向（1: アップトレンド、-1: ダウントレンド）


@jit(nopython=True)
def calculate_tfc(close: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    TFCの値とトレンド方向を計算する（高速化版）
    
    Args:
        close: 終値の配列
        upper: アッパーバンドの配列
        lower: ロワーバンドの配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: TFC値とトレンド方向の配列
    """
    length = len(close)
    tfc = np.zeros(length)
    trend = np.zeros(length)
    
    # 初期値の設定
    trend[0] = 1 if close[0] > upper[0] else -1
    tfc[0] = lower[0] if trend[0] == 1 else upper[0]
    
    # 2バー目以降の計算
    for i in range(1, length):
        if close[i] > upper[i-1]:
            trend[i] = 1
        elif close[i] < lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
            # バンドの調整
            if trend[i] == 1 and lower[i] < lower[i-1]:
                lower[i] = lower[i-1]
            elif trend[i] == -1 and upper[i] > upper[i-1]:
                upper[i] = upper[i-1]
        
        # トレンドに基づいてTFC値を設定
        if trend[i] == 1:
            tfc[i] = lower[i]
        else:
            tfc[i] = upper[i]
    
    return tfc, trend


class TFC(Indicator):
    """
    KAMAケルトナーTFC インジケーター
    
    KAMAケルトナーチャネルをベースに、TFCのトレンド判定ロジックを組み込んだインジケーター
    - TFC値: トレンドに応じて上限線または下限線を動的に選択
    - トレンド方向: 1（アップトレンド）または-1（ダウントレンド）
    """
    
    def __init__(
        self,
        kama_period: int = 10,
        kama_fast: int = 2,
        kama_slow: int = 30,
        atr_period: int = 10,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 2.0
    ):
        """
        コンストラクタ
        
        Args:
            kama_period: KAMAの効率比の計算期間（デフォルト: 10）
            kama_fast: KAMAの速い移動平均の期間（デフォルト: 2）
            kama_slow: KAMAの遅い移動平均の期間（デフォルト: 30）
            atr_period: ATRの期間（デフォルト: 10）
            upper_multiplier: アッパーバンドのATR乗数（デフォルト: 2.0）
            lower_multiplier: ロワーバンドのATR乗数（デフォルト: 2.0）
        """
        super().__init__(f"KAMAKeltnerTFC({kama_period}, {kama_fast}, {kama_slow}, {atr_period}, {upper_multiplier}, {lower_multiplier})")
        self.kama_keltner = KAMAKeltnerChannel(kama_period, kama_fast, kama_slow, atr_period, upper_multiplier, lower_multiplier)
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        KAMAケルトナーTFCを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            TFC値の配列
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
                close = data['close'].values
            else:
                close = data
            
            # KAMAケルトナーチャネルの計算
            self.kama_keltner.calculate(data)
            middle, upper, lower, _, _ = self.kama_keltner.get_bands()
            
            # TFCの計算
            tfc, trend = calculate_tfc(close, upper, lower)
            
            self._result = TFCResult(
                middle=middle,
                upper=upper,
                lower=lower,
                tfc=tfc,
                trend=trend
            )
            
            self._values = tfc  # 基底クラスの要件を満たすため
            return tfc
            
        except Exception:
            return None
    
    def get_trend(self) -> np.ndarray:
        """
        トレンド方向の配列を取得する
        
        Returns:
            np.ndarray: トレンド方向の配列（1: アップトレンド、-1: ダウントレンド）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.trend
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        中心線、上限線、下限線の値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限線, 下限線)の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return (
            self._result.middle,
            self._result.upper,
            self._result.lower
        ) 