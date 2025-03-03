#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .atr import ATR
from .kama import KaufmanAdaptiveMA


@dataclass
class KAMAKeltnerChannelResult:
    """KAMAケルトナーチャネルの計算結果"""
    middle: np.ndarray  # 中心線（KAMA）
    upper: np.ndarray   # 上限線（KAMA + upper_multiplier * ATR）
    lower: np.ndarray   # 下限線（KAMA - lower_multiplier * ATR）
    half_upper: np.ndarray  # 中間上限線（KAMA + upper_multiplier * 0.5 * ATR）
    half_lower: np.ndarray  # 中間下限線（KAMA - lower_multiplier * 0.5 * ATR）
    atr: np.ndarray     # ATRの値


@jit(nopython=True)
def calculate_kama_keltner(kama: np.ndarray, atr: np.ndarray, upper_multiplier: float, lower_multiplier: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    KAMAケルトナーチャネルのバンドを計算する（高速化版）
    
    Args:
        kama: KAMAの配列
        atr: ATRの配列
        upper_multiplier: アッパーバンドのATR乗数
        lower_multiplier: ロワーバンドのATR乗数
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            上限線、下限線、中間上限線、中間下限線の配列
    """
    upper = kama + (upper_multiplier * atr)
    lower = kama - (lower_multiplier * atr)
    half_upper = kama + (upper_multiplier * 0.5 * atr)
    half_lower = kama - (lower_multiplier * 0.5 * atr)
    
    return upper, lower, half_upper, half_lower


class KAMAKeltnerChannel(Indicator):
    """
    KAMAケルトナーチャネル インジケーター
    
    KAMAを中心線として使用し、ATRの倍数でバンドを形成する
    - 中心線: KAMA
    - 上限線: KAMA + upper_multiplier * ATR
    - 下限線: KAMA - lower_multiplier * ATR
    - 中間上限線: KAMA + upper_multiplier * 0.5 * ATR
    - 中間下限線: KAMA - lower_multiplier * 0.5 * ATR
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
        super().__init__(f"KAMAKeltner({kama_period}, {kama_fast}, {kama_slow}, {atr_period}, {upper_multiplier}, {lower_multiplier})")
        self.kama = KaufmanAdaptiveMA(kama_period, kama_fast, kama_slow)
        self.atr = ATR(atr_period)
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        KAMAケルトナーチャネルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            中心線（KAMA）の値を返す
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
            
            # KAMAとATRの計算
            kama_values = self.kama.calculate(data)
            atr_values = self.atr.calculate(data)
            
            if atr_values is None:
                return None
            
            # バンドの計算（高速化版）
            upper, lower, half_upper, half_lower = calculate_kama_keltner(
                kama_values,
                atr_values,
                self.upper_multiplier,
                self.lower_multiplier
            )
            
            self._result = KAMAKeltnerChannelResult(
                middle=kama_values,
                upper=upper,
                lower=lower,
                half_upper=half_upper,
                half_lower=half_lower,
                atr=atr_values
            )
            
            self._values = kama_values  # 基底クラスの要件を満たすため
            return kama_values
            
        except Exception:
            return None
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        すべてのバンドの値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限線, 下限線, 中間上限線, 中間下限線)の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return (
            self._result.middle,
            self._result.upper,
            self._result.lower,
            self._result.half_upper,
            self._result.half_lower
        )
    
    def get_atr(self) -> np.ndarray:
        """
        ATRの値を取得する
        
        Returns:
            np.ndarray: ATRの値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.atr 