#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .atr import ATR
from .gvidya import GVIDYA


@dataclass
class GVIDYAKeltnerChannelResult:
    """G-VIDYAケルトナーチャネルの計算結果"""
    middle: np.ndarray  # 中心線（G-VIDYA）
    upper: np.ndarray   # 上限線（G-VIDYA + upper_multiplier * ATR）
    lower: np.ndarray   # 下限線（G-VIDYA - lower_multiplier * ATR）
    half_upper: np.ndarray  # 中間上限線（G-VIDYA + upper_multiplier * 0.5 * ATR）
    half_lower: np.ndarray  # 中間下限線（G-VIDYA - lower_multiplier * 0.5 * ATR）
    atr: np.ndarray     # ATRの値


@jit(nopython=True)
def calculate_gvidya_keltner(gvidya: np.ndarray, atr: np.ndarray, upper_multiplier: float, lower_multiplier: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    G-VIDYAケルトナーチャネルのバンドを計算する（高速化版）
    
    Args:
        gvidya: G-VIDYAの配列
        atr: ATRの配列
        upper_multiplier: アッパーバンドのATR乗数
        lower_multiplier: ロワーバンドのATR乗数
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            上限線、下限線、中間上限線、中間下限線の配列
    """
    upper = gvidya + (upper_multiplier * atr)
    lower = gvidya - (lower_multiplier * atr)
    half_upper = gvidya + (upper_multiplier * 0.5 * atr)
    half_lower = gvidya - (lower_multiplier * 0.5 * atr)
    
    return upper, lower, half_upper, half_lower


class GVIDYAKeltnerChannel(Indicator):
    """
    G-VIDYAケルトナーチャネル インジケーター
    
    G-VIDYAを中心線として使用し、ATRの倍数でバンドを形成する
    - 中心線: G-VIDYA
    - 上限線: G-VIDYA + upper_multiplier * ATR
    - 下限線: G-VIDYA - lower_multiplier * ATR
    - 中間上限線: G-VIDYA + upper_multiplier * 0.5 * ATR
    - 中間下限線: G-VIDYA - lower_multiplier * 0.5 * ATR
    """
    
    def __init__(
        self,
        vidya_period: int = 46,
        sd_period: int = 28,
        gaussian_length: int = 4,
        gaussian_sigma: float = 2.0,
        atr_period: int = 14,
        upper_multiplier: float = 1.3,
        lower_multiplier: float = 1.3
    ):
        """
        コンストラクタ
        
        Args:
            vidya_period: VIDYA期間（デフォルト: 46）
            sd_period: 標準偏差の計算期間（デフォルト: 28）
            gaussian_length: ガウシアンフィルターの長さ（デフォルト: 4）
            gaussian_sigma: ガウシアンフィルターのシグマ（デフォルト: 2.0）
            atr_period: ATRの期間（デフォルト: 14）
            upper_multiplier: アッパーバンドのATR乗数（デフォルト: 1.3）
            lower_multiplier: ロワーバンドのATR乗数（デフォルト: 1.3）
        """
        super().__init__(f"GVIDYAKeltner({vidya_period}, {sd_period}, {gaussian_length}, {gaussian_sigma}, {atr_period}, {upper_multiplier}, {lower_multiplier})")
        self.gvidya = GVIDYA(vidya_period, sd_period, gaussian_length, gaussian_sigma)
        self.atr = ATR(atr_period)
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        G-VIDYAケルトナーチャネルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            中心線（G-VIDYA）の値を返す
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
            
            # G-VIDYAとATRの計算
            gvidya_values = self.gvidya.calculate(data)
            atr_values = self.atr.calculate(data)
            
            if atr_values is None:
                return None
            
            # バンドの計算（高速化版）
            upper, lower, half_upper, half_lower = calculate_gvidya_keltner(
                gvidya_values,
                atr_values,
                self.upper_multiplier,
                self.lower_multiplier
            )
            
            self._result = GVIDYAKeltnerChannelResult(
                middle=gvidya_values,
                upper=upper,
                lower=lower,
                half_upper=half_upper,
                half_lower=half_lower,
                atr=atr_values
            )
            
            self._values = gvidya_values  # 基底クラスの要件を満たすため
            return gvidya_values
            
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