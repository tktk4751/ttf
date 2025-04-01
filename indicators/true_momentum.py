#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Tuple, Union
from numba import jit
from dataclasses import dataclass

from .indicator import Indicator
from .kama import KaufmanAdaptiveMA
from .atr import ATR
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .trend_alpha import TrendAlpha
from .kama_bollinger_bands import DynamicKAMABollingerBands


@dataclass
class TrueMomentumResult:
    """トゥルーモメンタムの計算結果"""
    momentum: np.ndarray  # モメンタム値
    sqz_on: np.ndarray  # スクイーズオン状態
    sqz_off: np.ndarray  # スクイーズオフ状態
    no_sqz: np.ndarray  # スクイーズなし状態
    kama_values: np.ndarray  # KAMA値
    kama_upper: np.ndarray  # KAMAボリンジャー上限
    kama_lower: np.ndarray  # KAMAボリンジャー下限
    ta_middle: np.ndarray  # トレンドアルファ中心線
    ta_upper: np.ndarray  # トレンドアルファ上限
    ta_lower: np.ndarray  # トレンドアルファ下限
    efficiency_ratio: np.ndarray  # 効率比
    dynamic_momentum_period: np.ndarray  # 動的モメンタム期間


@jit(nopython=True)
def calculate_squeeze_states(
    bb_lower: np.ndarray,
    bb_upper: np.ndarray,
    kc_lower: np.ndarray,
    kc_upper: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    スクイーズ状態を計算（高速化版）
    
    Args:
        bb_lower: BBの下限
        bb_upper: BBの上限
        kc_lower: KCの下限
        kc_upper: KCの上限
        
    Returns:
        (sqz_on, sqz_off, no_sqz) のタプル
    """
    length = len(bb_lower)
    sqz_on = np.zeros(length, dtype=np.bool_)
    sqz_off = np.zeros(length, dtype=np.bool_)
    no_sqz = np.zeros(length, dtype=np.bool_)
    
    for i in range(length):
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(kc_upper[i]):
            continue
        
        if bb_lower[i] > kc_lower[i] and bb_upper[i] < kc_upper[i]:
            sqz_on[i] = True
        elif bb_lower[i] < kc_lower[i] and bb_upper[i] > kc_upper[i]:
            sqz_off[i] = True
        else:
            no_sqz[i] = True
    
    return sqz_on, sqz_off, no_sqz


@jit(nopython=True)
def calculate_dynamic_momentum_period(er: np.ndarray, max_period: int, min_period: int) -> np.ndarray:
    """
    効率比に基づいて動的なモメンタム期間を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        動的な期間の配列
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    periods = min_period + (1.0 - er) * (max_period - min_period)
    return np.round(periods).astype(np.int32)


@jit(nopython=True)
def calculate_rolling_max_min(high: np.ndarray, low: np.ndarray, periods: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    動的期間での最高値と最安値を計算（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        periods: 動的期間の配列
        
    Returns:
        (highest_high, lowest_low) のタプル
    """
    length = len(high)
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    max_period = int(np.nanmax(periods)) if len(periods) > 0 else 0
    if max_period <= 0:
        return highest_high, lowest_low
    
    for i in range(max_period, length):
        if np.isnan(periods[i]):
            continue
        
        period = int(periods[i])
        if period < 1:
            continue
            
        highest_high[i] = np.max(high[i-period+1:i+1])
        lowest_low[i] = np.min(low[i-period+1:i+1])
    
    return highest_high, lowest_low


@jit(nopython=True)
def calculate_dynamic_linear_regression(x: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """
    動的期間での線形回帰を計算（高速化版）
    
    Args:
        x: 入力配列
        periods: 動的期間の配列
        
    Returns:
        線形回帰の結果
    """
    length = len(x)
    result = np.full_like(x, np.nan)
    
    max_period = int(np.nanmax(periods)) if len(periods) > 0 else 0
    if max_period <= 0:
        return result
    
    for i in range(max_period, length):
        if np.isnan(periods[i]):
            continue
            
        period = int(periods[i])
        if period < 1:
            continue
            
        window = x[i-period+1:i+1]
        x_range = np.arange(period)
        
        # 線形回帰の係数を計算
        x_mean = np.mean(x_range)
        y_mean = np.mean(window)
        numerator = np.sum((x_range - x_mean) * (window - y_mean))
        denominator = np.sum((x_range - x_mean) ** 2)
        
        if denominator == 0:
            continue
            
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # 結果を計算
        result[i] = slope * (period - 1) + intercept
    
    return result


class TrueMomentum(Indicator):
    """
    トゥルーモメンタム インジケーター
    
    KAMAボリンジャーバンドとトレンドアルファを組み合わせてスクイーズモメンタムを拡張したインジケーター
    - KAMAボリンジャーバンド: 動的KAMA + 動的標準偏差乗数ベースのバンド
    - トレンドアルファ: 動的KAMA + 動的ATR乗数ベースのバンド
    - モメンタム: 動的期間の線形回帰に基づくモメンタム値
    
    効率比（ER）に基づいて以下のパラメータを動的に調整:
    - KAMAのfast/slow期間
    - 標準偏差の乗数
    - ATR期間とATR乗数
    - モメンタム計算の期間
    
    スクイーズ状態：
    - スクイーズオン: KAMAボリンジャーバンドがトレンドアルファバンドの内側にある
    - スクイーズオフ: KAMAボリンジャーバンドがトレンドアルファバンドの外側にある
    - スクイーズなし: その他の状態
    """
    
    def __init__(
        self,
        period: int = 20,
        max_std_mult: float = 2.0,
        min_std_mult: float = 1.0,
        max_kama_slow: int = 55,
        min_kama_slow: int = 30,
        max_kama_fast: int = 13,
        min_kama_fast: int = 2,
        max_atr_period: int = 120,
        min_atr_period: int = 13,
        max_atr_mult: float = 3.0,
        min_atr_mult: float = 1.0,
        max_momentum_period: int = 100,
        min_momentum_period: int = 20
    ):
        """
        コンストラクタ
        
        Args:
            period: 計算期間（KAMA、ボリンジャーバンド、トレンドアルファに共通）（デフォルト: 20）
            max_std_mult: 標準偏差乗数の最大値（デフォルト: 2.0）
            min_std_mult: 標準偏差乗数の最小値（デフォルト: 1.0）
            max_kama_slow: KAMAの遅い移動平均の最大期間（デフォルト: 55）
            min_kama_slow: KAMAの遅い移動平均の最小期間（デフォルト: 30）
            max_kama_fast: KAMAの速い移動平均の最大期間（デフォルト: 13）
            min_kama_fast: KAMAの速い移動平均の最小期間（デフォルト: 2）
            max_atr_period: ATR期間の最大値（デフォルト: 120）
            min_atr_period: ATR期間の最小値（デフォルト: 13）
            max_atr_mult: ATR乗数の最大値（デフォルト: 3.0）
            min_atr_mult: ATR乗数の最小値（デフォルト: 1.0）
            max_momentum_period: モメンタム計算の最大期間（デフォルト: 100）
            min_momentum_period: モメンタム計算の最小期間（デフォルト: 20）
        """
        super().__init__("TrueMomentum")
        
        self.period = period
        self.max_momentum_period = max_momentum_period
        self.min_momentum_period = min_momentum_period
        
        # KAMAボリンジャーバンドのインスタンス化
        self.kama_bb = DynamicKAMABollingerBands(
            period=period,
            max_std_mult=max_std_mult,
            min_std_mult=min_std_mult,
            max_kama_slow=max_kama_slow,
            min_kama_slow=min_kama_slow,
            max_kama_fast=max_kama_fast,
            min_kama_fast=min_kama_fast
        )
        
        # トレンドアルファのインスタンス化
        self.trend_alpha = TrendAlpha(
            kama_period=period,
            max_kama_slow=max_kama_slow,
            min_kama_slow=min_kama_slow,
            max_kama_fast=max_kama_fast,
            min_kama_fast=min_kama_fast,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_atr_mult,
            min_multiplier=min_atr_mult
        )
        
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        トゥルーモメンタムを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            モメンタム値の配列
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
            else:
                if data.ndim == 2:
                    high = data[:, 1]  # high
                    low = data[:, 2]   # low
                    close = data[:, 3] # close
                else:
                    close = data
                    high = close
                    low = close
            
            # データ長の検証
            data_length = len(close)
            
            # 1. KAMAボリンジャーバンドの計算
            kama_bb_result = self.kama_bb.calculate(data)
            
            # 2. トレンドアルファの計算
            trend_alpha_values = self.trend_alpha.calculate(data)
            ta_middle, ta_upper, ta_lower, _, _ = self.trend_alpha.get_bands()
            
            # 3. 効率比（ER）を取得
            er = self.kama_bb.get_efficiency_ratio()
            
            # 4. 動的モメンタム期間の計算
            dynamic_momentum_period = calculate_dynamic_momentum_period(
                er,
                self.max_momentum_period,
                self.min_momentum_period
            )
            
            # 5. スクイーズ状態の判定
            sqz_on, sqz_off, no_sqz = calculate_squeeze_states(
                kama_bb_result.lower, 
                kama_bb_result.upper, 
                ta_lower, 
                ta_upper
            )
            
            # 6. 動的期間での最高値と最安値を計算
            highest_high, lowest_low = calculate_rolling_max_min(high, low, dynamic_momentum_period)
            
            # 7. 動的モメンタム値の計算
            avg_hl = (highest_high + lowest_low) / 2
            avg_hlc = (avg_hl + kama_bb_result.middle) / 2
            
            momentum = calculate_dynamic_linear_regression(close - avg_hlc, dynamic_momentum_period)
            
            # 結果の保存
            self._result = TrueMomentumResult(
                momentum=momentum,
                sqz_on=sqz_on,
                sqz_off=sqz_off,
                no_sqz=no_sqz,
                kama_values=kama_bb_result.middle,
                kama_upper=kama_bb_result.upper,
                kama_lower=kama_bb_result.lower,
                ta_middle=ta_middle,
                ta_upper=ta_upper,
                ta_lower=ta_lower,
                efficiency_ratio=er,
                dynamic_momentum_period=dynamic_momentum_period
            )
            
            self._values = momentum  # 基底クラスの要件を満たすため
            return momentum
            
        except Exception as e:
            print(f"計算中にエラーが発生しました: {e}")
            return None
    
    def get_squeeze_states(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        スクイーズ状態を取得
        
        Returns:
            (sqz_on, sqz_off, no_sqz) のタプル
            - sqz_on: スクイーズオン状態
            - sqz_off: スクイーズオフ状態
            - no_sqz: スクイーズなし状態
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.sqz_on, self._result.sqz_off, self._result.no_sqz
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        全バンドの値を取得
        
        Returns:
            (kama_values, kama_upper, kama_lower, ta_upper, ta_lower) のタプル
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return (
            self._result.kama_values, 
            self._result.kama_upper, 
            self._result.kama_lower, 
            self._result.ta_upper, 
            self._result.ta_lower
        )
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比の値を取得
        
        Returns:
            効率比の配列
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.efficiency_ratio
    
    def get_dynamic_momentum_period(self) -> np.ndarray:
        """
        動的モメンタム期間を取得
        
        Returns:
            動的モメンタム期間の配列
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.dynamic_momentum_period 