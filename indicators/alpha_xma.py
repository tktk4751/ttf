#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .alma import calculate_alma


@dataclass
class AlphaXMAResult:
    """AlphaXMAの計算結果"""
    values: np.ndarray        # AlphaXMAの値（ALMAで平滑化済み）
    raw_values: np.ndarray    # 生のAlphaMA値（平滑化前）
    er: np.ndarray           # Efficiency Ratio
    dynamic_kama_period: np.ndarray  # 動的KAMAピリオド
    dynamic_fast_period: np.ndarray  # 動的Fast期間
    dynamic_slow_period: np.ndarray  # 動的Slow期間


@jit(nopython=True)
def calculate_dynamic_kama_period(er: np.ndarray, max_period: int, min_period: int) -> np.ndarray:
    """
    効率比に基づいて動的なKAMAピリオドを計算する（高速化版）
    
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
def calculate_dynamic_kama_constants(er: np.ndarray, 
                                    max_slow: int, min_slow: int,
                                    max_fast: int, min_fast: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    効率比に基づいて動的なKAMAのfast/slow期間を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_slow: 遅い移動平均の最大期間
        min_slow: 遅い移動平均の最小期間
        max_fast: 速い移動平均の最大期間
        min_fast: 速い移動平均の最小期間
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            動的なfast期間の配列、動的なslow期間の配列、fastの定数、slowの定数
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    fast_periods = min_fast + (1.0 - er) * (max_fast - min_fast)
    slow_periods = min_slow + (1.0 - er) * (max_slow - min_slow)
    
    fast_periods_rounded = np.round(fast_periods).astype(np.int32)
    slow_periods_rounded = np.round(slow_periods).astype(np.int32)
    
    # 定数の計算
    fast_constants = 2.0 / (fast_periods + 1.0)
    slow_constants = 2.0 / (slow_periods + 1.0)
    
    return fast_periods_rounded, slow_periods_rounded, fast_constants, slow_constants


@jit(nopython=True)
def calculate_alpha_xma(close: np.ndarray, er: np.ndarray, er_period: int,
                      kama_period: np.ndarray,
                      fast_constants: np.ndarray, slow_constants: np.ndarray,
                      alma_period: int, alma_offset: float, alma_sigma: float) -> np.ndarray:
    """
    AlphaXMAを計算する（高速化版）
    
    Args:
        close: 終値の配列
        er: 効率比の配列
        er_period: 効率比の計算期間
        kama_period: 動的なKAMAピリオドの配列
        fast_constants: 速い移動平均の定数配列
        slow_constants: 遅い移動平均の定数配列
        alma_period: ALMAの期間
        alma_offset: ALMAのオフセット (0-1)。1に近いほど最新のデータを重視
        alma_sigma: ALMAのシグマ。大きいほど重みの差が大きくなる
    
    Returns:
        AlphaXMAの配列
    """
    length = len(close)
    # 初期のスムーサーなしのアルファMAを計算するための配列
    initial_alpha_ma = np.full(length, np.nan)
    
    # 最初のAlphaMAは最初の価格
    initial_alpha_ma[0] = close[0]
    
    # 各時点でのAlphaMAを計算
    for i in range(1, length):
        if np.isnan(er[i]) or i < er_period:
            initial_alpha_ma[i] = initial_alpha_ma[i-1]
            continue
        
        # 現在の動的パラメータを取得
        curr_kama_period = int(kama_period[i])
        if curr_kama_period < 1:
            curr_kama_period = 1
        
        # 現在の時点での実際のER値を計算
        if i >= curr_kama_period:
            # 例外処理を簡略化（Numba対応）
            change = float(close[i] - close[i-curr_kama_period])
            volatility = float(np.sum(np.abs(np.diff(close[max(0, i-curr_kama_period):i+1]))))
            
            # ゼロ除算を防止
            if volatility < 1e-10:
                current_er = 0.0
            else:
                current_er = abs(change) / volatility
            
            # スムージング定数の計算
            smoothing_constant = (current_er * (fast_constants[i] - slow_constants[i]) + slow_constants[i]) ** 2
            
            # 0-1の範囲に制限
            if smoothing_constant < 0.0:
                smoothing_constant = 0.0
            elif smoothing_constant > 1.0:
                smoothing_constant = 1.0
            
            # AlphaMAの計算
            initial_alpha_ma[i] = float(initial_alpha_ma[i-1]) + float(smoothing_constant * (close[i] - initial_alpha_ma[i-1]))
        else:
            initial_alpha_ma[i] = initial_alpha_ma[i-1]
    
    # 計算したAlphaMAをALMAで平滑化
    alpha_xma = calculate_alma(initial_alpha_ma, alma_period, alma_offset, alma_sigma)
    
    return alpha_xma


class AlphaXMA(Indicator):
    """
    AlphaXMA (Alpha Exponential Moving Average) インジケーター
    
    AlphaMAにALMA (Arnaud Legoux Moving Average) フィルターを適用した改良版。
    効率比（ER）に基づいて以下のパラメータを動的に調整する適応型移動平均線：
    - KAMAピリオド自体
    - KAMAのfast期間
    - KAMAのslow期間
    
    その後、結果をALMAで平滑化して、ノイズをさらに除去し、
    よりスムーズなシグナルを得ることができます。
    
    特徴:
    - ERピリオドとKAMAピリオドを分離
    - すべてのパラメータが効率比に応じて動的に調整される
    - トレンドが強い時：短いピリオドと速い反応
    - レンジ相場時：長いピリオドとノイズ除去
    - ALMAフィルタリングでさらなるノイズ除去とトレンドへの素早い反応を両立
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_kama_period: int = 144,
        min_kama_period: int = 10,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2,
        alma_period: int = 9,
        alma_offset: float = 0.85,
        alma_sigma: float = 6
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_kama_period: KAMAピリオドの最大値（デフォルト: 144）
            min_kama_period: KAMAピリオドの最小値（デフォルト: 10）
            max_slow_period: 遅い移動平均の最大期間（デフォルト: 89）
            min_slow_period: 遅い移動平均の最小期間（デフォルト: 30）
            max_fast_period: 速い移動平均の最大期間（デフォルト: 13）
            min_fast_period: 速い移動平均の最小期間（デフォルト: 2）
            alma_period: ALMAの期間（デフォルト: 9）
            alma_offset: ALMAのオフセット(0-1)（デフォルト: 0.85）
            alma_sigma: ALMAのシグマ（デフォルト: 6）
        """
        super().__init__(
            f"AlphaXMA({er_period}, {max_kama_period}, {min_kama_period}, "
            f"{max_slow_period}, {min_slow_period}, {max_fast_period}, {min_fast_period}, "
            f"ALMA:{alma_period},{alma_offset:.2f},{alma_sigma})"
        )
        self.er_period = er_period
        self.max_kama_period = max_kama_period
        self.min_kama_period = min_kama_period
        self.max_slow_period = max_slow_period
        self.min_slow_period = min_slow_period
        self.max_fast_period = max_fast_period
        self.min_fast_period = min_fast_period
        self.alma_period = alma_period
        self.alma_offset = alma_offset
        self.alma_sigma = alma_sigma
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        AlphaXMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            AlphaXMAの値
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameには'close'カラムが必要です")
                close = data['close'].values
            else:
                if data.ndim == 2:
                    close = data[:, 3]  # close
                else:
                    close = data  # 1次元配列として扱う
            
            # データ長の検証
            data_length = len(close)
            self._validate_period(self.er_period, data_length)
            self._validate_period(self.alma_period, data_length)
            
            # 効率比（ER）の計算
            er = calculate_efficiency_ratio_for_period(close, self.er_period)
            
            # 動的なKAMAピリオドの計算
            dynamic_kama_period = calculate_dynamic_kama_period(
                er,
                self.max_kama_period,
                self.min_kama_period
            )
            
            # 動的なfast/slow期間の計算
            fast_periods, slow_periods, fast_constants, slow_constants = calculate_dynamic_kama_constants(
                er,
                self.max_slow_period,
                self.min_slow_period,
                self.max_fast_period,
                self.min_fast_period
            )
            
            # AlphaXMAの計算
            alpha_xma_values = calculate_alpha_xma(
                close,
                er,
                self.er_period,
                dynamic_kama_period,
                fast_constants,
                slow_constants,
                self.alma_period,
                self.alma_offset,
                self.alma_sigma
            )
            
            # 生のAlphaMA値を計算（平滑化前）
            raw_alpha_ma = np.full_like(close, np.nan)
            raw_alpha_ma[0] = close[0]
            for i in range(1, len(close)):
                if np.isnan(er[i]) or i < self.er_period:
                    raw_alpha_ma[i] = raw_alpha_ma[i-1]
                    continue
                
                curr_kama_period = int(dynamic_kama_period[i])
                if curr_kama_period < 1:
                    curr_kama_period = 1
                
                if i >= curr_kama_period:
                    change = float(close[i] - close[i-curr_kama_period])
                    volatility = float(np.sum(np.abs(np.diff(close[max(0, i-curr_kama_period):i+1]))))
                    
                    if volatility < 1e-10:
                        current_er = 0.0
                    else:
                        current_er = abs(change) / volatility
                    
                    smoothing_constant = (current_er * (fast_constants[i] - slow_constants[i]) + slow_constants[i]) ** 2
                    
                    if smoothing_constant < 0.0:
                        smoothing_constant = 0.0
                    elif smoothing_constant > 1.0:
                        smoothing_constant = 1.0
                    
                    raw_alpha_ma[i] = float(raw_alpha_ma[i-1]) + float(smoothing_constant * (close[i] - raw_alpha_ma[i-1]))
                else:
                    raw_alpha_ma[i] = raw_alpha_ma[i-1]
            
            # 結果の保存
            self._result = AlphaXMAResult(
                values=alpha_xma_values,
                raw_values=raw_alpha_ma,
                er=er,
                dynamic_kama_period=dynamic_kama_period,
                dynamic_fast_period=fast_periods,
                dynamic_slow_period=slow_periods
            )
            
            self._values = alpha_xma_values
            return alpha_xma_values
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"AlphaXMA計算中にエラー: {error_msg}\n{stack_trace}")
            # エラー時に初期化済みの配列を返す
            if 'close' in locals():
                self._values = np.zeros_like(close)
                return self._values
            return np.array([])
    
    def get_raw_values(self) -> np.ndarray:
        """
        平滑化前の生のAlphaMA値を取得する
        
        Returns:
            np.ndarray: 平滑化前のAlphaMA値
        """
        if self._result is None:
            return np.array([])
        return self._result.raw_values
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.er
    
    def get_dynamic_periods(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        動的な期間の値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (KAMAピリオド, Fast期間, Slow期間)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
        return (
            self._result.dynamic_kama_period,
            self._result.dynamic_fast_period,
            self._result.dynamic_slow_period
        ) 