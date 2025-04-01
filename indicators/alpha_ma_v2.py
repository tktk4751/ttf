#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period


@dataclass
class AlphaMAV2Result:
    """AlphaMAV2の計算結果"""
    values: np.ndarray         # AlphaMAV2の値
    er: np.ndarray             # Efficiency Ratio
    dynamic_period: np.ndarray # 動的期間


@jit(nopython=True)
def calculate_dynamic_period(
    er: np.ndarray, 
    max_period: int, 
    min_period: int
) -> np.ndarray:
    """
    効率比に基づいて動的な期間を計算する（高速化版）
    
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
def calculate_rsx_smoothing(data: np.ndarray, length: int) -> np.ndarray:
    """
    RSXの平滑化関数を使用してデータを平滑化する（高速化版）
    
    Args:
        data: 入力データ配列
        length: 平滑化期間
    
    Returns:
        平滑化されたデータの配列
    """
    size = len(data)
    f8 = data.copy()  # 元のデータをコピー
    smoothed = np.zeros(size)
    
    # 初期値の設定
    f28 = np.zeros(size)
    f30 = np.zeros(size)
    f38 = np.zeros(size)
    f40 = np.zeros(size)
    f48 = np.zeros(size)
    f50 = np.zeros(size)
    
    # パラメータの計算
    f18 = 3.0 / (length + 2.0)
    f20 = 1.0 - f18
    
    for i in range(1, size):
        # フィルタリング（1段階目）
        f28[i] = f20 * f28[i-1] + f18 * f8[i]
        f30[i] = f18 * f28[i] + f20 * f30[i-1]
        smoothed_1 = f28[i] * 1.5 - f30[i] * 0.5
        
        # フィルタリング（2段階目）
        f38[i] = f20 * f38[i-1] + f18 * smoothed_1
        f40[i] = f18 * f38[i] + f20 * f40[i-1]
        smoothed_2 = f38[i] * 1.5 - f40[i] * 0.5
        
        # フィルタリング（3段階目）
        f48[i] = f20 * f48[i-1] + f18 * smoothed_2
        f50[i] = f18 * f48[i] + f20 * f50[i-1]
        smoothed[i] = f48[i] * 1.5 - f50[i] * 0.5
    
    return smoothed


@jit(nopython=True)
def calculate_alpha_ma_v2(close: np.ndarray, er: np.ndarray, dynamic_periods: np.ndarray) -> np.ndarray:
    """
    AlphaMAV2を計算する（高速化版）
    
    Args:
        close: 終値の配列
        er: 効率比の配列
        dynamic_periods: 動的な期間の配列
    
    Returns:
        AlphaMAV2の配列
    """
    length = len(close)
    alpha_ma_v2 = np.zeros(length)
    
    # 最大期間を取得（計算開始点の決定用）
    max_period = int(np.max(dynamic_periods))
    if max_period < 1:
        max_period = 1
    
    # 初期値の設定（最初のポイントはデータそのまま）
    if length > 0:
        alpha_ma_v2[0] = close[0]
    
    # 各時点での平滑化を計算
    for i in range(max_period, length):
        # その時点での動的な期間を取得
        curr_period = int(dynamic_periods[i])
        if curr_period < 1:
            curr_period = 1
        
        # その時点でのウィンドウを取得（効率のため、必要な部分の配列だけを使用）
        window = close[max(0, i-curr_period*3):i+1]  # 3倍の長さで十分なバッファを確保
        
        # RSXの平滑化関数を使用して平滑化
        smoothed = calculate_rsx_smoothing(window, curr_period)
        
        # 最後の値（現在の時点の平滑化値）を使用
        alpha_ma_v2[i] = smoothed[-1]
    
    return alpha_ma_v2


class AlphaMAV2(Indicator):
    """
    AlphaMAV2 (Alpha Moving Average Version 2) インジケーター
    
    効率比（ER）に基づいて期間を動的に調整し、RSXの平滑化関数を使用する
    適応型移動平均線。
    
    特徴:
    - RSXの多段階平滑化アルゴリズムを使用
    - 期間が効率比に応じて動的に調整される
    - トレンドが強い時：短い期間と速い反応
    - レンジ相場時：長い期間とノイズ除去
    - 優れたノイズ除去とシグナルの明確さを両立
    """
    
    def __init__(
        self,
        er_period: int = 10,
        max_period: int = 34,
        min_period: int = 5
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 10）
            max_period: 平滑化期間の最大値（デフォルト: 34）
            min_period: 平滑化期間の最小値（デフォルト: 5）
        """
        super().__init__(f"AlphaMAV2({er_period}, {min_period}-{max_period})")
        self.er_period = er_period
        self.max_period = max_period
        self.min_period = min_period
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        AlphaMAV2を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            AlphaMAV2の値
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
            
            # 効率比（ER）の計算
            er = calculate_efficiency_ratio_for_period(close, self.er_period)
            
            # 動的な期間の計算
            dynamic_periods = calculate_dynamic_period(
                er,
                self.max_period,
                self.min_period
            )
            
            # AlphaMAV2の計算
            alpha_ma_v2_values = calculate_alpha_ma_v2(
                close,
                er,
                dynamic_periods
            )
            
            # 結果の保存
            self._result = AlphaMAV2Result(
                values=alpha_ma_v2_values,
                er=er,
                dynamic_period=dynamic_periods
            )
            
            self._values = alpha_ma_v2_values
            return alpha_ma_v2_values
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"AlphaMAV2計算中にエラー: {error_msg}\n{stack_trace}")
            # エラー時に初期化済みの配列を返す
            if 'close' in locals():
                self._values = np.zeros_like(close)
                return self._values
            return np.array([])
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            効率比の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.er
    
    def get_dynamic_period(self) -> np.ndarray:
        """
        動的な期間の値を取得する
        
        Returns:
            動的な期間の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.dynamic_period
    
    def get_signal_line(self, offset: int = 1) -> np.ndarray:
        """
        シグナルラインを計算する（移動平均のオフセット）
        
        Args:
            offset: オフセット期間（デフォルト: 1）
        
        Returns:
            シグナルラインの値
        """
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        # オフセットが有効かチェック
        if offset <= 0:
            return self._values
        
        # 単純な移動平均でシグナルラインを作成
        signal = np.zeros_like(self._values)
        for i in range(offset, len(self._values)):
            signal[i] = np.mean(self._values[i-offset:i+1])
        
        return signal
    
    def get_crossover_signals(self, offset: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        クロスオーバー・クロスアンダーのシグナルを取得
        
        Args:
            offset: シグナルラインのオフセット（デフォルト: 1）
        
        Returns:
            (クロスオーバーシグナル, クロスアンダーシグナル)のタプル
            シグナルは、条件を満たす場合は1、そうでない場合は0
        """
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        # シグナルラインの取得
        signal = self.get_signal_line(offset)
        
        # 1つ前の値を取得（最初の要素には前の値がないので同じ値を使用）
        prev_values = np.roll(self._values, 1)
        prev_values[0] = self._values[0]
        prev_signal = np.roll(signal, 1)
        prev_signal[0] = signal[0]
        
        # クロスオーバー: 前の値がシグナル未満で、現在の値がシグナル以上
        crossover = np.where(
            (prev_values < prev_signal) & (self._values >= signal),
            1, 0
        )
        
        # クロスアンダー: 前の値がシグナル以上で、現在の値がシグナル未満
        crossunder = np.where(
            (prev_values >= prev_signal) & (self._values < signal),
            1, 0
        )
        
        return crossover, crossunder 