#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Tuple, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .rsx import calculate_rsx_numba, RSX
from .efficiency_ratio import EfficiencyRatio


@jit(nopython=True)
def calculate_adaptive_period(
    er_values: np.ndarray,
    min_period: int = 14,
    max_period: int = 21
) -> np.ndarray:
    """
    効率比（Efficiency Ratio）に基づいて適応的にRSXの期間を計算する
    
    Args:
        er_values: 効率比の配列
        min_period: 最小期間（デフォルト: 14）
        max_period: 最大期間（デフォルト: 21）
    
    Returns:
        適応的なRSX期間の配列（整数値）
    """
    # ERの値が0〜1の範囲なので、これをmin_period〜max_periodの範囲にマッピング
    # ERが高い（トレンドが強い）→ 短い期間（より反応的に）
    # ERが低い（レンジ相場）→ 長い期間（よりスムーズに）
    adaptive_periods = np.round(max_period - er_values * (max_period - min_period)).astype(np.int32)
    
    return adaptive_periods


@jit(nopython=True)
def calculate_adaptive_levels(
    er_values: np.ndarray,
    min_high_level: float = 85.0,
    max_high_level: float = 90.0,
    min_low_level: float = 10.0,
    max_low_level: float = 15.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    効率比（Efficiency Ratio）に基づいて適応的に高値/安値レベルを計算する
    
    Args:
        er_values: 効率比の配列
        min_high_level: 最小高値レベル（デフォルト: 85.0）
        max_high_level: 最大高値レベル（デフォルト: 90.0）
        min_low_level: 最小安値レベル（デフォルト: 10.0）
        max_low_level: 最大安値レベル（デフォルト: 15.0）
    
    Returns:
        (適応的な高値レベルの配列, 適応的な安値レベルの配列)のタプル
    """
    # ERが高い（トレンドが強い）→ より厳しいレベル（買われすぎ/売られすぎの判断を厳格に）
    # ERが低い（レンジ相場）→ より緩いレベル（買われすぎ/売られすぎの判断を緩やかに）
    adaptive_high_levels = min_high_level + er_values * (max_high_level - min_high_level)
    adaptive_low_levels = min_low_level + er_values * (max_low_level - min_low_level)
    
    return adaptive_high_levels, adaptive_low_levels


class AlphaRSX(Indicator):
    """
    Alpha RSX インディケーター
    
    Jurik RSX (Relative Strength eXtended)の拡張版。
    効率比（Efficiency Ratio）に基づいて以下のパラメータを動的に調整:
    - 計算期間（最小14、最大21）
    - 買われすぎレベル（最小85、最大90）
    - 売られすぎレベル（最小10、最大15）
    
    市場の状態に応じて最適なパラメータを自動調整することで、
    より正確なシグナルと市場の状況に適応した分析が可能。
    
    特徴:
    - 強いトレンド時：より短い期間（反応的）と厳格なレベル設定
    - レンジ相場時：より長い期間（スムーズ）と緩やかなレベル設定
    - RSXの優れたノイズ除去能力と反応速度を維持
    
    使用方法:
    - 買われすぎ/売られすぎレベルの確認（動的に調整される）
    - クロスオーバー/クロスアンダーでのエントリー/エグジットの確認
    - トレンドの方向と強さの分析
    """
    
    def __init__(
        self,
        er_period: int = 10,
        min_period: int = 14,
        max_period: int = 21,
        min_high_level: float = 85.0,
        max_high_level: float = 90.0,
        min_low_level: float = 10.0,
        max_low_level: float = 15.0
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 10）
            min_period: RSXの最小期間（デフォルト: 14）
            max_period: RSXの最大期間（デフォルト: 21）
            min_high_level: 最小買われすぎレベル（デフォルト: 85.0）
            max_high_level: 最大買われすぎレベル（デフォルト: 90.0）
            min_low_level: 最小売られすぎレベル（デフォルト: 10.0）
            max_low_level: 最大売られすぎレベル（デフォルト: 15.0）
        """
        super().__init__(f"AlphaRSX({min_period}-{max_period})")
        self.er_period = er_period
        self.min_period = min_period
        self.max_period = max_period
        self.min_high_level = min_high_level
        self.max_high_level = max_high_level
        self.min_low_level = min_low_level
        self.max_low_level = max_low_level
        
        # サブインジケーター
        self.er = EfficiencyRatio(period=er_period)
        
        # 適応的なパラメーター用の配列
        self._adaptive_periods = None
        self._adaptive_high_levels = None
        self._adaptive_low_levels = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Alpha RSXを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、HLC3が計算されます（High, Low, Closeの平均）
        
        Returns:
            Alpha RSXの配列（0-100の範囲）
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
        self._validate_period(self.max_period, data_length)
        
        # 効率比（ER）の計算
        er_values = self.er.calculate(data)
        
        # 適応的なパラメーターの計算
        self._adaptive_periods = calculate_adaptive_period(
            er_values, 
            self.min_period, 
            self.max_period
        )
        
        self._adaptive_high_levels, self._adaptive_low_levels = calculate_adaptive_levels(
            er_values,
            self.min_high_level,
            self.max_high_level,
            self.min_low_level,
            self.max_low_level
        )
        
        # 通常のRSXよりも厳密な計算が必要
        # 各ポイントで異なる期間を使用するので、各時点ごとに計算する
        size = len(prices)
        self._values = np.zeros(size)
        
        # 最初の部分は常に0（データが不足）
        # 最大期間から計算開始
        for i in range(self.max_period, size):
            # その時点での適応的なperiodを取得
            adaptive_period = int(self._adaptive_periods[i])
            
            # その時点でのRSXを計算（適応的な期間を使用）
            # 効率のため、必要な部分の配列だけを使用
            window = prices[i-adaptive_period:i+1]
            rsx_value = calculate_rsx_numba(window, adaptive_period)[-1]
            
            self._values[i] = rsx_value
        
        return self._values
    
    def get_adaptive_periods(self) -> np.ndarray:
        """
        適応的な期間を取得
        
        Returns:
            適応的な期間の配列
        """
        if self._adaptive_periods is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._adaptive_periods
    
    def get_adaptive_levels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        適応的な高値/安値レベルを取得
        
        Returns:
            (適応的な高値レベルの配列, 適応的な安値レベルの配列)のタプル
        """
        if self._adaptive_high_levels is None or self._adaptive_low_levels is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._adaptive_high_levels, self._adaptive_low_levels
    
    def get_overbought_oversold(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        買われすぎ・売られすぎのシグナルを取得（適応的なレベルを使用）
        
        Returns:
            (買われすぎシグナル, 売られすぎシグナル)のタプル
            シグナルは、条件を満たす場合は1、そうでない場合は0
        """
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        # 適応的なレベルを使用して買われすぎ/売られすぎを判定
        overbought_signal = np.where(self._values >= self._adaptive_high_levels, 1, 0)
        oversold_signal = np.where(self._values <= self._adaptive_low_levels, 1, 0)
        
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