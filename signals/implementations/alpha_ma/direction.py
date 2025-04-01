#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.alpha_ma import AlphaMA


@jit(nopython=True)
def calculate_direction_signal(short_ma: np.ndarray, long_ma: np.ndarray) -> np.ndarray:
    """
    方向性シグナルを計算する（高速化版）
    
    Args:
        short_ma: 短期移動平均の配列
        long_ma: 長期移動平均の配列
    
    Returns:
        シグナルの配列 (1: ロング方向, -1: ショート方向)
    """
    length = len(short_ma)
    signals = np.zeros(length)
    
    for i in range(length):
        if short_ma[i] > long_ma[i]:
            signals[i] = 1  # ロング方向
        else:
            signals[i] = -1  # ショート方向
    
    return signals


@jit(nopython=True)
def calculate_triple_direction_signal(short_ma: np.ndarray, middle_ma: np.ndarray, long_ma: np.ndarray) -> np.ndarray:
    """
    3本の移動平均を使用した方向性シグナルを計算する（高速化版）
    
    Args:
        short_ma: 短期移動平均の配列
        middle_ma: 中期移動平均の配列
        long_ma: 長期移動平均の配列
    
    Returns:
        シグナルの配列 (1: 上昇配列, -1: 下降配列, 0: その他)
    """
    length = len(short_ma)
    signals = np.zeros(length)
    
    for i in range(length):
        if short_ma[i] > middle_ma[i] and middle_ma[i] > long_ma[i]:
            signals[i] = 1  # 上昇配列
        elif short_ma[i] < middle_ma[i] and middle_ma[i] < long_ma[i]:
            signals[i] = -1  # 下降配列
        else:
            signals[i] = 0  # その他
    
    return signals


@jit(nopython=True)
def calculate_circulation_stages(short_ma: np.ndarray, middle_ma: np.ndarray, long_ma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    移動平均線大循環のステージとシグナルを計算する（高速化版）
    
    Args:
        short_ma: 短期移動平均の配列
        middle_ma: 中期移動平均の配列
        long_ma: 長期移動平均の配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: ステージの配列とシグナルの配列
    """
    length = len(short_ma)
    stages = np.zeros(length)
    signals = np.zeros(length)
    
    for i in range(length):
        # ステージ1: 短期 > 中期 > 長期
        if short_ma[i] > middle_ma[i] and middle_ma[i] > long_ma[i]:
            stages[i] = 1
            signals[i] = 1  # ロング方向
        
        # ステージ2: 中期 > 短期 > 長期
        elif middle_ma[i] > short_ma[i] and short_ma[i] > long_ma[i]:
            stages[i] = 2
            signals[i] = 1  # ロング方向
        
        # ステージ3: 中期 > 長期 > 短期
        elif middle_ma[i] > long_ma[i] and long_ma[i] > short_ma[i]:
            stages[i] = 3
            signals[i] = -1  # ショート方向
        
        # ステージ4: 長期 > 中期 > 短期
        elif long_ma[i] > middle_ma[i] and middle_ma[i] > short_ma[i]:
            stages[i] = 4
            signals[i] = -1  # ショート方向
        
        # ステージ5: 長期 > 短期 > 中期
        elif long_ma[i] > short_ma[i] and short_ma[i] > middle_ma[i]:
            stages[i] = 5
            signals[i] = -1  # ショート方向
        
        # ステージ6: 短期 > 長期 > 中期
        elif short_ma[i] > long_ma[i] and long_ma[i] > middle_ma[i]:
            stages[i] = 6
            signals[i] = 1  # ロング方向
        
        # その他の配列パターン
        else:
            stages[i] = 0
            signals[i] = 0  # 中立
    
    return stages, signals


class AlphaMATrendFollowingSignal(BaseSignal, IDirectionSignal):
    """
    AlphaMAを使用した方向性シグナル
    - 短期AlphaMA > 長期AlphaMA: ロング方向 (1)
    - 短期AlphaMA < 長期AlphaMA: ショート方向 (-1)
    
    特徴:
    - 効率比（ER）に基づいて動的に調整されるAlphaMAを使用
    - トレンドが強い時は短いピリオドで速く反応
    - レンジ相場時は長いピリオドでノイズを除去
    """
    
    def __init__(
        self,
        short_er_period: int = 21,
        long_er_period: int = 21,
        short_max_kama_period: int = 89,
        short_min_kama_period: int = 8,
        long_max_kama_period: int = 144,
        long_min_kama_period: int = 21,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2
    ):
        """
        コンストラクタ
        
        Args:
            short_er_period: 短期AlphaMAの効率比の計算期間
            long_er_period: 長期AlphaMAの効率比の計算期間
            short_max_kama_period: 短期AlphaMAのKAMAピリオドの最大値
            short_min_kama_period: 短期AlphaMAのKAMAピリオドの最小値
            long_max_kama_period: 長期AlphaMAのKAMAピリオドの最大値
            long_min_kama_period: 長期AlphaMAのKAMAピリオドの最小値
            max_slow_period: 遅い移動平均の最大期間
            min_slow_period: 遅い移動平均の最小期間
            max_fast_period: 速い移動平均の最大期間
            min_fast_period: 速い移動平均の最小期間
        """
        params = {
            'short_er_period': short_er_period,
            'long_er_period': long_er_period,
            'short_max_kama_period': short_max_kama_period,
            'short_min_kama_period': short_min_kama_period,
            'long_max_kama_period': long_max_kama_period,
            'long_min_kama_period': long_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period
        }
        super().__init__(
            f"AlphaMATrendFollowing({short_min_kama_period}, {long_min_kama_period})",
            params
        )
        
        # AlphaMAインジケーターの初期化
        self._short_alpha_ma = AlphaMA(
            er_period=short_er_period,
            max_kama_period=short_max_kama_period,
            min_kama_period=short_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        
        self._long_alpha_ma = AlphaMA(
            er_period=long_er_period,
            max_kama_period=long_max_kama_period,
            min_kama_period=long_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング方向, -1: ショート方向)
        """
        # AlphaMAの計算
        short_alpha_ma = self._short_alpha_ma.calculate(data)
        long_alpha_ma = self._long_alpha_ma.calculate(data)
        
        # 方向性シグナルの生成（高速化版）
        signals = calculate_direction_signal(short_alpha_ma, long_alpha_ma)
        
        return signals


class AlphaMACirculationSignal(BaseSignal, IDirectionSignal):
    """
    AlphaMAを使用した方向性シグナル（移動平均線大循環）
    
    ステージの定義:
    1: 短期 > 中期 > 長期  （安定上昇相場）
    2: 中期 > 短期 > 長期  （上昇相場の終焉）
    3: 中期 > 長期 > 短期   (下降相場の入口)
    4: 長期 > 中期 > 短期   (安定下降相場)
    5: 長期 > 短期 > 中期   (下降相場の終焉)
    6: 短期 > 長期 > 中期   (上昇相場の入口)
    
    特徴:
    - 効率比（ER）に基づいて動的に調整されるAlphaMAを使用
    - トレンドが強い時は短いピリオドで速く反応
    - レンジ相場時は長いピリオドでノイズを除去
    """
    
    def __init__(
        self,
        er_period: int = 21,
        short_max_kama_period: int = 55,
        short_min_kama_period: int = 3,
        middle_max_kama_period: int = 144,
        middle_min_kama_period: int = 21,
        long_max_kama_period: int = 377,
        long_min_kama_period: int = 55,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2
    ):
        """
        コンストラクタ
        
        Args:
            short_er_period: 短期AlphaMAの効率比の計算期間
            middle_er_period: 中期AlphaMAの効率比の計算期間
            long_er_period: 長期AlphaMAの効率比の計算期間
            short_max_kama_period: 短期AlphaMAのKAMAピリオドの最大値
            short_min_kama_period: 短期AlphaMAのKAMAピリオドの最小値
            middle_max_kama_period: 中期AlphaMAのKAMAピリオドの最大値
            middle_min_kama_period: 中期AlphaMAのKAMAピリオドの最小値
            long_max_kama_period: 長期AlphaMAのKAMAピリオドの最大値
            long_min_kama_period: 長期AlphaMAのKAMAピリオドの最小値
            max_slow_period: 遅い移動平均の最大期間
            min_slow_period: 遅い移動平均の最小期間
            max_fast_period: 速い移動平均の最大期間
            min_fast_period: 速い移動平均の最小期間
        """
        params = {
            'short_er_period': er_period,
            'middle_er_period': er_period,
            'long_er_period': er_period,
            'short_max_kama_period': short_max_kama_period,
            'short_min_kama_period': short_min_kama_period,
            'middle_max_kama_period': middle_max_kama_period,
            'middle_min_kama_period': middle_min_kama_period,
            'long_max_kama_period': long_max_kama_period,
            'long_min_kama_period': long_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period
        }
        super().__init__(
            f"AlphaMACirculation({short_min_kama_period}, {middle_min_kama_period}, {long_min_kama_period})",
            params
        )
        
        # AlphaMAインジケーターの初期化
        self._short_alpha_ma = AlphaMA(
            er_period=er_period,
            max_kama_period=short_max_kama_period,
            min_kama_period=short_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        
        self._middle_alpha_ma = AlphaMA(
            er_period=er_period,
            max_kama_period=middle_max_kama_period,
            min_kama_period=middle_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        
        self._long_alpha_ma = AlphaMA(
            er_period=er_period,
            max_kama_period=long_max_kama_period,
            min_kama_period=long_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング方向, -1: ショート方向, 0: 中立)
        """
        # AlphaMAの計算
        short_alpha_ma = self._short_alpha_ma.calculate(data)
        middle_alpha_ma = self._middle_alpha_ma.calculate(data)
        long_alpha_ma = self._long_alpha_ma.calculate(data)
        
        # ステージとシグナルの生成（高速化版）
        _, signals = calculate_circulation_stages(
            short_alpha_ma, middle_alpha_ma, long_alpha_ma
        )
        
        return signals
    
    def get_stage(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        現在のステージを返す
        
        Args:
            data: 価格データ
        
        Returns:
            ステージ番号の配列 (1-6)
            1: 短期 > 中期 > 長期  （安定上昇相場）
            2: 中期 > 短期 > 長期  （上昇相場の終焉）
            3: 中期 > 長期 > 短期   (下降相場の入口)
            4: 長期 > 中期 > 短期   (安定下降相場)
            5: 長期 > 短期 > 中期   (下降相場の終焉)
            6: 短期 > 長期 > 中期   (上昇相場の入口)
        """
        # AlphaMAの計算
        short_alpha_ma = self._short_alpha_ma.calculate(data)
        middle_alpha_ma = self._middle_alpha_ma.calculate(data)
        long_alpha_ma = self._long_alpha_ma.calculate(data)
        
        # ステージの生成（高速化版）
        stages, _ = calculate_circulation_stages(
            short_alpha_ma, middle_alpha_ma, long_alpha_ma
        )
        
        return stages


class AlphaMADirectionSignal2(BaseSignal, IDirectionSignal):
    """AlphaMAディレクションシグナル2
    
    n期間AlphaMAが終値より下にあるときは1（上昇トレンド）、
    上にあるときは-1（下降トレンド）を出力する
    
    特徴:
    - 効率比（ER）に基づいて動的に調整されるAlphaMAを使用
    - トレンドが強い時は短いピリオドで速く反応
    - レンジ相場時は長いピリオドでノイズを除去
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_kama_period: int = 233,
        min_kama_period: int = 55,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2
    ):
        """
        初期化
        
        Args:
            er_period: 効率比の計算期間
            max_kama_period: KAMAピリオドの最大値
            min_kama_period: KAMAピリオドの最小値
            max_slow_period: 遅い移動平均の最大期間
            min_slow_period: 遅い移動平均の最小期間
            max_fast_period: 速い移動平均の最大期間
            min_fast_period: 速い移動平均の最小期間
        """
        params = {
            'er_period': er_period,
            'max_kama_period': max_kama_period,
            'min_kama_period': min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period
        }
        super().__init__(f"AlphaMADirectionSignal2({min_kama_period})", params)
        
        # AlphaMAインジケーターの初期化
        self.alpha_ma = AlphaMA(
            er_period=er_period,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            1（上昇トレンド）または-1（下降トレンド）の配列
        """
        # データの準備
        if isinstance(data, pd.DataFrame):
            close = data['close'].values
        else:
            if data.ndim == 2:
                close = data[:, 3]  # close
            else:
                close = data  # 1次元配列として扱う
        
        # AlphaMAの計算
        alpha_ma_values = self.alpha_ma.calculate(data)
        
        # シグナルの生成
        signals = np.where(close > alpha_ma_values, 1, -1)
        
        return signals


class AlphaMATripleDirectionSignal(BaseSignal, IDirectionSignal):
    """
    3本のAlphaMAを使用したディレクションシグナル
    
    シグナル条件:
    1: 短期 > 中期 > 長期 （完全な上昇配列）
    -1: 短期 < 中期 < 長期 （完全な下降配列）
    0: その他の配列パターン
    
    特徴:
    - 効率比（ER）に基づいて動的に調整されるAlphaMAを使用
    - トレンドが強い時は短いピリオドで速く反応
    - レンジ相場時は長いピリオドでノイズを除去
    """
    
    def __init__(
        self,
        short_er_period: int = 21,
        middle_er_period: int = 21,
        long_er_period: int = 21,
        short_max_kama_period: int = 89,
        short_min_kama_period: int = 8,
        middle_max_kama_period: int = 144,
        middle_min_kama_period: int = 21,
        long_max_kama_period: int = 233,
        long_min_kama_period: int = 55,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2
    ):
        """
        コンストラクタ
        
        Args:
            short_er_period: 短期AlphaMAの効率比の計算期間
            middle_er_period: 中期AlphaMAの効率比の計算期間
            long_er_period: 長期AlphaMAの効率比の計算期間
            short_max_kama_period: 短期AlphaMAのKAMAピリオドの最大値
            short_min_kama_period: 短期AlphaMAのKAMAピリオドの最小値
            middle_max_kama_period: 中期AlphaMAのKAMAピリオドの最大値
            middle_min_kama_period: 中期AlphaMAのKAMAピリオドの最小値
            long_max_kama_period: 長期AlphaMAのKAMAピリオドの最大値
            long_min_kama_period: 長期AlphaMAのKAMAピリオドの最小値
            max_slow_period: 遅い移動平均の最大期間
            min_slow_period: 遅い移動平均の最小期間
            max_fast_period: 速い移動平均の最大期間
            min_fast_period: 速い移動平均の最小期間
        """
        params = {
            'short_er_period': short_er_period,
            'middle_er_period': middle_er_period,
            'long_er_period': long_er_period,
            'short_max_kama_period': short_max_kama_period,
            'short_min_kama_period': short_min_kama_period,
            'middle_max_kama_period': middle_max_kama_period,
            'middle_min_kama_period': middle_min_kama_period,
            'long_max_kama_period': long_max_kama_period,
            'long_min_kama_period': long_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period
        }
        super().__init__(
            f"AlphaMATripleDirection({short_min_kama_period}, {middle_min_kama_period}, {long_min_kama_period})",
            params
        )
        
        # AlphaMAインジケーターの初期化
        self._short_alpha_ma = AlphaMA(
            er_period=short_er_period,
            max_kama_period=short_max_kama_period,
            min_kama_period=short_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        
        self._middle_alpha_ma = AlphaMA(
            er_period=middle_er_period,
            max_kama_period=middle_max_kama_period,
            min_kama_period=middle_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        
        self._long_alpha_ma = AlphaMA(
            er_period=long_er_period,
            max_kama_period=long_max_kama_period,
            min_kama_period=long_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: 上昇配列, -1: 下降配列, 0: その他)
        """
        # AlphaMAの計算
        short_alpha_ma = self._short_alpha_ma.calculate(data)
        middle_alpha_ma = self._middle_alpha_ma.calculate(data)
        long_alpha_ma = self._long_alpha_ma.calculate(data)
        
        # シグナルの生成（高速化版）
        signals = calculate_triple_direction_signal(
            short_alpha_ma, middle_alpha_ma, long_alpha_ma
        )
        
        return signals 