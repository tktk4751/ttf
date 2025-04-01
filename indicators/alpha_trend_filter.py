#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from .indicator import Indicator
from .alpha_trend_index import AlphaTrendIndex
from .efficiency_ratio import calculate_efficiency_ratio_for_period


@dataclass
class AlphaTrendFilterResult:
    """アルファトレンドフィルターの計算結果"""
    values: np.ndarray          # アルファトレンドフィルターの値（0-1の範囲で正規化）
    trend_index: np.ndarray     # アルファトレンドインデックスの値
    er: np.ndarray              # サイクル効率比（CER）
    combined_rms: np.ndarray    # トレンドインデックスとERの二乗平均平方根（RMS）
    rms_window: np.ndarray      # RMS計算のウィンドウサイズ
    dynamic_threshold: np.ndarray  # 動的しきい値


@njit(fastmath=True)
def calculate_simple_average_combination(trend_index: np.ndarray, er: np.ndarray, weight: float = 0.6) -> np.ndarray:
    """
    トレンドインデックスと効率比の単純な重み付き平均を計算する
    
    Args:
        trend_index: アルファトレンドインデックスの配列
        er: 効率比の配列
        weight: トレンドインデックスの重み (0-1)
        
    Returns:
        単純平均の配列
    """
    n = len(trend_index)
    result = np.zeros(n, dtype=np.float64)
    er_abs = np.abs(er)
    
    for i in range(n):
        if np.isnan(trend_index[i]) or np.isnan(er_abs[i]):
            result[i] = np.nan
        else:
            # 重み付き平均
            result[i] = weight * trend_index[i] + (1.0 - weight) * er_abs[i]
    
    return result


@njit(fastmath=True, parallel=True)
def calculate_rms_combination(trend_index: np.ndarray, er: np.ndarray, weight: float = 0.6, window: np.ndarray = None) -> np.ndarray:
    """
    トレンドインデックスと効率比の二乗平均平方根（RMS）を計算する
    
    Args:
        trend_index: アルファトレンドインデックスの配列
        er: 効率比の配列
        weight: トレンドインデックスの重み (0-1)
        window: 各ポイントでのRMS計算ウィンドウサイズの配列（Noneの場合は全データを使用）
        
    Returns:
        RMS値の配列
    """
    n = len(trend_index)
    result = np.zeros(n, dtype=np.float64)
    er_abs = np.abs(er)
    
    for i in prange(n):
        if np.isnan(trend_index[i]) or np.isnan(er_abs[i]):
            result[i] = np.nan
            continue
        
        # ウィンドウサイズの決定
        if window is not None and i >= 0 and i < len(window):
            win_size = int(window[i])
            if win_size < 1:
                win_size = 1
        else:
            win_size = i + 1  # 利用可能な全データ
        
        # 利用可能なデータポイント数を制限
        start_idx = max(0, i - win_size + 1)
        
        # トレンドインデックスとER（絶対値）のウィンドウを取得
        ti_window = trend_index[start_idx:i+1]
        er_window = er_abs[start_idx:i+1]
        
        # 重み付きRMS計算
        if len(ti_window) > 0:
            ti_squared_sum = np.sum(ti_window * ti_window)
            er_squared_sum = np.sum(er_window * er_window)
            
            ti_rms = np.sqrt(ti_squared_sum / len(ti_window))
            er_rms = np.sqrt(er_squared_sum / len(ti_window))
            
            # 重み付き組み合わせ
            result[i] = weight * ti_rms + (1.0 - weight) * er_rms
        else:
            result[i] = 0.0
    
    return result


@njit(fastmath=True)
def calculate_sigmoid_enhanced_combination(trend_index: np.ndarray, er: np.ndarray, weight: float = 0.6) -> np.ndarray:
    """
    トレンドインデックスと効率比を組み合わせ、シグモイド関数で強調する
    
    Args:
        trend_index: アルファトレンドインデックスの配列
        er: 効率比の配列
        weight: トレンドインデックスの重み (0-1)
        
    Returns:
        組み合わせた値の配列
    """
    n = len(trend_index)
    result = np.zeros(n, dtype=np.float64)
    er_abs = np.abs(er)
    
    for i in range(n):
        if np.isnan(trend_index[i]) or np.isnan(er_abs[i]):
            result[i] = np.nan
        else:
            # 重み付き平均
            combined = weight * trend_index[i] + (1.0 - weight) * er_abs[i]
            
            # シグモイド関数による強調（0.5を中心に差を強調）
            if combined != 0.5:
                sigmoid = 1.0 / (1.0 + np.exp(-6.0 * (combined - 0.5)))
                result[i] = sigmoid
            else:
                result[i] = 0.5
    
    return result


@njit(fastmath=True)
def calculate_dynamic_threshold(
    er: np.ndarray,
    max_threshold: float,
    min_threshold: float
) -> np.ndarray:
    """
    効率比に基づいて動的なしきい値を計算する（高速化版）
    
    効率比が高いほどしきい値は大きく、効率比が低いほどしきい値は小さくなる
    
    Args:
        er: 効率比の配列
        max_threshold: しきい値の最大値
        min_threshold: しきい値の最小値
    
    Returns:
        np.ndarray: 動的なしきい値の配列
    """
    # ERが高い（トレンドが強い）ほどしきい値は高く、
    # ERが低い（トレンドが弱い）ほどしきい値は低くなる
    return min_threshold + np.abs(er) * (max_threshold - min_threshold)


@njit(fastmath=True)
def calculate_alpha_trend_filter(
    trend_index: np.ndarray, 
    er: np.ndarray, 
    combination_weight: float, 
    combination_method: int,
    rms_window: np.ndarray,
    max_threshold: float,
    min_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    アルファトレンドフィルターのメイン計算ロジック
    
    Args:
        trend_index: アルファトレンドインデックスの配列
        er: 効率比の配列
        combination_weight: 組み合わせの重み
        combination_method: 組み合わせ方法 (0=sigmoid, 1=rms, 2=simple)
        rms_window: RMS計算ウィンドウサイズ
        max_threshold: しきい値の最大値
        min_threshold: しきい値の最小値
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (組み合わせ値, 動的しきい値)
    """
    # 選択された組み合わせ方法に基づいて値を計算
    if combination_method == 1:  # RMS
        combined_values = calculate_rms_combination(
            trend_index, er, combination_weight, rms_window
        )
    elif combination_method == 2:  # Simple
        combined_values = calculate_simple_average_combination(
            trend_index, er, combination_weight
        )
    else:  # Sigmoid (default)
        combined_values = calculate_sigmoid_enhanced_combination(
            trend_index, er, combination_weight
        )
    
    # 動的しきい値の計算
    dynamic_threshold = calculate_dynamic_threshold(
        er, max_threshold, min_threshold
    )
    
    return combined_values, dynamic_threshold


class AlphaTrendFilter(Indicator):
    """
    アルファトレンドフィルター（AlphaTrendFilter）インジケーター
    
    アルファトレンドインデックスとサイクル効率比（CER）を様々な方法で組み合わせた
    高度なトレンド/レンジ検出フィルターです。
    
    特徴:
    - アルファトレンドインデックスの信頼性の高いトレンド/レンジ検出
    - サイクル効率比（CER）との複数の組み合わせ方法：
      * シグモイド強調（デフォルト）: 値の差をシグモイド関数で強調
      * 二乗平均平方根（RMS）: 値の二乗平均平方根を計算
      * 単純平均: 値の重み付き平均を計算
    - 動的に調整されるパラメータによる市場状態への適応
    - 動的しきい値による柔軟なトレンド/レンジ判定
    - 0-1の範囲で正規化された出力値
    
    解釈:
    - 値がしきい値以上: 強いトレンド相場、明確な方向性
    - 値がしきい値未満: 強いレンジ相場、方向性の欠如
    """
    
    def __init__(
        self,
        max_chop_period: int = 89,
        min_chop_period: int = 21,
        max_atr_period: int = 89,
        min_atr_period: int = 21,
        max_stddev_period: int = 13,
        min_stddev_period: int = 5,
        max_lookback_period: int = 13,
        min_lookback_period: int = 5,
        max_rms_window: int = 13,
        min_rms_window: int = 5,
        max_threshold: float = 0.75,
        min_threshold: float = 0.55,
        # サイクル効率比(CER)のパラメーター
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 62,
        cycle_part: float = 0.5,
        # 組み合わせパラメータ
        combination_weight: float = 0.6,
        combination_method: str = "sigmoid"  # "sigmoid", "rms", "simple"
    ):
        """
        コンストラクタ
        
        Args:
            max_chop_period: チョピネス期間の最大値（デフォルト: 89）
            min_chop_period: チョピネス期間の最小値（デフォルト: 21）
            max_atr_period: ATR期間の最大値（デフォルト: 89）
            min_atr_period: ATR期間の最小値（デフォルト: 21）
            max_stddev_period: 標準偏差期間の最大値（デフォルト: 13）
            min_stddev_period: 標準偏差期間の最小値（デフォルト: 5）
            max_lookback_period: 最小標準偏差を探す最大ルックバック期間（デフォルト: 13）
            min_lookback_period: 最小標準偏差を探す最小ルックバック期間（デフォルト: 5）
            max_rms_window: RMS計算の最大ウィンドウサイズ（デフォルト: 13）
            min_rms_window: RMS計算の最小ウィンドウサイズ（デフォルト: 5）
            max_threshold: しきい値の最大値（デフォルト: 0.75）
            min_threshold: しきい値の最小値（デフォルト: 0.55）
            cycle_detector_type: ドミナントサイクル検出器の種類（デフォルト: 'hody_dc'）
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 62）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            combination_weight: トレンドインデックスとCERの組み合わせ重み（デフォルト: 0.6）
            combination_method: 組み合わせ方法
                "sigmoid": シグモイド強調（デフォルト）
                "rms": 二乗平均平方根
                "simple": 単純平均
        """
        super().__init__(
            f"AlphaTrendFilter({max_chop_period}, {min_chop_period})"
        )
        self.max_chop_period = max_chop_period
        self.min_chop_period = min_chop_period
        self.max_atr_period = max_atr_period
        self.min_atr_period = min_atr_period
        self.max_stddev_period = max_stddev_period
        self.min_stddev_period = min_stddev_period
        self.max_lookback_period = max_lookback_period
        self.min_lookback_period = min_lookback_period
        self.max_rms_window = max_rms_window
        self.min_rms_window = min_rms_window
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.combination_weight = combination_weight
        
        # 組み合わせ方法を数値に変換（Numbaで文字列比較を避けるため）
        if combination_method == "rms":
            self._combination_method_id = 1
        elif combination_method == "simple":
            self._combination_method_id = 2
        else:  # "sigmoid" or any other value
            self._combination_method_id = 0
        
        self.combination_method = combination_method
        
        # アルファトレンドインデックスインジケーターのインスタンス化
        self.alpha_trend_index = AlphaTrendIndex(
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_stddev_period=max_stddev_period,
            min_stddev_period=min_stddev_period,
            max_lookback_period=max_lookback_period,
            min_lookback_period=min_lookback_period,
            # サイクル効率比(CER)のパラメーター
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part
        )
        
        self._result = None
    
    def calculate(self, open_prices, high_prices, low_prices, close_prices) -> AlphaTrendFilterResult:
        """
        アルファトレンドフィルターを計算する

        Args:
            open_prices: 始値の配列
            high_prices: 高値の配列
            low_prices: 安値の配列
            close_prices: 終値の配列

        Returns:
            AlphaTrendFilterResult: 計算結果
        """
        
        # 配列をNumpy配列に変換
        o = np.asarray(open_prices, dtype=np.float64)
        h = np.asarray(high_prices, dtype=np.float64)
        l = np.asarray(low_prices, dtype=np.float64)
        c = np.asarray(close_prices, dtype=np.float64)
        
        # アルファトレンドインデックスを計算
        trend_index_result = self.alpha_trend_index.calculate(o, h, l, c)
        trend_index = trend_index_result.values
        er = trend_index_result.er
        
        # 効率比に基づいて動的RMSウィンドウサイズを計算
        from .alpha_trend_index import calculate_dynamic_period
        rms_window = calculate_dynamic_period(
            er, self.max_rms_window, self.min_rms_window
        )
        
        # アルファトレンドフィルターを計算
        combined_values, dynamic_threshold = calculate_alpha_trend_filter(
            trend_index, 
            er, 
            self.combination_weight, 
            self._combination_method_id,
            rms_window,
            self.max_threshold,
            self.min_threshold
        )
        
        # 結果オブジェクトを作成
        result = AlphaTrendFilterResult(
            values=combined_values,
            trend_index=trend_index,
            er=er,
            combined_rms=combined_values,  # 互換性のために名前を維持
            rms_window=rms_window,
            dynamic_threshold=dynamic_threshold
        )
        
        # 内部の結果変数に保存
        self._result = result
        self._values = combined_values
        
        return result
    
    def get_trend_index(self) -> np.ndarray:
        """
        アルファトレンドインデックスの値を取得する
        
        Returns:
            np.ndarray: アルファトレンドインデックスの値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.trend_index
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.er
    
    def get_combined_rms(self) -> np.ndarray:
        """
        組み合わせRMS値を取得する
        
        Returns:
            np.ndarray: トレンドインデックスとERの組み合わせRMS値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.combined_rms
    
    def get_rms_window(self) -> np.ndarray:
        """
        RMS計算のウィンドウサイズを取得する
        
        Returns:
            np.ndarray: RMS計算のウィンドウサイズ
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.rms_window
    
    def get_dynamic_threshold(self) -> np.ndarray:
        """
        動的しきい値を取得する
        
        Returns:
            np.ndarray: 動的しきい値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.dynamic_threshold
    
    def get_state(self) -> np.ndarray:
        """
        フィルターの状態を取得する
        
        Returns:
            np.ndarray: 状態値の配列
                1: 値がしきい値以上（トレンド相場）
                -1: 値がしきい値未満（レンジ相場）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        state = np.zeros_like(self._result.values)
        state[self._result.values >= self._result.dynamic_threshold] = 1  # トレンド相場
        state[self._result.values < self._result.dynamic_threshold] = -1  # レンジ相場
        
        return state
    
    def reset(self) -> None:
        """
        インジケーターの状態をリセットする
        """
        super().reset()
        self.alpha_trend_index.reset()
        self._result = None 