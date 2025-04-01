#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from .indicator import Indicator
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .hyper_smoother import hyper_smoother
from .cycle_efficiency_ratio import CycleEfficiencyRatio
from .alpha_atr import AlphaATR


@dataclass
class AlphaTrendIndexResult:
    """アルファトレンドインデックスの計算結果"""
    values: np.ndarray          # アルファトレンドインデックスの値（0-1の範囲）
    er: np.ndarray              # サイクル効率比（CER）
    dynamic_chop_period: np.ndarray  # 動的チョピネス期間
    dynamic_atr_period: np.ndarray   # 動的ATR期間
    dynamic_stddev_period: np.ndarray # 動的標準偏差期間
    tr: np.ndarray              # True Range
    atr: np.ndarray             # Average True Range (AlphaATR)
    choppiness_index: np.ndarray # Choppiness Index（元の値）
    range_index: np.ndarray     # Range Index（元の値）
    stddev_factor: np.ndarray   # 標準偏差係数


@njit(fastmath=True)
def calculate_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    True Rangeを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        True Range の配列
    """
    length = len(high)
    tr = np.zeros(length, dtype=np.float64)
    
    # 最初の要素は単純なレンジ
    tr[0] = high[0] - low[0]
    
    # 2番目以降の要素はTRを計算
    for i in range(1, length):
        tr1 = high[i] - low[i]  # 当日の高値 - 当日の安値
        tr2 = abs(high[i] - close[i-1])  # |当日の高値 - 前日の終値|
        tr3 = abs(low[i] - close[i-1])  # |当日の安値 - 前日の終値|
        tr[i] = max(tr1, tr2, tr3)
    
    return tr


@njit(fastmath=True)
def calculate_normalized_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    正規化True Rangeを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        正規化True Range の配列
    """
    length = len(high)
    ntr = np.zeros(length, dtype=np.float64)
    
    # 2番目以降の要素はNormalized TRを計算
    for i in range(1, length):
        tr1 = high[i] - low[i]  # 当日の高値 - 当日の安値
        tr2 = abs(high[i] - close[i-1])  # |当日の高値 - 前日の終値|
        tr3 = abs(low[i] - close[i-1])  # |当日の安値 - 前日の終値|
        tr_val = max(tr1, tr2, tr3)
        # クローズで正規化
        if close[i] > 0:
            ntr[i] = tr_val / close[i]
        else:
            ntr[i] = tr_val
    
    return ntr


@njit(fastmath=True)
def calculate_dynamic_period(er: np.ndarray, max_period: int, min_period: int) -> np.ndarray:
    """
    効率比に基づいて動的な期間を計算する
    
    Args:
        er: 効率比の配列
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        動的な期間の配列
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    # 反転した効率比（1-ER）を使う
    inv_er = 1.0 - np.abs(er)
    periods = min_period + inv_er * (max_period - min_period)
    return np.round(periods).astype(np.int32)


@njit(fastmath=True)
def calculate_sma(src: np.ndarray, length: int) -> np.ndarray:
    """
    単純移動平均（SMA）を計算する
    
    Args:
        src: 入力データ配列
        length: 期間
        
    Returns:
        計算されたSMA
    """
    n = len(src)
    result = np.zeros(n, dtype=np.float64)
    
    # 最初のlength-1ポイントは不完全なウィンドウ
    for i in range(n):
        if i < length:
            # 不完全なウィンドウでは利用可能なデータの平均を取る
            result[i] = np.mean(src[:i+1])
        else:
            # 完全なウィンドウ
            result[i] = np.mean(src[i-length+1:i+1])
    
    return result


@njit(fastmath=True)
def calculate_atr(tr: np.ndarray, period: np.ndarray) -> np.ndarray:
    """
    動的期間を使用してAverage True Rangeを計算する
    
    Args:
        tr: True Range配列
        period: 期間配列（各バーの期間）
        
    Returns:
        ATR配列
    """
    n = len(tr)
    atr = np.zeros(n, dtype=np.float64)
    
    # 異なる期間ごとにATRを別々に計算する必要がある
    for i in range(1, n):
        # 直前のバーと異なる期間の場合、計算を再調整
        if i > 0 and (i == 1 or period[i] != period[i-1]):
            # 現在の期間に対応するインデックスを見つける
            current_period = int(period[i])
            start_idx = max(0, i - current_period)
            
            # このバーのATRを計算（SMA）
            if i - start_idx > 0:  # データが十分ある場合
                atr[i] = np.mean(tr[start_idx:i+1])
            else:
                atr[i] = tr[i]  # データが不足している場合はTRを使用
        else:
            # 期間が同じ場合はSMAを継続
            if i == 0:
                atr[i] = tr[i]
            else:
                current_period = int(period[i])
                if current_period > 0:
                    alpha = 1.0 / current_period
                    atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
                else:
                    atr[i] = tr[i]
    
    return atr


@njit(fastmath=True, parallel=True)
def calculate_choppiness_index(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """
    動的期間によるチョピネス指数を計算する
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period: 動的な期間の配列
        tr: True Rangeの配列
    
    Returns:
        チョピネス指数の配列（0-100の範囲）
    """
    length = len(high)
    chop = np.zeros(length, dtype=np.float64)
    
    for i in prange(1, length):
        curr_period = int(period[i])
        if curr_period < 2:
            curr_period = 2
        
        if i < curr_period:
            continue
        
        # True Range の合計
        tr_sum = np.sum(tr[i - curr_period + 1:i + 1])
        
        # 期間内の最高値と最安値を取得
        idx_start = i - curr_period + 1
        period_high = np.max(high[idx_start:i + 1])
        period_low = np.min(low[idx_start:i + 1])
        
        # 価格レンジの計算
        price_range = period_high - period_low
        
        # チョピネス指数の計算
        if price_range > 0 and curr_period > 1 and tr_sum > 0:
            log_period = np.log10(float(curr_period))
            chop_val = 100.0 * np.log10(tr_sum / price_range) / log_period
            # 値を0-100の範囲に制限
            chop[i] = max(0.0, min(100.0, chop_val))
        else:
            chop[i] = 0.0
    
    return chop


@njit(fastmath=True, parallel=True)
def calculate_stddev_factor(atr: np.ndarray, period: np.ndarray, lookback_period: np.ndarray) -> np.ndarray:
    """
    ATRの標準偏差係数を計算する
    
    Args:
        atr: ATR配列
        period: 期間配列（各バーの期間）
        lookback_period: 標準偏差の計算に使用する動的ルックバック期間
        
    Returns:
        標準偏差係数
    """
    n = len(atr)
    stddev = np.zeros(n, dtype=np.float64)
    lowest_stddev = np.full(n, np.inf, dtype=np.float64)  # 最小値を追跡するための配列
    stddev_factor = np.ones(n, dtype=np.float64)  # デフォルトは1
    
    for i in prange(n):
        # 標準偏差の計算期間を決定
        current_period = int(period[i])
        if current_period <= 1:
            current_period = 2
            
        # ルックバック期間を決定
        current_lookback = int(lookback_period[i])
        if current_lookback <= 0:
            current_lookback = 1
        
        if i >= current_period - 1:
            # 利用可能なデータの取得
            start_idx = max(0, i - current_period + 1)
            atr_window = atr[start_idx:i+1]
            
            # PineScriptのSMAを使用した計算方法
            stddev_a = np.mean(np.power(atr_window, 2))
            stddev_b = np.power(np.sum(atr_window), 2) / np.power(len(atr_window), 2)
            curr_stddev = np.sqrt(max(0.0, stddev_a - stddev_b))
            
            stddev[i] = curr_stddev
            
            # 最小標準偏差の更新（ルックバック期間内で）
            if i == current_period - 1 or (i > 0 and curr_stddev < lowest_stddev[i-1]):
                lowest_stddev[i] = curr_stddev
            else:
                lowest_lookback_start = max(0, i - current_lookback + 1)
                if i > lowest_lookback_start:
                    lowest_in_period = np.min(stddev[lowest_lookback_start:i+1])
                    lowest_stddev[i] = lowest_in_period
                else:
                    lowest_stddev[i] = stddev[i]
            
            # 標準偏差係数の計算
            if stddev[i] > 0:
                stddev_factor[i] = lowest_stddev[i] / stddev[i]
            else:
                stddev_factor[i] = 1.0
        elif i > 0:
            # データ不足の場合は前の値を使用
            stddev[i] = stddev[i-1]
            lowest_stddev[i] = lowest_stddev[i-1]
            stddev_factor[i] = stddev_factor[i-1]
    
    return stddev_factor


@njit(fastmath=True)
def calculate_alpha_trend_index(
    chop: np.ndarray,
    stddev_factor: np.ndarray
) -> np.ndarray:
    """
    アルファトレンドインデックスを計算する
    
    Args:
        chop: チョピネス指数の配列（0-100の範囲）
        stddev_factor: 標準偏差ファクターの配列
    
    Returns:
        アルファトレンドインデックスの配列（0-1の範囲、1に近いほど強いトレンド）
    """
    # チョピネス指数と標準偏差係数を組み合わせたレンジインデックスを計算
    range_index = chop * stddev_factor
    
    # トレンド指数として常に反転し、0-1に正規化
    trend_index = 1.0 - (range_index / 100.0)
    
    return trend_index


@njit(fastmath=True)
def calculate_alpha_trend_index_batch(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray,
    er: np.ndarray, 
    max_chop_period: int, 
    min_chop_period: int,
    max_stddev_period: int, 
    min_stddev_period: int,
    max_lookback_period: int, 
    min_lookback_period: int,
    atr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    アルファトレンドインデックスを一括計算する

    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        er: 効率比の配列
        max_chop_period: チョピネス期間の最大値
        min_chop_period: チョピネス期間の最小値
        max_stddev_period: 標準偏差期間の最大値
        min_stddev_period: 標準偏差期間の最小値
        max_lookback_period: 標準偏差ルックバック期間の最大値
        min_lookback_period: 標準偏差ルックバック期間の最小値
        atr: 既に計算されたATR配列

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            (アルファトレンドインデックス, 動的チョピネス期間, 動的標準偏差期間, チョピネス指数, 標準偏差係数)
    """
    # 動的期間の計算
    dynamic_chop_period = calculate_dynamic_period(
        er, max_chop_period, min_chop_period
    )
    
    dynamic_stddev_period = calculate_dynamic_period(
        er, max_stddev_period, min_stddev_period
    )
    
    dynamic_lookback_period = calculate_dynamic_period(
        er, max_lookback_period, min_lookback_period
    )
    
    # True Rangeの計算
    tr = calculate_tr(high, low, close)
    
    # チョピネスインデックスの計算
    chop_index = calculate_choppiness_index(high, low, close, dynamic_chop_period, tr)
    
    # 標準偏差係数の計算
    stddev_factor = calculate_stddev_factor(atr, dynamic_stddev_period, dynamic_lookback_period)
    
    # トレンドインデックスの計算
    trend_index = calculate_alpha_trend_index(chop_index, stddev_factor)
    
    return trend_index, dynamic_chop_period, dynamic_stddev_period, chop_index, stddev_factor


class AlphaTrendIndex(Indicator):
    """
    アルファトレンドインデックス（Alpha Trend Index）インジケーター
    
    サイクル効率比（CER）を使用してチョピネスインデックス、アルファATR、
    標準偏差などを動的に調整する高度なトレンド/レンジ検出指標です。
    
    特徴:
    - サイクル効率比（CER）を使用して、現在のサイクルに基づいた適応的な計算
    - アルファATRを使用した高度なボラティリティ測定
    - チョピネスインデックスと標準偏差係数を組み合わせて正規化したトレンド指標
    - 市場状態に応じて各期間が自動調整される
    - 0-1の範囲で表示（1に近いほど強いトレンド、0に近いほど強いレンジ）
    
    CERはドミナントサイクル検出を使用して効率比を計算するため、
    より正確な市場状態の把握が可能になります。
    """
    
    def __init__(
        self,
        max_chop_period: int = 21,
        min_chop_period: int = 8,
        max_atr_period: int = 21,
        min_atr_period: int = 10,
        max_stddev_period: int = 21,
        min_stddev_period: int = 14,
        max_lookback_period: int = 14,
        min_lookback_period: int = 7,
        # サイクル効率比(CER)のパラメーター
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        # アルファATRのパラメーター
        smoother_type: str = 'alma'  # 'alma'または'hyper'
    ):
        """
        コンストラクタ
        
        Args:
            max_chop_period: チョピネス期間の最大値
            min_chop_period: チョピネス期間の最小値
            max_atr_period: ATR期間の最大値
            min_atr_period: ATR期間の最小値
            max_stddev_period: 標準偏差期間の最大値
            min_stddev_period: 標準偏差期間の最小値
            max_lookback_period: 標準偏差の最小値を探す期間の最大値
            min_lookback_period: 標準偏差の最小値を探す期間の最小値
            cycle_detector_type: サイクル検出器の種類
                'dudi_dc' - 二重微分
                'hody_dc' - ホモダイン判別機（デフォルト）
                'phac_dc' - 位相累積
                'dudi_dce' - 拡張二重微分
                'hody_dce' - 拡張ホモダイン判別機
                'phac_dce' - 拡張位相累積
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            smoother_type: 平滑化アルゴリズムのタイプ（'alma'または'hyper'）
        """
        super().__init__(
            f"AlphaTrendIndex(CER, {max_chop_period}, {min_chop_period})"
        )
        
        self.max_chop_period = max_chop_period
        self.min_chop_period = min_chop_period
        self.max_atr_period = max_atr_period
        self.min_atr_period = min_atr_period
        self.max_stddev_period = max_stddev_period
        self.min_stddev_period = min_stddev_period
        self.max_lookback_period = max_lookback_period
        self.min_lookback_period = min_lookback_period
        self.smoother_type = smoother_type
        
        # サイクル効率比(CER)関連のパラメータ
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        
        # サイクル効率比(CER)のインスタンス化
        self.cycle_er = CycleEfficiencyRatio(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part
        )
        
        # アルファATRのインスタンス化
        self.alpha_atr_indicator = AlphaATR(
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            smoother_type=smoother_type
        )
        
        self._result = None
    
    def calculate(self, open_prices, high_prices, low_prices, close_prices) -> AlphaTrendIndexResult:
        """
        アルファトレンドインデックスを計算する

        Args:
            open_prices: 始値の配列
            high_prices: 高値の配列
            low_prices: 安値の配列
            close_prices: 終値の配列

        Returns:
            AlphaTrendIndexResult: 計算結果
        """
        
        # 配列をNumpy配列に変換
        o = np.asarray(open_prices, dtype=np.float64)
        h = np.asarray(high_prices, dtype=np.float64)
        l = np.asarray(low_prices, dtype=np.float64)
        c = np.asarray(close_prices, dtype=np.float64)
        
        # サイクル効率比(CER)を計算
        data = {
            'open': o,
            'high': h,
            'low': l,
            'close': c
        }
        df = pd.DataFrame(data)
        er = self.cycle_er.calculate(df)
        
        # 動的ATR期間の計算
        dynamic_atr_period = calculate_dynamic_period(
            er, self.max_atr_period, self.min_atr_period
        )
        
        # データフレーム作成（アルファATR計算用）
        df_data = pd.DataFrame({
            'high': h,
            'low': l,
            'close': c
        })
        
        # アルファATRの計算（外部効率比を使用）
        alpha_atr_values = self.alpha_atr_indicator.calculate(df_data, external_er=er)
        atr = self.alpha_atr_indicator.get_absolute_atr()  # 金額ベースの値を使用
        
        # 一括計算
        trend_index, dynamic_chop_period, dynamic_stddev_period, chop_index, stddev_factor = (
            calculate_alpha_trend_index_batch(
                h, l, c, er, 
                self.max_chop_period, self.min_chop_period,
                self.max_stddev_period, self.min_stddev_period,
                self.max_lookback_period, self.min_lookback_period,
                atr
            )
        )
        
        # True Rangeの計算（ここでは結果の完全性のために保持）
        tr = calculate_tr(h, l, c)
        
        # 結果オブジェクトを作成
        result = AlphaTrendIndexResult(
            values=trend_index,
            er=er,
            dynamic_chop_period=dynamic_chop_period,
            dynamic_atr_period=dynamic_atr_period,
            dynamic_stddev_period=dynamic_stddev_period,
            tr=tr,
            atr=atr,
            choppiness_index=chop_index,
            range_index=1.0 - chop_index,  # レンジインデックスはチョピネスの反転
            stddev_factor=stddev_factor
        )
        
        # 内部の結果変数に保存
        self._result = result
        self._values = trend_index
        
        return result

    def get_stddev_factor(self) -> np.ndarray:
        """
        標準偏差係数の値を取得する
        
        Returns:
            標準偏差係数の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.stddev_factor
    
    def get_choppiness_index(self) -> np.ndarray:
        """
        チョピネス指数の値を取得する
        
        Returns:
            チョピネス指数の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.choppiness_index
    
    def get_range_index(self) -> np.ndarray:
        """
        レンジインデックスの値を取得する
        
        Returns:
            レンジインデックスの配列
        """
        if self._result is None:
            return np.array([])
        return self._result.range_index

    def reset(self) -> None:
        """
        インジケーターの状態をリセットする
        """
        super().reset()
        self.cycle_er.reset()
        self.alpha_atr_indicator.reset()
        self._result = None 