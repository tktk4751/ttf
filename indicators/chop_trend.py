#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange
import traceback
import math

# Assuming these base classes/helpers exist in the same directory or are importable
try:
    from .indicator import Indicator
    from .atr import ATR
    from .ehlers_unified_dc import EhlersUnifiedDC
except ImportError:
    # Fallback for potential execution context issues
    print("Warning: Could not import from relative path. Assuming base classes are available.")
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class ATR:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return None
        def get_values(self): return np.array([])
        def get_dynamic_periods(self): return np.array([])
        def reset(self): pass
    class EhlersUnifiedDC:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return np.full(len(data), 10.0)
        def reset(self): pass


class ChopTrendResult(NamedTuple):
    """CHOPトレンド計算結果"""
    values: np.ndarray          # CHOPトレンドインデックスの値（0-1の範囲）
    trend_signals: np.ndarray   # 1=up, -1=down, 0=range
    current_trend: str          # 'up', 'down', 'range'
    current_trend_value: int    # 1, -1, 0
    dominant_cycle: np.ndarray  # ドミナントサイクル値（チョピネス期間として使用）
    dynamic_atr_period: np.ndarray   # 動的ATR期間
    choppiness_index: np.ndarray # Choppiness Index（元の値）
    range_index: np.ndarray     # Range Index（元の値）
    stddev_factor: np.ndarray   # 標準偏差係数 (固定期間で計算)
    tr: np.ndarray              # True Range
    atr: np.ndarray             # Average True Range
    fixed_threshold: float      # 固定しきい値
    trend_state: np.ndarray     # トレンド状態 (1=トレンド、0=レンジ、NaN=不明)


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
def calculate_stddev_factor(atr: np.ndarray) -> np.ndarray:
    """
    ATRの標準偏差係数を計算する (固定期間: 期間=14, ルックバック=14)

    Args:
        atr: ATR配列
    Returns:
        標準偏差係数
    """
    n = len(atr)
    fixed_period = 14
    fixed_lookback = 14
    stddev = np.zeros(n, dtype=np.float64)
    lowest_stddev = np.full(n, np.inf, dtype=np.float64)
    stddev_factor = np.ones(n, dtype=np.float64)

    for i in prange(n):
        if i >= fixed_period - 1:
            start_idx = i - fixed_period + 1
            atr_window = atr[start_idx:i+1]

            # PineScriptのSMAを使用した計算方法を維持
            stddev_a = np.mean(np.power(atr_window, 2))
            stddev_b = np.power(np.sum(atr_window), 2) / np.power(len(atr_window), 2)
            curr_stddev = np.sqrt(max(0.0, stddev_a - stddev_b))
            stddev[i] = curr_stddev

            # 最小標準偏差の更新（固定ルックバック期間内で）
            lowest_lookback_start = max(0, i - fixed_lookback + 1)
            # windowがlookback期間より短い場合も考慮
            valid_stddev_window = stddev[lowest_lookback_start : i + 1]
            # infを除外して最小値を計算 
            valid_stddev_window_finite = valid_stddev_window[np.isfinite(valid_stddev_window)]
            if len(valid_stddev_window_finite) > 0:
                 lowest_stddev[i] = np.min(valid_stddev_window_finite)
            else:
                 # 期間内に有効な標準偏差がない場合は現在の値を使用
                 lowest_stddev[i] = stddev[i] if np.isfinite(stddev[i]) else np.inf

            # 標準偏差係数の計算
            if stddev[i] > 0 and np.isfinite(lowest_stddev[i]):
                stddev_factor[i] = lowest_stddev[i] / stddev[i]
            elif i > 0:
                 stddev_factor[i] = stddev_factor[i-1] # 前の値を使用
            else:
                 stddev_factor[i] = 1.0 # 初期値

        elif i > 0:
            # データ不足の場合は前の値を使用
            stddev[i] = stddev[i-1]
            lowest_stddev[i] = lowest_stddev[i-1]
            stddev_factor[i] = stddev_factor[i-1]
        else:
            # 最初の要素はNaNまたはデフォルト値
             stddev[i] = np.nan
             lowest_stddev[i] = np.inf
             stddev_factor[i] = 1.0

    return stddev_factor


@njit(fastmath=True)
def calculate_chop_trend_index(
    chop: np.ndarray,
    stddev_factor: np.ndarray
) -> np.ndarray:
    """
    CHOPトレンドインデックスを計算する
    
    Args:
        chop: チョピネス指数の配列（0-100の範囲）
        stddev_factor: 標準偏差ファクターの配列
    
    Returns:
        CHOPトレンドインデックスの配列（0-1の範囲、1に近いほど強いトレンド）
    """
    # チョピネス指数と標準偏差係数を組み合わせたレンジインデックスを計算
    range_index = chop * stddev_factor
    
    # トレンド指数として常に反転し、0-1に正規化
    trend_index = 1.0 - (range_index / 100.0)
    
    # 値を0-1の範囲にクリップ
    trend_index = np.maximum(0.0, np.minimum(1.0, trend_index))
    
    return trend_index


@njit(fastmath=True, cache=True)
def calculate_wilder_smoothing_ct(values: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's Smoothing（CHOPトレンド用）
    
    Args:
        values: 値の配列
        period: 期間
    
    Returns:
        Wilder's Smoothing適用後の配列
    """
    length = len(values)
    result = np.full(length, np.nan)
    
    if period <= 0 or length < period:
        return result
    
    # 最初の値は単純移動平均で計算
    result[period-1] = np.mean(values[:period])
    
    # 2番目以降はWilder's Smoothingで計算
    # Smoothed(t) = ((period-1) * Smoothed(t-1) + Value(t)) / period
    for i in range(period, length):
        result[i] = ((period - 1) * result[i-1] + values[i]) / period
    
    return result


@njit(fastmath=True, cache=True)
def calculate_wma_numba_ct(prices: np.ndarray, period: int) -> np.ndarray:
    """
    重み付き移動平均 (WMA) を計算する (Numba JIT、NaN対応)
    
    Args:
        prices: 価格の配列 (np.float64 を想定)
        period: 期間
    
    Returns:
        WMA値の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)

    if period <= 0 or length < period:
        return result

    weights = np.arange(1.0, period + 1.0) # 重み 1, 2, ..., period
    weights_sum = period * (period + 1.0) / 2.0 # 合計 = n(n+1)/2

    if weights_sum < 1e-9:
        return result

    for i in range(period - 1, length):
        window_prices = prices[i - period + 1 : i + 1]

        # ウィンドウ内にNaNが含まれていないかチェック
        has_nan = False
        for j in range(period):
            if np.isnan(window_prices[j]):
                has_nan = True
                break
        
        if not has_nan:
            wma_value = 0.0
            for j in range(period):
                wma_value += window_prices[j] * weights[j]
            result[i] = wma_value / weights_sum

    return result


@njit(fastmath=True, cache=True)
def calculate_hma_numba_ct(prices: np.ndarray, period: int) -> np.ndarray:
    """
    ハル移動平均線 (HMA) を計算する (Numba JIT)
    
    Args:
        prices: 価格の配列 (np.float64 を想定)
        period: 期間 (2以上)
    
    Returns:
        HMA値の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)

    if period <= 1 or length == 0:
        return result

    period_half = int(period / 2)
    period_sqrt = int(math.sqrt(period))

    # HMA計算に必要な最小期間をチェック
    min_len_for_hma = period_sqrt - 1 + max(period_half, period)
    if length < min_len_for_hma:
         return result

    if period_half <= 0 or period_sqrt <= 0:
        return result

    # 中間WMAを計算
    wma_half = calculate_wma_numba_ct(prices, period_half)
    wma_full = calculate_wma_numba_ct(prices, period)

    # 差分系列を計算: 2 * WMA(period/2) - WMA(period)
    diff_wma = np.full(length, np.nan)
    valid_indices = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff_wma[valid_indices] = 2.0 * wma_half[valid_indices] - wma_full[valid_indices]
    
    if np.all(np.isnan(diff_wma)):
        return result

    # 最終的なHMAを計算: WMA(diff_wma, sqrt(period))
    hma_values = calculate_wma_numba_ct(diff_wma, period_sqrt)

    return hma_values


@njit(fastmath=True, cache=True)
def calculate_alma_numba_ct(prices: np.ndarray, period: int, offset: float = 0.85, sigma: float = 6.0) -> np.ndarray:
    """
    ALMAを計算する（Numba JIT、NaN対応改善）
    
    Args:
        prices: 価格の配列
        period: 期間
        offset: オフセット (0-1)。1に近いほど最新のデータを重視
        sigma: シグマ。大きいほど重みの差が大きくなる
    
    Returns:
        ALMA値の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)
    
    if period <= 0 or length == 0:
        return result
    
    # ウィンドウサイズが価格データより大きい場合は調整
    window_size = period
    
    # ウェイトの計算
    m = offset * (window_size - 1)
    s = window_size / sigma
    weights = np.zeros(window_size)
    weights_sum = 0.0
    
    for i in range(window_size):
        weight = np.exp(-((i - m) ** 2) / (2 * s * s))
        weights[i] = weight
        weights_sum += weight
    
    # 重みの正規化 (ゼロ除算防止)
    if weights_sum > 1e-9:
        weights = weights / weights_sum
    else:
        weights = np.full(window_size, 1.0 / window_size)
    
    # ALMAの計算
    for i in range(length):
        # ウィンドウに必要なデータがあるか確認
        window_start_idx = i - window_size + 1
        if window_start_idx < 0:
            continue

        # ウィンドウ内のNaNを確認
        window_prices = prices[window_start_idx : i + 1]
        if np.any(np.isnan(window_prices)):
            continue

        # ALMAの計算
        alma_value = 0.0
        for j in range(window_size):
            alma_value += window_prices[j] * weights[j]
        result[i] = alma_value
    
    return result


@njit(fastmath=True, cache=True)
def calculate_ema_numba_ct(prices: np.ndarray, period: int) -> np.ndarray:
    """
    EMA (Exponential Moving Average) を計算する（Numba JIT）
    
    Args:
        prices: 価格の配列
        period: 期間
    
    Returns:
        EMA値の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)
    
    if period <= 0 or length == 0:
        return result
    
    # EMAの平滑化係数
    alpha = 2.0 / (period + 1.0)
    
    # 最初の値を設定（最初の有効な値を使用）
    first_valid_idx = -1
    for i in range(length):
        if not np.isnan(prices[i]):
            result[i] = prices[i]
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return result  # 全てNaNの場合
    
    # EMAの計算
    for i in range(first_valid_idx + 1, length):
        if not np.isnan(prices[i]):
            if not np.isnan(result[i-1]):
                result[i] = alpha * prices[i] + (1.0 - alpha) * result[i-1]
            else:
                result[i] = prices[i]
    
    return result


@njit(fastmath=True, cache=True)
def calculate_zlema_numba_ct(prices: np.ndarray, period: int) -> np.ndarray:
    """
    ZLEMA (Zero Lag EMA) を計算する（Numba JIT）
    
    Args:
        prices: 価格の配列
        period: 期間
    
    Returns:
        ZLEMA値の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)
    
    if period <= 0 or length == 0:
        return result
    
    # ラグ期間の計算
    lag = int((period - 1) / 2)
    
    if lag >= length:
        return result
    
    # ラグ補正された価格系列を作成
    adjusted_prices = np.full(length, np.nan)
    
    for i in range(lag, length):
        if not np.isnan(prices[i]) and not np.isnan(prices[i - lag]):
            # adjusted_src = src + (src - src[lag])
            adjusted_prices[i] = prices[i] + (prices[i] - prices[i - lag])
    
    # 調整された価格系列にEMAを適用
    result = calculate_ema_numba_ct(adjusted_prices, period)
    
    return result


@njit(fastmath=True, cache=True)
def calculate_chop_trend_with_smoothing(
    trend_index: np.ndarray,
    smoothing_method: int,
    smoothing_period: int
) -> np.ndarray:
    """
    指定されたスムージング方法でCHOPトレンドインデックスを平滑化する
    
    Args:
        trend_index: CHOPトレンドインデックスの配列
        smoothing_method: スムージング方法 (0: なし, 1: Wilder's, 2: HMA, 3: ALMA, 4: ZLEMA)
        smoothing_period: スムージング期間
    
    Returns:
        平滑化されたCHOPトレンドインデックスの配列
    """
    if smoothing_method == 0:  # スムージングなし
        return trend_index
    elif smoothing_method == 1:  # Wilder's Smoothing
        return calculate_wilder_smoothing_ct(trend_index, smoothing_period)
    elif smoothing_method == 2:  # HMA
        return calculate_hma_numba_ct(trend_index, smoothing_period)
    elif smoothing_method == 3:  # ALMA
        return calculate_alma_numba_ct(trend_index, smoothing_period, 0.85, 6.0)  # デフォルトパラメータ
    elif smoothing_method == 4:  # ZLEMA
        return calculate_zlema_numba_ct(trend_index, smoothing_period)
    else:
        # デフォルトはスムージングなし
        return trend_index


@njit(fastmath=True)
def calculate_chop_trend_index_batch(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    dominant_cycle: np.ndarray,
    atr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CHOPトレンドインデックスを一括計算する

    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        dominant_cycle: ドミナントサイクル値の配列（チョピネス期間として使用）
        atr: ATR の配列

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            (CHOPトレンドインデックス, チョピネス指数, 標準偏差係数)
    """
    # True Rangeの計算
    tr = calculate_tr(high, low, close)

    # チョピネスインデックスの計算 (DC値を期間として使用)
    # DC値が0や負にならないように最小値を2にクリップ
    chop_period = np.maximum(2, dominant_cycle).astype(np.int32)
    chop_index = calculate_choppiness_index(high, low, close, chop_period, tr)

    # 標準偏差係数の計算 (固定期間を使用)
    stddev_factor = calculate_stddev_factor(atr)

    # トレンドインデックスの計算
    trend_index = calculate_chop_trend_index(chop_index, stddev_factor)

    return trend_index, chop_index, stddev_factor


@jit(nopython=True, cache=True)
def calculate_trend_signals_with_range(values: np.ndarray, slope_index: int, range_threshold: float = 0.005) -> np.ndarray:
    """
    トレンド信号を計算する（range状態対応版）(Numba JIT)
    
    Args:
        values: インジケーター値の配列
        slope_index: スロープ判定期間
        range_threshold: range判定の閾値（相対的変化率）
    
    Returns:
        trend_signals: 1=up, -1=down, 0=range のNumPy配列
    """
    length = len(values)
    trend_signals = np.zeros(length, dtype=np.int8)
    
    # 統計的閾値計算用のウィンドウサイズ（固定）
    stats_window = 21
    
    for i in range(slope_index, length):
        if not np.isnan(values[i]) and not np.isnan(values[i - slope_index]):
            current = values[i]
            previous = values[i - slope_index]
            
            # 基本的な変化量
            change = current - previous
            
            # 相対的変化率の計算
            base_value = max(abs(current), abs(previous), 1e-10)  # ゼロ除算防止
            relative_change = abs(change) / base_value
            
            # 統計的閾値の計算（過去の変動の標準偏差）
            start_idx = max(slope_index, i - stats_window + 1)
            if start_idx < i - slope_index:
                # 過去の変化率を計算
                historical_changes = np.zeros(i - start_idx)
                for j in range(start_idx, i):
                    if not np.isnan(values[j]) and not np.isnan(values[j - slope_index]):
                        hist_current = values[j]
                        hist_previous = values[j - slope_index]
                        hist_base = max(abs(hist_current), abs(hist_previous), 1e-10)
                        historical_changes[j - start_idx] = abs(hist_current - hist_previous) / hist_base
                
                # 標準偏差ベースの閾値のみを使用
                if len(historical_changes) > 0:
                    # 標準偏差ベースの閾値
                    std_threshold = np.std(historical_changes) * 0.5  # 0.5倍の標準偏差
                    
                    # 最終的なrange閾値は、固定閾値と標準偏差閾値の大きい方
                    effective_threshold = max(range_threshold, std_threshold)
                else:
                    effective_threshold = range_threshold
            else:
                effective_threshold = range_threshold
            
            # トレンド判定
            if relative_change < effective_threshold:
                # 変化が小さすぎる場合はrange
                trend_signals[i] = 0  # range
            elif change > 0:
                # 上昇トレンド
                trend_signals[i] = 1  # up
            else:
                # 下降トレンド
                trend_signals[i] = -1  # down
    
    return trend_signals


@jit(nopython=True, cache=True)
def calculate_current_trend_with_range(trend_signals: np.ndarray) -> tuple:
    """
    現在のトレンド状態を計算する（range対応版）(Numba JIT)
    
    Args:
        trend_signals: トレンド信号配列 (1=up, -1=down, 0=range)
    
    Returns:
        tuple: (current_trend_index, current_trend_value)
               current_trend_index: 0=range, 1=up, 2=down (trend_names用のインデックス)
               current_trend_value: 0=range, 1=up, -1=down (実際のトレンド値)
    """
    length = len(trend_signals)
    if length == 0:
        return 0, 0  # range
    
    # 最新の値を取得
    latest_trend = trend_signals[-1]
    
    if latest_trend == 1:  # up
        return 1, 1   # up
    elif latest_trend == -1:  # down
        return 2, -1   # down
    else:  # range
        return 0, 0  # range


@njit(fastmath=True)
def calculate_fixed_threshold_trend_state(
    trend_index: np.ndarray,
    fixed_threshold: float
) -> np.ndarray:
    """
    固定しきい値に基づいてトレンド状態を計算する
    
    Args:
        trend_index: CHOPトレンドインデックスの配列
        fixed_threshold: 固定しきい値
    
    Returns:
        トレンド状態の配列（1=トレンド、0=レンジ、NaN=不明）
    """
    length = len(trend_index)
    trend_state = np.full(length, np.nan)
    
    for i in range(length):
        if np.isnan(trend_index[i]):
            continue
        trend_state[i] = 1.0 if trend_index[i] >= fixed_threshold else 0.0
    
    return trend_state


class ChopTrend(Indicator):
    """
    CHOPトレンドインジケーター
    
    EhlersUnifiedDCを使用してチョピネス期間を動的に決定し、
    ATRと固定期間の標準偏差係数を組み合わせてトレンド/レンジを検出する指標。
    XTrendIndexをベースに、CATRをATRに変更し、スムージング機能を追加。
    
    特徴:
    - EhlersUnifiedDCを使用してチョピネス期間を動的に決定
    - ATRを使用したボラティリティ測定（HMA、ALMA、ZLEMAスムージング対応）
    - チョピネスインデックスと固定期間の標準偏差係数を組み合わせて正規化したトレンド指標
    - 市場状態に応じてチョピネス期間とATR期間が自動調整される
    - 0-1の範囲で表示（1に近いほど強いトレンド、0に近いほど強いレンジ）
    - 固定しきい値によるトレンド/レンジ状態の判定
    - トレンド方向判定機能（up, down, range）
    - CHOPトレンドインデックス自体のスムージング機能（Wilder's、HMA、ALMA、ZLEMA対応）
    """
    
    def __init__(
        self,
        # EhlersUnifiedDC パラメータ
        detector_type: str = 'cycle_period2',
        cycle_part: float = 1.0,
        max_cycle: int = 120,
        min_cycle: int = 21,
        max_output: int = 89,
        min_output: int = 13,
        src_type: str = 'hlc3',
        lp_period: int = 21,
        hp_period: int = 120,

        # ATR パラメータ
        atr_period: int = 13,
        atr_smoothing_method: str = 'alma',
        use_dynamic_atr_period: bool = True,
        
        # CHOPトレンドスムージングパラメータ  
        chop_smoothing_method: str = 'none',
        chop_smoothing_period: int = 5,
        
        # トレンド判定パラメータ
        slope_index: int = 1,
        range_threshold: float = 0.005,

        # 固定しきい値のパラメータ
        fixed_threshold: float = 0.65
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: EhlersUnifiedDCで使用する検出器タイプ (デフォルト: 'cycle_period2')
            cycle_part: DCのサイクル部分の倍率 (デフォルト: 1.0)
            max_cycle: DCの最大サイクル期間 (デフォルト: 120)
            min_cycle: DCの最小サイクル期間 (デフォルト: 21)
            max_output: DCの最大出力値 (デフォルト: 89)
            min_output: DCの最小出力値 (デフォルト: 13)
            src_type: DC計算に使用する価格ソース ('close', 'hlc3', etc.) (デフォルト: 'hlc3')
            lp_period: 拡張DC用のローパスフィルター期間 (デフォルト: 21)
            hp_period: 拡張DC用のハイパスフィルター期間 (デフォルト: 120)
            atr_period: ATRの期間 (デフォルト: 13)
            atr_smoothing_method: ATRで使用する平滑化アルゴリズム ('alma', 'hma', 'zlema', 'wilder', 'none') (デフォルト: 'alma')
            use_dynamic_atr_period: 動的ATR期間を使用するかどうか (デフォルト: True)
            chop_smoothing_method: CHOPトレンドインデックスのスムージング方法 ('none', 'wilder', 'hma', 'alma', 'zlema') (デフォルト: 'none')
            chop_smoothing_period: CHOPトレンドスムージング期間 (デフォルト: 5)
            slope_index: トレンド判定期間 (デフォルト: 1)
            range_threshold: range判定の基本閾値 (デフォルト: 0.005)
            fixed_threshold: 固定しきい値 (デフォルト: 0.65)
        """
        # CHOPスムージング方法の検証と変換
        chop_smoothing_methods = {'none': 0, 'wilder': 1, 'hma': 2, 'alma': 3, 'zlema': 4}
        if chop_smoothing_method.lower() not in chop_smoothing_methods:
            raise ValueError(f"サポートされていないCHOPスムージング方法: {chop_smoothing_method}. 使用可能: {list(chop_smoothing_methods.keys())}")
        
        self.chop_smoothing_method_str = chop_smoothing_method.lower()
        self.chop_smoothing_method_int = chop_smoothing_methods[self.chop_smoothing_method_str]
        self.chop_smoothing_period = chop_smoothing_period
        
        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_atr_period else ""
        atr_str = f"_atr({atr_smoothing_method})" if atr_smoothing_method != 'none' else ""
        chop_smooth_str = f"_chop_smooth({chop_smoothing_method})" if chop_smoothing_method != 'none' else ""
        super().__init__(
            f"ChopTrend({max_output},{min_output}{dynamic_str}{atr_str}{chop_smooth_str},slope={slope_index},th={fixed_threshold})"
        )
        
        # ドミナントサイクル検出器 (EhlersUnifiedDC) のパラメータ
        self.detector_type = detector_type
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.src_type = src_type
        self.lp_period = lp_period
        self.hp_period = hp_period

        # ATRパラメータ
        self.atr_period = atr_period
        self.atr_smoothing_method = atr_smoothing_method
        self.use_dynamic_atr_period = use_dynamic_atr_period
        
        # トレンド判定パラメータ
        self.slope_index = slope_index
        self.range_threshold = range_threshold

        # 固定しきい値のパラメータ
        self.fixed_threshold = fixed_threshold

        # ドミナントサイクル検出器の初期化 (EhlersUnifiedDCを使用)
        self.dc_detector = EhlersUnifiedDC(
            detector_type=self.detector_type,
            cycle_part=self.cycle_part,
            max_cycle=self.max_cycle,
            min_cycle=self.min_cycle,
            max_output=self.max_output,
            min_output=self.min_output,
            src_type=self.src_type,
            lp_period=self.lp_period,
            hp_period=self.hp_period
        )

        # ATRのインスタンス化
        self.atr_indicator = ATR(
             period=self.atr_period,
             smoothing_method=self.atr_smoothing_method,
             use_dynamic_period=self.use_dynamic_atr_period,
             cycle_part=self.cycle_part,
             detector_type=self.detector_type,
             max_cycle=self.max_cycle,
             min_cycle=self.min_cycle,
             max_output=self.max_output,
             min_output=self.min_output,
             slope_index=self.slope_index,
             range_threshold=self.range_threshold,
             lp_period=self.lp_period,
             hp_period=self.hp_period
        )

        self._cache = {}
        self._result: Optional[ChopTrendResult] = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrameの場合は形状と端点でハッシュを計算
                shape_tuple = data.shape
                first_row_tuple = tuple(data.iloc[0]) if len(data) > 0 else ()
                last_row_tuple = tuple(data.iloc[-1]) if len(data) > 0 else ()
                data_repr_tuple = (shape_tuple, first_row_tuple, last_row_tuple)
                data_hash_val = hash(data_repr_tuple)
            elif isinstance(data, np.ndarray):
                # NumPy配列の場合はバイト表現でハッシュ
                data_hash_val = hash(data.tobytes())
            else:
                # その他のデータ型は文字列表現でハッシュ化
                data_hash_val = hash(str(data))

        except Exception as e:
            self.logger.warning(f"データハッシュ計算中にエラー: {e}. データ全体の文字列表現を使用します。", exc_info=True)
            data_hash_val = hash(str(data)) # fallback

        # パラメータ文字列の作成
        param_str = (
            f"{self.detector_type}_{self.cycle_part}_{self.max_cycle}_{self.min_cycle}_"
            f"{self.max_output}_{self.min_output}_{self.src_type}_{self.lp_period}_{self.hp_period}_"
            f"{self.atr_period}_{self.atr_smoothing_method}_{self.use_dynamic_atr_period}_"
            f"{self.chop_smoothing_method_str}_{self.chop_smoothing_period}_"
            f"{self.slope_index}_{self.range_threshold:.3f}_{self.fixed_threshold}"
        )
        return f"{data_hash_val}_{hash(param_str)}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ChopTrendResult:
        """
        CHOPトレンドを計算する

        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合は'open', 'high', 'low', 'close'カラムが必要

        Returns:
            ChopTrendResult: 計算結果
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
            self.logger.warning("入力データが空です。空の結果を返します。")
            empty_result = ChopTrendResult(
                values=np.array([]),
                trend_signals=np.array([], dtype=np.int8),
                current_trend='range',
                current_trend_value=0,
                dominant_cycle=np.array([]),
                dynamic_atr_period=np.array([]),
                choppiness_index=np.array([]),
                range_index=np.array([]),
                stddev_factor=np.array([]),
                tr=np.array([]),
                atr=np.array([]),
                fixed_threshold=self.fixed_threshold,
                trend_state=np.array([])
            )
            return empty_result
            
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._cache and self._result is not None:
                # データ長が一致するか確認
                if len(self._result.values) == current_data_len:
                    return ChopTrendResult(
                        values=self._result.values.copy(),
                        trend_signals=self._result.trend_signals.copy(),
                        current_trend=self._result.current_trend,
                        current_trend_value=self._result.current_trend_value,
                        dominant_cycle=self._result.dominant_cycle.copy(),
                        dynamic_atr_period=self._result.dynamic_atr_period.copy(),
                        choppiness_index=self._result.choppiness_index.copy(),
                        range_index=self._result.range_index.copy(),
                        stddev_factor=self._result.stddev_factor.copy(),
                        tr=self._result.tr.copy(),
                        atr=self._result.atr.copy(),
                        fixed_threshold=self._result.fixed_threshold,
                        trend_state=self._result.trend_state.copy()
                    )
                else:
                    self.logger.debug(f"キャッシュのデータ長が異なるため再計算します。")
                    # キャッシュを無効化
                    del self._cache[data_hash]
                    self._result = None

            # データ検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
                # openがない場合はcloseで代用
                h = np.asarray(data['high'].values, dtype=np.float64)
                l = np.asarray(data['low'].values, dtype=np.float64)
                c = np.asarray(data['close'].values, dtype=np.float64)
                # DataFrameを渡す必要がある場合
                df_data = data

            elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 4:
                h = np.asarray(data[:, 1], dtype=np.float64)
                l = np.asarray(data[:, 2], dtype=np.float64)
                c = np.asarray(data[:, 3], dtype=np.float64)
                 # DataFrameが必要なインジケータ用に一時的に作成
                df_data = pd.DataFrame({'open': data[:, 0], 'high': h, 'low': l, 'close': c})
            else:
                raise ValueError("データはPandas DataFrameまたは4列以上のNumPy配列である必要があります")

            # ドミナントサイクルの計算 (EhlersUnifiedDCを使用)
            # このDC値がチョピネス期間として使われる
            dominant_cycle = self.dc_detector.calculate(df_data)

            # ATRの計算
            atr_result = self.atr_indicator.calculate(df_data)
            atr = atr_result.values  # ATR値
            dynamic_atr_period = self.atr_indicator.get_dynamic_periods() if self.use_dynamic_atr_period else np.full_like(atr, self.atr_period)

            # 一括計算 (CHOP Trend Index, Choppiness, StdDev Factor)
            trend_index, chop_index, stddev_factor = calculate_chop_trend_index_batch(
                h, l, c,
                dominant_cycle, # チョピネス期間としてDC値を使用
                atr             # ATR
            )

            # CHOPトレンドインデックスのスムージング適用
            if self.chop_smoothing_method_str != 'none':
                trend_index = calculate_chop_trend_with_smoothing(
                    trend_index, 
                    self.chop_smoothing_method_int, 
                    self.chop_smoothing_period
                )

            # 値を0-1の範囲に正規化（0以下は0、1以上は1にクリップ）
            trend_index = np.clip(trend_index, 0.0, 1.0)

            # True Rangeの計算 (結果オブジェクト用)
            tr = calculate_tr(h, l, c)

            # トレンド状態の計算（固定しきい値使用）
            trend_state = calculate_fixed_threshold_trend_state(
                trend_index, self.fixed_threshold
            )

            # トレンド判定
            trend_signals = calculate_trend_signals_with_range(trend_index, self.slope_index, self.range_threshold)
            trend_index_val, trend_value = calculate_current_trend_with_range(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index_val]

            # 結果オブジェクトを作成
            result = ChopTrendResult(
                values=trend_index,
                trend_signals=trend_signals,
                current_trend=current_trend,
                current_trend_value=trend_value,
                dominant_cycle=dominant_cycle, # チョピネスに使ったDC値
                dynamic_atr_period=dynamic_atr_period, # ATRが計算した期間
                choppiness_index=chop_index,
                range_index=100.0 - chop_index, # レンジインデックス (0-100)
                stddev_factor=stddev_factor,
                tr=tr,
                atr=atr, # ATR値
                fixed_threshold=self.fixed_threshold,
                trend_state=trend_state
            )

            self._result = result
            self._cache[data_hash] = self._result
            self._values = trend_index # Indicatorクラスの標準出力

            return ChopTrendResult(
                values=result.values.copy(),
                trend_signals=result.trend_signals.copy(),
                current_trend=result.current_trend,
                current_trend_value=result.current_trend_value,
                dominant_cycle=result.dominant_cycle.copy(),
                dynamic_atr_period=result.dynamic_atr_period.copy(),
                choppiness_index=result.choppiness_index.copy(),
                range_index=result.range_index.copy(),
                stddev_factor=result.stddev_factor.copy(),
                tr=result.tr.copy(),
                atr=result.atr.copy(),
                fixed_threshold=result.fixed_threshold,
                trend_state=result.trend_state.copy()
            )

        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ChopTrend計算中にエラー: {error_msg}\n{stack_trace}")
            # エラー時は空の結果を返す
            self._result = None
            self._values = np.full(current_data_len, np.nan)
            if data_hash in self._cache:
                del self._cache[data_hash]
            
            error_result = ChopTrendResult(
                 values=np.full(current_data_len, np.nan),
                 trend_signals=np.zeros(current_data_len, dtype=np.int8),
                 current_trend='range',
                 current_trend_value=0,
                 dominant_cycle=np.full(current_data_len, np.nan),
                 dynamic_atr_period=np.full(current_data_len, np.nan),
                 choppiness_index=np.full(current_data_len, np.nan),
                 range_index=np.full(current_data_len, np.nan),
                 stddev_factor=np.full(current_data_len, np.nan),
                 tr=np.full(current_data_len, np.nan),
                 atr=np.full(current_data_len, np.nan),
                 fixed_threshold=self.fixed_threshold,
                 trend_state=np.full(current_data_len, np.nan)
            )
            return error_result

    # --- Getter Methods ---
    def get_values(self) -> Optional[np.ndarray]:
        """CHOPトレンド値のみを取得する（後方互換性のため）"""
        if self._result is not None:
            return self._result.values.copy()
        return None

    def get_trend_signals(self) -> Optional[np.ndarray]:
        """トレンド信号を取得する"""
        if self._result is not None:
            return self._result.trend_signals.copy()
        return None

    def get_current_trend(self) -> str:
        """現在のトレンド状態を取得する"""
        if self._result is not None:
            return self._result.current_trend
        return 'range'

    def get_current_trend_value(self) -> int:
        """現在のトレンド値を取得する"""
        if self._result is not None:
            return self._result.current_trend_value
        return 0

    def get_result(self) -> Optional[ChopTrendResult]:
        """計算結果全体を取得する"""
        return self._result

    def get_dominant_cycle(self) -> np.ndarray:
        """チョピネス期間の計算に使用したドミナントサイクル値を取得する"""
        if self._result is None: return np.array([])
        return self._result.dominant_cycle.copy()

    def get_dynamic_atr_period(self) -> np.ndarray:
        """ATRが計算した動的期間を取得する"""
        if self._result is None: return np.array([])
        return self._result.dynamic_atr_period.copy()

    def get_stddev_factor(self) -> np.ndarray:
        """標準偏差係数の値を取得する"""
        if self._result is None: return np.array([])
        return self._result.stddev_factor.copy()

    def get_choppiness_index(self) -> np.ndarray:
        """チョピネス指数の値を取得する"""
        if self._result is None: return np.array([])
        return self._result.choppiness_index.copy()

    def get_range_index(self) -> np.ndarray:
        """レンジインデックスの値を取得する (0-100)"""
        if self._result is None: return np.array([])
        return self._result.range_index.copy()

    def get_true_range(self) -> np.ndarray:
        """True Rangeの値を取得する"""
        if self._result is None: return np.array([])
        return self._result.tr.copy()

    def get_atr(self) -> np.ndarray:
        """ATR値を取得する"""
        if self._result is None: return np.array([])
        return self._result.atr.copy()

    def get_fixed_threshold(self) -> float:
        """固定しきい値を取得する"""
        return self.fixed_threshold

    def get_trend_state(self) -> np.ndarray:
        """トレンド状態を取得する（1=トレンド、0=レンジ、NaN=不明）"""
        if self._result is None: return np.array([])
        return self._result.trend_state.copy()

    def get_chop_smoothing_method(self) -> str:
        """CHOPトレンドスムージング方法を取得する"""
        return self.chop_smoothing_method_str

    def get_chop_smoothing_period(self) -> int:
        """CHOPトレンドスムージング期間を取得する"""
        return self.chop_smoothing_period

    def reset(self) -> None:
        """インジケーターの状態をリセットする"""
        super().reset()
        self.dc_detector.reset()
        self.atr_indicator.reset()
        self._result = None
        self._cache = {}
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。") 