#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
import pandas as pd
from numba import njit
import traceback

from .indicator import Indicator
from .price_source import PriceSource

# 条件付きインポート（動的期間用）
try:
    from .cycle.ehlers_unified_dc import EhlersUnifiedDC
    EHLERS_UNIFIED_DC_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC
        EHLERS_UNIFIED_DC_AVAILABLE = True
    except ImportError:
        EhlersUnifiedDC = None
        EHLERS_UNIFIED_DC_AVAILABLE = False

# 統合カルマンフィルターインポート
try:
    from .kalman.unified_kalman import UnifiedKalman
    UNIFIED_KALMAN_AVAILABLE = True
except ImportError:
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.kalman.unified_kalman import UnifiedKalman
        UNIFIED_KALMAN_AVAILABLE = True
    except ImportError:
        UnifiedKalman = None
        UNIFIED_KALMAN_AVAILABLE = False

# HyperTrendIndexインポート
try:
    from .hyper_trend_index import HyperTrendIndex
    HYPER_TREND_INDEX_AVAILABLE = True
except ImportError:
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.hyper_trend_index import HyperTrendIndex
        HYPER_TREND_INDEX_AVAILABLE = True
    except ImportError:
        HyperTrendIndex = None
        HYPER_TREND_INDEX_AVAILABLE = False

# Ultimate Smootherインポート
try:
    from .smoother.ultimate_smoother import UltimateSmoother
    ULTIMATE_SMOOTHER_AVAILABLE = True
except ImportError:
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.smoother.ultimate_smoother import UltimateSmoother
        ULTIMATE_SMOOTHER_AVAILABLE = True
    except ImportError:
        UltimateSmoother = None
        ULTIMATE_SMOOTHER_AVAILABLE = False


@dataclass
class HyperTripleFRAMAResult:
    """ハイパートリプルFRAMAの計算結果"""
    frama_values: np.ndarray           # 通常のFRAMA値（alpha_multiplier1）
    second_frama_values: np.ndarray    # 2本目のFRAMA値（alpha_multiplier2）
    third_frama_values: np.ndarray     # 3本目のFRAMA値（alpha_multiplier3）
    fractal_dimension: np.ndarray      # フラクタル次元
    alpha: np.ndarray                  # 通常のアルファ値
    second_alpha: np.ndarray           # 2本目のアルファ値
    third_alpha: np.ndarray            # 3本目のアルファ値
    filtered_prices: np.ndarray        # 最終的な平滑化後の価格
    smoothing_applied: str             # 適用された平滑化方法


@njit(fastmath=True, cache=True)
def calculate_dynamic_fc_sc_period_vec(indicator_values: np.ndarray, fc_min: float, fc_max: float, sc_min: float, sc_max: float, period_min: int, period_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    インジケーター値（HyperERまたはHyperADX）に基づいて動的にFC/SC/ピリオドパラメーターを計算する（ベクトル化版）
    
    Args:
        indicator_values: インジケーター値の配列（0-1の範囲、HyperERまたはHyperADX）
        fc_min: FC最小値（インジケーター値高い時に使用）
        fc_max: FC最大値（インジケーター値低い時に使用）  
        sc_min: SC最小値（インジケーター値高い時に使用）
        sc_max: SC最大値（インジケーター値低い時に使用）
        period_min: ピリオド最小値（インジケーター値高い時に使用）
        period_max: ピリオド最大値（インジケーター値低い時に使用）
    
    Returns:
        動的FC値配列、動的SC値配列、動的ピリオド値配列のタプル
        
    Note:
        - HyperER: 高い値=効率的（高速レスポンス）、低い値=非効率的（低速レスポンス）
        - HyperADX: 高い値=強いトレンド（高速レスポンス）、低い値=レンジ相場（低速レスポンス）
    """
    length = len(indicator_values)
    dynamic_fc = np.zeros(length, dtype=np.float64)
    dynamic_sc = np.zeros(length, dtype=np.float64)
    dynamic_periods = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        indicator_value = indicator_values[i] if not np.isnan(indicator_values[i]) else 0.0
        
        # インジケーター値高い（効率的/トレンド）→ FC小さく（高速）、SC小さく（高速）、ピリオド小さく（高速）
        # インジケーター値低い（非効率/レンジ）→ FC大きく（低速）、SC大きく（低速）、ピリオド大きく（低速）
        dynamic_fc[i] = fc_min + (1.0 - indicator_value) * (fc_max - fc_min)
        dynamic_sc[i] = sc_min + (1.0 - indicator_value) * (sc_max - sc_min)
        
        # ピリオドは偶数である必要があるため、偶数に調整
        raw_period = period_min + (1.0 - indicator_value) * (period_max - period_min)
        adjusted_period = int(raw_period)
        if adjusted_period % 2 != 0:
            adjusted_period += 1  # 奇数の場合は偶数にする
        
        # 範囲チェック（偶数範囲に調整）
        if adjusted_period < period_min:
            adjusted_period = period_min if period_min % 2 == 0 else period_min + 1
        elif adjusted_period > period_max:
            adjusted_period = period_max if period_max % 2 == 0 else period_max - 1
        
        dynamic_periods[i] = float(adjusted_period)
    
    return dynamic_fc, dynamic_sc, dynamic_periods


@njit(fastmath=True, cache=True)
def calculate_hyper_triple_frama_core(
    price: np.ndarray, 
    high: np.ndarray, 
    low: np.ndarray, 
    n: int, 
    fc: int, 
    sc: int, 
    alpha_multiplier1: float = 1.0,
    alpha_multiplier2: float = 0.5, 
    alpha_multiplier3: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ハイパートリプルFRAMA（3本のFRAMAを同時計算）
    
    Args:
        price: 価格データ
        high: 高値データ
        low: 安値データ
        n: 期間（偶数である必要がある）
        fc: Fast Constant
        sc: Slow Constant
        alpha_multiplier1: 1本目のアルファ調整係数（デフォルト: 1.0）
        alpha_multiplier2: 2本目のアルファ調整係数（デフォルト: 0.5）
        alpha_multiplier3: 3本目のアルファ調整係数（デフォルト: 0.1）
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        1本目FRAMA値、2本目FRAMA値、3本目FRAMA値、フラクタル次元、1本目アルファ値、2本目アルファ値、3本目アルファ値
    """
    length = len(price)
    
    # 結果配列を初期化
    frama1 = np.zeros(length, dtype=np.float64)
    frama2 = np.zeros(length, dtype=np.float64)
    frama3 = np.zeros(length, dtype=np.float64)
    dimension = np.zeros(length, dtype=np.float64)
    alpha1 = np.zeros(length, dtype=np.float64)
    alpha2 = np.zeros(length, dtype=np.float64)
    alpha3 = np.zeros(length, dtype=np.float64)
    
    # 初期値設定
    for i in range(length):
        frama1[i] = np.nan
        frama2[i] = np.nan
        frama3[i] = np.nan
        dimension[i] = np.nan
        alpha1[i] = np.nan
        alpha2[i] = np.nan
        alpha3[i] = np.nan
    
    # 計算に必要な最小期間
    min_period = n
    
    # w = log(2/(SC+1))
    w = np.log(2.0 / (sc + 1))
    
    # 最初の期間は価格をそのまま使用
    for i in range(min(min_period, length)):
        if not np.isnan(price[i]):
            frama1[i] = price[i]
            frama2[i] = price[i]
            frama3[i] = price[i]
            alpha1[i] = 1.0
            alpha2[i] = alpha_multiplier2
            alpha3[i] = alpha_multiplier3
    
    # メインループ
    for i in range(min_period, length):
        if np.isnan(price[i]):
            frama1[i] = frama1[i-1] if i > 0 else np.nan
            frama2[i] = frama2[i-1] if i > 0 else np.nan
            frama3[i] = frama3[i-1] if i > 0 else np.nan
            continue
        
        # len1 = len/2
        len1 = n // 2
        
        # H1 = highest(high,len1)
        # L1 = lowest(low,len1)
        h1 = -np.inf
        l1 = np.inf
        for j in range(len1):
            if i - j >= 0:
                if high[i - j] > h1:
                    h1 = high[i - j]
                if low[i - j] < l1:
                    l1 = low[i - j]
        
        # N1 = (H1-L1)/len1
        n1 = (h1 - l1) / len1
        
        # H2 = highest(high,len)[len1]
        # L2 = lowest(low,len)[len1]
        h2 = -np.inf
        l2 = np.inf
        for j in range(len1, n):
            if i - j >= 0:
                if high[i - j] > h2:
                    h2 = high[i - j]
                if low[i - j] < l2:
                    l2 = low[i - j]
        
        # N2 = (H2-L2)/len1
        n2 = (h2 - l2) / len1
        
        # H3 = highest(high,len)
        # L3 = lowest(low,len)
        h3 = -np.inf
        l3 = np.inf
        for j in range(n):
            if i - j >= 0:
                if high[i - j] > h3:
                    h3 = high[i - j]
                if low[i - j] < l3:
                    l3 = low[i - j]
        
        # N3 = (H3-L3)/len
        n3 = (h3 - l3) / n
        
        # dimen1 = (log(N1+N2)-log(N3))/log(2)
        # dimen = iff(N1>0 and N2>0 and N3>0,dimen1,nz(dimen1[1]))
        if n1 > 0 and n2 > 0 and n3 > 0:
            dimen1 = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
            dimen = dimen1
        else:
            dimen = dimension[i-1] if i > 0 else 1.0
        
        dimension[i] = dimen
        
        # alpha1 = exp(w*(dimen-1))
        alpha_base = np.exp(w * (dimen - 1.0))
        
        # oldalpha = alpha1>1?1:(alpha1<0.01?0.01:alpha1)
        if alpha_base > 1.0:
            oldalpha = 1.0
        elif alpha_base < 0.01:
            oldalpha = 0.01
        else:
            oldalpha = alpha_base
        
        # oldN = (2-oldalpha)/oldalpha
        oldN = (2.0 - oldalpha) / oldalpha
        
        # N = (((SC-FC)*(oldN-1))/(SC-1))+FC
        N = (((sc - fc) * (oldN - 1.0)) / (sc - 1.0)) + fc
        
        # alpha_ = 2/(N+1)
        alpha_ = 2.0 / (N + 1.0)
        
        # alpha = alpha_<2/(SC+1)?2/(SC+1):(alpha_>1?1:alpha_)
        min_alpha = 2.0 / (sc + 1.0)
        if alpha_ < min_alpha:
            final_alpha = min_alpha
        elif alpha_ > 1.0:
            final_alpha = 1.0
        else:
            final_alpha = alpha_
        
        # 各ラインのアルファ値を計算
        alpha1_val = final_alpha * alpha_multiplier1
        alpha2_val = final_alpha * alpha_multiplier2
        alpha3_val = final_alpha * alpha_multiplier3
        
        alpha1[i] = alpha1_val
        alpha2[i] = alpha2_val
        alpha3[i] = alpha3_val
        
        # 各FRAMAラインの計算
        if i == min_period:
            frama1[i] = price[i]
            frama2[i] = price[i]
            frama3[i] = price[i]
        else:
            frama1[i] = (1.0 - alpha1_val) * frama1[i-1] + alpha1_val * price[i]
            frama2[i] = (1.0 - alpha2_val) * frama2[i-1] + alpha2_val * price[i]
            frama3[i] = (1.0 - alpha3_val) * frama3[i-1] + alpha3_val * price[i]
    
    return frama1, frama2, frama3, dimension, alpha1, alpha2, alpha3


@njit(fastmath=True, cache=True)
def calculate_hyper_triple_frama_indicator_adaptation_core(
    price: np.ndarray, 
    high: np.ndarray, 
    low: np.ndarray, 
    n: int, 
    dynamic_fc: np.ndarray, 
    dynamic_sc: np.ndarray, 
    dynamic_periods: np.ndarray = None,
    alpha_multiplier1: float = 1.0,
    alpha_multiplier2: float = 0.5,
    alpha_multiplier3: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    インジケーター動的適応ハイパートリプルFRAMA計算（HyperERまたはHyperADX対応）
    
    Args:
        price: 価格データ
        high: 高値データ
        low: 安値データ
        n: 基本期間
        dynamic_fc: 動的FC値配列
        dynamic_sc: 動的SC値配列
        dynamic_periods: 動的期間配列（オプション）
        alpha_multiplier1: 1本目のアルファ調整係数
        alpha_multiplier2: 2本目のアルファ調整係数
        alpha_multiplier3: 3本目のアルファ調整係数
    
    Returns:
        1本目FRAMA値、2本目FRAMA値、3本目FRAMA値、フラクタル次元、1本目アルファ値、2本目アルファ値、3本目アルファ値のタプル
    """
    length = len(price)
    
    # 結果配列を初期化
    frama1 = np.zeros(length, dtype=np.float64)
    frama2 = np.zeros(length, dtype=np.float64)
    frama3 = np.zeros(length, dtype=np.float64)
    dimension = np.zeros(length, dtype=np.float64)
    alpha1 = np.zeros(length, dtype=np.float64)
    alpha2 = np.zeros(length, dtype=np.float64)
    alpha3 = np.zeros(length, dtype=np.float64)
    
    # 初期値設定
    for i in range(length):
        frama1[i] = np.nan
        frama2[i] = np.nan
        frama3[i] = np.nan
        dimension[i] = np.nan
        alpha1[i] = np.nan
        alpha2[i] = np.nan
        alpha3[i] = np.nan
    
    for i in range(length):
        # 動的期間または固定期間を使用
        current_period = n
        if dynamic_periods is not None and i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
            current_period = max(4, int(dynamic_periods[i]))
            if current_period % 2 != 0:
                current_period += 1
        
        # 動的FC/SC値を取得（デフォルト値付き）
        current_fc = int(dynamic_fc[i]) if i < len(dynamic_fc) and not np.isnan(dynamic_fc[i]) else 1
        current_sc = int(dynamic_sc[i]) if i < len(dynamic_sc) and not np.isnan(dynamic_sc[i]) else 198
        
        len1 = current_period // 2
        min_period = max(current_period - 1, 0)
        
        if i < min_period:
            continue
        
        # フラクタル次元計算（既存ロジック）
        h1 = -np.inf
        l1 = np.inf
        for j in range(len1):
            if i - j >= 0:
                if high[i - j] > h1:
                    h1 = high[i - j]
                if low[i - j] < l1:
                    l1 = low[i - j]
        
        n1 = (h1 - l1) / len1
        
        h2 = -np.inf
        l2 = np.inf
        for j in range(len1, current_period):
            if i - j >= 0:
                if high[i - j] > h2:
                    h2 = high[i - j]
                if low[i - j] < l2:
                    l2 = low[i - j]
        
        n2 = (h2 - l2) / len1
        
        h3 = -np.inf
        l3 = np.inf
        for j in range(current_period):
            if i - j >= 0:
                if high[i - j] > h3:
                    h3 = high[i - j]
                if low[i - j] < l3:
                    l3 = low[i - j]
        
        n3 = (h3 - l3) / current_period
        
        # フラクタル次元計算
        if n1 > 0 and n2 > 0 and n3 > 0:
            dimen1 = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
            dimen = dimen1
        else:
            dimen = dimension[i-1] if i > 0 else 1.0
        
        dimension[i] = dimen
        
        # w = log(2/(SC+1)) - 動的SC使用
        w = np.log(2.0 / (current_sc + 1))
        
        # アルファ計算（動的FC/SC使用）
        alpha_base = np.exp(w * (dimen - 1.0))
        
        if alpha_base > 1.0:
            oldalpha = 1.0
        elif alpha_base < 0.01:
            oldalpha = 0.01
        else:
            oldalpha = alpha_base
        
        oldN = (2.0 - oldalpha) / oldalpha
        N = (((current_sc - current_fc) * (oldN - 1.0)) / (current_sc - 1.0)) + current_fc
        alpha_ = 2.0 / (N + 1.0)
        
        min_alpha = 2.0 / (current_sc + 1.0)
        if alpha_ < min_alpha:
            final_alpha = min_alpha
        elif alpha_ > 1.0:
            final_alpha = 1.0
        else:
            final_alpha = alpha_
        
        # 各ラインのアルファ値を計算
        alpha1_val = final_alpha * alpha_multiplier1
        alpha2_val = final_alpha * alpha_multiplier2
        alpha3_val = final_alpha * alpha_multiplier3
        
        alpha1[i] = alpha1_val
        alpha2[i] = alpha2_val
        alpha3[i] = alpha3_val
        
        # FRAMA計算
        if i == min_period:
            frama1[i] = price[i]
            frama2[i] = price[i]
            frama3[i] = price[i]
        else:
            frama1[i] = (1.0 - alpha1_val) * frama1[i-1] + alpha1_val * price[i]
            frama2[i] = (1.0 - alpha2_val) * frama2[i-1] + alpha2_val * price[i]
            frama3[i] = (1.0 - alpha3_val) * frama3[i-1] + alpha3_val * price[i]
    
    return frama1, frama2, frama3, dimension, alpha1, alpha2, alpha3


@njit(fastmath=True, cache=True)
def calculate_hyper_triple_frama_dynamic_core(
    price: np.ndarray, 
    high: np.ndarray, 
    low: np.ndarray, 
    n: int, 
    fc: int, 
    sc: int, 
    dynamic_periods: np.ndarray = None,
    alpha_multiplier1: float = 1.0, 
    alpha_multiplier2: float = 0.5,
    alpha_multiplier3: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    動的期間対応ハイパートリプルFRAMAを計算する
    
    Args:
        price: 価格データ
        high: 高値データ
        low: 安値データ
        n: 基本期間（偶数である必要がある）
        fc: Fast Constant
        sc: Slow Constant
        dynamic_periods: 動的期間配列（オプション）
        alpha_multiplier1: 1本目のアルファ調整係数
        alpha_multiplier2: 2本目のアルファ調整係数
        alpha_multiplier3: 3本目のアルファ調整係数
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        1本目FRAMA値、2本目FRAMA値、3本目FRAMA値、フラクタル次元、1本目アルファ値、2本目アルファ値、3本目アルファ値
    """
    length = len(price)
    
    # 結果配列を初期化
    frama1 = np.zeros(length, dtype=np.float64)
    frama2 = np.zeros(length, dtype=np.float64)
    frama3 = np.zeros(length, dtype=np.float64)
    dimension = np.zeros(length, dtype=np.float64)
    alpha1 = np.zeros(length, dtype=np.float64)
    alpha2 = np.zeros(length, dtype=np.float64)
    alpha3 = np.zeros(length, dtype=np.float64)
    
    # 初期値設定
    for i in range(length):
        frama1[i] = np.nan
        frama2[i] = np.nan
        frama3[i] = np.nan
        dimension[i] = np.nan
        alpha1[i] = np.nan
        alpha2[i] = np.nan
        alpha3[i] = np.nan
    
    w = -4.6
    
    for i in range(length):
        # 動的期間または固定期間を使用
        current_period = n
        if dynamic_periods is not None and i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
            # 偶数に調整
            current_period = max(4, int(dynamic_periods[i]))
            if current_period % 2 != 0:
                current_period += 1  # 奇数の場合は偶数にする
        
        len1 = current_period // 2
        min_period = max(current_period - 1, 0)
        
        if i < min_period:
            continue
        
        # N1 = (highest(high,len1)-lowest(low,len1))/len1
        h1 = -np.inf
        l1 = np.inf
        for j in range(len1):
            if i - j >= 0:
                if high[i - j] > h1:
                    h1 = high[i - j]
                if low[i - j] < l1:
                    l1 = low[i - j]
        
        n1 = (h1 - l1) / len1
        
        # N2 = (highest(high,len2)[len1]-lowest(low,len2)[len1])/len1
        h2 = -np.inf
        l2 = np.inf
        for j in range(len1, current_period):
            if i - j >= 0:
                if high[i - j] > h2:
                    h2 = high[i - j]
                if low[i - j] < l2:
                    l2 = low[i - j]
        
        n2 = (h2 - l2) / len1
        
        # N3 = (highest(high,len)-lowest(low,len))/len
        h3 = -np.inf
        l3 = np.inf
        for j in range(current_period):
            if i - j >= 0:
                if high[i - j] > h3:
                    h3 = high[i - j]
                if low[i - j] < l3:
                    l3 = low[i - j]
        
        n3 = (h3 - l3) / current_period
        
        # フラクタル次元の計算
        if n1 > 0 and n2 > 0 and n3 > 0:
            dimen1 = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
            dimen = dimen1
        else:
            dimen = dimension[i-1] if i > 0 else 1.0
        
        dimension[i] = dimen
        
        # アルファ計算
        alpha_base = np.exp(w * (dimen - 1.0))
        
        if alpha_base > 1.0:
            oldalpha = 1.0
        elif alpha_base < 0.01:
            oldalpha = 0.01
        else:
            oldalpha = alpha_base
        
        oldN = (2.0 - oldalpha) / oldalpha
        N = (((sc - fc) * (oldN - 1.0)) / (sc - 1.0)) + fc
        alpha_ = 2.0 / (N + 1.0)
        
        min_alpha = 2.0 / (sc + 1.0)
        if alpha_ < min_alpha:
            final_alpha = min_alpha
        elif alpha_ > 1.0:
            final_alpha = 1.0
        else:
            final_alpha = alpha_
        
        # 各ラインのアルファ値を計算
        alpha1_val = final_alpha * alpha_multiplier1
        alpha2_val = final_alpha * alpha_multiplier2
        alpha3_val = final_alpha * alpha_multiplier3
        
        alpha1[i] = alpha1_val
        alpha2[i] = alpha2_val
        alpha3[i] = alpha3_val
        
        # 各FRAMAラインの計算
        if i == min_period:
            frama1[i] = price[i]
            frama2[i] = price[i]
            frama3[i] = price[i]
        else:
            frama1[i] = (1.0 - alpha1_val) * frama1[i-1] + alpha1_val * price[i]
            frama2[i] = (1.0 - alpha2_val) * frama2[i-1] + alpha2_val * price[i]
            frama3[i] = (1.0 - alpha3_val) * frama3[i-1] + alpha3_val * price[i]
    
    return frama1, frama2, frama3, dimension, alpha1, alpha2, alpha3


class HyperTripleFRAMA(Indicator):
    """
    ハイパートリプルFRAMA - 3本のFRAMAを同時計算するFractal Adaptive Moving Average
    By John Ehlers (改良版)
    
    フラクタル適応移動平均の3本同時計算版：
    - 価格データのフラクタル次元を計算
    - フラクタル次元に基づいて適応的にアルファ値を調整
    - 3本のFRAMAを同時に計算（異なるアルファ乗数を使用）
    - トレンド時は高速、レンジ時は低速で反応
    
    特徴:
    - 1本目FRAMA: デフォルトアルファ乗数1.0（通常のFRAMA）
    - 2本目FRAMA: デフォルトアルファ乗数0.5（より滑らか）
    - 3本目FRAMA: デフォルトアルファ乗数0.1（最も滑らか）
    - アルファ調整係数により柔軟な設定が可能
    - 複数のプライスソースに対応
    """
    
    def __init__(
        self,
        period: int = 16,              # 期間（偶数である必要がある、HyperER動的適応時は無視される）
        src_type: str = 'hl2',         # ソースタイプ
        fc: int = 1,                   # Fast Constant
        sc: int = 198,                 # Slow Constant
        alpha_multiplier1: float = 1.0, # 1本目のアルファ調整係数
        alpha_multiplier2: float = 0.5, # 2本目のアルファ調整係数
        alpha_multiplier3: float = 0.1, # 3本目のアルファ調整係数
        # 動的期間パラメータ
        period_mode: str = 'fixed',    # 'fixed' または 'dynamic'
        cycle_detector_type: str = 'hody_e',
        lp_period: int = 13,
        hp_period: int = 124,
        cycle_part: float = 0.5,
        max_cycle: int = 89,
        min_cycle: int = 8,
        max_output: int = 124,
        min_output: int = 8,
        # 動的適応パラメータ
        enable_indicator_adaptation: bool = True,  # インジケーター動的適応を有効にするか
        adaptation_indicator: str = 'hyper_er',    # 適応に使用するインジケーター ('hyper_er', 'hyper_adx', 'hyper_trend_index')
        hyper_er_period: int = 14,                 # HyperER計算期間
        hyper_er_midline_period: int = 100,        # HyperERミッドライン期間
        hyper_adx_period: int = 14,                # HyperADX計算期間
        hyper_adx_midline_period: int = 100,       # HyperADXミッドライン期間
        hyper_trend_index_period: int = 14,        # HyperTrendIndex計算期間
        hyper_trend_index_midline_period: int = 100, # HyperTrendIndexミッドライン期間
        fc_min: float = 1.0,                       # FC最小値（インジケーター値高い時）
        fc_max: float = 8.0,                      # FC最大値（インジケーター値低い時）
        sc_min: float = 50.0,                      # SC最小値（インジケーター値高い時）
        sc_max: float = 250.0,                     # SC最大値（インジケーター値低い時）
        period_min: int = 4,                       # ピリオド最小値（インジケーター値高い時、偶数）
        period_max: int = 24,                      # ピリオド最大値（インジケーター値低い時、偶数）
        # 平滑化モード設定
        smoothing_mode: str = 'kalman_ultimate',              # 'none', 'kalman', 'ultimate', 'kalman_ultimate'
        # 統合カルマンフィルターパラメータ
        kalman_filter_type: str = 'simple',     # カルマンフィルタータイプ
        kalman_process_noise: float = 1e-5,        # プロセスノイズ
        kalman_min_observation_noise: float = 1e-6, # 最小観測ノイズ
        kalman_adaptation_window: int = 5,         # 適応ウィンドウ
        # Ultimate Smootherパラメータ
        ultimate_smoother_period: int = 10         # Ultimate Smoother期間
    ):
        """
        コンストラクタ
        
        Args:
            period: 期間（偶数である必要がある、デフォルト: 16）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            fc: Fast Constant（デフォルト: 1）
            sc: Slow Constant（デフォルト: 198）
            alpha_multiplier1: 1本目のアルファ調整係数（デフォルト: 1.0）
            alpha_multiplier2: 2本目のアルファ調整係数（デフォルト: 0.5）
            alpha_multiplier3: 3本目のアルファ調整係数（デフォルト: 0.1）
            ... その他のパラメータ（HyperFRAMAと同様）
        """
        # 平滑化モード設定の検証
        valid_smoothing_modes = ['none', 'kalman', 'ultimate', 'kalman_ultimate']
        if smoothing_mode not in valid_smoothing_modes:
            raise ValueError(f"無効な平滑化モード: {smoothing_mode}。有効なオプション: {', '.join(valid_smoothing_modes)}")
        
        # 動的適応文字列の作成
        adaptation_str = ""
        if enable_indicator_adaptation:
            if adaptation_indicator == 'hyper_er':
                adaptation_str += f"_hyper_er({hyper_er_period},{hyper_er_midline_period},period_{period_min}-{period_max})"
            elif adaptation_indicator == 'hyper_adx':
                adaptation_str += f"_hyper_adx({hyper_adx_period},{hyper_adx_midline_period},period_{period_min}-{period_max})"
            elif adaptation_indicator == 'hyper_trend_index':
                adaptation_str += f"_hyper_trend_index({hyper_trend_index_period},{hyper_trend_index_midline_period},period_{period_min}-{period_max})"
        if period_mode == 'dynamic':
            adaptation_str += f"_dynamic({cycle_detector_type})"
        if smoothing_mode != 'none':
            adaptation_str += f"_smooth({smoothing_mode})"
        
        # 指標名の作成（インジケーター動的適応時はピリオド範囲を表示）
        period_str = f"{period_min}-{period_max}" if enable_indicator_adaptation else str(period)
        indicator_name = f"HyperTripleFRAMA(period={period_str}, src={src_type}, fc={fc}, sc={sc}, alpha1={alpha_multiplier1}, alpha2={alpha_multiplier2}, alpha3={alpha_multiplier3}{adaptation_str})"
        super().__init__(indicator_name)
        
        # パラメータの検証
        if period < 2:
            raise ValueError("期間は2以上である必要があります")
        if period % 2 != 0:
            raise ValueError("期間は偶数である必要があります")
        if fc < 1:
            raise ValueError("FC（Fast Constant）は1以上である必要があります")
        if sc < fc:
            raise ValueError("SC（Slow Constant）はFC以上である必要があります")
        
        if alpha_multiplier1 <= 0 or alpha_multiplier1 > 1:
            raise ValueError("alpha_multiplier1は0より大きく1以下である必要があります")
        if alpha_multiplier2 <= 0 or alpha_multiplier2 > 1:
            raise ValueError("alpha_multiplier2は0より大きく1以下である必要があります")
        if alpha_multiplier3 <= 0 or alpha_multiplier3 > 1:
            raise ValueError("alpha_multiplier3は0より大きく1以下である必要があります")
        
        # インジケーター動的適応パラメーターの検証
        if enable_indicator_adaptation:
            if adaptation_indicator not in ['hyper_er', 'hyper_adx', 'hyper_trend_index']:
                raise ValueError("adaptation_indicatorは'hyper_er'、'hyper_adx'、または'hyper_trend_index'である必要があります")
            if period_min < 2:
                raise ValueError("period_minは2以上である必要があります")
            if period_max < period_min:
                raise ValueError("period_maxはperiod_min以上である必要があります")
            if period_min % 2 != 0:
                raise ValueError("period_minは偶数である必要があります")
            if period_max % 2 != 0:
                raise ValueError("period_maxは偶数である必要があります")
        
        # パラメータを保存
        self.period = period
        self.src_type = src_type.lower()
        self.fc = fc
        self.sc = sc
        self.alpha_multiplier1 = alpha_multiplier1
        self.alpha_multiplier2 = alpha_multiplier2
        self.alpha_multiplier3 = alpha_multiplier3
        
        # インジケーター動的適応パラメータ
        self.enable_indicator_adaptation = enable_indicator_adaptation
        self.adaptation_indicator = adaptation_indicator
        self.hyper_er_period = hyper_er_period
        self.hyper_er_midline_period = hyper_er_midline_period
        self.hyper_adx_period = hyper_adx_period
        self.hyper_adx_midline_period = hyper_adx_midline_period
        self.hyper_trend_index_period = hyper_trend_index_period
        self.hyper_trend_index_midline_period = hyper_trend_index_midline_period
        self.fc_min = fc_min
        self.fc_max = fc_max
        self.sc_min = sc_min
        self.sc_max = sc_max
        self.period_min = period_min
        self.period_max = period_max
        
        # 平滑化モード設定
        self.smoothing_mode = smoothing_mode
        
        # 統合カルマンフィルターパラメータ
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_min_observation_noise = kalman_min_observation_noise
        self.kalman_adaptation_window = kalman_adaptation_window
        
        # Ultimate Smootherパラメータ
        self.ultimate_smoother_period = ultimate_smoother_period
        
        # 動的期間パラメータ
        self.period_mode = period_mode.lower()
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        
        # 動的期間検証
        if self.period_mode not in ['fixed', 'dynamic']:
            raise ValueError(f"無効な期間モード: {period_mode}")
        
        # 動的適応インジケーターの初期化（遅延インポート）
        self.hyper_er = None
        self.hyper_adx = None
        self.hyper_trend_index = None
        self._last_hyper_er_values = None
        self._last_hyper_adx_values = None
        self._last_hyper_trend_index_values = None
        self._hyper_er_initialized = False
        self._hyper_adx_initialized = False
        self._hyper_trend_index_initialized = False
        
        # 平滑化フィルターの初期化
        self.kalman_filter = None
        self.ultimate_smoother = None
        
        # カルマンフィルターの初期化（必要な場合）
        if self.smoothing_mode in ['kalman', 'kalman_ultimate']:
            if not UNIFIED_KALMAN_AVAILABLE:
                self.logger.error("統合カルマンフィルターが利用できません。indicators.kalman.unified_kalmanをインポートできません。")
                self.smoothing_mode = 'none'
                self.logger.warning("平滑化機能を無効にしました")
            else:
                try:
                    self.kalman_filter = UnifiedKalman(
                        filter_type=self.kalman_filter_type,
                        src_type=self.src_type,
                        process_noise=self.kalman_process_noise,
                        min_observation_noise=self.kalman_min_observation_noise,
                        adaptation_window=self.kalman_adaptation_window
                    )
                    self.logger.info(f"統合カルマンフィルターを初期化しました: {self.kalman_filter_type}")
                except Exception as e:
                    self.logger.error(f"統合カルマンフィルターの初期化に失敗: {e}")
                    self.smoothing_mode = 'none'
                    self.logger.warning("平滑化機能を無効にしました")
        
        # Ultimate Smootherの初期化（必要な場合）
        if self.smoothing_mode in ['ultimate', 'kalman_ultimate']:
            if not ULTIMATE_SMOOTHER_AVAILABLE:
                self.logger.error("Ultimate Smootherが利用できません。indicators.smoother.ultimate_smootherをインポートできません。")
                if self.smoothing_mode == 'ultimate':
                    self.smoothing_mode = 'none'
                elif self.smoothing_mode == 'kalman_ultimate':
                    self.smoothing_mode = 'kalman'
                self.logger.warning("Ultimate Smoother機能を無効にしました")
            else:
                try:
                    self.ultimate_smoother = UltimateSmoother(
                        period=self.ultimate_smoother_period,
                        src_type=self.src_type,
                        period_mode='fixed'
                    )
                    self.logger.info(f"Ultimate Smootherを初期化しました: 期間={self.ultimate_smoother_period}")
                except Exception as e:
                    self.logger.error(f"Ultimate Smootherの初期化に失敗: {e}")
                    if self.smoothing_mode == 'ultimate':
                        self.smoothing_mode = 'none'
                    elif self.smoothing_mode == 'kalman_ultimate':
                        self.smoothing_mode = 'kalman'
                    self.logger.warning("Ultimate Smoother機能を無効にしました")
        
        # ドミナントサイクル検出器の初期化
        self.dc_detector = None
        self._last_dc_values = None
        if self.period_mode == 'dynamic' and EHLERS_UNIFIED_DC_AVAILABLE:
            try:
                self.dc_detector = EhlersUnifiedDC(
                    detector_type=self.cycle_detector_type,
                    cycle_part=self.cycle_part,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    src_type=self.src_type,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period
                )
            except Exception as e:
                self.logger.warning(f"ドミナントサイクル検出器の初期化に失敗しました: {e}")
                self.period_mode = 'fixed'
        elif self.period_mode == 'dynamic' and not EHLERS_UNIFIED_DC_AVAILABLE:
            self.logger.warning("EhlersUnifiedDCが利用できません。固定期間モードに変更します。")
            self.period_mode = 'fixed'
        
        # ソースタイプの検証（PriceSourceで処理されるため削除可能だが、互換性のため残す）
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        try:
            # データ情報の取得
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0].get('close', data.iloc[0, -1])) if length > 0 else 0.0
                last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1])) if length > 0 else 0.0
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_val = float(data[0, -1])
                        last_val = float(data[-1, -1])
                    else:
                        first_val = float(data[0])
                        last_val = float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            # パラメータ情報
            params_sig = f"{self.period}_{self.src_type}_{self.fc}_{self.sc}_{self.alpha_multiplier1}_{self.alpha_multiplier2}_{self.alpha_multiplier3}"
            
            # ハッシュ計算
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.period}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperTripleFRAMAResult:
        """
        ハイパートリプルFRAMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            HyperTripleFRAMAResult: 3本のFRAMA値、フラクタル次元、各アルファ値を含む結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return HyperTripleFRAMAResult(
                    frama_values=cached_result.frama_values.copy(),
                    second_frama_values=cached_result.second_frama_values.copy(),
                    third_frama_values=cached_result.third_frama_values.copy(),
                    fractal_dimension=cached_result.fractal_dimension.copy(),
                    alpha=cached_result.alpha.copy(),
                    second_alpha=cached_result.second_alpha.copy(),
                    third_alpha=cached_result.third_alpha.copy(),
                    filtered_prices=cached_result.filtered_prices.copy(),
                    smoothing_applied=cached_result.smoothing_applied
                )
            
            # 1. ソース価格データを取得
            source_price = PriceSource.calculate_source(data, self.src_type)
            
            # 2. 平滑化処理（モードに応じて選択）
            smoothed_price, smoothing_applied = self._apply_smoothing(source_price, data)
            
            # 3. HyperTripleFRAMA計算用の価格データを設定
            price = smoothed_price
            
            # 高値・安値データの取得（FRAMAのフラクタル次元計算に必要）
            if isinstance(data, pd.DataFrame):
                if 'high' not in data.columns or 'low' not in data.columns:
                    raise ValueError("DataFrameには'high'と'low'カラムが必要です")
                high = data['high'].to_numpy()
                low = data['low'].to_numpy()
            else:
                # NumPy配列の場合
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                high = data[:, 1]  # high
                low = data[:, 2]   # low
            
            # NumPy配列に変換（float64型で統一）
            price = np.asarray(price, dtype=np.float64)
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < self.period:
                self.logger.warning(f"データ長({data_length})が期間({self.period})より短いです")
            
            # インジケーター動的適応の計算（オプション）
            dynamic_fc = None
            dynamic_sc = None
            dynamic_indicator_periods = None
            if self.enable_indicator_adaptation:
                # HyperERまたはHyperADXを使用
                if self.adaptation_indicator == 'hyper_er':
                    # 遅延インポートでHyperERを初期化
                    if not self._hyper_er_initialized:
                        try:
                            from .trend_filter.hyper_er import HyperER
                            self.hyper_er = HyperER(
                                period=self.hyper_er_period,
                                midline_period=self.hyper_er_midline_period,
                                er_src_type=self.src_type
                            )
                            self._hyper_er_initialized = True
                        except Exception as e:
                            self.logger.warning(f"HyperERインジケーターの初期化に失敗しました: {e}")
                            self.enable_indicator_adaptation = False
                    
                    if self.hyper_er is not None:
                        try:
                            hyper_er_result = self.hyper_er.calculate(data)
                            if hyper_er_result is not None and hasattr(hyper_er_result, 'values'):
                                indicator_values = np.asarray(hyper_er_result.values, dtype=np.float64)
                                dynamic_fc, dynamic_sc, dynamic_indicator_periods = calculate_dynamic_fc_sc_period_vec(
                                    indicator_values, self.fc_min, self.fc_max, self.sc_min, self.sc_max, self.period_min, self.period_max
                                )
                                self._last_hyper_er_values = indicator_values.copy()
                        except Exception as e:
                            self.logger.warning(f"HyperER動的適応計算に失敗しました: {e}")
                            # フォールバック: 前回の値を使用または固定値
                            if self._last_hyper_er_values is not None:
                                dynamic_fc, dynamic_sc, dynamic_indicator_periods = calculate_dynamic_fc_sc_period_vec(
                                    self._last_hyper_er_values, self.fc_min, self.fc_max, self.sc_min, self.sc_max, self.period_min, self.period_max
                                )
                
                elif self.adaptation_indicator == 'hyper_adx':
                    # 遅延インポートでHyperADXを初期化
                    if not self._hyper_adx_initialized:
                        try:
                            from .trend_filter.hyper_adx import HyperADX
                            self.hyper_adx = HyperADX(
                                period=self.hyper_adx_period,
                                midline_period=self.hyper_adx_midline_period
                            )
                            self._hyper_adx_initialized = True
                        except Exception as e:
                            self.logger.warning(f"HyperADXインジケーターの初期化に失敗しました: {e}")
                            self.enable_indicator_adaptation = False
                    
                    if self.hyper_adx is not None:
                        try:
                            hyper_adx_result = self.hyper_adx.calculate(data)
                            if hyper_adx_result is not None and hasattr(hyper_adx_result, 'values'):
                                indicator_values = np.asarray(hyper_adx_result.values, dtype=np.float64)
                                dynamic_fc, dynamic_sc, dynamic_indicator_periods = calculate_dynamic_fc_sc_period_vec(
                                    indicator_values, self.fc_min, self.fc_max, self.sc_min, self.sc_max, self.period_min, self.period_max
                                )
                                self._last_hyper_adx_values = indicator_values.copy()
                        except Exception as e:
                            self.logger.warning(f"HyperADX動的適応計算に失敗しました: {e}")
                            # フォールバック: 前回の値を使用または固定値
                            if self._last_hyper_adx_values is not None:
                                dynamic_fc, dynamic_sc, dynamic_indicator_periods = calculate_dynamic_fc_sc_period_vec(
                                    self._last_hyper_adx_values, self.fc_min, self.fc_max, self.sc_min, self.sc_max, self.period_min, self.period_max
                                )
                
                elif self.adaptation_indicator == 'hyper_trend_index':
                    # 遅延インポートでHyperTrendIndexを初期化
                    if not self._hyper_trend_index_initialized:
                        try:
                            if not HYPER_TREND_INDEX_AVAILABLE:
                                raise ImportError("HyperTrendIndexが利用できません")
                            
                            self.hyper_trend_index = HyperTrendIndex(
                                period=self.hyper_trend_index_period,
                                midline_period=self.hyper_trend_index_midline_period,
                                src_type=self.src_type
                            )
                            self._hyper_trend_index_initialized = True
                        except Exception as e:
                            self.logger.warning(f"HyperTrendIndexインジケーターの初期化に失敗しました: {e}")
                            self.enable_indicator_adaptation = False
                    
                    if self.hyper_trend_index is not None:
                        try:
                            hyper_trend_index_result = self.hyper_trend_index.calculate(data)
                            if hyper_trend_index_result is not None and hasattr(hyper_trend_index_result, 'values'):
                                indicator_values = np.asarray(hyper_trend_index_result.values, dtype=np.float64)
                                dynamic_fc, dynamic_sc, dynamic_indicator_periods = calculate_dynamic_fc_sc_period_vec(
                                    indicator_values, self.fc_min, self.fc_max, self.sc_min, self.sc_max, self.period_min, self.period_max
                                )
                                self._last_hyper_trend_index_values = indicator_values.copy()
                        except Exception as e:
                            self.logger.warning(f"HyperTrendIndex動的適応計算に失敗しました: {e}")
                            # フォールバック: 前回の値を使用または固定値
                            if self._last_hyper_trend_index_values is not None:
                                dynamic_fc, dynamic_sc, dynamic_indicator_periods = calculate_dynamic_fc_sc_period_vec(
                                    self._last_hyper_trend_index_values, self.fc_min, self.fc_max, self.sc_min, self.sc_max, self.period_min, self.period_max
                                )
            
            # 動的期間の計算（オプション）
            dynamic_periods = None
            
            # インジケーター動的適応のピリオドを優先、次にドミナントサイクル検出
            if dynamic_indicator_periods is not None:
                # インジケーター動的適応のピリオドを使用
                dynamic_periods = dynamic_indicator_periods
            elif self.period_mode == 'dynamic' and self.dc_detector is not None:
                try:
                    dc_result = self.dc_detector.calculate(data)
                    if dc_result is not None:
                        dynamic_periods = np.asarray(dc_result, dtype=np.float64)
                        self._last_dc_values = dynamic_periods.copy()
                except Exception as e:
                    self.logger.warning(f"ドミナントサイクル検出に失敗しました: {e}")
                    # フォールバック: 前回の値を使用
                    if self._last_dc_values is not None:
                        dynamic_periods = self._last_dc_values
            
            # ハイパートリプルFRAMAの計算
            if self.enable_indicator_adaptation and dynamic_fc is not None and dynamic_sc is not None:
                # インジケーター動的適応版を使用
                frama1_values, frama2_values, frama3_values, fractal_dim, alpha1_values, alpha2_values, alpha3_values = calculate_hyper_triple_frama_indicator_adaptation_core(
                    price, high, low, self.period, dynamic_fc, dynamic_sc, dynamic_periods, 
                    self.alpha_multiplier1, self.alpha_multiplier2, self.alpha_multiplier3
                )
            elif self.period_mode == 'dynamic' and dynamic_periods is not None:
                # 動的期間対応版を使用
                frama1_values, frama2_values, frama3_values, fractal_dim, alpha1_values, alpha2_values, alpha3_values = calculate_hyper_triple_frama_dynamic_core(
                    price, high, low, self.period, self.fc, self.sc, dynamic_periods, 
                    self.alpha_multiplier1, self.alpha_multiplier2, self.alpha_multiplier3
                )
            else:
                # 固定期間版を使用
                frama1_values, frama2_values, frama3_values, fractal_dim, alpha1_values, alpha2_values, alpha3_values = calculate_hyper_triple_frama_core(
                    price, high, low, self.period, self.fc, self.sc, 
                    self.alpha_multiplier1, self.alpha_multiplier2, self.alpha_multiplier3
                )
            
            # 結果の保存
            result = HyperTripleFRAMAResult(
                frama_values=frama1_values.copy(),
                second_frama_values=frama2_values.copy(),
                third_frama_values=frama3_values.copy(),
                fractal_dimension=fractal_dim.copy(),
                alpha=alpha1_values.copy(),
                second_alpha=alpha2_values.copy(),
                third_alpha=alpha3_values.copy(),
                filtered_prices=smoothed_price.copy() if isinstance(smoothed_price, np.ndarray) else np.array(smoothed_price),
                smoothing_applied=smoothing_applied
            )
            
            # キャッシュを更新
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = frama1_values  # 基底クラスの要件を満たすため（1本目FRAMA値をメインとする）
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ハイパートリプルFRAMA計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = HyperTripleFRAMAResult(
                frama_values=np.array([]),
                second_frama_values=np.array([]),
                third_frama_values=np.array([]),
                fractal_dimension=np.array([]),
                alpha=np.array([]),
                second_alpha=np.array([]),
                third_alpha=np.array([]),
                filtered_prices=np.array([]),
                smoothing_applied='error'
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """1本目FRAMA値のみを取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.frama_values.copy()
    
    def get_frama_values(self) -> Optional[np.ndarray]:
        """
        1本目FRAMA値を取得する
        
        Returns:
            np.ndarray: 1本目FRAMA値
        """
        return self.get_values()
    
    def get_second_frama_values(self) -> Optional[np.ndarray]:
        """
        2本目FRAMA値を取得する
        
        Returns:
            np.ndarray: 2本目FRAMA値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.second_frama_values.copy()
    
    def get_third_frama_values(self) -> Optional[np.ndarray]:
        """
        3本目FRAMA値を取得する
        
        Returns:
            np.ndarray: 3本目FRAMA値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.third_frama_values.copy()
    
    def get_fractal_dimension(self) -> Optional[np.ndarray]:
        """
        フラクタル次元を取得する
        
        Returns:
            np.ndarray: フラクタル次元の値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.fractal_dimension.copy()
    
    def get_alpha(self) -> Optional[np.ndarray]:
        """
        1本目アルファ値を取得する
        
        Returns:
            np.ndarray: 1本目アルファ値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.alpha.copy()
    
    def get_second_alpha(self) -> Optional[np.ndarray]:
        """
        2本目アルファ値を取得する
        
        Returns:
            np.ndarray: 2本目アルファ値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.second_alpha.copy()
    
    def get_third_alpha(self) -> Optional[np.ndarray]:
        """
        3本目アルファ値を取得する
        
        Returns:
            np.ndarray: 3本目アルファ値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.third_alpha.copy()
    
    def get_smoothed_prices(self) -> Optional[np.ndarray]:
        """
        平滑化後の価格を取得する
        
        Returns:
            np.ndarray: 平滑化後の価格
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.filtered_prices.copy()
    
    def get_filtered_prices(self) -> Optional[np.ndarray]:
        """
        平滑化後の価格を取得する（互換性のため）
        
        Returns:
            np.ndarray: 平滑化後の価格
        """
        return self.get_smoothed_prices()
    
    def get_smoothing_applied(self) -> Optional[str]:
        """
        適用された平滑化方法を取得する
        
        Returns:
            str: 適用された平滑化方法
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.smoothing_applied
    
    def _apply_smoothing(self, source_price: np.ndarray, data: Union[pd.DataFrame, np.ndarray]) -> tuple[np.ndarray, str]:
        """
        指定された平滑化モードに応じて価格を平滑化する
        
        Args:
            source_price: 元の価格データ
            data: 元の価格データ（DataFrameまたはNumPy配列）
            
        Returns:
            tuple[np.ndarray, str]: 平滑化後の価格、適用された平滑化方法
        """
        smoothed_price = source_price.copy()
        applied_method = 'none'
        
        try:
            if self.smoothing_mode == 'none':
                # 平滑化なし
                applied_method = 'none'
                
            elif self.smoothing_mode == 'kalman':
                # カルマンフィルターのみ
                if self.kalman_filter is not None:
                    kalman_result = self.kalman_filter.calculate(data)
                    if kalman_result is not None and hasattr(kalman_result, 'values'):
                        smoothed_price = kalman_result.values
                        applied_method = 'kalman'
                        self.logger.debug("カルマンフィルターを適用しました")
                    else:
                        self.logger.warning("カルマンフィルター結果が無効。元の価格を使用します。")
                        applied_method = 'kalman_failed'
                else:
                    self.logger.warning("カルマンフィルターが初期化されていません。")
                    applied_method = 'kalman_unavailable'
                    
            elif self.smoothing_mode == 'ultimate':
                # Ultimate Smootherのみ
                if self.ultimate_smoother is not None:
                    # 価格データをDataFrame形式に変換してUltimate Smootherに渡す
                    temp_df = self._create_temp_dataframe(smoothed_price, data)
                    ultimate_result = self.ultimate_smoother.calculate(temp_df)
                    if ultimate_result is not None and hasattr(ultimate_result, 'values'):
                        smoothed_price = ultimate_result.values
                        applied_method = 'ultimate'
                        self.logger.debug("Ultimate Smootherを適用しました")
                    else:
                        self.logger.warning("Ultimate Smoother結果が無効。元の価格を使用します。")
                        applied_method = 'ultimate_failed'
                else:
                    self.logger.warning("Ultimate Smootherが初期化されていません。")
                    applied_method = 'ultimate_unavailable'
                    
            elif self.smoothing_mode == 'kalman_ultimate':
                # カルマンフィルター → Ultimate Smootherの順で二次平滑化
                kalman_applied = False
                ultimate_applied = False
                
                # 第1段階: カルマンフィルター
                if self.kalman_filter is not None:
                    kalman_result = self.kalman_filter.calculate(data)
                    if kalman_result is not None and hasattr(kalman_result, 'values'):
                        smoothed_price = kalman_result.values
                        kalman_applied = True
                        self.logger.debug("カルマンフィルターを適用しました")
                    else:
                        self.logger.warning("カルマンフィルター結果が無効。")
                else:
                    self.logger.warning("カルマンフィルターが初期化されていません。")
                
                # 第2段階: Ultimate Smoother（カルマンフィルター後の価格に対して）
                if self.ultimate_smoother is not None:
                    # カルマンフィルター後の価格をDataFrame形式に変換
                    temp_df = self._create_temp_dataframe(smoothed_price, data)
                    ultimate_result = self.ultimate_smoother.calculate(temp_df)
                    if ultimate_result is not None and hasattr(ultimate_result, 'values'):
                        smoothed_price = ultimate_result.values
                        ultimate_applied = True
                        self.logger.debug("Ultimate Smootherを適用しました")
                    else:
                        self.logger.warning("Ultimate Smoother結果が無効。")
                else:
                    self.logger.warning("Ultimate Smootherが初期化されていません。")
                
                # 適用結果の記録
                if kalman_applied and ultimate_applied:
                    applied_method = 'kalman_ultimate'
                elif kalman_applied:
                    applied_method = 'kalman_only'
                elif ultimate_applied:
                    applied_method = 'ultimate_only'
                else:
                    applied_method = 'kalman_ultimate_failed'
                    
        except Exception as e:
            self.logger.error(f"平滑化処理中にエラー: {e}")
            applied_method = f'{self.smoothing_mode}_error'
        
        return smoothed_price, applied_method
    
    def _create_temp_dataframe(self, price_values: np.ndarray, original_data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Ultimate Smoother用の一時的なDataFrameを作成する
        
        Args:
            price_values: 価格値の配列
            original_data: 元のデータ
            
        Returns:
            pd.DataFrame: Ultimate Smoother用の一時DataFrame
        """
        try:
            if isinstance(original_data, pd.DataFrame):
                # DataFrameの場合は必要なカラムをコピーしてprice_valuesで置き換え
                temp_df = original_data.copy()
                # ソースタイプに応じて適切なカラムを更新
                if self.src_type == 'close' and 'close' in temp_df.columns:
                    temp_df['close'] = price_values
                elif self.src_type in ['hlc3', 'hl2', 'ohlc4']:
                    # 複合ソースの場合はcloseを更新
                    if 'close' in temp_df.columns:
                        temp_df['close'] = price_values
                return temp_df
            else:
                # NumPy配列の場合は簡単なDataFrameを作成
                if original_data.ndim == 2 and original_data.shape[1] >= 4:
                    # OHLC形式の場合
                    temp_df = pd.DataFrame({
                        'open': original_data[:, 0],
                        'high': original_data[:, 1],
                        'low': original_data[:, 2],
                        'close': price_values
                    })
                else:
                    # 単一価格の場合
                    temp_df = pd.DataFrame({
                        'close': price_values,
                        'high': price_values,
                        'low': price_values,
                        'open': price_values
                    })
                return temp_df
        except Exception as e:
            self.logger.error(f"一時DataFrame作成中にエラー: {e}")
            # フォールバック: 簡単なDataFrameを作成
            return pd.DataFrame({
                'close': price_values,
                'high': price_values,
                'low': price_values,
                'open': price_values
            })
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        if self.kalman_filter:
            self.kalman_filter.reset()
        if self.ultimate_smoother:
            self.ultimate_smoother.reset()
        self._result_cache = {}
        self._cache_keys = []