#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, Literal
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, int64, boolean

from .indicator import Indicator
from .price_source import PriceSource
from .c_atr import CATR
from .z_adaptive_ma import ZAdaptiveMA
from .cycle_efficiency_ratio import CycleEfficiencyRatio
from .x_trend_index import XTrendIndex
from .z_adaptive_trend_index import ZAdaptiveTrendIndex
from .roc_persistence import ROCPersistence
from .alma import ALMA
from .cycle_rsx import CycleRSX
from .hma import HMA
from .hyper_smoother import hyper_smoother


@njit(fastmath=True, cache=True)
def calculate_ema_numba(data: np.ndarray, period: int) -> np.ndarray:
    """
    指数移動平均（EMA）を計算する（Numba最適化版）
    
    Args:
        data: 入力データ配列
        period: EMA期間
    
    Returns:
        EMA値の配列
    """
    length = len(data)
    result = np.full(length, np.nan)
    
    if length == 0 or period <= 0:
        return result
    
    # アルファ値の計算
    alpha = 2.0 / (period + 1.0)
    
    # 最初の有効値を見つけてEMAの初期値に設定
    start_idx = 0
    while start_idx < length and np.isnan(data[start_idx]):
        start_idx += 1
    
    if start_idx >= length:
        return result  # 全てNaN
    
    # 初期値を設定
    result[start_idx] = data[start_idx]
    
    # EMA計算
    for i in range(start_idx + 1, length):
        if not np.isnan(data[i]):
            if not np.isnan(result[i-1]):
                result[i] = alpha * data[i] + (1.0 - alpha) * result[i-1]
            else:
                result[i] = data[i]
        else:
            result[i] = result[i-1] if not np.isnan(result[i-1]) else np.nan
    
    return result


@dataclass
class ZAdaptiveChannelResult:
    """Zアダプティブチャネルの計算結果"""
    middle: np.ndarray        # 中心線（ZAdaptiveMA）
    upper: np.ndarray         # 上限バンド
    lower: np.ndarray         # 下限バンド
    er: np.ndarray            # Efficiency Ratio (CER)
    dynamic_multiplier: np.ndarray  # 動的乗数
    z_atr: np.ndarray         # CATR値
    max_mult_values: np.ndarray  # 動的に計算されたmax_multiplier値
    min_mult_values: np.ndarray  # 動的に計算されたmin_multiplier値
    multiplier_trigger: np.ndarray  # 乗数計算に使用されたトリガー値
    # X-Trend Index関連の結果を追加
    x_trend_values: np.ndarray  # X-Trend Indexの値
    x_trend_upper_multiplier: np.ndarray  # X-Trend Index調整されたアッパーバンド乗数
    x_trend_lower_multiplier: np.ndarray  # X-Trend Index調整されたロワーバンド乗数
    # 乗数平滑化結果を追加
    upper_multiplier_smoothed: np.ndarray  # 平滑化されたアッパーバンド乗数
    lower_multiplier_smoothed: np.ndarray  # 平滑化されたロワーバンド乗数


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True, target='parallel')
def calculate_dynamic_multiplier_vec(trigger: float, max_mult: float, min_mult: float) -> float:
    """
    トリガー値に基づいて動的乗数を計算する（ベクトル化&並列版）
    
    Args:
        trigger: トリガー値（0〜1.0の範囲、トレンドインデックスやCERの絶対値）
        max_mult: 最大乗数（トレンド時に使用）
        min_mult: 最小乗数（非トレンド時に使用）
    
    Returns:
        動的乗数値
    """
    # トリガー値はCERの場合は絶対値を使用、XトレンドやZアダプティブトレンドの場合はそのまま使用
    # どちらにしても0〜1の範囲で扱う
    
    # ZChannelと同様の計算式を使用：トリガー値が高いほどバンド幅が小さくなる
    return max_mult - trigger * (max_mult - min_mult)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True, target='parallel')
def calculate_dynamic_max_multiplier(trigger: float, max_max_mult: float, min_max_mult: float) -> float:
    """
    トリガー値に基づいて動的最大乗数を計算する（ベクトル化&並列版）
    
    Args:
        trigger: トリガー値（0〜1.0の範囲、トレンドインデックスやCERの絶対値）
        max_max_mult: 最大乗数の最大値（強いトレンド時に使用）
        min_max_mult: 最大乗数の最小値（中程度のトレンド時に使用）
    
    Returns:
        動的最大乗数値
    """
    # トリガー値は0〜1の範囲で扱う
    
    # ZChannelと同様の計算式：トリガー値が高いほど最大乗数が小さくなる
    return max_max_mult - trigger * (max_max_mult - min_max_mult)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True, target='parallel')
def calculate_dynamic_min_multiplier(trigger: float, max_min_mult: float, min_min_mult: float) -> float:
    """
    トリガー値に基づいて動的最小乗数を計算する（ベクトル化&並列版）
    
    Args:
        trigger: トリガー値（0〜1.0の範囲、トレンドインデックスやCERの絶対値）
        max_min_mult: 最小乗数の最大値（中程度のトレンド時に使用）
        min_min_mult: 最小乗数の最小値（非トレンド時に使用）
    
    Returns:
        動的最小乗数値
    """
    # トリガー値は0〜1の範囲で扱う
    
    # ZChannelと同様の計算式：トリガー値が高いほど最小乗数が小さくなる
    return max_min_mult - trigger * (max_min_mult - min_min_mult)


@njit(float64[:](float64[:], float64, float64), fastmath=True, parallel=True, cache=True)
def calculate_dynamic_multiplier_optimized(trigger: np.ndarray, max_mult: float, min_mult: float) -> np.ndarray:
    """
    トリガー値に基づいて動的乗数を計算する（最適化&並列版）
    
    Args:
        trigger: トリガー値の配列（0〜1.0の範囲）
        max_mult: 最大乗数（トレンド時に使用）
        min_mult: 最小乗数（非トレンド時に使用）
    
    Returns:
        動的乗数値の配列
    """
    result = np.empty_like(trigger)
    
    for i in prange(len(trigger)):
        # トリガー値は0〜1の範囲で扱う
        
        # ZChannelと同様の計算式を使用：トリガー値が高いほどバンド幅が小さくなる
        result[i] = max_mult - trigger[i] * (max_mult - min_mult)
    
    return result


@njit(fastmath=True, parallel=True, cache=True)
def calculate_z_adaptive_channel_optimized(
    z_ma: np.ndarray,
    z_atr: np.ndarray,
    dynamic_multiplier: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Zアダプティブチャネルを計算する（最適化&並列版）
    
    Args:
        z_ma: ZAdaptiveMAの配列（中心線）
        z_atr: CATRの配列（ボラティリティ測定・金額ベース）
        dynamic_multiplier: 動的乗数の配列
    
    Returns:
        中心線、上限バンド、下限バンドのタプル
    """
    length = len(z_ma)
    
    # 結果用の配列を初期化（中心線は既存配列を再利用）
    middle = z_ma  # 直接参照で最適化（コピー不要）
    upper = np.empty(length, dtype=np.float64)
    lower = np.empty(length, dtype=np.float64)
    
    # 並列処理で各時点でのバンドを計算
    for i in prange(length):
        # 動的乗数をATRに適用
        if np.isnan(z_ma[i]) or np.isnan(z_atr[i]) or np.isnan(dynamic_multiplier[i]):
            upper[i] = np.nan
            lower[i] = np.nan
            continue
            
        band_width = z_atr[i] * dynamic_multiplier[i]
        
        # 金額ベースのATRを使用（絶対値）
        upper[i] = z_ma[i] + band_width
        lower[i] = z_ma[i] - band_width
    
    return middle, upper, lower


@vectorize(['float64(float64, float64)'], nopython=True, fastmath=True, cache=True, target='parallel')
def calculate_new_simple_dynamic_multiplier_vec(cer: float, x_trend: float) -> float:
    """
    新しいシンプルな動的乗数を計算する（ベクトル化&並列版）
    
    Args:
        cer: CER値（サイクル効率比）
        x_trend: XTRENDINDEXの値（0〜1.0の範囲）
    
    Returns:
        動的乗数値 = 15 - (CER*10) - (XTRENDINDEX*10)
        （計算結果が0.5以下の場合は0.5に制限）
    """
    # CERは負の値も取り得るため、絶対値を使用
    abs_cer = abs(cer) if not np.isnan(cer) else 0.0
    # X_TRENDは0-1の範囲
    safe_x_trend = x_trend if not np.isnan(x_trend) else 0.0
    
    # 動的乗数を計算
    result = 18.0 - (abs_cer * 10.0) - (safe_x_trend * 10.0)
    
    # 0.5以下の場合は0.5に制限
    return max(result, 0.5)


@njit(float64[:](float64[:], float64[:]), fastmath=True, parallel=True, cache=True)
def calculate_new_simple_dynamic_multiplier_optimized(cer: np.ndarray, x_trend: np.ndarray) -> np.ndarray:
    """
    新しいシンプルな動的乗数を計算する（最適化&並列版）
    
    Args:
        cer: CER値の配列（サイクル効率比）
        x_trend: XTRENDINDEXの値の配列（0〜1.0の範囲）
    
    Returns:
        動的乗数値の配列 = 16 - (CER*10) - (XTRENDINDEX*10)
        （計算結果が0.5以下の場合は0.5に制限）
    """
    result = np.empty_like(cer)
    
    for i in prange(len(cer)):
        # CERは負の値も取り得るため、絶対値を使用
        abs_cer = abs(cer[i]) if not np.isnan(cer[i]) else 0.0
        # X_TRENDは0-1の範囲
        safe_x_trend = x_trend[i] if not np.isnan(x_trend[i]) else 0.0
        
        # 動的乗数を計算
        multiplier = 15.0 - (abs_cer * 10.0) - (safe_x_trend * 10.0)
        
        # 0.5以下の場合は0.5に制限
        result[i] = max(multiplier, 0.5)
    
    return result


@njit(fastmath=True, parallel=True, cache=True)
def adjust_multipliers_with_roc_persistence(
    base_multiplier: np.ndarray,
    roc_persistence_values: np.ndarray,
    roc_directions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ROC継続性に基づいてアッパーバンドとロワーバンドの乗数を調整する
    
    新しいロジック:
    - roc_directionsが-1の場合: アッパーバンド乗数 = 動的乗数 - 10*abs(ROCPersistenceのvalues)
    - roc_directionsが1の場合: アッパーバンドは通常の動的乗数を使用
    - roc_directionsが1の場合: ロワーバンド乗数 = 動的乗数 - 10*abs(ROCPersistenceのvalues)
    - roc_directionsが-1の場合: ロワーバンドは通常の動的乗数を使用
    - 調整後の乗数が0.5以下になったら、0.5とする
    
    Args:
        base_multiplier: 基本の動的乗数
        roc_persistence_values: ROC継続性の値（-1から1）
        roc_directions: ROC方向（1=正、-1=負、0=ゼロ）
    
    Returns:
        調整されたアッパーバンド乗数とロワーバンド乗数のタプル
    """
    length = len(base_multiplier)
    upper_multiplier = np.empty(length, dtype=np.float64)
    lower_multiplier = np.empty(length, dtype=np.float64)
    
    for i in prange(length):
        # 基本値で初期化
        upper_mult = base_multiplier[i]
        lower_mult = base_multiplier[i]
        
        # NaN値の場合はそのまま
        if (np.isnan(base_multiplier[i]) or 
            np.isnan(roc_persistence_values[i]) or 
            np.isnan(roc_directions[i])):
            upper_multiplier[i] = upper_mult
            lower_multiplier[i] = lower_mult
            continue
        
        # ROC方向に基づく乗数調整
        if roc_directions[i] == -1:  # 負の方向（下降）
            # アッパーバンド乗数のみを調整
            # アッパーバンド乗数 = 動的乗数 - 10*abs(ROCPersistenceのvalues)
            upper_mult = base_multiplier[i] - 10.0 * abs(roc_persistence_values[i])
            # ロワーバンドは通常の動的乗数を使用
            lower_mult = base_multiplier[i]
            
        elif roc_directions[i] == 1:  # 正の方向（上昇）
            # ロワーバンド乗数のみを調整
            # ロワーバンド乗数 = 動的乗数 - 10*abs(ROCPersistenceのvalues)
            lower_mult = base_multiplier[i] - 10.0 * abs(roc_persistence_values[i])
            # アッパーバンドは通常の動的乗数を使用
            upper_mult = base_multiplier[i]
        
        else:  # roc_directions[i] == 0（ゼロの場合）
            # 両方とも通常の動的乗数を使用
            upper_mult = base_multiplier[i]
            lower_mult = base_multiplier[i]
        
        # 乗数が0.5以下にならないよう制限（最小値0.5）
        upper_multiplier[i] = max(upper_mult, 0.5)
        lower_multiplier[i] = max(lower_mult, 0.5)
    
    return upper_multiplier, lower_multiplier


@njit(fastmath=True, parallel=True, cache=True)
def calculate_z_adaptive_channel_with_roc_persistence(
    z_ma: np.ndarray,
    z_atr: np.ndarray,
    upper_multiplier: np.ndarray,
    lower_multiplier: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ROC継続性を考慮したZアダプティブチャネルを計算する（最適化&並列版）
    
    Args:
        z_ma: ZAdaptiveMAの配列（中心線）
        z_atr: CATRの配列（ボラティリティ測定・金額ベース）
        upper_multiplier: アッパーバンド用の調整乗数
        lower_multiplier: ロワーバンド用の調整乗数
    
    Returns:
        中心線、上限バンド、下限バンドのタプル
    """
    length = len(z_ma)
    
    # 結果用の配列を初期化（中心線は既存配列を再利用）
    middle = z_ma  # 直接参照で最適化（コピー不要）
    upper = np.empty(length, dtype=np.float64)
    lower = np.empty(length, dtype=np.float64)
    
    # 並列処理で各時点でのバンドを計算
    for i in prange(length):
        # NaN値チェック
        if (np.isnan(z_ma[i]) or np.isnan(z_atr[i]) or 
            np.isnan(upper_multiplier[i]) or np.isnan(lower_multiplier[i])):
            upper[i] = np.nan
            lower[i] = np.nan
            continue
        
        # 各バンドで異なる乗数を適用
        upper_band_width = z_atr[i] * upper_multiplier[i]
        lower_band_width = z_atr[i] * lower_multiplier[i]
        
        # 金額ベースのATRを使用（絶対値）
        upper[i] = z_ma[i] + upper_band_width
        lower[i] = z_ma[i] - lower_band_width
    
    return middle, upper, lower




@njit(fastmath=True, parallel=True, cache=True)
def adjust_multipliers_with_x_trend_index(
    upper_multiplier: np.ndarray,
    lower_multiplier: np.ndarray,
    x_trend_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X-Trend Indexに基づいて乗数を動的に調整する
    
    調整ロジック:
    - X-Trend Index ≤ 0.5: レンジ相場 -> 乗数100%使用
    - X-Trend Index = 0.9: トレンド相場 -> 乗数50%使用
    - 0.5 < X-Trend < 0.9: 線形補間で100%から50%まで調整
    - アッパーバンド、ロワーバンドとも同じロジックを適用
    - 最小乗数0.5以下の場合は0.5に制限
    
    Args:
        upper_multiplier: アッパーバンド乗数配列
        lower_multiplier: ロワーバンド乗数配列
        x_trend_values: X-Trend Index値配列 (0-1の範囲)
    
    Returns:
        調整されたアッパーバンド乗数とロワーバンド乗数のタプル
    """
    length = len(upper_multiplier)
    adjusted_upper = np.empty(length, dtype=np.float64)
    adjusted_lower = np.empty(length, dtype=np.float64)
    
    for i in prange(length):
        # デフォルトは元の乗数を使用
        adj_upper = upper_multiplier[i]
        adj_lower = lower_multiplier[i]
        
        # NaN値の場合はそのまま
        if (np.isnan(upper_multiplier[i]) or 
            np.isnan(lower_multiplier[i]) or 
            np.isnan(x_trend_values[i])):
            adjusted_upper[i] = adj_upper
            adjusted_lower[i] = adj_lower
            continue
        
        x_trend = x_trend_values[i]
        
        # X-Trend Indexに基づく使用率計算 (アッパー、ロワーとも同じロジック)
        if x_trend <= 0.5:
            # X-Trend <= 0.5: 100%使用 (レンジ相場)
            usage_percentage = 1.0
        elif x_trend >= 0.9:
            # X-Trend >= 0.9: 50%使用 (強いトレンド相場)
            usage_percentage = 0.5
        else:
            # 0.5 < X-Trend < 0.9: 線形補間 (100%から50%へ)
            # percentage = 1.0 - (x_trend - 0.5) * (1.0 - 0.5) / (0.9 - 0.5)
            # percentage = 1.0 - (x_trend - 0.5) * 0.5 / 0.4
            # percentage = 1.0 - (x_trend - 0.5) * 1.25
            usage_percentage = 1.0 - (x_trend - 0.5) * 1.25
        
        # 乗数に使用率を適用
        adj_upper = upper_multiplier[i] * usage_percentage
        adj_lower = lower_multiplier[i] * usage_percentage
        
        # 最小値制限 (最小動的乗数0.5)
        adjusted_upper[i] = max(adj_upper, 0.5)  # 最小乗数0.5
        adjusted_lower[i] = max(adj_lower, 0.5)  # 最小乗数0.5
    
    return adjusted_upper, adjusted_lower


@vectorize(['float64(float64)'], nopython=True, fastmath=True, cache=True, target='parallel')
def calculate_simple_adjustment_multiplier_vec(trigger: float) -> float:
    """
    シンプルアジャストメント動的乗数を計算する（ベクトル化&並列版）
    
    Args:
        trigger: トリガー値（0〜1.0の範囲、トレンドインデックスやCERの絶対値）
    
    Returns:
        動的乗数値 = MAX - trigger * (MAX - MIN)
        MAX = 8.0, MIN = 0.5
        トリガー値が0の時は8.0、トリガー値が1の時は0.5
    """
    # 定数定義
    MAX_MULTIPLIER = 6.0
    MIN_MULTIPLIER = 1.0
    
    # トリガー値をクランプ（0-1の範囲に制限）
    safe_trigger = min(max(trigger if not np.isnan(trigger) else 0.0, 0.0), 1.0)
    
    # 線形補間で動的乗数を計算
    # トリガー値が高いほど乗数が小さくなる（ボラティリティが低下）
    result = MAX_MULTIPLIER - safe_trigger * (MAX_MULTIPLIER - MIN_MULTIPLIER)
    
    return result


@njit(float64[:](float64[:]), fastmath=True, parallel=True, cache=True)
def calculate_simple_adjustment_multiplier_optimized(trigger: np.ndarray) -> np.ndarray:
    """
    シンプルアジャストメント動的乗数を計算する（最適化&並列版）
    
    Args:
        trigger: トリガー値の配列（0〜1.0の範囲）
    
    Returns:
        動的乗数値の配列 = MAX - trigger * (MAX - MIN)
        MAX = 8.0, MIN = 0.5
        トリガー値が0の時は8.0、トリガー値が1の時は0.5
    """
    # 定数定義
    MAX_MULTIPLIER = 6.0
    MIN_MULTIPLIER = 1.0
    DIFF = MAX_MULTIPLIER - MIN_MULTIPLIER  # 7.5
    
    result = np.empty_like(trigger)
    
    for i in prange(len(trigger)):
        # トリガー値をクランプ（0-1の範囲に制限）
        safe_trigger = min(max(trigger[i] if not np.isnan(trigger[i]) else 0.0, 0.0), 1.0)
        
        # 線形補間で動的乗数を計算
        # トリガー値が高いほど乗数が小さくなる（ボラティリティが低下）
        result[i] = MAX_MULTIPLIER - safe_trigger * DIFF
    
    return result


class ZAdaptiveChannel(Indicator):
    """
    ZAdaptiveChannel（Zアダプティブチャネル）インディケーター
    
    特徴:
    - 効率比(CER)、Xトレンドインデックス、Zアダプティブトレンドインデックスのいずれかに基づいて
      動的に調整されるチャネル幅
    - ZAdaptiveMAを中心線として使用
    - CATRを使用したボラティリティベースのバンド
    - トレンド強度に応じて自動調整されるATR乗数
    - 3つの乗数計算方法:
      * adaptive: 従来の動的乗数計算（パラメータで設定可能な範囲）
      * simple: シンプルな動的乗数計算（15 - (CER*10) - (XTRENDINDEX*10)）
      * simple_adjustment: シンプルアジャストメント（MAX=8からMIN=0.5まで線形補間）
    - サイクルRSXベースの動的乗数調整（オプション）
    - 乗数の平滑化（ALMA、HMA、HyperSmoother、EMA）
    
    使用方法:
    - 動的なサポート/レジスタンスレベルの特定
    - トレンドの方向性とボラティリティに基づくエントリー/エグジット
    - トレンド分析
    """
    
    def __init__(
        self,
        # 動的乗数の範囲パラメータ
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 3.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        src_type: str = 'hlc3',       # 'open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4'
        
        # 乗数計算方法選択
        multiplier_method: str = 'simple_adjustment',  # 'adaptive', 'simple', 'simple_adjustment'
        
        # トリガーソース選択
        multiplier_source: str = 'cer',  # 'cer', 'x_trend', 'z_trend'
        ma_source: str = 'x_trend',          # ZAdaptiveMAに渡すソース（'cer', 'x_trend'）
        
        # X-Trend Index調整の有効化
        use_x_trend_adjustment: bool = True,
        
        # 乗数平滑化オプション
        multiplier_smoothing_method: str = 'none',  # 'none', 'alma', 'hma', 'hyper', 'ema'
        multiplier_smoothing_period: int = 4,        # 平滑化期間
        alma_offset: float = 0.85,                   # ALMA用オフセット
        alma_sigma: float = 6,                       # ALMA用シグマ
        
        # CERパラメータ
        detector_type: str = 'dudi_e',     # CER用ドミナントサイクル検出器タイプ
        cycle_part: float = 0.4,           # CER用サイクル部分
        lp_period: int = 5,               # CER用ローパスフィルター期間
        hp_period: int = 100,              # CER用ハイパスフィルター期間
        max_cycle: int = 120,              # CER用最大サイクル期間
        min_cycle: int = 10,               # CER用最小サイクル期間
        max_output: int = 75,             # CER用最大出力値
        min_output: int = 5,              # CER用最小出力値
        use_kalman_filter: bool = False,   # CER用カルマンフィルター使用有無
        
        # ZAdaptiveMA用パラメータ
        fast_period: int = 2,             # 速い移動平均の期間（固定値）
        slow_period: int = 30,            # 遅い移動平均の期間（固定値）
        
        # Xトレンドインデックスパラメータ（multiplier_source='x_trend'またはma_source='x_trend'の場合に使用）
        x_detector_type: str = 'dudi_e',
        x_cycle_part: float = 0.7,
        x_max_cycle: int = 120,
        x_min_cycle: int = 5,
        x_max_output: int = 55,
        x_min_output: int = 8,
        x_smoother_type: str = 'alma',
        
        # 固定しきい値のパラメータ（XTrendIndex用）
        fixed_threshold: float = 0.65,
        
        # ROC Persistenceパラメータ
        roc_detector_type: str = 'hody_e',
        roc_max_persistence_periods: int = 89,
        roc_smooth_persistence: bool = False,
        roc_persistence_smooth_period: int = 3,
        roc_smooth_roc: bool = True,
        roc_alma_period: int = 5,
        roc_alma_offset: float = 0.85,
        roc_alma_sigma: float = 6,
        roc_signal_threshold: float = 0.0,
        
        # Cycle RSXパラメータ
        cycle_rsx_detector_type: str = 'dudi_e',
        cycle_rsx_lp_period: int = 5,
        cycle_rsx_hp_period: int = 89,
        cycle_rsx_cycle_part: float = 0.4,
        cycle_rsx_max_cycle: int = 55,
        cycle_rsx_min_cycle: int = 5,
        cycle_rsx_max_output: int = 34,
        cycle_rsx_min_output: int = 3,
        cycle_rsx_src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            max_max_multiplier: 最大乗数の最大値（動的乗数使用時）
            min_max_multiplier: 最大乗数の最小値（動的乗数使用時）
            max_min_multiplier: 最小乗数の最大値（動的乗数使用時）
            min_min_multiplier: 最小乗数の最小値（動的乗数使用時）
            src_type: ソースタイプ
            
            multiplier_method: 乗数計算方法
                'adaptive': 従来の動的乗数計算（デフォルト）
                'simple': シンプルな動的乗数計算（15 - (CER*10) - (XTRENDINDEX*10)）
                'simple_adjustment': シンプルアジャストメント動的乗数計算（MAX=8からMIN=0.5まで線形補間）
            
            multiplier_source: 乗数計算に使用するトリガーのソース
                'cer': サイクル効率比（デフォルト）
                'x_trend': Xトレンドインデックス
                'z_trend': Zアダプティブトレンドインデックス
            
            ma_source: ZAdaptiveMAに渡すソース
                'cer': サイクル効率比（デフォルト）
                'x_trend': Xトレンドインデックス
            
            use_roc_persistence: ROC Persistence調整の有効化
            
            use_cycle_rsx_adjustment: サイクルRSX調整の有効化
                - 固定の線形補間アルゴリズムを使用：
                - アッパーバンド乗数（RSX ≤ 50）：RSX=50で100%、RSX=10で10%使用
                - ロワーバンド乗数（RSX ≥ 50）：RSX=50で100%、RSX=90で10%使用
            
            use_x_trend_adjustment: X-Trend Index調整の有効化
            
            multiplier_smoothing_method: 乗数平滑化方法
                'none': 平滑化なし（デフォルト）
                'alma': ALMA平滑化
                'hma': HMA平滑化
                'hyper': HyperSmoother平滑化
                'ema': EMA平滑化
            multiplier_smoothing_period: 平滑化期間
            alma_offset: ALMA用オフセット
            alma_sigma: ALMA用シグマ
            
            detector_type: CER用ドミナントサイクル検出器タイプ
            cycle_part: CER用サイクル部分
            lp_period: CER用ローパスフィルター期間
            hp_period: CER用ハイパスフィルター期間
            max_cycle: CER用最大サイクル期間
            min_cycle: CER用最小サイクル期間
            max_output: CER用最大出力値
            min_output: CER用最小出力値
            use_kalman_filter: CER用カルマンフィルター使用有無
            
            fast_period: 速い移動平均の期間（固定値）
            slow_period: 遅い移動平均の期間（固定値）
            
            x_detector_type: Xトレンド用検出器タイプ
            x_cycle_part: Xトレンド用サイクル部分
            x_max_cycle: Xトレンド用最大サイクル期間
            x_min_cycle: Xトレンド用最小サイクル期間
            x_max_output: Xトレンド用最大出力値
            x_min_output: Xトレンド用最小出力値
            x_smoother_type: Xトレンド用平滑化タイプ
            
            fixed_threshold: 固定しきい値（XTrendIndex用）
            
            roc_detector_type: ROC Persistence用検出器タイプ
            roc_max_persistence_periods: ROC Persistence用最大持続期間
            roc_smooth_persistence: ROC Persistence平滑化有無
            roc_persistence_smooth_period: ROC Persistence平滑化期間
            roc_smooth_roc: ROC Persistence平滑化ROC有無
            roc_alma_period: ROC Persistence用ALMA期間
            roc_alma_offset: ROC Persistence用ALMAオフセット
            roc_alma_sigma: ROC Persistence用ALMAシグマ
            roc_signal_threshold: ROC Persistence信号閾値
            
            cycle_rsx_detector_type: サイクルRSX用検出器タイプ
            cycle_rsx_lp_period: サイクルRSX用ローパスフィルター期間
            cycle_rsx_hp_period: サイクルRSX用ハイパスフィルター期間
            cycle_rsx_cycle_part: サイクルRSX用サイクル部分
            cycle_rsx_max_cycle: サイクルRSX用最大サイクル期間
            cycle_rsx_min_cycle: サイクルRSX用最小サイクル期間
            cycle_rsx_max_output: サイクルRSX用最大出力値
            cycle_rsx_min_output: サイクルRSX用最小出力値
            cycle_rsx_src_type: サイクルRSX用ソースタイプ
        """
        # 有効なmultiplier_methodをチェック
        if multiplier_method not in ['adaptive', 'simple', 'simple_adjustment']:
            self.logger.warning(f"無効なmultiplier_method: {multiplier_method}。'adaptive'を使用します。")
            multiplier_method = 'adaptive'
        
        # 有効なmultiplier_sourceをチェック
        if multiplier_source not in ['cer', 'x_trend', 'z_trend']:
            self.logger.warning(f"無効なmultiplier_source: {multiplier_source}。'cer'を使用します。")
            multiplier_source = 'cer'
        
        # 有効なma_sourceをチェック
        if ma_source not in ['cer', 'x_trend']:
            self.logger.warning(f"無効なma_source: {ma_source}。'cer'を使用します。")
            ma_source = 'cer'
        
        # 有効なmultiplier_smoothing_methodをチェック
        if multiplier_smoothing_method not in ['none', 'alma', 'hma', 'hyper', 'ema']:
            self.logger.warning(f"無効なmultiplier_smoothing_method: {multiplier_smoothing_method}。'none'を使用します。")
            multiplier_smoothing_method = 'none'
        
        super().__init__(f"ZAdaptiveChannel({multiplier_method},{multiplier_source},{ma_source},{max_max_multiplier},{cycle_part},{src_type})")
        
        # パラメータの保存
        self.max_max_multiplier = max_max_multiplier
        self.min_max_multiplier = min_max_multiplier
        self.max_min_multiplier = max_min_multiplier
        self.min_min_multiplier = min_min_multiplier
        self.src_type = src_type
        self.multiplier_source = multiplier_source
        self.ma_source = ma_source
        self.multiplier_method = multiplier_method

        
        self.use_x_trend_adjustment = use_x_trend_adjustment
        
        # 乗数平滑化パラメータ
        self.multiplier_smoothing_method = multiplier_smoothing_method
        self.multiplier_smoothing_period = multiplier_smoothing_period
        self.alma_offset = alma_offset
        self.alma_sigma = alma_sigma
        
        # 依存オブジェクトの初期化
        # 1. CycleEfficiencyRatio (常に初期化)
        self.cycle_er = CycleEfficiencyRatio(
            detector_type=detector_type,
            cycle_part=cycle_part,
            lp_period=lp_period,
            hp_period=hp_period,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            use_kalman_filter=use_kalman_filter,
            src_type=src_type
        )
        
        # 2. Xトレンドインデックス (multiplier_source='x_trend'またはma_source='x_trend'またはmultiplier_method='simple'またはuse_x_trend_adjustmentの場合)
        self.x_trend_index = None
        if multiplier_source == 'x_trend' or ma_source == 'x_trend' or multiplier_method == 'simple' or use_x_trend_adjustment:
            self.x_trend_index = XTrendIndex(
                detector_type=x_detector_type,
                cycle_part=x_cycle_part,
                max_cycle=x_max_cycle,
                min_cycle=x_min_cycle,
                max_output=x_max_output,
                min_output=x_min_output,
                src_type=src_type,
                lp_period=lp_period,
                hp_period=hp_period,
                smoother_type=x_smoother_type,
                fixed_threshold=fixed_threshold
            )
        
        # 3. Zアダプティブトレンドインデックス (multiplier_source='z_trend'の場合)
        self.z_trend_index = None
        if multiplier_source == 'z_trend':
            self.z_trend_index = ZAdaptiveTrendIndex(
                detector_type=x_detector_type,
                cycle_part=x_cycle_part,
                max_cycle=x_max_cycle,
                min_cycle=x_min_cycle,
                max_output=x_max_output,
                min_output=x_min_output,
                src_type=src_type,
                lp_period=lp_period,
                hp_period=hp_period,
                smoother_type=x_smoother_type,
                # CER パラメータ
                cer_detector_type=detector_type,
                cer_lp_period=lp_period,
                cer_hp_period=hp_period,
                cer_cycle_part=cycle_part,
                cer_max_cycle=max_cycle,
                cer_min_cycle=min_cycle,
                cer_max_output=max_output,
                cer_min_output=min_output,
                cer_src_type=src_type,
                use_kalman_filter=use_kalman_filter,

            )
        
 
        
        # 6. ZAdaptiveMA
        self._z_adaptive_ma = ZAdaptiveMA(fast_period=fast_period, slow_period=slow_period)
        
        # 7. CATR
        self._c_atr = CATR()
        
        # 8. 乗数平滑化用インジケーター（選択された方法のみ初期化）
        self._multiplier_smoother_upper = None
        self._multiplier_smoother_lower = None
        
        if self.multiplier_smoothing_method == 'alma':
            self._multiplier_smoother_upper = ALMA(
                period=self.multiplier_smoothing_period,
                offset=self.alma_offset,
                sigma=self.alma_sigma,
                src_type='close'  # 乗数値は1次元配列なので'close'として扱う
            )
            self._multiplier_smoother_lower = ALMA(
                period=self.multiplier_smoothing_period,
                offset=self.alma_offset,
                sigma=self.alma_sigma,
                src_type='close'
            )
        elif self.multiplier_smoothing_method == 'hma':
            self._multiplier_smoother_upper = HMA(
                period=self.multiplier_smoothing_period,
                src_type='close'
            )
            self._multiplier_smoother_lower = HMA(
                period=self.multiplier_smoothing_period,
                src_type='close'
            )
        # hyper と ema は関数ベースなので初期化不要
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 10  # キャッシュの最大サイズ
        self._cache_keys = []  # キャッシュキーの順序管理用
    
    def _smooth_multipliers(self, upper_multiplier: np.ndarray, lower_multiplier: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        選択された方法で乗数を平滑化する
        
        Args:
            upper_multiplier: アッパーバンド乗数
            lower_multiplier: ロワーバンド乗数
        
        Returns:
            平滑化されたアッパーバンド乗数とロワーバンド乗数のタプル
        """
        if self.multiplier_smoothing_method == 'none':
            return upper_multiplier.copy(), lower_multiplier.copy()
        
        try:
            if self.multiplier_smoothing_method == 'alma':
                if self._multiplier_smoother_upper is not None and self._multiplier_smoother_lower is not None:
                    upper_smoothed = self._multiplier_smoother_upper.calculate(upper_multiplier)
                    lower_smoothed = self._multiplier_smoother_lower.calculate(lower_multiplier)
                    return upper_smoothed, lower_smoothed
                    
            elif self.multiplier_smoothing_method == 'hma':
                if self._multiplier_smoother_upper is not None and self._multiplier_smoother_lower is not None:
                    upper_smoothed = self._multiplier_smoother_upper.calculate(upper_multiplier)
                    lower_smoothed = self._multiplier_smoother_lower.calculate(lower_multiplier)
                    return upper_smoothed, lower_smoothed
                    
            elif self.multiplier_smoothing_method == 'hyper':
                upper_smoothed = hyper_smoother(upper_multiplier, self.multiplier_smoothing_period)
                lower_smoothed = hyper_smoother(lower_multiplier, self.multiplier_smoothing_period)
                return upper_smoothed, lower_smoothed
                
            elif self.multiplier_smoothing_method == 'ema':
                upper_smoothed = calculate_ema_numba(upper_multiplier, self.multiplier_smoothing_period)
                lower_smoothed = calculate_ema_numba(lower_multiplier, self.multiplier_smoothing_period)
                return upper_smoothed, lower_smoothed
                
        except Exception as e:
            self.logger.warning(f"乗数平滑化中にエラー: {str(e)}。元の乗数を使用します。")
        
        # エラー時またはサポートされていない方法の場合は元の値を返す
        return upper_multiplier.copy(), lower_multiplier.copy()
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を生成（高速化版）
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        # DataFrameの場合はサイズと最初と最後の値のみを使用
        if isinstance(data, pd.DataFrame):
            shape = data.shape
            # 最初と最後の10行のみ使用（大きなデータセットの場合も高速）
            if len(data) > 20:
                first_last = (
                    tuple(data.iloc[0].values) + 
                    tuple(data.iloc[-1].values) +
                    (data.shape[0],)  # データの長さも含める
                )
            else:
                # 小さなデータセットはすべて使用
                first_last = tuple(data.values.flatten()[-20:])
        else:
            shape = data.shape
            # NumPy配列も同様
            if len(data) > 20:
                if data.ndim > 1:
                    first_last = tuple(data[0]) + tuple(data[-1]) + (data.shape[0],)
                else:
                    first_last = (data[0], data[-1], data.shape[0])
            else:
                first_last = tuple(data.flatten()[-20:])
            
        # パラメータとサイズ、データのサンプルを組み合わせたハッシュを返す
        params_str = (
            f"{self.src_type}_{self.max_max_multiplier}_{self.min_max_multiplier}_"
            f"{self.max_min_multiplier}_{self.min_min_multiplier}_{self.multiplier_method}_"
            f"{self.multiplier_source}_{self.ma_source}_"
            f"{self.multiplier_smoothing_method}_{self.multiplier_smoothing_period}"
        )
        
        return f"{params_str}_{hash(first_last + (shape,))}"
    
    def _calculate_trigger_values(self, data) -> np.ndarray:
        """
        選択されたソースに基づいてトリガー値を計算
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 0-1の範囲のトリガー値
        """
        if self.multiplier_source == 'cer':
            # CERの場合は絶対値を取って0-1に正規化
            raw_er = self.cycle_er.calculate(data)
            trigger_values = np.abs(raw_er)
            return trigger_values
            
        elif self.multiplier_source == 'x_trend':
            # Xトレンドインデックスは既に0-1の範囲なのでそのまま使用
            result = self.x_trend_index.calculate(data)
            return result.values
            
        elif self.multiplier_source == 'z_trend':
            # Zアダプティブトレンドインデックスも既に0-1の範囲なのでそのまま使用
            result = self.z_trend_index.calculate(data)
            return result.values
            
        else:
            # デフォルトはCER
            raw_er = self.cycle_er.calculate(data)
            trigger_values = np.abs(raw_er)
            return trigger_values
    
    def _calculate_ma_source_values(self, data) -> np.ndarray:
        """
        ZAdaptiveMAに渡すソース値を計算
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: MAソース値
        """
        if self.ma_source == 'cer':
            # CERの場合はそのまま返す（生の値）
            return self.cycle_er.calculate(data)
            
        elif self.ma_source == 'x_trend':
            # Xトレンドインデックスの場合
            if self.x_trend_index is None:
                # フォールバックとしてCERを使用
                self.logger.warning("XTrendIndexが初期化されていません。CERを使用します。")
                return self.cycle_er.calculate(data)
            result = self.x_trend_index.calculate(data)
            # XTrendIndexの値は0-1の範囲なので、そのまま返す
            return result.values
            
        else:
            # デフォルトはCER
            return self.cycle_er.calculate(data)

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Zアダプティブチャネルを計算（高速化版）
        
        Args:
            data: DataFrame または numpy 配列
        
        Returns:
            np.ndarray: 中心線（ZAdaptiveMA）の値
        """
        try:
            # データハッシュを計算して、キャッシュが有効かどうかを確認
            data_hash = self._get_data_hash(data)
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新（最も新しく使われたキーを最後に）
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                return self._result_cache[data_hash].middle
            
            # 1. 効率比の計算（CER）- 常に計算（他の計算で使用される可能性があるため）
            er = self.cycle_er.calculate(data)
            
            # 2. トリガー値の計算（乗数計算用）
            trigger_values = self._calculate_trigger_values(data)
            
            # 3. ZAdaptiveMAのソース値を計算
            ma_source_values = self._calculate_ma_source_values(data)
            
            # 4. 動的な乗数値の計算（並列化・ベクトル化版）
            if self.multiplier_method == 'simple':
                # 4a. 新しいシンプルな動的乗数の計算: 15 - (CER*10) - (XTRENDINDEX*10)
                if self.x_trend_index is None:
                    raise ValueError("シンプル方式にはXTRENDINDEXが必要ですが、初期化されていません。")
                
                # XTRENDINDEXの値を取得
                x_trend_result = self.x_trend_index.calculate(data)
                x_trend_values = x_trend_result.values
                
                # 新しい計算式を使用
                dynamic_multiplier = calculate_new_simple_dynamic_multiplier_vec(er, x_trend_values)
                
                # シンプル計算の場合、max_mult_valuesとmin_mult_valuesは使用しないが、
                # 結果の一貫性のため、動的乗数と同じ値で初期化
                max_mult_values = dynamic_multiplier.copy()
                min_mult_values = dynamic_multiplier.copy()
                
            elif self.multiplier_method == 'simple_adjustment':
                # 4b. シンプルアジャストメント動的乗数の計算: MAX=8からMIN=0.5まで線形補間
                # シンプルアジャストメント計算式を使用
                dynamic_multiplier = calculate_simple_adjustment_multiplier_vec(trigger_values)
                
                # シンプルアジャストメント計算の場合、max_mult_valuesとmin_mult_valuesは使用しないが、
                # 結果の一貫性のため、動的乗数と同じ値で初期化
                max_mult_values = dynamic_multiplier.copy()
                min_mult_values = dynamic_multiplier.copy()
                
            else:
                # 4c. 従来のアダプティブ動的乗数の計算
                # 4.1. 動的な最大乗数の計算
                max_mult_values = calculate_dynamic_max_multiplier(
                    trigger_values, 
                    self.max_max_multiplier, 
                    self.min_max_multiplier
                )
                
                # 4.2. 動的な最小乗数の計算
                min_mult_values = calculate_dynamic_min_multiplier(
                    trigger_values, 
                    self.max_min_multiplier, 
                    self.min_min_multiplier
                )
                
                # 4.3. 線形計算を使用して最終的な動的乗数を計算
                dynamic_multiplier = calculate_dynamic_multiplier_vec(trigger_values, max_mult_values, min_mult_values)

            # デフォルトの乗数を設定 (エラー修正のため追加)
            upper_multiplier = dynamic_multiplier.copy()
            lower_multiplier = dynamic_multiplier.copy()
            
            
            # 7. X-Trend Indexの計算と乗数調整（有効な場合）
            x_trend_values = np.full_like(dynamic_multiplier, np.nan)
            x_trend_upper_multiplier = upper_multiplier.copy()
            x_trend_lower_multiplier = lower_multiplier.copy()
            
            if self.use_x_trend_adjustment and self.x_trend_index is not None:
                try:
                    # X-Trend Indexを計算
                    x_trend_result = self.x_trend_index.calculate(data)
                    x_trend_values = x_trend_result.values
                    
                    # X-Trend Indexに基づく乗数調整を適用
                    x_trend_upper_multiplier, x_trend_lower_multiplier = adjust_multipliers_with_x_trend_index(
                        upper_multiplier,
                        lower_multiplier,
                        x_trend_values
                    )
                    
                    # 最終的な乗数として更新
                    upper_multiplier = x_trend_upper_multiplier
                    lower_multiplier = x_trend_lower_multiplier
                    
                except Exception as e:
                    self.logger.warning(f"X-Trend Index計算中にエラー: {str(e)}。基本乗数を使用します。")
            
            # 8. 乗数の平滑化
            upper_multiplier_smoothed, lower_multiplier_smoothed = self._smooth_multipliers(
                upper_multiplier, lower_multiplier
            )
            
            # 9. ZAdaptiveMAの計算（中心線）- 選択されたソースを使用
            z_ma = self._z_adaptive_ma.calculate(data, ma_source_values)
            
            # 10. CATRの計算（external_erパラメータは不要になった）
            self._c_atr.calculate(data)
            
            # 金額ベースのCATRを取得 - 重要: バンド計算には金額ベースのATRを使用する
            z_atr = self._c_atr.get_absolute_atr()
            
            # 11. Zアダプティブチャネルの計算（平滑化された乗数を使用）
            middle, upper, lower = calculate_z_adaptive_channel_with_roc_persistence(
                z_ma,
                z_atr,
                upper_multiplier_smoothed,
                lower_multiplier_smoothed
            )
            
            # 結果をキャッシュ
            result = ZAdaptiveChannelResult(
                middle=middle,
                upper=upper,
                lower=lower,
                er=er,
                dynamic_multiplier=dynamic_multiplier,
                z_atr=z_atr,
                max_mult_values=max_mult_values,
                min_mult_values=min_mult_values,
                multiplier_trigger=trigger_values,
                # X-Trend Index関連の結果を追加
                x_trend_values=x_trend_values,
                x_trend_upper_multiplier=x_trend_upper_multiplier,
                x_trend_lower_multiplier=x_trend_lower_multiplier,
                # 乗数平滑化結果を追加
                upper_multiplier_smoothed=upper_multiplier_smoothed,
                lower_multiplier_smoothed=lower_multiplier_smoothed
            )
            
            # キャッシュサイズ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            return middle  # 元のインターフェイスと互換性を保つため中心線を返す
            
        except Exception as e:
            self.logger.error(f"Zアダプティブチャネル計算中にエラー: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return np.array([])
    
    def get_cycle_rsx_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクルRSXの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクルRSXの値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result.cycle_rsx_values
        except Exception as e:
            self.logger.error(f"サイクルRSX値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_cycle_rsx_upper_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクルRSX調整されたアッパーバンド乗数を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクルRSX調整されたアッパーバンド乗数の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result.cycle_rsx_upper_multiplier
        except Exception as e:
            self.logger.error(f"サイクルRSXアッパーバンド乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_cycle_rsx_lower_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクルRSX調整されたロワーバンド乗数を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクルRSX調整されたロワーバンド乗数の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result.cycle_rsx_lower_multiplier
        except Exception as e:
            self.logger.error(f"サイクルRSXロワーバンド乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_x_trend_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X-Trend Indexの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X-Trend Indexの値
        """
        if data is not None:
            self.calculate(data)
        
        data_hash = self._get_data_hash(data) if data is not None else (list(self._result_cache.keys())[-1] if self._result_cache else None)
        
        if data_hash and data_hash in self._result_cache:
            result = self._result_cache[data_hash]
            return result.x_trend_values
        else:
            return np.array([])
    
    def get_x_trend_upper_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X-Trend Index調整されたアッパーバンド乗数を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X-Trend Index調整されたアッパーバンド乗数
        """
        if data is not None:
            self.calculate(data)
        
        data_hash = self._get_data_hash(data) if data is not None else (list(self._result_cache.keys())[-1] if self._result_cache else None)
        
        if data_hash and data_hash in self._result_cache:
            result = self._result_cache[data_hash]
            return result.x_trend_upper_multiplier
        else:
            return np.array([])
    
    def get_x_trend_lower_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X-Trend Index調整されたロワーバンド乗数を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X-Trend Index調整されたロワーバンド乗数
        """
        if data is not None:
            self.calculate(data)
        
        data_hash = self._get_data_hash(data) if data is not None else (list(self._result_cache.keys())[-1] if self._result_cache else None)
        
        if data_hash and data_hash in self._result_cache:
            result = self._result_cache[data_hash]
            return result.x_trend_lower_multiplier
        else:
            return np.array([])
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        バンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([]), np.array([]), np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.middle, result.upper, result.lower
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            return np.array([]), np.array([]), np.array([])
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        効率比（CER）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 効率比の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.er
        except Exception as e:
            self.logger.error(f"効率比取得中にエラー: {str(e)}")
            return np.array([])
    
    # 後方互換性のため
    def get_cycle_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        効率比（CER）の値を取得（後方互換性のため）
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 効率比の値
        """
        return self.get_efficiency_ratio(data)
    
    def get_multiplier_trigger(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        乗数計算に使用されたトリガー値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トリガー値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.multiplier_trigger
        except Exception as e:
            self.logger.error(f"トリガー値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.dynamic_multiplier
        except Exception as e:
            self.logger.error(f"動的乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_z_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CATR値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CATR値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.z_atr
        except Exception as e:
            self.logger.error(f"ZATR取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_max_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的最大乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的最大乗数の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.max_mult_values
        except Exception as e:
            self.logger.error(f"動的最大乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_min_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的最小乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的最小乗数の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.min_mult_values
        except Exception as e:
            self.logger.error(f"動的最小乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_upper_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        調整されたアッパーバンド乗数を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 調整されたアッパーバンド乗数の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result.upper_multiplier
        except Exception as e:
            self.logger.error(f"アッパーバンド乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_lower_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        調整されたロワーバンド乗数を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 調整されたロワーバンド乗数の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result.lower_multiplier
        except Exception as e:
            self.logger.error(f"ロワーバンド乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_roc_persistence_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ROC継続性の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ROC継続性の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result.roc_persistence_values
        except Exception as e:
            self.logger.error(f"ROC継続性値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_roc_directions(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ROC方向を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ROC方向（1=正、-1=負、0=ゼロ）
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result.roc_directions
        except Exception as e:
            self.logger.error(f"ROC方向取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_smoothed_multipliers(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        平滑化された乗数を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (平滑化されたアッパーバンド乗数, 平滑化されたロワーバンド乗数)のタプル
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([]), np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result.upper_multiplier_smoothed, result.lower_multiplier_smoothed
        except Exception as e:
            self.logger.error(f"平滑化された乗数取得中にエラー: {str(e)}")
            return np.array([]), np.array([])
    
    def get_upper_multiplier_smoothed(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        平滑化されたアッパーバンド乗数を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 平滑化されたアッパーバンド乗数の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result.upper_multiplier_smoothed
        except Exception as e:
            self.logger.error(f"平滑化アッパーバンド乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_lower_multiplier_smoothed(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        平滑化されたロワーバンド乗数を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 平滑化されたロワーバンド乗数の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                result = next(iter(self._result_cache.values()))
                
            return result.lower_multiplier_smoothed
        except Exception as e:
            self.logger.error(f"平滑化ロワーバンド乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def reset(self) -> None:
        """
        状態をリセット
        """
        # キャッシュをクリア
        self._result_cache = {}
        self._cache_keys = []
        
        # 依存オブジェクトもリセット
        self.cycle_er.reset()
        if self.x_trend_index is not None:
            self.x_trend_index.reset()
        if self.z_trend_index is not None:
            self.z_trend_index.reset()
        self._z_adaptive_ma.reset()
        self._c_atr.reset()
        if self.roc_persistence is not None:
            self.roc_persistence.reset()
        if self.cycle_rsx is not None:
            self.cycle_rsx.reset()
        
        # 乗数平滑化インジケーターもリセット
        if self._multiplier_smoother_upper is not None:
            self._multiplier_smoother_upper.reset()
        if self._multiplier_smoother_lower is not None:
            self._multiplier_smoother_lower.reset() 