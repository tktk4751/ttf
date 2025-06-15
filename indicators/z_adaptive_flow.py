#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Optional, Literal
import numpy as np
import pandas as pd
from numba import jit, prange, njit, float64, int64, boolean

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    # from .adaptive_period import AdaptivePeriod  # 固定乗数を使用するため不要
    from .hma import HMA
    from .alma import ALMA
    from .z_adaptive_ma import ZAdaptiveMA
    from .zlema import ZLEMA
    from .volatility import volatility
    from .atr import ATR
    from .efficiency_ratio import EfficiencyRatio
except ImportError:
    # Fallback for potential execution context issues
    print("Warning: Could not import from relative path. Assuming base classes are available.")
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class PriceSource:
        @staticmethod
        def calculate_source(data, src_type):
            if isinstance(data, pd.DataFrame):
                if src_type == 'close': return data['close'].values
                elif src_type == 'high': return data['high'].values
                elif src_type == 'low': return data['low'].values
                elif src_type == 'hlc3': return ((data['high'] + data['low'] + data['close']) / 3).values
                else: return data['close'].values
            else:
                return data[:, 3] if data.ndim > 1 and data.shape[1] > 3 else data


@dataclass
class ZAdaptiveFlowResult:
    """Z Adaptive Flowの計算結果"""
    basis: np.ndarray                    # 基準線（fast_ma + slow_ma) / 2
    level: np.ndarray                    # トレンドレベル（上下切り替わる線）
    upper: np.ndarray                    # 上限バンド
    lower: np.ndarray                    # 下限バンド
    trend_state: np.ndarray              # 1=bullish, -1=bearish
    volatility: np.ndarray               # 使用されたボラティリティ
    slow_multiplier: np.ndarray          # スロー期間乗数（動的適応）
    volatility_multiplier: np.ndarray    # ボラティリティ乗数（動的適応）
    fast_ma: np.ndarray                  # ファストMA
    slow_ma: np.ndarray                  # スローMA
    long_signals: np.ndarray             # ロングシグナル
    short_signals: np.ndarray            # ショートシグナル
    smoothed_volatility: np.ndarray      # 平滑化されたボラティリティ


@njit(fastmath=True, cache=True)
def calculate_ema_numba(data: np.ndarray, period: int) -> np.ndarray:
    """
    指数移動平均（EMA）を計算する（Numba最適化版、NaN対応強化）
    
    Args:
        data: 入力データ配列
        period: EMA期間
    
    Returns:
        EMA値の配列
    """
    length = len(data)
    result = np.full(length, np.nan)
    
    if length == 0 or period < 1:
        return result
    
    # 期間の調整
    effective_period = min(period, length)
    
    # アルファ値の計算
    alpha = 2.0 / (effective_period + 1.0)
    
    # 最初の有効値を見つけてEMAの初期値に設定
    start_idx = 0
    while start_idx < length and (np.isnan(data[start_idx]) or data[start_idx] == 0.0):
        start_idx += 1
    
    if start_idx >= length:
        # 全ての値が無効な場合、0で初期化を試行
        start_idx = 0
        initial_value = 1e-8  # 極小値で初期化
    else:
        initial_value = data[start_idx]
    
    # 初期値を設定
    result[start_idx] = initial_value
    
    # EMA計算（改良版）
    for i in range(start_idx + 1, length):
        current_value = data[i]
        
        # NaNまたは0の場合の処理
        if np.isnan(current_value) or current_value == 0.0:
            if not np.isnan(result[i-1]):
                # 前の値を保持
                result[i] = result[i-1]
            else:
                result[i] = 1e-8  # 極小値
        else:
            if not np.isnan(result[i-1]):
                # 通常のEMA計算
                result[i] = alpha * current_value + (1.0 - alpha) * result[i-1]
            else:
                # 前の値がNaNの場合は現在の値で開始
                result[i] = current_value
    
    # 結果の後処理：0になった値を極小値に置換
    for i in range(length):
        if result[i] == 0.0:
            result[i] = 1e-8
    
    return result


@njit(fastmath=True, cache=True)
def calculate_robust_sma_numba(data: np.ndarray, period: int) -> np.ndarray:
    """
    堅牢な単純移動平均を計算する（Numba最適化版、NaN対応）
    
    Args:
        data: 入力データ配列
        period: 期間
    
    Returns:
        SMA値の配列
    """
    length = len(data)
    result = np.full(length, np.nan)
    
    if length == 0 or period < 1:
        return result
    
    for i in range(length):
        # 計算範囲を決定
        start_idx = max(0, i - period + 1)
        end_idx = i + 1
        
        # ウィンドウ内の有効な値を集計
        sum_val = 0.0
        count = 0
        
        for j in range(start_idx, end_idx):
            if not np.isnan(data[j]) and data[j] > 0.0:
                sum_val += data[j]
                count += 1
        
        if count > 0:
            result[i] = sum_val / count
        elif i > 0 and not np.isnan(result[i-1]):
            # 有効な値がない場合は前の値を保持
            result[i] = result[i-1]
        else:
            result[i] = 1e-8  # 極小値
    
    return result


@njit(fastmath=True, parallel=False, cache=True)
def calculate_trend_state_and_level(
    close_prices: np.ndarray,
    upper_band: np.ndarray,
    lower_band: np.ndarray,
    basis: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pine Scriptの get_trend_state ロジックを実装
    トレンド状態とレベルを計算する
    
    Args:
        close_prices: 終値配列
        upper_band: 上限バンド
        lower_band: 下限バンド
        basis: 基準線
    
    Returns:
        trend_state, level のタプル
    """
    length = len(close_prices)
    trend_state = np.zeros(length, dtype=np.int8)
    level = np.full(length, np.nan)
    
    if length == 0:
        return trend_state, level
    
    # 最初の有効なデータポイントを見つける
    first_valid_idx = -1
    for i in range(length):
        if (not np.isnan(close_prices[i]) and 
            not np.isnan(upper_band[i]) and 
            not np.isnan(lower_band[i]) and 
            not np.isnan(basis[i])):
            first_valid_idx = i
            break
    
    # 有効なデータが見つからない場合の処理を改善
    if first_valid_idx == -1:
        # まず、バンドとベーシスを使った推定値で初期化を試行
        default_trend = -1  # デフォルトは下降トレンド
        
        for i in range(length):
            trend_state[i] = default_trend
            
            # レベル値の設定（優先順位付き）
            if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]):
                # 両方のバンドが有効な場合、デフォルトトレンドに応じて設定
                level[i] = upper_band[i] if default_trend == -1 else lower_band[i]
            elif not np.isnan(upper_band[i]):
                level[i] = upper_band[i]
            elif not np.isnan(lower_band[i]):
                level[i] = lower_band[i]
            elif not np.isnan(basis[i]):
                level[i] = basis[i]
            elif not np.isnan(close_prices[i]):
                level[i] = close_prices[i]
            else:
                level[i] = 0.0  # 最終フォールバック
        
        return trend_state, level
    
    # 初期状態の設定（有効なデータが見つかった場合）
    first_close = close_prices[first_valid_idx]
    first_basis = basis[first_valid_idx]
    first_upper = upper_band[first_valid_idx]
    first_lower = lower_band[first_valid_idx]
    
    # 初期トレンド判定
    current_trend = 1 if first_close > first_basis else -1
    
    # 初期レベル設定
    if current_trend == 1:
        current_level = first_lower  # 上昇トレンド時は下限バンドがサポート
    else:
        current_level = first_upper  # 下降トレンド時は上限バンドがレジスタンス
    
    # 最初の有効インデックスまでの値を埋める
    for i in range(first_valid_idx + 1):
        trend_state[i] = current_trend
        if i < first_valid_idx:
            # 有効インデックス前の値は推定値で埋める
            if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]):
                level[i] = upper_band[i] if current_trend == -1 else lower_band[i]
            elif not np.isnan(upper_band[i]):
                level[i] = upper_band[i]
            elif not np.isnan(lower_band[i]):
                level[i] = lower_band[i]
            elif not np.isnan(basis[i]):
                level[i] = basis[i]
            else:
                level[i] = current_level  # 計算済みの初期レベル
        else:
            level[i] = current_level
    
    # メインループ - 有効インデックス以降を処理
    for i in range(first_valid_idx + 1, length):
        current_close = close_prices[i]
        current_upper = upper_band[i]
        current_lower = lower_band[i]
        current_basis = basis[i]
        
        # データ有効性チェック
        if (np.isnan(current_close) or 
            np.isnan(current_upper) or 
            np.isnan(current_lower) or 
            np.isnan(current_basis)):
            # NaN値の場合は前の値を保持し、可能な限り推定値を設定
            trend_state[i] = current_trend
            
            # レベル値の設定（有効な値がある場合は使用）
            if not np.isnan(current_upper) and not np.isnan(current_lower):
                level[i] = current_upper if current_trend == -1 else current_lower
            elif not np.isnan(current_upper):
                level[i] = current_upper
            elif not np.isnan(current_lower):
                level[i] = current_lower
            elif not np.isnan(current_basis):
                level[i] = current_basis
            else:
                level[i] = current_level  # 前の値を保持
            continue
        
        # Pine Scriptロジック: トレンド状態とレベルの更新
        if current_trend == 1:  # 上昇トレンド中
            if current_close <= current_level:
                # 価格がサポートレベルを下抜け → 下降トレンドに転換
                current_trend = -1
                current_level = current_upper
            else:
                # 上昇トレンド継続 → サポートレベルを上方修正
                current_level = max(current_level, current_lower)
        
        else:  # 下降トレンド中 (current_trend == -1)
            if current_close >= current_level:
                # 価格がレジスタンスレベルを上抜け → 上昇トレンドに転換
                current_trend = 1
                current_level = current_lower
            else:
                # 下降トレンド継続 → レジスタンスレベルを下方修正
                current_level = min(current_level, current_upper)
        
        # 結果を配列に格納
        trend_state[i] = current_trend
        level[i] = current_level
    
    # 最終チェック：Level配列にNaNが残っていないか確認と修正
    for i in range(length):
        if np.isnan(level[i]):
            # フォールバック値を設定（優先順位付き）
            if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]):
                level[i] = upper_band[i] if trend_state[i] == -1 else lower_band[i]
            elif not np.isnan(upper_band[i]):
                level[i] = upper_band[i]
            elif not np.isnan(lower_band[i]):
                level[i] = lower_band[i]
            elif not np.isnan(basis[i]):
                level[i] = basis[i]
            elif not np.isnan(close_prices[i]):
                level[i] = close_prices[i]
            else:
                # 前後の有効な値を探す
                found_fallback = False
                for j in range(max(0, i-5), min(length, i+6)):
                    if not np.isnan(level[j]):
                        level[i] = level[j]
                        found_fallback = True
                        break
                if not found_fallback:
                    level[i] = 0.0  # 最終フォールバック
    
    return trend_state, level


@njit(fastmath=True, parallel=True, cache=True)
def detect_signals(trend_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    トレンド変化に基づいてシグナルを検出
    
    Args:
        trend_state: トレンド状態配列 (1=bullish, -1=bearish)
    
    Returns:
        long_signals, short_signals のタプル（ブール配列）
    """
    length = len(trend_state)
    long_signals = np.zeros(length, dtype=boolean)
    short_signals = np.zeros(length, dtype=boolean)
    
    if length <= 1:
        return long_signals, short_signals
    
    for i in range(1, length):
        # ロングシグナル: -1から1への変化
        if trend_state[i] == 1 and trend_state[i-1] == -1:
            long_signals[i] = True
        
        # ショートシグナル: 1から-1への変化
        if trend_state[i] == -1 and trend_state[i-1] == 1:
            short_signals[i] = True
    
    return long_signals, short_signals


@njit(fastmath=True, parallel=True, cache=True)
def apply_dynamic_slow_multiplier(
    base_period: int,
    multiplier_values: np.ndarray
) -> np.ndarray:
    """
    動的スロー期間乗数を適用
    
    Args:
        base_period: ベース期間
        multiplier_values: 乗数配列（2-5の範囲）
    
    Returns:
        動的スロー期間配列
    """
    result = np.empty_like(multiplier_values)
    
    for i in prange(len(multiplier_values)):
        if np.isnan(multiplier_values[i]):
            result[i] = float(base_period * 2)  # デフォルト乗数2
        else:
            # 2-5の範囲にクリップ
            clipped_mult = max(2.0, min(5.0, multiplier_values[i]))
            result[i] = float(base_period) * clipped_mult
    
    return result


def calculate_trend_state_and_level_fallback(
    close_prices: np.ndarray,
    upper_band: np.ndarray,
    lower_band: np.ndarray,
    basis: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    トレンド状態とレベルを計算する（Numba無しバックアップ版）
    
    Args:
        close_prices: 終値配列
        upper_band: 上限バンド
        lower_band: 下限バンド
        basis: 基準線
    
    Returns:
        trend_state, level のタプル
    """
    length = len(close_prices)
    trend_state = np.zeros(length, dtype=np.int8)
    level = np.full(length, np.nan, dtype=np.float64)
    
    if length == 0:
        return trend_state, level
    
    # 最初の有効なデータポイントを見つける
    first_valid_idx = -1
    for i in range(length):
        if (not np.isnan(close_prices[i]) and 
            not np.isnan(upper_band[i]) and 
            not np.isnan(lower_band[i]) and 
            not np.isnan(basis[i])):
            first_valid_idx = i
            break
    
    # print(f"デバッグ（フォールバック関数）: first_valid_idx = {first_valid_idx}")
    
    if first_valid_idx == -1:
        # 有効なデータが見つからない場合は部分的に使用可能なデータで補完
        # print("デバッグ（フォールバック関数）: 有効なデータが見つからないため、部分補完を実行")
        default_trend = -1
        
        for i in range(length):
            trend_state[i] = default_trend
            
            # レベル値の設定（優先順位付き）
            if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]):
                level[i] = upper_band[i] if default_trend == -1 else lower_band[i]
            elif not np.isnan(upper_band[i]):
                level[i] = upper_band[i]
            elif not np.isnan(lower_band[i]):
                level[i] = lower_band[i]
            elif not np.isnan(basis[i]):
                level[i] = basis[i]
            elif not np.isnan(close_prices[i]):
                level[i] = close_prices[i]
            else:
                level[i] = 0.0
        
        return trend_state, level
    
    # 初期状態の設定
    first_close = close_prices[first_valid_idx]
    first_basis = basis[first_valid_idx]
    first_upper = upper_band[first_valid_idx]
    first_lower = lower_band[first_valid_idx]
    
    # 初期トレンド判定
    current_trend = 1 if first_close > first_basis else -1
    current_level = first_lower if current_trend == 1 else first_upper
    
    # print(f"デバッグ（フォールバック関数）: 初期設定 - trend={current_trend}, level={current_level}")
    
    # 最初の有効インデックスまでの値を埋める
    for i in range(first_valid_idx + 1):
        trend_state[i] = current_trend
        if i < first_valid_idx:
            # 有効インデックス前の値は推定値で埋める
            if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]):
                level[i] = upper_band[i] if current_trend == -1 else lower_band[i]
            elif not np.isnan(upper_band[i]):
                level[i] = upper_band[i]
            elif not np.isnan(lower_band[i]):
                level[i] = lower_band[i]
            elif not np.isnan(basis[i]):
                level[i] = basis[i]
            else:
                level[i] = current_level
        else:
            level[i] = current_level
    
    # メインループ
    for i in range(first_valid_idx + 1, length):
        current_close = close_prices[i]
        current_upper = upper_band[i]
        current_lower = lower_band[i]
        current_basis = basis[i]
        
        # データ有効性チェック
        if (np.isnan(current_close) or 
            np.isnan(current_upper) or 
            np.isnan(current_lower) or 
            np.isnan(current_basis)):
            # NaN値の場合は前の値を保持し、可能な限り推定値を設定
            trend_state[i] = current_trend
            
            if not np.isnan(current_upper) and not np.isnan(current_lower):
                level[i] = current_upper if current_trend == -1 else current_lower
            elif not np.isnan(current_upper):
                level[i] = current_upper
            elif not np.isnan(current_lower):
                level[i] = current_lower
            elif not np.isnan(current_basis):
                level[i] = current_basis
            else:
                level[i] = current_level
            continue
        
        # トレンド状態とレベルの更新
        if current_trend == 1:  # 上昇トレンド中
            if current_close <= current_level:
                current_trend = -1
                current_level = current_upper
            else:
                current_level = max(current_level, current_lower)
        else:  # 下降トレンド中
            if current_close >= current_level:
                current_trend = 1
                current_level = current_lower
            else:
                current_level = min(current_level, current_upper)
        
        trend_state[i] = current_trend
        level[i] = current_level
    
    # 最終チェック
    # nan_count = np.isnan(level).sum()
    # print(f"デバッグ（フォールバック関数）: 最終チェック - NaN数: {nan_count}/{length}")
    
    return trend_state, level


class ZAdaptiveFlow(Indicator):
    """
    Z Adaptive Flow インジケーター
    
    Pine ScriptのAdaptive Trend Flowを参考に実装された動的適応型トレンドフォローインジケーター
    
    特徴:
    - 選択可能なMAタイプ（HMA、ALMA、ZAdaptiveMA、ZLEMA）
    - 選択可能なボラティリティ（Volatility、ATR）
    - 固定乗数:
      * スロー期間乗数: 4 (固定)
      * ボラティリティ乗数: 2 (固定)
    - トレンド状態判定とシグナル生成
    - 全インジケーターの初期化引数をサポート
    """
    
    def __init__(
        self,
        # 基本パラメータ
        length: int = 10,                    # メイン期間
        smooth_length: int = 14,             # ボラティリティ平滑化期間
        src_type: str = 'hlc3',             # 価格ソース
        
        # MAタイプ選択
        ma_type: str = 'zlema',               # 'hma', 'alma', 'z_adaptive_ma', 'zlema'
        
        # MA共通パラメータ
        ma_period: Optional[int] = None,           # None の場合 length を使用
        ma_use_dynamic_period: bool = True,        # 動的期間を使用するかどうか
        ma_slope_index: int = 1,                   # スロープインデックス
        ma_range_threshold: float = 0.005,         # range閾値
        
        # ボラティリティタイプ選択
        volatility_type: str = 'volatility', # 'volatility', 'atr'
        
        # 固定乗数パラメータ
        slow_period_multiplier: float = 4.0,     # スロー期間乗数（固定）
        volatility_multiplier: float = 2.0,     # ボラティリティ乗数（固定）
        
        # 共通サイクル検出器パラメータ（全インジケーター共通）
        detector_type: str = 'absolute_ultimate',
        cycle_part: float = 1.0,
        lp_period: int = 10,
        hp_period: int = 48,
        max_cycle: int = 233,
        min_cycle: int = 13,
        max_output: int = 144,
        min_output: int = 13,
        
        # ALMA固有パラメータ（ALMAのみ）
        alma_offset: float = 0.85,
        alma_sigma: float = 6,
        
        # ZAdaptiveMA固有パラメータ（ZAdaptiveMAのみ）
        z_adaptive_ma_fast_period: int = 2,
        z_adaptive_ma_slow_period: int = 144,
        
        # EfficiencyRatio固有パラメータ（ZAdaptiveMA使用時）
        efficiency_ratio_period: int = 5,
        efficiency_ratio_smoothing_method: str = 'hma',
        efficiency_ratio_use_dynamic_period: bool = True,
        efficiency_ratio_slope_index: int = 3,
        efficiency_ratio_range_threshold: float = 0.005,
        efficiency_ratio_smoother_period: int = 13,
        
        # ATR固有パラメータ
        atr_period: Optional[int] = None,           # None の場合 length を使用
        atr_smoothing_method: str = 'alma',         # ATR用平滑化方法
        atr_use_dynamic_period: bool = False,
        atr_slope_index: int = 1,
        atr_range_threshold: float = 0.005,
        
        # volatility固有パラメータ
        volatility_period_mode: str = 'fixed',      # 'adaptive' または 'fixed'
        volatility_fixed_period: Optional[int] = None,  # None の場合 length を使用
        volatility_calculation_mode: str = 'return',    # 'return' または 'price'
        volatility_return_type: str = 'log',            # 'log' または 'simple'
        volatility_ddof: int = 1,                       # 自由度調整
        volatility_smoother_type: str = 'hma',          # 'wilder', 'alma', 'hma', 'zlema', 'none'
        volatility_smoother_period: Optional[int] = None, # None の場合 smooth_length を使用
        
        # 以下のパラメータは互換性のため残すが使用しない
        adaptive_trigger: str = 'chop_trend',
        adaptive_power: float = 1.0,
        adaptive_invert: bool = False,
        adaptive_reverse_mapping: bool = False,
        **trigger_params
    ):
        """
        コンストラクタ
        
        Args:
            length: メイン期間
            smooth_length: ボラティリティ平滑化期間
            src_type: 価格ソース
            ma_type: MAタイプ ('hma', 'alma', 'z_adaptive_ma', 'zlema')
            volatility_type: ボラティリティタイプ ('volatility', 'atr')
            slow_period_multiplier: スロー期間乗数（固定値、デフォルト4.0）
            volatility_multiplier: ボラティリティ乗数（固定値、デフォルト2.0）
            
            # MA共通パラメータ（全MAタイプで使用）
            ma_period: MA用期間（None の場合 length を使用）
            ma_use_dynamic_period: MAで動的期間を使用するかどうか
            ma_slope_index: MA用スロープインデックス
            ma_range_threshold: MA用range閾値
            
            # 共通サイクル検出器パラメータ（全インジケーター共通）
            detector_type: サイクル検出器タイプ
            cycle_part: サイクル検出器のパート
            lp_period: 低周波サイクル検出器の期間
            hp_period: 高周波サイクル検出器の期間
            max_cycle: 最大サイクル
            min_cycle: 最小サイクル
            max_output: 最大出力
            min_output: 最小出力
            
            # ALMA固有パラメータ（ALMAのみ）
            alma_offset: ALMA用オフセット
            alma_sigma: ALMA用シグマ
            
            # ZAdaptiveMA固有パラメータ（ZAdaptiveMAのみ）
            z_adaptive_ma_fast_period: ZAdaptiveMA用ファスト期間
            z_adaptive_ma_slow_period: ZAdaptiveMA用スロー期間
            
            # EfficiencyRatio固有パラメータ（ZAdaptiveMA使用時）
            efficiency_ratio_period: EfficiencyRatio用期間
            efficiency_ratio_smoothing_method: EfficiencyRatio用平滑化方法
            efficiency_ratio_use_dynamic_period: EfficiencyRatioで動的期間を使用するかどうか
            efficiency_ratio_slope_index: EfficiencyRatio用スロープインデックス
            efficiency_ratio_range_threshold: EfficiencyRatio用range閾値
            efficiency_ratio_smoother_period: EfficiencyRatio用平滑化期間
            
            # ATR固有パラメータ
            atr_period: ATRの期間
            atr_smoothing_method: ATR用平滑化方法
            atr_use_dynamic_period: ATRで動的期間を使用するかどうか
            atr_slope_index: ATR用スロープインデックス
            atr_range_threshold: ATR用range閾値
            
            # volatility固有パラメータ
            volatility_period_mode: Volatility用期間モード
            volatility_fixed_period: Volatility用固定期間
            volatility_calculation_mode: Volatility用計算モード ('return': リターンベース, 'price': 価格ベース)
            volatility_return_type: Volatility用リターンタイプ
            volatility_ddof: Volatility用自由度調整
            volatility_smoother_type: Volatility用平滑化タイプ
            volatility_smoother_period: Volatility用平滑化期間
            
            # 以下は互換性のため残すが、内部では使用しない
            adaptive_trigger: 旧AdaptivePeriod用トリガーインジケーター（無視される）
            adaptive_power: 旧AdaptivePeriod用べき乗値（無視される）
            adaptive_invert: 旧AdaptivePeriod用反転フラグ（無視される）
            adaptive_reverse_mapping: 旧AdaptivePeriod用逆マッピングフラグ（無視される）
            **trigger_params: 旧トリガーインジケーター用追加パラメータ（無視される）
        """
        # パラメータ検証
        if ma_type not in ['hma', 'alma', 'z_adaptive_ma', 'zlema']:
            raise ValueError(f"無効なMAタイプ: {ma_type}")
        
        if volatility_type not in ['volatility', 'atr']:
            raise ValueError(f"無効なボラティリティタイプ: {volatility_type}")
        
        super().__init__(f"ZAdaptiveFlow({length},{ma_type},{volatility_type},slow_mult={slow_period_multiplier},vol_mult={volatility_multiplier})")
        
        # パラメータ保存
        self.length = length
        self.smooth_length = smooth_length
        self.src_type = src_type
        self.ma_type = ma_type
        self.volatility_type = volatility_type
        
        # 固定乗数パラメータ
        self.slow_period_multiplier = float(slow_period_multiplier)
        self.volatility_multiplier = float(volatility_multiplier)
        self.slow_period = int(length * self.slow_period_multiplier)  # 固定スロー期間
        
        # 共通サイクル検出器パラメータ（全インジケーター共通）
        self.detector_type = detector_type
        self.cycle_part = cycle_part
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        
        # HMA固有パラメータ
        self.hma_period = ma_period
        self.hma_use_dynamic_period = ma_use_dynamic_period
        self.hma_slope_index = ma_slope_index
        self.hma_range_threshold = ma_range_threshold
        
        # ALMA固有パラメータ
        self.alma_period = ma_period
        self.alma_offset = alma_offset
        self.alma_sigma = alma_sigma
        self.alma_use_dynamic_period = ma_use_dynamic_period
        self.alma_slope_index = ma_slope_index
        self.alma_range_threshold = ma_range_threshold
        
        # ZLEMA固有パラメータ
        self.zlema_period = ma_period
        self.zlema_use_dynamic_period = ma_use_dynamic_period
        self.zlema_slope_index = ma_slope_index
        self.zlema_range_threshold = ma_range_threshold
        
        # ZAdaptiveMA固有パラメータ
        self.z_adaptive_ma_fast_period = z_adaptive_ma_fast_period
        self.z_adaptive_ma_slow_period = z_adaptive_ma_slow_period
        self.z_adaptive_ma_slope_index = ma_slope_index
        self.z_adaptive_ma_range_threshold = ma_range_threshold
        
        # EfficiencyRatio固有パラメータ（ZAdaptiveMA使用時）
        self.efficiency_ratio_period = efficiency_ratio_period
        self.efficiency_ratio_smoothing_method = efficiency_ratio_smoothing_method
        self.efficiency_ratio_use_dynamic_period = efficiency_ratio_use_dynamic_period
        self.efficiency_ratio_slope_index = efficiency_ratio_slope_index
        self.efficiency_ratio_range_threshold = efficiency_ratio_range_threshold
        self.efficiency_ratio_smoother_period = efficiency_ratio_smoother_period
        
        # ATR固有パラメータ
        self.atr_period = atr_period
        self.atr_smoothing_method = atr_smoothing_method
        self.atr_use_dynamic_period = atr_use_dynamic_period
        self.atr_slope_index = atr_slope_index
        self.atr_range_threshold = atr_range_threshold
        
        # volatility固有パラメータ
        self.volatility_period_mode = volatility_period_mode
        self.volatility_fixed_period = volatility_fixed_period
        self.volatility_calculation_mode = volatility_calculation_mode
        self.volatility_return_type = volatility_return_type
        self.volatility_ddof = volatility_ddof
        self.volatility_smoother_type = volatility_smoother_type
        self.volatility_smoother_period = volatility_smoother_period
        
        # 互換性のためのパラメータ（内部では使用しない）
        self.adaptive_trigger = adaptive_trigger
        self.adaptive_power = adaptive_power
        self.adaptive_invert = adaptive_invert
        self.adaptive_reverse_mapping = adaptive_reverse_mapping
        self.trigger_params = trigger_params
        
        # 依存オブジェクト初期化
        self._initialize_dependencies()
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 5
        self._cache_keys = []
    
    def _initialize_dependencies(self):
        """依存オブジェクトの初期化"""
        try:
            # MA インスタンス初期化
            if self.ma_type == 'hma':
                # HMA用パラメータの設定（Noneの場合はlengthを使用）
                hma_fast_period = self.hma_period if self.hma_period is not None else self.length
                hma_slow_period = self.slow_period
                
                self.fast_ma = HMA(
                    period=hma_fast_period,
                    src_type=self.src_type,
                    use_dynamic_period=self.hma_use_dynamic_period,
                    cycle_part=self.cycle_part,
                    detector_type=self.detector_type,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    slope_index=self.hma_slope_index,
                    range_threshold=self.hma_range_threshold,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period
                )
                self.slow_ma = HMA(
                    period=hma_slow_period,
                    src_type=self.src_type,
                    use_dynamic_period=self.hma_use_dynamic_period,
                    cycle_part=self.cycle_part,
                    detector_type=self.detector_type,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    slope_index=self.hma_slope_index,
                    range_threshold=self.hma_range_threshold,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period
                )
            elif self.ma_type == 'alma':
                # ALMA用パラメータの設定（Noneの場合はlengthを使用）
                alma_fast_period = self.alma_period if self.alma_period is not None else self.length
                alma_slow_period = self.slow_period
                
                self.fast_ma = ALMA(
                    period=alma_fast_period,
                    offset=self.alma_offset,
                    sigma=self.alma_sigma,
                    src_type=self.src_type,
                    use_dynamic_period=self.alma_use_dynamic_period,
                    cycle_part=self.cycle_part,
                    detector_type=self.detector_type,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    slope_index=self.alma_slope_index,
                    range_threshold=self.alma_range_threshold,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period
                )
                self.slow_ma = ALMA(
                    period=alma_slow_period,
                    offset=self.alma_offset,
                    sigma=self.alma_sigma,
                    src_type=self.src_type,
                    use_dynamic_period=self.alma_use_dynamic_period,
                    cycle_part=self.cycle_part,
                    detector_type=self.detector_type,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    slope_index=self.alma_slope_index,
                    range_threshold=self.alma_range_threshold,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period
                )
            elif self.ma_type == 'z_adaptive_ma':
                # EfficiencyRatioインスタンスを作成
                self.efficiency_ratio_indicator = EfficiencyRatio(
                    period=self.efficiency_ratio_period,
                    src_type=self.src_type,
                    smoothing_method=self.efficiency_ratio_smoothing_method,
                    use_dynamic_period=self.efficiency_ratio_use_dynamic_period,
                    cycle_part=self.cycle_part,
                    detector_type=self.detector_type,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    slope_index=self.efficiency_ratio_slope_index,
                    range_threshold=self.efficiency_ratio_range_threshold,
                    smoother_period=self.efficiency_ratio_smoother_period,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period
                )
                
                self.fast_ma = ZAdaptiveMA(
                    fast_period=self.z_adaptive_ma_fast_period,
                    slow_period=self.length,
                    src_type=self.src_type,
                    slope_index=self.z_adaptive_ma_slope_index,
                    range_threshold=self.z_adaptive_ma_range_threshold
                )
                self.slow_ma = ZAdaptiveMA(
                    fast_period=self.z_adaptive_ma_fast_period,
                    slow_period=self.slow_period,
                    src_type=self.src_type,
                    slope_index=self.z_adaptive_ma_slope_index,
                    range_threshold=self.z_adaptive_ma_range_threshold
                )
            elif self.ma_type == 'zlema':
                # ZLEMA用パラメータの設定（Noneの場合はlengthを使用）
                zlema_fast_period = self.zlema_period if self.zlema_period is not None else self.length
                zlema_slow_period = self.slow_period
                
                self.fast_ma = ZLEMA(
                    period=zlema_fast_period,
                    src_type=self.src_type,
                    use_dynamic_period=self.zlema_use_dynamic_period,
                    cycle_part=self.cycle_part,
                    detector_type=self.detector_type,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    slope_index=self.zlema_slope_index,
                    range_threshold=self.zlema_range_threshold,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period
                )
                self.slow_ma = ZLEMA(
                    period=zlema_slow_period,
                    src_type=self.src_type,
                    use_dynamic_period=self.zlema_use_dynamic_period,
                    cycle_part=self.cycle_part,
                    detector_type=self.detector_type,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    slope_index=self.zlema_slope_index,
                    range_threshold=self.zlema_range_threshold,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period
                )
            
            # ボラティリティ インスタンス初期化
            if self.volatility_type == 'volatility':
                # volatility用パラメータの設定
                vol_fixed_period = self.volatility_fixed_period if self.volatility_fixed_period is not None else self.length
                vol_smoother_period = self.volatility_smoother_period if self.volatility_smoother_period is not None else self.smooth_length
                
                self.volatility_indicator = volatility(
                    period_mode=self.volatility_period_mode,
                    fixed_period=vol_fixed_period,
                    calculation_mode=self.volatility_calculation_mode,
                    return_type=self.volatility_return_type,
                    ddof=self.volatility_ddof,
                    smoother_type=self.volatility_smoother_type,
                    smoother_period=vol_smoother_period,
                    detector_type=self.detector_type,
                    cycle_part=self.cycle_part,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    src_type=self.src_type
                )
            else:  # 'atr'
                # ATR用パラメータの設定
                atr_period = self.atr_period if self.atr_period is not None else self.length
                
                self.volatility_indicator = ATR(
                    period=atr_period,
                    smoothing_method=self.atr_smoothing_method,
                    use_dynamic_period=self.atr_use_dynamic_period,
                    cycle_part=self.cycle_part,
                    detector_type=self.detector_type,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    slope_index=self.atr_slope_index,
                    range_threshold=self.atr_range_threshold,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period
                )
            
        except Exception as e:
            self.logger.error(f"依存オブジェクト初期化中にエラー: {str(e)}")
            raise
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を生成"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    first_close = float(data.iloc[0].get('close', data.iloc[0, -1]))
                    last_close = float(data.iloc[-1].get('close', data.iloc[-1, -1]))
                    data_signature = (length, first_close, last_close)
                else:
                    data_signature = (0, 0.0, 0.0)
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_val = float(data[0, -1])
                        last_val = float(data[-1, -1])
                    else:
                        first_val = float(data[0])
                        last_val = float(data[-1])
                    data_signature = (length, first_val, last_val)
                else:
                    data_signature = (0, 0.0, 0.0)
            
            params_signature = (
                self.length,
                self.ma_type,
                self.volatility_type,
                self.slow_period_multiplier,
                self.volatility_multiplier
            )
            
            return f"{hash(data_signature)}_{hash(params_signature)}"
            
        except Exception:
            return f"{id(data)}_{self.ma_type}_{self.volatility_type}"
    
    def _calculate_ma_with_dynamic_period(self, data, period_array, is_slow=False):
        """動的期間でMAを計算（期間配列を使用）"""
        try:
            # 期間配列を整数に変換（安全な変換）
            periods = np.asarray(period_array, dtype=np.float64)
            periods = np.round(periods).astype(np.int32)
            periods = np.maximum(periods, 2)  # 最小期間2（MAの最小要件）
            
            length = len(data)
            result = np.full(length, np.nan)
            
            # 各時点で異なる期間のMAを計算
            for i in range(length):
                current_period = int(periods[i])
                
                if i < current_period - 1:
                    continue
                
                try:
                    # 現在の期間でMAを計算
                    if self.ma_type == 'hma':
                        # HMAを一時的に作成（期間を動的に変更）
                        temp_hma = HMA(
                            period=current_period,
                            src_type=self.src_type,
                            slope_index=self.hma_slope_index,
                            range_threshold=self.hma_range_threshold
                        )
                        # 現在位置までのデータで計算
                        window_data = data[:i+1] if isinstance(data, np.ndarray) else data.iloc[:i+1]
                        hma_result = temp_hma.calculate(window_data)
                        if hasattr(hma_result, 'values') and len(hma_result.values) > 0 and not np.isnan(hma_result.values[-1]):
                            result[i] = hma_result.values[-1]
                    
                    elif self.ma_type == 'alma':
                        # ALMAを一時的に作成
                        temp_alma = ALMA(
                            period=current_period,
                            offset=self.alma_offset,
                            sigma=self.alma_sigma,
                            src_type=self.src_type,
                            use_dynamic_period=False
                        )
                        window_data = data[:i+1] if isinstance(data, np.ndarray) else data.iloc[:i+1]
                        alma_result = temp_alma.calculate(window_data)
                        if hasattr(alma_result, 'values') and len(alma_result.values) > 0 and not np.isnan(alma_result.values[-1]):
                            result[i] = alma_result.values[-1]
                    
                    elif self.ma_type == 'zlema':
                        # ZLEMAを一時的に作成
                        temp_zlema = ZLEMA(
                            period=current_period,
                            src_type=self.src_type,
                            use_dynamic_period=False,
                            slope_index=self.zlema_slope_index,
                            range_threshold=self.zlema_range_threshold
                        )
                        window_data = data[:i+1] if isinstance(data, np.ndarray) else data.iloc[:i+1]
                        zlema_result = temp_zlema.calculate(window_data)
                        if hasattr(zlema_result, 'values') and len(zlema_result.values) > 0 and not np.isnan(zlema_result.values[-1]):
                            result[i] = zlema_result.values[-1]
                    
                    # ZAdaptiveMAは外部ERを必要とするため、簡易EMAでフォールバック
                    elif self.ma_type == 'z_adaptive_ma':
                        # EfficiencyRatioからERを計算
                        if hasattr(self, 'efficiency_ratio_indicator'):
                            window_data = data[:i+1] if isinstance(data, np.ndarray) else data.iloc[:i+1]
                            er_result = self.efficiency_ratio_indicator.calculate(window_data)
                            external_er = er_result.values if hasattr(er_result, 'values') else er_result
                            
                            if len(external_er) > 0:
                                # ZAdaptiveMAを一時的に作成してERを渡す
                                temp_zma = ZAdaptiveMA(
                                    fast_period=self.z_adaptive_ma_fast_period,
                                    slow_period=current_period,
                                    src_type=self.src_type
                                )
                                zma_result = temp_zma.calculate(window_data, external_er=external_er)
                                if hasattr(zma_result, 'values') and len(zma_result.values) > 0 and not np.isnan(zma_result.values[-1]):
                                    result[i] = zma_result.values[-1]
                        else:
                            # フォールバック: 簡易EMA
                            src_prices = PriceSource.calculate_source(data, self.src_type)
                            window_prices = src_prices[:i+1]
                            if len(window_prices) >= current_period:
                                ema_result = calculate_ema_numba(window_prices, current_period)
                                if len(ema_result) > 0 and not np.isnan(ema_result[-1]):
                                    result[i] = ema_result[-1]
                
                except Exception as e:
                    # 個別計算でエラーが発生した場合はスキップ
                    continue
            
            return result
            
        except Exception as e:
            self.logger.warning(f"動的MA計算中にエラー: {str(e)}。フォールバックを使用します。")
            # エラー時は固定期間でフォールバック
            fixed_period = int(np.nanmean(period_array)) if len(period_array) > 0 else (self.length * 2 if is_slow else self.length)
            fixed_period = max(2, fixed_period)  # 最小期間2
            return self._calculate_fixed_ma(data, fixed_period)
    
    def _calculate_fixed_ma(self, data, period):
        """固定期間でMAを計算"""
        try:
            # 期間の最小値チェック
            period = max(2, int(period))
            
            if self.ma_type == 'hma':
                temp_ma = HMA(
                    period=period,
                    src_type=self.src_type,
                    slope_index=self.hma_slope_index,
                    range_threshold=self.hma_range_threshold
                )
            elif self.ma_type == 'alma':
                temp_ma = ALMA(
                    period=period,
                    offset=self.alma_offset,
                    sigma=self.alma_sigma,
                    src_type=self.src_type,
                    use_dynamic_period=False
                )
            elif self.ma_type == 'z_adaptive_ma':
                # ZAdaptiveMAの場合、EfficiencyRatioからERを計算
                if hasattr(self, 'efficiency_ratio_indicator'):
                    er_result = self.efficiency_ratio_indicator.calculate(data)
                    external_er = er_result.values if hasattr(er_result, 'values') else er_result
                    
                    temp_ma = ZAdaptiveMA(
                        fast_period=self.z_adaptive_ma_fast_period,
                        slow_period=period,
                        src_type=self.src_type
                    )
                    result = temp_ma.calculate(data, external_er=external_er)
                else:
                    # フォールバック: 簡易EMA
                    self.logger.warning("EfficiencyRatioが利用できません。EMAフォールバックを使用します。")
                    src_prices = PriceSource.calculate_source(data, self.src_type)
                    result = calculate_ema_numba(src_prices, max(2, period))
                    return result
            elif self.ma_type == 'zlema':
                temp_ma = ZLEMA(
                    period=period,
                    src_type=self.src_type,
                    use_dynamic_period=False,
                    slope_index=self.zlema_slope_index,
                    range_threshold=self.zlema_range_threshold
                )
            
            result = temp_ma.calculate(data)
            if hasattr(result, 'values'):
                return result.values
            else:
                return result
                
        except Exception as e:
            self.logger.warning(f"固定MA計算中にエラー: {str(e)}。EMAフォールバックを使用します。")
            # 最終フォールバック：EMA
            src_prices = PriceSource.calculate_source(data, self.src_type)
            return calculate_ema_numba(src_prices, max(2, period))

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ZAdaptiveFlowResult:
        """
        Z Adaptive Flowを計算
        
        Args:
            data: 価格データ
        
        Returns:
            ZAdaptiveFlowResult: 計算結果
        """
        try:
            # データハッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                return self._result_cache[data_hash]
            
            # データ長チェック
            data_length = len(data)
            if data_length == 0:
                self.logger.warning("入力データが空です。")
                return self._create_empty_result()
            
            # 価格ソース抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            close_prices = PriceSource.calculate_source(data, 'close')
            
            # 1. 固定乗数の準備
            data_length = len(data)
            slow_multiplier = np.full(data_length, self.slow_period_multiplier)
            volatility_multiplier = np.full(data_length, self.volatility_multiplier)
            
            # 2. MA計算（固定期間を使用）
            # ファストMA（固定期間）
            if self.ma_type == 'z_adaptive_ma':
                # ZAdaptiveMAの場合、EfficiencyRatioからERを計算
                er_result = self.efficiency_ratio_indicator.calculate(data)
                external_er = er_result.values if hasattr(er_result, 'values') else er_result
                
                fast_ma_result = self.fast_ma.calculate(data, external_er=external_er)
            else:
                fast_ma_result = self.fast_ma.calculate(data)
                
            if hasattr(fast_ma_result, 'values'):
                fast_ma_values = fast_ma_result.values
            elif isinstance(fast_ma_result, np.ndarray):
                fast_ma_values = fast_ma_result
            else:
                fast_ma_values = np.array(fast_ma_result)
            
            # スローMA（固定期間）
            if self.ma_type == 'z_adaptive_ma':
                # ZAdaptiveMAの場合、EfficiencyRatioからERを計算
                slow_ma_result = self.slow_ma.calculate(data, external_er=external_er)
            else:
                slow_ma_result = self.slow_ma.calculate(data)
                
            if hasattr(slow_ma_result, 'values'):
                slow_ma_values = slow_ma_result.values
            elif isinstance(slow_ma_result, np.ndarray):
                slow_ma_values = slow_ma_result
            else:
                slow_ma_values = np.array(slow_ma_result)
            
            # 3. Basis線の計算
            basis = (fast_ma_values + slow_ma_values) / 2.0
            
            # 4. ボラティリティの計算
            volatility_result = self.volatility_indicator.calculate(data)
            
            # ボラティリティ結果の処理
            if hasattr(volatility_result, 'values'):
                volatility_values = volatility_result.values
            elif hasattr(volatility_result, 'volatility'):
                volatility_values = volatility_result.volatility
            elif isinstance(volatility_result, np.ndarray):
                volatility_values = volatility_result
            else:
                # フォールバック: 簡単な価格変動率ベースのボラティリティ
                self.logger.warning("ボラティリティ計算結果が無効です。フォールバックを使用します。")
                returns = np.diff(np.log(src_prices))
                returns = np.concatenate([np.array([0]), returns])  # 長さを合わせる
                volatility_values = np.abs(returns)
            
            # ボラティリティ値の検証
            if len(volatility_values) != data_length:
                self.logger.warning(f"ボラティリティ配列の長さが不正です: {len(volatility_values)} != {data_length}")
                # 長さを調整
                if len(volatility_values) < data_length:
                    # 不足分をNaNで埋める
                    padding = np.full(data_length - len(volatility_values), np.nan)
                    volatility_values = np.concatenate([padding, volatility_values])
                else:
                    # 余分な部分を切り取る
                    volatility_values = volatility_values[-data_length:]
            
            # 5. ボラティリティの平滑化（改良版）
            # まず元のボラティリティ値をクリーンアップ
            clean_volatility = volatility_values.copy()
            
            # NaNや0の値を処理
            nan_mask = np.isnan(clean_volatility) | (clean_volatility <= 0)
            if nan_mask.all():
                # 全て無効な場合はフォールバック値
                clean_volatility = np.full_like(clean_volatility, 1e-6)
                self.logger.warning("ボラティリティ値が全て無効です。フォールバック値を使用します。")
            else:
                # 無効値を直前の有効値または平均値で補間
                valid_values = clean_volatility[~nan_mask]
                if len(valid_values) > 0:
                    fallback_value = np.mean(valid_values)
                    # 前方補間
                    last_valid = fallback_value
                    for i in range(len(clean_volatility)):
                        if nan_mask[i]:
                            clean_volatility[i] = last_valid
                        else:
                            last_valid = clean_volatility[i]
            
            # 平滑化期間の調整
            if self.smooth_length > len(clean_volatility):
                actual_smooth_length = max(2, len(clean_volatility) // 3)
                self.logger.warning(f"平滑化期間を{self.smooth_length}から{actual_smooth_length}に調整しました。")
            elif self.smooth_length < 2:
                actual_smooth_length = 2
            else:
                actual_smooth_length = self.smooth_length
            
            # 複数の平滑化手法を試行
            smoothed_volatility = None
            
            # 方法1: 改良EMAを試行
            try:
                smoothed_volatility = calculate_ema_numba(clean_volatility, actual_smooth_length)
                if not np.isnan(smoothed_volatility).all() and np.sum(smoothed_volatility > 0) > len(smoothed_volatility) * 0.5:
                    # 成功
                    pass
                else:
                    smoothed_volatility = None
            except:
                smoothed_volatility = None
            
            # 方法2: 堅牢SMAを試行
            if smoothed_volatility is None or np.isnan(smoothed_volatility).all():
                try:
                    smoothed_volatility = calculate_robust_sma_numba(clean_volatility, actual_smooth_length)
                    if not np.isnan(smoothed_volatility).all():
                        self.logger.info("堅牢SMAで平滑化を実行しました。")
                    else:
                        smoothed_volatility = None
                except:
                    smoothed_volatility = None
            
            # 方法3: 最終フォールバック（手動移動平均）
            if smoothed_volatility is None or np.isnan(smoothed_volatility).all():
                self.logger.warning("自動平滑化が失敗しました。手動移動平均を使用します。")
                window_size = min(3, len(clean_volatility))
                smoothed_volatility = np.zeros_like(clean_volatility)
                
                for i in range(len(clean_volatility)):
                    start_idx = max(0, i - window_size + 1)
                    window_data = clean_volatility[start_idx:i+1]
                    valid_data = window_data[window_data > 0]
                    
                    if len(valid_data) > 0:
                        smoothed_volatility[i] = np.mean(valid_data)
                    elif i > 0:
                        smoothed_volatility[i] = smoothed_volatility[i-1]
                    else:
                        smoothed_volatility[i] = 1e-6
            
            # 最終的な値の検証と修正
            if np.isnan(smoothed_volatility).any():
                nan_indices = np.isnan(smoothed_volatility)
                smoothed_volatility[nan_indices] = clean_volatility[nan_indices]
                
            # 値が0以下の場合の修正
            zero_mask = smoothed_volatility <= 0
            if zero_mask.any():
                smoothed_volatility[zero_mask] = np.maximum(clean_volatility[zero_mask], 1e-8)
            
            # ボラティリティの最小値保証（0除算回避）
            min_vol = 1e-8
            smoothed_volatility = np.maximum(smoothed_volatility, min_vol)
            clean_volatility = np.maximum(clean_volatility, min_vol)
            
            # デバッグ情報（最初の数値のみ表示）
            if np.isnan(smoothed_volatility).any():
                nan_count = np.isnan(smoothed_volatility).sum()
                self.logger.warning(f"平滑化後もNaN値が{nan_count}個残っています。")
            else:
                valid_count = np.sum(smoothed_volatility > min_vol)
                self.logger.debug(f"平滑化ボラティリティ: 有効値数={valid_count}/{len(smoothed_volatility)}")
            
            # 6. 固定ボラティリティ乗数適用
            upper_band = basis + (smoothed_volatility * self.volatility_multiplier)
            lower_band = basis - (smoothed_volatility * self.volatility_multiplier)
            
            # 7. トレンド状態とレベルの計算
            # 入力データの詳細確認（デバッグ用、通常はコメントアウト）
            # print(f"デバッグ: 入力データサイズ確認")
            # print(f"  - close_prices: {len(close_prices)}")
            # print(f"  - upper_band: {len(upper_band)}")
            # print(f"  - lower_band: {len(lower_band)}")
            # print(f"  - basis: {len(basis)}")
            
            # まずNumba版を試行
            # print("デバッグ: Numba版のcalculate_trend_state_and_levelを試行")
            try:
                trend_state, level = calculate_trend_state_and_level(
                    close_prices, upper_band, lower_band, basis
                )
                
                # 結果をチェック
                level_nan_count = np.isnan(level).sum()
                if level_nan_count == len(level):
                    # print("デバッグ: Numba版が失敗、フォールバック版を使用")
                    trend_state, level = calculate_trend_state_and_level_fallback(
                        close_prices, upper_band, lower_band, basis
                    )
                # else:
                #     print("デバッグ: Numba版が成功")
            except Exception as e:
                # print(f"デバッグ: Numba版でエラー発生: {str(e)}, フォールバック版を使用")
                trend_state, level = calculate_trend_state_and_level_fallback(
                    close_prices, upper_band, lower_band, basis
                )
            
            # 8. シグナル検出
            long_signals, short_signals = detect_signals(trend_state)
            
            # 結果作成（固定乗数値を保持）
            result = ZAdaptiveFlowResult(
                basis=basis,
                level=level,
                upper=upper_band,
                lower=lower_band,
                trend_state=trend_state,
                volatility=volatility_values,  # 元のボラティリティ値
                slow_multiplier=slow_multiplier,  # 固定値配列
                volatility_multiplier=volatility_multiplier,  # 固定値配列
                fast_ma=fast_ma_values,
                slow_ma=slow_ma_values,
                long_signals=long_signals,
                short_signals=short_signals,
                smoothed_volatility=smoothed_volatility
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Z Adaptive Flow計算中にエラー: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._create_empty_result()
    
    def _create_empty_result(self) -> ZAdaptiveFlowResult:
        """空の結果を作成"""
        empty_array = np.array([])
        empty_bool_array = np.array([], dtype=bool)
        empty_int_array = np.array([], dtype=np.int8)
        
        return ZAdaptiveFlowResult(
            basis=empty_array,
            level=empty_array,
            upper=empty_array,
            lower=empty_array,
            trend_state=empty_int_array,
            volatility=empty_array,
            slow_multiplier=empty_array,
            volatility_multiplier=empty_array,
            fast_ma=empty_array,
            slow_ma=empty_array,
            long_signals=empty_bool_array,
            short_signals=empty_bool_array,
            smoothed_volatility=empty_array
        )
    
    def get_detailed_result(self, data: Union[pd.DataFrame, np.ndarray] = None) -> ZAdaptiveFlowResult:
        """
        詳細な計算結果を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            ZAdaptiveFlowResult: 詳細な計算結果
        """
        if data is not None:
            return self.calculate(data)
        
        # 最新の結果を使用
        if not self._result_cache:
            return self._create_empty_result()
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result
    
    def get_trend_lines(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        メインのトレンドライン（basis, level）を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            basis, level のタプル
        """
        result = self.get_detailed_result(data)
        return result.basis, result.level
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        バンド（upper, lower）を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            upper, lower のタプル
        """
        result = self.get_detailed_result(data)
        return result.upper, result.lower
    
    def get_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        シグナル（long_signals, short_signals）を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            long_signals, short_signals のタプル
        """
        result = self.get_detailed_result(data)
        return result.long_signals, result.short_signals
    
    def get_trend_state(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド状態を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            trend_state: 1=bullish, -1=bearish
        """
        result = self.get_detailed_result(data)
        return result.trend_state
    
    def get_current_trend(self, data: Union[pd.DataFrame, np.ndarray] = None) -> str:
        """
        現在のトレンド状態を文字列で取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            'bullish', 'bearish', または 'unknown'
        """
        trend_state = self.get_trend_state(data)
        if len(trend_state) == 0:
            return 'unknown'
        
        latest_trend = trend_state[-1]
        if latest_trend == 1:
            return 'bullish'
        elif latest_trend == -1:
            return 'bearish'
        else:
            return 'unknown'
    
    def reset(self) -> None:
        """インジケーターの状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        
        # 依存オブジェクトのリセット
        if hasattr(self, 'fast_ma') and self.fast_ma:
            self.fast_ma.reset()
        if hasattr(self, 'slow_ma') and self.slow_ma:
            self.slow_ma.reset()
        if hasattr(self, 'volatility_indicator') and self.volatility_indicator:
            self.volatility_indicator.reset()
        if hasattr(self, 'efficiency_ratio_indicator') and self.efficiency_ratio_indicator:
            self.efficiency_ratio_indicator.reset()
        
        self.logger.debug(f"インジケーター '{self.name}' がリセットされました。") 