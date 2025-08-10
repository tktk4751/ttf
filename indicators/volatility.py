#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit
import traceback
import math

# --- 依存関係のインポート ---

from .indicator import Indicator
from .price_source import PriceSource
from .alma import calculate_alma_numba as calculate_alma
from .hma import calculate_hma_numba
from .zlema import calculate_zlema_numba, calculate_ema_numba
from .cycle.ehlers_unified_dc import EhlersUnifiedDC


@dataclass
class volatilityResult:
    """ボラティリティの計算結果"""
    values: np.ndarray        # 標準偏差（%ベース、比率）
    absolute_values: np.ndarray  # 価格ベース: 価格変動率の標準偏差, リターンベース: 絶対値ベースの標準偏差
    returns: np.ndarray       # 計算に使用されたリターン
    volatility_period: np.ndarray  # ドミナントサイクルまたは固定期間
    dc_values: np.ndarray     # ドミナントサイクル値（適応モード時のみ）


@njit(fastmath=True, parallel=True, cache=True)
def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    対数収益率を計算する（並列高速化版）
    
    Args:
        prices: 価格の配列
    
    Returns:
        対数収益率の配列
    """
    length = len(prices)
    returns = np.zeros(length, dtype=np.float64)
    
    if length > 1:
        # 最初の要素は0
        returns[0] = 0.0
        
        # 対数収益率の計算（並列処理）
        for i in prange(1, length):
            if prices[i-1] > 0 and prices[i] > 0:
                returns[i] = np.log(prices[i] / prices[i-1])
            else:
                returns[i] = 0.0
    
    return returns


@njit(fastmath=True, parallel=True, cache=True)
def calculate_simple_returns(prices: np.ndarray) -> np.ndarray:
    """
    単純収益率を計算する（並列高速化版）
    
    Args:
        prices: 価格の配列
    
    Returns:
        単純収益率の配列
    """
    length = len(prices)
    returns = np.zeros(length, dtype=np.float64)
    
    if length > 1:
        # 最初の要素は0
        returns[0] = 0.0
        
        # 単純収益率の計算（並列処理）
        for i in prange(1, length):
            if prices[i-1] > 0:
                returns[i] = (prices[i] - prices[i-1]) / prices[i-1]
            else:
                returns[i] = 0.0
    
    return returns


@njit(fastmath=True, cache=True)
def calculate_rolling_std_numba(
    returns: np.ndarray,
    period: int,
    ddof: int = 1
) -> float:
    """
    指定された期間の標準偏差を計算する（Numba最適化版）
    
    Args:
        returns: 収益率の配列
        period: 計算期間
        ddof: 自由度調整（デフォルト: 1）
    
    Returns:
        標準偏差
    """
    if len(returns) < period or period < 2:
        return np.nan
    
    # 平均を計算
    mean_val = 0.0
    for i in range(period):
        mean_val += returns[i]
    mean_val /= period
    
    # 分散を計算
    variance = 0.0
    for i in range(period):
        diff = returns[i] - mean_val
        variance += diff * diff
    
    # 自由度調整
    variance /= (period - ddof)
    
    return np.sqrt(variance)


@njit(fastmath=True, cache=True)
def calculate_rolling_std_prices_numba(
    prices: np.ndarray,
    period: int,
    ddof: int = 1
) -> float:
    """
    価格ベースの指定された期間の標準偏差を計算する（Numba最適化版）
    修正版：価格の絶対的な標準偏差ではなく、価格変動率の標準偏差を計算
    
    Args:
        prices: 価格の配列
        period: 計算期間
        ddof: 自由度調整（デフォルト: 1）
    
    Returns:
        価格変動率の標準偏差（比率ベース）
    """
    if len(prices) < period or period < 2:
        return np.nan
    
    # 価格変動率を計算（期間内の各価格 / 平均価格 - 1）
    mean_price = 0.0
    for i in range(period):
        mean_price += prices[i]
    mean_price /= period
    
    if mean_price <= 0:
        return np.nan
    
    # 各価格変動率を計算し、その標準偏差を求める
    variance = 0.0
    for i in range(period):
        if prices[i] > 0:
            price_change_ratio = (prices[i] / mean_price) - 1.0
            variance += price_change_ratio * price_change_ratio
    
    # 自由度調整
    variance /= (period - ddof)
    
    return np.sqrt(variance)


@njit(fastmath=True, cache=True)
def calculate_wilder_smoothing(values: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's Smoothing（従来のATR計算方法）
    
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
def calculate_volatility_with_smoothing(
    values: np.ndarray,
    period: int,
    smoothing_method: int
) -> np.ndarray:
    """
    指定されたスムージング方法でボラティリティを平滑化する
    
    Args:
        values: ボラティリティ値の配列
        period: スムージング期間
        smoothing_method: スムージング方法 (0: なし, 1: Wilder's, 2: HMA, 3: ALMA, 4: ZLEMA)
    
    Returns:
        平滑化されたボラティリティ値の配列
    """
    if smoothing_method == 0:  # スムージングなし
        return values
    elif smoothing_method == 1:  # Wilder's Smoothing
        return calculate_wilder_smoothing(values, period)
    elif smoothing_method == 2:  # HMA
        return calculate_hma_numba(values, period)
    elif smoothing_method == 3:  # ALMA
        return calculate_alma(values, period, 0.85, 6.0)  # デフォルトパラメータ
    elif smoothing_method == 4:  # ZLEMA
        return calculate_zlema_numba(values, period)
    else:
        # デフォルトはスムージングなし
        return values


@njit(fastmath=True, parallel=True, cache=True)
def calculate_volatility(
    prices: np.ndarray,
    volatility_period: np.ndarray,
    max_period: int,
    return_type: str = 'log',      # 'log' または 'simple'
    smoothing_method: int = 0,     # 0: なし, 1: Wilder's, 2: HMA, 3: ALMA, 4: ZLEMA
    smoother_period: int = 14,     # スムージング期間
    ddof: int = 1,                 # 自由度調整
    calculation_mode: str = 'return'  # 'return' または 'price'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ボラティリティを計算する（並列高速化版）
    
    Args:
        prices: 価格の配列
        volatility_period: ドミナントサイクルまたは固定期間の配列
        max_period: 最大期間（計算開始位置用）
        return_type: リターンタイプ（'log'または'simple'）
        smoothing_method: 平滑化アルゴリズムのタイプ（0-4）
        smoother_period: スムージング期間
        ddof: 自由度調整（デフォルト: 1）
        calculation_mode: 計算モード（'return': リターンベース, 'price': 価格ベース）
    
    Returns:
        (標準偏差の配列, リターンの配列)
    """
    length = len(prices)
    volatility = np.zeros(length, dtype=np.float64)
    
    # リターンの計算（リターンベースの場合）
    if calculation_mode == 'return':
        if return_type == 'log':
            returns = calculate_log_returns(prices)
        else:  # 'simple'
            returns = calculate_simple_returns(prices)
    else:
        # 価格ベースの場合、returnsは空配列
        returns = np.zeros(length, dtype=np.float64)
    
    # 各時点での標準偏差を計算
    for i in prange(max_period, length):
        # その時点での期間を取得
        curr_period = int(volatility_period[i])
        if curr_period < 2:
            curr_period = 2
            
        # 現在位置までのデータを取得
        start_idx = max(0, i - curr_period + 1)
        end_idx = i + 1
        
        if end_idx - start_idx >= curr_period:
            if calculation_mode == 'return':
                # リターンベースの標準偏差
                window_returns = returns[start_idx:end_idx]
                std_val = calculate_rolling_std_numba(window_returns, curr_period, ddof)
            else:
                # 価格ベースの標準偏差
                window_prices = prices[start_idx:end_idx]
                std_val = calculate_rolling_std_prices_numba(window_prices, curr_period, ddof)
            
            if not np.isnan(std_val):
                volatility[i] = std_val
            else:
                volatility[i] = np.nan
    
    return volatility, returns


@njit(fastmath=True, parallel=True, cache=True)
def calculate_percent_volatility(absolute_volatility: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """
    絶対値ベースのボラティリティから%ベースのボラティリティを計算する（並列高速化版）
    
    Args:
        absolute_volatility: 絶対値ベースのボラティリティ配列
        prices: 価格の配列
    
    Returns:
        %ベースのボラティリティ配列
    """
    length = len(absolute_volatility)
    percent_volatility = np.zeros_like(absolute_volatility, dtype=np.float64)
    
    # 並列処理で高速化
    for i in prange(length):
        if not np.isnan(absolute_volatility[i]) and prices[i] > 0:
            percent_volatility[i] = absolute_volatility[i] / prices[i]
    
    return percent_volatility


def apply_smoothing_wrapper(
    values: np.ndarray,
    smoother_type: str,
    smoother_period: int
) -> np.ndarray:
    """
    指定されたスムージングアルゴリズムを適用する（ラッパー関数、NaN対応強化）
    
    Args:
        values: 平滑化する値の配列
        smoother_type: 平滑化アルゴリズムのタイプ
        smoother_period: スムージング期間
    
    Returns:
        平滑化された値の配列
    """
    if len(values) == 0:
        return values
    
    # 入力値の前処理
    clean_values = values.copy()
    
    # NaNや異常値の処理
    nan_mask = np.isnan(clean_values) | (clean_values <= 0)
    if nan_mask.all():
        # 全て無効な場合はフォールバック値
        clean_values = np.full_like(clean_values, 1e-8)
    else:
        # 無効値を補間
        valid_values = clean_values[~nan_mask]
        if len(valid_values) > 0:
            fallback_value = np.mean(valid_values)
            # 前方補間
            last_valid = fallback_value
            for i in range(len(clean_values)):
                if nan_mask[i]:
                    clean_values[i] = last_valid
                else:
                    last_valid = clean_values[i]
    
    # 期間の調整
    effective_period = min(smoother_period, len(clean_values))
    if effective_period < 2:
        effective_period = 2
    
    # スムージング方法の変換と実行
    smoothing_methods = {'none': 0, 'wilder': 1, 'hma': 2, 'alma': 3, 'zlema': 4}
    smoothing_method_int = smoothing_methods.get(smoother_type.lower(), 0)
    
    try:
        # 基本的な平滑化を試行
        if smoothing_method_int == 0:  # なし
            result = clean_values
        else:
            result = calculate_volatility_with_smoothing(clean_values, effective_period, smoothing_method_int)
        
        # 結果の検証
        if result is None or np.isnan(result).all():
            raise ValueError("平滑化結果が無効です")
        
        # NaN値の後処理
        if np.isnan(result).any():
            nan_indices = np.isnan(result)
            result[nan_indices] = clean_values[nan_indices]
        
        # 最小値の保証
        result = np.maximum(result, 1e-8)
        
        return result
        
    except Exception as e:
        # フォールバック: 単純な移動平均
        try:
            window_size = min(3, len(clean_values))
            result = np.zeros_like(clean_values)
            
            for i in range(len(clean_values)):
                start_idx = max(0, i - window_size + 1)
                window_data = clean_values[start_idx:i+1]
                valid_data = window_data[window_data > 0]
                
                if len(valid_data) > 0:
                    result[i] = np.mean(valid_data)
                elif i > 0:
                    result[i] = result[i-1]
                else:
                    result[i] = 1e-8
            
            return np.maximum(result, 1e-8)
            
        except:
            # 最終フォールバック
            return np.maximum(clean_values, 1e-8)


class VolatilityResult(NamedTuple):
    """Volatility計算結果"""
    values: np.ndarray
    trend_signals: np.ndarray  # 1=up, -1=down, 0=range
    current_trend: str  # 'up', 'down', 'range'
    current_trend_value: int  # 1, -1, 0


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


class volatility(Indicator):
    """
    ボラティリティインジケーター
    
    特徴:
    - ドミナントサイクル検出器から適応期間を決定、または固定期間を使用
    - リターンベースまたは価格ベースの標準偏差計算を選択可能
    - 対数収益率または単純収益率ベースの標準偏差計算（リターンベース時）
    - Wilder's、ALMA、HMA、ZLEMAによる平滑化（オプション）
    - 絶対値ベースと%ベースの両方の値を提供
    
    使用方法:
    - ボラティリティに基づいたリスク管理
    - ポジションサイジングの調整
    - ボラティリティブレイクアウト戦略
    - 異なる価格帯の銘柄間でのボラティリティ比較（%ベース）
    """
    
    def __init__(
        self,
        # --- ボラティリティ計算パラメータ ---
        period_mode: str = 'adaptive',           # 'adaptive' または 'fixed'
        fixed_period: int = 21,                  # 固定期間モード時の期間
        calculation_mode: str = 'price',        # 'return' または 'price'
        return_type: str = 'log',                # 'log' または 'simple'（リターンベース時のみ）
        ddof: int = 1,                          # 自由度調整
        smoother_type: str = 'hma',             # 'wilder', 'alma', 'hma', 'zlema', または 'none'
        smoother_period: int = 14,               # スムージング期間
        # --- ドミナントサイクル検出器パラメータ（適応モード時） ---
        detector_type: str = 'cycle_period2',           # 検出器タイプ
        cycle_part: float = 1.0,                # サイクル部分の倍率
        lp_period: int = 13,
        hp_period: int = 50,
        max_cycle: int = 55,                    # 最大サイクル期間
        min_cycle: int = 5,                     # 最小サイクル期間
        max_output: int = 34,                   # 最大出力値
        min_output: int = 2,                    # 最小出力値（標準偏差には最低2が必要）
        # --- 価格ソースパラメータ ---
        src_type: str = 'hlc3'                 # 価格ソース
    ):
        """
        コンストラクタ
        
        Args:
            period_mode: 期間モード
                - 'adaptive': ドミナントサイクル検出器から適応期間を決定
                - 'fixed': 固定期間を使用
            fixed_period: 固定期間モード時の期間（デフォルト: 21）
            calculation_mode: 計算モード（デフォルト: 'return'）
                'return' - リターンベースの標準偏差
                'price' - 価格ベースの標準偏差
            return_type: リターンタイプ（デフォルト: 'log'、リターンベース時のみ）
                'log' - 対数収益率
                'simple' - 単純収益率
            ddof: 自由度調整（デフォルト: 1）
            smoother_type: 平滑化アルゴリズムのタイプ（デフォルト: 'none'）
                'wilder' - Wilder's Smoothing
                'alma' - ALMA（Arnaud Legoux Moving Average）
                'hma' - ハル移動平均（Hull Moving Average）
                'zlema' - Zero Lag EMA
                'none' - 平滑化なし
            smoother_period: スムージング期間（デフォルト: 14）
            detector_type: 検出器タイプ（適応モード時）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 55）
            min_cycle: 最小サイクル期間（デフォルト: 5）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 2）
            src_type: 計算に使用する価格ソース（デフォルト: 'close'）
        """
        mode_str = f"{period_mode}_{fixed_period}" if period_mode == 'fixed' else f"{period_mode}_{detector_type}"
        calc_str = f"calc_{calculation_mode}"
        ret_str = f"_{return_type}" if calculation_mode == 'return' else ""
        smooth_str = f"_{smoother_type}_{smoother_period}" if smoother_type != 'none' else ""
        super().__init__(
            f"volatility({mode_str},{calc_str}{ret_str}{smooth_str},src={src_type})"
        )
        
        # パラメータの保存
        self.period_mode = period_mode.lower()
        self.fixed_period = fixed_period
        self.calculation_mode = calculation_mode.lower()
        self.return_type = return_type.lower()
        self.ddof = ddof
        self.smoother_type = smoother_type.lower()
        self.smoother_period = smoother_period
        self.detector_type = detector_type
        self.cycle_part = cycle_part
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = max(min_output, 2)  # 標準偏差には最低2が必要
        self.src_type = src_type.lower()
        
        # パラメータ検証
        if self.period_mode not in ['adaptive', 'fixed']:
            raise ValueError(f"無効なperiod_mode: {period_mode}。'adaptive'または'fixed'を指定してください。")
        if self.calculation_mode not in ['return', 'price']:
            raise ValueError(f"無効なcalculation_mode: {calculation_mode}。'return'または'price'を指定してください。")
        if self.return_type not in ['log', 'simple']:
            raise ValueError(f"無効なreturn_type: {return_type}。'log'または'simple'を指定してください。")
        if self.smoother_type not in ['wilder', 'alma', 'hma', 'zlema', 'none']:
            raise ValueError(f"無効なsmoother_type: {smoother_type}。'wilder'、'alma'、'hma'、'zlema'、または'none'を指定してください。")
        
        # スムージング方法の変換
        smoothing_methods = {'none': 0, 'wilder': 1, 'hma': 2, 'alma': 3, 'zlema': 4}
        self.smoothing_method_int = smoothing_methods[self.smoother_type]
        
        # ドミナントサイクル検出器を初期化（適応モード時のみ）
        self.dc_detector = None
        if self.period_mode == 'adaptive':
            self.dc_detector = EhlersUnifiedDC(
                detector_type=self.detector_type,
                cycle_part=self.cycle_part,
                lp_period=self.lp_period,
                hp_period=self.hp_period,
                max_cycle=self.max_cycle,
                min_cycle=self.min_cycle,
                max_output=self.max_output,
                min_output=self.min_output
            )
        
        # 依存ツールの初期化
        self.price_source_extractor = PriceSource()
        
        # 結果キャッシュ
        self._result = None
        self._data_hash = None
        
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する（超高速版）"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    close_val = float(data.iloc[0].get('close', data.iloc[0, 3]))
                    last_close = float(data.iloc[-1].get('close', data.iloc[-1, 3]))
                    data_signature = (length, close_val, last_close)
                else:
                    data_signature = (0, 0.0, 0.0)
            else:
                # NumPy配列の場合
                length = len(data)
                if length > 0 and data.ndim > 1 and data.shape[1] >= 4:
                    data_signature = (length, float(data[0, 3]), float(data[-1, 3]))
                else:
                    data_signature = (0, 0.0, 0.0)
            
            # パラメータの最小セット
            params_sig = f"{self.period_mode}_{self.fixed_period}_{self.calculation_mode}_{self.return_type}_{self.smoother_type}_{self.smoother_period}_{self.src_type}"
            if self.period_mode == 'adaptive':
                params_sig += f"_{self.detector_type}_{self.max_output}_{self.min_output}"
            
            # 超高速ハッシュ
            return f"{hash(data_signature)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.period_mode}_{self.calculation_mode}_{self.return_type}_{self.smoother_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ボラティリティを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            標準偏差の値（%ベース）
        """
        try:
            current_data_len = len(data) if hasattr(data, '__len__') else 0
            if current_data_len == 0:
                self.logger.warning("入力データが空です。")
                return np.array([])

            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values

            # ソース価格取得
            src_prices = PriceSource.calculate_source(data, self.src_type)
            if src_prices is None or len(src_prices) == 0:
                self.logger.warning(f"価格ソース '{self.src_type}' 取得失敗/空。")
                return np.full(current_data_len, np.nan)
            
            if not isinstance(src_prices, np.ndarray):
                src_prices = np.array(src_prices)
            if src_prices.dtype != np.float64:
                try:
                    src_prices = src_prices.astype(np.float64)
                except ValueError:
                    self.logger.error(f"価格ソース '{self.src_type}' をfloat64に変換できませんでした。")
                    return np.full(current_data_len, np.nan)

            self._data_hash = data_hash  # 新しいハッシュを保存

            # 期間の決定
            data_length = len(src_prices)
            if self.period_mode == 'adaptive':
                # ドミナントサイクルから期間を決定
                if self.dc_detector is None:
                    self.logger.error("適応モードですが、ドミナントサイクル検出器が初期化されていません。")
                    return np.full(current_data_len, np.nan)
                
                dc_values = self.dc_detector.calculate(data)
                volatility_period = np.asarray(dc_values, dtype=np.float64)
                
                # 最大期間を取得
                max_period_value_float = np.nanmax(volatility_period)
                if np.isnan(max_period_value_float):
                    max_period_value = self.fixed_period  # フォールバック
                    self.logger.warning("ドミナントサイクルが全てNaNです。固定期間をフォールバックとして使用します。")
                else:
                    max_period_value = int(max_period_value_float)
                    if max_period_value < 2:
                        max_period_value = 2
            else:
                # 固定期間を使用
                volatility_period = np.full(data_length, self.fixed_period, dtype=np.float64)
                max_period_value = self.fixed_period
                dc_values = np.full(data_length, self.fixed_period, dtype=np.float64)

            # データ長の検証
            if data_length < max_period_value:
                self.logger.warning(f"データ長({data_length})が必要な最大期間({max_period_value})より短いため、計算できません。")
                self._reset_results()
                return np.full(current_data_len, np.nan)

            # ボラティリティの計算（並列版）
            volatility_values, returns_values = calculate_volatility(
                src_prices,
                volatility_period,
                max_period_value,
                self.return_type,
                self.smoothing_method_int,
                self.smoother_period,
                self.ddof,
                self.calculation_mode
            )
            
            # 平滑化の適用
            if self.smoother_type != 'none':
                # 平滑化前の値の検証
                if np.isnan(volatility_values).all():
                    self.logger.warning("ボラティリティ値が全てNaNです。平滑化をスキップします。")
                    smoothed_values = volatility_values
                else:
                    # NaN値の数をチェック
                    nan_count = np.isnan(volatility_values).sum()
                    if nan_count > 0:
                        self.logger.debug(f"平滑化前のNaN値数: {nan_count}/{len(volatility_values)}")
                    
                    try:
                        smoothed_values = apply_smoothing_wrapper(
                            volatility_values,
                            self.smoother_type,
                            self.smoother_period
                        )
                        
                        # 平滑化結果の検証
                        smoothed_nan_count = np.isnan(smoothed_values).sum()
                        if smoothed_nan_count > 0:
                            self.logger.debug(f"平滑化後のNaN値数: {smoothed_nan_count}/{len(smoothed_values)}")
                        
                        # 平滑化が成功したかチェック
                        if np.isnan(smoothed_values).all():
                            self.logger.warning("平滑化結果が全てNaNです。元の値を使用します。")
                            smoothed_values = volatility_values
                        elif smoothed_nan_count > len(smoothed_values) * 0.7:
                            self.logger.warning("平滑化結果のNaN率が高すぎます。元の値を使用します。")
                            smoothed_values = volatility_values
                        else:
                            # 部分的なNaN値を修正
                            if smoothed_nan_count > 0:
                                nan_mask = np.isnan(smoothed_values)
                                smoothed_values[nan_mask] = volatility_values[nan_mask]
                        
                        volatility_values = smoothed_values
                        
                    except Exception as e:
                        self.logger.warning(f"平滑化処理中にエラー: {str(e)}。元の値を使用します。")
                        # エラー時は元の値をそのまま使用
            else:
                self.logger.debug("平滑化はスキップされます（smoother_type='none'）。")
            
            # 絶対値ベースの標準偏差を保存
            absolute_volatility_values = volatility_values

            # %ベースの標準偏差に変換
            if self.calculation_mode == 'price':
                # 価格ベースの場合、修正版では既に比率ベース（価格変動率の標準偏差）なので
                # リターンベースと同じように扱う
                percent_volatility_values = absolute_volatility_values
            else:
                # リターンベースの場合、すでに比率なので%ベースとして使用
                percent_volatility_values = absolute_volatility_values

            # 結果の保存
            self._result = volatilityResult(
                values=np.copy(percent_volatility_values),           # %ベースの標準偏差
                absolute_values=np.copy(absolute_volatility_values), # 価格ベース: 価格変動率の標準偏差, リターンベース: 絶対値ベースの標準偏差
                returns=np.copy(returns_values),                     # リターン
                volatility_period=np.copy(volatility_period),        # 使用された期間
                dc_values=np.copy(dc_values)                         # ドミナントサイクル値
            )
            
            self._values = percent_volatility_values
            return percent_volatility_values
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ボラティリティ計算中にエラー: {error_msg}\n{stack_trace}")
            self._reset_results()
            return np.array([])

    def _reset_results(self):
        """内部結果とキャッシュをリセットする"""
        self._result = None
        self._data_hash = None
        self._values = None

    def get_dc_values(self) -> np.ndarray:
        """
        ドミナントサイクルの値を取得する（適応モード時のみ）
        
        Returns:
            np.ndarray: ドミナントサイクルの値
        """
        if self._result is None:
            return np.array([])
        return self._result.dc_values
    
    def get_returns(self) -> np.ndarray:
        """
        計算に使用されたリターンを取得する（リターンベース時のみ）
        
        Returns:
            np.ndarray: リターンの値
        """
        if self._result is None:
            return np.array([])
        return self._result.returns
    
    def get_volatility_period(self) -> np.ndarray:
        """
        使用された期間を取得する
        
        Returns:
            np.ndarray: 期間の値
        """
        if self._result is None:
            return np.array([])
        return self._result.volatility_period
    
    def get_percent_volatility(self) -> np.ndarray:
        """
        %ベースのボラティリティを取得する
        
        Returns:
            np.ndarray: %ベースのボラティリティ値（100倍されたパーセンテージ値）
        """
        if self._result is None:
            return np.array([])
        return self._result.values * 100  # 100倍して返す
    
    def get_absolute_volatility(self) -> np.ndarray:
        """
        絶対値ベースのボラティリティを取得する
        
        Returns:
            np.ndarray: 絶対値ベースのボラティリティ値
        """
        if self._result is None:
            return np.array([])
        return self._result.absolute_values
    
    def get_volatility_multiple(self, multiplier: float = 1.0) -> np.ndarray:
        """
        %ベースのボラティリティの倍数を取得する
        
        Args:
            multiplier: ボラティリティの倍数（デフォルト: 1.0）
            
        Returns:
            np.ndarray: %ベースのボラティリティ × 倍数
        """
        volatility = self.get_percent_volatility()
        return volatility * multiplier
    
    def get_absolute_volatility_multiple(self, multiplier: float = 1.0) -> np.ndarray:
        """
        絶対値ベースのボラティリティの倍数を取得する
        
        Args:
            multiplier: ボラティリティの倍数（デフォルト: 1.0）
            
        Returns:
            np.ndarray: 絶対値ベースのボラティリティ × 倍数
        """
        abs_volatility = self.get_absolute_volatility()
        return abs_volatility * multiplier
    
    def get_annualized_volatility(self, periods_per_year: int = 252) -> np.ndarray:
        """
        年率換算ボラティリティを取得する
        
        Args:
            periods_per_year: 年間の期間数（デフォルト: 252 = 営業日数）
            
        Returns:
            np.ndarray: 年率換算ボラティリティ（%ベース）
        """
        volatility = self.get_percent_volatility()
        return volatility * np.sqrt(periods_per_year)
        
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._reset_results()
        if self.dc_detector and hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset()
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。") 