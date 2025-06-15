#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange
import traceback

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
        def calculate(self, data): return np.full(len(data), 14.0)
        def reset(self): pass


class ADXVolResult(NamedTuple):
    """ADX_VOL計算結果"""
    values: np.ndarray              # ADX_VOL値（0-1の範囲）
    trend_signals: np.ndarray       # 1=up, -1=down, 0=range
    current_trend: str              # 'up', 'down', 'range'
    current_trend_value: int        # 1, -1, 0
    dynamic_periods: np.ndarray     # 動的期間の配列
    normalized_adx: np.ndarray      # 正規化ADX値
    stddev_factor: np.ndarray       # ATR標準偏差係数
    atr: np.ndarray                 # ATR値
    dynamic_atr_period: np.ndarray  # 動的ATR期間
    fixed_threshold: float          # 固定しきい値
    trend_state: np.ndarray         # トレンド状態 (1=トレンド、0=レンジ、NaN=不明)


@jit(nopython=True)
def calculate_tr(high: np.ndarray, low: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    """
    True Rangeを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        prev_close: 1期前の終値の配列
    
    Returns:
        True Range の配列
    """
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    tr = np.maximum(tr1, tr2)
    tr = np.maximum(tr, tr3)
    
    return tr


@jit(nopython=True)
def calculate_dm(high: np.ndarray, low: np.ndarray) -> tuple:
    """
    Directional Movement（+DM, -DM）を計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
    
    Returns:
        tuple: (+DM, -DM)の配列
    """
    length = len(high)
    plus_dm = np.zeros(length)
    minus_dm = np.zeros(length)
    
    for i in range(1, length):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        
        if (up > down) and (up > 0):
            plus_dm[i] = up
        else:
            plus_dm[i] = 0
        
        if (down > up) and (down > 0):
            minus_dm[i] = down
        else:
            minus_dm[i] = 0
    
    return plus_dm, minus_dm


@jit(nopython=True)
def calculate_normalized_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int
) -> np.ndarray:
    """
    正規化されたADXを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period: 計算期間
    
    Returns:
        正規化されたADX値の配列（0-1の範囲）
    """
    length = len(high)
    tr = np.zeros(length)
    tr[0] = high[0] - low[0]
    tr[1:] = calculate_tr(high[1:], low[1:], close[:-1])
    
    # +DM, -DMの計算
    plus_dm, minus_dm = calculate_dm(high, low)
    
    # 指数移動平均の計算用の係数
    alpha = 2.0 / (period + 1.0)
    
    # TR, +DM, -DMの平滑化
    smoothed_tr = np.zeros(length)
    smoothed_plus_dm = np.zeros(length)
    smoothed_minus_dm = np.zeros(length)
    
    # 最初の値の初期化
    smoothed_tr[0] = tr[0]
    smoothed_plus_dm[0] = plus_dm[0]
    smoothed_minus_dm[0] = minus_dm[0]
    
    # 指数移動平均による平滑化
    for i in range(1, length):
        smoothed_tr[i] = (tr[i] * alpha) + (smoothed_tr[i-1] * (1 - alpha))
        smoothed_plus_dm[i] = (plus_dm[i] * alpha) + (smoothed_plus_dm[i-1] * (1 - alpha))
        smoothed_minus_dm[i] = (minus_dm[i] * alpha) + (smoothed_minus_dm[i-1] * (1 - alpha))
    
    # +DI, -DIの計算
    nonzero_tr = np.where(smoothed_tr == 0, 1e-10, smoothed_tr)
    plus_di = smoothed_plus_dm / nonzero_tr
    minus_di = smoothed_minus_dm / nonzero_tr
    
    # DXの計算
    dx_sum = plus_di + minus_di
    # ゼロ除算を防ぐ
    nonzero_sum = np.where(dx_sum == 0, 1e-10, dx_sum)
    dx = np.abs(plus_di - minus_di) / nonzero_sum
    
    # ADXの計算（DXの平滑化）
    adx = np.zeros(length)
    adx[0] = dx[0]
    
    for i in range(1, length):
        adx[i] = (dx[i] * alpha) + (adx[i-1] * (1 - alpha))
    
    return adx


@njit(fastmath=True, cache=True)
def calculate_dynamic_nadx_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period_array: np.ndarray,
    max_period: int
) -> np.ndarray:
    """
    動的期間で正規化ADXを計算する（Numba JIT）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period_array: 各時点での期間の配列
        max_period: 最大期間（計算開始位置用）
    
    Returns:
        正規化ADX値の配列
    """
    length = len(high)
    result = np.full(length, np.nan)
    
    # 各時点での動的正規化ADXを計算
    for i in range(max_period, length):
        # その時点でのドミナントサイクルから決定された期間を取得
        curr_period = int(period_array[i])
        if curr_period < 1:
            curr_period = 1
            
        # 現在位置までのデータを取得
        start_idx = max(0, i - curr_period * 2)  # ADXには余裕をもったウィンドウが必要
        window_high = high[start_idx:i+1]
        window_low = low[start_idx:i+1]
        window_close = close[start_idx:i+1]
        
        # 現在の期間で正規化ADXを計算
        if len(window_high) >= curr_period:
            nadx_values = calculate_normalized_adx(window_high, window_low, window_close, curr_period)
            
            # 最後の値を正規化ADXとして使用
            if len(nadx_values) > 0 and not np.isnan(nadx_values[-1]):
                result[i] = nadx_values[-1]
    
    return result


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
def calculate_adx_vol_index(
    normalized_adx: np.ndarray,
    stddev_factor: np.ndarray
) -> np.ndarray:
    """
    ADX_VOLインデックスを計算する
    
    Args:
        normalized_adx: 正規化ADX値の配列（0-1の範囲）
        stddev_factor: 標準偏差ファクターの配列
    
    Returns:
        ADX_VOLインデックスの配列（0-1の範囲、1に近いほど強いトレンド）
    """
    # ADXと標準偏差係数を組み合わせて信頼性を考慮したトレンド指標を計算
    adx_vol = normalized_adx * stddev_factor
    
    # 値を0-1の範囲にクリップ
    adx_vol = np.maximum(0.0, np.minimum(1.0, adx_vol))
    
    return adx_vol


@njit(fastmath=True)
def calculate_fixed_threshold_trend_state(
    trend_index: np.ndarray,
    fixed_threshold: float
) -> np.ndarray:
    """
    固定しきい値に基づいてトレンド状態を計算する
    
    Args:
        trend_index: ADX_VOLインデックスの配列
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


class ADXVol(Indicator):
    """
    ADX_VOL（ADX with Volatility）インジケーター
    
    正規化ADXにATRの標準偏差係数を組み合わせたインジケーター。
    トレンドの強さとボラティリティの安定性を同時に評価することで、
    より信頼性の高いトレンド検出を実現する。
    
    特徴:
    - 正規化ADX（0-1の範囲）とATR標準偏差係数の組み合わせ
    - 動的期間対応（EhlersUnifiedDC使用）
    - ATRのスムージング機能（HMA、ALMA、ZLEMA対応）
    - 固定しきい値によるトレンド/レンジ状態の判定
    - トレンド方向判定機能（up, down, range）
    """
    
    def __init__(self, 
                 # ADXパラメータ
                 period: int = 13,
                 use_dynamic_period: bool = True,
                 
                 # EhlersUnifiedDC パラメータ
                 detector_type: str = 'cycle_period2',
                 cycle_part: float = 0.5,
                 max_cycle: int = 55,
                 min_cycle: int = 5,
                 max_output: int = 15,
                 min_output: int = 3,
                 src_type: str = 'hlc3',
                 lp_period: int = 10,
                 hp_period: int = 48,

                 # ATR パラメータ
                 atr_period: int = 13,
                 atr_smoothing_method: str = 'alma',
                 use_dynamic_atr_period: bool = True,
                 
                 # トレンド判定パラメータ
                 slope_index: int = 1,
                 range_threshold: float = 0.005,

                 # 固定しきい値のパラメータ
                 fixed_threshold: float = 0.25):
        """
        コンストラクタ
        
        Args:
            period: ADXの期間（デフォルト: 13）
            use_dynamic_period: 動的期間を使用するかどうか（デフォルト: True）
            detector_type: EhlersUnifiedDCで使用する検出器タイプ（デフォルト: 'cycle_period2'）
            cycle_part: DCのサイクル部分の倍率（デフォルト: 0.5）
            max_cycle: DCの最大サイクル期間（デフォルト: 55）
            min_cycle: DCの最小サイクル期間（デフォルト: 5）
            max_output: DCの最大出力値（デフォルト: 15）
            min_output: DCの最小出力値（デフォルト: 3）
            src_type: DC計算に使用する価格ソース（デフォルト: 'hlc3'）
            lp_period: 拡張DC用のローパスフィルター期間（デフォルト: 10）
            hp_period: 拡張DC用のハイパスフィルター期間（デフォルト: 48）
            atr_period: ATRの期間（デフォルト: 13）
            atr_smoothing_method: ATRで使用する平滑化アルゴリズム（デフォルト: 'alma'）
            use_dynamic_atr_period: 動的ATR期間を使用するかどうか（デフォルト: True）
            slope_index: トレンド判定期間（デフォルト: 1）
            range_threshold: range判定の基本閾値（デフォルト: 0.005）
            fixed_threshold: 固定しきい値（デフォルト: 0.25）
        """
        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_period else ""
        atr_str = f"_atr({atr_smoothing_method})" if atr_smoothing_method != 'none' else ""
        super().__init__(
            f"ADX_VOL(p={period}{dynamic_str}{atr_str},slope={slope_index},th={fixed_threshold})"
        )
        
        # ADXパラメータ
        self.period = period
        self.use_dynamic_period = use_dynamic_period
        
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
        self.dc_detector = None
        self._last_dc_values = None  # 最後に計算されたDC値を保存
        if self.use_dynamic_period:
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
        self._result: Optional[ADXVolResult] = None
    
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
            f"{self.period}_{self.use_dynamic_period}_{self.detector_type}_{self.cycle_part}_"
            f"{self.max_cycle}_{self.min_cycle}_{self.max_output}_{self.min_output}_{self.src_type}_"
            f"{self.lp_period}_{self.hp_period}_{self.atr_period}_{self.atr_smoothing_method}_"
            f"{self.use_dynamic_atr_period}_{self.slope_index}_{self.range_threshold:.3f}_{self.fixed_threshold}"
        )
        return f"{data_hash_val}_{hash(param_str)}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ADXVolResult:
        """
        ADX_VOLを計算する

        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合は'open', 'high', 'low', 'close'カラムが必要

        Returns:
            ADXVolResult: 計算結果
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
            self.logger.warning("入力データが空です。空の結果を返します。")
            empty_result = ADXVolResult(
                values=np.array([]),
                trend_signals=np.array([], dtype=np.int8),
                current_trend='range',
                current_trend_value=0,
                dynamic_periods=np.array([]),
                normalized_adx=np.array([]),
                stddev_factor=np.array([]),
                atr=np.array([]),
                dynamic_atr_period=np.array([]),
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
                    return ADXVolResult(
                        values=self._result.values.copy(),
                        trend_signals=self._result.trend_signals.copy(),
                        current_trend=self._result.current_trend,
                        current_trend_value=self._result.current_trend_value,
                        dynamic_periods=self._result.dynamic_periods.copy(),
                        normalized_adx=self._result.normalized_adx.copy(),
                        stddev_factor=self._result.stddev_factor.copy(),
                        atr=self._result.atr.copy(),
                        dynamic_atr_period=self._result.dynamic_atr_period.copy(),
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

            # 動的期間の処理
            if self.use_dynamic_period:
                if self.dc_detector is None:
                    self.logger.error("動的期間モードですが、ドミナントサイクル検出器が初期化されていません。")
                    error_result = ADXVolResult(
                        values=np.full(current_data_len, np.nan),
                        trend_signals=np.zeros(current_data_len, dtype=np.int8),
                        current_trend='range',
                        current_trend_value=0,
                        dynamic_periods=np.array([]),
                        normalized_adx=np.array([]),
                        stddev_factor=np.array([]),
                        atr=np.array([]),
                        dynamic_atr_period=np.array([]),
                        fixed_threshold=self.fixed_threshold,
                        trend_state=np.array([])
                    )
                    return error_result
                
                # ドミナントサイクルの計算
                dc_values = self.dc_detector.calculate(df_data)
                period_array = np.asarray(dc_values, dtype=np.float64)
                
                # DC値を保存（get_dynamic_periods用）
                self._last_dc_values = period_array.copy()
                
                # 最大期間の取得
                max_period_value_float = np.nanmax(period_array)
                if np.isnan(max_period_value_float):
                    max_period_value = self.period  # デフォルト期間を使用
                    self.logger.warning("ドミナントサイクルが全てNaNです。デフォルト期間を使用します。")
                else:
                    max_period_value = int(max_period_value_float)
                    if max_period_value < 1:
                        max_period_value = self.period
                
                # 動的正規化ADXの計算
                normalized_adx = calculate_dynamic_nadx_numba(h, l, c, period_array, max_period_value)
            else:
                # 固定期間モード
                normalized_adx = calculate_normalized_adx(h, l, c, self.period)
                period_array = np.full_like(normalized_adx, self.period)

            # ATRの計算
            atr_result = self.atr_indicator.calculate(df_data)
            atr = atr_result.values  # ATR値
            dynamic_atr_period = self.atr_indicator.get_dynamic_periods() if self.use_dynamic_atr_period else np.full_like(atr, self.atr_period)

            # ATR標準偏差係数の計算
            stddev_factor = calculate_stddev_factor(atr)

            # ADX_VOLインデックスの計算
            adx_vol_values = calculate_adx_vol_index(normalized_adx, stddev_factor)

            # 値を0-1の範囲に正規化（0以下は0、1以上は1にクリップ）
            adx_vol_values = np.clip(adx_vol_values, 0.0, 1.0)

            # トレンド状態の計算（固定しきい値使用）
            trend_state = calculate_fixed_threshold_trend_state(
                adx_vol_values, self.fixed_threshold
            )

            # トレンド判定
            trend_signals = calculate_trend_signals_with_range(adx_vol_values, self.slope_index, self.range_threshold)
            trend_index, trend_value = calculate_current_trend_with_range(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]

            # 結果オブジェクトを作成
            result = ADXVolResult(
                values=adx_vol_values,
                trend_signals=trend_signals,
                current_trend=current_trend,
                current_trend_value=trend_value,
                dynamic_periods=period_array,
                normalized_adx=normalized_adx,
                stddev_factor=stddev_factor,
                atr=atr,
                dynamic_atr_period=dynamic_atr_period,
                fixed_threshold=self.fixed_threshold,
                trend_state=trend_state
            )

            self._result = result
            self._cache[data_hash] = self._result
            self._values = adx_vol_values # Indicatorクラスの標準出力

            return ADXVolResult(
                values=result.values.copy(),
                trend_signals=result.trend_signals.copy(),
                current_trend=result.current_trend,
                current_trend_value=result.current_trend_value,
                dynamic_periods=result.dynamic_periods.copy(),
                normalized_adx=result.normalized_adx.copy(),
                stddev_factor=result.stddev_factor.copy(),
                atr=result.atr.copy(),
                dynamic_atr_period=result.dynamic_atr_period.copy(),
                fixed_threshold=result.fixed_threshold,
                trend_state=result.trend_state.copy()
            )

        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ADX_VOL計算中にエラー: {error_msg}\n{stack_trace}")
            # エラー時は空の結果を返す
            self._result = None
            self._values = np.full(current_data_len, np.nan)
            if data_hash in self._cache:
                del self._cache[data_hash]
            
            error_result = ADXVolResult(
                 values=np.full(current_data_len, np.nan),
                 trend_signals=np.zeros(current_data_len, dtype=np.int8),
                 current_trend='range',
                 current_trend_value=0,
                 dynamic_periods=np.full(current_data_len, np.nan),
                 normalized_adx=np.full(current_data_len, np.nan),
                 stddev_factor=np.full(current_data_len, np.nan),
                 atr=np.full(current_data_len, np.nan),
                 dynamic_atr_period=np.full(current_data_len, np.nan),
                 fixed_threshold=self.fixed_threshold,
                 trend_state=np.full(current_data_len, np.nan)
            )
            return error_result

    # --- Getter Methods ---
    def get_values(self) -> Optional[np.ndarray]:
        """ADX_VOL値のみを取得する（後方互換性のため）"""
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

    def get_result(self) -> Optional[ADXVolResult]:
        """計算結果全体を取得する"""
        return self._result

    def get_dynamic_periods(self) -> np.ndarray:
        """動的期間の値を取得する"""
        if self._result is None: return np.array([])
        return self._result.dynamic_periods.copy()

    def get_normalized_adx(self) -> np.ndarray:
        """正規化ADX値を取得する"""
        if self._result is None: return np.array([])
        return self._result.normalized_adx.copy()

    def get_stddev_factor(self) -> np.ndarray:
        """ATR標準偏差係数の値を取得する"""
        if self._result is None: return np.array([])
        return self._result.stddev_factor.copy()

    def get_atr(self) -> np.ndarray:
        """ATR値を取得する"""
        if self._result is None: return np.array([])
        return self._result.atr.copy()

    def get_dynamic_atr_period(self) -> np.ndarray:
        """動的ATR期間を取得する"""
        if self._result is None: return np.array([])
        return self._result.dynamic_atr_period.copy()

    def get_fixed_threshold(self) -> float:
        """固定しきい値を取得する"""
        return self.fixed_threshold

    def get_trend_state(self) -> np.ndarray:
        """トレンド状態を取得する（1=トレンド、0=レンジ、NaN=不明）"""
        if self._result is None: return np.array([])
        return self._result.trend_state.copy()

    def reset(self) -> None:
        """インジケーターの状態をリセットする"""
        super().reset()
        if self.dc_detector and hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset()
        self.atr_indicator.reset()
        self._result = None
        self._cache = {}
        self._last_dc_values = None
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。") 