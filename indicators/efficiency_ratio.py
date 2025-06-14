#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit
import traceback
import math

# Assuming these base classes/helpers exist in the same directory or are importable
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .alma import calculate_alma_numba
    from .hma import calculate_hma_numba
    from .zlema import calculate_zlema_numba
    from .ehlers_unified_dc import EhlersUnifiedDC
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
                elif src_type == 'open': return data['open'].values
                elif src_type == 'high': return data['high'].values
                elif src_type == 'low': return data['low'].values
                elif src_type == 'hl2': return ((data['high'] + data['low']) / 2).values
                elif src_type == 'hlc3': return ((data['high'] + data['low'] + data['close']) / 3).values
                elif src_type == 'ohlc4': return ((data['open'] + data['high'] + data['low'] + data['close']) / 4).values
                else: return data['close'].values # Default to close
            else:
                return data[:, 3] if data.ndim > 1 and data.shape[1] > 3 else data
    # Dummy functions for fallback
    def calculate_alma_numba(values, period, offset=0.85, sigma=6.0):
        return np.full_like(values, np.nan)
    def calculate_hma_numba(values, period):
        return np.full_like(values, np.nan)
    def calculate_zlema_numba(values, period):
        return np.full_like(values, np.nan)
    class EhlersUnifiedDC:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return np.full(len(data), 10.0)
        def reset(self): pass


class EfficiencyRatioResult(NamedTuple):
    """EfficiencyRatio計算結果"""
    values: np.ndarray
    trend_signals: np.ndarray  # 1=up, -1=down, 0=range
    current_trend: str  # 'up', 'down', 'range'
    current_trend_value: int  # 1, -1, 0


@njit(fastmath=True, cache=True)
def calculate_efficiency_ratio(change: np.ndarray, volatility: np.ndarray) -> np.ndarray:
    """
    効率比（Efficiency Ratio）を計算する（高速化版）
    
    Args:
        change: 価格変化（終値の差分）の配列
        volatility: ボラティリティ（価格変化の絶対値の合計）の配列
    
    Returns:
        効率比の配列
    """
    return np.abs(change) / (volatility + 1e-10)  # ゼロ除算を防ぐ


@njit(fastmath=True, cache=True)
def calculate_wilder_smoothing(values: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's Smoothing
    
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
def calculate_er_with_smoothing(
    values: np.ndarray,
    period: int,
    smoothing_method: int
) -> np.ndarray:
    """
    指定されたスムージング方法でEfficiency Ratioを平滑化する
    
    Args:
        values: Efficiency Ratio値の配列
        period: スムージング期間
        smoothing_method: スムージング方法 (0: なし, 1: Wilder's, 2: HMA, 3: ALMA, 4: ZLEMA)
    
    Returns:
        平滑化されたEfficiency Ratio値の配列
    """
    if smoothing_method == 0:  # スムージングなし
        return values
    elif smoothing_method == 1:  # Wilder's Smoothing
        return calculate_wilder_smoothing(values, period)
    elif smoothing_method == 2:  # HMA
        return calculate_hma_numba(values, period)
    elif smoothing_method == 3:  # ALMA
        return calculate_alma_numba(values, period, 0.85, 6.0)  # デフォルトパラメータ
    elif smoothing_method == 4:  # ZLEMA
        return calculate_zlema_numba(values, period)
    else:
        # デフォルトはスムージングなし
        return values


@njit(fastmath=True, cache=True)
def calculate_efficiency_ratio_for_period(prices: np.ndarray, period: int) -> np.ndarray:
    """
    指定された期間の効率比（ER）を計算する（高速化版）
    
    Args:
        prices: 価格の配列（closeやhlc3などの計算済みソース）
        period: 計算期間
    
    Returns:
        効率比の配列（0-1の範囲）
        - 1に近いほど効率的な価格変動（強いトレンド）
        - 0に近いほど非効率な価格変動（レンジ・ノイズ）
    """
    length = len(prices)
    er = np.zeros(length)
    
    for i in range(period, length):
        change = prices[i] - prices[i-period]
        volatility = 0.0
        
        # ボラティリティの計算（価格変化の絶対値の合計）
        for j in range(i-period, i):
            volatility += abs(prices[j+1] - prices[j])
        
        if volatility > 1e-10:  # ゼロ除算防止
            er[i] = abs(change) / volatility
        else:
            er[i] = 0.0
    
    return er


@njit(fastmath=True, cache=True)
def calculate_dynamic_er_numba(
    prices: np.ndarray,
    period_array: np.ndarray,
    max_period: int
) -> np.ndarray:
    """
    動的期間でEfficiency Ratioを計算する（Numba JIT）
    
    Args:
        prices: 価格の配列
        period_array: 各時点での期間の配列
        max_period: 最大期間（計算開始位置用）
    
    Returns:
        Efficiency Ratio値の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)
    
    # 各時点での動的ERを計算
    for i in range(max_period, length):
        # その時点でのドミナントサイクルから決定された期間を取得
        curr_period = int(period_array[i])
        if curr_period < 1:
            curr_period = 1
            
        # 現在位置から必要なデータを取得
        if i >= curr_period:
            change = prices[i] - prices[i-curr_period]
            volatility = 0.0
            
            # ボラティリティの計算
            for j in range(i-curr_period, i):
                volatility += abs(prices[j+1] - prices[j])
            
            if volatility > 1e-10:  # ゼロ除算防止
                result[i] = abs(change) / volatility
            else:
                result[i] = 0.0
    
    return result


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


class EfficiencyRatio(Indicator):
    """
    効率比（Efficiency Ratio）インジケーター
    
    価格変動の効率性を測定する指標
    - 1に近いほど効率的な価格変動（強いトレンド）
    - 0に近いほど非効率な価格変動（レンジ・ノイズ）
    
    特徴:
    - 固定期間または動的期間（ドミナントサイクル）での計算に対応
    - Wilder's、HMA、ALMA、ZLEMAによる平滑化（オプション）
    - トレンド判定機能：slope_index期間前との比較でトレンド方向を判定
    - range状態判定：統計的閾値を使用した高精度なレンジ相場検出
    
    使用方法：
    - 0.618以上: 効率的な価格変動（強いトレンド）
    - 0.382以下: 非効率な価格変動（レンジ・ノイズ）
    """
    
    def __init__(self, 
                 period: int = 5,
                 src_type: str = 'hlc3',
                 smoothing_method: str = 'hma',
                 use_dynamic_period: bool = True,
                 cycle_part: float = 1.0,
                 detector_type: str = 'absolute_ultimate',
                 max_cycle: int = 120,
                 min_cycle: int = 5,
                 max_output: int = 120,
                 min_output: int = 5,
                 slope_index: int = 3,
                 range_threshold: float = 0.005,
                 smoother_period: int = 13,
                 lp_period: int = 5,
                 hp_period: int = 120):
        """
        コンストラクタ
        
        Args:
            period: 期間（固定期間モード時）
            src_type: 価格ソース ('close', 'hlc3', etc.)
            smoothing_method: スムージング方法 ('none', 'wilder', 'hma', 'alma', 'zlema')
            use_dynamic_period: 動的期間を使用するかどうか
            cycle_part: サイクル部分の倍率（動的期間モード用）
            detector_type: 検出器タイプ（動的期間モード用）
            max_cycle: 最大サイクル期間（動的期間モード用）
            min_cycle: 最小サイクル期間（動的期間モード用）
            max_output: 最大出力値（動的期間モード用）
            min_output: 最小出力値（動的期間モード用）
            slope_index: トレンド判定期間 (1以上、デフォルト: 1)
            range_threshold: range判定の基本閾値（デフォルト: 0.005 = 0.5%）
            smoother_period: スムージング期間（デフォルト: 14）
            lp_period: ローパスフィルター期間（動的期間モード用、デフォルト: 10）
            hp_period: ハイパスフィルター期間（動的期間モード用、デフォルト: 48）
        """
        # スムージング方法の検証と変換
        smoothing_methods = {'none': 0, 'wilder': 1, 'hma': 2, 'alma': 3, 'zlema': 4}
        if smoothing_method.lower() not in smoothing_methods:
            raise ValueError(f"サポートされていないスムージング方法: {smoothing_method}. 使用可能: {list(smoothing_methods.keys())}")
        
        self.smoothing_method_str = smoothing_method.lower()
        self.smoothing_method_int = smoothing_methods[self.smoothing_method_str]
        
        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_period else ""
        smooth_str = f"_smooth({smoothing_method})" if smoothing_method != 'none' else ""
        super().__init__(f"ER(p={period},src={src_type}{dynamic_str}{smooth_str},slope={slope_index},range_th={range_threshold:.3f})")
        
        self.period = period
        self.src_type = src_type
        self.use_dynamic_period = use_dynamic_period
        self.slope_index = slope_index
        self.range_threshold = range_threshold
        self.smoother_period = smoother_period
        
        # 動的期間モード用パラメータ
        self.cycle_part = cycle_part
        self.detector_type = detector_type
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.lp_period = lp_period
        self.hp_period = hp_period
        
        # ドミナントサイクル検出器（動的期間モード用）
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
        
        self._cache = {}
        self._result: Optional[EfficiencyRatioResult] = None

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
        if self.use_dynamic_period:
            param_str = (f"p={self.period}_src={self.src_type}_smooth={self.smoothing_method_str}_dynamic={self.detector_type}_{self.max_output}_{self.min_output}_"
                        f"slope={self.slope_index}_range_th={self.range_threshold:.3f}")
        else:
            param_str = f"p={self.period}_src={self.src_type}_smooth={self.smoothing_method_str}_slope={self.slope_index}_range_th={self.range_threshold:.3f}"

        return f"{data_hash_val}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> EfficiencyRatioResult:
        """
        効率比を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
        
        Returns:
            EfficiencyRatioResult: ER値とトレンド情報を含む結果
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
            self.logger.warning("入力データが空です。空の結果を返します。")
            empty_result = EfficiencyRatioResult(
                values=np.array([]),
                trend_signals=np.array([], dtype=np.int8),
                current_trend='range',
                current_trend_value=0
            )
            return empty_result
            
        try:
            data_hash = self._get_data_hash(data)

            # キャッシュチェック
            if data_hash in self._cache and self._result is not None:
                # データ長が一致するか確認
                if len(self._result.values) == current_data_len:
                    return EfficiencyRatioResult(
                        values=self._result.values.copy(),
                        trend_signals=self._result.trend_signals.copy(),
                        current_trend=self._result.current_trend,
                        current_trend_value=self._result.current_trend_value
                    )
                else:
                    self.logger.debug(f"キャッシュのデータ長が異なるため再計算します。")
                    # キャッシュを無効化
                    del self._cache[data_hash]
                    self._result = None

            # PriceSourceを使って価格データを取得
            prices = PriceSource.calculate_source(data, self.src_type)
            
            # Numbaのためにnumpy配列かつfloat64であることを保証
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)
            if prices.dtype != np.float64:
                prices = prices.astype(np.float64)

            # データ長の検証
            data_length = len(prices)
            if data_length == 0:
                self.logger.warning("価格データが空です。空の結果を返します。")
                empty_result = EfficiencyRatioResult(
                    values=np.array([]),
                    trend_signals=np.array([], dtype=np.int8),
                    current_trend='range',
                    current_trend_value=0
                )
                self._result = empty_result
                self._cache[data_hash] = self._result
                return empty_result

            if self.use_dynamic_period:
                # 動的期間モード
                if self.dc_detector is None:
                    self.logger.error("動的期間モードですが、ドミナントサイクル検出器が初期化されていません。")
                    error_result = EfficiencyRatioResult(
                        values=np.full(current_data_len, np.nan),
                        trend_signals=np.zeros(current_data_len, dtype=np.int8),
                        current_trend='range',
                        current_trend_value=0
                    )
                    return error_result
                
                # ドミナントサイクルの計算
                dc_values = self.dc_detector.calculate(data)
                period_array = np.asarray(dc_values, dtype=np.float64)
                
                # DC値を保存（get_dynamic_periods用）
                self._last_dc_values = period_array.copy()
                
                # デバッグ情報の出力
                valid_dc = period_array[~np.isnan(period_array)]
                if len(valid_dc) > 0:
                    self.logger.info(f"動的期間統計 - 平均: {valid_dc.mean():.1f}, 範囲: {valid_dc.min():.0f} - {valid_dc.max():.0f}")
                else:
                    self.logger.warning("ドミナントサイクル値が全てNaNです。")
                
                # 最大期間の取得
                max_period_value_float = np.nanmax(period_array)
                if np.isnan(max_period_value_float):
                    max_period_value = self.period  # デフォルト期間を使用
                    self.logger.warning("ドミナントサイクルが全てNaNです。デフォルト期間を使用します。")
                else:
                    max_period_value = int(max_period_value_float)
                    if max_period_value < 1:
                        max_period_value = self.period
                
                # データ長の検証
                min_len_required = max_period_value + 1
                if data_length < min_len_required:
                    self.logger.warning(f"データ長({data_length})が必要な最大期間({min_len_required})より短いため、計算できません。")
                    error_result = EfficiencyRatioResult(
                        values=np.full(current_data_len, np.nan),
                        trend_signals=np.zeros(current_data_len, dtype=np.int8),
                        current_trend='range',
                        current_trend_value=0
                    )
                    return error_result
                
                # 動的ERの計算
                er_values = calculate_dynamic_er_numba(prices, period_array, max_period_value)
            else:
                # 固定期間モード
                if self.period > data_length:
                    self.logger.warning(f"期間 ({self.period}) がデータ長 ({data_length}) より大きいです。")
                
                # 固定期間でのER計算
                er_values = calculate_efficiency_ratio_for_period(prices, self.period)

            # スムージングの適用
            if self.smoothing_method_str != 'none':
                er_values = calculate_er_with_smoothing(er_values, self.smoother_period, self.smoothing_method_int)

            # 値を0-1の範囲に正規化（0以下は0、1以上は1にクリップ）
            er_values = np.clip(er_values, 0.0, 1.0)

            # トレンド判定
            trend_signals = calculate_trend_signals_with_range(er_values, self.slope_index, self.range_threshold)
            trend_index, trend_value = calculate_current_trend_with_range(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]

            result = EfficiencyRatioResult(
                values=er_values,
                trend_signals=trend_signals,
                current_trend=current_trend,
                current_trend_value=trend_value
            )

            # 計算結果を保存
            self._result = result
            self._cache[data_hash] = self._result
            return EfficiencyRatioResult(
                values=result.values.copy(),
                trend_signals=result.trend_signals.copy(),
                current_trend=result.current_trend,
                current_trend_value=result.current_trend_value
            )
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"EfficiencyRatio '{self.name}' 計算中に予期せぬエラー: {error_msg}\n{stack_trace}")
            # Return NaNs matching the input data length
            self._result = None # Clear result on error
            error_result = EfficiencyRatioResult(
                values=np.full(current_data_len, np.nan),
                trend_signals=np.zeros(current_data_len, dtype=np.int8),
                current_trend='range',
                current_trend_value=0
            )
            return error_result

    def get_values(self) -> Optional[np.ndarray]:
        """ER値のみを取得する（後方互換性のため）"""
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

    def get_dynamic_periods(self) -> np.ndarray:
        """
        動的期間の値を取得する（動的期間モードのみ）
        
        Returns:
            動的期間の配列
        """
        if not self.use_dynamic_period:
            return np.array([])
        
        # 最後に計算されたドミナントサイクル値を返す
        if self._last_dc_values is not None:
            return self._last_dc_values.copy()
        
        return np.array([])

    def reset(self) -> None:
        """インジケータの状態（キャッシュ、結果）をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}
        self._last_dc_values = None
        if self.dc_detector and hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset()
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。")

    def __str__(self) -> str:
        """文字列表現"""
        return self.name 