#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit
import traceback

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .efficiency_ratio import calculate_efficiency_ratio, calculate_efficiency_ratio_for_period
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
    # Dummy efficiency ratio functions
    def calculate_efficiency_ratio(prices, period): return np.zeros(len(prices))
    def calculate_efficiency_ratio_for_period(prices, period): return [0.0]


class KAMAResult(NamedTuple):
    """KAMA計算結果"""
    values: np.ndarray
    is_bullish: np.ndarray
    is_bearish: np.ndarray
    current_trend: str  # 'bullish', 'bearish', 'neutral'
    is_currently_bullish: bool
    is_currently_bearish: bool


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


@jit(nopython=True)
def calculate_smoothing_constant(er: np.ndarray, fast: float, slow: float) -> np.ndarray:
    """
    スムージング定数を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        fast: 速い移動平均の期間から計算した定数
        slow: 遅い移動平均の期間から計算した定数
    
    Returns:
        スムージング定数の配列
    """
    return (er * (fast - slow) + slow) ** 2


@jit(nopython=True)
def calculate_kama(close: np.ndarray, period: int, fast_period: int, slow_period: int) -> np.ndarray:
    """
    カウフマン適応移動平均線を計算する（高速化版）
    
    Args:
        close: 終値の配列
        period: 効率比の計算期間
        fast_period: 速い移動平均の期間
        slow_period: 遅い移動平均の期間
    
    Returns:
        KAMAの配列
    """
    length = len(close)
    kama = np.full(length, np.nan)
    
    if length < period:
        return kama
    
    # 最初のKAMAは単純な平均値
    kama[period-1] = np.mean(close[:period])
    
    # 定数の計算
    fast = 2.0 / (fast_period + 1.0)
    slow = 2.0 / (slow_period + 1.0)
    
    # 各時点でのKAMAを計算
    for i in range(period, length):
        # 効率比の計算
        er = calculate_efficiency_ratio_for_period(close[max(0, i-period+1):i+1], period)[0]
        
        # スムージング定数の計算
        sc = (er * (fast - slow) + slow) ** 2
        
        # KAMAの計算
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama


@jit(nopython=True)
def calculate_dynamic_kama(close: np.ndarray, periods: np.ndarray, fast_period: int, slow_period: int, max_period: int) -> np.ndarray:
    """
    動的期間対応KAMA (Dynamic KAMA) を計算する (Numba JIT)

    Args:
        close: 終値の配列
        periods: 各時点での期間の配列
        fast_period: 速い移動平均の期間
        slow_period: 遅い移動平均の期間
        max_period: 最大期間

    Returns:
        Dynamic KAMA値の配列
    """
    length = len(close)
    kama = np.full(length, np.nan)
    
    if length == 0:
        return kama
    
    # 定数の計算
    fast = 2.0 / (fast_period + 1.0)
    slow = 2.0 / (slow_period + 1.0)
    
    for i in range(length):
        period = int(periods[i])
        if period <= 0 or i < period - 1:
            continue
            
        # 初期値の設定
        if i == period - 1 or np.isnan(kama[i-1]):
            kama[i] = np.mean(close[max(0, i-period+1):i+1])
            continue
            
        # 効率比の計算（簡易版）
        window_start = max(0, i - period + 1)
        window_close = close[window_start:i+1]
        
        # 価格変動の総和
        total_change = abs(window_close[-1] - window_close[0])
        
        # 日々の変動の総和
        daily_changes = 0.0
        for j in range(1, len(window_close)):
            daily_changes += abs(window_close[j] - window_close[j-1])
        
        # 効率比の計算
        if daily_changes > 1e-9:
            er = total_change / daily_changes
        else:
            er = 0.0
            
        # スムージング定数の計算
        sc = (er * (fast - slow) + slow) ** 2
        
        # KAMAの計算
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama


class KaufmanAdaptiveMA(Indicator):
    """
    カウフマン適応移動平均線（KAMA）インジケーター
    
    価格のトレンドの効率性に基づいて、移動平均線の感度を自動的に調整する
    - トレンドが強い時：速い移動平均に近づく
    - トレンドが弱い時：遅い移動平均に近づく
    
    特徴:
    - 計算に使用する価格ソースを選択可能 ('close', 'hlc3', etc.)
    - 動的期間対応：外部で計算された期間配列を使用可能
    - 固定期間でも動的期間でも使用可能
    - トレンド判定機能：slope_index期間前との比較でトレンド方向を判定
    - range状態判定：統計的閾値を使用した高精度なレンジ相場検出
    """
    
    def __init__(self, 
                 period: int = 10, 
                 fast_period: int = 2, 
                 slow_period: int = 34,
                 src_type: str = 'hlc3',
                 slope_index: int = 1,
                 range_threshold: float = 0.005):
        """
        コンストラクタ
        
        Args:
            period: 効率比の計算期間（デフォルト: 10）
            fast_period: 速い移動平均の期間（デフォルト: 2）
            slow_period: 遅い移動平均の期間（デフォルト: 30）
            src_type: 価格ソース ('close', 'hlc3', etc.)
            slope_index: トレンド判定期間 (1以上、デフォルト: 1)
            range_threshold: range判定の基本閾値（デフォルト: 0.005 = 0.5%）
        """
        if not isinstance(slope_index, int) or slope_index < 1:
            raise ValueError(f"slope_indexは1以上の整数である必要がありますが、'{slope_index}'が指定されました")
            
        super().__init__(f"KAMA(p={period},fast={fast_period},slow={slow_period},src={src_type},slope={slope_index},range_th={range_threshold:.3f})")
        self.period = period
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.src_type = src_type.lower()
        self.slope_index = slope_index
        self.range_threshold = range_threshold
        
        self._cache = {}
        self._result: Optional[KAMAResult] = None

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
        param_str = f"p={self.period}_fast={self.fast_period}_slow={self.slow_period}_src={self.src_type}_slope={self.slope_index}_range_th={self.range_threshold:.3f}"

        return f"{data_hash_val}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> KAMAResult:
        """
        固定期間KAMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            KAMAResult: KAMA値とトレンド情報を含む結果
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
            self.logger.warning("入力データが空です。空の結果を返します。")
            return KAMAResult(
                values=np.array([]),
                is_bullish=np.array([], dtype=bool),
                is_bearish=np.array([], dtype=bool),
                current_trend='range',
                is_currently_bullish=False,
                is_currently_bearish=False
            )

        try:
            data_hash = self._get_data_hash(data)

            # キャッシュチェック
            if data_hash in self._cache and self._result is not None:
                # データ長が一致するか確認
                if len(self._result.values) == current_data_len:
                    return KAMAResult(
                        values=self._result.values.copy(),
                        is_bullish=self._result.is_bullish.copy(),
                        is_bearish=self._result.is_bearish.copy(),
                        current_trend=self._result.current_trend,
                        is_currently_bullish=self._result.is_currently_bullish,
                        is_currently_bearish=self._result.is_currently_bearish
                    ) # Return a copy to prevent external modification
                else:
                    self.logger.debug(f"キャッシュのデータ長が異なるため再計算します。")
                    # キャッシュを無効化
                    del self._cache[data_hash]
                    self._result = None

            # --- PriceSourceを使用してソース価格を取得 ---
            src_prices = PriceSource.calculate_source(data, self.src_type)

            # src_pricesがNoneまたは空でないか確認
            if src_prices is None or len(src_prices) == 0:
                self.logger.warning(f"価格ソース '{self.src_type}' の取得に失敗またはデータが空です。NaN結果を返します。")
                return KAMAResult(
                    values=np.full(current_data_len, np.nan),
                    is_bullish=np.full(current_data_len, False, dtype=bool),
                    is_bearish=np.full(current_data_len, False, dtype=bool),
                    current_trend='range',
                    is_currently_bullish=False,
                    is_currently_bearish=False
                )

            # Numbaのためにnumpy配列かつfloat64であることを保証
            if not isinstance(src_prices, np.ndarray):
                src_prices = np.array(src_prices)
            if src_prices.dtype != np.float64:
                try:
                    src_prices = src_prices.astype(np.float64)
                except ValueError:
                    self.logger.error(f"価格ソース '{self.src_type}' をfloat64に変換できませんでした。")
                    return KAMAResult(
                        values=np.full(current_data_len, np.nan),
                        is_bullish=np.full(current_data_len, False, dtype=bool),
                        is_bearish=np.full(current_data_len, False, dtype=bool),
                        current_trend='range',
                        is_currently_bullish=False,
                        is_currently_bearish=False
                    )
        
            # データ長の検証
            data_length = len(src_prices)
            if data_length == 0:
                self.logger.warning("有効な価格データが空です。空の結果を返します。")
                empty_result = KAMAResult(
                    values=np.array([]),
                    is_bullish=np.array([], dtype=bool),
                    is_bearish=np.array([], dtype=bool),
                    current_trend='range',
                    is_currently_bullish=False,
                    is_currently_bearish=False
                )
                self._result = empty_result
                self._cache[data_hash] = self._result
                return empty_result

            if self.period > data_length:
                self.logger.warning(f"効率比計算期間 ({self.period}) がデータ長 ({data_length}) より大きいです。")

            # Ensure input array is C-contiguous for Numba
            if not src_prices.flags['C_CONTIGUOUS']:
                src_prices = np.ascontiguousarray(src_prices)
        
            # KAMAの計算（高速化版）
            kama_values = calculate_kama(
                src_prices,
                self.period,
                self.fast_period,
                self.slow_period
            )
        
            # --- トレンド判定計算（range対応版） ---
            trend_signals = calculate_trend_signals_with_range(
                kama_values, self.slope_index, self.range_threshold
            )

            # 現在のトレンド状態を計算
            trend_index, trend_value = calculate_current_trend_with_range(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]

            # is_bullish/is_bearishは使用しないため、空の配列を設定
            is_bullish = np.full(len(kama_values), False, dtype=bool)
            is_bearish = np.full(len(kama_values), False, dtype=bool)

            result = KAMAResult(
                values=kama_values,
                is_bullish=is_bullish,
                is_bearish=is_bearish,
                current_trend=current_trend,
                is_currently_bullish=(trend_value == 1),
                is_currently_bearish=(trend_value == -1)
            )

            self._result = result
            self._cache[data_hash] = self._result
            return KAMAResult(
                values=result.values.copy(),
                is_bullish=result.is_bullish.copy(),
                is_bearish=result.is_bearish.copy(),
                current_trend=result.current_trend,
                is_currently_bullish=result.is_currently_bullish,
                is_currently_bearish=result.is_currently_bearish
            )

        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"KAMA '{self.name}' 計算中に予期せぬエラー: {error_msg}\n{stack_trace}")
            self._result = None # Clear result on error
            return KAMAResult(
                values=np.full(current_data_len, np.nan),
                is_bullish=np.full(current_data_len, False, dtype=bool),
                is_bearish=np.full(current_data_len, False, dtype=bool),
                current_trend='range',
                is_currently_bullish=False,
                is_currently_bearish=False
            )

    def calculate_with_dynamic_periods(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        periods: Union[np.ndarray, list]
    ) -> KAMAResult:
        """
        動的期間を使用してKAMAを計算する

        Args:
            data: 価格データ (pd.DataFrame or np.ndarray)
            periods: 各時点での期間配列

        Returns:
            KAMAResult: 動的期間KAMA値とトレンド情報を含む結果
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
            self.logger.warning("入力データが空です。空の結果を返します。")
            return KAMAResult(
                values=np.array([]),
                is_bullish=np.array([], dtype=bool),
                is_bearish=np.array([], dtype=bool),
                current_trend='range',
                is_currently_bullish=False,
                is_currently_bearish=False
            )

        try:
            # 期間配列をNumPy配列に変換
            if not isinstance(periods, np.ndarray):
                periods = np.array(periods)
            if periods.dtype != np.float64:
                periods = periods.astype(np.float64)

            # データ長チェック
            if len(periods) != current_data_len:
                self.logger.error(f"期間配列の長さ({len(periods)})がデータ長({current_data_len})と異なります。")
                return KAMAResult(
                    values=np.full(current_data_len, np.nan),
                    is_bullish=np.full(current_data_len, False, dtype=bool),
                    is_bearish=np.full(current_data_len, False, dtype=bool),
                    current_trend='range',
                    is_currently_bullish=False,
                    is_currently_bearish=False
                )

            # --- PriceSourceを使用してソース価格を取得 ---
            src_prices = PriceSource.calculate_source(data, self.src_type)

            if src_prices is None or len(src_prices) == 0:
                self.logger.warning(f"価格ソース '{self.src_type}' の取得に失敗またはデータが空です。NaN結果を返します。")
                return KAMAResult(
                    values=np.full(current_data_len, np.nan),
                    is_bullish=np.full(current_data_len, False, dtype=bool),
                    is_bearish=np.full(current_data_len, False, dtype=bool),
                    current_trend='range',
                    is_currently_bullish=False,
                    is_currently_bearish=False
                )

            # Numba用にfloat64配列に変換
            if not isinstance(src_prices, np.ndarray):
                src_prices = np.array(src_prices)
            if src_prices.dtype != np.float64:
                try:
                    src_prices = src_prices.astype(np.float64)
                except ValueError:
                    self.logger.error(f"価格ソース '{self.src_type}' をfloat64に変換できませんでした。")
                    return KAMAResult(
                        values=np.full(current_data_len, np.nan),
                        is_bullish=np.full(current_data_len, False, dtype=bool),
                        is_bearish=np.full(current_data_len, False, dtype=bool),
                        current_trend='range',
                        is_currently_bullish=False,
                        is_currently_bearish=False
                    )

            # 最大期間を取得
            max_period_value = int(np.nanmax(periods))
            if np.isnan(max_period_value) or max_period_value < 1:
                max_period_value = 1

            # データ長検証
            data_length = len(src_prices)
            if data_length < max_period_value:
                self.logger.warning(f"データ長({data_length})が必要な最大期間({max_period_value})より短いため、計算できません。")
                return KAMAResult(
                    values=np.full(current_data_len, np.nan),
                    is_bullish=np.full(current_data_len, False, dtype=bool),
                    is_bearish=np.full(current_data_len, False, dtype=bool),
                    current_trend='range',
                    is_currently_bullish=False,
                    is_currently_bearish=False
                )

            # Numba用に配列をC-contiguousにする
            if not src_prices.flags['C_CONTIGUOUS']:
                src_prices = np.ascontiguousarray(src_prices)
            if not periods.flags['C_CONTIGUOUS']:
                periods = np.ascontiguousarray(periods)

            # 動的期間KAMA計算
            kama_values = calculate_dynamic_kama(src_prices, periods, self.fast_period, self.slow_period, max_period_value)

            # --- トレンド判定計算（range対応版） ---
            trend_signals = calculate_trend_signals_with_range(
                kama_values, self.slope_index, self.range_threshold
            )

            # 現在のトレンド状態を計算
            trend_index, trend_value = calculate_current_trend_with_range(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]

            # is_bullish/is_bearishは使用しないため、空の配列を設定
            is_bullish = np.full(len(kama_values), False, dtype=bool)
            is_bearish = np.full(len(kama_values), False, dtype=bool)

            return KAMAResult(
                values=kama_values,
                is_bullish=is_bullish,
                is_bearish=is_bearish,
                current_trend=current_trend,
                is_currently_bullish=(trend_value == 1),
                is_currently_bearish=(trend_value == -1)
            )

        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"動的期間KAMA計算中に予期せぬエラー: {error_msg}\n{stack_trace}")
            return KAMAResult(
                values=np.full(current_data_len, np.nan),
                is_bullish=np.full(current_data_len, False, dtype=bool),
                is_bearish=np.full(current_data_len, False, dtype=bool),
                current_trend='range',
                is_currently_bullish=False,
                is_currently_bearish=False
            )

    def get_values(self) -> Optional[np.ndarray]:
        """KAMA値のみを取得する（後方互換性のため）"""
        if self._result is not None:
            return self._result.values.copy()
        return None

    def get_trend_signals(self) -> tuple:
        """トレンド信号を取得する"""
        if self._result is not None:
            return self._result.is_bullish.copy(), self._result.is_bearish.copy()
        return None, None

    def get_current_trend(self) -> str:
        """現在のトレンド状態を取得する"""
        if self._result is not None:
            return self._result.current_trend
        return 'range'

    def is_currently_bullish_trend(self) -> bool:
        """現在が上昇トレンドかどうか"""
        if self._result is not None:
            return self._result.is_currently_bullish
        return False

    def is_currently_bearish_trend(self) -> bool:
        """現在が下降トレンドかどうか"""
        if self._result is not None:
            return self._result.is_currently_bearish
        return False

    def reset(self) -> None:
        """インジケータの状態（キャッシュ、結果）をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。") 