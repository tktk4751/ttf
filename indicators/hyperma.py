#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit
import traceback # For detailed error logging

# Assuming these base classes/helpers exist in the same directory or are importable
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .hyper_smoother import calculate_hyper_smoother_numba
    from .cycle.ehlers_unified_dc import EhlersUnifiedDC
except ImportError:
    # Fallback for potential execution context issues, adjust as needed
    # This might happen if running the file directly without the package structure
    print("Warning: Could not import from relative path. Assuming base classes are available.")
    # Define dummy classes if needed for static analysis, but real execution needs correct imports
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
    
    # Fallback hyper smoother function
    @jit(nopython=True)
    def calculate_hyper_smoother_numba(data: np.ndarray, length: int) -> np.ndarray:
        size = len(data)
        smoothed = np.zeros(size)
        
        # 初期値の設定
        f28 = np.zeros(size)
        f30 = np.zeros(size)
        vC = np.zeros(size)
        
        f38 = np.zeros(size)
        f40 = np.zeros(size)
        v10 = np.zeros(size)
        
        f48 = np.zeros(size)
        f50 = np.zeros(size)
        v14 = np.zeros(size)
        
        # パラメータの計算
        f18 = 3.0 / (length + 2.0)
        f20 = 1.0 - f18
        
        for i in range(1, size):
            # フィルタリング（1段階目）
            f28[i] = f20 * f28[i-1] + f18 * data[i]
            f30[i] = f18 * f28[i] + f20 * f30[i-1]
            vC[i] = f28[i] * 1.5 - f30[i] * 0.5
            
            # フィルタリング（2段階目）
            f38[i] = f20 * f38[i-1] + f18 * vC[i]
            f40[i] = f18 * f38[i] + f20 * f40[i-1]
            v10[i] = f38[i] * 1.5 - f40[i] * 0.5
            
            # フィルタリング（3段階目）
            f48[i] = f20 * f48[i-1] + f18 * v10[i]
            f50[i] = f18 * f48[i] + f20 * f50[i-1]
            v14[i] = f48[i] * 1.5 - f50[i] * 0.5
            
            # 最終的な平滑化値
            smoothed[i] = v14[i]
        
        return smoothed

    # Dummy EhlersUnifiedDC class for fallback
    class EhlersUnifiedDC:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return np.full(len(data), 20.0)


class HyperMAResult(NamedTuple):
    """HyperMA計算結果"""
    values: np.ndarray
    is_bullish: np.ndarray
    is_bearish: np.ndarray
    current_trend: str  # 'bullish', 'bearish', 'neutral'
    is_currently_bullish: bool
    is_currently_bearish: bool


@jit(nopython=True, cache=True)
def calculate_hyperma_numba(prices: np.ndarray, length: int) -> np.ndarray:
    """
    ハイパースムーサー移動平均線 (HyperMA) を計算する (Numba JIT)

    Args:
        prices: 価格の配列 (np.float64 を想定)
        length: 期間 (1以上)

    Returns:
        HyperMA値の配列
    """
    data_length = len(prices)
    result = np.full(data_length, np.nan)

    if length <= 0 or data_length == 0:
        return result

    if data_length < length:
        # データが短すぎる場合、計算可能な範囲で実行
        return calculate_hyper_smoother_numba(prices, length)

    # ハイパースムーサーを適用
    smoothed_values = calculate_hyper_smoother_numba(prices, length)

    return smoothed_values


@jit(nopython=True, cache=True)
def calculate_dynamic_hyperma_numba(prices: np.ndarray, periods: np.ndarray, max_period: int) -> np.ndarray:
    """
    動的期間対応ハイパースムーサー移動平均線 (Dynamic HyperMA) を計算する (Numba JIT)

    Args:
        prices: 価格の配列 (np.float64 を想定)
        periods: 各時点での期間の配列
        max_period: 最大期間

    Returns:
        Dynamic HyperMA値の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)
    
    if length == 0:
        return result
    
    # 動的期間でのハイパースムーサー適用
    # 各時点で異なる期間を使用するため、逐次計算
    for i in range(length):
        period = int(periods[i])
        if period <= 0 or i < period - 1:
            continue
            
        # 現在位置までのウィンドウデータを取得
        window_start = max(0, i - period + 1)
        window_data = prices[window_start:i + 1]
        
        # ウィンドウにNaNが含まれていないかチェック
        has_nan = False
        for j in range(len(window_data)):
            if np.isnan(window_data[j]):
                has_nan = True
                break
        
        if not has_nan and len(window_data) >= period:
            # ハイパースムーサーを適用
            smoothed_window = calculate_hyper_smoother_numba(window_data, period)
            # 最新の値を結果に設定
            if len(smoothed_window) > 0:
                result[i] = smoothed_window[-1]
    
    return result


@jit(nopython=True, cache=True)
def calculate_trend_signals(values: np.ndarray, slope_index: int) -> tuple:
    """
    トレンド信号を計算する (Numba JIT)
    
    Args:
        values: インジケーター値の配列
        slope_index: スロープ判定期間
    
    Returns:
        tuple: (is_bullish, is_bearish) のNumPy配列
    """
    length = len(values)
    is_bullish = np.full(length, False)
    is_bearish = np.full(length, False)
    
    for i in range(slope_index, length):
        if not np.isnan(values[i]) and not np.isnan(values[i - slope_index]):
            if values[i] > values[i - slope_index]:
                is_bullish[i] = True
                is_bearish[i] = False
            elif values[i] < values[i - slope_index]:
                is_bullish[i] = False
                is_bearish[i] = True
            # Equal case: both remain False
    
    return is_bullish, is_bearish


@jit(nopython=True, cache=True)
def calculate_current_trend(is_bullish: np.ndarray, is_bearish: np.ndarray) -> tuple:
    """
    現在のトレンド状態を計算する (Numba JIT)
    
    Args:
        is_bullish: 上昇トレンド判定配列
        is_bearish: 下降トレンド判定配列
    
    Returns:
        tuple: (current_trend_index, is_currently_bullish, is_currently_bearish)
               current_trend_index: 0=neutral, 1=bullish, 2=bearish
    """
    length = len(is_bullish)
    if length == 0:
        return 0, False, False  # neutral, not bullish, not bearish
    
    # 最新の値を取得
    latest_bullish = is_bullish[-1]
    latest_bearish = is_bearish[-1]
    
    if latest_bullish:
        return 1, True, False   # bullish
    elif latest_bearish:
        return 2, False, True   # bearish
    else:
        return 0, False, False  # neutral


class HyperMA(Indicator):
    """
    ハイパースムーサー移動平均線 (HyperMA) インジケーター

    ハイパースムーサーアルゴリズムを使用して、3段階のフィルタリングにより
    非常に滑らかな移動平均線を提供します。

    特徴:
    - 非常になめらかで、ノイズを効果的に除去する移動平均線。
    - 計算に使用する価格ソースを選択可能 ('close', 'hlc3', etc.)。
    - 動的期間対応：外部で計算された期間配列を使用可能。
    - 固定期間でも動的期間でも使用可能。
    - Ehlers Unified DCを使用した自動動的期間計算オプション。
    - トレンド判定機能：slope_index期間前との比較でトレンド方向を判定。
    """

    def __init__(self,
                 length: int = 14,
                 src_type: str = 'close',
                 slope_index: int = 1,
                 # 動的期間計算オプション
                 use_dynamic_periods: bool = False,
                 # Ehlers Unified DCパラメータ
                 ehlers_detector_type: str = 'phac_e',
                 ehlers_cycle_part: float = 0.5,
                 ehlers_max_cycle: int = 50,
                 ehlers_min_cycle: int = 6,
                 ehlers_max_output: int = 34,
                 ehlers_min_output: int = 8,
                 ehlers_src_type: str = 'hlc3',
                 ehlers_use_kalman: bool = False,
                 ehlers_lp_period: int = 5,
                 ehlers_hp_period: int = 55):
        """
        コンストラクタ

        Args:
            length: 期間 (1以上である必要があります)
            src_type: 価格ソース ('close', 'hlc3', etc.)
            slope_index: トレンド判定期間 (1以上、デフォルト: 1)
            use_dynamic_periods: 動的期間を使用するかどうか
            ehlers_detector_type: Ehlers DC検出器タイプ
            ehlers_cycle_part: Ehlerサイクル部分
            ehlers_max_cycle: Ehlers最大サイクル期間
            ehlers_min_cycle: Ehlers最小サイクル期間
            ehlers_max_output: Ehlers最大出力値（動的期間の最大値）
            ehlers_min_output: Ehlers最小出力値（動的期間の最小値）
            ehlers_src_type: EhlersDCの価格ソース
            ehlers_use_kalman: Ehlersでカルマンフィルター使用
            ehlers_lp_period: Ehlersローパスフィルター期間
            ehlers_hp_period: Ehlersハイパスフィルター期間
        """
        if not isinstance(length, int) or length <= 0:
             raise ValueError(f"期間は1以上の整数である必要がありますが、'{length}'が指定されました")
        
        if not isinstance(slope_index, int) or slope_index < 1:
            raise ValueError(f"slope_indexは1以上の整数である必要がありますが、'{slope_index}'が指定されました")

        super().__init__(f"HyperMA(l={length},src={src_type},slope={slope_index},dyn={'Y' if use_dynamic_periods else 'N'})")

        self.length = length
        self.src_type = src_type.lower() # Ensure lowercase for consistency
        self.slope_index = slope_index

        # 動的期間関連パラメータ
        self.use_dynamic_periods = use_dynamic_periods
        
        # Ehlers Unified DC関連パラメータ
        self.ehlers_detector_type = ehlers_detector_type
        self.ehlers_cycle_part = ehlers_cycle_part
        self.ehlers_max_cycle = ehlers_max_cycle
        self.ehlers_min_cycle = ehlers_min_cycle
        self.ehlers_max_output = ehlers_max_output
        self.ehlers_min_output = ehlers_min_output
        self.ehlers_src_type = ehlers_src_type
        self.ehlers_use_kalman = ehlers_use_kalman
        self.ehlers_lp_period = ehlers_lp_period
        self.ehlers_hp_period = ehlers_hp_period
        
        # Ehlers Unified DCインスタンス（動的期間使用時のみ）
        self._ehlers_dc = None
        if self.use_dynamic_periods:
            try:
                self._ehlers_dc = EhlersUnifiedDC(
                    detector_type=self.ehlers_detector_type,
                    cycle_part=self.ehlers_cycle_part,
                    max_cycle=self.ehlers_max_cycle,
                    min_cycle=self.ehlers_min_cycle,
                    max_output=self.ehlers_max_output,
                    min_output=self.ehlers_min_output,
                    src_type=self.ehlers_src_type,
                    use_kalman_filter=self.ehlers_use_kalman,
                    lp_period=self.ehlers_lp_period,
                    hp_period=self.ehlers_hp_period
                )
            except Exception as e:
                self.logger.warning(f"Ehlers Unified DCの初期化に失敗しました: {e}. 固定期間モードにフォールバックします。")
                self.use_dynamic_periods = False
                self._ehlers_dc = None

        self._cache = {}
        self._result: Optional[HyperMAResult] = None

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

        # パラメータ文字列の作成（動的期間パラメータも含む）
        param_str = f"l={self.length}_src={self.src_type}_slope={self.slope_index}_dyn={self.use_dynamic_periods}"
        if self.use_dynamic_periods:
            param_str += f"_ehlDet={self.ehlers_detector_type}"

        return f"{data_hash_val}_{param_str}"


    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperMAResult:
        """
        HyperMA を計算する（固定期間または動的期間）

        Args:
            data: 価格データ (pd.DataFrame or np.ndarray)。OHLC形式を期待。

        Returns:
            HyperMAResult: HyperMA値とトレンド情報を含む結果
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
             self.logger.warning("入力データが空です。空の結果を返します。")
             return HyperMAResult(
                 values=np.array([]),
                 is_bullish=np.array([], dtype=bool),
                 is_bearish=np.array([], dtype=bool),
                 current_trend='neutral',
                 is_currently_bullish=False,
                 is_currently_bearish=False
             )
             
        try:
            # 動的期間を使用する場合
            if self.use_dynamic_periods and self._ehlers_dc is not None:
                self.logger.debug("動的期間モードでHyperMAを計算中...")
                
                # Ehlers Unified DCでドミナントサイクルを計算
                dominant_cycles = self._ehlers_dc.calculate(data)
                
                # ドミナントサイクルを動的期間に変換（範囲制限）
                dynamic_periods = np.clip(dominant_cycles, self.ehlers_min_output, self.ehlers_max_output)
                
                # NaN値を前の有効値で埋める
                dynamic_periods = pd.Series(dynamic_periods).ffill().fillna(self.length).values
                
                # 動的期間でHyperMAを計算
                return self.calculate_with_dynamic_periods(data, dynamic_periods)
            
            # 固定期間モードの場合（元のロジック）
            data_hash = self._get_data_hash(data)

            # キャッシュチェック
            if data_hash in self._cache and self._result is not None:
                 # データ長が一致するか確認
                if len(self._result.values) == current_data_len:
                     return HyperMAResult(
                         values=self._result.values.copy(),
                         is_bullish=self._result.is_bullish.copy(),
                         is_bearish=self._result.is_bearish.copy(),
                         current_trend=self._result.current_trend,
                         is_currently_bullish=self._result.is_currently_bullish,
                         is_currently_bearish=self._result.is_currently_bearish
                     ) # Return a copy to prevent external modification
                else:
                    self.logger.debug(f"キャッシュのデータ長 ({len(self._result.values)}) が現在のデータ長 ({current_data_len}) と異なるため再計算します。")
                    # キャッシュを無効化
                    del self._cache[data_hash]
                    self._result = None

            # --- PriceSourceを使用してソース価格を取得 ---
            src_prices = PriceSource.calculate_source(data, self.src_type)

             # src_pricesがNoneまたは空でないか確認
            if src_prices is None or len(src_prices) == 0:
                self.logger.warning(f"価格ソース '{self.src_type}' の取得に失敗またはデータが空です。NaN結果を返します。")
                return HyperMAResult(
                    values=np.full(current_data_len, np.nan),
                    is_bullish=np.full(current_data_len, False, dtype=bool),
                    is_bearish=np.full(current_data_len, False, dtype=bool),
                    current_trend='neutral',
                    is_currently_bullish=False,
                    is_currently_bearish=False
                )

            # Numbaのためにnumpy配列かつfloat64であることを保証
            if not isinstance(src_prices, np.ndarray):
                src_prices = np.array(src_prices)
            if src_prices.dtype != np.float64:
                # NaNが含まれている可能性があるため、astypeで変換
                 try:
                     src_prices = src_prices.astype(np.float64)
                 except ValueError:
                      self.logger.error(f"価格ソース '{self.src_type}' をfloat64に変換できませんでした。NaNが含まれているか、数値以外のデータが存在する可能性があります。")
                      return HyperMAResult(
                          values=np.full(current_data_len, np.nan),
                          is_bullish=np.full(current_data_len, False, dtype=bool),
                          is_bearish=np.full(current_data_len, False, dtype=bool),
                          current_trend='neutral',
                          is_currently_bullish=False,
                          is_currently_bearish=False
                      )

            # データ長の検証
            data_length = len(src_prices)
            if data_length == 0:
                self.logger.warning("有効な価格データが空です。空の結果を返します。")
                empty_result = HyperMAResult(
                    values=np.array([]),
                    is_bullish=np.array([], dtype=bool),
                    is_bearish=np.array([], dtype=bool),
                    current_trend='neutral',
                    is_currently_bullish=False,
                    is_currently_bearish=False
                )
                self._result = empty_result
                self._cache[data_hash] = self._result
                return empty_result

            # 期間に対するデータ長のチェック
            if data_length < self.length:
                 self.logger.warning(f"データ長 ({data_length}) がHyperMA期間 ({self.length}) に対して短すぎる可能性があります。結果はNaNが多くなります。")
                 # 計算は続行するが、結果はほぼNaNになる

            # --- HyperMA計算 (Numba) ---
            # Ensure input array is C-contiguous for Numba
            if not src_prices.flags['C_CONTIGUOUS']:
                src_prices = np.ascontiguousarray(src_prices)

            hyperma_values = calculate_hyperma_numba(src_prices, self.length)

            # --- トレンド判定計算 ---
            is_bullish, is_bearish = calculate_trend_signals(hyperma_values, self.slope_index)

            # 現在のトレンド状態を計算
            trend_index, currently_bullish, currently_bearish = calculate_current_trend(is_bullish, is_bearish)
            trend_names = ['neutral', 'bullish', 'bearish']
            current_trend = trend_names[trend_index]

            result = HyperMAResult(
                values=hyperma_values,
                is_bullish=is_bullish,
                is_bearish=is_bearish,
                current_trend=current_trend,
                is_currently_bullish=currently_bullish,
                is_currently_bearish=currently_bearish
            )

            self._result = result
            self._cache[data_hash] = self._result
            return HyperMAResult(
                values=result.values.copy(),
                is_bullish=result.is_bullish.copy(),
                is_bearish=result.is_bearish.copy(),
                current_trend=result.current_trend,
                is_currently_bullish=result.is_currently_bullish,
                is_currently_bearish=result.is_currently_bearish
            ) # Return a copy

        except Exception as e:
            # Log the full error with traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"HyperMA '{self.name}' 計算中に予期せぬエラー: {error_msg}\n{stack_trace}")
            # Return NaNs matching the input data length
            self._result = None # Clear result on error
            return HyperMAResult(
                values=np.full(current_data_len, np.nan),
                is_bullish=np.full(current_data_len, False, dtype=bool),
                is_bearish=np.full(current_data_len, False, dtype=bool),
                current_trend='neutral',
                is_currently_bullish=False,
                is_currently_bearish=False
            )

    def calculate_with_dynamic_periods(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        periods: Union[np.ndarray, list]
    ) -> HyperMAResult:
        """
        動的期間を使用してHyperMAを計算する

        Args:
            data: 価格データ (pd.DataFrame or np.ndarray)
            periods: 各時点での期間配列

        Returns:
            HyperMAResult: 動的期間HyperMA値とトレンド情報を含む結果
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
            self.logger.warning("入力データが空です。空の結果を返します。")
            return HyperMAResult(
                values=np.array([]),
                is_bullish=np.array([], dtype=bool),
                is_bearish=np.array([], dtype=bool),
                current_trend='neutral',
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
                return HyperMAResult(
                    values=np.full(current_data_len, np.nan),
                    is_bullish=np.full(current_data_len, False, dtype=bool),
                    is_bearish=np.full(current_data_len, False, dtype=bool),
                    current_trend='neutral',
                    is_currently_bullish=False,
                    is_currently_bearish=False
                )

            # --- PriceSourceを使用してソース価格を取得 ---
            src_prices = PriceSource.calculate_source(data, self.src_type)

            if src_prices is None or len(src_prices) == 0:
                self.logger.warning(f"価格ソース '{self.src_type}' の取得に失敗またはデータが空です。NaN結果を返します。")
                return HyperMAResult(
                    values=np.full(current_data_len, np.nan),
                    is_bullish=np.full(current_data_len, False, dtype=bool),
                    is_bearish=np.full(current_data_len, False, dtype=bool),
                    current_trend='neutral',
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
                    return HyperMAResult(
                        values=np.full(current_data_len, np.nan),
                        is_bullish=np.full(current_data_len, False, dtype=bool),
                        is_bearish=np.full(current_data_len, False, dtype=bool),
                        current_trend='neutral',
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
                return HyperMAResult(
                    values=np.full(current_data_len, np.nan),
                    is_bullish=np.full(current_data_len, False, dtype=bool),
                    is_bearish=np.full(current_data_len, False, dtype=bool),
                    current_trend='neutral',
                    is_currently_bullish=False,
                    is_currently_bearish=False
                )

            # Numba用に配列をC-contiguousにする
            if not src_prices.flags['C_CONTIGUOUS']:
                src_prices = np.ascontiguousarray(src_prices)
            if not periods.flags['C_CONTIGUOUS']:
                periods = np.ascontiguousarray(periods)

            # 動的期間HyperMA計算
            hyperma_values = calculate_dynamic_hyperma_numba(src_prices, periods, max_period_value)

            # --- トレンド判定計算 ---
            is_bullish, is_bearish = calculate_trend_signals(hyperma_values, self.slope_index)

            # 現在のトレンド状態を計算
            trend_index, currently_bullish, currently_bearish = calculate_current_trend(is_bullish, is_bearish)
            trend_names = ['neutral', 'bullish', 'bearish']
            current_trend = trend_names[trend_index]

            return HyperMAResult(
                values=hyperma_values,
                is_bullish=is_bullish,
                is_bearish=is_bearish,
                current_trend=current_trend,
                is_currently_bullish=currently_bullish,
                is_currently_bearish=currently_bearish
            )

        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"動的期間HyperMA計算中に予期せぬエラー: {error_msg}\n{stack_trace}")
            return HyperMAResult(
                values=np.full(current_data_len, np.nan),
                is_bullish=np.full(current_data_len, False, dtype=bool),
                is_bearish=np.full(current_data_len, False, dtype=bool),
                current_trend='neutral',
                is_currently_bullish=False,
                is_currently_bearish=False
            )

    def get_values(self) -> Optional[np.ndarray]:
        """HyperMA値のみを取得する（後方互換性のため）"""
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
        return 'neutral'

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
        if self._ehlers_dc is not None:
            self._ehlers_dc.reset()
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。") 