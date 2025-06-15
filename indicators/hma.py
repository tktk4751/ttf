#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit
import math
import traceback # For detailed error logging

# Assuming these base classes/helpers exist in the same directory or are importable
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .ehlers_unified_dc import EhlersUnifiedDC
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
    # Dummy EhlersUnifiedDC class for fallback
    class EhlersUnifiedDC:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return np.full(len(data), 20.0)
        def reset(self): pass


class HMAResult(NamedTuple):
    """HMA計算結果"""
    values: np.ndarray
    trend_signals: np.ndarray  # 1=up, -1=down, 0=range
    current_trend: str  # 'up', 'down', 'range'
    current_trend_value: int  # 1, -1, 0


@jit(nopython=True, cache=True)
def calculate_wma_numba(prices: np.ndarray, period: int) -> np.ndarray:
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

    if period <= 0 or length < period: # WMA requires at least 'period' data points
        return result

    weights = np.arange(1.0, period + 1.0) # 重み 1, 2, ..., period
    weights_sum = period * (period + 1.0) / 2.0 # 合計 = n(n+1)/2

    if weights_sum < 1e-9: # ゼロ除算防止 (period=0 の場合など)
        return result

    for i in range(period - 1, length): # Start calculation from the first full window
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
        # else: result[i] remains NaN

    return result

@jit(nopython=True, cache=True)
def calculate_hma_numba(prices: np.ndarray, period: int) -> np.ndarray:
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

    if period <= 1 or length == 0: # HMAには期間2以上が必要
        return result

    period_half = int(period / 2)
    period_sqrt = int(math.sqrt(period))

    # HMA計算に必要な最小期間をチェック
    min_len_for_hma = period_sqrt - 1 + max(period_half, period)
    if length < min_len_for_hma:
         #print(f"Data length {length} too short for HMA period {period}. Min required: {min_len_for_hma}")
         return result # データが短すぎる場合、計算不可

    if period_half <= 0 or period_sqrt <= 0:
        return result # 不正な中間期間

    # 中間WMAを計算
    wma_half = calculate_wma_numba(prices, period_half)
    wma_full = calculate_wma_numba(prices, period)

    # 差分系列を計算: 2 * WMA(period/2) - WMA(period)
    # WMA計算から生じるNaNを処理
    diff_wma = np.full(length, np.nan)
    # NaNでない場合のみ計算
    valid_indices = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff_wma[valid_indices] = 2.0 * wma_half[valid_indices] - wma_full[valid_indices]
    
    # If diff_wma contains only NaNs (e.g., due to short input or all NaNs in wmas), return NaNs
    if np.all(np.isnan(diff_wma)):
        #print("Intermediate diff_wma calculation resulted in all NaNs.")
        return result # No valid data to compute final WMA

    # 最終的なHMAを計算: WMA(diff_wma, sqrt(period))
    hma_values = calculate_wma_numba(diff_wma, period_sqrt)

    return hma_values


@jit(nopython=True, cache=True)
def calculate_dynamic_hma_numba(
    prices: np.ndarray,
    period_array: np.ndarray,
    max_period: int
) -> np.ndarray:
    """
    動的期間でHMAを計算する（Numba JIT）
    
    Args:
        prices: 価格の配列
        period_array: 各時点での期間の配列
        max_period: 最大期間（計算開始位置用）
    
    Returns:
        HMA値の配列
    """
    length = len(prices)
    result = np.full(length, np.nan)
    
    # 各時点での動的HMAを計算
    for i in range(max_period, length):
        # その時点でのドミナントサイクルから決定された期間を取得
        curr_period = int(period_array[i])
        if curr_period < 2:  # HMAには最低2期間が必要
            curr_period = 2
            
        # 現在位置までの価格データを取得（効率化のためウィンドウサイズを制限）
        start_idx = max(0, i - curr_period * 2)  # HMAには余裕をもったウィンドウが必要
        window = prices[start_idx:i+1]
        
        # 現在の期間でHMAを計算
        hma_values = calculate_hma_numba(window, curr_period)
        
        # 最後の値をHMAとして使用
        if len(hma_values) > 0 and not np.isnan(hma_values[-1]):
            result[i] = hma_values[-1]
    
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


class HMA(Indicator):
    """
    ハル移動平均線 (Hull Moving Average - HMA) インジケーター

    HMA = WMA(2 * WMA(Price, period/2) - WMA(Price, period), sqrt(period))

    特徴:
    - 非常になめらかで、価格変動への反応が速い移動平均線。
    - 計算に使用する価格ソースを選択可能 ('close', 'hlc3', etc.)。
    - 固定期間または動的期間（ドミナントサイクル）での計算に対応。
    - トレンド判定機能：slope_index期間前との比較でトレンド方向を判定
    - range状態判定：統計的閾値を使用した高精度なレンジ相場検出
    """

    def __init__(self,
                 period: int = 21,
                 src_type: str = 'hlc3',
                 use_dynamic_period: bool = True,
                 cycle_part: float = 1.0,
                 detector_type: str = 'absolute_ultimate',
                 max_cycle: int = 233,
                 min_cycle: int = 13,
                 max_output: int = 120,
                 min_output: int = 5,
                 slope_index: int = 1,
                 range_threshold: float = 0.005,
                 lp_period: int = 10,
                 hp_period: int = 48):
        """
        コンストラクタ

        Args:
            period: 期間（固定期間モードで使用、2以上である必要があります）
            src_type: 価格ソース ('close', 'hlc3', etc.)
            use_dynamic_period: 動的期間を使用するかどうか
            cycle_part: サイクル部分の倍率（動的期間モード用）
            detector_type: 検出器タイプ（動的期間モード用）
            max_cycle: 最大サイクル期間（動的期間モード用）
            min_cycle: 最小サイクル期間（動的期間モード用）
            max_output: 最大出力値（動的期間モード用）
            min_output: 最小出力値（動的期間モード用）
            slope_index: トレンド判定期間 (1以上、デフォルト: 1)
            range_threshold: range判定の基本閾値（デフォルト: 0.005 = 0.5%）
            lp_period: ローパスフィルター期間（動的期間モード用、デフォルト: 10）
            hp_period: ハイパスフィルター期間（動的期間モード用、デフォルト: 48）
        """
        if not isinstance(period, int) or period <= 1:
             raise ValueError(f"期間は2以上の整数である必要がありますが、'{period}'が指定されました")

        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_period else ""
        super().__init__(f"HMA(p={period},src={src_type}{dynamic_str},slope={slope_index},range_th={range_threshold:.3f})")

        self.period = period
        self.src_type = src_type.lower() # Ensure lowercase for consistency
        self.use_dynamic_period = use_dynamic_period
        self.slope_index = slope_index
        self.range_threshold = range_threshold
        
        # 動的期間モード用パラメータ
        self.cycle_part = cycle_part
        self.detector_type = detector_type
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.lp_period = lp_period
        self.hp_period = hp_period

        self.price_source_extractor = PriceSource()
        
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
        self._result: Optional[HMAResult] = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        # src_typeに基づいて必要なカラムを決定
        required_cols = set()
        st = self.src_type
        if st == 'open': required_cols.add('open')
        elif st == 'high': required_cols.add('high')
        elif st == 'low': required_cols.add('low')
        elif st == 'close': required_cols.add('close')
        elif st == 'hl2': required_cols.update(['high', 'low'])
        elif st == 'hlc3': required_cols.update(['high', 'low', 'close'])
        elif st == 'ohlc4': required_cols.update(['open', 'high', 'low', 'close'])
        # Add other source types if PriceSource supports them
        else: required_cols.add('close') # Default

        data_hash_val = None
        try:
            if isinstance(data, pd.DataFrame):
                # データフレームのカラム名を小文字に統一してチェック
                df_cols_lower = {col.lower(): col for col in data.columns}
                present_cols = [df_cols_lower[req_col] for req_col in required_cols if req_col in df_cols_lower]

                if not present_cols:
                     # 必要なカラムがない場合でも、データ形状などでハッシュを試みる
                    shape_tuple = data.shape
                    first_row_tuple = tuple(data.iloc[0]) if len(data) > 0 else ()
                    last_row_tuple = tuple(data.iloc[-1]) if len(data) > 0 else ()
                    data_repr_tuple = (shape_tuple, first_row_tuple, last_row_tuple)
                    data_hash_val = hash(data_repr_tuple)
                    self.logger.debug(f"必要なカラム {required_cols} が見つかりません。形状と端点でハッシュを計算: {data_hash_val}")
                else:
                    # 必要なカラムのデータのみでハッシュを計算
                    data_values = data[present_cols].values
                    data_hash_val = hash(data_values.tobytes())

            elif isinstance(data, np.ndarray):
                 # NumPy配列の場合、src_typeに応じて列インデックスを決定 (OHLCV想定)
                col_indices = []
                if st == 'open' and data.ndim > 1 and data.shape[1] > 0: col_indices.append(0)
                elif st == 'high' and data.ndim > 1 and data.shape[1] > 1: col_indices.append(1)
                elif st == 'low' and data.ndim > 1 and data.shape[1] > 2: col_indices.append(2)
                elif st == 'close' and data.ndim > 1 and data.shape[1] > 3: col_indices.append(3)
                 # hl2, hlc3 などはDataFrameに変換してから処理するのが安全だが、ここでは単純化
                elif st in ['hl2', 'hlc3', 'ohlc4'] and data.ndim > 1 and data.shape[1] >= 4:
                    # これらのソースタイプは複数列必要なので、全列のハッシュを使うか、
                    # calculate内でDataFrame変換後に取得する方が確実。ここでは全列ハッシュを使用。
                     data_hash_val = hash(data.tobytes())
                elif data.ndim == 1: # 1次元配列ならそれが価格ソースとみなす
                    data_hash_val = hash(data.tobytes())
                
                if data_hash_val is None: # 上記で決まらなかった場合
                    if col_indices and data.ndim == 2 and data.shape[1] > max(col_indices):
                        try:
                            data_values = data[:, col_indices]
                            data_hash_val = hash(data_values.tobytes())
                        except IndexError:
                            data_hash_val = hash(data.tobytes()) # fallback
                    else: # デフォルト (close) or 1D array or ambiguous
                        data_hash_val = hash(data.tobytes())
            else:
                # サポート外のデータ型は文字列表現でハッシュ化
                data_hash_val = hash(str(data))

        except Exception as e:
             self.logger.warning(f"データハッシュ計算中にエラー: {e}. データ全体の文字列表現を使用します。", exc_info=True)
             data_hash_val = hash(str(data)) # fallback

        # パラメータ文字列の作成
        if self.use_dynamic_period:
            param_str = (f"p={self.period}_src={self.src_type}_"
                        f"dynamic={self.detector_type}_{self.max_output}_{self.min_output}")
        else:
            param_str = f"p={self.period}_src={self.src_type}"

        return f"{data_hash_val}_{param_str}"


    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HMAResult:
        """
        HMAを計算する

        Args:
            data: 価格データ (pd.DataFrame or np.ndarray)。OHLC形式を期待。

        Returns:
            HMAResult: HMA値とトレンド情報を含む結果
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
             self.logger.warning("入力データが空です。空の配列を返します。")
             empty_result = HMAResult(
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
                     # self.logger.debug(f"キャッシュヒット: {data_hash}")
                     return HMAResult(
                         values=self._result.values.copy(),
                         trend_signals=self._result.trend_signals.copy(),
                         current_trend=self._result.current_trend,
                         current_trend_value=self._result.current_trend_value
                     )
                else:
                    self.logger.debug(f"キャッシュのデータ長 ({len(self._result.values)}) が現在のデータ長 ({current_data_len}) と異なるため再計算します。 Hash: {data_hash}")
                    # キャッシュを無効化
                    del self._cache[data_hash]
                    self._result = None

            # --- データ検証とソース価格取得 ---
            # PriceSourceを使ってソース価格を取得
            src_prices = PriceSource.calculate_source(data, self.src_type)

            # データ長の検証
            data_length = len(src_prices)
            if data_length == 0:
                self.logger.warning("価格データが空です。空の配列を返します。")
                empty_result = HMAResult(
                    values=np.array([]),
                    trend_signals=np.array([], dtype=np.int8),
                    current_trend='range',
                    current_trend_value=0
                )
                self._result = empty_result
                self._cache[data_hash] = self._result
                return empty_result

            # Numbaのためにnumpy配列かつfloat64であることを保証
            if not isinstance(src_prices, np.ndarray):
                src_prices = np.array(src_prices)
            if src_prices.dtype != np.float64:
                # NaNが含まれている可能性があるため、astypeで変換
                try:
                    src_prices = src_prices.astype(np.float64)
                except ValueError:
                    self.logger.error(f"価格ソース '{self.src_type}' をfloat64に変換できませんでした。NaNが含まれているか、数値以外のデータが存在する可能性があります。")
                    error_result = HMAResult(
                        values=np.full(current_data_len, np.nan),
                        trend_signals=np.zeros(current_data_len, dtype=np.int8),
                        current_trend='range',
                        current_trend_value=0
                    )
                    return error_result

            if self.use_dynamic_period:
                # 動的期間モード
                if self.dc_detector is None:
                    self.logger.error("動的期間モードですが、ドミナントサイクル検出器が初期化されていません。")
                    error_result = HMAResult(
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
                    if max_period_value < 2:  # HMAには最低2期間が必要
                        max_period_value = max(2, self.period)
                
                # データ長の検証
                min_len_required = max_period_value + int(math.sqrt(max_period_value)) - 1
                if data_length < min_len_required:
                    self.logger.warning(f"データ長({data_length})が必要な最大期間({min_len_required})より短いため、計算できません。")
                    error_result = HMAResult(
                        values=np.full(current_data_len, np.nan),
                        trend_signals=np.zeros(current_data_len, dtype=np.int8),
                        current_trend='range',
                        current_trend_value=0
                    )
                    return error_result
                
                # 動的HMAの計算
                hma_values = calculate_dynamic_hma_numba(src_prices, period_array, max_period_value)
            else:
                # 固定期間モード
                min_len_required = self.period + int(math.sqrt(self.period)) -1 # Roughly the lookback of the final WMA
                if data_length < min_len_required:
                     self.logger.warning(f"データ長 ({data_length}) がHMA期間 ({self.period}) に対して短すぎる可能性があります (最小目安: {min_len_required})。結果はNaNが多くなります。")
                     # 計算は続行するが、結果はほぼNaNになる

                # --- HMA計算 (Numba) ---
                # Numba関数には float64 配列を渡す
                if src_prices.dtype != np.float64:
                     self.logger.warning(f"Numba関数に渡す価格データの型が {src_prices.dtype} です。float64に変換します。")
                     try:
                         src_prices = src_prices.astype(np.float64)
                     except ValueError:
                          self.logger.error("最終価格データをfloat64に変換できませんでした。NaN配列を返します。")
                          error_result = HMAResult(
                              values=np.full(current_data_len, np.nan),
                              trend_signals=np.zeros(current_data_len, dtype=np.int8),
                              current_trend='range',
                              current_trend_value=0
                          )
                          return error_result

                # Ensure input array is C-contiguous for Numba
                if not src_prices.flags['C_CONTIGUOUS']:
                    src_prices = np.ascontiguousarray(src_prices)

                # self.logger.debug(f"calculate_hma_numba を呼び出します。period={self.period}, data length={len(src_prices)}")
                hma_values = calculate_hma_numba(src_prices, self.period)
                # self.logger.debug(f"calculate_hma_numba 完了。結果の長さ: {len(hma_values)}")

            # トレンド判定
            trend_signals = calculate_trend_signals_with_range(hma_values, self.slope_index, self.range_threshold)
            trend_index, trend_value = calculate_current_trend_with_range(trend_signals)
            trend_names = ['range', 'up', 'down']
            current_trend = trend_names[trend_index]

            result = HMAResult(
                values=hma_values,
                trend_signals=trend_signals,
                current_trend=current_trend,
                current_trend_value=trend_value
            )

            self._result = result
            self._cache[data_hash] = self._result
            return HMAResult(
                values=result.values.copy(),
                trend_signals=result.trend_signals.copy(),
                current_trend=result.current_trend,
                current_trend_value=result.current_trend_value
            )

        except Exception as e:
            # Log the full error with traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"HMA '{self.name}' 計算中に予期せぬエラー: {error_msg}\n{stack_trace}")
            # Return NaNs matching the input data length
            self._result = None # Clear result on error
            error_result = HMAResult(
                values=np.full(current_data_len, np.nan),
                trend_signals=np.zeros(current_data_len, dtype=np.int8),
                current_trend='range',
                current_trend_value=0
            )
            return error_result

    def get_values(self) -> Optional[np.ndarray]:
        """HMA値のみを取得する（後方互換性のため）"""
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
