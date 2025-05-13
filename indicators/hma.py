#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional
import numpy as np
import pandas as pd
from numba import jit
import math
import traceback # For detailed error logging

# Assuming these base classes/helpers exist in the same directory or are importable
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .kalman_filter import KalmanFilter
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
        def ensure_dataframe(self, data):
             if isinstance(data, np.ndarray):
                 # Basic conversion assuming OHLC or just Close
                 cols = ['open', 'high', 'low', 'close'] if data.ndim == 2 and data.shape[1] >= 4 else ['close']
                 if data.ndim == 1:
                     return pd.DataFrame({'close': data})
                 elif data.ndim == 2 and data.shape[1] < len(cols):
                     return pd.DataFrame(data, columns=cols[:data.shape[1]])
                 elif data.ndim == 2:
                      return pd.DataFrame(data, columns=cols)
                 else:
                      raise ValueError("Cannot convert numpy array to DataFrame")
             elif isinstance(data, pd.DataFrame):
                 return data
             else:
                 raise TypeError("Data must be pd.DataFrame or np.ndarray")
        def get_price(self, df, src_type):
            src_type = src_type.lower()
            if src_type == 'close': return df['close'].values
            elif src_type == 'open': return df['open'].values
            elif src_type == 'high': return df['high'].values
            elif src_type == 'low': return df['low'].values
            elif src_type == 'hl2': return ((df['high'] + df['low']) / 2).values
            elif src_type == 'hlc3': return ((df['high'] + df['low'] + df['close']) / 3).values
            elif src_type == 'ohlc4': return ((df['open'] + df['high'] + df['low'] + df['close']) / 4).values
            else: return df['close'].values # Default to close
    class KalmanFilter:
        def __init__(self, **kwargs): self.logger = self._get_logger()
        def calculate(self, data): return None # Dummy implementation
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)


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


class HMA(Indicator):
    """
    ハル移動平均線 (Hull Moving Average - HMA) インジケーター

    HMA = WMA(2 * WMA(Price, period/2) - WMA(Price, period), sqrt(period))

    特徴:
    - 非常になめらかで、価格変動への反応が速い移動平均線。
    - 計算に使用する価格ソースを選択可能 ('close', 'hlc3', etc.)。
    - オプションで価格ソースにカルマンフィルターを適用可能（デフォルトは無効）。
    """

    def __init__(self,
                 period: int = 9,
                 src_type: str = 'close',
                 use_kalman_filter: bool = False, # HMA自体が平滑化するためデフォルト無効
                 kalman_measurement_noise: float = 1.0,
                 kalman_process_noise: float = 0.01,
                 kalman_n_states: int = 5):
        """
        コンストラクタ

        Args:
            period: 期間 (2以上である必要があります)
            src_type: 価格ソース ('close', 'hlc3', etc.)
            use_kalman_filter: ソース価格にカルマンフィルターを適用するかどうか
            kalman_*: カルマンフィルターのパラメータ
        """
        if not isinstance(period, int) or period <= 1:
             raise ValueError(f"期間は2以上の整数である必要がありますが、'{period}'が指定されました")

        kalman_str = f"_kalman={'Y' if use_kalman_filter else 'N'}" if use_kalman_filter else ""
        super().__init__(f"HMA(p={period},src={src_type}{kalman_str})")

        self.period = period
        self.src_type = src_type.lower() # Ensure lowercase for consistency
        self.use_kalman_filter = use_kalman_filter
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_process_noise = kalman_process_noise
        self.kalman_n_states = kalman_n_states

        self.price_source_extractor = PriceSource()
        self.kalman_filter = None
        if self.use_kalman_filter:
            # KalmanFilterの初期化時に自身のロガーを渡すなど、必要に応じて調整
            self.kalman_filter = KalmanFilter(
                price_source=self.src_type, # KalmanFilterもソースタイプを知る必要がある場合
                measurement_noise=self.kalman_measurement_noise,
                process_noise=self.kalman_process_noise,
                n_states=self.kalman_n_states
                # logger=self.logger # 必要であればロガーを共有
            )

        self._cache = {}
        self._result: Optional[np.ndarray] = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する (ALMAから流用し、HMA用に調整)"""
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
        param_str = (
            f"p={self.period}_src={self.src_type}_"
            f"kalman={self.use_kalman_filter}"
        )
        # カルマンフィルター使用時のみ、関連パラメータをハッシュに含める
        if self.use_kalman_filter:
            param_str += f"_{self.kalman_measurement_noise}_{self.kalman_process_noise}_{self.kalman_n_states}"

        return f"{data_hash_val}_{param_str}"


    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        HMAを計算する

        Args:
            data: 価格データ (pd.DataFrame or np.ndarray)。OHLC形式を期待。

        Returns:
            HMA値の配列 (np.ndarray)
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
             self.logger.warning("入力データが空です。空の配列を返します。")
             return np.array([])
             
        try:
            data_hash = self._get_data_hash(data)

            # キャッシュチェック
            if data_hash in self._cache and self._result is not None:
                 # データ長が一致するか確認
                if len(self._result) == current_data_len:
                     # self.logger.debug(f"キャッシュヒット: {data_hash}")
                     return self._result.copy() # Return a copy to prevent external modification
                else:
                    self.logger.debug(f"キャッシュのデータ長 ({len(self._result)}) が現在のデータ長 ({current_data_len}) と異なるため再計算します。 Hash: {data_hash}")
                    # キャッシュを無効化
                    del self._cache[data_hash]
                    self._result = None

            # --- データ検証とソース価格取得 ---
            # PriceSourceにDataFrameを渡すことを保証
            prices_df = self.price_source_extractor.ensure_dataframe(data)
            if prices_df is None or prices_df.empty:
                self.logger.warning("DataFrameへの変換または価格ソースの抽出に失敗しました。NaN配列を返します。")
                return np.full(current_data_len, np.nan)

            src_prices = self.price_source_extractor.get_price(prices_df, self.src_type)

             # src_pricesがNoneまたは空でないか確認
            if src_prices is None or len(src_prices) == 0:
                self.logger.warning(f"価格ソース '{self.src_type}' の取得に失敗またはデータが空です。NaN配列を返します。")
                return np.full(current_data_len, np.nan)

            # Numbaのためにnumpy配列かつfloat64であることを保証
            if not isinstance(src_prices, np.ndarray):
                src_prices = np.array(src_prices)
            if src_prices.dtype != np.float64:
                # NaNが含まれている可能性があるため、astypeで変換
                 try:
                     src_prices = src_prices.astype(np.float64)
                 except ValueError:
                      self.logger.error(f"価格ソース '{self.src_type}' をfloat64に変換できませんでした。NaNが含まれているか、数値以外のデータが存在する可能性があります。")
                      return np.full(current_data_len, np.nan)


            # --- Optional Kalman Filtering ---
            effective_src_prices = src_prices
            if self.use_kalman_filter and self.kalman_filter:
                self.logger.debug("カルマンフィルターを適用します...")
                # KalmanFilterにはDataFrameを渡す (OHLCが必要な場合があるため)
                kalman_input_data = prices_df
                try:
                    filtered_prices = self.kalman_filter.calculate(kalman_input_data)

                    if filtered_prices is not None and len(filtered_prices) > 0:
                        # フィルター結果がnumpy配列であることを確認
                        if isinstance(filtered_prices, pd.Series):
                            filtered_prices = filtered_prices.values
                        elif not isinstance(filtered_prices, np.ndarray):
                             self.logger.warning("カルマンフィルターの出力が予期しない型です。フィルター結果は使用しません。")
                             filtered_prices = None # Skip using this result

                        # データ長の一致を確認
                        if filtered_prices is not None and len(filtered_prices) == len(src_prices):
                            effective_src_prices = filtered_prices
                            # Ensure float64 for numba
                            if effective_src_prices.dtype != np.float64:
                                 try:
                                     effective_src_prices = effective_src_prices.astype(np.float64)
                                 except ValueError:
                                     self.logger.error("カルマンフィルター適用後のデータをfloat64に変換できませんでした。元の価格を使用します。")
                                     effective_src_prices = src_prices # Revert
                            self.logger.debug("カルマンフィルター適用完了。")
                        elif filtered_prices is not None:
                             self.logger.warning(f"カルマンフィルター適用後のデータ長 ({len(filtered_prices)}) が元のデータ長 ({len(src_prices)}) と異なります。フィルター結果は使用しません。")
                             # effective_src_prices は変更しない (元の src_prices のまま)
                        else:
                             # filtered_prices が None か空だった場合
                             self.logger.warning("カルマンフィルターの計算結果がNoneまたは空です。元のソース価格を使用します。")
                    else:
                        self.logger.warning("カルマンフィルター計算失敗または空の結果。元のソース価格を使用します。")

                except Exception as kf_e:
                    self.logger.error(f"カルマンフィルター計算中にエラーが発生しました: {kf_e}", exc_info=True)
                    # Fallback to using original source prices
                    self.logger.warning("エラーのため、元のソース価格を使用します。")
                    effective_src_prices = src_prices # Ensure fallback

            # 再度データ長の検証 (フィルター適用後)
            data_length = len(effective_src_prices)
            if data_length == 0:
                self.logger.warning("有効な価格データが空です（フィルター適用後？）。空の配列を返します。")
                self._result = np.array([])
                self._cache[data_hash] = self._result
                return self._result.copy()

            # 期間に対するデータ長のチェック
            min_len_required = self.period + int(math.sqrt(self.period)) -1 # Roughly the lookback of the final WMA
            if data_length < min_len_required:
                 self.logger.warning(f"データ長 ({data_length}) がHMA期間 ({self.period}) に対して短すぎる可能性があります (最小目安: {min_len_required})。結果はNaNが多くなります。")
                 # 計算は続行するが、結果はほぼNaNになる

            # --- HMA計算 (Numba) ---
            # Numba関数には float64 配列を渡す
            if effective_src_prices.dtype != np.float64:
                 self.logger.warning(f"Numba関数に渡す価格データの型が {effective_src_prices.dtype} です。float64に変換します。")
                 try:
                     effective_src_prices = effective_src_prices.astype(np.float64)
                 except ValueError:
                      self.logger.error("最終価格データをfloat64に変換できませんでした。NaN配列を返します。")
                      return np.full(current_data_len, np.nan)


            # Ensure input array is C-contiguous for Numba
            if not effective_src_prices.flags['C_CONTIGUOUS']:
                effective_src_prices = np.ascontiguousarray(effective_src_prices)

            # self.logger.debug(f"calculate_hma_numba を呼び出します。period={self.period}, data length={len(effective_src_prices)}")
            hma_values = calculate_hma_numba(effective_src_prices, self.period)
            # self.logger.debug(f"calculate_hma_numba 完了。結果の長さ: {len(hma_values)}")

            self._result = hma_values
            self._cache[data_hash] = self._result
            return self._result.copy() # Return a copy

        except Exception as e:
            # Log the full error with traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"HMA '{self.name}' 計算中に予期せぬエラー: {error_msg}\n{stack_trace}")
            # Return NaNs matching the input data length
            self._result = None # Clear result on error
            return np.full(current_data_len, np.nan)


    def reset(self) -> None:
        """インジケータの状態（キャッシュ、結果、カルマンフィルター）をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}
        if self.kalman_filter and hasattr(self.kalman_filter, 'reset'):
            self.kalman_filter.reset()
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。")
