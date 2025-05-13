#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional
import numpy as np
import pandas as pd
from numba import jit
import traceback

# Import base classes and hyper_smoother
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .kalman_filter import KalmanFilter
    from .hyper_smoother import hyper_smoother, calculate_hyper_smoother_numba # Import the smoother function
except ImportError:
    # Fallback imports and dummy classes (as in HMA)
    print("Warning: Could not import from relative path. Assuming base classes/functions are available.")
    class Indicator: # Dummy
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class PriceSource: # Dummy
        def ensure_dataframe(self, data):
            if isinstance(data, np.ndarray):
                 cols = ['open', 'high', 'low', 'close'] if data.ndim == 2 and data.shape[1] >= 4 else ['close']
                 if data.ndim == 1: return pd.DataFrame({'close': data})
                 elif data.ndim == 2 and data.shape[1] < len(cols): return pd.DataFrame(data, columns=cols[:data.shape[1]])
                 elif data.ndim == 2: return pd.DataFrame(data, columns=cols)
                 else: raise ValueError("Cannot convert numpy array to DataFrame")
            elif isinstance(data, pd.DataFrame): return data
            else: raise TypeError("Data must be pd.DataFrame or np.ndarray")
        def get_price(self, df, src_type):
            st = src_type.lower()
            if st == 'close': return df['close'].values
            elif st == 'hlc3': return ((df['high'] + df['low'] + df['close']) / 3).values
            # Add other sources as needed
            else: return df['close'].values # Default
    class KalmanFilter: # Dummy
        def __init__(self, **kwargs): self.logger = self._get_logger()
        def calculate(self, data): return None # Dummy
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    # Dummy hyper_smoother if import fails
    @jit(nopython=True)
    def calculate_hyper_smoother_numba(data: np.ndarray, length: int) -> np.ndarray:
        size = len(data)
        smoothed = np.zeros(size)
        f28, f30, vC = np.zeros(size), np.zeros(size), np.zeros(size)
        f38, f40, v10 = np.zeros(size), np.zeros(size), np.zeros(size)
        f48, f50, v14 = np.zeros(size), np.zeros(size), np.zeros(size)
        f18 = 3.0 / (length + 2.0)
        f20 = 1.0 - f18
        for i in range(1, size):
            f28[i] = f20 * f28[i-1] + f18 * data[i]
            f30[i] = f18 * f28[i] + f20 * f30[i-1]
            vC[i] = f28[i] * 1.5 - f30[i] * 0.5
            f38[i] = f20 * f38[i-1] + f18 * vC[i]
            f40[i] = f18 * f38[i] + f20 * f40[i-1]
            v10[i] = f38[i] * 1.5 - f40[i] * 0.5
            f48[i] = f20 * f48[i-1] + f18 * v10[i]
            f50[i] = f18 * f48[i] + f20 * f50[i-1]
            v14[i] = f48[i] * 1.5 - f50[i] * 0.5
            smoothed[i] = v14[i]
            if np.isnan(data[i]): # Basic NaN propagation
                 smoothed[i] = np.nan
        smoothed[0] = np.nan # Usually first value is not reliable
        return smoothed
    def hyper_smoother(data: Union[pd.Series, np.ndarray], length: int = 14) -> np.ndarray:
        if isinstance(data, pd.Series): values = data.values
        else: values = data
        if len(values) < length: raise ValueError("Data length must be greater than period")
        # Basic NaN handling before passing to Numba
        values_nan_handled = np.where(np.isnan(values), 0.0, values) # Replace NaN temporarily
        smoothed = calculate_hyper_smoother_numba(values_nan_handled, length)
        # Re-introduce NaNs where original data had NaNs
        smoothed[np.isnan(values)] = np.nan
        return smoothed


class HyperMA(Indicator):
    """
    ハイパー移動平均線 (HyperMA) インジケーター

    価格ソースをハイパースムーサーで平滑化し、その結果を直接返します。
    （以前のバージョンではSMAを適用していましたが、冗長なため削除されました）

    特徴:
    - ハイパースムーサーによる強力なノイズ除去。
    - 計算に使用する価格ソースを選択可能 ('close', 'hlc3', etc.)。
    - オプションで元の価格ソースにカルマンフィルターを適用可能。
    """

    def __init__(self,
                 period: int = 14,
                 src_type: str = 'close',
                 use_kalman_filter: bool = False,
                 kalman_measurement_noise: float = 1.0,
                 kalman_process_noise: float = 0.01,
                 kalman_n_states: int = 5):
        """
        コンストラクタ

        Args:
            period: ハイパースムーサーと単純移動平均 (SMA) の共通期間
            src_type: 価格ソース ('close', 'hlc3', etc.)
            use_kalman_filter: ソース価格にカルマンフィルターを適用するかどうか
            kalman_*: カルマンフィルターのパラメータ
        """
        if not isinstance(period, int) or period <= 0:
             raise ValueError(f"期間は正の整数である必要がありますが、'{period}'が指定されました")

        kalman_str = f"_kalman={'Y' if use_kalman_filter else 'N'}" if use_kalman_filter else ""
        super().__init__(f"HyperMA(p={period},src={src_type}{kalman_str})")

        self.period = period
        self.src_type = src_type.lower()
        self.use_kalman_filter = use_kalman_filter
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_process_noise = kalman_process_noise
        self.kalman_n_states = kalman_n_states

        self.price_source_extractor = PriceSource()
        self.kalman_filter = None
        if self.use_kalman_filter:
            self.kalman_filter = KalmanFilter(
                price_source=self.src_type,
                measurement_noise=self.kalman_measurement_noise,
                process_noise=self.kalman_process_noise,
                n_states=self.kalman_n_states
            )

        self._cache = {}
        self._result: Optional[np.ndarray] = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        # src_typeに基づいて必要なカラムを決定 (HMAと同様)
        required_cols = set()
        st = self.src_type
        if st == 'open': required_cols.add('open')
        elif st == 'high': required_cols.add('high')
        elif st == 'low': required_cols.add('low')
        elif st == 'close': required_cols.add('close')
        elif st == 'hl2': required_cols.update(['high', 'low'])
        elif st == 'hlc3': required_cols.update(['high', 'low', 'close'])
        elif st == 'ohlc4': required_cols.update(['open', 'high', 'low', 'close'])
        else: required_cols.add('close') # Default

        data_hash_val = None
        try:
            # データハッシュ計算ロジックは HMA と同様
            if isinstance(data, pd.DataFrame):
                df_cols_lower = {col.lower(): col for col in data.columns}
                present_cols = [df_cols_lower[req_col] for req_col in required_cols if req_col in df_cols_lower]
                if not present_cols:
                    shape_tuple = data.shape
                    first_row_tuple = tuple(data.iloc[0]) if len(data) > 0 else ()
                    last_row_tuple = tuple(data.iloc[-1]) if len(data) > 0 else ()
                    data_hash_val = hash((shape_tuple, first_row_tuple, last_row_tuple))
                else:
                    data_values = data[present_cols].values
                    data_hash_val = hash(data_values.tobytes())
            elif isinstance(data, np.ndarray):
                data_hash_val = hash(data.tobytes()) # NumPy はシンプルにバイト列でハッシュ
            else:
                data_hash_val = hash(str(data))
        except Exception as e:
             self.logger.warning(f"データハッシュ計算中にエラー: {e}. フォールバックを使用します。", exc_info=True)
             data_hash_val = hash(str(data))

        # パラメータ文字列の作成 (period を使用)
        param_str = (
            f"p={self.period}_src={self.src_type}_"
            f"kalman={self.use_kalman_filter}"
        )
        if self.use_kalman_filter:
            param_str += f"_{self.kalman_measurement_noise}_{self.kalman_process_noise}_{self.kalman_n_states}"

        return f"{data_hash_val}_{param_str}"

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        HyperMAを計算する

        Args:
            data: 価格データ (pd.DataFrame or np.ndarray)

        Returns:
            HyperMA値の配列 (np.ndarray)
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
            self.logger.warning("入力データが空です。空の配列を返します。")
            return np.array([])

        try:
            data_hash = self._get_data_hash(data)

            if data_hash in self._cache and self._result is not None:
                if len(self._result) == current_data_len:
                    return self._result.copy()
                else:
                    self.logger.debug(f"キャッシュデータ長不一致のため再計算: {data_hash}")
                    del self._cache[data_hash]
                    self._result = None

            prices_df = self.price_source_extractor.ensure_dataframe(data)
            if prices_df is None or prices_df.empty:
                self.logger.warning("DataFrame変換/ソース抽出失敗。NaN配列を返します。")
                return np.full(current_data_len, np.nan)

            src_prices = self.price_source_extractor.get_price(prices_df, self.src_type)
            if src_prices is None or len(src_prices) == 0:
                self.logger.warning(f"価格ソース '{self.src_type}' 取得失敗/空。NaN配列を返します。")
                return np.full(current_data_len, np.nan)

            if not isinstance(src_prices, np.ndarray):
                src_prices = np.array(src_prices)
            if src_prices.dtype != np.float64:
                try:
                    src_prices = src_prices.astype(np.float64)
                except ValueError:
                    self.logger.error(f"価格ソース '{self.src_type}' をfloat64に変換できませんでした。")
                    return np.full(current_data_len, np.nan)

            effective_src_prices = src_prices
            if self.use_kalman_filter and self.kalman_filter:
                self.logger.debug("カルマンフィルターを適用します...")
                try:
                    filtered_prices = self.kalman_filter.calculate(prices_df)
                    if filtered_prices is not None and len(filtered_prices) == len(src_prices):
                        if isinstance(filtered_prices, pd.Series): filtered_prices = filtered_prices.values
                        if filtered_prices.dtype != np.float64:
                            filtered_prices = filtered_prices.astype(np.float64)
                        effective_src_prices = filtered_prices
                        self.logger.debug("カルマンフィルター適用完了。")
                    else:
                        self.logger.warning("カルマンフィルター計算失敗/結果長不一致。元のソース価格を使用します。")
                except Exception as kf_e:
                    self.logger.error(f"カルマンフィルター計算中にエラー: {kf_e}", exc_info=True)
                    self.logger.warning("エラーのため、元のソース価格を使用します。")

            data_length = len(effective_src_prices)
            if data_length < self.period:
                self.logger.warning(f"データ長 ({data_length}) が期間 ({self.period}) 未満です。NaN配列を返します。")
                return np.full(current_data_len, np.nan)
            
            if not effective_src_prices.flags['C_CONTIGUOUS']:
                effective_src_prices = np.ascontiguousarray(effective_src_prices)
            
            nan_mask = np.isnan(effective_src_prices)
            temp_prices = pd.Series(effective_src_prices).ffill().bfill().values
            if temp_prices.dtype != np.float64:
                 temp_prices = temp_prices.astype(np.float64)
            
            self.logger.debug(f"ハイパースムーサーを適用します (期間={self.period})...")
            smoothed_prices = calculate_hyper_smoother_numba(temp_prices, self.period)
            smoothed_prices[nan_mask] = np.nan
            self.logger.debug("ハイパースムーサー適用完了。")

            # --- SMA Calculation --- (このセクションを削除またはコメントアウト)
            # if len(smoothed_prices) < self.period:
            #     self.logger.warning(f"平滑化後のデータ長 ({len(smoothed_prices)}) が期間 ({self.period}) 未満です。NaNが多くなります。")
            #
            # if smoothed_prices.dtype != np.float64:
            #      try:
            #          smoothed_prices = smoothed_prices.astype(np.float64)
            #      except ValueError:
            #          self.logger.error("平滑化後のデータをfloat64に変換できませんでした。NaN配列を返します。")
            #          return np.full(current_data_len, np.nan)
            #
            # if not smoothed_prices.flags['C_CONTIGUOUS']:
            #     smoothed_prices = np.ascontiguousarray(smoothed_prices)
            #
            # self.logger.debug(f"SMAを計算します (期間={self.period})...")
            # hyper_ma_values = calculate_sma_numba(smoothed_prices, self.period) # この行を削除
            # self.logger.debug("SMA計算完了。")

            # hyper_smoother の結果をそのまま使用する
            hyper_ma_values = smoothed_prices # この行を追加/変更

            self._result = hyper_ma_values
            self._cache[data_hash] = self._result
            return self._result.copy()

        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"HyperMA '{self.name}' 計算中に予期せぬエラー: {error_msg}\n{stack_trace}")
            self._result = None
            return np.full(current_data_len, np.nan)

    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}
        if self.kalman_filter and hasattr(self.kalman_filter, 'reset'):
            self.kalman_filter.reset()
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。")

# # --- Example Usage --- 
# if __name__ == '__main__':
#     import logging
#     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     # サンプルデータ作成 (HMAと同様)
#     data_len = 200
#     close_prices = np.linspace(100, 150, data_len) + np.random.randn(data_len) * 3
#     high_prices = close_prices + np.abs(np.random.randn(data_len)) * 1.5
#     low_prices = close_prices - np.abs(np.random.randn(data_len)) * 1.5
#     open_prices = close_prices + np.random.randn(data_len) * 0.8
    
#     # NaN値をいくつか挿入
#     close_prices[20:25] = np.nan
#     close_prices[50] = np.nan
#     high_prices[30] = np.nan
    
#     sample_data_df = pd.DataFrame({
#         'open': open_prices,
#         'high': high_prices,
#         'low': low_prices,
#         'close': close_prices
#     })

#     # --- HyperMA インスタンス化と計算 ---
#     hyper_ma_indicator = HyperMA(period=14, src_type='close')
#     hyper_ma_kalman = HyperMA(period=14, src_type='close', use_kalman_filter=True)

#     print("\n--- Calculating HyperMA (p=14, src=close) on DataFrame ---")
#     hyper_ma_values = hyper_ma_indicator.calculate(sample_data_df)
#     print(f"Result length: {len(hyper_ma_values)}")
#     print("Last 15 values:")
#     print(hyper_ma_values[-15:])

#     # キャッシュテスト
#     print("\n--- Calculating HyperMA again (should use cache) ---")
#     hyper_ma_cached = hyper_ma_indicator.calculate(sample_data_df)
#     assert np.allclose(hyper_ma_values, hyper_ma_cached, equal_nan=True)
#     print("Cache test passed.")

#     print("\n--- Calculating HyperMA (p=14, src=close, Kalman=Y) on DataFrame ---")
#     hyper_ma_kalman_values = hyper_ma_kalman.calculate(sample_data_df)
#     print(f"Result length: {len(hyper_ma_kalman_values)}")
#     print("Last 15 values:")
#     print(hyper_ma_kalman_values[-15:])

#     # 短いデータでのテスト
#     print("\n--- Calculating HyperMA on short data ---")
#     short_data = sample_data_df.iloc[:20] # period=14より長い
#     hyper_ma_short = hyper_ma_indicator.calculate(short_data)
#     print(f"Result length: {len(hyper_ma_short)}")
#     print("Values:")
#     print(hyper_ma_short)

#     # NaNのみのデータ
#     print("\n--- Calculating HyperMA on all NaN data ---")
#     nan_data = sample_data_df.copy()
#     nan_data['close'] = np.nan
#     hyper_ma_nan = hyper_ma_indicator.calculate(nan_data)
#     print(f"Result length: {len(hyper_ma_nan)}")
#     print("Values (should be all NaN):")
#     print(hyper_ma_nan[-15:])
#     assert np.all(np.isnan(hyper_ma_nan))

#     # エラーケース (不正な期間)
#     try:
#         print("\n--- Testing invalid period ---")
#         HyperMA(period=0)
#     except ValueError as e:
#         print(f"Caught expected error: {e}") 