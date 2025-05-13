#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from .indicator import Indicator
from .atr import ATR
from .price_source import PriceSource
from .kalman_filter import apply_kalman_filter_numba # Import the Numba function

# Numbaで使用するためのヘルパー関数 (Supertrend計算でも使用)
@njit(fastmath=True)
def nz(value: float, replacement: float = 0.0) -> float:
    """
    NaNの場合はreplacementを返す（Numba互換）
    """
    if np.isnan(value):
        return replacement
    return value

@dataclass
class KalmanHullSupertrendResult:
    """KalmanHullSupertrendの計算結果"""
    supertrend: np.ndarray  # Supertrendラインの値
    direction: np.ndarray   # トレンド方向（1=上昇トレンド、-1=下降トレンド）
    kalman_hma: np.ndarray  # Kalman Hull Moving Averageの値
    upper_band: np.ndarray  # 計算された上限バンド
    lower_band: np.ndarray  # 計算された下限バンド
    atr: np.ndarray         # ATR値

# --- Kalman Hull Moving Average ---
@njit(fastmath=True)
def calculate_khma(
    prices: np.ndarray,
    length: float, # measurementNoiseに対応
    process_noise: float,
    n_states: int = 5
) -> np.ndarray:
    """
    Kalman Hull Moving Average (KHMA) を計算する (Numba JIT)
    PineScriptのKHMA関数に相当。
    """
    # lengthが0以下の場合の処理を追加
    if length <= 0:
        # measurement_noiseが0以下になるため、apply_kalman_filter内で処理されるが、
        # sqrt(length)も問題になるため、ここで対処
        # 元の価格を返すか、NaN配列を返すか
        return np.full_like(prices, np.nan)

    length_half = length / 2.0
    # length_halfも0以下チェック
    if length_half <= 0:
         length_half = 1e-9 # apply_kalman_filter_numba内で処理されるように

    # sqrt_lengthも負にならないように
    sqrt_length = np.sqrt(max(0.0, length)) # Ensure argument is float
    sqrt_length_rounded = np.round(sqrt_length)
    # sqrt_length_roundedも0以下チェック
    if sqrt_length_rounded <= 0:
         sqrt_length_rounded = 1e-9 # apply_kalman_filter_numba内で処理されるように


    # Kalman Filterを3回適用 (インポートしたNumba関数を使用)
    kalman_len_half = apply_kalman_filter_numba(prices, length_half, process_noise, n_states)
    kalman_len = apply_kalman_filter_numba(prices, length, process_noise, n_states)

    # 2 * KF(len/2) - KF(len)
    # NaNが含まれる場合の計算に注意
    intermediate_series = 2.0 * kalman_len_half - kalman_len

    # 最終的なKalman Filter適用 (インポートしたNumba関数を使用)
    khma = apply_kalman_filter_numba(intermediate_series, sqrt_length_rounded, process_noise, n_states)

    return khma

# --- Supertrend ---
@njit(fastmath=True, parallel=True)
def calculate_supertrend_bands_trend(
    close: np.ndarray,
    src: np.ndarray, # 通常はKHMA
    atr: np.ndarray,
    factor: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Supertrendのバンドとトレンド方向を計算する (Numba JIT, Parallel)
    PineScriptのsupertrend関数と条件ロジックに相当。
    """
    length = len(close)
    upper_band = np.full(length, np.nan, dtype=np.float64)
    lower_band = np.full(length, np.nan, dtype=np.float64)
    supertrend_line = np.full(length, np.nan, dtype=np.float64)
    direction = np.zeros(length, dtype=np.int8) # 1 for long, -1 for short

    # 最初の有効なインデックスを見つける (srcとatrの両方が必要)
    start_idx = -1
    for i in range(length):
        if not np.isnan(close[i]) and not np.isnan(src[i]) and not np.isnan(atr[i]) and atr[i] > 1e-9: # ATRがほぼ0でないことも確認
            start_idx = i
            break

    if start_idx == -1: # 有効なデータがない場合
        return upper_band, lower_band, supertrend_line, direction

    # 初期値の設定
    # PineScript: if na(atr[1]) direction := 1
    # Pythonでは最初の有効なatrを持つインデックスで初期化
    # 最初のトレンド方向は、最初の終値と初期バンドで決定するのが自然
    initial_up = src[start_idx] + factor * atr[start_idx]
    initial_lo = src[start_idx] - factor * atr[start_idx]
    upper_band[start_idx] = initial_up
    lower_band[start_idx] = initial_lo

    if close[start_idx] > initial_up:
        direction[start_idx] = 1
        supertrend_line[start_idx] = initial_lo
    elif close[start_idx] < initial_lo:
        direction[start_idx] = -1
        supertrend_line[start_idx] = initial_up
    else: # バンド内にある場合
        # PineScriptの `na(atr[1])` のケースに近い挙動をさせるなら、デフォルト1
        direction[start_idx] = 1
        supertrend_line[start_idx] = initial_lo

    # メインループ (並列化のためprangeを使用)
    # ループ内で前の値に依存するため、並列化は困難。通常のrangeに戻す。
    # @njit(fastmath=True) # parallel=True を削除
    # def calculate_supertrend_bands_trend_sequential(...):
    #     ... (ループを for i in range(start_idx + 1, length): に変更) ...
    # この関数自体を sequential にするか、呼び出し側でループするか。
    # ここでは関数内で sequential ループを行う。
    for i in range(start_idx + 1, length):
        # 必要な値がNaNでないかチェック
        if np.isnan(close[i]) or np.isnan(src[i]) or np.isnan(atr[i]) or atr[i] <= 1e-9:
            # データ欠損時は前の値を引き継ぐ
            upper_band[i] = upper_band[i-1]
            lower_band[i] = lower_band[i-1]
            supertrend_line[i] = supertrend_line[i-1]
            direction[i] = direction[i-1]
            continue

        # 現在のバンド計算
        up = src[i] + factor * atr[i]
        lo = src[i] - factor * atr[i]

        # 前のバンドと終値を取得 (NaNの場合は前の有効な値を引き継ぐ想定だが、ここではnzで0にする)
        # より正確には、前の有効なインデックスの値を保持する必要がある。
        # Numbaでは複雑になるため、nzで近似する。
        prev_lower_band = nz(lower_band[i-1])
        prev_upper_band = nz(upper_band[i-1])
        prev_close = nz(close[i-1])
        # prev_supertrend = nz(supertrend_line[i-1]) # PineScriptのロジックでは使われていない

        # バンドの更新ロジック (PineScriptの := 演算子に相当)
        # lowerBand := lowerBand > prevLowerBand or close[1] < prevLowerBand ? lowerBand : prevLowerBand
        current_lower_band = lo
        if not (lo > prev_lower_band or prev_close < prev_lower_band):
             current_lower_band = prev_lower_band
        lower_band[i] = current_lower_band


        # upperBand := upperBand < prevUpperBand or close[1] > prevUpperBand ? upperBand : prevUpperBand
        current_upper_band = up
        if not (up < prev_upper_band or prev_close > prev_upper_band):
             current_upper_band = prev_upper_band
        upper_band[i] = current_upper_band


        # トレンド方向の決定ロジック (PineScriptのロジックを再現)
        # prevSuperTrend = superTrend[1] # 前のSupertrend値が必要
        prev_supertrend_val = nz(supertrend_line[i-1])
        prev_upper_band_val = nz(upper_band[i-1]) # nzは不要かも、上で計算済み
        prev_lower_band_val = nz(lower_band[i-1]) # nzは不要かも、上で計算済み

        current_direction = direction[i-1] # デフォルトは維持

        # if na(atr[1]) -> これは最初のステップで処理済み
        # else if prevSuperTrend == prevUpperBand
        #     direction := close > upperBand ? -1 : 1
        # else (prevSuperTrend == prevLowerBand)
        #     direction := close < lowerBand ? 1 : -1

        # 前のSupertrendがどちらのバンドだったかで条件分岐
        # 注意: 浮動小数点比較のため、完全一致は危険。許容誤差を設けるか、前の方向で判断する方が安全。
        # ここでは前の方向 `direction[i-1]` を使う方がロバスト。
        if direction[i-1] == 1: # 前が上昇トレンド (SupertrendはLower Bandだったはず)
            if close[i] < lower_band[i]: # 下抜けたらトレンド転換
                current_direction = -1
            # else 維持
        elif direction[i-1] == -1: # 前が下降トレンド (SupertrendはUpper Bandだったはず)
            if close[i] > upper_band[i]: # 上抜けたらトレンド転換
                current_direction = 1
            # else 維持

        direction[i] = current_direction

        # Supertrendラインの決定
        if current_direction == 1:
            supertrend_line[i] = lower_band[i]
        else: # current_direction == -1
            supertrend_line[i] = upper_band[i]

    return upper_band, lower_band, supertrend_line, direction


class KalmanHullSupertrend(Indicator):
    """
    Kalman Hull Supertrend インジケーター

    PineScriptの "Kalman Hull Supertrend [BackQuant]" をPythonで実装。
    Kalman Filterで平滑化された価格にHull Moving Averageを適用し、
    その結果をSupertrendのソースとして使用する。\

    
    """

    def __init__(
        self,
        price_source: str = 'close',
        measurement_noise: float = 3.0,
        process_noise: float = 0.01,
        atr_period: int = 12,
        factor: float = 1.7,
        kalman_n_states: int = 5, # PineScriptのN=5に対応
        warmup_periods: Optional[int] = None # ATR等のウォームアップ期間
    ):
        """
        コンストラクタ

        Args:
            price_source (str): Kalman FilterとKHMAに使用する価格 ('close', 'hlc3', etc.)
            measurement_noise (float): Kalman Filterの測定ノイズ (PineScriptのmeasurementNoise)
                                      0より大きい値である必要があります。
            process_noise (float): Kalman Filterのプロセスノイズ (PineScriptのprocessNoise)
            atr_period (int): ATRの計算期間
            factor (float): SupertrendのATR乗数
            kalman_n_states (int): Kalman Filterの状態数 (PineScriptのN)
            warmup_periods (Optional[int]): 計算に必要な最小期間。Noneの場合、atr_periodから推定。
        """
        if measurement_noise <= 0:
             raise ValueError("measurement_noiseは0より大きい必要があります。")
        if atr_period <= 0:
             raise ValueError("atr_periodは0より大きい必要があります。")
        if factor <= 0:
             raise ValueError("factorは0より大きい必要があります。")
        if kalman_n_states <= 0:
             raise ValueError("kalman_n_statesは0より大きい必要があります。")


        self._price_source_key = price_source
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.atr_period = atr_period
        self.factor = factor
        self.kalman_n_states = kalman_n_states

        # ウォームアップ期間を設定 (ATR期間 + KHMAの内部計算に必要な期間を考慮)
        # KHMAは内部で複数回フィルタをかけるため、余裕を持たせる
        # apply_kalman_filterは状態を持つため、実質的なウォームアップは入力系列長に依存するが、
        # ATRの期間が主要な制約となることが多い。
        estimated_warmup = atr_period + kalman_n_states + 5 # 余裕分 (KHMAの依存性は複雑なので簡略化)
        self._warmup_periods = warmup_periods if warmup_periods is not None else estimated_warmup

        # インジケータ名を生成
        name = (f"KHSupertrend(src={price_source}, measN={measurement_noise}, procN={process_noise}, "
                f"atrP={atr_period}, factor={factor})")
        super().__init__(name, self._warmup_periods)

        # サブインジケータ
        self.atr_calculator = ATR(period=atr_period, use_ema=False) # PineScriptのta.atrはSMAベース
        self.price_source_extractor = PriceSource()

        # 結果キャッシュ
        self._result: Optional[KalmanHullSupertrendResult] = None
        self._data_hash: Optional[int] = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> int:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムと設定値をハッシュ
            cols_to_hash = ['high', 'low', 'close']
            # price_sourceがohlc4などの場合も考慮
            if self._price_source_key == 'hl2':
                 cols_to_hash = ['high', 'low']
            elif self._price_source_key == 'hlc3':
                 cols_to_hash = ['high', 'low', 'close']
            elif self._price_source_key == 'ohlc4':
                 cols_to_hash = ['open', 'high', 'low', 'close']
            elif self._price_source_key == 'open':
                 cols_to_hash = ['open']
            elif self._price_source_key == 'high':
                 cols_to_hash = ['high']
            elif self._price_source_key == 'low':
                 cols_to_hash = ['low']
            # 'close' は常に必要 (Supertrend計算のため)
            if 'close' not in cols_to_hash:
                 cols_to_hash.append('close')
            if 'high' not in cols_to_hash: # ATR計算のため
                 cols_to_hash.append('high')
            if 'low' not in cols_to_hash: # ATR計算のため
                 cols_to_hash.append('low')

            # 重複削除
            cols_to_hash = sorted(list(set(cols_to_hash)))


            relevant_cols = [col for col in cols_to_hash if col in data.columns]
            if len(relevant_cols) != len(cols_to_hash):
                 missing = set(cols_to_hash) - set(relevant_cols)
                 raise ValueError(f"DataFrameに必要なカラムが見つかりません: {missing}")

            data_tuple = tuple(map(tuple, (data[col].values for col in relevant_cols)))
            param_tuple = (self._price_source_key, self.measurement_noise, self.process_noise,
                           self.atr_period, self.factor, self.kalman_n_states)
            return hash((data_tuple, param_tuple))
        else:
            # NumPy配列の場合は形状と内容、パラメータをハッシュ
            # OHLCデータなどを想定
            if data.ndim != 2 or data.shape[1] < 4:
                 raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")

            param_tuple = (self._price_source_key, self.measurement_noise, self.process_noise,
                           self.atr_period, self.factor, self.kalman_n_states)
            # NumPy配列の内容をタプルに変換してハッシュ可能にする
            try:
                # 全データをハッシュすると重いので、一部を代表値とする
                # shape, first row, last row, mean of close
                shape_tuple = data.shape
                first_row = tuple(data[0]) if len(data) > 0 else ()
                last_row = tuple(data[-1]) if len(data) > 0 else ()
                mean_close = np.mean(data[:, 3]) if len(data) > 0 else 0.0 # Assuming close is 4th col
                data_repr_tuple = (shape_tuple, first_row, last_row, mean_close)
            except Exception: # フォールバック
                 data_repr_tuple = data.shape

            return hash((data_repr_tuple, param_tuple))


    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Kalman Hull Supertrendを計算する

        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                  DataFrameの場合、'high', 'low', 'close'カラムが必要。
                  'open'も'ohlc4'ソースの場合に必要。
                  NumPy配列の場合、OHLCの順で列が並んでいることを想定。

        Returns:
            np.ndarray: トレンド方向の配列（1=上昇、-1=下降、0=計算不可）
        """
        try:
            # データハッシュを計算し、キャッシュを確認
            current_hash = self._get_data_hash(data)
            if self._result is not None and current_hash == self._data_hash:
                return self._result.direction # キャッシュされたトレンド方向を返す

            # データを準備
            prices_df = self.price_source_extractor.ensure_dataframe(data)
            src_prices = self.price_source_extractor.get_price(prices_df, self._price_source_key)
            close_prices = self.price_source_extractor.get_price(prices_df, 'close')
            # high_prices = self.price_source_extractor.get_price(prices_df, 'high') # ATR計算機が内部で使う
            # low_prices = self.price_source_extractor.get_price(prices_df, 'low')   # ATR計算機が内部で使う

            # データ長の検証
            if len(src_prices) < self._warmup_periods:
                self.logger.warning(f"データ長 ({len(src_prices)}) がウォームアップ期間 ({self._warmup_periods}) より短いため、計算をスキップします。")
                nan_array = np.full(len(src_prices), np.nan)
                int_zero_array = np.zeros(len(src_prices), dtype=np.int8) # 方向は0で初期化
                self._result = KalmanHullSupertrendResult(
                    supertrend=nan_array, direction=int_zero_array, kalman_hma=nan_array,
                    upper_band=nan_array, lower_band=nan_array, atr=nan_array
                )
                self._values = self._result.direction
                self._data_hash = current_hash
                return self._values


            # 1. ATR計算
            # ATR計算機はDataFrameを受け取る想定
            atr_values = self.atr_calculator.calculate(prices_df)

            # 2. Kalman Hull Moving Average (KHMA) 計算
            # KHMA計算関数はNumPy配列を受け取る
            khma_values = calculate_khma(
                src_prices, self.measurement_noise, self.process_noise, self.kalman_n_states
            )

            # 3. Supertrend計算
            # Supertrend計算関数もNumPy配列を受け取る
            upper_band, lower_band, supertrend_line, direction = calculate_supertrend_bands_trend(
                close_prices, khma_values, atr_values, self.factor
            )

            # 結果を保存
            self._result = KalmanHullSupertrendResult(
                supertrend=supertrend_line,
                direction=direction,
                kalman_hma=khma_values,
                upper_band=upper_band,
                lower_band=lower_band,
                atr=atr_values
            )
            self._values = self._result.direction # Indicatorクラスの規約
            self._data_hash = current_hash # ハッシュを更新

            return self._values

        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"{self.name} 計算中にエラー: {error_msg}\n{stack_trace}")

            # エラー時はデフォルト値を返す
            data_len = len(data) if hasattr(data, '__len__') else 0
            nan_array = np.full(data_len, np.nan)
            int_zero_array = np.zeros(data_len, dtype=np.int8)
            # 結果オブジェクトが存在しない場合（初回エラーなど）は作成
            if self._result is None:
                 self._result = KalmanHullSupertrendResult(
                     supertrend=nan_array, direction=int_zero_array, kalman_hma=nan_array,
                     upper_band=nan_array, lower_band=nan_array, atr=nan_array
                 )
                 self._values = int_zero_array

            # 既存の結果があれば、方向を0で埋める
            self._result.direction[:] = 0
            self._values = self._result.direction
            # エラー発生時はキャッシュを無効化（次回再計算させる）
            self._data_hash = None
            return self._values


    def get_supertrend(self) -> Optional[np.ndarray]:
        """Supertrendラインの値を取得する"""
        return self._result.supertrend if self._result else None

    def get_direction(self) -> Optional[np.ndarray]:
        """トレンド方向（1 or -1）を取得する"""
        return self._result.direction if self._result else None

    def get_kalman_hma(self) -> Optional[np.ndarray]:
        """Kalman Hull Moving Averageの値を取得する"""
        return self._result.kalman_hma if self._result else None

    def get_bands(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Supertrendの計算に使用された上限バンドと下限バンドを取得する"""
        if self._result:
            return self._result.upper_band, self._result.lower_band
        return None, None # タプルで返す

    def get_atr(self) -> Optional[np.ndarray]:
        """計算に使用されたATRの値を取得する"""
        return self._result.atr if self._result else None

    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._result = None
        self._data_hash = None
        self.atr_calculator.reset()
        # price_source_extractorは状態を持たない想定
