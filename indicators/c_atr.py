#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit
import traceback # エラーハンドリング用にインポート

# --- 依存関係のインポート ---

from .indicator import Indicator
from .price_source import PriceSource
from .kalman_filter import KalmanFilter
from .alma import calculate_alma_numba as calculate_alma
from .hyper_smoother import hyper_smoother, calculate_hyper_smoother_numba
from .cycle.ehlers_unified_dc import EhlersUnifiedDC


@dataclass
class CATRResult:
    """CATRの計算結果"""
    values: np.ndarray        # CATRの値（%ベース）
    absolute_values: np.ndarray  # CATRの値（金額ベース）
    tr: np.ndarray           # True Range
    atr_period: np.ndarray  # ドミナントサイクルから決定されたATR期間
    dc_values: np.ndarray    # ドミナントサイクル値


@vectorize(['float64(float64, float64)'], nopython=True, fastmath=True, cache=True)
def max_vec(a: float, b: float) -> float:
    """aとbの最大値を返す（ベクトル化版）"""
    return max(a, b)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def max3_vec(a: float, b: float, c: float) -> float:
    """a, b, cの最大値を返す（ベクトル化版）"""
    return max(a, max(b, c))


@njit(fastmath=True, parallel=True, cache=True)
def calculate_true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    True Rangeを計算する（並列高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        True Range の配列
    """
    length = len(high)
    tr = np.zeros(length, dtype=np.float64)
    
    # 最初の要素は単純なレンジ
    if length > 0:
        tr[0] = high[0] - low[0]
    
    # 一時配列を使用して計算効率化
    tr1 = np.zeros(length, dtype=np.float64)
    tr2 = np.zeros(length, dtype=np.float64)
    tr3 = np.zeros(length, dtype=np.float64)
    
    # 各要素のTRの計算を分解して並列化
    for i in prange(1, length):
        # 当日の高値 - 当日の安値
        tr1[i] = high[i] - low[i]
        # |当日の高値 - 前日の終値|
        tr2[i] = abs(high[i] - close[i-1])
        # |当日の安値 - 前日の終値|
        tr3[i] = abs(low[i] - close[i-1])
    
    # 最大値を計算（並列処理）
    for i in prange(1, length):
        tr[i] = max(tr1[i], max(tr2[i], tr3[i]))
    
    return tr


@njit(fastmath=True, parallel=True, cache=True)
def calculate_c_atr(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray,
    atr_period: np.ndarray,
    max_period: int,
    smoother_type: str = 'alma'  # 'alma'または'hyper'
) -> np.ndarray:
    """
    CATRを計算する（並列高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        atr_period: サイクル検出器から直接決定されたATR期間の配列
        max_period: 最大期間（計算開始位置用）
        smoother_type: 平滑化アルゴリズムのタイプ（'alma'または'hyper'）
    
    Returns:
        CATRの値を返す
    """
    length = len(high)
    c_atr = np.zeros(length, dtype=np.float64)
    
    # True Rangeの計算
    tr = calculate_true_range(high, low, close)
    
    # 各時点での平滑化を計算
    for i in prange(max_period, length):
        # その時点でのドミナントサイクルから決定された期間を取得
        curr_period = int(atr_period[i])
        if curr_period < 1:
            curr_period = 1
            
        # 現在位置までのTRデータを取得（効率化のためウィンドウサイズを制限）
        start_idx = max(0, i-curr_period*2)
        window = tr[start_idx:i+1]
        
        # 選択された平滑化アルゴリズムを適用
        if smoother_type == 'alma':
            # ALMAを使用して平滑化（固定パラメータ：offset=0.85, sigma=6）
            smoothed_values = calculate_alma(window, curr_period, 0.85, 6.0)
        else:  # 'hyper'
            # ハイパースムーサーを使用して平滑化
            smoothed_values = calculate_hyper_smoother_numba(window, curr_period)
        
        # 最後の値をATRとして使用
        if len(smoothed_values) > 0 and not np.isnan(smoothed_values[-1]):
            c_atr[i] = smoothed_values[-1]
    
    return c_atr


@njit(fastmath=True, parallel=True, cache=True)
def calculate_percent_atr(absolute_atr: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    金額ベースのATRから%ベースのATRを計算する（並列高速化版）
    
    Args:
        absolute_atr: 金額ベースのATR配列
        close: 終値の配列
    
    Returns:
        %ベースのATR配列
    """
    length = len(absolute_atr)
    percent_atr = np.zeros_like(absolute_atr, dtype=np.float64)
    
    # 並列処理で高速化
    for i in prange(length):
        if not np.isnan(absolute_atr[i]) and close[i] > 0:
            percent_atr[i] = absolute_atr[i] / close[i]
    
    return percent_atr


class CATR(Indicator):
    """
    CATR（Cycle Average True Range）インジケーター
    
    特徴:
    - ドミナントサイクル検出器から直接ATR期間を決定
    - ALMAまたはハイパースムーサーによる平滑化
    - 金額ベースと%ベースの両方の値を提供
    
    使用方法:
    - ボラティリティに基づいた利益確定・損切りレベルの設定
    - ATRチャネルやボラティリティストップの構築
    - 異なる価格帯の銘柄間でのボラティリティ比較（%ベース）
    """
    
    def __init__(
        self,
        detector_type: str = 'phac_e',             # 検出器タイプ
        cycle_part: float = 0.5,                 # サイクル部分の倍率
        lp_period: int = 13,
        hp_period: int = 50,
        max_cycle: int = 55,                     # 最大サイクル期間
        min_cycle: int = 5,                      # 最小サイクル期間
        max_output: int = 34,                    # 最大出力値
        min_output: int = 5,                     # 最小出力値
        smoother_type: str = 'alma',              # 'alma'または'hyper'
        # --- 追加: XMA風パラメータ ---
        src_type: str = 'hlc3',                 # 価格ソース ('close', 'hlc3', etc.)
        use_kalman_filter: bool = False,         # カルマンフィルターを使用するか
        kalman_measurement_noise: float = 1.0,   # カルマン: 測定ノイズ
        kalman_process_noise: float = 0.01,      # カルマン: プロセスノイズ
        kalman_n_states: int = 5                 # カルマン: 状態数
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: 検出器タイプ
                - 'hody': ホモダイン判別機（デフォルト）
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 55）
            min_cycle: 最小サイクル期間（デフォルト: 5）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 5）
            smoother_type: 平滑化アルゴリズムのタイプ（デフォルト: 'alma'）
                'alma' - ALMA（Arnaud Legoux Moving Average）
                'hyper' - ハイパースムーサー
            src_type: 計算に使用する価格ソース（デフォルト: 'close'）
            use_kalman_filter: ソース価格にカルマンフィルターを適用するか（デフォルト: False）
            kalman_measurement_noise: カルマンフィルターの測定ノイズ（デフォルト: 1.0）
            kalman_process_noise: カルマンフィルターのプロセスノイズ（デフォルト: 0.01）
            kalman_n_states: カルマンフィルターの状態数（デフォルト: 5）
        """
        kalman_str = f"_kalman={'Y' if use_kalman_filter else 'N'}" if use_kalman_filter else ""
        super().__init__(
            f"CATR({detector_type},{max_output},{min_output},{smoother_type},src={src_type}{kalman_str})"
        )
        
        # パラメータの保存
        self.detector_type = detector_type
        self.cycle_part = cycle_part
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.smoother_type = smoother_type
        # --- 追加パラメータ保存 ---
        self.src_type = src_type.lower()
        self.use_kalman_filter = use_kalman_filter
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_process_noise = kalman_process_noise
        self.kalman_n_states = kalman_n_states
        
        # ドミナントサイクル検出器を初期化
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
        
        # --- 追加: 依存ツールの初期化 ---
        self.price_source_extractor = PriceSource()
        self.kalman_filter = None
        if self.use_kalman_filter:
            self.kalman_filter = KalmanFilter(
                measurement_noise=self.kalman_measurement_noise,
                process_noise=self.kalman_process_noise,
                n_states=self.kalman_n_states
            )
        
        # 結果キャッシュ
        self._result = None
        self._data_hash = None  # データキャッシュ用ハッシュ
        
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する（超高速版）"""
        # 超高速化のため最小限のサンプリング
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    high_val = float(data.iloc[0].get('high', data.iloc[0, 1]))
                    low_val = float(data.iloc[0].get('low', data.iloc[0, 2]))
                    close_val = float(data.iloc[0].get('close', data.iloc[0, 3]))
                    last_close = float(data.iloc[-1].get('close', data.iloc[-1, 3]))
                    data_signature = (length, high_val, low_val, close_val, last_close)
                else:
                    data_signature = (0, 0.0, 0.0, 0.0, 0.0)
            else:
                # NumPy配列の場合
                length = len(data)
                if length > 0 and data.ndim > 1 and data.shape[1] >= 4:
                    data_signature = (length, float(data[0, 1]), float(data[0, 2]), 
                                    float(data[0, 3]), float(data[-1, 3]))
                else:
                    data_signature = (0, 0.0, 0.0, 0.0, 0.0)
            
            # パラメータの最小セット
            params_sig = f"{self.detector_type}_{self.max_output}_{self.min_output}_{self.smoother_type}_{self.src_type}"
            
            # 超高速ハッシュ
            return f"{hash(data_signature)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.detector_type}_{self.smoother_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        CATRを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            CATRの値（%ベース）
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

            # --- データ検証とソース価格取得 (PriceSource.calculate を使用) ---
            # 必要なOHLCデータを取得
            # Note: TR計算には high, low, close が必要
            high_prices = PriceSource.calculate_source(data, 'high')
            low_prices = PriceSource.calculate_source(data, 'low')
            close_prices = PriceSource.calculate_source(data, 'close')

            # データ長の検証
            data_length = len(close_prices)
            if data_length == 0:
                self.logger.warning("価格データが空です。空の配列を返します。")
                self._reset_results() # 結果をリセット
                self._data_hash = data_hash
                return np.array([])

            self._data_hash = data_hash  # 新しいハッシュを保存

            # TR計算に必要な high, low, close は既に取得済み
            high = np.asarray(high_prices, dtype=np.float64)
            low = np.asarray(low_prices, dtype=np.float64)
            close = np.asarray(close_prices, dtype=np.float64)

            # %ATR計算用のソース価格 (Kalman適用前)
            src_prices = PriceSource.calculate_source(data, self.src_type) # 再度取得
            if src_prices is None or len(src_prices) == 0:
                 self.logger.warning(f"価格ソース '{self.src_type}' 取得失敗/空。")
                 return np.full(current_data_len, np.nan)
            if not isinstance(src_prices, np.ndarray): src_prices = np.array(src_prices)
            if src_prices.dtype != np.float64:
                 try: src_prices = src_prices.astype(np.float64)
                 except ValueError:
                      self.logger.error(f"価格ソース '{self.src_type}' をfloat64に変換できませんでした。")
                      return np.full(current_data_len, np.nan)

            # --- カルマンフィルター適用 (Optional) ---
            effective_src_prices = src_prices
            if self.use_kalman_filter and self.kalman_filter:
                self.logger.debug("カルマンフィルター適用中...")
                try:
                    # KalmanFilterのcalculateにソース価格を渡す
                    filtered_prices = self.kalman_filter.calculate(effective_src_prices)
                    if filtered_prices is not None and len(filtered_prices) == current_data_len:
                        if not isinstance(filtered_prices, np.ndarray): filtered_prices = np.array(filtered_prices)
                        try:
                             if filtered_prices.dtype != np.float64: filtered_prices = filtered_prices.astype(np.float64)
                             effective_src_prices = filtered_prices
                             self.logger.debug("カルマンフィルター適用完了。")
                        except ValueError:
                             self.logger.error("カルマンフィルター結果をfloat64に変換できませんでした。元のソース価格を使用。")
                    else:
                         self.logger.warning("カルマンフィルター計算失敗/結果長不一致。元のソース価格を使用。")
                except Exception as kf_e:
                     self.logger.error(f"カルマンフィルター計算エラー: {kf_e}", exc_info=False)
                     self.logger.warning("エラーのため、元のソース価格を使用。")

            # ドミナントサイクルの計算 - 直接ATR期間として使用
            # dc_detectorには元のDataFrameを渡す（内部で必要な価格を使う想定）
            dc_values = self.dc_detector.calculate(data)
            atr_period = np.asarray(dc_values, dtype=np.float64)

            # 最大ATR期間の最大値を取得（計算開始位置用）
            # NaNを除外して最大値を取得し、整数に変換。NaNのみの場合は0になる
            max_period_value_float = np.nanmax(atr_period)
            if np.isnan(max_period_value_float):
                 max_period_value = 10 # デフォルト最小期間 (例)
                 self.logger.warning("ATR期間が全てNaNです。デフォルトの最大期間を使用します。")
            else:
                 max_period_value = int(max_period_value_float)
                 if max_period_value < 1:
                     max_period_value = 10 # 最小期間制限

            # データ長の検証
            data_length = len(high)
            if data_length < max_period_value:
                 # データ長が足りない場合は警告し、空の結果を返す
                 self.logger.warning(f"データ長({data_length})が必要な最大期間({max_period_value})より短いため、計算できません。")
                 self._reset_results()
                 return np.full(current_data_len, np.nan)

            # CATRの計算（並列版 - 高速化） - 元のhigh, low, closeを使用
            c_atr_values = calculate_c_atr(
                high,
                low,
                close,
                atr_period,
                max_period_value,
                self.smoother_type
            )
            
            # 金額ベースのATR値を保存
            absolute_atr_values = c_atr_values

            # %ベースのATR値に変換（指定されたソース価格に対する比率）（並列版 - 高速化）
            percent_atr_values = calculate_percent_atr(absolute_atr_values, effective_src_prices)

            # True Rangeを計算
            tr_values = calculate_true_range(high, low, close)

            # 結果の保存（参照問題を避けるためコピーを作成）
            self._result = CATRResult(
                values=np.copy(percent_atr_values),           # %ベースのATR
                absolute_values=np.copy(absolute_atr_values), # 金額ベースのATR
                tr=np.copy(tr_values),                        # True Range
                atr_period=np.copy(atr_period),
                dc_values=np.copy(dc_values)                  # ドミナントサイクル値を保存
            )
            
            self._values = percent_atr_values  # 標準インジケーターインターフェース用
            return percent_atr_values
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"CATR計算中にエラー: {error_msg}\n{stack_trace}")
            # エラー時は前回の結果を維持せず、リセットして空配列を返す
            self._reset_results()
            return np.array([])

    def _reset_results(self):
        """内部結果とキャッシュをリセットする"""
        self._result = None
        self._data_hash = None
        # _values もリセットする (Indicatorクラスのキャッシュかもしれないため)
        self._values = None

    def get_dc_values(self) -> np.ndarray:
        """
        ドミナントサイクルの値を取得する
        
        Returns:
            np.ndarray: ドミナントサイクルの値
        """
        if self._result is None:
            return np.array([])
        return self._result.dc_values
    
    def get_true_range(self) -> np.ndarray:
        """
        True Range (TR)の値を取得する
        
        Returns:
            np.ndarray: TRの値
        """
        if self._result is None:
            return np.array([])
        return self._result.tr
    
    def get_atr_period(self) -> np.ndarray:
        """
        ATR期間の値を取得する
        
        Returns:
            np.ndarray: ATR期間の値
        """
        if self._result is None:
            return np.array([])
        return self._result.atr_period
    
    def get_percent_atr(self) -> np.ndarray:
        """
        %ベースのATRを取得する
        
        Returns:
            np.ndarray: %ベースのATR値（100倍されたパーセンテージ値）
        """
        if self._result is None:
            return np.array([])
        return self._result.values * 100  # 100倍して返す
    
    def get_absolute_atr(self) -> np.ndarray:
        """
        金額ベースのATRを取得する
        
        Returns:
            np.ndarray: 金額ベースのATR値
        """
        if self._result is None:
            return np.array([])
        return self._result.absolute_values
    
    def get_atr_multiple(self, multiplier: float = 1.0) -> np.ndarray:
        """
        %ベースのATRの倍数を取得する
        
        Args:
            multiplier: ATRの倍数（デフォルト: 1.0）
            
        Returns:
            np.ndarray: %ベースのATR × 倍数
        """
        atr = self.get_percent_atr()
        return atr * multiplier
    
    def get_absolute_atr_multiple(self, multiplier: float = 1.0) -> np.ndarray:
        """
        金額ベースのATRの倍数を取得する
        
        Args:
            multiplier: ATRの倍数（デフォルト: 1.0）
            
        Returns:
            np.ndarray: 金額ベースのATR × 倍数
        """
        abs_atr = self.get_absolute_atr()
        return abs_atr * multiplier
        
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._reset_results() # 内部結果とキャッシュをリセット
        if self.dc_detector and hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset()
        # --- 追加: カルマンフィルターのリセット ---
        if self.kalman_filter and hasattr(self.kalman_filter, 'reset'):
            self.kalman_filter.reset()
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。") 