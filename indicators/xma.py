#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange
import math
import traceback

# --- 依存関係のインポート ---
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .kalman_filter import KalmanFilter
    from .cycle_efficiency_ratio import CycleEfficiencyRatio
    from .x_trend_index import XTrendIndex
    # MA計算用のNumba関数をインポート
    from .alma import calculate_alma_numba # 元の関数
    from .hyper_smoother import calculate_hyper_smoother_numba # 元の関数
    from .hma import calculate_wma_numba # HMAで使用
except ImportError:
    # フォールバック (テストや静的解析用)
    print("Warning: Could not import from relative path. Assuming base classes/functions are available.")
    # ... (ALMA, HMA, HyperMA にあるようなダミークラス/関数定義) ...
    # 簡単のため省略。実行には実際のモジュールが必要。
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class PriceSource: # Dummy
        def ensure_dataframe(self, data): return pd.DataFrame({'close': data}) if isinstance(data, np.ndarray) else data
        def get_price(self, df, src_type): return df['close'].values
    class KalmanFilter: # Dummy
        def __init__(self, **kwargs): pass
        def calculate(self, data): return data['close'].values if isinstance(data, pd.DataFrame) else data
        def reset(self): pass
    class CycleEfficiencyRatio: # Dummy
         def __init__(self, **kwargs): self.logger = self._get_logger()
         def calculate(self, data): return np.random.rand(len(data)) * 2 - 1 # Dummy [-1, 1]
         def reset(self): pass
         def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class XTrendIndex: # Dummy
         def __init__(self, **kwargs): self.logger = self._get_logger()
         def calculate(self, data):
              n = len(data)
              return type('obj', (object,), {
                  'values': np.random.rand(n), 'er': np.random.rand(n),
                  'dominant_cycle': np.full(n, 14.0), 'dynamic_atr_period': np.full(n, 14.0),
                  'choppiness_index': np.random.rand(n)*100, 'range_index': np.random.rand(n)*100,
                  'stddev_factor': np.random.rand(n)+0.5, 'tr': np.random.rand(n),
                  'atr': np.random.rand(n), 'dynamic_threshold': np.random.rand(n)*0.2+0.5,
                  'trend_state': np.random.randint(0, 2, n).astype(float)
              })()
         def reset(self): pass
         def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    # Dummy MA calcs
    @njit
    def calculate_alma_numba(prices, period, offset, sigma): return np.full_like(prices, np.nan)
    @njit
    def calculate_hyper_smoother_numba(data, length): return np.full_like(data, np.nan)
    @njit
    def calculate_wma_numba(prices, period): return np.full_like(prices, np.nan)

# --- Numba ヘルパー関数 ---

@njit(fastmath=True, parallel=True, cache=True)
def calculate_dynamic_period_kama_style(
    trigger_values: np.ndarray,
    fast_period: int,
    slow_period: int,
    min_period_limit: int = 1 # 計算結果の最小期間
) -> np.ndarray:
    """
    KAMAスタイルの動的期間計算（Numba JIT）- 並列処理・ベクトル化最適化版
    トリガー値（CERまたはXTI）に基づいて動的期間を計算
    
    Args:
        trigger_values: トリガー値（0-1の範囲）の配列
        fast_period: 最速期間（通常2）
        slow_period: 最遅期間（通常30）
        min_period_limit: 最小期間の下限
        
    Returns:
        動的期間の配列
    """
    n = len(trigger_values)
    periods = np.empty(n, dtype=np.float64)
    
    # 事前計算で定数を用意
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc_diff = fast_sc - slow_sc
    
    # 並列処理でループを最適化
    for i in prange(n):
        # NaNチェック
        if np.isnan(trigger_values[i]):
            periods[i] = slow_period
            continue
            
        # トリガー値をバインド (0-1の範囲に制限)
        trigger = max(0.0, min(1.0, trigger_values[i]))
        
        # 動的期間計算: KAMA方式
        sc = trigger * sc_diff + slow_sc # スムージング定数
        period = 2.0 / sc - 1.0 # 期間 = 2/alpha - 1
        
        # 最小期間制限
        periods[i] = max(min_period_limit, period)
        
    return periods


# --- 動的MA計算用のNumba関数 --- (変更なし)

@njit(fastmath=True, parallel=True, cache=True)
def calculate_dynamic_alma_numba(prices: np.ndarray, dynamic_periods: np.ndarray, offset: float, sigma: float) -> np.ndarray:
    """動的期間を使用してALMAを計算する（並列処理最適化版）"""
    length = len(prices)
    result = np.full(length, np.nan, dtype=np.float64)
    last_valid_alma = np.nan

    # 早期リターン条件
    if length == 0:
        return np.zeros(0, dtype=np.float64)

    # プリアロケーション：重みを格納する最大サイズの配列（最大期間に基づく）
    max_period = np.max(dynamic_periods) if length > 0 else 0
    if max_period <= 0 or np.isnan(max_period):
        return result  # 計算不可能な場合は早期終了

    for i in prange(length):
        period = int(dynamic_periods[i])  # 整数に変換
        if period <= 0 or np.isnan(period) or i < period - 1:
            result[i] = last_valid_alma  # 前の値を引き継ぐ
            continue

        # ALMA計算のコア処理
        window_size = period
        m = offset * (window_size - 1)
        s = window_size / sigma
        
        # シグマが極端に小さい場合を避ける
        if abs(s) < 1e-9:
            result[i] = last_valid_alma
            continue

        # ウィンドウ内の価格を取得
        window_start_idx = i - window_size + 1
        window_prices = prices[window_start_idx : i + 1]
        
        # ウィンドウ内のNaNをチェック
        has_nan = False
        for j in range(window_size):
            if np.isnan(window_prices[j]):
                has_nan = True
                break
                
        if has_nan:
            result[i] = last_valid_alma  # NaNがあれば前の値を引き継ぐ
            continue

        # 重みの計算と正規化
        weights_sum = 0.0
        alma_value = 0.0
        
        for j in range(window_size):
            exponent = -((j - m) ** 2) / (2 * s * s)
            # 極端に小さい指数の処理
            weight = 0.0 if exponent < -700 else np.exp(exponent)
            weights_sum += weight
            # 重みと価格の積を直接累積（メモリ節約）
            alma_value += window_prices[j] * weight
            
        # 正規化（ゼロ除算防止）
        if weights_sum > 1e-9:
            alma_value /= weights_sum
        else:
            # 極端なケースの処理
            alma_value = np.mean(window_prices)  # 単純平均をフォールバックとして使用
            
        result[i] = alma_value
        last_valid_alma = alma_value  # 有効な値を更新

    return result

@njit(fastmath=True, parallel=True, cache=True)
def calculate_dynamic_hyper_smoother_numba(data: np.ndarray, dynamic_periods: np.ndarray) -> np.ndarray:
    """動的期間を使用してHyper Smootherを計算する（並列処理最適化版）"""
    size = len(data)
    smoothed = np.full(size, np.nan, dtype=np.float64)
    if size == 0:
        # 空の配列を返す際の型推論エラー対策
        return np.zeros(0, dtype=np.float64)

    # 大きな配列での効率化のためにバッチ処理を実装
    batch_size = min(1000, size)  # バッチサイズを制限
    batches = (size + batch_size - 1) // batch_size  # 必要なバッチ数を計算
    
    # 各バッチで独立して処理（並列化に適している）
    for batch in prange(batches):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, size)
        
        # 各バッチの初期状態変数
        f28, f30, vC = np.nan, np.nan, np.nan
        f38, f40, v10 = np.nan, np.nan, np.nan
        f48, f50, v14 = np.nan, np.nan, np.nan
        last_valid_smooth = np.nan

        for i in range(start_idx, end_idx):
            if np.isnan(data[i]):
                smoothed[i] = last_valid_smooth  # NaNの場合、前の値を使用
                continue

            # 最初の有効なデータで状態を初期化
            if np.isnan(f28):
                f28 = f30 = vC = data[i]
                f38 = f40 = v10 = data[i]
                f48 = f50 = v14 = data[i]
                smoothed[i] = data[i]
                last_valid_smooth = data[i]
                continue

            period = max(1, int(dynamic_periods[i]))  # 期間値を安全に変換（最小1）
            if np.isnan(period) or period <= 0:
                smoothed[i] = last_valid_smooth  # 無効な期間は前の値を使用
                continue

            # フィルタ係数の最適化計算
            f18 = 3.0 / (period + 2.0)
            f20 = 1.0 - f18

            # 状態変数の更新（内部状態がNaNでないことを保証）
            f28_prev = f28
            f30_prev = f30
            f38_prev = f38
            f40_prev = f40
            f48_prev = f48
            f50_prev = f50

            # ステージ1
            f28 = f20 * f28_prev + f18 * data[i]
            f30 = f18 * f28 + f20 * f30_prev
            vC = 1.5 * f28 - 0.5 * f30

            # ステージ2
            f38 = f20 * f38_prev + f18 * vC
            f40 = f18 * f38 + f20 * f40_prev
            v10 = 1.5 * f38 - 0.5 * f40

            # ステージ3
            f48 = f20 * f48_prev + f18 * v10
            f50 = f18 * f48 + f20 * f50_prev
            v14 = 1.5 * f48 - 0.5 * f50

            smoothed[i] = v14
            last_valid_smooth = v14

    return smoothed

@njit(fastmath=True, parallel=True, cache=True)
def calculate_dynamic_wma_numba(prices: np.ndarray, dynamic_periods: np.ndarray) -> np.ndarray:
    """動的期間を使用してWMAを計算する（並列処理最適化版）"""
    length = len(prices)
    result = np.full(length, np.nan, dtype=np.float64)
    if length == 0:
        # 空の配列を返す際の型推論エラー対策
        return np.zeros(0, dtype=np.float64)
        
    # 最大期間を事前計算 (メモリ確保の最適化)
    max_period = np.max(dynamic_periods) if length > 0 else 0
    if max_period <= 0 or np.isnan(max_period):
        return result
        
    # 重み計算を事前に行い、再利用する（パフォーマンス向上）
    max_period_int = int(max_period)
    all_weights = np.arange(1.0, max_period_int + 1.0, dtype=np.float64)
    
    for i in prange(length):
        period = int(dynamic_periods[i])
        if period <= 0 or np.isnan(period) or i < period - 1:
            continue  # 結果は既にNaNで初期化済み

        # ウィンドウの取得
        window_start = i - period + 1
        window_prices = prices[window_start:i+1]
        
        # NaNチェック - 高速化のためにnp.isnanを使用
        if np.any(np.isnan(window_prices)):
            continue
            
        # この期間用の重みを取得（事前計算した配列から）
        weights = all_weights[:period]
        weights_sum = period * (period + 1.0) / 2.0
        
        # 行列演算による高速なWMA計算
        result[i] = np.sum(window_prices * weights) / weights_sum
        
    # NaN値のfill forward処理（前の有効な値で埋める）
    last_valid_value = np.nan
    for i in range(length):
        if not np.isnan(result[i]):
            last_valid_value = result[i]
        elif not np.isnan(last_valid_value):
            result[i] = last_valid_value
            
    return result

@njit(fastmath=True, parallel=True, cache=True)
def calculate_dynamic_hma_numba(prices: np.ndarray, dynamic_periods: np.ndarray) -> np.ndarray:
    """動的期間を使用してHMAを計算する（並列処理最適化版）"""
    length = len(prices)
    if length == 0:
        # 空の配列を返す際の型推論エラー対策
        return np.zeros(0, dtype=np.float64)
        
    # 動的期間の前処理 (整数変換と無効値チェック)
    periods = np.round(dynamic_periods).astype(np.int32)
    
    # NaN・負の値・ゼロ期間を検出
    invalid_mask = np.isnan(periods) | (periods <= 0)
    
    # 有効な期間のみで半分と平方根の期間を計算
    half_periods = np.zeros_like(periods)
    sqrt_periods = np.zeros_like(periods)
    
    # 有効な期間のみ処理
    valid_indices = ~invalid_mask
    if np.any(valid_indices):
        valid_periods = periods[valid_indices]
        half_periods[valid_indices] = np.maximum(1, np.floor(valid_periods / 2)).astype(np.int32)
        sqrt_periods[valid_indices] = np.maximum(1, np.floor(np.sqrt(valid_periods))).astype(np.int32)
    
    # 中間WMAを計算
    wma_half = calculate_dynamic_wma_numba(prices, half_periods)
    wma_full = calculate_dynamic_wma_numba(prices, periods)
    
    # 2*WMA(n/2) - WMA(n)の計算
    diff_wma = np.full(length, np.nan, dtype=np.float64)
    not_nan_half = ~np.isnan(wma_half)
    not_nan_full = ~np.isnan(wma_full)
    calc_indices = not_nan_half & not_nan_full  # 両方のWMAが有効な位置
    
    if np.any(calc_indices):
        diff_wma[calc_indices] = 2.0 * wma_half[calc_indices] - wma_full[calc_indices]
    
    # 最終WMA計算
    result = calculate_dynamic_wma_numba(diff_wma, sqrt_periods)
    
    return result


# --- XMA Indicator Class ---

class XMA(Indicator):
    """
    X Moving Average (XMA) インジケーター

    サイクル効率比(CER)またはXTrendIndexに基づいて期間を動的に調整する適応型移動平均線。
    MAタイプとして ALMA, HyperMA, HMA を選択可能。

    特徴:
    - MAタイプを選択可能 (ALMA, HyperMA, HMA)
    - 適応トリガーを選択可能 (CER, XTrendIndex)
    - ピリオドを動的に調整: period = 2 / alpha - 1
      (alphaはトリガーと固定fast/slow期間(2/30)からKAMAスタイルで計算)
    - 価格ソース選択とオプションのカルマンフィルター適用
    """

    SUPPORTED_MA_TYPES = ['alma', 'hyperma', 'hma']
    SUPPORTED_TRIGGER_TYPES = ['cer', 'xtrend']
    FIXED_FAST_PERIOD = 2
    FIXED_SLOW_PERIOD = 55

    def __init__(
        self,
        # --- XMA Core Params ---
        ma_type: str = 'alma',
        trigger_type: str = 'cer', # 'cer' or 'xtrend'
        src_type: str = 'close',
        use_kalman_filter: bool = False,

        # --- Kalman Filter Params (if use_kalman_filter=True) ---
        kalman_measurement_noise: float = 1.0,
        kalman_process_noise: float = 0.01,
        kalman_n_states: int = 5,

        # --- MA Specific Params ---
        alma_offset: float = 0.85, # if ma_type='alma'
        alma_sigma: float = 6.0,   # if ma_type='alma'

        # --- Trigger: CER Params (if trigger_type='cer') ---
        cer_detector_type: str = 'hody',
        cer_lp_period: int = 5,
        cer_hp_period: int = 144,
        cer_cycle_part: float = 0.5,
        cer_max_cycle: int = 144,  # 追加: CER最大サイクル
        cer_min_cycle: int = 5,    # 追加: CER最小サイクル
        cer_max_output: int = 40,  # 追加: CER最大出力
        cer_min_output: int = 10,  # 追加: CER最小出力
        cer_src_type: str = None,  # 更新: CER用ソースタイプ（デフォルトでXMAと同じ）
        cer_use_kalman_filter: bool = None, # 更新: CER用カルマンフィルター（デフォルトでXMAと同じ）

        # --- Trigger: XTrendIndex Params (if trigger_type='xtrend') ---
        # --- XT DC Params ---
        xt_dc_detector_type: str = 'hody',
        xt_dc_cycle_part: float = 0.5,
        xt_dc_max_cycle: int = 55,
        xt_dc_min_cycle: int = 5,
        xt_dc_max_output: int = 34,
        xt_dc_min_output: int = 5,
        xt_dc_src_type: str = 'hlc3', # Note: Separate src for XTrend's internal DC
        xt_dc_lp_period: int = 5,
        xt_dc_hp_period: int = 55,
        # --- XT CATR Smoother Param ---
        xt_catr_smoother_type: str = 'alma',
        # --- XT Internal CER Params ---
        xt_cer_detector_type: str = 'hody',
        xt_cer_lp_period: int = 5,
        xt_cer_hp_period: int = 144,
        xt_cer_cycle_part: float = 0.5,
        # --- XT Threshold Params ---
        xt_max_threshold: float = 0.75,
        xt_min_threshold: float = 0.55
    ):
        """
        XMA（X Moving Average）インジケーター

        トリガー値（CERまたはXTrendIndex）に応じて期間を動的に調整する適応型移動平均線。
        ALMAなどの異なるMAタイプを選択可能。

        Args:
            ma_type (str): 移動平均タイプ ('alma', 'hyperma', 'hma')
            trigger_type (str): トリガータイプ ('cer' or 'xtrend')
            src_type (str): 価格ソース ('close', 'hlc3', etc.)
            use_kalman_filter (bool): カルマンフィルターを使用するか
            kalman_measurement_noise (float): カルマンフィルターの測定ノイズ
            kalman_process_noise (float): カルマンフィルターのプロセスノイズ
            kalman_n_states (int): カルマンフィルターの状態数
            alma_offset (float): ALMA用オフセット
            alma_sigma (float): ALMA用シグマ
            cer_detector_type (str): CER用検出器タイプ
            cer_lp_period (int): CER用LPフィルター期間
            cer_hp_period (int): CER用HPフィルター期間
            cer_cycle_part (float): CER用サイクル部分
            cer_max_cycle (int): CER用最大サイクル期間
            cer_min_cycle (int): CER用最小サイクル期間
            cer_max_output (int): CER用最大出力期間
            cer_min_output (int): CER用最小出力期間
            cer_src_type (str): CER用ソースタイプ
            cer_use_kalman_filter (bool): CER用カルマンフィルターフラグ
            xt_*: XTrendIndex用パラメータ
        """
        if ma_type not in self.SUPPORTED_MA_TYPES:
            raise ValueError(f"Unsupported MA type: {ma_type}. Supported types: {self.SUPPORTED_MA_TYPES}")
        if trigger_type not in self.SUPPORTED_TRIGGER_TYPES:
            raise ValueError(f"Unsupported trigger type: {trigger_type}. Supported types: {self.SUPPORTED_TRIGGER_TYPES}")
        
        # Base settings
        super().__init__(f"XMA(ma={ma_type},trig={trigger_type},src={src_type}_kalman={'Y' if use_kalman_filter else 'N'})")
        
        # Core params
        self.ma_type = ma_type.lower()
        self.trigger_type = trigger_type.lower()
        self.src_type = src_type.lower()
        self.use_kalman_filter = use_kalman_filter
        
        # Kalman filter params
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_process_noise = kalman_process_noise
        self.kalman_n_states = kalman_n_states
        self.kalman_filter = None  # Will be initialized on demand
        
        # MA Specific params
        self.alma_offset = alma_offset
        self.alma_sigma = alma_sigma
        
        # CER params - store these as instance attributes and set defaults where needed
        self.cer_detector_type = cer_detector_type
        self.cer_lp_period = cer_lp_period
        self.cer_hp_period = cer_hp_period
        self.cer_cycle_part = cer_cycle_part
        self.cer_max_cycle = cer_max_cycle
        self.cer_min_cycle = cer_min_cycle
        self.cer_max_output = cer_max_output
        self.cer_min_output = cer_min_output
        self.cer_src_type = src_type.lower() if cer_src_type is None else cer_src_type.lower()
        self.cer_use_kalman_filter = use_kalman_filter if cer_use_kalman_filter is None else cer_use_kalman_filter
        
        # XTrendIndex params - store these as instance attributes
        self.xt_dc_detector_type = xt_dc_detector_type
        self.xt_dc_cycle_part = xt_dc_cycle_part
        self.xt_dc_max_cycle = xt_dc_max_cycle
        self.xt_dc_min_cycle = xt_dc_min_cycle
        self.xt_dc_max_output = xt_dc_max_output
        self.xt_dc_min_output = xt_dc_min_output
        self.xt_dc_src_type = xt_dc_src_type
        self.xt_dc_lp_period = xt_dc_lp_period
        self.xt_dc_hp_period = xt_dc_hp_period
        self.xt_catr_smoother_type = xt_catr_smoother_type
        self.xt_cer_detector_type = xt_cer_detector_type
        self.xt_cer_lp_period = xt_cer_lp_period
        self.xt_cer_hp_period = xt_cer_hp_period
        self.xt_cer_cycle_part = xt_cer_cycle_part
        self.xt_max_threshold = xt_max_threshold
        self.xt_min_threshold = xt_min_threshold
        
        # トリガーインスタンスの初期化
        self._trigger_instance = None

        # --- 結果保存用変数 --- 
        self._cache = {}
        self._values = None
        self._trigger = None
        self._dynamic_periods = None
        self._data_hash = None


    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        data_hash_val = None
        try:
            if isinstance(data, pd.DataFrame):
                # Simplified hash based on shape and first/last row values
                shape_tuple = data.shape
                first_row = tuple(data.iloc[0]) if not data.empty else ()
                last_row = tuple(data.iloc[-1]) if not data.empty else ()
                data_hash_val = hash((shape_tuple, first_row, last_row))
            elif isinstance(data, np.ndarray):
                data_hash_val = hash(data.tobytes())
            else:
                data_hash_val = hash(str(data))
        except Exception as e:
             self.logger.warning(f"データハッシュ計算エラー: {e}. フォールバック使用.", exc_info=False)
             data_hash_val = hash(str(data))

        # Hash all relevant parameters
        param_tuple = (
            self.ma_type, self.trigger_type, self.src_type, self.use_kalman_filter,
            self.alma_offset, self.alma_sigma,
            self.kalman_measurement_noise, self.kalman_process_noise, self.kalman_n_states,
            self.cer_detector_type, self.cer_lp_period, self.cer_hp_period, self.cer_cycle_part,
            self.cer_max_cycle, self.cer_min_cycle, self.cer_max_output, self.cer_min_output,
            self.cer_src_type, self.cer_use_kalman_filter,
            self.xt_dc_detector_type, self.xt_dc_cycle_part, self.xt_dc_max_cycle, self.xt_dc_min_cycle,
            self.xt_dc_max_output, self.xt_dc_min_output, self.xt_dc_src_type,
            self.xt_dc_lp_period, self.xt_dc_hp_period, self.xt_catr_smoother_type,
            self.xt_cer_detector_type, self.xt_cer_lp_period, self.xt_cer_hp_period, self.xt_cer_cycle_part,
            self.xt_max_threshold, self.xt_min_threshold,
            self.FIXED_FAST_PERIOD, self.FIXED_SLOW_PERIOD
        )
        param_hash = hash(param_tuple)

        return f"{data_hash_val}_{param_hash}"


    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        XMA（X Moving Average）を計算する（最適化版）

        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
            
        Returns:
            np.ndarray: XMA値の配列
        """
        try:
            # データチェック
            if data is None or (isinstance(data, pd.DataFrame) and data.empty) or \
               (isinstance(data, np.ndarray) and len(data) == 0):
                self.logger.warning("空のデータが渡されました")
                return np.array([])
                
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            current_hash = self._get_data_hash(data)
            if self._values is not None and current_hash == self._data_hash:
                return self._values.copy()
                
            self._data_hash = current_hash
            self._reset_results()  # 初期化
            
            self.logger.debug("XMA計算開始...")
            
            # --- ソースデータの準備 ---
            # 型変換の最適化: 早期にNumPy配列に変換して処理速度を向上
            source = None
            if self.use_kalman_filter:
                self.logger.debug("カルマンフィルター前処理適用中...")
                # カルマンフィルターの初期化（必要な場合のみ）
                if self.kalman_filter is None:
                    self.kalman_filter = KalmanFilter(
                        price_source=self.src_type,
                        measurement_noise=self.kalman_measurement_noise,
                        process_noise=self.kalman_process_noise,
                        n_states=self.kalman_n_states
                    )
                source = self.kalman_filter.calculate(data)
            else:
                # 直接ソースを使用
                source = PriceSource.calculate_source(data, self.src_type)
                
            # 早期にNumPy配列に変換
            source_np = np.asarray(source, dtype=np.float64)
            source_length = len(source_np)
                
            # ソースデータが空か確認
            if source_length == 0:
                self.logger.warning("ソースデータが空です")
                return np.array([])
                
            # --- トリガー値の取得 ---
            trigger_values = None
            if self.trigger_type == 'cer':
                self.logger.debug("CERトリガー計算中...")
                # CER（サイクル効率比）をトリガーとして使用
                # 既存のインスタンスを再利用（存在する場合）
                if not isinstance(self._trigger_instance, CycleEfficiencyRatio):
                    self._trigger_instance = CycleEfficiencyRatio(
                        detector_type=self.cer_detector_type,
                        lp_period=self.cer_lp_period,
                        hp_period=self.cer_hp_period,
                        cycle_part=self.cer_cycle_part,
                        max_cycle=self.cer_max_cycle,
                        min_cycle=self.cer_min_cycle,
                        max_output=self.cer_max_output,
                        min_output=self.cer_min_output,
                        src_type=self.cer_src_type,
                        use_kalman_filter=self.cer_use_kalman_filter,
                        kalman_measurement_noise=self.kalman_measurement_noise,
                        kalman_process_noise=self.kalman_process_noise,
                        kalman_n_states=self.kalman_n_states
                    )
                # トリガー計算
                trigger_values = self._trigger_instance.calculate(data)
                
            elif self.trigger_type == 'xtrend':
                self.logger.debug("XTrendIndexトリガー計算中...")
                # XTrendIndexをトリガーとして使用
                # 既存のインスタンスを再利用（存在する場合）
                if not isinstance(self._trigger_instance, XTrendIndex):
                    self._trigger_instance = XTrendIndex(
                        # DC設定
                        detector_type=self.xt_dc_detector_type,
                        cycle_part=self.xt_dc_cycle_part,
                        max_cycle=self.xt_dc_max_cycle,
                        min_cycle=self.xt_dc_min_cycle,
                        max_output=self.xt_dc_max_output,
                        min_output=self.xt_dc_min_output,
                        src_type=self.xt_dc_src_type,
                        lp_period=self.xt_dc_lp_period,
                        hp_period=self.xt_dc_hp_period,
                        # CATR設定
                        smoother_type=self.xt_catr_smoother_type,
                        # CER設定
                        cer_detector_type=self.xt_cer_detector_type,
                        cer_lp_period=self.xt_cer_lp_period,
                        cer_hp_period=self.xt_cer_hp_period,
                        cer_cycle_part=self.xt_cer_cycle_part,
                        # しきい値
                        max_threshold=self.xt_max_threshold,
                        min_threshold=self.xt_min_threshold
                    )
                
                # トリガー計算
                result = self._trigger_instance.calculate(data)
                trigger_values = result.values  # XTrendIndexResultオブジェクトから値を取得
            else:
                raise ValueError(f"サポートされていないトリガータイプ: {self.trigger_type}")
                
            # トリガー値をNumPy配列に変換し、絶対値をとってクリップ (0-1の範囲に制限)
            trigger_np = np.asarray(trigger_values, dtype=np.float64)
            trigger_values_clipped = np.clip(np.abs(trigger_np), 0.0, 1.0)
            self._trigger = trigger_values_clipped
            
            # --- 動的ピリオド計算 (最適化版) --- 
            self.logger.debug("動的ピリオド計算中...")
            
            # Numba関数を使用して高速に動的期間を計算
            dynamic_periods = calculate_dynamic_period_kama_style(
                trigger_values_clipped,
                self.FIXED_FAST_PERIOD,
                self.FIXED_SLOW_PERIOD,
                min_period_limit=1  # 最小期間を1に設定
            )
            
            # NaNを処理してforward fill
            # 一旦pandasを使用（numpy-onlyソリューションの方が良い場合は後で最適化）
            periods_series = pd.Series(dynamic_periods)
            periods_filled = periods_series.ffill().fillna(self.FIXED_SLOW_PERIOD)
            
            # 整数に変換
            dynamic_periods = np.round(periods_filled.values).astype(np.int32)
            
            # 結果を保存
            self._dynamic_periods = dynamic_periods
            
            # 配列で期間が正しく計算できたか確認
            self.logger.debug(f"動的期間計算完了: 最小値={np.min(dynamic_periods)}, 最大値={np.max(dynamic_periods)}")
            
            # --- 動的MAの計算 ---
            self.logger.debug(f"動的MA計算中... MAタイプ: {self.ma_type}")
            
            # MAタイプに基づいて計算（最適化したNumba関数を使用）
            result = None
            if self.ma_type == 'alma':
                self.logger.debug(f"ALMAでスムージング中... offset={self.alma_offset}, sigma={self.alma_sigma}")
                result = calculate_dynamic_alma_numba(source_np, dynamic_periods, self.alma_offset, self.alma_sigma)
            elif self.ma_type == 'hyperma':
                self.logger.debug("HyperMAでスムージング中...")
                result = calculate_dynamic_hyper_smoother_numba(source_np, dynamic_periods)
            elif self.ma_type == 'hma':
                self.logger.debug("HMAでスムージング中...")
                result = calculate_dynamic_hma_numba(source_np, dynamic_periods)
            else:
                raise ValueError(f"サポートされていないMAタイプ: {self.ma_type}")
                
            # 結果を保存して返す
            self._values = result
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"XMA '{self.name}' 計算中に予期せぬエラー: {error_msg}\n{stack_trace}")
            return np.array([])

    def _reset_results(self):
        """内部結果をリセットする"""
        self._values = None
        self._trigger = None
        self._dynamic_periods = None
        self._cache = {}

    # --- Getterメソッド --- 
    def get_trigger_values(self) -> Optional[np.ndarray]:
        """計算に使用された元のトリガー値（絶対値化やクリップ前）を取得する"""
        return self._trigger.copy() if self._trigger is not None else None

    def get_dynamic_periods(self) -> Optional[np.ndarray]:
        """計算に使用された最終的な動的期間を取得する"""
        return self._dynamic_periods.copy() if self._dynamic_periods is not None else None

    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._reset_results()
        if hasattr(self, 'kalman_filter') and self.kalman_filter and hasattr(self.kalman_filter, 'reset'):
            self.kalman_filter.reset()
        if hasattr(self, '_trigger_instance') and self._trigger_instance and hasattr(self._trigger_instance, 'reset'):
            self._trigger_instance.reset()
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。")

# # --- Example Usage (Optional) ---
# if __name__ == '__main__':
#     import logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#     # --- サンプルデータ作成 --- 
#     data_len = 200
#     close_prices = np.linspace(100, 120, data_len) + np.random.randn(data_len) * 2 + np.sin(np.linspace(0, 10 * np.pi, data_len)) * 5
#     high_prices = close_prices + np.abs(np.random.randn(data_len)) * 1.0
#     low_prices = close_prices - np.abs(np.random.randn(data_len)) * 1.0
#     open_prices = close_prices + np.random.randn(data_len) * 0.5
#     close_prices[30:40] = np.nan # NaN挿入

#     sample_data_df = pd.DataFrame({
#         'open': open_prices, 'high': high_prices, 'low': low_prices, 'close': close_prices
#     })

#     # --- XMA インスタンス化と計算 --- 
#     print("\n--- Calculating XMA (ALMA, CER trigger) ---")
#     xma_alma_cer = XMA(ma_type='alma', trigger_type='cer', src_type='hlc3')
#     xma_alma_cer_values = xma_alma_cer.calculate(sample_data_df)
#     print(f"Result length: {len(xma_alma_cer_values)}")
#     print("Last 10 values:", xma_alma_cer_values[-10:])
#     periods_alma_cer = xma_alma_cer.get_dynamic_periods()
#     if periods_alma_cer is not None: print("Last 10 periods:", periods_alma_cer[-10:])

#     print("\n--- Calculating XMA (HyperMA, XTrend trigger) ---")
#     xma_hyper_xtrend = XMA(
#         ma_type='hyperma',
#         trigger_type='xtrend',
#         src_type='close',
#         use_kalman_filter=True,
#         # XTrendIndexのパラメータをいくつか変更してみる例
#         xt_dc_max_cycle=89,
#         xt_dc_max_output=55,
#         xt_catr_smoother_type='hyperma' # XTrend内部のCATRもHyperMAに
#     )
#     xma_hyper_xtrend_values = xma_hyper_xtrend.calculate(sample_data_df)
#     print(f"Result length: {len(xma_hyper_xtrend_values)}")
#     print("Last 10 values:", xma_hyper_xtrend_values[-10:])
#     periods_hyper_xtrend = xma_hyper_xtrend.get_dynamic_periods()
#     if periods_hyper_xtrend is not None: print("Last 10 periods:", periods_hyper_xtrend[-10:])

#     print("\n--- Calculating XMA (HMA, CER trigger) ---")
#     xma_hma_cer = XMA(
#         ma_type='hma',
#         trigger_type='cer',
#         src_type='hl2',
#         # CERのパラメータを変更してみる例
#         cer_hp_period=89
#     )
#     xma_hma_cer_values = xma_hma_cer.calculate(sample_data_df)
#     print(f"Result length: {len(xma_hma_cer_values)}")
#     print("Last 10 values:", xma_hma_cer_values[-10:])
#     periods_hma_cer = xma_hma_cer.get_dynamic_periods()
#     if periods_hma_cer is not None: print("Last 10 periods:", periods_hma_cer[-10:])

#     # キャッシュテスト
#     print("\n--- Calculating XMA (HMA, CER trigger) again (cache test) ---")
#     xma_hma_cer_cached = xma_hma_cer.calculate(sample_data_df)
#     assert np.allclose(xma_hma_cer_values, xma_hma_cer_cached, equal_nan=True)
#     print("Cache test passed.") 