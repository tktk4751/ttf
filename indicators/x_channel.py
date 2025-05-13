#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import jit, vectorize, njit, prange
import traceback

# --- 依存関係のインポート ---
try:
    from .indicator import Indicator
    from .xma import XMA
    from .c_atr import CATR
    from .cycle_efficiency_ratio import CycleEfficiencyRatio
    from .x_trend_index import XTrendIndex
    from .price_source import PriceSource
    from .kalman_filter import KalmanFilter
except ImportError:
    # フォールバック (テストや静的解析用)
    print("Warning: Could not import from relative path. Assuming base classes/functions are available.")
    # ダミークラス定義（ここでは省略）
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    # 他のダミークラスも必要に応じて追加


@dataclass
class XChannelResult:
    """Xチャネルの計算結果"""
    middle: np.ndarray          # 中心線 (XMA)
    upper: np.ndarray           # 上限バンド
    lower: np.ndarray           # 下限バンド
    trigger_values: np.ndarray  # 乗数計算に使用したトリガー値 (CER or XTrendIndex)
    dynamic_multiplier: np.ndarray # 動的ATR乗数 (非線形計算)
    catr: np.ndarray            # CATR値 (金額ベース)
    max_mult_values: np.ndarray # 動的に計算された最大乗数値
    min_mult_values: np.ndarray # 動的に計算された最小乗数値
    # オプションで追加情報 (デバッグ用など)
    xma_values: Optional[np.ndarray] = None
    catr_result: Optional[Any] = None
    trigger_instance_result: Optional[Any] = None


@vectorize(['float64(float64)'], nopython=True, fastmath=True, target='parallel', cache=True)
def calculate_nonlinear_multiplier_vec(trigger_value: float) -> float:
    """
    トリガー値 (0-1) に基づいて非線形なATR乗数を計算する（ベクトル化・並列処理最適化版）
    トリガー値が高い（トレンド強い）ほど乗数は小さくなる (min_multに近づく)
    トリガー値が低い（レンジ強い）ほど乗数は大きくなる (max_multに近づく)

    Args:
        trigger_value: トリガー値 (0から1の範囲)

    Returns:
        動的な乗数の値
    """
    # 定数値（コンパイル時に最適化される）
    MAX_MULT = 9.0
    MIN_MULT = 0.3
    K = 2.0 # べき乗の指数
    RANGE = MAX_MULT - MIN_MULT

    # NaN処理（早期リターン）
    if np.isnan(trigger_value):
        return MAX_MULT

    # トリガー値を0-1の範囲にクリップ（高速なmin/max）
    clamped_trigger = max(0.0, min(1.0, trigger_value))
    
    # 非線形計算（最適化）: multiplier = min + (max - min) * (1 - trigger^k)
    # pow関数を避けて直接乗算（k=2の場合）
    trigger_squared = clamped_trigger * clamped_trigger
    multiplier = MIN_MULT + RANGE * (1.0 - trigger_squared)
    
    return multiplier


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_multiplier_vec(cer: float, max_mult: float, min_mult: float) -> float:
    """
    サイクル効率比に基づいて動的なATR乗数を計算する（ベクトル化版）
    
    Args:
        cer: サイクル効率比の値
        max_mult: 最大乗数
        min_mult: 最小乗数
    
    Returns:
        動的な乗数の値
    """
    # CERが高い（トレンドが強い）ほど乗数は小さく、
    # CERが低い（トレンドが弱い）ほど乗数は大きくなる
    if np.isnan(cer):
        return np.nan
    cer_abs = abs(cer)
    return max_mult - cer_abs * (max_mult - min_mult)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_max_multiplier(cer: float, max_max_mult: float, min_max_mult: float) -> float:
    """
    サイクル効率比に基づいて動的な最大ATR乗数を計算する（ベクトル化版）
    
    Args:
        cer: サイクル効率比の値
        max_max_mult: 最大乗数の最大値（例：8.0）
        min_max_mult: 最大乗数の最小値（例：3.0）
    
    Returns:
        動的な最大乗数の値
    """
    if np.isnan(cer):
        return np.nan
    cer_abs = abs(cer)
    # CERが低い（トレンドが弱い）ほど最大乗数は大きく、
    # CERが高い（トレンドが強い）ほど最大乗数は小さくなる
    return max_max_mult - cer_abs * (max_max_mult - min_max_mult)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_min_multiplier(cer: float, max_min_mult: float, min_min_mult: float) -> float:
    """
    サイクル効率比に基づいて動的な最小ATR乗数を計算する（ベクトル化版）
    
    Args:
        cer: サイクル効率比の値
        max_min_mult: 最小乗数の最大値（例：1.5）
        min_min_mult: 最小乗数の最小値（例：0.3）
    
    Returns:
        動的な最小乗数の値
    """
    if np.isnan(cer):
        return np.nan
    cer_abs = abs(cer)
    # CERが低い（トレンドが弱い）ほど最小乗数は小さく、
    # CERが高い（トレンドが強い）ほど最小乗数は大きくなる
    return max_min_mult - cer_abs * (max_min_mult - min_min_mult)


@njit(fastmath=True, parallel=True, cache=True)
def calculate_x_channel(
    xma: np.ndarray,
    catr_absolute: np.ndarray,
    dynamic_multiplier: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Xチャネル（X Channel）の計算（ベクトル化・並列化最適化版）

    Args:
        xma: 中心線（XMA）の配列
        catr_absolute: 絶対CATR値の配列
        dynamic_multiplier: 動的ATR乗数の配列

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (中心線, 上限バンド, 下限バンド)のタプル
    """
    # データ長の確認
    len_xma = len(xma)
    len_catr = len(catr_absolute)
    len_multiplier = len(dynamic_multiplier)
    
    # 配列長が異なる場合は最小の長さを使用
    length = min(len_xma, len_catr, len_multiplier)
    if length <= 0:
        # 空の配列の場合は空の結果を返す (Numba型推論エラー対応版)
        # 事前に型付けされた空の配列を作成
        empty_array = np.zeros(0, dtype=np.float64)
        return empty_array, empty_array, empty_array
    
    # 必要に応じて配列をトリミング
    xma_adj = xma[-length:] if len_xma > length else xma
    catr_adj = catr_absolute[-length:] if len_catr > length else catr_absolute
    mult_adj = dynamic_multiplier[-length:] if len_multiplier > length else dynamic_multiplier
    
    # 結果配列の初期化
    upper_band = np.empty(length, dtype=np.float64)
    lower_band = np.empty(length, dtype=np.float64)
    
    # バンド計算（ベクトル化と並列処理の組み合わせ）
    # 小さな配列ではベクトル化が、大きな配列では並列処理が効率的
    if length < 1000:
        # ベクトル化計算（小さな配列向け）
        atr_width = catr_adj * mult_adj
        upper_band = xma_adj + atr_width
        lower_band = xma_adj - atr_width
    else:
        # パフォーマンス最適化のためにバッチに分割して並列処理
        batch_size = max(1000, length // 16)  # 16はコアごとのバッチ
        num_batches = (length + batch_size - 1) // batch_size  # 切り上げ除算

        for batch_idx in prange(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, length)
            
            # 各バッチで処理
            for i in range(start_idx, end_idx):
                atr_width = catr_adj[i] * mult_adj[i]
                upper_band[i] = xma_adj[i] + atr_width
                lower_band[i] = xma_adj[i] - atr_width
    
    return xma_adj, upper_band, lower_band


class XChannel(Indicator):
    """
    Xチャネル（X Channel）インジケーター（最適化版）

    特徴:
    - 中心線にXMA（X Moving Average）を使用
    - バンド幅の計算にCATR（Cycle Average True Range）を使用
    - ATR乗数がトリガー値（CERまたはXTrendIndex）に基づいて非線形に調整される
    - 価格ソースとカルマンフィルターを選択可能
    - NumbaとNumPyによる高速化対応

    市場状態に応じた最適な挙動:
    - トレンド強い（トリガー値高い）:
      - 狭いバンド幅（小さい乗数）でトレンドをタイトに追従
    - トレンド弱い（トリガー値低い）:
      - 広いバンド幅（大きい乗数）でレンジ相場の振れ幅を捉える
    """
    # 乗数の上限・下限値を定数として定義
    MAX_MULT = 10.0
    MIN_MULT = 0.1
    # 動的乗数の範囲
    MAX_MAX_MULT = 8.0    # 最大乗数の最大値
    MIN_MAX_MULT = 3.0    # 最大乗数の最小値
    MAX_MIN_MULT = 1.5    # 最小乗数の最大値
    MIN_MIN_MULT = 0.5    # 最小乗数の最小値
    # キャッシュサイズ制限
    MAX_CACHE_SIZE = 5

    def __init__(
        self,
        # Kalman params (shared for XMA/CATR/Trigger if used)
        kalman_measurement_noise: float = 1.0,
        kalman_process_noise: float = 0.01,
        kalman_n_states: int = 5,
        # --- 動的乗数パラメータ ---
        max_max_multiplier: float = 9.0,    # 最大乗数の最大値
        min_max_multiplier: float = 3.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.3,    # 最小乗数の最大値
        min_min_multiplier: float = 0.3,    # 最小乗数の最小値
        # --- XMA Params ---
        # Alma specific (passed to XMA if ma_type='alma')
        xma_alma_offset: float = 0.85,
        xma_alma_sigma: float = 6.0,
        xma_ma_type: str = 'alma',
        xma_src_type: str = 'close', # XMA calculation source
        xma_use_kalman_filter: bool = False, # Kalman for XMA source
        xma_trigger_type: str = 'cer', # XMA's internal trigger
        # --- XMA's internal CER params ---
        xma_cer_detector_type: str = 'hody',
        xma_cer_lp_period: int = 5,
        xma_cer_hp_period: int = 144,
        xma_cer_cycle_part: float = 0.5,
        xma_cer_max_cycle: int = 144,  # 追加: CER最大サイクル
        xma_cer_min_cycle: int = 5,    # 追加: CER最小サイクル
        xma_cer_max_output: int = 40,  # 追加: CER最大出力
        xma_cer_min_output: int = 10,  # 追加: CER最小出力
        xma_cer_src_type: str = None,  # 追加: CER用ソースタイプ
        xma_cer_use_kalman_filter: bool = None, # 追加: CER用カルマンフィルター
        # --- XMA's internal XTrend params ---
        xma_xt_dc_detector_type: str = 'hody',
        xma_xt_dc_cycle_part: float = 0.5,
        xma_xt_dc_max_cycle: int = 55,
        xma_xt_dc_min_cycle: int = 5,
        xma_xt_dc_max_output: int = 34,
        xma_xt_dc_min_output: int = 5,
        xma_xt_dc_src_type: str = 'hlc3',
        xma_xt_dc_lp_period: int = 5,
        xma_xt_dc_hp_period: int = 55,
        xma_xt_catr_smoother_type: str = 'alma',
        xma_xt_cer_detector_type: str = 'hody',
        xma_xt_cer_lp_period: int = 5,
        xma_xt_cer_hp_period: int = 144,
        xma_xt_cer_cycle_part: float = 0.5,
        xma_xt_max_threshold: float = 0.75,
        xma_xt_min_threshold: float = 0.55,

        # --- CATR Params ---
        catr_detector_type: str = 'hody', # DC for CATR
        catr_cycle_part: float = 0.5,
        catr_lp_period: int = 5,
        catr_hp_period: int = 55,
        catr_max_cycle: int = 55,
        catr_min_cycle: int = 5,
        catr_max_output: int = 34,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'alma',
        catr_src_type: str = 'close', # CATR calculation source (can differ from XMA)
        catr_use_kalman_filter: bool = False, # Kalman for CATR source

        # --- Multiplier Trigger Params ---
        multiplier_trigger_type: str = 'cer', # 'cer' or 'xtrend'
        # CER trigger params
        mult_cer_detector_type: str = 'hody_e', # Use enhanced for trigger?
        mult_cer_lp_period: int = 5,
        mult_cer_hp_period: int = 55,
        mult_cer_cycle_part: float = 0.55,
        mult_cer_max_cycle: int = 55,
        mult_cer_min_cycle: int = 3,
        mult_cer_max_output: int = 55,
        mult_cer_min_output: int = 5,
        mult_cer_src_type: str = 'hlc3', # Source for multiplier trigger CER
        mult_cer_use_kalman_filter: bool = False, # Kalman for multiplier trigger CER source
        # XTrend trigger params
        mult_xt_dc_detector_type: str = 'phac_e',
        mult_xt_dc_cycle_part: float = 0.55,
        mult_xt_dc_max_cycle: int = 100,
        mult_xt_dc_min_cycle: int = 3,
        mult_xt_dc_max_output: int = 55,
        mult_xt_dc_min_output: int = 13,
        mult_xt_dc_src_type: str = 'hlc3',
        mult_xt_dc_lp_period: int = 5,
        mult_xt_dc_hp_period: int = 55,
        mult_xt_catr_smoother_type: str = 'alma',
        mult_xt_cer_detector_type: str = 'hody',
        mult_xt_cer_lp_period: int = 5,
        mult_xt_cer_hp_period: int = 144,
        mult_xt_cer_cycle_part: float = 0.5,
        mult_xt_max_threshold: float = 0.75,
        mult_xt_min_threshold: float = 0.55

        # General Params
        # src_type: str = 'close', # Overall source? Or let internal handle? -> Let internal handle
        # use_kalman_filter: bool = False # Overall kalman? -> Let internal handle

    ):
        """
        コンストラクタ（最適化版）
        (パラメータ詳細は各コンポーネントのドキュメント参照)
        """
        # Build indicator name dynamically
        trigger_name = f"multTrig={multiplier_trigger_type}"
        xma_name = f"xma={xma_ma_type}"
        catr_name = f"catr={catr_smoother_type}"
        indicator_name = f"XChannel({xma_name},{catr_name},{trigger_name})"
        super().__init__(indicator_name)

        # 事前チェック - トリガータイプの検証
        multiplier_trigger_type = multiplier_trigger_type.lower()
        if multiplier_trigger_type not in ['cer', 'xtrend']:
            raise ValueError("multiplier_trigger_type must be 'cer' or 'xtrend'")

        # 依存関係を効率的にインスタンス化するための変数格納
        # --- Store Params (コンパクトに整理)---
        # Kalman params (共有パラメータ)
        self.kalman_params = {
            'measurement_noise': kalman_measurement_noise,
            'process_noise': kalman_process_noise,
            'n_states': kalman_n_states
        }
        
        # 動的乗数パラメータ
        self.max_max_multiplier = max_max_multiplier
        self.min_max_multiplier = min_max_multiplier
        self.max_min_multiplier = max_min_multiplier
        self.min_min_multiplier = min_min_multiplier
        
        # XMA Params (必要なパラメータのみを辞書に格納)
        self.xma_params = {
            # Core
            'ma_type': xma_ma_type.lower(),
            'trigger_type': xma_trigger_type.lower(),
            'src_type': xma_src_type.lower(),
            'use_kalman_filter': xma_use_kalman_filter,
            # Alma specific
            'alma_offset': xma_alma_offset,
            'alma_sigma': xma_alma_sigma,
            # CER params
            'cer_detector_type': xma_cer_detector_type,
            'cer_lp_period': xma_cer_lp_period,
            'cer_hp_period': xma_cer_hp_period,
            'cer_cycle_part': xma_cer_cycle_part,
            'cer_max_cycle': xma_cer_max_cycle,
            'cer_min_cycle': xma_cer_min_cycle,
            'cer_max_output': xma_cer_max_output,
            'cer_min_output': xma_cer_min_output,
            'cer_src_type': xma_cer_src_type,
            'cer_use_kalman_filter': xma_cer_use_kalman_filter,
            # XTrend params
            'xt_dc_detector_type': xma_xt_dc_detector_type,
            'xt_dc_cycle_part': xma_xt_dc_cycle_part,
            'xt_dc_max_cycle': xma_xt_dc_max_cycle,
            'xt_dc_min_cycle': xma_xt_dc_min_cycle,
            'xt_dc_max_output': xma_xt_dc_max_output,
            'xt_dc_min_output': xma_xt_dc_min_output,
            'xt_dc_src_type': xma_xt_dc_src_type,
            'xt_dc_lp_period': xma_xt_dc_lp_period,
            'xt_dc_hp_period': xma_xt_dc_hp_period,
            'xt_catr_smoother_type': xma_xt_catr_smoother_type,
            'xt_cer_detector_type': xma_xt_cer_detector_type,
            'xt_cer_lp_period': xma_xt_cer_lp_period,
            'xt_cer_hp_period': xma_xt_cer_hp_period,
            'xt_cer_cycle_part': xma_xt_cer_cycle_part,
            'xt_max_threshold': xma_xt_max_threshold,
            'xt_min_threshold': xma_xt_min_threshold,
        }
        
        # CATR Params
        self.catr_params = {
            'detector_type': catr_detector_type,
            'cycle_part': catr_cycle_part,
            'lp_period': catr_lp_period,
            'hp_period': catr_hp_period,
            'max_cycle': catr_max_cycle,
            'min_cycle': catr_min_cycle,
            'max_output': catr_max_output,
            'min_output': catr_min_output,
            'smoother_type': catr_smoother_type,
            'src_type': catr_src_type,
            'use_kalman_filter': catr_use_kalman_filter,
        }
        
        # Multiplier Trigger Params
        self.multiplier_trigger_type = multiplier_trigger_type
        
        # CER Params for Multiplier Trigger
        self.mult_cer_params = {
            'detector_type': mult_cer_detector_type,
            'lp_period': mult_cer_lp_period,
            'hp_period': mult_cer_hp_period,
            'cycle_part': mult_cer_cycle_part,
            'max_cycle': mult_cer_max_cycle,
            'min_cycle': mult_cer_min_cycle,
            'max_output': mult_cer_max_output,
            'min_output': mult_cer_min_output,
            'src_type': mult_cer_src_type,
            'use_kalman_filter': mult_cer_use_kalman_filter,
        }
        
        # XTrend Params for Multiplier Trigger
        self.mult_xt_params = {
            'detector_type': mult_xt_dc_detector_type,
            'cycle_part': mult_xt_dc_cycle_part,
            'max_cycle': mult_xt_dc_max_cycle,
            'min_cycle': mult_xt_dc_min_cycle,
            'max_output': mult_xt_dc_max_output,
            'min_output': mult_xt_dc_min_output,
            'src_type': mult_xt_dc_src_type,
            'lp_period': mult_xt_dc_lp_period,
            'hp_period': mult_xt_dc_hp_period,
            'catr_smoother_type': mult_xt_catr_smoother_type,
            'cer_detector_type': mult_xt_cer_detector_type,
            'cer_lp_period': mult_xt_cer_lp_period,
            'cer_hp_period': mult_xt_cer_hp_period,
            'cer_cycle_part': mult_xt_cer_cycle_part,
            'max_threshold': mult_xt_max_threshold,
            'min_threshold': mult_xt_min_threshold,
        }

        # --- 遅延初期化変数の準備 ---
        # 必要になるまでインスタンス化しないようにする（リソース節約）
        self._xma = None
        self._catr = None
        self._catr_required_cer = None
        self._multiplier_trigger_instance = None
        
        # Kalman filters - 必要に応じて初期化
        self._xma_kalman = None
        self._catr_kalman = None
        self._mult_cer_kalman = None

        # Result cache
        self._result = None
        self._cache = {}  # {data_hash: XChannelResult} 形式
        self._cache_keys = []  # キャッシュキーの順序を保持（LRU実装のため）
        self._data_hash = None


    def _initialize_components(self):
        """コンポーネントの遅延初期化（必要時のみ）"""
        kalman_params = self.kalman_params
        
        # 1. XMAの初期化（必要時のみ）
        if self._xma is None:
            self.logger.debug("XMAコンポーネントを初期化中...")
            
            # パラメータを展開して渡す（**演算子を活用）
            self._xma = XMA(
                **{k: v for k, v in self.xma_params.items()},
                kalman_measurement_noise=kalman_params['measurement_noise'],
                kalman_process_noise=kalman_params['process_noise'],
                kalman_n_states=kalman_params['n_states']
            )
        
        # 2. CER初期化（乗数トリガー用とCATR用）
        if self._catr_required_cer is None:
            self.logger.debug("CERコンポーネント（CATR用）を初期化中...")
            
            # CER params with Kalman - 必要なパラメータを全て含める
            self._catr_required_cer = CycleEfficiencyRatio(
                detector_type=self.mult_cer_params['detector_type'],
                lp_period=self.mult_cer_params['lp_period'],
                hp_period=self.mult_cer_params['hp_period'],
                cycle_part=self.mult_cer_params['cycle_part'],
                max_cycle=self.mult_cer_params['max_cycle'],
                min_cycle=self.mult_cer_params['min_cycle'],
                max_output=self.mult_cer_params['max_output'],
                min_output=self.mult_cer_params['min_output'],
                src_type=self.mult_cer_params['src_type'],
                use_kalman_filter=self.mult_cer_params['use_kalman_filter'],
                kalman_measurement_noise=kalman_params['measurement_noise'],
                kalman_process_noise=kalman_params['process_noise'],
                kalman_n_states=kalman_params['n_states']
            )
        
        # 3. CATR初期化
        if self._catr is None:
            self.logger.debug("CATRコンポーネントを初期化中...")
            
            # パラメータを展開して渡す
            self._catr = CATR(
                **{k: v for k, v in self.catr_params.items()},
                kalman_measurement_noise=kalman_params['measurement_noise'],
                kalman_process_noise=kalman_params['process_noise'],
                kalman_n_states=kalman_params['n_states']
            )
        
        # 4. 乗数トリガーインスタンス初期化
        if self._multiplier_trigger_instance is None:
            self.logger.debug(f"乗数トリガー（{self.multiplier_trigger_type}）を初期化中...")
            
            if self.multiplier_trigger_type == 'cer':
                # CERを乗数トリガーとして使用（既に初期化済み）
                self._multiplier_trigger_instance = self._catr_required_cer
            else:  # 'xtrend'
                # XTrend params
                self._multiplier_trigger_instance = XTrendIndex(
                    detector_type=self.mult_xt_params['detector_type'],
                    cycle_part=self.mult_xt_params['cycle_part'],
                    max_cycle=self.mult_xt_params['max_cycle'],
                    min_cycle=self.mult_xt_params['min_cycle'],
                    max_output=self.mult_xt_params['max_output'],
                    min_output=self.mult_xt_params['min_output'],
                    src_type=self.mult_xt_params['src_type'],
                    lp_period=self.mult_xt_params['lp_period'],
                    hp_period=self.mult_xt_params['hp_period'],
                    smoother_type=self.mult_xt_params['catr_smoother_type'],
                    cer_detector_type=self.mult_xt_params['cer_detector_type'],
                    cer_lp_period=self.mult_xt_params['cer_lp_period'],
                    cer_hp_period=self.mult_xt_params['cer_hp_period'],
                    cer_cycle_part=self.mult_xt_params['cer_cycle_part'],
                    max_threshold=self.mult_xt_params['max_threshold'],
                    min_threshold=self.mult_xt_params['min_threshold']
                )
        
        return True  # 初期化完了


    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データと全パラメータに基づいてハッシュ値を計算する"""
        data_hash_val = None
        try:
            # Determine relevant columns based on all used src_types
            all_src_types = {
                self.xma_params['src_type'],
                self.catr_params['src_type'],
                self.mult_cer_params['src_type'] if self.multiplier_trigger_type == 'cer' else self.mult_xt_params['src_type']
            }
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
            # XMA params
            self.xma_params['ma_type'], self.xma_params['src_type'], self.xma_params['use_kalman_filter'],
            self.xma_params['alma_offset'], self.xma_params['alma_sigma'],
            # CATR params
            self.catr_params['detector_type'], self.catr_params['smoother_type'], self.catr_params['src_type'], self.catr_params['use_kalman_filter'],
            # Multiplier Trigger
            self.multiplier_trigger_type,
            # Kalman general
            self.kalman_params['measurement_noise'], self.kalman_params['process_noise'], self.kalman_params['n_states'],
            # CER/XTrend主要パラメータ
            self.mult_cer_params['detector_type'], self.mult_cer_params['cycle_part'], self.mult_cer_params['max_cycle'],
            self.mult_xt_params['detector_type'], self.mult_xt_params['max_threshold'], self.mult_xt_params['min_threshold']
        )
        param_hash = hash(param_tuple)

        return f"{data_hash_val}_{param_hash}"


    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Xチャネルを計算する（最適化版）
        
        Args:
            data: 価格データ (pandas.DataFrame または numpy.ndarray)
            
        Returns:
            中心線（XMA）の配列
        """
        try:
            # データの事前チェック - 早期リターン最適化
            if data is None or (isinstance(data, pd.DataFrame) and data.empty) or \
               (isinstance(data, np.ndarray) and data.size == 0):
                self.logger.warning("入力データが空です")
                return np.array([])
                
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            current_hash = self._get_data_hash(data)
            if current_hash in self._cache:
                # LRUキャッシュの更新（使用されたキーを最新にする）
                self._cache_keys.remove(current_hash)
                self._cache_keys.append(current_hash)
                self._result = self._cache[current_hash]
                return self._result.middle
                
            if self._result is not None and current_hash == self._data_hash:
                return self._result.middle
            
            # コンポーネントの遅延初期化
            self._initialize_components()
                
            # 新しいハッシュを保存
            self._data_hash = current_hash

            # --- 1. XMA計算 ---
            self.logger.debug("XMA計算中...")
            xma_values = self._xma.calculate(data)
            
            # XMA計算結果をチェック
            if xma_values is None or len(xma_values) == 0:
                self.logger.error("XMAの計算結果が空です")
                return np.array([])
            
            # --- 2. 乗数トリガー計算 ---
            self.logger.debug("乗数トリガー計算中...")
            trigger_values, er_values = self._calculate_trigger_and_er(data)
            
            # 無効なトリガー値をチェック
            if trigger_values is None or len(trigger_values) == 0:
                self.logger.error("有効なトリガー値が計算できませんでした")
                return np.array([])
            
            # --- 3. 非線形乗数計算 - ベクトル化関数を使用 ---
            self.logger.debug("非線形乗数計算中...")
            # NumPy配列に変換して高速化（ベクトル処理のため）
            trigger_np = np.asarray(trigger_values, dtype=np.float64)
            # 絶対値を取り、0-1の範囲にクリッピング
            trigger_abs = np.abs(trigger_np)
            
            # 動的な最大・最小乗数の計算
            max_mult_values = calculate_dynamic_max_multiplier(
                trigger_abs,
                self.max_max_multiplier,
                self.min_max_multiplier
            )
            
            min_mult_values = calculate_dynamic_min_multiplier(
                trigger_abs,
                self.max_min_multiplier,
                self.min_min_multiplier
            )
            
            # 動的ATR乗数の計算
            dynamic_multiplier = calculate_dynamic_multiplier_vec(
                trigger_abs,
                max_mult_values,
                min_mult_values
            )
            
            # --- 4. CATR計算 ---
            self.logger.debug("CATR計算中...")
            if er_values is None or len(er_values) == 0:
                self.logger.error("有効なER値がありません。CATRの計算ができません。")
                return np.array([])
                
            # CATRにER値を渡して計算
            try:
                self._catr.calculate(data, external_er=er_values)
                catr_absolute_values = self._catr.get_absolute_atr()
                
                # CATRの結果をチェック
                if catr_absolute_values is None or len(catr_absolute_values) == 0:
                    self.logger.error("CATR計算結果が空です")
                    return np.array([])
                    
            except Exception as e:
                self.logger.error(f"CATR計算中にエラーが発生しました: {str(e)}")
                return np.array([])
            
            # --- 5. 配列サイズの調整（Numba関数内でも処理されるが、事前にチェック）---
            try:
                xma_values_np = np.asarray(xma_values, dtype=np.float64)
                catr_absolute_np = np.asarray(catr_absolute_values, dtype=np.float64)
                
                # 配列の長さを確認（デバッグ情報として）
                xma_len = len(xma_values_np)
                catr_len = len(catr_absolute_np)
                mult_len = len(dynamic_multiplier)
                
                # 配列長の不一致を警告
                if xma_len != catr_len or xma_len != mult_len:
                    self.logger.warning(f"配列サイズの不一致: XMA={xma_len}, CATR={catr_len}, Multiplier={mult_len}")
                
                # --- 6. Xチャネル計算（最適化されたNumba関数を使用）---
                self.logger.debug("Xチャネル計算中...")
                
                # Numba関数呼び出しをtry-exceptで囲む
                middle, upper, lower = calculate_x_channel(
                    xma_values_np,
                    catr_absolute_np,
                    dynamic_multiplier
                )
                
                # 結果チェック
                if middle is None or len(middle) == 0:
                    self.logger.error("チャネル計算結果が空です")
                    return np.array([])
                
            except Exception as e:
                self.logger.error(f"チャネル計算中にエラーが発生しました: {str(e)}")
                return np.array([])
            
            # --- 7. 結果の保存とキャッシュ更新 ---
            result = XChannelResult(
                middle=middle,
                upper=upper,
                lower=lower,
                trigger_values=trigger_values,
                dynamic_multiplier=dynamic_multiplier,
                catr=catr_absolute_values,
                xma_values=xma_values,
                catr_result=self._catr,
                trigger_instance_result=self._trigger_instance_result,
                max_mult_values=max_mult_values,
                min_mult_values=min_mult_values,
            )
            
            # LRUキャッシュに保存（最大サイズ制限）
            self._cache[current_hash] = result
            self._cache_keys.append(current_hash)
            
            # キャッシュが最大サイズを超えたら最も古いエントリを削除
            if len(self._cache_keys) > self.MAX_CACHE_SIZE:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]
            
            self._result = result
            return middle
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"XChannel計算エラー: {error_msg}\n{stack_trace}")
            return np.array([])
            
    def _calculate_trigger_and_er(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        トリガー値とER値を計算する（効率化のため分離）
        
        Args:
            data: 価格データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (トリガー値, ER値)
        """
        trigger_values = None
        er_values = None
        
        try:
            # コンポーネント初期化を確認
            if self._catr_required_cer is None or self._multiplier_trigger_instance is None:
                self._initialize_components()  # 必要なコンポーネントを初期化

            if self.multiplier_trigger_type == 'cer':
                # CERを乗数トリガーとして使用
                er_values = self._catr_required_cer.calculate(data)
                trigger_values = np.abs(er_values)  # CERの絶対値
                self._trigger_instance_result = self._catr_required_cer
            else:  # 'xtrend'
                # XTrendIndexを乗数トリガーとして使用
                result = self._multiplier_trigger_instance.calculate(data)
                trigger_values = result.values  # XTrendIndexResultオブジェクトから値を取得
                
                # XTrendの場合もER値を取得（CATRに必要）
                if hasattr(self._multiplier_trigger_instance, 'get_er'):
                    er_values = self._multiplier_trigger_instance.get_er()
                else:
                    # 無ければCERを計算してER値を取得
                    if self._catr_required_cer is None:
                        # CERの初期化が必要
                        kalman_params = self.kalman_params
                        self._catr_required_cer = CycleEfficiencyRatio(
                            detector_type=self.mult_cer_params['detector_type'],
                            lp_period=self.mult_cer_params['lp_period'],
                            hp_period=self.mult_cer_params['hp_period'],
                            cycle_part=self.mult_cer_params['cycle_part'],
                            max_cycle=self.mult_cer_params['max_cycle'],
                            min_cycle=self.mult_cer_params['min_cycle'],
                            max_output=self.mult_cer_params['max_output'],
                            min_output=self.mult_cer_params['min_output'],
                            src_type=self.mult_cer_params['src_type'],
                            use_kalman_filter=self.mult_cer_params['use_kalman_filter'],
                            kalman_measurement_noise=kalman_params['measurement_noise'],
                            kalman_process_noise=kalman_params['process_noise'],
                            kalman_n_states=kalman_params['n_states']
                        )
                    er_values = self._catr_required_cer.calculate(data)
                    
                self._trigger_instance_result = self._multiplier_trigger_instance
                
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"トリガー計算エラー: {error_msg}")
            # エラー時は空の配列を返す
            return np.array([]), np.array([])
            
        return trigger_values, er_values


    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Xチャネルのバンド値を取得する

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                (中心線, 上限バンド, 下限バンド)の値
        """
        if self._result is None:
            empty = np.array([])
            return empty, empty, empty
        return self._result.middle.copy(), self._result.upper.copy(), self._result.lower.copy()

    def get_trigger_values(self) -> np.ndarray:
        """
        乗数計算に使用されたトリガー値（0-1の範囲）を取得する

        Returns:
            np.ndarray: トリガー値
        """
        if self._result is None:
            return np.array([])
        return self._result.trigger_values.copy()

    def get_dynamic_multiplier(self) -> np.ndarray:
        """
        動的ATR乗数の値を取得する

        Returns:
            np.ndarray: 動的ATR乗数の値
        """
        if self._result is None:
            return np.array([])
        return self._result.dynamic_multiplier.copy()

    def get_dynamic_max_multiplier(self) -> np.ndarray:
        """
        動的な最大ATR乗数の値を取得する
        
        Returns:
            np.ndarray: 動的最大ATR乗数の値
        """
        if self._result is None:
            return np.array([])
        return self._result.max_mult_values.copy()
        
    def get_dynamic_min_multiplier(self) -> np.ndarray:
        """
        動的な最小ATR乗数の値を取得する
        
        Returns:
            np.ndarray: 動的最小ATR乗数の値
        """
        if self._result is None:
            return np.array([])
        return self._result.min_mult_values.copy()

    def get_catr(self) -> np.ndarray:
        """
        CATR値（金額ベース）を取得する

        Returns:
            np.ndarray: CATR値
        """
        if self._result is None:
            return np.array([])
        return self._result.catr.copy()

    def get_result(self) -> Optional[XChannelResult]:
        """
        計算結果の全詳細を含むオブジェクトを取得する

        Returns:
            Optional[XChannelResult]: 計算結果オブジェクト、または計算されていない場合はNone
        """
        return self._result # Returns a reference, copy if mutation is a concern

    def reset(self) -> None:
        """
        インジケーターの状態をリセットする（最適化版）
        キャッシュをクリアして、コンポーネントを再初期化可能な状態に戻します。
        """
        super().reset()
        
        # 結果とキャッシュをクリア
        self._result = None
        self._cache.clear()
        self._cache_keys.clear()
        self._data_hash = None
        
        # 初期化済みのコンポーネントをリセット
        for component_name in ['_xma', '_catr', '_catr_required_cer', '_multiplier_trigger_instance']:
            component = getattr(self, component_name, None)
            if component and hasattr(component, 'reset'):
                try:
                    component.reset()
                    self.logger.debug(f"{component_name}をリセットしました")
                except Exception as e:
                    self.logger.warning(f"{component_name}のリセット中にエラーが発生しました: {e}")
                
        # Kalman filtersをリセット
        for kalman_name in ['_xma_kalman', '_catr_kalman', '_mult_cer_kalman']:
            kalman = getattr(self, kalman_name, None)
            if kalman and hasattr(kalman, 'reset'):
                try:
                    kalman.reset()
                    self.logger.debug(f"{kalman_name}をリセットしました")
                except Exception as e:
                    self.logger.warning(f"{kalman_name}のリセット中にエラーが発生しました: {e}")

        self.logger.debug(f"インジケーター '{self.name}' をリセットしました") 