#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import jit, njit, prange
import traceback

from .indicator import Indicator
from .hyper_smoother import hyper_smoother
from .cycle_efficiency_ratio import CycleEfficiencyRatio
from .c_atr import CATR
from .ehlers_unified_dc import EhlersUnifiedDC # または EhlersHoDyDC

# 定数
DEFAULT_CYCLE_PART = 0.5
DEFAULT_MAX_CYCLE = 55
DEFAULT_MIN_CYCLE = 5
DEFAULT_MAX_OUTPUT = 34
DEFAULT_MIN_OUTPUT = 5
DEFAULT_LP_PERIOD = 5
DEFAULT_HP_PERIOD = 55
DEFAULT_SMOOTHER = 'alma'
DEFAULT_DC_DETECTOR = 'hody'


@dataclass
class CTrendIndexResult:
    """Cトレンドインデックスの計算結果"""
    values: np.ndarray          # Cトレンドインデックスの値（0-1の範囲）
    er: np.ndarray              # サイクル効率比（CER）
    chop_period: np.ndarray     # チョピネス期間（DC検出器から）
    atr_period: np.ndarray      # ATR期間（CATRから）
    stddev_period: np.ndarray   # 標準偏差期間（DC検出器から）
    lookback_period: np.ndarray # 標準偏差ルックバック期間（DC検出器から）
    tr: np.ndarray              # True Range
    atr: np.ndarray             # Average True Range (CATR絶対値)
    choppiness_index: np.ndarray # Choppiness Index（元の値）
    range_index: np.ndarray     # Range Index（チョピネスの反転）
    stddev_factor: np.ndarray   # 標準偏差係数
    chop_dc: np.ndarray         # チョピネス期間用ドミナントサイクル値
    stddev_dc: np.ndarray       # 標準偏差期間用ドミナントサイクル値
    lookback_dc: np.ndarray     # 標準偏差ルックバック期間用ドミナントサイクル値
    dynamic_threshold: np.ndarray  # 動的しきい値
    trend_state: np.ndarray     # トレンド状態 (1=トレンド、0=レンジ、NaN=不明)


@njit(fastmath=True)
def calculate_tr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    True Rangeを計算する（高速化版）

    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列

    Returns:
        True Range の配列
    """
    length = len(high)
    tr = np.zeros(length, dtype=np.float64)

    if length > 0:
        tr[0] = high[0] - low[0]

    for i in range(1, length):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)

    return tr


@njit(fastmath=True, parallel=True)
def calculate_choppiness_index(high: np.ndarray, low: np.ndarray, close: np.ndarray, chop_period: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """
    ドミナントサイクル期間によるチョピネス指数を計算する

    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        chop_period: ドミナントサイクルから決定されたチョピネス期間の配列
        tr: True Rangeの配列

    Returns:
        チョピネス指数の配列（0-100の範囲）
    """
    length = len(high)
    chop = np.full(length, np.nan, dtype=np.float64) # 初期値はNaN

    for i in prange(length):
        curr_period = int(chop_period[i])
        if curr_period < 2 or np.isnan(curr_period):
            curr_period = 2 # 最小期間を保証

        if i < curr_period - 1:
            continue

        # True Range の合計
        tr_sum = np.sum(tr[i - curr_period + 1 : i + 1])

        # 期間内の最高値と最安値を取得
        idx_start = i - curr_period + 1
        period_high = np.max(high[idx_start : i + 1])
        period_low = np.min(low[idx_start : i + 1])

        # 価格レンジの計算
        price_range = period_high - period_low

        # チョピネス指数の計算
        if price_range > 1e-9 and curr_period > 1 and tr_sum > 1e-9: # ゼロ除算と無意味な計算を避ける
            log_period = np.log10(float(curr_period))
            if log_period > 1e-9: # ゼロ除算を避ける
                chop_val = 100.0 * np.log10(tr_sum / price_range) / log_period
                # 値を0-100の範囲に制限
                chop[i] = max(0.0, min(100.0, chop_val))
            else:
                chop[i] = 0.0 # 期間1の場合はチョピネス0
        else:
            chop[i] = 0.0

    return chop


@njit(fastmath=True, parallel=True)
def calculate_stddev_factor(atr: np.ndarray, stddev_period: np.ndarray, lookback_period: np.ndarray) -> np.ndarray:
    """
    ATRの標準偏差係数を計算する（DC期間を使用）

    Args:
        atr: ATR配列
        stddev_period: 標準偏差の計算に使用するドミナントサイクル期間
        lookback_period: 最小標準偏差のルックバックに使用するドミナントサイクル期間

    Returns:
        標準偏差係数
    """
    n = len(atr)
    stddev = np.full(n, np.nan, dtype=np.float64)
    lowest_stddev = np.full(n, np.nan, dtype=np.float64)
    stddev_factor = np.full(n, np.nan, dtype=np.float64)

    for i in prange(n):
        current_stddev_period = int(stddev_period[i])
        if current_stddev_period <= 1 or np.isnan(current_stddev_period):
            current_stddev_period = 2

        current_lookback_period = int(lookback_period[i])
        if current_lookback_period <= 0 or np.isnan(current_lookback_period):
             current_lookback_period = 1

        if i >= current_stddev_period - 1:
            start_idx = max(0, i - current_stddev_period + 1)
            atr_window = atr[start_idx : i + 1]

            if len(atr_window) > 1:
                mean_sq = np.mean(np.power(atr_window, 2))
                sq_mean = np.power(np.mean(atr_window), 2)
                curr_stddev = np.sqrt(max(0.0, mean_sq - sq_mean))
                stddev[i] = curr_stddev

                # 最小標準偏差の計算（ルックバック期間内で）
                lookback_start_idx = max(0, i - current_lookback_period + 1)
                valid_stddevs = stddev[lookback_start_idx : i + 1]
                # NaNを除外して最小値を計算
                valid_stddevs_no_nan = valid_stddevs[~np.isnan(valid_stddevs)]
                if len(valid_stddevs_no_nan) > 0:
                   lowest_stddev[i] = np.min(valid_stddevs_no_nan)
                else:
                   lowest_stddev[i] = curr_stddev # 有効な過去の標準偏差がない場合は現在の値

                # 標準偏差係数の計算
                if not np.isnan(stddev[i]) and stddev[i] > 1e-9 and not np.isnan(lowest_stddev[i]):
                    stddev_factor[i] = lowest_stddev[i] / stddev[i]
                elif not np.isnan(stddev[i]) and stddev[i] <= 1e-9:
                     stddev_factor[i] = 1.0 # 標準偏差がほぼゼロならファクターは1
                else:
                    stddev_factor[i] = np.nan # 計算できない場合

            else: # 期間が短すぎる場合
                stddev[i] = 0.0
                lowest_stddev[i] = 0.0
                stddev_factor[i] = 1.0

        #elif i > 0: # 十分なデータがない場合、前の値を引き継ぐ（NaN伝播を防ぐためコメントアウト）
        #    stddev[i] = stddev[i-1]
        #    lowest_stddev[i] = lowest_stddev[i-1]
        #    stddev_factor[i] = stddev_factor[i-1]

    return stddev_factor


@njit(fastmath=True)
def calculate_c_trend_index(
    chop: np.ndarray,
    stddev_factor: np.ndarray
) -> np.ndarray:
    """
    Cトレンドインデックスを計算する

    Args:
        chop: チョピネス指数の配列（0-100の範囲）
        stddev_factor: 標準偏差ファクターの配列

    Returns:
        Cトレンドインデックスの配列（0-1の範囲、1に近いほど強いトレンド）
    """
    # チョピネス指数と標準偏差係数を組み合わせたレンジインデックスを計算
    # NaNを適切に処理
    range_index = np.full_like(chop, np.nan, dtype=np.float64)
    valid_mask = ~np.isnan(chop) & ~np.isnan(stddev_factor)
    range_index[valid_mask] = chop[valid_mask] * stddev_factor[valid_mask]

    # トレンド指数として常に反転し、0-1に正規化
    trend_index = 1.0 - (range_index / 100.0)

    return trend_index


@njit(fastmath=True)
def calculate_c_trend_index_batch(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray, # CATRの絶対値
    chop_period: np.ndarray,
    stddev_period: np.ndarray,
    lookback_period: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cトレンドインデックスを一括計算する（DC期間ベース）

    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        atr: CATRの絶対値の配列
        chop_period: チョピネス期間（DC検出器から）
        stddev_period: 標準偏差期間（DC検出器から）
        lookback_period: 標準偏差ルックバック期間（DC検出器から）

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            (Cトレンドインデックス, チョピネス指数, 標準偏差係数)
    """
    # True Rangeの計算
    tr = calculate_tr(high, low, close)

    # チョピネスインデックスの計算
    chop_index = calculate_choppiness_index(high, low, close, chop_period, tr)

    # 標準偏差係数の計算
    stddev_factor = calculate_stddev_factor(atr, stddev_period, lookback_period)

    # トレンドインデックスの計算
    trend_index = calculate_c_trend_index(chop_index, stddev_factor)

    return trend_index, chop_index, stddev_factor


@njit(fastmath=True)
def calculate_dynamic_threshold(
    er: np.ndarray,
    max_threshold: float,
    min_threshold: float
) -> np.ndarray:
    """
    効率比に基づいて動的なしきい値を計算する（高速化版）

    Args:
        er: 効率比の配列
        max_threshold: しきい値の最大値
        min_threshold: しきい値の最小値

    Returns:
        動的なしきい値の配列
    """
    length = len(er)
    threshold = np.full(length, np.nan, dtype=np.float64)

    for i in range(length):
        if np.isnan(er[i]):
            continue

        # ERの絶対値を使用
        er_abs = abs(er[i])

        # ERが高いほど（トレンドが強いほど）しきい値は高く
        # ERが低いほど（レンジ相場ほど）しきい値は低く
        threshold[i] = min_threshold + er_abs * (max_threshold - min_threshold)

    return threshold


class CTrendIndex(Indicator):
    """
    Cトレンドインデックス（Cycle Trend Index）インジケーター

    サイクル効率比（CER）とドミナントサイクル検出器を使用して、CATR（Cycle ATR）、
    チョピネスインデックス、標準偏差などを計算し、トレンド/レンジ状態を判定する指標です。
    ZTrendIndexを改良し、ATRにCATRを使用し、各期間（チョピネス、標準偏差、ルックバック）の
    決定をそれぞれ独立したドミナントサイクル検出器で行います。

    特徴:
    - サイクル効率比（CER）による適応的な計算
    - CATRを使用したサイクル適応型ボラティリティ測定
    - ドミナントサイクル検出器を使用してチョピネス、標準偏差、ルックバック期間を動的に決定
    - チョピネスインデックスと標準偏差係数を組み合わせて正規化したトレンド指標 (0-1)
    - 動的しきい値によるトレンド/レンジ状態の判定

    Args:
        # 各DC検出器共通パラメータ (デフォルト値は DEFAULT_* 定数を使用)
        chop_detector_type (str): チョピネス期間用DCタイプ
        chop_cycle_part (float): チョピネス期間用サイクル部分
        chop_lp_period (int): チョピネス期間用LPフィルター期間
        chop_hp_period (int): チョピネス期間用HPフィルター期間
        chop_max_cycle (int): チョピネス期間用最大サイクル
        chop_min_cycle (int): チョピネス期間用最小サイクル
        chop_max_output (int): チョピネス期間用最大出力
        chop_min_output (int): チョピネス期間用最小出力

        stddev_detector_type (str): 標準偏差期間用DCタイプ
        stddev_cycle_part (float): 標準偏差期間用サイクル部分
        stddev_lp_period (int): 標準偏差期間用LPフィルター期間
        stddev_hp_period (int): 標準偏差期間用HPフィルター期間
        stddev_max_cycle (int): 標準偏差期間用最大サイクル
        stddev_min_cycle (int): 標準偏差期間用最小サイクル
        stddev_max_output (int): 標準偏差期間用最大出力
        stddev_min_output (int): 標準偏差期間用最小出力

        lookback_detector_type (str): ルックバック期間用DCタイプ
        lookback_cycle_part (float): ルックバック期間用サイクル部分
        lookback_lp_period (int): ルックバック期間用LPフィルター期間
        lookback_hp_period (int): ルックバック期間用HPフィルター期間
        lookback_max_cycle (int): ルックバック期間用最大サイクル
        lookback_min_cycle (int): ルックバック期間用最小サイクル
        lookback_max_output (int): ルックバック期間用最大出力
        lookback_min_output (int): ルックバック期間用最小出力

        # CATR パラメータ
        catr_detector_type (str): CATR用DCタイプ
        catr_cycle_part (float): CATR用サイクル部分
        catr_lp_period (int): CATR用LPフィルター期間
        catr_hp_period (int): CATR用HPフィルター期間
        catr_max_cycle (int): CATR用最大サイクル
        catr_min_cycle (int): CATR用最小サイクル
        catr_max_output (int): CATR用最大出力
        catr_min_output (int): CATR用最小出力
        catr_smoother_type (str): CATR用平滑化タイプ

        # CER パラメータ
        cer_cycle_detector_type (str): CER用サイクル検出器タイプ
        cer_lp_period (int): CER用LPフィルター期間
        cer_hp_period (int): CER用HPフィルター期間
        cer_cycle_part (float): CER用サイクル部分

        # その他
        max_threshold (float): 動的しきい値の最大値
        min_threshold (float): 動的しきい値の最小値
        dc_detector_class (type): 使用するDC検出器クラス
    """

    def __init__(
        self,
        # チョピネス期間用DCパラメータ
        chop_detector_type: str = 'dudi_e',
        chop_cycle_part: float = 0.5,
        chop_lp_period: int = 5,
        chop_hp_period: int = 144,
        chop_max_cycle: int = 89,
        chop_min_cycle: int = 5,
        chop_max_output: int = 55,
        chop_min_output: int = 5,

        # 標準偏差期間用DCパラメータ
        stddev_detector_type: str = 'dudi_e',
        stddev_cycle_part: float = 0.5,
        stddev_lp_period: int = 5,
        stddev_hp_period: int = 144,
        stddev_max_cycle: int = 144,
        stddev_min_cycle: int = 10,
        stddev_max_output: int = 21,
        stddev_min_output: int = 8,

        # ルックバック期間用DCパラメータ
        lookback_detector_type: str = 'dudi_e',
        lookback_cycle_part: float = 0.5,
        lookback_lp_period: int = 5,
        lookback_hp_period: int = 89,
        lookback_max_cycle: int = 55,
        lookback_min_cycle: int = 10,
        lookback_max_output: int = 15,
        lookback_min_output: int = 8,

        # CATR パラメータ
        catr_detector_type: str = 'phac_e',
        catr_cycle_part: float = 0.5,
        catr_lp_period: int = 5,
        catr_hp_period: int = 144,
        catr_max_cycle: int = 89,
        catr_min_cycle: int = 5,
        catr_max_output: int = 55,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'alma',

        # CER パラメータ
        cer_cycle_detector_type: str = 'hody_e',
        cer_lp_period: int = 5,
        cer_hp_period: int = 144,
        cer_cycle_part: float = 0.5,

        # その他
        max_threshold: float = 0.75,
        min_threshold: float = 0.55,
        dc_detector_class: type = EhlersUnifiedDC
    ):
        # インジケーター名の生成（主要な設定をいくつか含める）
        name = (f"CTrendIndex(chop={chop_detector_type}_{chop_max_output},"
                f"std={stddev_detector_type}_{stddev_max_output},"
                f"catr={catr_detector_type}_{catr_max_output})")
        super().__init__(name)

        # パラメータをインスタンス変数として保存
        self.chop_detector_type = chop_detector_type
        self.chop_cycle_part = chop_cycle_part
        self.chop_lp_period = chop_lp_period
        self.chop_hp_period = chop_hp_period
        self.chop_max_cycle = chop_max_cycle
        self.chop_min_cycle = chop_min_cycle
        self.chop_max_output = chop_max_output
        self.chop_min_output = chop_min_output

        self.stddev_detector_type = stddev_detector_type
        self.stddev_cycle_part = stddev_cycle_part
        self.stddev_lp_period = stddev_lp_period
        self.stddev_hp_period = stddev_hp_period
        self.stddev_max_cycle = stddev_max_cycle
        self.stddev_min_cycle = stddev_min_cycle
        self.stddev_max_output = stddev_max_output
        self.stddev_min_output = stddev_min_output

        self.lookback_detector_type = lookback_detector_type
        self.lookback_cycle_part = lookback_cycle_part
        self.lookback_lp_period = lookback_lp_period
        self.lookback_hp_period = lookback_hp_period
        self.lookback_max_cycle = lookback_max_cycle
        self.lookback_min_cycle = lookback_min_cycle
        self.lookback_max_output = lookback_max_output
        self.lookback_min_output = lookback_min_output

        self.catr_detector_type = catr_detector_type
        self.catr_cycle_part = catr_cycle_part
        self.catr_lp_period = catr_lp_period
        self.catr_hp_period = catr_hp_period
        self.catr_max_cycle = catr_max_cycle
        self.catr_min_cycle = catr_min_cycle
        self.catr_max_output = catr_max_output
        self.catr_min_output = catr_min_output
        self.catr_smoother_type = catr_smoother_type

        self.cer_cycle_detector_type = cer_cycle_detector_type
        self.cer_lp_period = cer_lp_period
        self.cer_hp_period = cer_hp_period
        self.cer_cycle_part = cer_cycle_part

        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.dc_detector_class = dc_detector_class

        # DC検出器の初期化 (個別の引数を使用)
        self.chop_dc_detector = self.dc_detector_class(
            detector_type=self.chop_detector_type, cycle_part=self.chop_cycle_part,
            lp_period=self.chop_lp_period, hp_period=self.chop_hp_period,
            max_cycle=self.chop_max_cycle, min_cycle=self.chop_min_cycle,
            max_output=self.chop_max_output, min_output=self.chop_min_output
        )
        self.stddev_dc_detector = self.dc_detector_class(
            detector_type=self.stddev_detector_type, cycle_part=self.stddev_cycle_part,
            lp_period=self.stddev_lp_period, hp_period=self.stddev_hp_period,
            max_cycle=self.stddev_max_cycle, min_cycle=self.stddev_min_cycle,
            max_output=self.stddev_max_output, min_output=self.stddev_min_output
        )
        self.lookback_dc_detector = self.dc_detector_class(
            detector_type=self.lookback_detector_type, cycle_part=self.lookback_cycle_part,
            lp_period=self.lookback_lp_period, hp_period=self.lookback_hp_period,
            max_cycle=self.lookback_max_cycle, min_cycle=self.lookback_min_cycle,
            max_output=self.lookback_max_output, min_output=self.lookback_min_output
        )

        # CycleEfficiencyRatio の初期化 (個別の引数を使用)
        self.cycle_er = CycleEfficiencyRatio(
            cycle_detector_type=self.cer_cycle_detector_type, lp_period=self.cer_lp_period,
            hp_period=self.cer_hp_period, cycle_part=self.cer_cycle_part
        )

        # CATR の初期化 (個別の引数を使用)
        self.c_atr_indicator = CATR(
            detector_type=self.catr_detector_type, cycle_part=self.catr_cycle_part,
            lp_period=self.catr_lp_period, hp_period=self.catr_hp_period,
            max_cycle=self.catr_max_cycle, min_cycle=self.catr_min_cycle,
            max_output=self.catr_max_output, min_output=self.catr_min_output,
            smoother_type=self.catr_smoother_type
        )

        self._result = None
        self._data_hash = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            cols = ['open', 'high', 'low', 'close']
            # 存在するカラムのみを対象にする
            data_tuple = tuple(tuple(data[col].values) for col in cols if col in data.columns)
        elif isinstance(data, np.ndarray):
             # NumPy配列の形状と最初の数要素、最後の数要素でハッシュを代表させる（大規模データ対策）
             # より堅牢にするには、完全なバイトハッシュが必要な場合もある
             head = tuple(data[:5].flat) if len(data) > 0 else ()
             tail = tuple(data[-5:].flat) if len(data) > 5 else ()
             data_tuple = (data.shape, head, tail)
        else:
             # サポート外のデータ型
             raise TypeError("Unsupported data type for hashing.")


        # パラメータ文字列の生成 (重要なパラメータのみ)
        param_str = (
            f"{self.chop_detector_type}_{self.chop_cycle_part}_{self.chop_lp_period}_{self.chop_hp_period}_"
            f"{self.chop_max_cycle}_{self.chop_min_cycle}_{self.chop_max_output}_{self.chop_min_output}_"
            f"{self.stddev_detector_type}_{self.stddev_cycle_part}_{self.stddev_lp_period}_{self.stddev_hp_period}_"
            f"{self.stddev_max_cycle}_{self.stddev_min_cycle}_{self.stddev_max_output}_{self.stddev_min_output}_"
            f"{self.lookback_detector_type}_{self.lookback_cycle_part}_{self.lookback_lp_period}_{self.lookback_hp_period}_"
            f"{self.lookback_max_cycle}_{self.lookback_min_cycle}_{self.lookback_max_output}_{self.lookback_min_output}_"
            f"{self.catr_detector_type}_{self.catr_cycle_part}_{self.catr_lp_period}_{self.catr_hp_period}_"
            f"{self.catr_max_cycle}_{self.catr_min_cycle}_{self.catr_max_output}_{self.catr_min_output}_{self.catr_smoother_type}_"
            f"{self.cer_cycle_detector_type}_{self.cer_lp_period}_{self.cer_hp_period}_{self.cer_cycle_part}_"
            f"{self.max_threshold}_{self.min_threshold}_{self.dc_detector_class.__name__}"
        )
        # データとパラメータのハッシュを組み合わせる
        return f"{hash(data_tuple)}_{hash(param_str)}"

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> CTrendIndexResult:
        """
        Cトレンドインデックスを計算する

        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合は'open', 'high', 'low', 'close'カラムが必要

        Returns:
            CTrendIndexResult: 計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result

            self._data_hash = data_hash

            # データ検証とNumPy配列への変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
                h = np.asarray(data['high'].values, dtype=np.float64)
                l = np.asarray(data['low'].values, dtype=np.float64)
                c = np.asarray(data['close'].values, dtype=np.float64)
                # open は DC計算で必要になる場合がある
                o = np.asarray(data.get('open', c), dtype=np.float64)
                input_df_for_dc = data # DC計算用にDataFrameを保持
            elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 4:
                o = np.asarray(data[:, 0], dtype=np.float64)
                h = np.asarray(data[:, 1], dtype=np.float64)
                l = np.asarray(data[:, 2], dtype=np.float64)
                c = np.asarray(data[:, 3], dtype=np.float64)
                # DC計算用にDataFrameを作成
                input_df_for_dc = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c})
            else:
                raise ValueError("入力データはOHLCカラムを持つDataFrameまたは形状(N, >=4)のNumPy配列である必要があります")


            # 1. サイクル効率比(CER)を計算
            er = self.cycle_er.calculate(input_df_for_dc)

            # 2. CATRを計算 (CERが必要)
            # CATRは内部でDCを計算するが、ここでは外部CERを使用
            self.c_atr_indicator.calculate(input_df_for_dc, external_er=er)
            atr_absolute = self.c_atr_indicator.get_absolute_atr() # 絶対値ATRを取得
            atr_period_from_catr = self.c_atr_indicator.get_atr_period() # CATRが使った期間

            # 3. 各期間用のドミナントサイクルを計算
            chop_dc_values = self.chop_dc_detector.calculate(input_df_for_dc)
            stddev_dc_values = self.stddev_dc_detector.calculate(input_df_for_dc)
            lookback_dc_values = self.lookback_dc_detector.calculate(input_df_for_dc)

            # 4. Cトレンドインデックスのコア計算（バッチ処理）
            trend_index, chop_index, stddev_factor = calculate_c_trend_index_batch(
                h, l, c, atr_absolute,
                chop_dc_values,
                stddev_dc_values,
                lookback_dc_values
            )

            # 5. True Rangeの計算 (結果オブジェクト用)
            tr = calculate_tr(h, l, c)

            # 6. 動的しきい値の計算
            dynamic_threshold = calculate_dynamic_threshold(
                er, self.max_threshold, self.min_threshold
            )

            # 7. トレンド状態の計算
            length = len(trend_index)
            trend_state = np.full(length, np.nan) # デフォルトはNaN (不明)
            valid_indices = ~np.isnan(trend_index) & ~np.isnan(dynamic_threshold)
            trend_state[valid_indices & (trend_index >= dynamic_threshold)] = 1.0 # トレンド
            trend_state[valid_indices & (trend_index < dynamic_threshold)] = 0.0  # レンジ

            # 8. 結果オブジェクトを作成
            self._result = CTrendIndexResult(
                values=trend_index,
                er=er,
                chop_period=chop_dc_values,
                atr_period=atr_period_from_catr, # CATRが内部で使った期間
                stddev_period=stddev_dc_values,
                lookback_period=lookback_dc_values,
                tr=tr,
                atr=atr_absolute,
                choppiness_index=chop_index,
                range_index=100.0 - chop_index, # レンジインデックスはチョピネスの反転 (0-100) -> 要確認：Zでは 1-trend_index/(100?)
                stddev_factor=stddev_factor,
                chop_dc=chop_dc_values,
                stddev_dc=stddev_dc_values,
                lookback_dc=lookback_dc_values,
                dynamic_threshold=dynamic_threshold,
                trend_state=trend_state
            )

            self._values = trend_index # Indicator基底クラス用

            return self._result

        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"CTrendIndex計算中にエラー: {error_msg}
{stack_trace}")
            # エラー発生時は空の結果または前回成功時の結果を返す（ここでは空を返す）
            # 以前の値を返す場合は self._result if self._result else CTrendIndexResult(...) のような処理
            # 空の numpy 配列で初期化された結果を返す
            dummy_len = len(data) if hasattr(data, '__len__') else 0
            nan_array = np.full(dummy_len, np.nan)
            return CTrendIndexResult(
                values=np.copy(nan_array), er=np.copy(nan_array), chop_period=np.copy(nan_array),
                atr_period=np.copy(nan_array), stddev_period=np.copy(nan_array), lookback_period=np.copy(nan_array),
                tr=np.copy(nan_array), atr=np.copy(nan_array), choppiness_index=np.copy(nan_array),
                range_index=np.copy(nan_array), stddev_factor=np.copy(nan_array), chop_dc=np.copy(nan_array),
                stddev_dc=np.copy(nan_array), lookback_dc=np.copy(nan_array), dynamic_threshold=np.copy(nan_array),
                trend_state=np.copy(nan_array)
            )

    # --- Getterメソッド ---
    def get_result(self) -> Optional[CTrendIndexResult]:
        """計算結果全体を返す"""
        return self._result

    def get_stddev_factor(self) -> np.ndarray:
        """標準偏差係数を取得"""
        return self._result.stddev_factor if self._result else np.array([])

    def get_choppiness_index(self) -> np.ndarray:
        """チョピネス指数を取得"""
        return self._result.choppiness_index if self._result else np.array([])

    def get_range_index(self) -> np.ndarray:
        """レンジインデックスを取得"""
        # 注意: ZTrendIndexでは 1.0 - (chop * factor / 100.0) だったが、
        # CTrendIndexResultでは 100.0 - chop と定義。整合性を確認・修正必要。
        # ここではResultの定義に従う。
        return self._result.range_index if self._result else np.array([])

    def get_chop_dc(self) -> np.ndarray:
        """チョピネス期間用ドミナントサイクル値を取得"""
        return self._result.chop_dc if self._result else np.array([])

    def get_stddev_dc(self) -> np.ndarray:
        """標準偏差期間用ドミナントサイクル値を取得"""
        return self._result.stddev_dc if self._result else np.array([])

    def get_lookback_dc(self) -> np.ndarray:
        """標準偏差ルックバック期間用ドミナントサイクル値を取得"""
        return self._result.lookback_dc if self._result else np.array([])

    def get_dynamic_threshold(self) -> np.ndarray:
        """動的しきい値を取得"""
        return self._result.dynamic_threshold if self._result else np.array([])

    def get_trend_state(self) -> np.ndarray:
        """トレンド状態を取得（1=トレンド, 0=レンジ, NaN=不明）"""
        return self._result.trend_state if self._result else np.array([])

    def get_atr_period(self) -> np.ndarray:
        """CATRが使用したATR期間を取得"""
        return self._result.atr_period if self._result else np.array([])

    def get_absolute_atr(self) -> np.ndarray:
        """CATRの絶対値（金額ベース）を取得"""
        return self._result.atr if self._result else np.array([])

    def get_efficiency_ratio(self) -> np.ndarray:
        """サイクル効率比(CER)を取得"""
        return self._result.er if self._result else np.array([])

    def reset(self) -> None:
        """インジケーターの状態をリセットする"""
        super().reset()
        self.cycle_er.reset()
        self.c_atr_indicator.reset()
        self.chop_dc_detector.reset()
        self.stddev_dc_detector.reset()
        self.lookback_dc_detector.reset()
        self._result = None
        self._data_hash = None 