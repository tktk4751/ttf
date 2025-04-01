#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from .indicator import Indicator
from .hyper_smoother import hyper_smoother
from .cycle_efficiency_ratio import CycleEfficiencyRatio
from .z_atr import ZATR
from .ehlers_hody_dc import EhlersHoDyDC


@dataclass
class ZTrendIndexResult:
    """Zトレンドインデックスの計算結果"""
    values: np.ndarray          # Zトレンドインデックスの値（0-1の範囲）
    er: np.ndarray              # サイクル効率比（CER）
    dynamic_chop_period: np.ndarray  # 動的チョピネス期間
    dynamic_atr_period: np.ndarray   # 動的ATR期間
    dynamic_stddev_period: np.ndarray # 動的標準偏差期間
    tr: np.ndarray              # True Range
    atr: np.ndarray             # Average True Range (ZATR)
    choppiness_index: np.ndarray # Choppiness Index（元の値）
    range_index: np.ndarray     # Range Index（元の値）
    stddev_factor: np.ndarray   # 標準偏差係数
    max_chop_dc: np.ndarray     # 最大チョピネス期間用ドミナントサイクル値
    min_chop_dc: np.ndarray     # 最小チョピネス期間用ドミナントサイクル値
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
    
    # 最初の要素は単純なレンジ
    tr[0] = high[0] - low[0]
    
    # 2番目以降の要素はTRを計算
    for i in range(1, length):
        tr1 = high[i] - low[i]  # 当日の高値 - 当日の安値
        tr2 = abs(high[i] - close[i-1])  # |当日の高値 - 前日の終値|
        tr3 = abs(low[i] - close[i-1])  # |当日の安値 - 前日の終値|
        tr[i] = max(tr1, tr2, tr3)
    
    return tr


@njit(fastmath=True)
def calculate_dynamic_chop_period(er: np.ndarray, max_periods: np.ndarray, min_periods: np.ndarray) -> np.ndarray:
    """
    効率比とドミナントサイクルに基づいて動的なチョピネス期間を計算する
    
    Args:
        er: 効率比の配列
        max_periods: 最大期間の配列（ドミナントサイクルから計算）
        min_periods: 最小期間の配列（ドミナントサイクルから計算）
    
    Returns:
        動的なチョピネス期間の配列
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    # 反転した効率比（1-ER）を使う
    inv_er = 1.0 - np.abs(er)
    periods = min_periods + inv_er * (max_periods - min_periods)
    return np.round(periods).astype(np.int32)


@njit(fastmath=True, parallel=True)
def calculate_choppiness_index(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """
    動的期間によるチョピネス指数を計算する
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period: 動的な期間の配列
        tr: True Rangeの配列
    
    Returns:
        チョピネス指数の配列（0-100の範囲）
    """
    length = len(high)
    chop = np.zeros(length, dtype=np.float64)
    
    for i in prange(1, length):
        curr_period = int(period[i])
        if curr_period < 2:
            curr_period = 2
        
        if i < curr_period:
            continue
        
        # True Range の合計
        tr_sum = np.sum(tr[i - curr_period + 1:i + 1])
        
        # 期間内の最高値と最安値を取得
        idx_start = i - curr_period + 1
        period_high = np.max(high[idx_start:i + 1])
        period_low = np.min(low[idx_start:i + 1])
        
        # 価格レンジの計算
        price_range = period_high - period_low
        
        # チョピネス指数の計算
        if price_range > 0 and curr_period > 1 and tr_sum > 0:
            log_period = np.log10(float(curr_period))
            chop_val = 100.0 * np.log10(tr_sum / price_range) / log_period
            # 値を0-100の範囲に制限
            chop[i] = max(0.0, min(100.0, chop_val))
        else:
            chop[i] = 0.0
    
    return chop


@njit(fastmath=True, parallel=True)
def calculate_stddev_factor(atr: np.ndarray, period: np.ndarray, lookback_period: np.ndarray) -> np.ndarray:
    """
    ATRの標準偏差係数を計算する
    
    Args:
        atr: ATR配列
        period: 期間配列（各バーの期間）
        lookback_period: 標準偏差の計算に使用する動的ルックバック期間
        
    Returns:
        標準偏差係数
    """
    n = len(atr)
    stddev = np.zeros(n, dtype=np.float64)
    lowest_stddev = np.full(n, np.inf, dtype=np.float64)  # 最小値を追跡するための配列
    stddev_factor = np.ones(n, dtype=np.float64)  # デフォルトは1
    
    for i in prange(n):
        # 標準偏差の計算期間を決定
        current_period = int(period[i])
        if current_period <= 1:
            current_period = 2
            
        # ルックバック期間を決定
        current_lookback = int(lookback_period[i])
        if current_lookback <= 0:
            current_lookback = 1
        
        if i >= current_period - 1:
            # 利用可能なデータの取得
            start_idx = max(0, i - current_period + 1)
            atr_window = atr[start_idx:i+1]
            
            # PineScriptのSMAを使用した計算方法
            stddev_a = np.mean(np.power(atr_window, 2))
            stddev_b = np.power(np.sum(atr_window), 2) / np.power(len(atr_window), 2)
            curr_stddev = np.sqrt(max(0.0, stddev_a - stddev_b))
            
            stddev[i] = curr_stddev
            
            # 最小標準偏差の更新（ルックバック期間内で）
            if i == current_period - 1 or (i > 0 and curr_stddev < lowest_stddev[i-1]):
                lowest_stddev[i] = curr_stddev
            else:
                lowest_lookback_start = max(0, i - current_lookback + 1)
                if i > lowest_lookback_start:
                    lowest_in_period = np.min(stddev[lowest_lookback_start:i+1])
                    lowest_stddev[i] = lowest_in_period
                else:
                    lowest_stddev[i] = stddev[i]
            
            # 標準偏差係数の計算
            if stddev[i] > 0:
                stddev_factor[i] = lowest_stddev[i] / stddev[i]
            else:
                stddev_factor[i] = 1.0
        elif i > 0:
            # データ不足の場合は前の値を使用
            stddev[i] = stddev[i-1]
            lowest_stddev[i] = lowest_stddev[i-1]
            stddev_factor[i] = stddev_factor[i-1]
    
    return stddev_factor


@njit(fastmath=True)
def calculate_z_trend_index(
    chop: np.ndarray,
    stddev_factor: np.ndarray
) -> np.ndarray:
    """
    Zトレンドインデックスを計算する
    
    Args:
        chop: チョピネス指数の配列（0-100の範囲）
        stddev_factor: 標準偏差ファクターの配列
    
    Returns:
        Zトレンドインデックスの配列（0-1の範囲、1に近いほど強いトレンド）
    """
    # チョピネス指数と標準偏差係数を組み合わせたレンジインデックスを計算
    range_index = chop * stddev_factor
    
    # トレンド指数として常に反転し、0-1に正規化
    trend_index = 1.0 - (range_index / 100.0)
    
    return trend_index


@njit(fastmath=True)
def calculate_dynamic_period(er: np.ndarray, max_period: int, min_period: int) -> np.ndarray:
    """
    効率比に基づいて動的な期間を計算する（標準関数版）
    
    Args:
        er: 効率比の配列
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        動的な期間の配列
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    # 反転した効率比（1-ER）を使う
    inv_er = 1.0 - np.abs(er)
    periods = min_period + inv_er * (max_period - min_period)
    return np.round(periods).astype(np.int32)


@njit(fastmath=True)
def calculate_z_trend_index_batch(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray,
    er: np.ndarray, 
    max_chop_periods: np.ndarray, 
    min_chop_periods: np.ndarray,
    max_stddev_period: int, 
    min_stddev_period: int,
    max_lookback_period: int, 
    min_lookback_period: int,
    atr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Zトレンドインデックスを一括計算する

    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        er: 効率比の配列
        max_chop_periods: チョピネス期間の最大値（ドミナントサイクル）
        min_chop_periods: チョピネス期間の最小値（ドミナントサイクル）
        max_stddev_period: 標準偏差期間の最大値
        min_stddev_period: 標準偏差期間の最小値
        max_lookback_period: 標準偏差ルックバック期間の最大値
        min_lookback_period: 標準偏差ルックバック期間の最小値
        atr: 既に計算されたATR配列

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            (Zトレンドインデックス, 動的チョピネス期間, 動的標準偏差期間, チョピネス指数, 標準偏差係数)
    """
    # 動的チョピネス期間の計算（ドミナントサイクルベース）
    dynamic_chop_period = calculate_dynamic_chop_period(
        er, max_chop_periods, min_chop_periods
    )
    
    # 動的標準偏差期間の計算（標準関数版）
    dynamic_stddev_period = calculate_dynamic_period(
        er, max_stddev_period, min_stddev_period
    )
    
    # 動的ルックバック期間の計算（標準関数版）
    dynamic_lookback_period = calculate_dynamic_period(
        er, max_lookback_period, min_lookback_period
    )
    
    # True Rangeの計算
    tr = calculate_tr(high, low, close)
    
    # チョピネスインデックスの計算
    chop_index = calculate_choppiness_index(high, low, close, dynamic_chop_period, tr)
    
    # 標準偏差係数の計算
    stddev_factor = calculate_stddev_factor(atr, dynamic_stddev_period, dynamic_lookback_period)
    
    # トレンドインデックスの計算
    trend_index = calculate_z_trend_index(chop_index, stddev_factor)
    
    return trend_index, dynamic_chop_period, dynamic_stddev_period, chop_index, stddev_factor


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
    threshold = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(er[i]):
            threshold[i] = np.nan
            continue
        
        # ERの絶対値を使用
        er_abs = abs(er[i])
        
        # ERが高いほど（トレンドが強いほど）しきい値は高く
        # ERが低いほど（レンジ相場ほど）しきい値は低く
        threshold[i] = min_threshold + er_abs * (max_threshold - min_threshold)
    
    return threshold


class ZTrendIndex(Indicator):
    """
    Zトレンドインデックス（Z Trend Index）インジケーター
    
    サイクル効率比（CER）とドミナントサイクル検出器を使用してチョピネスインデックス、ZATR、
    標準偏差などを動的に調整する高度なトレンド/レンジ検出指標です。
    AlphaTrendIndexを改良し、チョピネス期間とATR期間の調整をドミナントサイクル検出器で行います。
    
    特徴:
    - サイクル効率比（CER）を使用して、現在のサイクルに基づいた適応的な計算
    - ドミナントサイクル検出器を使用してチョピネス期間とATR期間を動的に決定
    - ZATRを使用した高度なボラティリティ測定
    - チョピネスインデックスと標準偏差係数を組み合わせて正規化したトレンド指標
    - 市場状態に応じて各期間が自動調整される
    - 0-1の範囲で表示（1に近いほど強いトレンド、0に近いほど強いレンジ）
    - 動的しきい値によるトレンド/レンジ状態の判定
    
    CERとドミナントサイクル検出を組み合わせることで、より正確な市場サイクルの追跡が可能になります。
    """
    
    def __init__(
        self,
        # 最大チョピネス期間用ドミナントサイクル設定
        max_chop_dc_cycle_part: float = 0.5,
        max_chop_dc_max_cycle: int = 144,
        max_chop_dc_min_cycle: int = 10,
        max_chop_dc_max_output: int = 34,
        max_chop_dc_min_output: int = 13,
        
        # 最小チョピネス期間用ドミナントサイクル設定
        min_chop_dc_cycle_part: float = 0.25,
        min_chop_dc_max_cycle: int = 55,
        min_chop_dc_min_cycle: int = 5,
        min_chop_dc_max_output: int = 13,
        min_chop_dc_min_output: int = 5,
        
        # 標準偏差と標準偏差ルックバック期間の設定
        max_stddev_period: int = 21,
        min_stddev_period: int = 14,
        max_lookback_period: int = 14,
        min_lookback_period: int = 7,
        
        # サイクル効率比(CER)のパラメーター
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        
        # ZATR用パラメータ（既存のZATRクラスをそのまま使用）
        smoother_type: str = 'alma',  # 'alma'または'hyper'

        # 動的しきい値のパラメータ
        max_threshold: float = 0.75,
        min_threshold: float = 0.55
    ):
        """
        コンストラクタ
        
        Args:
            max_chop_dc_cycle_part: 最大チョピネス期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            max_chop_dc_max_cycle: 最大チョピネス期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 144）
            max_chop_dc_min_cycle: 最大チョピネス期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 10）
            max_chop_dc_max_output: 最大チョピネス期間用ドミナントサイクル計算用の最大出力値（デフォルト: 34）
            max_chop_dc_min_output: 最大チョピネス期間用ドミナントサイクル計算用の最小出力値（デフォルト: 13）
            
            min_chop_dc_cycle_part: 最小チョピネス期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.25）
            min_chop_dc_max_cycle: 最小チョピネス期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            min_chop_dc_min_cycle: 最小チョピネス期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            min_chop_dc_max_output: 最小チョピネス期間用ドミナントサイクル計算用の最大出力値（デフォルト: 13）
            min_chop_dc_min_output: 最小チョピネス期間用ドミナントサイクル計算用の最小出力値（デフォルト: 5）
            
            max_stddev_period: 標準偏差期間の最大値（デフォルト: 21）
            min_stddev_period: 標準偏差期間の最小値（デフォルト: 14）
            max_lookback_period: 標準偏差の最小値を探す期間の最大値（デフォルト: 14）
            min_lookback_period: 標準偏差の最小値を探す期間の最小値（デフォルト: 7）
            
            cycle_detector_type: サイクル検出器の種類（デフォルト: 'hody_dc'）
                'dudi_dc' - 二重微分
                'hody_dc' - ホモダイン判別機
                'phac_dc' - 位相累積
                'dudi_dce' - 拡張二重微分
                'hody_dce' - 拡張ホモダイン判別機
                'phac_dce' - 拡張位相累積
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            smoother_type: 平滑化アルゴリズムのタイプ（'alma'または'hyper'）
            max_threshold: しきい値の最大値（デフォルト: 0.75）
            min_threshold: しきい値の最小値（デフォルト: 0.55）
        """
        super().__init__(
            f"ZTrendIndex({cycle_detector_type}, {max_chop_dc_max_output}, {min_chop_dc_max_output})"
        )
        
        # 最大チョピネス期間用ドミナントサイクル設定
        self.max_chop_dc_cycle_part = max_chop_dc_cycle_part
        self.max_chop_dc_max_cycle = max_chop_dc_max_cycle
        self.max_chop_dc_min_cycle = max_chop_dc_min_cycle
        self.max_chop_dc_max_output = max_chop_dc_max_output
        self.max_chop_dc_min_output = max_chop_dc_min_output
        
        # 最小チョピネス期間用ドミナントサイクル設定
        self.min_chop_dc_cycle_part = min_chop_dc_cycle_part
        self.min_chop_dc_max_cycle = min_chop_dc_max_cycle
        self.min_chop_dc_min_cycle = min_chop_dc_min_cycle
        self.min_chop_dc_max_output = min_chop_dc_max_output
        self.min_chop_dc_min_output = min_chop_dc_min_output
        
        # 標準偏差とルックバック期間
        self.max_stddev_period = max_stddev_period
        self.min_stddev_period = min_stddev_period
        self.max_lookback_period = max_lookback_period
        self.min_lookback_period = min_lookback_period
        
        # 平滑化タイプ
        self.smoother_type = smoother_type
        
        # サイクル効率比(CER)関連のパラメータ
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        
        # 動的しきい値のパラメータ
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        
        # 最大チョピネス期間用ドミナントサイクル検出器を初期化
        self.max_chop_dc_detector = EhlersHoDyDC(
            cycle_part=self.max_chop_dc_cycle_part,
            max_cycle=self.max_chop_dc_max_cycle,
            min_cycle=self.max_chop_dc_min_cycle,
            max_output=self.max_chop_dc_max_output,
            min_output=self.max_chop_dc_min_output,
            src_type='hlc3'
        )
        
        # 最小チョピネス期間用ドミナントサイクル検出器を初期化
        self.min_chop_dc_detector = EhlersHoDyDC(
            cycle_part=self.min_chop_dc_cycle_part,
            max_cycle=self.min_chop_dc_max_cycle,
            min_cycle=self.min_chop_dc_min_cycle,
            max_output=self.min_chop_dc_max_output,
            min_output=self.min_chop_dc_min_output,
            src_type='hlc3'
        )
        
        # サイクル効率比(CER)のインスタンス化
        self.cycle_er = CycleEfficiencyRatio(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part
        )
        
        # ZATRのインスタンス化
        self.z_atr_indicator = ZATR(
            smoother_type=smoother_type
        )
        
        self._result = None
        self._data_hash = None  # データキャッシュ用ハッシュ
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            cols = ['open', 'high', 'low', 'close']
            data_hash = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            # NumPy配列の場合は全体をハッシュする
            data_hash = hash(tuple(map(tuple, data)))
        
        # パラメータ値を含めることで、同じデータでもパラメータが異なる場合に再計算する
        param_str = (
            f"{self.max_chop_dc_cycle_part}_{self.max_chop_dc_max_cycle}_{self.max_chop_dc_min_cycle}_"
            f"{self.max_chop_dc_max_output}_{self.max_chop_dc_min_output}_"
            f"{self.min_chop_dc_cycle_part}_{self.min_chop_dc_max_cycle}_{self.min_chop_dc_min_cycle}_"
            f"{self.min_chop_dc_max_output}_{self.min_chop_dc_min_output}_"
            f"{self.max_stddev_period}_{self.min_stddev_period}_{self.max_lookback_period}_{self.min_lookback_period}_"
            f"{self.cycle_detector_type}_{self.lp_period}_{self.hp_period}_{self.cycle_part}_{self.smoother_type}_"
            f"{self.max_threshold}_{self.min_threshold}"
        )
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ZTrendIndexResult:
        """
        Zトレンドインデックスを計算する

        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合は'open', 'high', 'low', 'close'カラムが必要

        Returns:
            ZTrendIndexResult: 計算結果
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
                
                o = np.asarray(data['open'].values if 'open' in data.columns else np.zeros_like(data['close'].values), dtype=np.float64)
                h = np.asarray(data['high'].values, dtype=np.float64)
                l = np.asarray(data['low'].values, dtype=np.float64)
                c = np.asarray(data['close'].values, dtype=np.float64)
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    o = np.asarray(data[:, 0], dtype=np.float64)  # open
                    h = np.asarray(data[:, 1], dtype=np.float64)  # high
                    l = np.asarray(data[:, 2], dtype=np.float64)  # low
                    c = np.asarray(data[:, 3], dtype=np.float64)  # close
                else:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # 最大チョピネス期間用ドミナントサイクルの計算
            max_chop_dc_values = self.max_chop_dc_detector.calculate(data)
            
            # 最小チョピネス期間用ドミナントサイクルの計算
            min_chop_dc_values = self.min_chop_dc_detector.calculate(data)
            
            # サイクル効率比(CER)を計算
            df_data = pd.DataFrame({
                'open': o,
                'high': h,
                'low': l,
                'close': c
            })
            er = self.cycle_er.calculate(df_data)
            
            # 動的ATR期間の計算（ATRはZATRインスタンスに任せる）
            
            # ZATRの計算（サイクル効率比を使用）
            z_atr_values = self.z_atr_indicator.calculate(df_data, external_er=er)
            atr = self.z_atr_indicator.get_absolute_atr()  # 金額ベースの値を使用
            
            # 一括計算
            trend_index, dynamic_chop_period, dynamic_stddev_period, chop_index, stddev_factor = (
                calculate_z_trend_index_batch(
                    h, l, c, er, 
                    max_chop_dc_values, min_chop_dc_values,
                    self.max_stddev_period, self.min_stddev_period,
                    self.max_lookback_period, self.min_lookback_period,
                    atr
                )
            )
            
            # True Rangeの計算（ここでは結果の完全性のために保持）
            tr = calculate_tr(h, l, c)
            
            # 動的しきい値の計算
            dynamic_threshold = calculate_dynamic_threshold(
                er, self.max_threshold, self.min_threshold
            )
            
            # トレンド状態の計算（1=トレンド、0=レンジ、NaN=不明）
            length = len(trend_index)
            trend_state = np.full(length, np.nan)
            
            for i in range(length):
                if np.isnan(trend_index[i]) or np.isnan(dynamic_threshold[i]):
                    continue
                
                if trend_index[i] >= dynamic_threshold[i]:
                    trend_state[i] = 1.0  # トレンド
                else:
                    trend_state[i] = 0.0  # レンジ
            
            # 結果オブジェクトを作成
            result = ZTrendIndexResult(
                values=trend_index,
                er=er,
                dynamic_chop_period=dynamic_chop_period,
                dynamic_atr_period=self.z_atr_indicator.get_dynamic_period(),
                dynamic_stddev_period=dynamic_stddev_period,
                tr=tr,
                atr=atr,
                choppiness_index=chop_index,
                range_index=1.0 - chop_index,  # レンジインデックスはチョピネスの反転
                stddev_factor=stddev_factor,
                max_chop_dc=max_chop_dc_values,
                min_chop_dc=min_chop_dc_values,
                dynamic_threshold=dynamic_threshold,
                trend_state=trend_state
            )
            
            # 内部の結果変数に保存
            self._result = result
            self._values = trend_index
            
            return result
        
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ZTrendIndex計算中にエラー: {error_msg}\n{stack_trace}")
            if hasattr(self, '_values') and self._values is not None:
                return self._values
            return np.array([])
    
    def get_stddev_factor(self) -> np.ndarray:
        """
        標準偏差係数の値を取得する
        
        Returns:
            標準偏差係数の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.stddev_factor
    
    def get_choppiness_index(self) -> np.ndarray:
        """
        チョピネス指数の値を取得する
        
        Returns:
            チョピネス指数の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.choppiness_index
    
    def get_range_index(self) -> np.ndarray:
        """
        レンジインデックスの値を取得する
        
        Returns:
            レンジインデックスの配列
        """
        if self._result is None:
            return np.array([])
        return self._result.range_index
    
    def get_max_chop_dc(self) -> np.ndarray:
        """
        最大チョピネス期間用ドミナントサイクル値を取得する
        
        Returns:
            ドミナントサイクル値の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.max_chop_dc
    
    def get_min_chop_dc(self) -> np.ndarray:
        """
        最小チョピネス期間用ドミナントサイクル値を取得する
        
        Returns:
            ドミナントサイクル値の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.min_chop_dc
    
    def get_dynamic_threshold(self) -> np.ndarray:
        """
        動的しきい値を取得する
        
        Returns:
            動的しきい値の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.dynamic_threshold
    
    def get_trend_state(self) -> np.ndarray:
        """
        トレンド状態を取得する（1=トレンド、0=レンジ、NaN=不明）
        
        Returns:
            トレンド状態の配列
        """
        if self._result is None:
            return np.array([])
        return self._result.trend_state
    
    def reset(self) -> None:
        """
        インジケーターの状態をリセットする
        """
        super().reset()
        self.cycle_er.reset()
        self.z_atr_indicator.reset()
        self.max_chop_dc_detector.reset()
        self.min_chop_dc_detector.reset()
        self._result = None
        self._data_hash = None 