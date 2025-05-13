#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import jit, vectorize, njit, prange

from .indicator import Indicator
from .z_ma import ZMA
from .z_atr import ZATR
from .cycle_efficiency_ratio import CycleEfficiencyRatio


@dataclass
class ZChannelResult:
    """Zチャネルの計算結果"""
    middle: np.ndarray        # 中心線（ZMA）
    upper: np.ndarray         # 上限バンド
    lower: np.ndarray         # 下限バンド
    cer: np.ndarray           # Cycle Efficiency Ratio
    dynamic_multiplier: np.ndarray  # 動的ATR乗数
    z_atr: np.ndarray         # ZATR値
    max_mult_values: np.ndarray  # 動的に計算されたmax_multiplier値
    min_mult_values: np.ndarray  # 動的に計算されたmin_multiplier値


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


@njit(fastmath=True)
def calculate_dynamic_multiplier(cer: np.ndarray, max_mult: float, min_mult: float) -> np.ndarray:
    """
    サイクル効率比に基づいて動的なATR乗数を計算する（高速化版）
    
    Args:
        cer: サイクル効率比の配列
        max_mult: 最大乗数
        min_mult: 最小乗数
    
    Returns:
        動的な乗数の配列
    """
    # CERが高い（トレンドが強い）ほど乗数は小さく、
    # CERが低い（トレンドが弱い）ほど乗数は大きくなる
    cer_abs = np.abs(cer)
    multipliers = max_mult - cer_abs * (max_mult - min_mult)
    return multipliers


@njit(fastmath=True, parallel=True, cache=True)
def calculate_z_channel(
    z_ma: np.ndarray,
    z_atr: np.ndarray,
    dynamic_multiplier: np.ndarray,
    use_percent: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Zチャネルを計算する（パラレル高速化版）
    
    Args:
        z_ma: ZMA値の配列
        z_atr: ZATR値の配列（金額ベースまたはパーセントベース）
        dynamic_multiplier: 動的乗数の配列
        use_percent: パーセントベースATRを使用するかどうか
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (中心線, 上限バンド, 下限バンド)の配列
    """
    length = len(z_ma)
    middle = np.copy(z_ma)
    upper = np.full_like(z_ma, np.nan, dtype=np.float64)
    lower = np.full_like(z_ma, np.nan, dtype=np.float64)
    
    # 有効なデータ判別用マスク
    valid_mask = ~(np.isnan(z_ma) | np.isnan(z_atr) | np.isnan(dynamic_multiplier))
    
    # バンド幅計算用の一時配列 - パラレル計算の準備
    band_width = np.zeros_like(z_ma, dtype=np.float64)
    
    # 並列計算で一度にバンド幅を計算
    for i in prange(length):
        if valid_mask[i]:
            if use_percent:
                # パーセントベースのATRを使用する場合、価格に対する比率
                band_width[i] = z_ma[i] * z_atr[i] * dynamic_multiplier[i]
            else:
                # 金額ベースのATRを使用する場合
                band_width[i] = z_atr[i] * dynamic_multiplier[i]
    
    # 一度にバンドを計算（並列処理）
    for i in prange(length):
        if valid_mask[i]:
            upper[i] = middle[i] + band_width[i]
            lower[i] = middle[i] - band_width[i]
    
    return middle, upper, lower


class ZChannel(Indicator):
    """
    Zチャネル（Z Channel）インジケーター
    
    特徴:
    - 中心線にZMA（Z Moving Average）を使用
    - バンド幅の計算にZATR（Z Average True Range）を使用
    - ATR乗数がサイクル効率比（CER）に基づいて動的に調整
    - 最大乗数と最小乗数もCERに基づいて動的に調整
    - ドミナントサイクル検出器による高度な適応性
    
    市場状態に応じた最適な挙動:
    - トレンド強い（CER高い）:
      - 狭いバンド幅（小さい乗数）でトレンドをタイトに追従
    - トレンド弱い（CER低い）:
      - 広いバンド幅（大きい乗数）でレンジ相場の振れ幅を捉える
    """
    
    def __init__(
        self,
        # 基本パラメータ
        detector_type: str = 'phac_e',
        cer_detector_type: str = None,  # CER用の検出器タイプ
        lp_period: int = 5,
        hp_period: int = 55,
        cycle_part: float = 0.382,
        max_multiplier: float = 7.0,  # 固定乗数を使用する場合
        min_multiplier: float = 1.0,  # 固定乗数を使用する場合
        # 動的乗数の範囲パラメータ（固定乗数の代わりに動的乗数を使用する場合）
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 3.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        smoother_type: str = 'alma',  # 'alma' または 'hyper'
        src_type: str = 'hlc3',       # 'open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4'
        
        # CER用パラメータ
        cer_max_cycle: int = 62,       # CER用の最大サイクル期間
        cer_min_cycle: int = 5,         # CER用の最小サイクル期間
        cer_max_output: int = 34,       # CER用の最大出力値
        cer_min_output: int = 5,        # CER用の最小出力値
        
        # ZMA用パラメータ
        zma_max_dc_cycle_part: float = 0.5,     # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_cycle: int = 100,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_cycle: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_output: int = 120,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_output: int = 22,        # ZMA: 最大期間用ドミナントサイクル計算用
        
        zma_min_dc_cycle_part: float = 0.25,    # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_cycle: int = 55,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_cycle: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_output: int = 13,        # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_output: int = 3,         # ZMA: 最小期間用ドミナントサイクル計算用
        
        # ZMA動的Slow最大用パラメータ
        zma_slow_max_dc_cycle_part: float = 0.5,
        zma_slow_max_dc_max_cycle: int = 144,
        zma_slow_max_dc_min_cycle: int = 5,
        zma_slow_max_dc_max_output: int = 89,
        zma_slow_max_dc_min_output: int = 22,
        
        # ZMA動的Slow最小用パラメータ
        zma_slow_min_dc_cycle_part: float = 0.5,
        zma_slow_min_dc_max_cycle: int = 89,
        zma_slow_min_dc_min_cycle: int = 5,
        zma_slow_min_dc_max_output: int = 21,
        zma_slow_min_dc_min_output: int = 8,
        
        # ZMA動的Fast最大用パラメータ
        zma_fast_max_dc_cycle_part: float = 0.5,
        zma_fast_max_dc_max_cycle: int = 55,
        zma_fast_max_dc_min_cycle: int = 5,
        zma_fast_max_dc_max_output: int = 15,
        zma_fast_max_dc_min_output: int = 3,
        
        zma_min_fast_period: int = 2,           # ZMA: 速い移動平均の最小期間（常に2で固定）
        zma_hyper_smooth_period: int = 0,       # ZMA: ハイパースムーサーの平滑化期間
        
        # ZATR用パラメータ
        zatr_max_dc_cycle_part: float = 0.7,    # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_cycle: int = 77,        # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_cycle: int = 5,         # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_output: int = 35,       # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_output: int = 5,        # ZATR: 最大期間用ドミナントサイクル計算用
        
        zatr_min_dc_cycle_part: float = 0.5,   # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_cycle: int = 34,        # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_cycle: int = 3,         # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_output: int = 13,       # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_output: int = 3         # ZATR: 最小期間用ドミナントサイクル計算用
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: 検出器タイプ（ZMAとZATRに使用）
                - 'hody': ホモダイン判別機（デフォルト）
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積
                - 'dft': 離散フーリエ変換
            cer_detector_type: CER用の検出器タイプ（指定しない場合はdetector_typeと同じ）
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_multiplier: ATR乗数の最大値（レガシーパラメータ、デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（レガシーパラメータ、デフォルト: 1.5）
            
            max_max_multiplier: 最大乗数の最大値（デフォルト: 8.0）
            min_max_multiplier: 最大乗数の最小値（デフォルト: 3.0）
            max_min_multiplier: 最小乗数の最大値（デフォルト: 1.5）
            min_min_multiplier: 最小乗数の最小値（デフォルト: 0.3）
            
            smoother_type: 平滑化アルゴリズムのタイプ（デフォルト: 'alma'）
                'alma' - ALMA（Arnaud Legoux Moving Average）
                'hyper' - ハイパースムーサー（3段階平滑化）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            
            # CER用パラメータ
            cer_max_cycle: CER用の最大サイクル期間
            cer_min_cycle: CER用の最小サイクル期間
            cer_max_output: CER用の最大出力値
            cer_min_output: CER用の最小出力値
            
            # ZMA用パラメータ
            zma_max_dc_cycle_part: ZMA最大期間用ドミナントサイクル計算用のサイクル部分
            zma_max_dc_max_cycle: ZMA最大期間用ドミナントサイクル計算用の最大サイクル期間
            zma_max_dc_min_cycle: ZMA最大期間用ドミナントサイクル計算用の最小サイクル期間
            zma_max_dc_max_output: ZMA最大期間用ドミナントサイクル計算用の最大出力値
            zma_max_dc_min_output: ZMA最大期間用ドミナントサイクル計算用の最小出力値
            
            zma_min_dc_cycle_part: ZMA最小期間用ドミナントサイクル計算用のサイクル部分
            zma_min_dc_max_cycle: ZMA最小期間用ドミナントサイクル計算用の最大サイクル期間
            zma_min_dc_min_cycle: ZMA最小期間用ドミナントサイクル計算用の最小サイクル期間
            zma_min_dc_max_output: ZMA最小期間用ドミナントサイクル計算用の最大出力値
            zma_min_dc_min_output: ZMA最小期間用ドミナントサイクル計算用の最小出力値
            
            # ZMA動的Slow最大用パラメータ
            zma_slow_max_dc_cycle_part: ZMA動的Slow最大用ドミナントサイクル計算用のサイクル部分
            zma_slow_max_dc_max_cycle: ZMA動的Slow最大用ドミナントサイクル計算用の最大サイクル期間
            zma_slow_max_dc_min_cycle: ZMA動的Slow最大用ドミナントサイクル計算用の最小サイクル期間
            zma_slow_max_dc_max_output: ZMA動的Slow最大用ドミナントサイクル計算用の最大出力値
            zma_slow_max_dc_min_output: ZMA動的Slow最大用ドミナントサイクル計算用の最小出力値
            
            # ZMA動的Slow最小用パラメータ
            zma_slow_min_dc_cycle_part: ZMA動的Slow最小用ドミナントサイクル計算用のサイクル部分
            zma_slow_min_dc_max_cycle: ZMA動的Slow最小用ドミナントサイクル計算用の最大サイクル期間
            zma_slow_min_dc_min_cycle: ZMA動的Slow最小用ドミナントサイクル計算用の最小サイクル期間
            zma_slow_min_dc_max_output: ZMA動的Slow最小用ドミナントサイクル計算用の最大出力値
            zma_slow_min_dc_min_output: ZMA動的Slow最小用ドミナントサイクル計算用の最小出力値
            
            # ZMA動的Fast最大用パラメータ
            zma_fast_max_dc_cycle_part: ZMA動的Fast最大用ドミナントサイクル計算用のサイクル部分
            zma_fast_max_dc_max_cycle: ZMA動的Fast最大用ドミナントサイクル計算用の最大サイクル期間
            zma_fast_max_dc_min_cycle: ZMA動的Fast最大用ドミナントサイクル計算用の最小サイクル期間
            zma_fast_max_dc_max_output: ZMA動的Fast最大用ドミナントサイクル計算用の最大出力値
            zma_fast_max_dc_min_output: ZMA動的Fast最大用ドミナントサイクル計算用の最小出力値
            
            zma_min_fast_period: ZMA速い移動平均の最小期間（常に2で固定）
            zma_hyper_smooth_period: ZMAハイパースムーサーの平滑化期間（0=平滑化しない）
            
            # ZATR用パラメータ
            zatr_max_dc_cycle_part: ZATR最大期間用ドミナントサイクル計算用のサイクル部分
            zatr_max_dc_max_cycle: ZATR最大期間用ドミナントサイクル計算用の最大サイクル期間
            zatr_max_dc_min_cycle: ZATR最大期間用ドミナントサイクル計算用の最小サイクル期間
            zatr_max_dc_max_output: ZATR最大期間用ドミナントサイクル計算用の最大出力値
            zatr_max_dc_min_output: ZATR最大期間用ドミナントサイクル計算用の最小出力値
            
            zatr_min_dc_cycle_part: ZATR最小期間用ドミナントサイクル計算用のサイクル部分
            zatr_min_dc_max_cycle: ZATR最小期間用ドミナントサイクル計算用の最大サイクル期間
            zatr_min_dc_min_cycle: ZATR最小期間用ドミナントサイクル計算用の最小サイクル期間
            zatr_min_dc_max_output: ZATR最小期間用ドミナントサイクル計算用の最大出力値
            zatr_min_dc_min_output: ZATR最小期間用ドミナントサイクル計算用の最小出力値
        """
        super().__init__(
            f"ZChannel({detector_type}, {max_multiplier}, {min_multiplier}, {smoother_type})"
        )
        # CERの検出器タイプが指定されていない場合、detector_typeと同じ値を使用
        if cer_detector_type is None:
            cer_detector_type = detector_type
            
        self.detector_type = detector_type
        self.cer_detector_type = cer_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self.max_max_multiplier = max_max_multiplier
        self.min_max_multiplier = min_max_multiplier
        self.max_min_multiplier = max_min_multiplier
        self.min_min_multiplier = min_min_multiplier
        self.smoother_type = smoother_type
        self.src_type = src_type
        
        # CER用パラメータ
        self.cer_max_cycle = cer_max_cycle
        self.cer_min_cycle = cer_min_cycle
        self.cer_max_output = cer_max_output
        self.cer_min_output = cer_min_output
        
        # ZMA用パラメータ
        self.zma_max_dc_cycle_part = zma_max_dc_cycle_part
        self.zma_max_dc_max_cycle = zma_max_dc_max_cycle
        self.zma_max_dc_min_cycle = zma_max_dc_min_cycle
        self.zma_max_dc_max_output = zma_max_dc_max_output
        self.zma_max_dc_min_output = zma_max_dc_min_output
        
        self.zma_min_dc_cycle_part = zma_min_dc_cycle_part
        self.zma_min_dc_max_cycle = zma_min_dc_max_cycle
        self.zma_min_dc_min_cycle = zma_min_dc_min_cycle
        self.zma_min_dc_max_output = zma_min_dc_max_output
        self.zma_min_dc_min_output = zma_min_dc_min_output
        
        # ZMA動的Slow最大用パラメータ
        self.zma_slow_max_dc_cycle_part = zma_slow_max_dc_cycle_part
        self.zma_slow_max_dc_max_cycle = zma_slow_max_dc_max_cycle
        self.zma_slow_max_dc_min_cycle = zma_slow_max_dc_min_cycle
        self.zma_slow_max_dc_max_output = zma_slow_max_dc_max_output
        self.zma_slow_max_dc_min_output = zma_slow_max_dc_min_output
        
        # ZMA動的Slow最小用パラメータ
        self.zma_slow_min_dc_cycle_part = zma_slow_min_dc_cycle_part
        self.zma_slow_min_dc_max_cycle = zma_slow_min_dc_max_cycle
        self.zma_slow_min_dc_min_cycle = zma_slow_min_dc_min_cycle
        self.zma_slow_min_dc_max_output = zma_slow_min_dc_max_output
        self.zma_slow_min_dc_min_output = zma_slow_min_dc_min_output
        
        # ZMA動的Fast最大用パラメータ
        self.zma_fast_max_dc_cycle_part = zma_fast_max_dc_cycle_part
        self.zma_fast_max_dc_max_cycle = zma_fast_max_dc_max_cycle
        self.zma_fast_max_dc_min_cycle = zma_fast_max_dc_min_cycle
        self.zma_fast_max_dc_max_output = zma_fast_max_dc_max_output
        self.zma_fast_max_dc_min_output = zma_fast_max_dc_min_output
        
        self.zma_min_fast_period = zma_min_fast_period
        self.zma_hyper_smooth_period = zma_hyper_smooth_period
        
        # ZATR用パラメータ
        self.zatr_max_dc_cycle_part = zatr_max_dc_cycle_part
        self.zatr_max_dc_max_cycle = zatr_max_dc_max_cycle
        self.zatr_max_dc_min_cycle = zatr_max_dc_min_cycle
        self.zatr_max_dc_max_output = zatr_max_dc_max_output
        self.zatr_max_dc_min_output = zatr_max_dc_min_output
        
        self.zatr_min_dc_cycle_part = zatr_min_dc_cycle_part
        self.zatr_min_dc_max_cycle = zatr_min_dc_max_cycle
        self.zatr_min_dc_min_cycle = zatr_min_dc_min_cycle
        self.zatr_min_dc_max_output = zatr_min_dc_max_output
        self.zatr_min_dc_min_output = zatr_min_dc_min_output
        
        # コンポーネントのインスタンス化
        self.cer = CycleEfficiencyRatio(
            detector_type=cer_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_cycle=cer_max_cycle,
            min_cycle=cer_min_cycle,
            max_output=cer_max_output,
            min_output=cer_min_output,
            src_type=src_type
        )
        
        # ZMA専用のCER
        self.zma_cer = CycleEfficiencyRatio(
            detector_type='hody_e',
            lp_period=5,
            hp_period=55,
            cycle_part=0.55,
            max_cycle=55,
            min_cycle=3,
            max_output=55,
            min_output=5,
            src_type=src_type
        )
        
        # ZATR専用のCER
        self.zatr_cer = CycleEfficiencyRatio(
            detector_type='phac_e',
            lp_period=5,
            hp_period=55,
            cycle_part=0.55,
            max_cycle=100,
            min_cycle=3,
            max_output=55,
            min_output=13,
            src_type=src_type
        )
        
        # ZMAの初期化
        self.zma = ZMA(
            # 基本パラメータ
            max_dc_cycle_part=zma_max_dc_cycle_part,
            max_dc_max_cycle=zma_max_dc_max_cycle,
            max_dc_min_cycle=zma_max_dc_min_cycle,
            max_dc_max_output=zma_max_dc_max_output,
            max_dc_min_output=zma_max_dc_min_output,
            
            min_dc_cycle_part=zma_min_dc_cycle_part,
            min_dc_max_cycle=zma_min_dc_max_cycle,
            min_dc_min_cycle=zma_min_dc_min_cycle,
            min_dc_max_output=zma_min_dc_max_output,
            min_dc_min_output=zma_min_dc_min_output,
            
            # 動的Slow最大用パラメータ
            slow_max_dc_cycle_part=zma_slow_max_dc_cycle_part,
            slow_max_dc_max_cycle=zma_slow_max_dc_max_cycle,
            slow_max_dc_min_cycle=zma_slow_max_dc_min_cycle,
            slow_max_dc_max_output=zma_slow_max_dc_max_output,
            slow_max_dc_min_output=zma_slow_max_dc_min_output,
            
            # 動的Slow最小用パラメータ
            slow_min_dc_cycle_part=zma_slow_min_dc_cycle_part,
            slow_min_dc_max_cycle=zma_slow_min_dc_max_cycle,
            slow_min_dc_min_cycle=zma_slow_min_dc_min_cycle,
            slow_min_dc_max_output=zma_slow_min_dc_max_output,
            slow_min_dc_min_output=zma_slow_min_dc_min_output,
            
            # 動的Fast最大用パラメータ
            fast_max_dc_cycle_part=zma_fast_max_dc_cycle_part,
            fast_max_dc_max_cycle=zma_fast_max_dc_max_cycle,
            fast_max_dc_min_cycle=zma_fast_max_dc_min_cycle,
            fast_max_dc_max_output=zma_fast_max_dc_max_output,
            fast_max_dc_min_output=zma_fast_max_dc_min_output,
            
            min_fast_period=zma_min_fast_period,
            hyper_smooth_period=zma_hyper_smooth_period,
            src_type=src_type,
            detector_type=detector_type
        )
        
        self.zatr = ZATR(
            detector_type=detector_type,  # ZATR用の検出器タイプ
            # ZATRクラスのパラメータに合わせて修正
            max_dc_cycle_part=zatr_max_dc_cycle_part,
            max_dc_max_cycle=zatr_max_dc_max_cycle,
            max_dc_min_cycle=zatr_max_dc_min_cycle,
            max_dc_max_output=zatr_max_dc_max_output,
            max_dc_min_output=zatr_max_dc_min_output,
            
            min_dc_cycle_part=zatr_min_dc_cycle_part,
            min_dc_max_cycle=zatr_min_dc_max_cycle,
            min_dc_min_cycle=zatr_min_dc_min_cycle,
            min_dc_max_output=zatr_min_dc_max_output,
            min_dc_min_output=zatr_min_dc_min_output,
            
            smoother_type=smoother_type
        )
        
        # コンピューターオブジェクトの初期化
        self._cache = {}
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする（高速化）
            cols = ['high', 'low', 'close']
            # NumPyでの高速ハッシュ計算
            data_values = np.vstack([data[col].values for col in cols if col in data.columns])
            data_hash = hash(data_values.tobytes())
        else:
            # NumPy配列の場合は全体をハッシュする（高速化）
            data_hash = hash(data.tobytes() if isinstance(data, np.ndarray) else str(data))
        
        # パラメータ値を含めることで、同じデータでもパラメータが異なる場合に再計算する
        # ZMAとZATRのパラメータもハッシュに含める
        param_str = (
            f"{self.detector_type}_{self.cer_detector_type}_{self.lp_period}_{self.hp_period}_{self.cycle_part}_"
            f"{self.max_multiplier}_{self.min_multiplier}_{self.smoother_type}_{self.src_type}_"
            f"{self.max_max_multiplier}_{self.min_max_multiplier}_{self.max_min_multiplier}_{self.min_min_multiplier}_"
            f"zma_{self.zma_max_dc_cycle_part}_{self.zma_max_dc_max_cycle}_{self.zma_max_dc_min_cycle}_"
            f"{self.zma_max_dc_max_output}_{self.zma_max_dc_min_output}_"
            f"{self.zma_min_dc_cycle_part}_{self.zma_min_dc_max_cycle}_{self.zma_min_dc_min_cycle}_"
            f"{self.zma_min_dc_max_output}_{self.zma_min_dc_min_output}_"
            f"{self.zma_slow_max_dc_cycle_part}_{self.zma_slow_max_dc_max_cycle}_{self.zma_slow_max_dc_min_cycle}_"
            f"{self.zma_slow_max_dc_max_output}_{self.zma_slow_max_dc_min_output}_"
            f"{self.zma_slow_min_dc_cycle_part}_{self.zma_slow_min_dc_max_cycle}_{self.zma_slow_min_dc_min_cycle}_"
            f"{self.zma_slow_min_dc_max_output}_{self.zma_slow_min_dc_min_output}_"
            f"{self.zma_fast_max_dc_cycle_part}_{self.zma_fast_max_dc_max_cycle}_{self.zma_fast_max_dc_min_cycle}_"
            f"{self.zma_fast_max_dc_max_output}_{self.zma_fast_max_dc_min_output}_"
            f"{self.zma_min_fast_period}_{self.zma_hyper_smooth_period}_"
            f"zatr_{self.zatr_max_dc_cycle_part}_{self.zatr_max_dc_max_cycle}_{self.zatr_max_dc_min_cycle}_"
            f"{self.zatr_max_dc_max_output}_{self.zatr_max_dc_min_output}_"
            f"{self.zatr_min_dc_cycle_part}_{self.zatr_min_dc_max_cycle}_{self.zatr_min_dc_min_cycle}_"
            f"{self.zatr_min_dc_max_output}_{self.zatr_min_dc_min_output}"
        )
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Zチャネルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high', 'low', 'close'カラムが必要
        
        Returns:
            中心線の値（ZMA）
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash in self._cache:
                return self._cache[data_hash]
            
            # データの検証と変換（必要最小限の処理）
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
            elif data.ndim != 2 or data.shape[1] < 4:
                raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # ZMA用のサイクル効率比（CER）の計算
            zma_cer = self.zma_cer.calculate(data)
            if zma_cer is None or len(zma_cer) == 0:
                raise ValueError("ZMA用のサイクル効率比（CER）の計算に失敗しました")
            
            # ZATR用のサイクル効率比（CER）の計算
            zatr_cer = self.zatr_cer.calculate(data)
            if zatr_cer is None or len(zatr_cer) == 0:
                raise ValueError("ZATR用のサイクル効率比（CER）の計算に失敗しました")
            
            # メインのCER（表示・計算用）も計算しておく
            cer = self.cer.calculate(data)
            
            # ZMAの計算（ZMA専用のサイクル効率比を使用）
            z_ma_values = self.zma.calculate(data, external_er=zma_cer)
            if z_ma_values is None:
                raise ValueError("ZMAの計算に失敗しました")
            
            # ZATRの計算（ZATR専用のサイクル効率比を使用）
            # 注: calculate()メソッドは%ベースのATRを返します
            z_atr_values = self.zatr.calculate(data, external_er=zatr_cer)
            if z_atr_values is None:
                raise ValueError("ZATRの計算に失敗しました")
            
            # 金額ベースのZATRを取得（パーセントベースではなく）
            # 重要: バンド計算には金額ベース(絶対値)のATRを使用する必要があります
            z_atr_absolute = self.zatr.get_absolute_atr()
            
            # データをNumPy配列に変換して計算を高速化
            cer_np = np.asarray(cer, dtype=np.float64)
            
            # 動的な最大・最小乗数の計算
            max_mult_values = calculate_dynamic_max_multiplier(
                cer_np,
                self.max_max_multiplier,
                self.min_max_multiplier
            )
            
            min_mult_values = calculate_dynamic_min_multiplier(
                cer_np,
                self.max_min_multiplier,
                self.min_min_multiplier
            )
            
            # 動的ATR乗数の計算（ベクトル化関数を使用）- 高速化
            dynamic_multiplier = calculate_dynamic_multiplier_vec(
                cer_np,
                max_mult_values,
                min_mult_values
            )
            
            # Zチャネルの計算（パラレル高速版）
            # 金額ベースのATRを使用
            middle, upper, lower = calculate_z_channel(
                np.asarray(z_ma_values, dtype=np.float64),
                np.asarray(z_atr_absolute, dtype=np.float64),
                np.asarray(dynamic_multiplier, dtype=np.float64),
                use_percent=False    # 金額ベース計算を使用
            )
            
            # 結果の保存（参照コピーを避けるためnp.copyを使用）
            result = ZChannelResult(
                middle=np.copy(middle),
                upper=np.copy(upper),
                lower=np.copy(lower),
                cer=np.copy(cer),
                dynamic_multiplier=np.copy(dynamic_multiplier),
                z_atr=np.copy(z_atr_absolute),  # 金額ベースのATRを保存
                max_mult_values=np.copy(max_mult_values),  # 動的最大乗数を保存
                min_mult_values=np.copy(min_mult_values)   # 動的最小乗数を保存
            )
            
            # 結果をクラス変数に保存
            self._result = result
            
            # 中心線を値として保存
            self._cache[data_hash] = middle
            return middle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ZChannel計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の配列を返す
            return np.array([])
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Zチャネルのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限バンド, 下限バンド)の値
        """
        if data is not None:
            # データが指定された場合は再計算
            self.calculate(data)
            
        # 結果確認
        if not hasattr(self, '_result') or self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
            
        return self._result.middle, self._result.upper, self._result.lower
    
    def get_cycle_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if data is not None:
            # データが指定された場合は再計算
            self.calculate(data)
            
        # 結果確認
        if not hasattr(self, '_result') or self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
            
        return self._result.cer
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的ATR乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的ATR乗数の値
        """
        if data is not None:
            # データが指定された場合は再計算
            self.calculate(data)
            
        # 結果確認
        if not hasattr(self, '_result') or self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
            
        return self._result.dynamic_multiplier
    
    def get_z_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZATR値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ZATR値
        """
        if data is not None:
            # データが指定された場合は再計算
            self.calculate(data)
            
        # 結果確認
        if not hasattr(self, '_result') or self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
            
        return self._result.z_atr
    
    def get_dynamic_max_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的な最大ATR乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的最大ATR乗数の値
        """
        if data is not None:
            # データが指定された場合は再計算
            self.calculate(data)
            
        # 結果確認
        if not hasattr(self, '_result') or self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
            
        return self._result.max_mult_values
    
    def get_dynamic_min_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的な最小ATR乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的最小ATR乗数の値
        """
        if data is not None:
            # データが指定された場合は再計算
            self.calculate(data)
            
        # 結果確認
        if not hasattr(self, '_result') or self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
            
        return self._result.min_mult_values
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self._cache = {}
        # メインCERのリセット
        self.cer.reset() if hasattr(self.cer, 'reset') else None
        # ZMA用CERのリセット
        self.zma_cer.reset() if hasattr(self.zma_cer, 'reset') else None
        # ZATR用CERのリセット
        self.zatr_cer.reset() if hasattr(self.zatr_cer, 'reset') else None
        # ZMAとZATRのリセット
        self.zma.reset() if hasattr(self.zma, 'reset') else None
        self.zatr.reset() if hasattr(self.zatr, 'reset') else None 