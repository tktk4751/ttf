#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit

from .indicator import Indicator
from .hyper_smoother import calculate_hyper_smoother_numba
from .cycle.ehlers_unified_dc import EhlersUnifiedDC


@dataclass
class ZMAResult:
    """ZMAの計算結果"""
    values: np.ndarray        # ZMAの値（スムージング済み）
    raw_values: np.ndarray    # 生のZMA値（スムージング前）
    er: np.ndarray            # サイクル効率比（CER）
    dynamic_kama_period: np.ndarray  # 動的KAMAピリオド
    dynamic_fast_period: np.ndarray  # 動的Fast期間
    dynamic_slow_period: np.ndarray  # 動的Slow期間
    dc_values: np.ndarray     # ドミナントサイクル値
    period_dc_values: np.ndarray    # 動的Fast最大用ドミナントサイクル値（以前のperiod_dc_values）


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_kama_period_vec(er: float, max_period: float, min_period: float) -> float:
    """
    効率比に基づいて動的なKAMAピリオドを計算する（ベクトル化版）
    
    Args:
        er: 効率比の値（ERまたはCER）
        max_period: 最大期間
        min_period: 最小期間
    
    Returns:
        動的な期間の値
    """
    if np.isnan(er):
        return np.nan
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    return np.round(min_period + (1.0 - abs(er)) * (max_period - min_period))


@njit(fastmath=True, parallel=True, cache=True)
def calculate_dynamic_kama_period(er: np.ndarray, max_period: np.ndarray, min_period: np.ndarray) -> np.ndarray:
    """
    効率比に基づいて動的なKAMAピリオドを計算する（高速化版）
    
    Args:
        er: 効率比の配列（ERまたはCER）
        max_period: 最大期間の配列
        min_period: 最小期間の配列
    
    Returns:
        動的な期間の配列
    """
    length = len(er)
    periods = np.zeros(length, dtype=np.float64)
    
    # 並列処理で高速化
    for i in prange(length):
        if np.isnan(er[i]):
            periods[i] = np.nan
        else:
            # ERが高い（トレンドが強い）ほど期間は短く、
            # ERが低い（トレンドが弱い）ほど期間は長くなる
            er_abs = abs(er[i])
            periods[i] = min_period[i] + (1.0 - er_abs) * (max_period[i] - min_period[i])
    
    return np.round(periods).astype(np.int32)


@njit(fastmath=True, parallel=True, cache=True)
def calculate_dynamic_kama_constants(er: np.ndarray, 
                                    max_slow: int, min_slow: int,
                                    max_fast: int, min_fast: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    効率比に基づいて動的なKAMAのfast/slow期間を計算する（高速化版）
    
    Args:
        er: 効率比の配列（ERまたはCER）
        max_slow: 遅い移動平均の最大期間
        min_slow: 遅い移動平均の最小期間
        max_fast: 速い移動平均の最大期間
        min_fast: 速い移動平均の最小期間
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            動的なfast期間の配列、動的なslow期間の配列、fastの定数、slowの定数
    """
    # ERが高い（トレンドが強い）ほど期間は短く、
    # ERが低い（トレンドが弱い）ほど期間は長くなる
    er_abs = np.abs(er)
    
    # 期間計算 - ベクトル化
    fast_periods = np.zeros_like(er, dtype=np.float64)
    slow_periods = np.zeros_like(er, dtype=np.float64)
    
    # 並列処理で高速化
    for i in prange(len(er)):
        if not np.isnan(er_abs[i]):
            fast_periods[i] = min_fast + (1.0 - er_abs[i]) * (max_fast - min_fast)
            slow_periods[i] = min_slow + (1.0 - er_abs[i]) * (max_slow - min_slow)
        else:
            fast_periods[i] = min_fast
            slow_periods[i] = min_slow
    
    fast_periods_rounded = np.round(fast_periods).astype(np.int32)
    slow_periods_rounded = np.round(slow_periods).astype(np.int32)
    
    # 定数の計算
    fast_constants = 2.0 / (fast_periods + 1.0)
    slow_constants = 2.0 / (slow_periods + 1.0)
    
    return fast_periods_rounded, slow_periods_rounded, fast_constants, slow_constants


@njit(fastmath=True, cache=True)
def calculate_z_ma(prices: np.ndarray, er: np.ndarray, er_period: int,
                  kama_period: np.ndarray,
                  fast_constants: np.ndarray, slow_constants: np.ndarray) -> np.ndarray:
    """
    ZMAを計算する（高速化版）
    
    Args:
        prices: 価格の配列（closeやhlc3などの計算済みソース）
        er: 効率比の配列（ERまたはCER）
        er_period: 効率比の計算期間
        kama_period: 動的なKAMAピリオドの配列
        fast_constants: 速い移動平均の定数配列
        slow_constants: 遅い移動平均の定数配列
    
    Returns:
        ZMAの配列
    """
    length = len(prices)
    z_ma = np.full(length, np.nan, dtype=np.float64)
    
    # 最初のZMAは最初の価格
    if length > 0:
        z_ma[0] = prices[0]
    
    # 一時配列を使用して計算効率化
    changes = np.zeros(length, dtype=np.float64)
    volatilities = np.zeros(length, dtype=np.float64)
    current_ers = np.zeros(length, dtype=np.float64)
    smoothing_constants = np.zeros(length, dtype=np.float64)
    
    # 各時点でのZMAを計算
    for i in range(1, length):
        if np.isnan(er[i]) or i < er_period:
            z_ma[i] = z_ma[i-1]
            continue
        
        # 現在の動的パラメータを取得
        curr_kama_period = int(kama_period[i])
        if curr_kama_period < 1:
            curr_kama_period = 1
        
        # 現在の時点での実際のER値を計算
        if i >= curr_kama_period:
            # 変化とボラティリティの計算
            changes[i] = prices[i] - prices[i-curr_kama_period]
            
            # ボラティリティ計算を高速化
            volatility = 0.0
            start_idx = max(0, i-curr_kama_period)
            for j in range(start_idx+1, i+1):
                volatility += abs(prices[j] - prices[j-1])
            volatilities[i] = volatility
            
            # ゼロ除算を防止
            if volatilities[i] < 1e-10:
                current_ers[i] = 0.0
            else:
                current_ers[i] = abs(changes[i]) / volatilities[i]
            
            # スムージング定数の計算
            sc = (current_ers[i] * (fast_constants[i] - slow_constants[i]) + slow_constants[i]) ** 2
            
            # 0-1の範囲に制限
            smoothing_constants[i] = max(0.0, min(1.0, sc))
            
            # ZMAの計算
            z_ma[i] = z_ma[i-1] + smoothing_constants[i] * (prices[i] - z_ma[i-1])
        else:
            z_ma[i] = z_ma[i-1]
    
    return z_ma


class ZMA(Indicator):
    """
    ZMA (Z Moving Average) インジケーター
    
    サイクル効率比（CER）とドミナントサイクルに基づいて以下のパラメータを動的に調整する適応型移動平均線：
    - KAMAのmax/min期間をサイクル検出器で動的に決定
    - KAMAのfast期間
    - KAMAのslow期間
    
    特徴:
    - すべてのパラメータがサイクル効率比とドミナントサイクルに応じて動的に調整される
    - トレンドが強い時：短いピリオドと速い反応
    - レンジ相場時：長いピリオドとノイズ除去
    - オプションのハイパースムーサーによる追加平滑化
    - 固定値と動的値の両方をサポート
    """
    
    def __init__(
        self,
        max_dc_cycle_part: float = 0.5,      
        
            # 最大期間用ドミナントサイクル計算用
        max_dc_max_cycle: int = 144,              # 最大期間用ドミナントサイクル計算用
        max_dc_min_cycle: int = 5,               # 最大期間用ドミナントサイクル計算用
        max_dc_max_output: int = 89,             # 最大期間用ドミナントサイクル計算用
        max_dc_min_output: int = 22,              # 最大期間用ドミナントサイクル計算用
        max_dc_lp_period: int = 5,               # 最大期間用ドミナントサイクル計算用LPピリオド
        max_dc_hp_period: int = 55,              # 最大期間用ドミナントサイクル計算用HPピリオド
        
        min_dc_cycle_part: float = 0.25,          # 最小期間用ドミナントサイクル計算用
        min_dc_max_cycle: int = 55,               # 最小期間用ドミナントサイクル計算用
        min_dc_min_cycle: int = 5,                # 最小期間用ドミナントサイクル計算用
        min_dc_max_output: int = 13,              # 最小期間用ドミナントサイクル計算用
        min_dc_min_output: int = 3,               # 最小期間用ドミナントサイクル計算用
        min_dc_lp_period: int = 5,               # 最小期間用ドミナントサイクル計算用LPピリオド
        min_dc_hp_period: int = 34,              # 最小期間用ドミナントサイクル計算用HPピリオド
        
        # 動的Slow最大用パラメータ
        slow_max_dc_cycle_part: float = 0.5,
        slow_max_dc_max_cycle: int = 144,
        slow_max_dc_min_cycle: int = 5,
        slow_max_dc_max_output: int = 89,
        slow_max_dc_min_output: int = 22,
        slow_max_dc_lp_period: int = 5,          # Slow最大用ドミナントサイクル計算用LPピリオド
        slow_max_dc_hp_period: int = 55,         # Slow最大用ドミナントサイクル計算用HPピリオド
        
        # 動的Slow最小用パラメータ
        slow_min_dc_cycle_part: float = 0.5,
        slow_min_dc_max_cycle: int = 89,
        slow_min_dc_min_cycle: int = 5,
        slow_min_dc_max_output: int = 21,
        slow_min_dc_min_output: int = 8,
        slow_min_dc_lp_period: int = 5,          # Slow最小用ドミナントサイクル計算用LPピリオド
        slow_min_dc_hp_period: int = 34,         # Slow最小用ドミナントサイクル計算用HPピリオド
        
        # 動的Fast最大用パラメータ
        fast_max_dc_cycle_part: float = 0.5,
        fast_max_dc_max_cycle: int = 55,
        fast_max_dc_min_cycle: int = 5,
        fast_max_dc_max_output: int = 15,
        fast_max_dc_min_output: int = 3,
        fast_max_dc_lp_period: int = 5,          # Fast最大用ドミナントサイクル計算用LPピリオド
        fast_max_dc_hp_period: int = 21,         # Fast最大用ドミナントサイクル計算用HPピリオド
        
        min_fast_period: int = 2,
        hyper_smooth_period: int = 0,
        src_type: str = 'hlc3',
        detector_type: str = 'hody',
        
        # 動的パラメータか固定値かを選択するためのオプション
        use_dynamic_periods: bool = True,     # 動的期間を使用するかどうか
        fixed_fast_period: int = 2,           # 固定ファスト期間
        fixed_slow_period: int = 30,          # 固定スロー期間
    ):
        """
        コンストラクタ
        
        Args:
            max_dc_cycle_part: 最大期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            max_dc_max_cycle: 最大期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 144）
            max_dc_min_cycle: 最大期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            max_dc_max_output: 最大期間用ドミナントサイクル計算用の最大出力値（デフォルト: 89）
            max_dc_min_output: 最大期間用ドミナントサイクル計算用の最小出力値（デフォルト: 22）
            
            max_dc_lp_period: 最大期間用ドミナントサイクル計算用LPピリオド（デフォルト: 5）
            max_dc_hp_period: 最大期間用ドミナントサイクル計算用HPピリオド（デフォルト: 55）
            
            min_dc_cycle_part: 最小期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.25）
            min_dc_max_cycle: 最小期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            min_dc_min_cycle: 最小期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            min_dc_max_output: 最小期間用ドミナントサイクル計算用の最大出力値（デフォルト: 13）
            min_dc_min_output: 最小期間用ドミナントサイクル計算用の最小出力値（デフォルト: 3）
            
            min_dc_lp_period: 最小期間用ドミナントサイクル計算用LPピリオド（デフォルト: 5）
            min_dc_hp_period: 最小期間用ドミナントサイクル計算用HPピリオド（デフォルト: 34）
            
            # 動的Slow最大用パラメータ
            slow_max_dc_cycle_part: 動的Slow最大用サイクル部分（デフォルト: 0.5）
            slow_max_dc_max_cycle: 動的Slow最大用の最大サイクル期間（デフォルト: 144）
            slow_max_dc_min_cycle: 動的Slow最大用の最小サイクル期間（デフォルト: 5）
            slow_max_dc_max_output: 動的Slow最大用の最大出力値（デフォルト: 89）
            slow_max_dc_min_output: 動的Slow最大用の最小出力値（デフォルト: 22）
            
            slow_max_dc_lp_period: 動的Slow最大用ドミナントサイクル計算用LPピリオド（デフォルト: 5）
            slow_max_dc_hp_period: 動的Slow最大用ドミナントサイクル計算用HPピリオド（デフォルト: 55）
            
            # 動的Slow最小用パラメータ
            slow_min_dc_cycle_part: 動的Slow最小用サイクル部分（デフォルト: 0.5）
            slow_min_dc_max_cycle: 動的Slow最小用の最大サイクル期間（デフォルト: 89）
            slow_min_dc_min_cycle: 動的Slow最小用の最小サイクル期間（デフォルト: 5）
            slow_min_dc_max_output: 動的Slow最小用の最大出力値（デフォルト: 21）
            slow_min_dc_min_output: 動的Slow最小用の最小出力値（デフォルト: 8）
            
            slow_min_dc_lp_period: 動的Slow最小用ドミナントサイクル計算用LPピリオド（デフォルト: 5）
            slow_min_dc_hp_period: 動的Slow最小用ドミナントサイクル計算用HPピリオド（デフォルト: 34）
            
            # 動的Fast最大用パラメータ
            fast_max_dc_cycle_part: 動的Fast最大用サイクル部分（デフォルト: 0.5）
            fast_max_dc_max_cycle: 動的Fast最大用の最大サイクル期間（デフォルト: 55）
            fast_max_dc_min_cycle: 動的Fast最大用の最小サイクル期間（デフォルト: 5）
            fast_max_dc_max_output: 動的Fast最大用の最大出力値（デフォルト: 15）
            fast_max_dc_min_output: 動的Fast最大用の最小出力値（デフォルト: 3）
            
            fast_max_dc_lp_period: 動的Fast最大用ドミナントサイクル計算用LPピリオド（デフォルト: 5）
            fast_max_dc_hp_period: 動的Fast最大用ドミナントサイクル計算用HPピリオド（デフォルト: 21）
            
            min_fast_period: 速い移動平均の最小期間（デフォルト: 2、常に固定）
            hyper_smooth_period: ハイパースムーサーの平滑化期間（デフォルト: 0、0以下の場合は平滑化しない）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値
                - 'hlc3': (高値 + 安値 + 終値) / 3（デフォルト）
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
            detector_type: 検出器タイプ
                - 'hody': ホモダイン判別機（デフォルト）
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積
                - 'dft': 離散フーリエ変換
                
            use_dynamic_periods: 動的な期間を使用するかどうか（デフォルト: True）
                - True: サイクル検出器によって動的に期間を調整
                - False: 固定値のfast/slow期間を使用
            fixed_fast_period: 固定ファスト期間（デフォルト: 2、use_dynamic_periodsがFalseの場合に使用）
            fixed_slow_period: 固定スロー期間（デフォルト: 30、use_dynamic_periodsがFalseの場合に使用）
        """
        # 指標名の作成
        indicator_name = f"ZMA({max_dc_max_output}, {min_dc_max_output}, {min_fast_period}, {src_type})"
        if not use_dynamic_periods:
            indicator_name = f"ZMA(fast={fixed_fast_period}, slow={fixed_slow_period}, {src_type})"
            
        super().__init__(indicator_name)
        
        # 最大期間用パラメータ
        self.max_dc_cycle_part = max_dc_cycle_part
        self.max_dc_max_cycle = max_dc_max_cycle
        self.max_dc_min_cycle = max_dc_min_cycle
        self.max_dc_max_output = max_dc_max_output
        self.max_dc_min_output = max_dc_min_output
        self.max_dc_lp_period = max_dc_lp_period
        self.max_dc_hp_period = max_dc_hp_period
        
        # 最小期間用パラメータ
        self.min_dc_cycle_part = min_dc_cycle_part
        self.min_dc_max_cycle = min_dc_max_cycle
        self.min_dc_min_cycle = min_dc_min_cycle
        self.min_dc_max_output = min_dc_max_output
        self.min_dc_min_output = min_dc_min_output
        self.min_dc_lp_period = min_dc_lp_period
        self.min_dc_hp_period = min_dc_hp_period
        
        # 動的Slow最大用パラメータ
        self.slow_max_dc_cycle_part = slow_max_dc_cycle_part
        self.slow_max_dc_max_cycle = slow_max_dc_max_cycle
        self.slow_max_dc_min_cycle = slow_max_dc_min_cycle
        self.slow_max_dc_max_output = slow_max_dc_max_output
        self.slow_max_dc_min_output = slow_max_dc_min_output
        self.slow_max_dc_lp_period = slow_max_dc_lp_period
        self.slow_max_dc_hp_period = slow_max_dc_hp_period
        
        # 動的Slow最小用パラメータ
        self.slow_min_dc_cycle_part = slow_min_dc_cycle_part
        self.slow_min_dc_max_cycle = slow_min_dc_max_cycle
        self.slow_min_dc_min_cycle = slow_min_dc_min_cycle
        self.slow_min_dc_max_output = slow_min_dc_max_output
        self.slow_min_dc_min_output = slow_min_dc_min_output
        self.slow_min_dc_lp_period = slow_min_dc_lp_period
        self.slow_min_dc_hp_period = slow_min_dc_hp_period
        
        # 動的Fast最大用パラメータ
        self.fast_max_dc_cycle_part = fast_max_dc_cycle_part
        self.fast_max_dc_max_cycle = fast_max_dc_max_cycle
        self.fast_max_dc_min_cycle = fast_max_dc_min_cycle
        self.fast_max_dc_max_output = fast_max_dc_max_output
        self.fast_max_dc_min_output = fast_max_dc_min_output
        self.fast_max_dc_lp_period = fast_max_dc_lp_period
        self.fast_max_dc_hp_period = fast_max_dc_hp_period
        
        # KAMAのfast/slow期間
        self.min_fast_period = min_fast_period  # 動的モード時の最小値
        
        # 固定値オプション用のパラメータ
        self.use_dynamic_periods = use_dynamic_periods
        self.fixed_fast_period = fixed_fast_period
        self.fixed_slow_period = fixed_slow_period
        
        self.hyper_smooth_period = hyper_smooth_period
        self.src_type = src_type.lower()
        self.detector_type = detector_type.lower()
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        self._result = None
        self._data_hash = None  # データキャッシュ用ハッシュ
        
        # サイクル検出器の初期化（use_dynamic_periodsがTrueの場合のみ使用）
        if self.use_dynamic_periods:
            # 最大期間用ドミナントサイクル検出器を初期化
            self.max_dc_detector = EhlersUnifiedDC(
                detector_type=self.detector_type,
                cycle_part=self.max_dc_cycle_part,
                max_cycle=self.max_dc_max_cycle,
                min_cycle=self.max_dc_min_cycle,
                max_output=self.max_dc_max_output,
                min_output=self.max_dc_min_output,
                src_type='hlc3',
                lp_period=self.max_dc_lp_period,
                hp_period=self.max_dc_hp_period
            )
            
            # 最小期間用ドミナントサイクル検出器を初期化
            self.min_dc_detector = EhlersUnifiedDC(
                detector_type=self.detector_type,
                cycle_part=self.min_dc_cycle_part,
                max_cycle=self.min_dc_max_cycle,
                min_cycle=self.min_dc_min_cycle,
                max_output=self.min_dc_max_output,
                min_output=self.min_dc_min_output,
                src_type='hlc3',
                lp_period=self.min_dc_lp_period,
                hp_period=self.min_dc_hp_period
            )
            
            # 動的Slow最大用ドミナントサイクル検出器を初期化
            self.slow_max_dc_detector = EhlersUnifiedDC(
                detector_type=self.detector_type,
                cycle_part=self.slow_max_dc_cycle_part,
                max_cycle=self.slow_max_dc_max_cycle,
                min_cycle=self.slow_max_dc_min_cycle,
                max_output=self.slow_max_dc_max_output,
                min_output=self.slow_max_dc_min_output,
                src_type='hlc3',
                lp_period=self.slow_max_dc_lp_period,
                hp_period=self.slow_max_dc_hp_period
            )
            
            # 動的Slow最小用ドミナントサイクル検出器を初期化
            self.slow_min_dc_detector = EhlersUnifiedDC(
                detector_type=self.detector_type,
                cycle_part=self.slow_min_dc_cycle_part,
                max_cycle=self.slow_min_dc_max_cycle,
                min_cycle=self.slow_min_dc_min_cycle,
                max_output=self.slow_min_dc_max_output,
                min_output=self.slow_min_dc_min_output,
                src_type='hlc3',
                lp_period=self.slow_min_dc_lp_period,
                hp_period=self.slow_min_dc_hp_period
            )
            
            # 動的Fast最大用ドミナントサイクル検出器を初期化
            self.fast_max_dc_detector = EhlersUnifiedDC(
                detector_type=self.detector_type,
                cycle_part=self.fast_max_dc_cycle_part,
                max_cycle=self.fast_max_dc_max_cycle,
                min_cycle=self.fast_max_dc_min_cycle,
                max_output=self.fast_max_dc_max_output,
                min_output=self.fast_max_dc_min_output,
                src_type='hlc3',
                lp_period=self.fast_max_dc_lp_period,
                hp_period=self.fast_max_dc_hp_period
            )
        else:
            # 動的期間を使用しない場合はNoneに設定
            self.max_dc_detector = None
            self.min_dc_detector = None
            self.slow_max_dc_detector = None
            self.slow_min_dc_detector = None
            self.fast_max_dc_detector = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする（高速化）
            if 'close' in data.columns:
                data_hash = hash(data['close'].values.tobytes())
            else:
                # closeカラムがない場合は全カラムのハッシュ（高速化）
                cols = ['high', 'low', 'close', 'open']
                data_values = np.vstack([data[col].values for col in cols if col in data.columns])
                data_hash = hash(data_values.tobytes())
        else:
            # NumPy配列の場合（高速化）
            if isinstance(data, np.ndarray):
                if data.ndim == 2 and data.shape[1] >= 4:
                    # OHLCデータの場合はcloseだけハッシュ（高速化）
                    data_hash = hash(data[:, 3].tobytes())
                else:
                    # それ以外は全体をハッシュ（高速化）
                    data_hash = hash(data.tobytes())
            else:
                data_hash = hash(str(data))
        
        # 外部ERがある場合はそのハッシュも含める（高速化）
        external_er_hash = "no_external_er"
        if external_er is not None and isinstance(external_er, np.ndarray):
            external_er_hash = hash(external_er.tobytes())
        
        # パラメータ値を含める
        param_str = (
            f"{self.max_dc_cycle_part}_{self.max_dc_max_cycle}_{self.max_dc_min_cycle}_"
            f"{self.max_dc_max_output}_{self.max_dc_min_output}_"
            f"{self.max_dc_lp_period}_{self.max_dc_hp_period}_"
            f"{self.min_dc_cycle_part}_{self.min_dc_max_cycle}_{self.min_dc_min_cycle}_"
            f"{self.min_dc_max_output}_{self.min_dc_min_output}_"
            f"{self.min_dc_lp_period}_{self.min_dc_hp_period}_"
            f"{self.slow_max_dc_cycle_part}_{self.slow_max_dc_max_cycle}_{self.slow_max_dc_min_cycle}_"
            f"{self.slow_max_dc_max_output}_{self.slow_max_dc_min_output}_"
            f"{self.slow_max_dc_lp_period}_{self.slow_max_dc_hp_period}_"
            f"{self.slow_min_dc_cycle_part}_{self.slow_min_dc_max_cycle}_{self.slow_min_dc_min_cycle}_"
            f"{self.slow_min_dc_max_output}_{self.slow_min_dc_min_output}_"
            f"{self.slow_min_dc_lp_period}_{self.slow_min_dc_hp_period}_"
            f"{self.fast_max_dc_cycle_part}_{self.fast_max_dc_max_cycle}_{self.fast_max_dc_min_cycle}_"
            f"{self.fast_max_dc_max_output}_{self.fast_max_dc_min_output}_"
            f"{self.fast_max_dc_lp_period}_{self.fast_max_dc_hp_period}_"
            f"{self.min_fast_period}_"
            f"{self.hyper_smooth_period}_{self.src_type}_{self.detector_type}_{external_er_hash}"
        )
        return f"{data_hash}_{param_str}"
    
    def _calculate_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[int, int, int]:
        """
        ドミナントサイクルに基づいて動的なperiodを計算する
        
        Args:

        
            data: 価格データ（DataFrameまたはNumPy配列）
                
        Returns:
            Tuple[int, int, int]: 動的なmax_slow_period, min_slow_period, max_fast_period
        """
        try:
            # 3つの検出器を使用して各ドミナントサイクル値を計算
            slow_max_dc_values = self.slow_max_dc_detector.calculate(data)
            slow_min_dc_values = self.slow_min_dc_detector.calculate(data)
            fast_max_dc_values = self.fast_max_dc_detector.calculate(data)
            
            # 結果が有効か確認
            if (len(slow_max_dc_values) == 0 or np.isnan(slow_max_dc_values).all() or
                len(slow_min_dc_values) == 0 or np.isnan(slow_min_dc_values).all() or
                len(fast_max_dc_values) == 0 or np.isnan(fast_max_dc_values).all()):
                # 計算失敗または空の結果の場合はデフォルト値を返す
                return self.slow_max_dc_max_output, self.slow_min_dc_max_output, self.fast_max_dc_max_output
            
            # 最新のDC値を取得（末尾の値）
            latest_slow_max_dc = slow_max_dc_values[-1]
            latest_slow_min_dc = slow_min_dc_values[-1]
            latest_fast_max_dc = fast_max_dc_values[-1]
            
            # 動的なperiodを計算 - DCの値をそのまま使用
            dynamic_max_slow = int(latest_slow_max_dc)
            dynamic_min_slow = int(latest_slow_min_dc)
            dynamic_max_fast = int(latest_fast_max_dc)
            
            # 最小値の制限
            if dynamic_max_slow < 2:
                dynamic_max_slow = 2
            if dynamic_min_slow < 2:
                dynamic_min_slow = 2
            if dynamic_max_fast < 2:
                dynamic_max_fast = 2
            
            return dynamic_max_slow, dynamic_min_slow, dynamic_max_fast
            
        except Exception as e:
            # エラーが発生した場合はログに記録し、デフォルト値を返す
            self.logger.error(f"動的Period計算中にエラー: {str(e)}")
            return self.slow_max_dc_max_output, self.slow_min_dc_max_output, self.fast_max_dc_max_output
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray], external_er: Optional[np.ndarray] = None) -> np.ndarray:
        """
        ZMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
            external_er: 外部から提供されるサイクル効率比（CER）
                サイクル効率比はCycleEfficiencyRatioクラスから提供される必要があります
        
        Returns:
            ZMAの値（ハイパースムーサーで平滑化されている場合はその結果、そうでなければ生のZMA値）
        """
        try:
            # サイクル効率比（CER）の検証
            if external_er is None:
                raise ValueError("サイクル効率比（CER）は必須です。external_erパラメータを指定してください")
            
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data, external_er)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # 指定されたソースタイプの価格データを取得
            prices = self.calculate_source_values(data, self.src_type)
            
            # データ長の検証
            data_length = len(prices)
            
            # NumPy配列に変換して計算を高速化
            prices_np = np.asarray(prices, dtype=np.float64)
            
            # 最大期間用ドミナントサイクルの計算
            max_dc_values = self.max_dc_detector.calculate(data)
            max_dc_np = np.asarray(max_dc_values, dtype=np.float64)
            
            # 最小期間用ドミナントサイクルの計算
            min_dc_values = self.min_dc_detector.calculate(data)
            min_dc_np = np.asarray(min_dc_values, dtype=np.float64)
            
            # 動的なperiod用ドミナントサイクル計算
            slow_max_dc_values = self.slow_max_dc_detector.calculate(data)
            slow_min_dc_values = self.slow_min_dc_detector.calculate(data)
            fast_max_dc_values = self.fast_max_dc_detector.calculate(data)
            
            # 動的なperiod値を計算
            dynamic_max_slow, dynamic_min_slow, dynamic_max_fast = self._calculate_dynamic_periods(data)
            
            # サイクル効率比（CER）を使用
            er = np.asarray(external_er, dtype=np.float64)
            # 外部CERの長さが一致するか確認
            if len(er) != data_length:
                raise ValueError(f"サイクル効率比の長さ({len(er)})がデータ長({data_length})と一致しません")
            
            # 動的なKAMAピリオドの計算（並列処理版）
            dynamic_kama_period = calculate_dynamic_kama_period(
                er,
                max_dc_np,
                min_dc_np
            )
            
            # 動的なfast/slow期間の計算（パラレル処理版）
            # 動的なperiodパラメータを使用
            fast_periods, slow_periods, fast_constants, slow_constants = calculate_dynamic_kama_constants(
                er,
                dynamic_max_slow,
                dynamic_min_slow,
                dynamic_max_fast,
                self.min_fast_period  # Fast期間の最小値は常に2で固定
            )
            
            # ZMAの計算（高速化版）
            z_ma_raw = calculate_z_ma(
                prices_np,
                er,
                len(er) // 10,  # 近似値として使用
                dynamic_kama_period,
                fast_constants,
                slow_constants
            )
            
            # ハイパースムーサーによる平滑化（オプション）
            if self.hyper_smooth_period > 0:
                # ハイパースムーサーを適用
                z_ma_smoothed = calculate_hyper_smoother_numba(z_ma_raw, self.hyper_smooth_period)
                
                # 出力用に平滑化済み値を使用
                z_ma_values = z_ma_smoothed
            else:
                # 平滑化しない場合は生の値をそのまま使用
                z_ma_values = z_ma_raw
            
            # 結果の保存（参照問題を避けるためコピーを作成）
            self._result = ZMAResult(
                values=np.copy(z_ma_values),
                raw_values=np.copy(z_ma_raw),
                er=np.copy(er),
                dynamic_kama_period=np.copy(dynamic_kama_period),
                dynamic_fast_period=np.copy(fast_periods),
                dynamic_slow_period=np.copy(slow_periods),
                dc_values=np.copy(max_dc_values),  # 最大期間用DCを保存
                period_dc_values=np.copy(fast_max_dc_values)  # 動的Fast最大用DCを保存（以前のperiod_dc_values）
            )
            
            self._values = z_ma_values
            return z_ma_values
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ZMA計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は前回の結果を維持する
            if self._result is None:
                # 初回エラー時は空の配列を返す
                return np.array([])
            return self._result.values
    
    def get_raw_values(self) -> np.ndarray:
        """
        ハイパースムーサーによる平滑化前の生ZMA値を取得する
        
        Returns:
            np.ndarray: 生のZMA値（平滑化前）
        """
        if self._result is None:
            return np.array([])
        return self._result.raw_values
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比の値を取得する
        
        Returns:
            np.ndarray: サイクル効率比の値（CER）
        """
        if self._result is None:
            return np.array([])
        return self._result.er
    
    def get_max_dc_values(self) -> np.ndarray:
        """
        最大期間用ドミナントサイクルの値を取得する
        
        Returns:
            np.ndarray: 最大期間用ドミナントサイクルの値
        """
        if self._result is None:
            return np.array([])
        return self._result.dc_values
    
    def get_min_dc_values(self) -> np.ndarray:
        """
        最小期間用ドミナントサイクルの値を取得する
        
        Returns:
            np.ndarray: 最小期間用ドミナントサイクルの値
        """
        # 最小期間用DCは結果に保存していないため、再計算する必要がある
        return self.min_dc_detector.get_values()
    
    def get_period_dc_values(self) -> np.ndarray:
        """
        動的Period用ドミナントサイクルの値を取得する
        
        Returns:
            np.ndarray: 動的Fast最大用ドミナントサイクルの値
        """
        if self._result is None:
            return np.array([])
        return self._result.period_dc_values
    
    def get_dynamic_periods(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        動的な期間の値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (KAMAピリオド, Fast期間, Slow期間)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
        return (
            self._result.dynamic_kama_period,
            self._result.dynamic_fast_period,
            self._result.dynamic_slow_period
        )
    
    def get_slow_max_dc_values(self) -> np.ndarray:
        """
        動的Slow最大用ドミナントサイクルの値を取得する
        
        Returns:
            np.ndarray: 動的Slow最大用ドミナントサイクルの値
        """
        # 検出器から直接値を取得
        return self.slow_max_dc_detector.get_values()
    
    def get_slow_min_dc_values(self) -> np.ndarray:
        """
        動的Slow最小用ドミナントサイクルの値を取得する
        
        Returns:
            np.ndarray: 動的Slow最小用ドミナントサイクルの値
        """
        # 検出器から直接値を取得
        return self.slow_min_dc_detector.get_values()
    
    def get_fast_max_dc_values(self) -> np.ndarray:
        """
        動的Fast最大用ドミナントサイクルの値を取得する
        
        Returns:
            np.ndarray: 動的Fast最大用ドミナントサイクルの値
        """
        # 検出器から直接値を取得
        return self.fast_max_dc_detector.get_values()
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        if self.use_dynamic_periods:
            self.max_dc_detector.reset()
            self.min_dc_detector.reset()
            self.slow_max_dc_detector.reset()
            self.slow_min_dc_detector.reset()
            self.fast_max_dc_detector.reset() 