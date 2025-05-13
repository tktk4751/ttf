#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import vectorize, njit, prange

from .indicator import Indicator
from .ehlers_unified_dc import EhlersUnifiedDC
from .c_atr import CATR
from .z_ma import ZMA
from .cycle_efficiency_ratio import CycleEfficiencyRatio


@dataclass
class CZChannelResult:
    """CZチャネルの計算結果"""
    middle: np.ndarray        # 中心線（ZMA）
    upper: np.ndarray         # 上限バンド
    lower: np.ndarray         # 下限バンド
    cer: np.ndarray           # Cycle Efficiency Ratio
    dynamic_multiplier: np.ndarray  # 動的ATR乗数
    c_atr: np.ndarray         # CATR値
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
def calculate_cz_channel(
    z_ma: np.ndarray,
    c_atr: np.ndarray,
    dynamic_multiplier: np.ndarray,
    use_percent: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CZチャネルを計算する（パラレル高速化版）
    
    Args:
        z_ma: ZMA値の配列
        c_atr: CATR値の配列（金額ベースまたはパーセントベース）
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
    valid_mask = ~(np.isnan(z_ma) | np.isnan(c_atr) | np.isnan(dynamic_multiplier))
    
    # バンド幅計算用の一時配列 - パラレル計算の準備
    band_width = np.zeros_like(z_ma, dtype=np.float64)
    
    # 並列計算で一度にバンド幅を計算
    for i in prange(length):
        if valid_mask[i]:
            if use_percent:
                # パーセントベースのATRを使用する場合、価格に対する比率
                band_width[i] = z_ma[i] * c_atr[i] * dynamic_multiplier[i]
            else:
                # 金額ベースのATRを使用する場合
                band_width[i] = c_atr[i] * dynamic_multiplier[i]
    
    # 一度にバンドを計算（並列処理）
    for i in prange(length):
        if valid_mask[i]:
            upper[i] = middle[i] + band_width[i]
            lower[i] = middle[i] - band_width[i]
    
    return middle, upper, lower


class CZChannel(Indicator):
    """
    CZチャネル（CZ Channel）インジケーター
    
    特徴:
    - ZMA（Z Moving Average）を中心線として使用
    - CATR（Cycle Average True Range）をボラティリティとして使用
    - サイクル効率比（CER）に基づく動的乗数でチャネル幅を調整
    - トレンドの強さに応じて適応的にバンド幅を変化
    
    使用方法:
    - トレンドフォロー戦略におけるエントリー・エグジットポイントの決定
    - サポート・レジスタンスレベルの特定
    - 価格アクションの分析とボラティリティの視覚化
    - 相場の状態（トレンド/レンジ）の判断
    """
    
    def __init__(
        self,
        # 基本パラメータ
        detector_type: str = 'phac_e',
        cer_detector_type: str = None,  # CER用の検出器タイプ
        lp_period: int = 5,
        hp_period: int = 55,
        cycle_part: float = 0.7,
        max_multiplier: float = 7.0,  # 固定乗数を使用する場合
        min_multiplier: float = 1.0,  # 固定乗数を使用する場合
        # 動的乗数の範囲パラメータ（固定乗数の代わりに動的乗数を使用する場合）
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 6.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        smoother_type: str = 'alma',  # 'alma' または 'hyper'
        src_type: str = 'hlc3',       # 'open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4'
        
        # ZMA用パラメータ
        zma_max_dc_cycle_part: float = 0.5,     # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_cycle: int = 100,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_cycle: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_output: int = 120,       # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_output: int = 22,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_lp_period: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用LPピリオド
        zma_max_dc_hp_period: int = 55,         # ZMA: 最大期間用ドミナントサイクル計算用HPピリオド
        
        zma_min_dc_cycle_part: float = 0.25,    # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_cycle: int = 55,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_cycle: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_output: int = 13,        # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_output: int = 3,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_lp_period: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用LPピリオド
        zma_min_dc_hp_period: int = 34,         # ZMA: 最小期間用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Slow最大用パラメータ
        zma_slow_max_dc_cycle_part: float = 0.5,
        zma_slow_max_dc_max_cycle: int = 144,
        zma_slow_max_dc_min_cycle: int = 5,
        zma_slow_max_dc_max_output: int = 89,
        zma_slow_max_dc_min_output: int = 22,
        zma_slow_max_dc_lp_period: int = 5,      # ZMA: Slow最大用ドミナントサイクル計算用LPピリオド
        zma_slow_max_dc_hp_period: int = 55,     # ZMA: Slow最大用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Slow最小用パラメータ
        zma_slow_min_dc_cycle_part: float = 0.5,
        zma_slow_min_dc_max_cycle: int = 89,
        zma_slow_min_dc_min_cycle: int = 5,
        zma_slow_min_dc_max_output: int = 21,
        zma_slow_min_dc_min_output: int = 8,
        zma_slow_min_dc_lp_period: int = 5,      # ZMA: Slow最小用ドミナントサイクル計算用LPピリオド
        zma_slow_min_dc_hp_period: int = 34,     # ZMA: Slow最小用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Fast最大用パラメータ
        zma_fast_max_dc_cycle_part: float = 0.5,
        zma_fast_max_dc_max_cycle: int = 55,
        zma_fast_max_dc_min_cycle: int = 5,
        zma_fast_max_dc_max_output: int = 15,
        zma_fast_max_dc_min_output: int = 3,
        zma_fast_max_dc_lp_period: int = 5,      # ZMA: Fast最大用ドミナントサイクル計算用LPピリオド
        zma_fast_max_dc_hp_period: int = 21,     # ZMA: Fast最大用ドミナントサイクル計算用HPピリオド
        
        zma_min_fast_period: int = 2,           # ZMA: 速い移動平均の最小期間（常に2で固定）
        zma_hyper_smooth_period: int = 0,       # ZMA: ハイパースムーサーの平滑化期間
        
        # CATR用パラメータ
        catr_detector_type: str = 'hody',
        catr_cycle_part: float = 0.5,
        catr_lp_period: int = 5,
        catr_hp_period: int = 55,
        catr_max_cycle: int = 55,
        catr_min_cycle: int = 5,
        catr_max_output: int = 34,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'alma'
    ):
        """初期化"""
        super().__init__(f"CZChannel({detector_type}, {max_max_multiplier}, {min_min_multiplier}, {smoother_type})")
        
        # 基本パラメータの設定
        self.detector_type = detector_type
        self.cer_detector_type = cer_detector_type or detector_type
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
        
        # ZMA用パラメータの設定
        self.zma_params = {
            'max_dc': {
                'cycle_part': zma_max_dc_cycle_part,
                'max_cycle': zma_max_dc_max_cycle,
                'min_cycle': zma_max_dc_min_cycle,
                'max_output': zma_max_dc_max_output,
                'min_output': zma_max_dc_min_output,
                'lp_period': zma_max_dc_lp_period,
                'hp_period': zma_max_dc_hp_period
            },
            'min_dc': {
                'cycle_part': zma_min_dc_cycle_part,
                'max_cycle': zma_min_dc_max_cycle,
                'min_cycle': zma_min_dc_min_cycle,
                'max_output': zma_min_dc_max_output,
                'min_output': zma_min_dc_min_output,
                'lp_period': zma_min_dc_lp_period,
                'hp_period': zma_min_dc_hp_period
            },
            'slow_max_dc': {
                'cycle_part': zma_slow_max_dc_cycle_part,
                'max_cycle': zma_slow_max_dc_max_cycle,
                'min_cycle': zma_slow_max_dc_min_cycle,
                'max_output': zma_slow_max_dc_max_output,
                'min_output': zma_slow_max_dc_min_output,
                'lp_period': zma_slow_max_dc_lp_period,
                'hp_period': zma_slow_max_dc_hp_period
            },
            'slow_min_dc': {
                'cycle_part': zma_slow_min_dc_cycle_part,
                'max_cycle': zma_slow_min_dc_max_cycle,
                'min_cycle': zma_slow_min_dc_min_cycle,
                'max_output': zma_slow_min_dc_max_output,
                'min_output': zma_slow_min_dc_min_output,
                'lp_period': zma_slow_min_dc_lp_period,
                'hp_period': zma_slow_min_dc_hp_period
            },
            'fast_max_dc': {
                'cycle_part': zma_fast_max_dc_cycle_part,
                'max_cycle': zma_fast_max_dc_max_cycle,
                'min_cycle': zma_fast_max_dc_min_cycle,
                'max_output': zma_fast_max_dc_max_output,
                'min_output': zma_fast_max_dc_min_output,
                'lp_period': zma_fast_max_dc_lp_period,
                'hp_period': zma_fast_max_dc_hp_period
            },
            'min_fast_period': zma_min_fast_period,
            'hyper_smooth_period': zma_hyper_smooth_period
        }
        
        # CATR用パラメータの設定
        self.catr_params = {
            'detector_type': catr_detector_type,
            'cycle_part': catr_cycle_part,
            'lp_period': catr_lp_period,
            'hp_period': catr_hp_period,
            'max_cycle': catr_max_cycle,
            'min_cycle': catr_min_cycle,
            'max_output': catr_max_output,
            'min_output': catr_min_output,
            'smoother_type': catr_smoother_type
        }
        
        # インジケーターのインスタンス化
        self.zma = ZMA(
            # 基本パラメータ
            detector_type=self.detector_type,
            src_type=self.src_type,
            use_dynamic_periods=True,  # 動的期間を使用
            
            # 最大期間用ドミナントサイクルパラメータ
            max_dc_cycle_part=self.zma_params['max_dc']['cycle_part'],
            max_dc_max_cycle=self.zma_params['max_dc']['max_cycle'],
            max_dc_min_cycle=self.zma_params['max_dc']['min_cycle'],
            max_dc_max_output=self.zma_params['max_dc']['max_output'],
            max_dc_min_output=self.zma_params['max_dc']['min_output'],
            max_dc_lp_period=self.zma_params['max_dc']['lp_period'],
            max_dc_hp_period=self.zma_params['max_dc']['hp_period'],
            
            # 最小期間用ドミナントサイクルパラメータ
            min_dc_cycle_part=self.zma_params['min_dc']['cycle_part'],
            min_dc_max_cycle=self.zma_params['min_dc']['max_cycle'],
            min_dc_min_cycle=self.zma_params['min_dc']['min_cycle'],
            min_dc_max_output=self.zma_params['min_dc']['max_output'],
            min_dc_min_output=self.zma_params['min_dc']['min_output'],
            min_dc_lp_period=self.zma_params['min_dc']['lp_period'],
            min_dc_hp_period=self.zma_params['min_dc']['hp_period'],
            
            # 動的Slow最大用パラメータ
            slow_max_dc_cycle_part=self.zma_params['slow_max_dc']['cycle_part'],
            slow_max_dc_max_cycle=self.zma_params['slow_max_dc']['max_cycle'],
            slow_max_dc_min_cycle=self.zma_params['slow_max_dc']['min_cycle'],
            slow_max_dc_max_output=self.zma_params['slow_max_dc']['max_output'],
            slow_max_dc_min_output=self.zma_params['slow_max_dc']['min_output'],
            slow_max_dc_lp_period=self.zma_params['slow_max_dc']['lp_period'],
            slow_max_dc_hp_period=self.zma_params['slow_max_dc']['hp_period'],
            
            # 動的Slow最小用パラメータ
            slow_min_dc_cycle_part=self.zma_params['slow_min_dc']['cycle_part'],
            slow_min_dc_max_cycle=self.zma_params['slow_min_dc']['max_cycle'],
            slow_min_dc_min_cycle=self.zma_params['slow_min_dc']['min_cycle'],
            slow_min_dc_max_output=self.zma_params['slow_min_dc']['max_output'],
            slow_min_dc_min_output=self.zma_params['slow_min_dc']['min_output'],
            slow_min_dc_lp_period=self.zma_params['slow_min_dc']['lp_period'],
            slow_min_dc_hp_period=self.zma_params['slow_min_dc']['hp_period'],
            
            # 動的Fast最大用パラメータ
            fast_max_dc_cycle_part=self.zma_params['fast_max_dc']['cycle_part'],
            fast_max_dc_max_cycle=self.zma_params['fast_max_dc']['max_cycle'],
            fast_max_dc_min_cycle=self.zma_params['fast_max_dc']['min_cycle'],
            fast_max_dc_max_output=self.zma_params['fast_max_dc']['max_output'],
            fast_max_dc_min_output=self.zma_params['fast_max_dc']['min_output'],
            fast_max_dc_lp_period=self.zma_params['fast_max_dc']['lp_period'],
            fast_max_dc_hp_period=self.zma_params['fast_max_dc']['hp_period'],
            
            # その他のパラメータ
            min_fast_period=self.zma_params['min_fast_period'],
            hyper_smooth_period=self.zma_params['hyper_smooth_period']
        )
        
        self.catr = CATR(**self.catr_params)
        
        # CER検出器の設定
        self.cer_detector = EhlersUnifiedDC(
            detector_type=self.cer_detector_type,
            lp_period=self.lp_period,
            hp_period=self.hp_period,
            cycle_part=self.cycle_part,
            max_cycle=self.catr_params['max_cycle'],
            min_cycle=self.catr_params['min_cycle'],
            max_output=self.catr_params['max_output'],
            min_output=self.catr_params['min_output']
        ) 

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> CZChannelResult:
        """
        CZチャネルの計算を実行する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'open', 'high', 'low', 'close'カラムが必要
                NumPy配列の場合、[open, high, low, close]の4列が必要
        
        Returns:
            CZChannelResult: 計算結果を含むデータクラスのインスタンス
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    raise ValueError("DataFrameには'open', 'high', 'low', 'close'カラムが必要です")
                df = data
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
            
            # ZMAの計算
            z_ma = self.zma.calculate(df)
            
            # CATRの計算
            c_atr = self.catr.calculate(df).values
            
            # CERの計算
            cer = self.cer_detector.calculate_cer(df)
            
            # 動的乗数の計算
            max_mult_values = calculate_dynamic_max_multiplier(
                cer=cer,
                max_max_mult=self.max_max_multiplier,
                min_max_mult=self.min_max_multiplier
            )
            
            min_mult_values = calculate_dynamic_min_multiplier(
                cer=cer,
                max_min_mult=self.max_min_multiplier,
                min_min_mult=self.min_min_multiplier
            )
            
            dynamic_multiplier = calculate_dynamic_multiplier_vec(
                cer=cer,
                max_mult=max_mult_values,
                min_mult=min_mult_values
            )
            
            # チャネルの計算
            middle, upper, lower = calculate_cz_channel(
                z_ma=z_ma,
                c_atr=c_atr,
                dynamic_multiplier=dynamic_multiplier,
                use_percent=False
            )
            
            # 結果を返す
            return CZChannelResult(
                middle=middle,
                upper=upper,
                lower=lower,
                cer=cer,
                dynamic_multiplier=dynamic_multiplier,
                c_atr=c_atr,
                max_mult_values=max_mult_values,
                min_mult_values=min_mult_values
            )
            
        except Exception as e:
            self.logger.error(f"CZChannel計算中にエラー: {str(e)}")
            # エラー時は空の結果を返す
            empty = np.array([])
            return CZChannelResult(
                middle=empty,
                upper=empty,
                lower=empty,
                cer=empty,
                dynamic_multiplier=empty,
                c_atr=empty,
                max_mult_values=empty,
                min_mult_values=empty
            ) 