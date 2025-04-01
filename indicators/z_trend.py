#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from .indicator import Indicator
from .z_atr import ZATR
from .cycle_efficiency_ratio import CycleEfficiencyRatio
from .ehlers_hody_dc import EhlersHoDyDC


@dataclass
class ZTrendResult:
    """Z_Trendの計算結果"""
    upper_band: np.ndarray  # 上側のバンド価格
    lower_band: np.ndarray  # 下側のバンド価格
    trend: np.ndarray       # トレンド方向（1=上昇トレンド、-1=下降トレンド）
    smooth_upper: np.ndarray  # 75パーセンタイルの平滑化値
    smooth_lower: np.ndarray  # 25パーセンタイルの平滑化値
    cer: np.ndarray          # サイクル効率比
    dynamic_multiplier: np.ndarray  # 動的乗数
    dynamic_percentile_length: np.ndarray  # 動的パーセンタイル期間
    z_atr: np.ndarray   # ZATR値
    max_dc: np.ndarray  # 最大パーセンタイル期間用のドミナントサイクル
    min_dc: np.ndarray  # 最小パーセンタイル期間用のドミナントサイクル
    max_mult_values: np.ndarray  # 動的に計算されたmax_multiplier値
    min_mult_values: np.ndarray  # 動的に計算されたmin_multiplier値


@njit(fastmath=True)
def calculate_dynamic_multiplier(cer: np.ndarray, max_mult: float, min_mult: float) -> np.ndarray:
    """
    サイクル効率比に基づいて動的な乗数を計算する（高速化版）
    
    Args:
        cer: サイクル効率比の配列
        max_mult: 最大乗数
        min_mult: 最小乗数
    
    Returns:
        動的な乗数の配列
    """
    # CERが高い（トレンドが強い）ほど乗数は小さく、
    # CERが低い（トレンドが弱い）ほど乗数は大きくなる
    multipliers = max_mult - np.abs(cer) * (max_mult - min_mult)
    return multipliers


@njit(fastmath=True)
def calculate_dynamic_max_multiplier(cer: np.ndarray, max_max_mult: float, min_max_mult: float) -> np.ndarray:
    """
    サイクル効率比に基づいて動的な最大ATR乗数を計算する（高速化版）
    
    Args:
        cer: サイクル効率比の配列
        max_max_mult: 最大乗数の最大値（例：8.0）
        min_max_mult: 最大乗数の最小値（例：3.0）
    
    Returns:
        動的な最大乗数の配列
    """
    # CERが低い（トレンドが弱い）ほど最大乗数は大きく、
    # CERが高い（トレンドが強い）ほど最大乗数は小さくなる
    max_multipliers = max_max_mult - np.abs(cer) * (max_max_mult - min_max_mult)
    return max_multipliers


@njit(fastmath=True)
def calculate_dynamic_min_multiplier(cer: np.ndarray, max_min_mult: float, min_min_mult: float) -> np.ndarray:
    """
    サイクル効率比に基づいて動的な最小ATR乗数を計算する（高速化版）
    
    Args:
        cer: サイクル効率比の配列
        max_min_mult: 最小乗数の最大値（例：1.5）
        min_min_mult: 最小乗数の最小値（例：0.3）
    
    Returns:
        動的な最小乗数の配列
    """
    # CERが低い（トレンドが弱い）ほど最小乗数は小さく、
    # CERが高い（トレンドが強い）ほど最小乗数は大きくなる
    min_multipliers = max_min_mult - np.abs(cer) * (max_min_mult - min_min_mult)
    return min_multipliers


@njit(fastmath=True, parallel=True)
def calculate_dynamic_percentile_length(
    cer: np.ndarray,
    max_length: np.ndarray,
    min_length: np.ndarray
) -> np.ndarray:
    """
    動的なパーセンタイル計算期間を計算する（高速化版）
    
    Args:
        cer: サイクル効率比の配列
        max_length: 最大期間の配列（ドミナントサイクルから計算）
        min_length: 最小期間の配列（ドミナントサイクルから計算）
    
    Returns:
        動的な期間の配列
    """
    # CERが高い（トレンドが強い）ほど期間は短く、
    # CERが低い（トレンドが弱い）ほど期間は長くなる
    periods = min_length + (1.0 - np.abs(cer)) * (max_length - min_length)
    return np.round(periods).astype(np.int32)


@njit(fastmath=True, parallel=True)
def calculate_dynamic_percentiles(
    high: np.ndarray,
    low: np.ndarray,
    dynamic_length: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    動的なパーセンタイルを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        dynamic_length: 動的な期間配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 25パーセンタイル（下限）と75パーセンタイル（上限）の配列
    """
    length = len(high)
    smooth_lower = np.zeros_like(high)
    smooth_upper = np.zeros_like(high)
    
    for i in prange(length):
        current_length = min(i + 1, int(dynamic_length[i]))
        if current_length < 2:
            current_length = 2  # 最低2データポイント必要
            
        start_idx = max(0, i - current_length + 1)
        high_window = high[start_idx:i+1]
        low_window = low[start_idx:i+1]
        
        sorted_high = np.sort(high_window)
        sorted_low = np.sort(low_window)
        
        k25 = max(0, min(len(sorted_low) - 1, int(np.ceil(25/100.0 * len(sorted_low))) - 1))
        k75 = max(0, min(len(sorted_high) - 1, int(np.ceil(75/100.0 * len(sorted_high))) - 1))
        
        smooth_lower[i] = sorted_low[k25]
        smooth_upper[i] = sorted_high[k75]
    
    return smooth_lower, smooth_upper


@njit(fastmath=True, parallel=True)
def calculate_z_trend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    smooth_upper: np.ndarray,
    smooth_lower: np.ndarray,
    z_atr: np.ndarray,
    dynamic_multiplier: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z_Trendを計算する（高速化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        smooth_upper: 75パーセンタイルの平滑化値
        smooth_lower: 25パーセンタイルの平滑化値
        z_atr: ZATRの配列
        dynamic_multiplier: 動的乗数の配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 上側バンド、下側バンド、トレンド方向の配列
    """
    length = len(close)
    
    # バンドの初期化
    upper_band = np.zeros(length, dtype=np.float64)
    lower_band = np.zeros(length, dtype=np.float64)
    trend = np.zeros(length, dtype=np.int8)
    
    # 最初の有効なインデックスを特定（NaNを避ける）
    start_idx = 0
    for i in range(length):
        if not np.isnan(z_atr[i]) and not np.isnan(smooth_upper[i]) and not np.isnan(smooth_lower[i]):
            start_idx = i
            break
    
    # 最初の値を設定
    if start_idx < length:
        upper_band[start_idx] = smooth_upper[start_idx] + dynamic_multiplier[start_idx] * z_atr[start_idx]
        lower_band[start_idx] = smooth_lower[start_idx] - dynamic_multiplier[start_idx] * z_atr[start_idx]
        trend[start_idx] = 1 if close[start_idx] > upper_band[start_idx] else -1
    
    # バンドとトレンドの計算
    for i in prange(start_idx + 1, length):
        if np.isnan(z_atr[i]) or np.isnan(smooth_upper[i]) or np.isnan(smooth_lower[i]):
            upper_band[i] = upper_band[i-1]
            lower_band[i] = lower_band[i-1]
            trend[i] = trend[i-1]
            continue
        
        # 新しいバンドの計算
        new_upper = smooth_upper[i] + dynamic_multiplier[i] * z_atr[i]
        new_lower = smooth_lower[i] - dynamic_multiplier[i] * z_atr[i]
        
        # トレンドに基づいてバンドを更新
        if trend[i-1] == 1:  # 上昇トレンド
            lower_band[i] = max(new_lower, lower_band[i-1])
            upper_band[i] = new_upper
            if close[i] < lower_band[i]:
                trend[i] = -1
            else:
                trend[i] = 1
        else:  # 下降トレンド
            upper_band[i] = min(new_upper, upper_band[i-1])
            lower_band[i] = new_lower
            if close[i] > upper_band[i]:
                trend[i] = 1
            else:
                trend[i] = -1
    
    return upper_band, lower_band, trend


class ZTrend(Indicator):
    """
    Z_Trend（ゼットトレンド）インジケーター
    
    アルファトレンドの改良版：
    - 効率比（ER）からサイクル効率比（CER）に変更
    - ATRをZATRに変更
    - サイクル検出器による動的パラメータ最適化
    - ドミナントサイクルに基づく動的パーセンタイル期間の自動調整
    - ATR乗数の最大値と最小値もCERに基づいて動的に調整
    
    特徴：
    - 25-75パーセンタイルによるレベル計算（動的期間）
    - ZATRによる洗練されたボラティリティ測定
    - サイクル効率比（CER）による全パラメータの動的最適化
    - 市場サイクルに自動適応する高度なトレンドフォロー性能
    """
    
    def __init__(
        self,
        cycle_detector_type: str = 'dudi_e',
        lp_period: int = 5,
        hp_period: int = 55,
        cycle_part: float = 0.5,
        
        # CERのドミナントサイクル検出器用パラメータ
        cer_max_cycle = 233,
        cer_min_cycle = 13,
        cer_max_output = 144,
        cer_min_output = 21,
        
        # 最大パーセンタイル期間用（長期）ドミナントサイクル検出器のパラメータ
        max_percentile_dc_cycle_part: float = 0.5,
        max_percentile_dc_max_cycle: int = 233,
        max_percentile_dc_min_cycle: int = 13,
        max_percentile_dc_max_output: int = 144,
        max_percentile_dc_min_output: int = 21,
        
        # 最小パーセンタイル期間用（短期）ドミナントサイクル検出器のパラメータ
        min_percentile_dc_cycle_part: float = 0.5,
        min_percentile_dc_max_cycle: int = 55,
        min_percentile_dc_min_cycle: int = 5,
        min_percentile_dc_max_output: int = 34,
        min_percentile_dc_min_output: int = 8,
        
        # ZATR用ドミナントサイクル検出器のパラメータ
        zatr_max_dc_cycle_part: float = 0.5,
        zatr_max_dc_max_cycle: int = 55,
        zatr_max_dc_min_cycle: int = 5,
        zatr_max_dc_max_output: int = 55,
        zatr_max_dc_min_output: int = 5,
        zatr_min_dc_cycle_part: float = 0.25,
        zatr_min_dc_max_cycle: int = 34,
        zatr_min_dc_min_cycle: int = 3,
        zatr_min_dc_max_output: int = 13,
        zatr_min_dc_min_output: int = 3,
        
        # パーセンタイル乗数
        max_percentile_cycle_mult: float = 0.5,  # 最大パーセンタイル期間のサイクル乗数
        min_percentile_cycle_mult: float = 0.25,  # 最小パーセンタイル期間のサイクル乗数
        
        
        # 動的乗数の範囲
        max_max_multiplier: float = 5.0,    # 最大乗数の最大値
        min_max_multiplier: float = 2.5,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        
        # その他の設定
        smoother_type: str = 'alma',   # 平滑化アルゴリズム（'alma'または'hyper'）
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: サイクル検出器の種類
                'dudi_dc' - 二重微分
                'hody_dc' - ホモダイン判別機（デフォルト）
                'phac_dc' - 位相累積
                'dudi_dce' - 拡張二重微分
                'hody_dce' - 拡張ホモダイン判別機
                'phac_dce' - 拡張位相累積
            lp_period: ローパスフィルターの期間（デフォルト: 13）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            
            cer_max_cycle: CER用ドミナントサイクル検出器の最大サイクル（デフォルト: 233）
            cer_min_cycle: CER用ドミナントサイクル検出器の最小サイクル（デフォルト: 13）
            cer_max_output: CER用ドミナントサイクル検出器の最大出力（デフォルト: 144）
            cer_min_output: CER用ドミナントサイクル検出器の最小出力（デフォルト: 21）
            
            max_percentile_dc_cycle_part: 最大パーセンタイル期間用DCのサイクル部分（デフォルト: 0.5）
            max_percentile_dc_max_cycle: 最大パーセンタイル期間用DCの最大サイクル（デフォルト: 233）
            max_percentile_dc_min_cycle: 最大パーセンタイル期間用DCの最小サイクル（デフォルト: 13）
            max_percentile_dc_max_output: 最大パーセンタイル期間用DCの最大出力（デフォルト: 144）
            max_percentile_dc_min_output: 最大パーセンタイル期間用DCの最小出力（デフォルト: 21）
            
            min_percentile_dc_cycle_part: 最小パーセンタイル期間用DCのサイクル部分（デフォルト: 0.5）
            min_percentile_dc_max_cycle: 最小パーセンタイル期間用DCの最大サイクル（デフォルト: 55）
            min_percentile_dc_min_cycle: 最小パーセンタイル期間用DCの最小サイクル（デフォルト: 5）
            min_percentile_dc_max_output: 最小パーセンタイル期間用DCの最大出力（デフォルト: 34）
            min_percentile_dc_min_output: 最小パーセンタイル期間用DCの最小出力（デフォルト: 8）
            
            zatr_max_dc_cycle_part: ZATR最大DCのサイクル部分（デフォルト: 0.5）
            zatr_max_dc_max_cycle: ZATR最大DCの最大サイクル（デフォルト: 55）
            zatr_max_dc_min_cycle: ZATR最大DCの最小サイクル（デフォルト: 5）
            zatr_max_dc_max_output: ZATR最大DCの最大出力（デフォルト: 55）
            zatr_max_dc_min_output: ZATR最大DCの最小出力（デフォルト: 5）
            zatr_min_dc_cycle_part: ZATR最小DCのサイクル部分（デフォルト: 0.25）
            zatr_min_dc_max_cycle: ZATR最小DCの最大サイクル（デフォルト: 34）
            zatr_min_dc_min_cycle: ZATR最小DCの最小サイクル（デフォルト: 3）
            zatr_min_dc_max_output: ZATR最小DCの最大出力（デフォルト: 13）
            zatr_min_dc_min_output: ZATR最小DCの最小出力（デフォルト: 3）
            
            max_percentile_cycle_mult: 最大パーセンタイル期間のサイクル乗数（デフォルト: 0.5）
            min_percentile_cycle_mult: 最小パーセンタイル期間のサイクル乗数（デフォルト: 0.25）
            
            max_multiplier: ATR乗数の最大値（レガシーパラメータ、デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（レガシーパラメータ、デフォルト: 1.0）
            
            max_max_multiplier: 最大乗数の最大値（デフォルト: 8.0）
            min_max_multiplier: 最大乗数の最小値（デフォルト: 3.0）
            max_min_multiplier: 最小乗数の最大値（デフォルト: 1.5）
            min_min_multiplier: 最小乗数の最小値（デフォルト: 0.3）
            
            smoother_type: 平滑化アルゴリズムのタイプ（デフォルト: 'alma'）
                'alma' - ALMA（Arnaud Legoux Moving Average）
                'hyper' - ハイパースムーサー（3段階平滑化）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
        """
        super().__init__(
            f"ZTrend({cycle_detector_type}, {max_percentile_cycle_mult}, {min_percentile_cycle_mult}, "
            f" {smoother_type})"
        )
        
        # 基本パラメータを保存
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        
        # CERパラメータを保存
        self.cer_max_cycle = cer_max_cycle
        self.cer_min_cycle = cer_min_cycle
        self.cer_max_output = cer_max_output
        self.cer_min_output = cer_min_output
        
        # 最大・最小パーセンタイル期間用DCパラメータを保存
        self.max_percentile_dc_cycle_part = max_percentile_dc_cycle_part
        self.max_percentile_dc_max_cycle = max_percentile_dc_max_cycle
        self.max_percentile_dc_min_cycle = max_percentile_dc_min_cycle
        self.max_percentile_dc_max_output = max_percentile_dc_max_output
        self.max_percentile_dc_min_output = max_percentile_dc_min_output
        
        self.min_percentile_dc_cycle_part = min_percentile_dc_cycle_part
        self.min_percentile_dc_max_cycle = min_percentile_dc_max_cycle
        self.min_percentile_dc_min_cycle = min_percentile_dc_min_cycle
        self.min_percentile_dc_max_output = min_percentile_dc_max_output
        self.min_percentile_dc_min_output = min_percentile_dc_min_output
        
        # ZATRパラメータを保存
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
        
        # その他のパラメータを保存
        self.max_percentile_cycle_mult = max_percentile_cycle_mult
        self.min_percentile_cycle_mult = min_percentile_cycle_mult
        
        # 動的乗数の範囲パラメータ
        self.max_max_multiplier = max_max_multiplier
        self.min_max_multiplier = min_max_multiplier
        self.max_min_multiplier = max_min_multiplier
        self.min_min_multiplier = min_min_multiplier
        
        self.smoother_type = smoother_type
        self.src_type = src_type
        
        # サイクル効率比（CER）のインスタンス化
        self.cycle_er = CycleEfficiencyRatio(
            detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_cycle=cer_max_cycle,
            min_cycle=cer_min_cycle,
            max_output=cer_max_output,
            min_output=cer_min_output,
            src_type=src_type
        )
        
        # ZATRのインスタンス化
        self.z_atr = ZATR(
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
        
        # 最大パーセンタイル期間用のドミナントサイクル検出器
        self.max_dc_detector = EhlersHoDyDC(
            cycle_part=max_percentile_dc_cycle_part,
            max_cycle=max_percentile_dc_max_cycle,
            min_cycle=max_percentile_dc_min_cycle,
            max_output=max_percentile_dc_max_output,
            min_output=max_percentile_dc_min_output,
            src_type=src_type
        )
        
        # 最小パーセンタイル期間用のドミナントサイクル検出器
        self.min_dc_detector = EhlersHoDyDC(
            cycle_part=min_percentile_dc_cycle_part,
            max_cycle=min_percentile_dc_max_cycle,
            min_cycle=min_percentile_dc_min_cycle,
            max_output=min_percentile_dc_max_output,
            min_output=min_percentile_dc_min_output,
            src_type=src_type
        )
        
        self._result = None
        self._data_hash = None  # データキャッシュ用ハッシュ
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            cols = ['high', 'low', 'close']
            data_hash = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            # NumPy配列の場合は全体をハッシュする
            data_hash = hash(tuple(map(tuple, data)))
        
        # パラメータ値を含めることで、同じデータでもパラメータが異なる場合に再計算する
        param_str = (
            f"{self.cycle_detector_type}_{self.lp_period}_{self.hp_period}_{self.cycle_part}_"
            f"{self.max_percentile_cycle_mult}_{self.min_percentile_cycle_mult}_"
            f"{self.smoother_type}_{self.src_type}_"
            f"{self.max_max_multiplier}_{self.min_max_multiplier}_{self.max_min_multiplier}_{self.min_min_multiplier}"
        )
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Z_Trendを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high', 'low', 'close'カラムが必要
        
        Returns:
            トレンド方向の配列（1=上昇、-1=下降）
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.trend
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    high = data[:, 1]  # high
                    low = data[:, 2]   # low
                    close = data[:, 3] # close
                else:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # データ長の検証
            data_length = len(high)
            if data_length < 10:  # 最低限必要なデータ量
                raise ValueError("データが少なすぎます（最低10ポイント必要）")
            
            # サイクル効率比（CER）の計算
            cer = self.cycle_er.calculate(data)
            
            # ドミナントサイクルの検出（最大・最小パーセンタイル期間用）
            max_dc = self.max_dc_detector.calculate(data)
            min_dc = self.min_dc_detector.calculate(data)
            
            # ドミナントサイクルから動的なパーセンタイル期間の最大値・最小値を計算
            max_percentile_length = np.round(max_dc * self.max_percentile_cycle_mult).astype(np.int32)
            min_percentile_length = np.round(min_dc * self.min_percentile_cycle_mult).astype(np.int32)
            
            # 値の範囲を制限
            max_percentile_length = np.clip(max_percentile_length, 21, 89)
            min_percentile_length = np.clip(min_percentile_length, 8, 34)
            
            # ZATRの計算（サイクル効率比を使用）
            # 金額ベースのATRを使用
            z_atr_values = self.z_atr.calculate(data, external_er=cer)
            z_atr_absolute = self.z_atr.get_absolute_atr()  # 金額ベースのZATR
            
            # 動的なパーセンタイル計算期間
            dynamic_percentile_length = calculate_dynamic_percentile_length(
                cer,
                max_percentile_length,
                min_percentile_length
            )
            
            # 動的な最大・最小乗数の計算
            max_mult_values = calculate_dynamic_max_multiplier(
                cer,
                self.max_max_multiplier,
                self.min_max_multiplier
            )
            
            min_mult_values = calculate_dynamic_min_multiplier(
                cer,
                self.max_min_multiplier,
                self.min_min_multiplier
            )
            
            # 動的乗数の計算
            dynamic_multiplier = calculate_dynamic_multiplier(
                cer,
                max_mult_values,
                min_mult_values
            )
            
            # 動的パーセンタイルの計算
            smooth_lower, smooth_upper = calculate_dynamic_percentiles(
                high, low, dynamic_percentile_length
            )
            
            # Z_Trendの計算
            upper_band, lower_band, trend = calculate_z_trend(
                high, low, close,
                smooth_upper, smooth_lower,
                z_atr_absolute,  # 金額ベースのZATR
                dynamic_multiplier
            )
            
            # 結果の保存
            self._result = ZTrendResult(
                upper_band=upper_band,
                lower_band=lower_band,
                trend=trend,
                smooth_upper=smooth_upper,
                smooth_lower=smooth_lower,
                cer=cer,
                dynamic_multiplier=dynamic_multiplier,
                dynamic_percentile_length=dynamic_percentile_length,
                z_atr=z_atr_absolute,
                max_dc=max_dc,
                min_dc=min_dc,
                max_mult_values=max_mult_values,
                min_mult_values=min_mult_values
            )
            
            # 基底クラスの要件を満たすため
            self._values = trend
            return trend
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ZTrend計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の配列を返す
            if self._result is None:
                # 初回エラー時は空の結果を作成
                empty_array = np.array([])
                self._result = ZTrendResult(
                    upper_band=empty_array,
                    lower_band=empty_array,
                    trend=empty_array,
                    smooth_upper=empty_array,
                    smooth_lower=empty_array,
                    cer=empty_array,
                    dynamic_multiplier=empty_array,
                    dynamic_percentile_length=empty_array,
                    z_atr=empty_array,
                    max_dc=empty_array,
                    min_dc=empty_array,
                    max_mult_values=empty_array,
                    min_mult_values=empty_array
                )
                self._values = empty_array
            
            return self._values
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Z_Trendのバンド値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (上限バンド, 下限バンド)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty
        return self._result.upper_band, self._result.lower_band
    
    def get_trend(self) -> np.ndarray:
        """
        トレンド方向を取得する
        
        Returns:
            np.ndarray: トレンド方向の配列（1=上昇、-1=下降）
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.trend
    
    def get_percentiles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        25パーセンタイルと75パーセンタイルの値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (25パーセンタイル, 75パーセンタイル)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty
        return self._result.smooth_lower, self._result.smooth_upper
    
    def get_cycle_er(self) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.cer
    
    def get_dynamic_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        動的パラメータの値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (動的乗数, 動的パーセンタイル期間)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty
        return self._result.dynamic_multiplier, self._result.dynamic_percentile_length
    
    def get_z_atr(self) -> np.ndarray:
        """
        ZATRの値を取得する
        
        Returns:
            np.ndarray: ZATRの値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.z_atr
    
    def get_dominant_cycles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        パーセンタイル計算に使用したドミナントサイクルを取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (最大パーセンタイル用DC, 最小パーセンタイル用DC)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty
        return self._result.max_dc, self._result.min_dc
    
    def get_dynamic_max_multiplier(self) -> np.ndarray:
        """
        動的な最大ATR乗数の値を取得する
        
        Returns:
            np.ndarray: 動的最大ATR乗数の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.max_mult_values
    
    def get_dynamic_min_multiplier(self) -> np.ndarray:
        """
        動的な最小ATR乗数の値を取得する
        
        Returns:
            np.ndarray: 動的最小ATR乗数の値
        """
        if self._result is None:
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
        self.cycle_er.reset() if hasattr(self.cycle_er, 'reset') else None
        self.z_atr.reset() if hasattr(self.z_atr, 'reset') else None
        self.max_dc_detector.reset() if hasattr(self.max_dc_detector, 'reset') else None
        self.min_dc_detector.reset() if hasattr(self.min_dc_detector, 'reset') else None 