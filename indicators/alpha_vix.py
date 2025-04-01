#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .alpha_atr import AlphaATR
from .alpha_volatility import AlphaVolatility
from .alpha_filter import AlphaFilter
from .alpha_choppiness import AlphaChoppiness
from .efficiency_ratio import EfficiencyRatio


@dataclass
class AlphaVIXResult:
    """AlphaVIXの計算結果"""
    values: np.ndarray         # VIXの値（%ベース）
    absolute_values: np.ndarray # VIXの値（金額ベース）
    atr_values: np.ndarray     # ATR値（%ベース）
    vol_values: np.ndarray     # ボラティリティ値（%ベース）
    atr_abs_values: np.ndarray # ATR値（金額ベース）
    vol_abs_values: np.ndarray # ボラティリティ値（金額ベース）
    er: np.ndarray             # 効率比
    filter_values: Optional[np.ndarray] = None  # アルファフィルター値（使用時のみ）


@jit(nopython=True)
def calculate_simple_average(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    2つの配列の単純平均を計算する（高速化版）
    
    Args:
        arr1: 最初の配列
        arr2: 2番目の配列
    
    Returns:
        単純平均の配列
    """
    result = np.zeros_like(arr1)
    for i in range(len(arr1)):
        # 両方の値が有効な場合のみ計算
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            # 単純平均 = (a + b) / 2
            result[i] = (arr1[i] + arr2[i]) / 2.0
    
    return result


@jit(nopython=True)
def calculate_weighted_average(arr1: np.ndarray, arr2: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    2つの配列の加重平均を計算する（高速化版）
    
    Args:
        arr1: 最初の配列
        arr2: 2番目の配列
        weights: arr1の重み（0-1の範囲）。arr2の重みは (1-weights)
    
    Returns:
        加重平均の配列
    """
    result = np.zeros_like(arr1)
    for i in range(len(arr1)):
        # 両方の値が有効な場合のみ計算
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            # 加重平均 = a * w + b * (1-w)
            w = weights[i]
            if w < 0.0:
                w = 0.0
            elif w > 1.0:
                w = 1.0
            result[i] = arr1[i] * w + arr2[i] * (1.0 - w)
    
    return result


@jit(nopython=True)
def calculate_maximum(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    2つの配列の各要素の最大値を計算する（高速化版）
    
    Args:
        arr1: 最初の配列
        arr2: 2番目の配列
    
    Returns:
        最大値の配列
    """
    result = np.zeros_like(arr1)
    for i in range(len(arr1)):
        # 両方の値が有効な場合のみ計算
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            # 最大値 = max(a, b)
            result[i] = max(arr1[i], arr2[i])
    
    return result


@jit(nopython=True)
def calculate_harmonic_mean(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    2つの配列の調和平均を計算する（高速化版）
    
    Args:
        arr1: 最初の配列
        arr2: 2番目の配列
    
    Returns:
        調和平均の配列
    """
    result = np.zeros_like(arr1)
    for i in range(len(arr1)):
        # 両方の値が有効な場合のみ計算
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]) and arr1[i] > 0 and arr2[i] > 0:
            # 調和平均 = 2 / (1/a + 1/b)
            result[i] = 2.0 / (1.0/arr1[i] + 1.0/arr2[i])
    
    return result


@jit(nopython=True)
def calculate_logarithmic_mean(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    2つの配列の対数平均を計算する（高速化版）
    
    Args:
        arr1: 最初の配列
        arr2: 2番目の配列
    
    Returns:
        対数平均の配列
    """
    result = np.zeros_like(arr1)
    for i in range(len(arr1)):
        # 両方の値が有効かつ正の場合のみ計算
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]) and arr1[i] > 0 and arr2[i] > 0:
            # 対数平均 = exp((log(a) + log(b)) / 2)
            result[i] = np.exp((np.log(arr1[i]) + np.log(arr2[i])) / 2.0)
    
    return result


@jit(nopython=True)
def calculate_root_mean_square(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    2つの配列の二乗平均平方根（RMS）を計算する（高速化版）
    
    Args:
        arr1: 最初の配列
        arr2: 2番目の配列
    
    Returns:
        二乗平均平方根（RMS）の配列
    """
    result = np.zeros_like(arr1)
    for i in range(len(arr1)):
        # 両方の値が有効な場合のみ計算
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            # 二乗平均平方根 = sqrt((a^2 + b^2) / 2)
            result[i] = np.sqrt((arr1[i]**2 + arr2[i]**2) / 2.0)
    
    return result


@jit(nopython=True)
def calculate_adaptive_weighted_average(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    2つの配列の二乗和に基づく適応的加重平均を計算する（高速化版）
    
    重みは各値の二乗に比例して割り当てられる（より大きい値により大きな重みを与える）
    
    Args:
        arr1: 最初の配列
        arr2: 2番目の配列
    
    Returns:
        適応的加重平均の配列
    """
    result = np.zeros_like(arr1)
    for i in range(len(arr1)):
        # 両方の値が有効な場合のみ計算
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            # 各値の二乗を計算
            sq1 = arr1[i] * arr1[i]
            sq2 = arr2[i] * arr2[i]
            
            # 二乗和の計算
            sq_sum = sq1 + sq2
            
            # 二乗和が0でない場合のみ重み付けを適用
            if sq_sum > 0:
                # 重みを計算（それぞれの二乗を二乗和で割る）
                w1 = sq1 / sq_sum
                w2 = sq2 / sq_sum
                
                # 適応的加重平均 = a * w1 + b * w2
                result[i] = arr1[i] * w1 + arr2[i] * w2
            else:
                # 二乗和が0の場合は単純に0を設定
                result[i] = 0.0
    
    return result


@jit(nopython=True)
def calculate_exponential_weighted_average(arr1: np.ndarray, arr2: np.ndarray, bias: float = 0.5) -> np.ndarray:
    """
    2つの配列の指数加重平均を計算する（高速化版）
    
    Args:
        arr1: 最初の配列
        arr2: 2番目の配列
        bias: 指数バイアス(0-1)。0.5はATRとボラティリティを等しく重み付け。
              0に近いほどATRに、1に近いほどボラティリティに重みが増加。
    
    Returns:
        指数加重平均の配列
    """
    result = np.zeros_like(arr1)
    for i in range(len(arr1)):
        # 両方の値が有効な場合のみ計算
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            # 正規化した値の計算（0-1の範囲に）
            max_val = max(arr1[i], arr2[i])
            if max_val > 0:
                norm1 = arr1[i] / max_val
                norm2 = arr2[i] / max_val
                
                # 指数加重平均 = max_val * (norm1^(1-bias) * norm2^bias)
                # bias=0.5の場合は幾何平均と同等
                result[i] = max_val * (norm1 ** (1.0 - bias)) * (norm2 ** bias)
            else:
                result[i] = 0.0
    
    return result


@jit(nopython=True)
def calculate_filter_weighted_average(arr1: np.ndarray, arr2: np.ndarray, filter_values: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """
    アルファフィルター値を使用して2つの配列の適応的加重平均を計算する（高速化版）
    
    Args:
        arr1: 最初の配列 (ATR)
        arr2: 2番目の配列 (Volatility)
        filter_values: アルファフィルター値（0-1の範囲）
        alpha: フィルター値の増幅係数（デフォルト: 2.0）
    
    Returns:
        アルファフィルターによる加重平均の配列
    """
    result = np.zeros_like(arr1)
    for i in range(len(arr1)):
        # すべての値が有効な場合のみ計算
        if (not np.isnan(arr1[i]) and not np.isnan(arr2[i]) and 
            not np.isnan(filter_values[i])):
            
            # フィルター値の増幅（alphaが大きいほど差が強調される）
            # 0-1の値を保持するため、この式を使用: x^alpha / (x^alpha + (1-x)^alpha)
            filter_val = filter_values[i]
            if filter_val < 0.0:
                filter_val = 0.0
            elif filter_val > 1.0:
                filter_val = 1.0
                
            # フィルター値の増幅
            if filter_val == 0.0 or filter_val == 1.0:
                enhanced_filter = filter_val
            else:
                filter_pow = filter_val ** alpha
                one_minus_filter_pow = (1.0 - filter_val) ** alpha
                enhanced_filter = filter_pow / (filter_pow + one_minus_filter_pow)
            
            # トレンド強度（フィルター値）に基づいた重み付け：
            # - トレンドが強い（filter_val が高い）場合、ATRにより大きな重みを与える
            # - レンジ相場（filter_val が低い）場合、ボラティリティにより大きな重みを与える
            atr_weight = enhanced_filter
            vol_weight = 1.0 - enhanced_filter
            
            # 加重平均の計算
            result[i] = arr1[i] * atr_weight + arr2[i] * vol_weight
    
    return result


@jit(nopython=True)
def calculate_simplified_filter(chop_values: np.ndarray, er_values: np.ndarray, chop_weight: float = 0.6) -> np.ndarray:
    """
    チョピネスと効率比から簡易フィルター値を計算する（高速化版）
    
    Args:
        chop_values: アルファチョピネスの値（0-1の範囲、1に近いほどレンジ相場）
        er_values: 効率比の値（0-1の範囲、1に近いほど強いトレンド）
        chop_weight: チョピネスの重み（デフォルト: 0.6）
    
    Returns:
        簡易フィルター値（0-1の範囲、1に近いほど強いトレンド）
    """
    result = np.zeros_like(chop_values)
    er_weight = 1.0 - chop_weight
    
    for i in range(len(chop_values)):
        # 両方の値が有効な場合のみ計算
        if not np.isnan(chop_values[i]) and not np.isnan(er_values[i]):
            # チョピネスは逆転させる（1-chop）：チョピネスが低いほどトレンドが強い
            inverted_chop = 1.0 - chop_values[i]
            
            # 高度な組み合わせ方法 - 4種類の計算方法を実装
            
            # 1. 重み付け線形結合: チョピネスとERの重み付け平均（基本的な方法）
            linear_combine = chop_weight * inverted_chop + er_weight * er_values[i]
            
            # 2. 幾何平均による非線形結合: 乗算効果でより強いシグナルを生成
            # どちらか一方が0に近いと結果も0に近くなる（AND論理に近い）
            if inverted_chop > 0 and er_values[i] > 0:
                geometric_mean = (inverted_chop ** chop_weight) * (er_values[i] ** er_weight)
            else:
                geometric_mean = 0.0
            
            # 3. 調和重み付け: 小さい値により敏感（保守的な判断に適する）
            if inverted_chop > 0.01 and er_values[i] > 0.01:  # ゼロ除算防止
                harmonic_weight = 1.0 / ((chop_weight / inverted_chop) + (er_weight / er_values[i]))
            else:
                harmonic_weight = 0.0
            
            # 4. 二次形式: チョピネスとERの積を加えることで、両方が高い場合に追加ボーナス
            quadratic = linear_combine + (inverted_chop * er_values[i] * 0.5)
            quadratic = min(1.0, quadratic)  # 1.0を超えないように制限
            
            # 最終的な値として二次形式を使用（最も安定したシグナルを提供）
            result[i] = quadratic
            
            # シグモイド関数による強調（結果を0-1の範囲に保ちながら中間の値を強調）
            if result[i] != 0.5:
                # 係数を8.0から6.0に変更して、より滑らかな遷移を実現
                sigmoid = 1.0 / (1.0 + np.exp(-6.0 * (result[i] - 0.5)))
                result[i] = sigmoid
    
    return result


class AlphaVIX(Indicator):
    """
    アルファVIX（Alpha VIX）インジケーター
    
    特徴:
    - アルファATRとアルファボラティリティを二乗平均平方根(RMS)法で統合（sqrt((ATR^2 + Volatility^2) / 2)）
    - 金額ベースと%ベースの両方の値を提供
    - 効率比（ER）に基づいて期間を動的に調整（内部のATRとボラティリティ計算で）
    - より包括的なボラティリティ測定
    - アルファフィルターによる市場状態に基づく適応的重み付け（フィルターモード使用時）
    
    使用方法:
    - 市場のボラティリティレベルの総合的な評価
    - ボラティリティに基づいたポジションサイジングとリスク管理
    - 相場状況の判断（高ボラティリティ/低ボラティリティ環境の識別）
    - 異なる価格帯の銘柄間でのボラティリティ比較（%ベース）
    
    注意:
    - 有効な組み合わせ方法:
      1. 単純平均: (ATR + Volatility) / 2
      2. 加重平均: (ATR * w + Volatility * (1-w)) （wはER値）
      3. 最大値: max(ATR, Volatility) （保守的なリスク管理に適切）
      4. 調和平均: 2 / (1/ATR + 1/Volatility) （極端な値の影響を軽減）
      5. 対数平均: exp((log(ATR) + log(Volatility)) / 2) （乗法的な性質を持つ）
      6. 二乗平均平方根: sqrt((ATR^2 + Volatility^2) / 2) (ユークリッド距離の特性を持つ)
      7. 適応的加重平均: 二乗和に基づいて重みを動的に調整（大きい値により大きな重みを与える）
      8. 指数加重平均: 正規化された値の指数重み付け（バイアスパラメータで調整可能）
      9. フィルター加重平均: アルファフィルターの値に基づいて重み付け（市場状態に応じた適応的調整）
    """
    
    # 計算モード定数
    MODE_SIMPLE_AVERAGE = 'simple'        # 単純平均
    MODE_WEIGHTED_AVERAGE = 'weighted'    # 加重平均
    MODE_MAXIMUM = 'maximum'              # 最大値
    MODE_HARMONIC = 'harmonic'            # 調和平均
    MODE_LOGARITHMIC = 'logarithmic'      # 対数平均
    MODE_ROOT_MEAN_SQUARE = 'rms'         # 二乗平均平方根
    MODE_ADAPTIVE_WEIGHTED = 'adaptive'   # 適応的加重平均
    MODE_EXPONENTIAL_WEIGHTED = 'exponential'  # 指数加重平均
    MODE_FILTER_WEIGHTED = 'filter'       # フィルター加重平均
    
    def __init__(
        self,
        er_period: int = 21,
        max_period: int = 89,
        min_period: int = 13,
        smoothing_period: int = 14,
        mode: str = 'rms',  # デフォルトをrmsに変更
        exp_bias: float = 0.5,  # 指数加重平均のバイアス
        filter_alpha: float = 2.0,  # フィルター値の増幅係数
        use_filter: bool = False,  # アルファフィルターを使用するかどうか
        use_simplified_filter: bool = True,  # シンプルなフィルターを使用するかどうか
        # アルファフィルターのパラメータ
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        max_adx_period: int = 21,
        min_adx_period: int = 5,
        max_threshold: float = 0.55,
        min_threshold: float = 0.45,
        # シンプルフィルターのパラメータ
        chop_weight: float = 0.6,  # チョピネスの重み（0-1）
        filter_combination: str = 'quadratic'  # フィルター組み合わせ方法
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_period: VIX計算用の各インジケーターの最大期間（デフォルト: 89）
            min_period: VIX計算用の各インジケーターの最小期間（デフォルト: 13）
            smoothing_period: ボラティリティの平滑化期間（デフォルト: 14）
            mode: 計算モード（デフォルト: 'rms'）
                - 'simple': 単純平均 (ATR + Volatility) / 2
                - 'weighted': 加重平均 ATR * w + Volatility * (1-w)、wはER値
                - 'maximum': 最大値 max(ATR, Volatility)
                - 'harmonic': 調和平均 2 / (1/ATR + 1/Volatility)
                - 'logarithmic': 対数平均 exp((log(ATR) + log(Volatility)) / 2)
                - 'rms': 二乗平均平方根 sqrt((ATR^2 + Volatility^2) / 2)
                - 'adaptive': 適応的加重平均（二乗和による重み付け）
                - 'exponential': 指数加重平均（バイアスパラメータで調整）
                - 'filter': フィルター加重平均（アルファフィルターによる重み付け）
            exp_bias: 指数加重平均のバイアス（0-1）。0.5はATRとボラティリティを等しく重み付け。
                      0に近いほどATRに、1に近いほどボラティリティに重みを増加。
            filter_alpha: フィルター値の増幅係数（デフォルト: 2.0）。大きいほどフィルター値の効果が強調される。
            use_filter: アルファフィルターを使用するかどうか（デフォルト: False）
            use_simplified_filter: シンプルなフィルターを使用するかどうか（デフォルト: True）
                ADXを除外し、チョピネスとERだけを使用した軽量なフィルター
            max_chop_period: アルファチョピネス期間の最大値（デフォルト: 55）
            min_chop_period: アルファチョピネス期間の最小値（デフォルト: 8）
            max_adx_period: アルファADX期間の最大値（デフォルト: 21） - シンプルフィルターでは未使用
            min_adx_period: アルファADX期間の最小値（デフォルト: 5） - シンプルフィルターでは未使用
            max_threshold: しきい値の最大値（デフォルト: 0.55） - シンプルフィルターでは未使用
            min_threshold: しきい値の最小値（デフォルト: 0.45） - シンプルフィルターでは未使用
            chop_weight: チョピネスの重み（0-1）（デフォルト: 0.6）。シンプルフィルターでのみ使用。
                        高いほどチョピネスを重視、低いほど効率比を重視。
            filter_combination: フィルター値の組み合わせ方法（デフォルト: 'quadratic'）
                - 'linear': 単純な重み付け平均（基本的）
                - 'geometric': 幾何平均による非線形結合（乗算効果）
                - 'harmonic': 調和重み付け（小さい値に敏感）
                - 'quadratic': 二次形式（両指標が高い場合にボーナス）
        """
        # modeが'filter'の場合は自動的にuse_filterをTrueに設定
        if mode == self.MODE_FILTER_WEIGHTED:
            use_filter = True
            
        super().__init__(
            f"AlphaVIX({er_period}, {max_period}, {min_period}, {smoothing_period}, {mode})"
        )
        self.er_period = er_period
        self.max_period = max_period
        self.min_period = min_period
        self.smoothing_period = smoothing_period
        self.mode = mode
        self.exp_bias = max(0.0, min(1.0, exp_bias))  # 0-1の範囲に制限
        self.filter_alpha = filter_alpha
        self.use_filter = use_filter
        self.use_simplified_filter = use_simplified_filter
        self.chop_weight = max(0.0, min(1.0, chop_weight))  # 0-1の範囲に制限
        self.filter_combination = filter_combination
        
        # 組み合わせ方法の検証
        valid_combinations = ['linear', 'geometric', 'harmonic', 'quadratic']
        if filter_combination not in valid_combinations:
            self.logger.warning(f"無効なフィルター組み合わせ方法: {filter_combination}。デフォルトの'quadratic'を使用します。")
            self.filter_combination = 'quadratic'
        
        # 計算モードの検証
        valid_modes = [
            self.MODE_SIMPLE_AVERAGE,
            self.MODE_WEIGHTED_AVERAGE, 
            self.MODE_MAXIMUM, 
            self.MODE_HARMONIC, 
            self.MODE_LOGARITHMIC,
            self.MODE_ROOT_MEAN_SQUARE,
            self.MODE_ADAPTIVE_WEIGHTED,
            self.MODE_EXPONENTIAL_WEIGHTED,
            self.MODE_FILTER_WEIGHTED
        ]
        if mode not in valid_modes:
            raise ValueError(f"無効な計算モード: {mode}。有効なモード: {', '.join(valid_modes)}")
        
        # サブインジケーターを初期化
        self.alpha_atr = AlphaATR(
            er_period=er_period,
            max_atr_period=max_period,
            min_atr_period=min_period
        )
        
        self.alpha_volatility = AlphaVolatility(
            er_period=er_period,
            max_vol_period=max_period,
            min_vol_period=min_period,
            smoothing_period=smoothing_period
        )
        
        # アルファフィルターを初期化（use_filterがTrueの場合のみ）
        self.alpha_filter = None
        self.alpha_choppiness = None
        self.er_indicator = None
        
        # フィルターのどちらかを初期化
        if self.use_filter:
            if self.use_simplified_filter:
                # シンプルフィルター：ADXなしでチョピネスとERだけを使用
                self.alpha_choppiness = AlphaChoppiness(
                    er_period=er_period,
                    max_chop_period=max_chop_period,
                    min_chop_period=min_chop_period
                )
                self.er_indicator = EfficiencyRatio(er_period)
            else:
                # 通常のAlphaFilter
                self.alpha_filter = AlphaFilter(
                    er_period=er_period,
                    max_chop_period=max_chop_period,
                    min_chop_period=min_chop_period,
                    max_adx_period=max_adx_period,
                    min_adx_period=min_adx_period,
                    max_threshold=max_threshold,
                    min_threshold=min_threshold
                )
        
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファVIXを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            VIXの値（%ベース）
        """
        try:
            # サブインジケーターを計算
            self.alpha_atr.calculate(data)
            self.alpha_volatility.calculate(data)
            
            # 各インジケーターの値を取得
            atr_percent = self.alpha_atr.get_percent_atr()
            atr_absolute = self.alpha_atr.get_absolute_atr()
            vol_percent = self.alpha_volatility.get_percent_volatility()
            vol_absolute = self.alpha_volatility.get_absolute_volatility()
            
            # アルファフィルターを計算（use_filterがTrueの場合のみ）
            filter_values = None
            if self.use_filter:
                if self.use_simplified_filter and self.alpha_choppiness is not None and self.er_indicator is not None:
                    # シンプルフィルターの計算（チョピネスとERのみ）
                    chop_values = self.alpha_choppiness.calculate(data)
                    er_values = self.er_indicator.calculate(data)
                    
                    # 高度な組み合わせロジックを適用
                    filter_values = np.zeros_like(chop_values)
                    
                    for i in range(len(chop_values)):
                        if not np.isnan(chop_values[i]) and not np.isnan(er_values[i]):
                            # チョピネスは逆転させる（1-chop）
                            inverted_chop = 1.0 - chop_values[i]
                            
                            if self.filter_combination == 'linear':
                                # 線形結合
                                combined = self.chop_weight * inverted_chop + (1.0 - self.chop_weight) * er_values[i]
                            elif self.filter_combination == 'geometric':
                                # 幾何平均
                                if inverted_chop > 0 and er_values[i] > 0:
                                    combined = (inverted_chop ** self.chop_weight) * (er_values[i] ** (1.0 - self.chop_weight))
                                else:
                                    combined = 0.0
                            elif self.filter_combination == 'harmonic':
                                # 調和重み付け
                                if inverted_chop > 0.01 and er_values[i] > 0.01:
                                    combined = 1.0 / ((self.chop_weight / inverted_chop) + ((1.0 - self.chop_weight) / er_values[i]))
                                else:
                                    combined = 0.0
                            else:  # 'quadratic'
                                # 二次形式
                                linear = self.chop_weight * inverted_chop + (1.0 - self.chop_weight) * er_values[i]
                                combined = linear + (inverted_chop * er_values[i] * 0.5)
                                combined = min(1.0, combined)
                            
                            # シグモイド関数による強調
                            if combined != 0.5:
                                sigmoid = 1.0 / (1.0 + np.exp(-6.0 * (combined - 0.5)))
                                filter_values[i] = sigmoid
                            else:
                                filter_values[i] = 0.5
                        else:
                            filter_values[i] = np.nan
                elif not self.use_simplified_filter and self.alpha_filter is not None:
                    # 通常のAlphaFilterを使用
                    filter_values = self.alpha_filter.calculate(data)
            
            # 計算モードに基づいて計算
            if self.mode == self.MODE_SIMPLE_AVERAGE:
                vix_percent = calculate_simple_average(atr_percent, vol_percent)
                vix_absolute = calculate_simple_average(atr_absolute, vol_absolute)
            elif self.mode == self.MODE_WEIGHTED_AVERAGE:
                # 加重平均の重みとして効率比を使用
                er = self.alpha_atr.get_efficiency_ratio()
                vix_percent = calculate_weighted_average(atr_percent, vol_percent, er)
                vix_absolute = calculate_weighted_average(atr_absolute, vol_absolute, er)
            elif self.mode == self.MODE_MAXIMUM:
                vix_percent = calculate_maximum(atr_percent, vol_percent)
                vix_absolute = calculate_maximum(atr_absolute, vol_absolute)
            elif self.mode == self.MODE_HARMONIC:
                vix_percent = calculate_harmonic_mean(atr_percent, vol_percent)
                vix_absolute = calculate_harmonic_mean(atr_absolute, vol_absolute)
            elif self.mode == self.MODE_LOGARITHMIC:
                vix_percent = calculate_logarithmic_mean(atr_percent, vol_percent)
                vix_absolute = calculate_logarithmic_mean(atr_absolute, vol_absolute)
            elif self.mode == self.MODE_ROOT_MEAN_SQUARE:
                vix_percent = calculate_root_mean_square(atr_percent, vol_percent)
                vix_absolute = calculate_root_mean_square(atr_absolute, vol_absolute)
            elif self.mode == self.MODE_ADAPTIVE_WEIGHTED:
                vix_percent = calculate_adaptive_weighted_average(atr_percent, vol_percent)
                vix_absolute = calculate_adaptive_weighted_average(atr_absolute, vol_absolute)
            elif self.mode == self.MODE_EXPONENTIAL_WEIGHTED:
                vix_percent = calculate_exponential_weighted_average(atr_percent, vol_percent, self.exp_bias)
                vix_absolute = calculate_exponential_weighted_average(atr_absolute, vol_absolute, self.exp_bias)
            elif self.mode == self.MODE_FILTER_WEIGHTED and filter_values is not None:
                vix_percent = calculate_filter_weighted_average(atr_percent, vol_percent, filter_values, self.filter_alpha)
                vix_absolute = calculate_filter_weighted_average(atr_absolute, vol_absolute, filter_values, self.filter_alpha)
            else:
                # デフォルトはRMSを使用
                vix_percent = calculate_root_mean_square(atr_percent, vol_percent)
                vix_absolute = calculate_root_mean_square(atr_absolute, vol_absolute)
            
            # 効率比を取得（どちらからでも取得可能なので、ATRから取得）
            er = self.alpha_atr.get_efficiency_ratio()
            
            # 結果の保存
            self._result = AlphaVIXResult(
                values=vix_percent,
                absolute_values=vix_absolute,
                atr_values=atr_percent,
                vol_values=vol_percent,
                atr_abs_values=atr_absolute,
                vol_abs_values=vol_absolute,
                er=er,
                filter_values=filter_values  # アルファフィルター値を保存
            )
            
            self._values = vix_percent  # 標準インジケーターインターフェース用
            return vix_percent
            
        except Exception as e:
            self.logger.error(f"AlphaVIX計算中にエラー: {str(e)}")
            return np.array([])
    
    def get_percent_vix(self) -> np.ndarray:
        """
        %ベースのVIXを取得する
        
        Returns:
            np.ndarray: %ベースのVIX値（100倍されたパーセンテージ値）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.values * 100  # 100倍して返す
    
    def get_absolute_vix(self) -> np.ndarray:
        """
        金額ベースのVIXを取得する
        
        Returns:
            np.ndarray: 金額ベースのVIX値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.absolute_values
    
    def get_percent_atr(self) -> np.ndarray:
        """
        %ベースのATRを取得する
        
        Returns:
            np.ndarray: %ベースのATR値（100倍されたパーセンテージ値）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.atr_values * 100  # 100倍して返す
    
    def get_percent_volatility(self) -> np.ndarray:
        """
        %ベースのボラティリティを取得する
        
        Returns:
            np.ndarray: %ベースのボラティリティ値（100倍されたパーセンテージ値）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.vol_values * 100  # 100倍して返す
    
    def get_absolute_atr(self) -> np.ndarray:
        """
        金額ベースのATRを取得する
        
        Returns:
            np.ndarray: 金額ベースのATR値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.atr_abs_values
    
    def get_absolute_volatility(self) -> np.ndarray:
        """
        金額ベースのボラティリティを取得する
        
        Returns:
            np.ndarray: 金額ベースのボラティリティ値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.vol_abs_values
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.er
    
    def get_vix_multiple(self, multiplier: float = 1.0) -> np.ndarray:
        """
        %ベースのVIXの倍数を取得する
        
        Args:
            multiplier: VIXの倍数（デフォルト: 1.0）
            
        Returns:
            np.ndarray: %ベースのVIX × 倍数（100倍されたパーセンテージ値）
        """
        vix = self.get_percent_vix()
        return vix * multiplier
    
    def get_absolute_vix_multiple(self, multiplier: float = 1.0) -> np.ndarray:
        """
        金額ベースのVIXの倍数を取得する
        
        Args:
            multiplier: VIXの倍数（デフォルト: 1.0）
            
        Returns:
            np.ndarray: 金額ベースのVIX × 倍数
        """
        abs_vix = self.get_absolute_vix()
        return abs_vix * multiplier
    
    def get_filter_values(self) -> Optional[np.ndarray]:
        """
        アルファフィルターの値を取得する（use_filterがTrueの場合のみ）
        
        Returns:
            np.ndarray: アルファフィルター値、またはNone（フィルターが使用されていない場合）
        """
        if not self.use_filter or self.alpha_filter is None:
            return None
        return self.alpha_filter._values 