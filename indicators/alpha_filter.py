#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from numba import jit

from .indicator import Indicator
from .alpha_choppiness import AlphaChoppiness
from .efficiency_ratio import EfficiencyRatio, calculate_efficiency_ratio_for_period


@dataclass
class AlphaFilterResult:
    """アルファフィルターの計算結果"""
    values: np.ndarray        # アルファフィルター値
    er: np.ndarray            # 効率比
    alpha_chop: np.ndarray    # アルファチョピネス値
    dynamic_period: np.ndarray  # 動的期間
    threshold: np.ndarray     # 動的しきい値


@jit(nopython=True)
def calculate_alpha_filter(alpha_chop_values: np.ndarray, er_values: np.ndarray, chop_weight: float, er_weight: float, length: int) -> np.ndarray:
    """
    アルファフィルターを計算する（高速化版）
    
    Args:
        alpha_chop_values: アルファチョピネスの値（0-1の範囲、1に近いほどレンジ相場）
        er_values: 効率比の値（0-1の範囲、1に近いほど強いトレンド）
        chop_weight: チョピネスの重み
        er_weight: 効率比の重み
        length: データ長
    
    Returns:
        アルファフィルターの値（0-1の範囲、1に近いほど強いトレンド）
    """
    result = np.zeros(length)
    
    for i in range(length):
        # NaN値のチェック - Numbaでは直接np.isnanを使用できないため、比較で代用
        chop_is_nan = alpha_chop_values[i] != alpha_chop_values[i]
        er_is_nan = er_values[i] != er_values[i]
        
        if chop_is_nan or er_is_nan:
            result[i] = 0.0
        else:
            # チョピネスは逆転させる（1-chop）：チョピネスが低いほどトレンドが強い
            inverted_chop = 1.0 - alpha_chop_values[i]
            
            # 重み付け平均の計算
            weighted_sum = (inverted_chop * chop_weight) + (er_values[i] * er_weight)
            total_weight = chop_weight + er_weight
            
            if total_weight > 0:
                result[i] = weighted_sum / total_weight
            else:
                result[i] = 0.5  # デフォルト値
            
            # シグモイド関数による強調（結果を0-1の範囲に保ちながら中間の値を強調）
            if result[i] != 0.5:
                # sigmoid(8*(x-0.5))を0-1に再スケール
                sigmoid = 1.0 / (1.0 + np.exp(-8.0 * (result[i] - 0.5)))
                result[i] = sigmoid
    
    return result


@jit(nopython=True)
def calculate_dynamic_threshold(er: np.ndarray, max_threshold: float, min_threshold: float) -> np.ndarray:
    """
    効率比に基づいて動的なしきい値を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_threshold: 最大しきい値
        min_threshold: 最小しきい値
    
    Returns:
        動的なしきい値の配列
    """
    # ERが高い（トレンドが強い）ほどしきい値は高く、
    # ERが低い（トレンドが弱い）ほどしきい値は低くなる
    return min_threshold + er * (max_threshold - min_threshold)


class AlphaFilter(Indicator):
    """
    アルファフィルター（Alpha Filter）インジケーター
    
    AlphaChoppinessとEfficiencyRatioを組み合わせた
    軽量かつ効果的な市場状態フィルタリング指標
    
    - 0.7以上：非常に強いトレンド
    - 0.5-0.7：中程度のトレンド
    - 0.3-0.5：弱いトレンド/レンジ転換の可能性
    - 0.3以下：明確なレンジ相場
    
    特徴：
    - すべてのコンポーネントが効率比（ER）に基づいて動的に最適化
    - チョピネスと効率比を組み合わせた最適なトレンド判定
    - シグモイド関数による明確な判別強化
    - 効率比に基づく動的しきい値の調整
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        chop_weight: float = 0.6,  # チョピネスの重み（0-1）
        er_weight: float = 0.4,    # 効率比の重み（0-1）
        max_threshold: float = 0.55,
        min_threshold: float = 0.45,
        # 以下のパラメータは後方互換性のために残しておく（使用されない）
        max_adx_period: int = 21,
        min_adx_period: int = 5
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_chop_period: アルファチョピネス期間の最大値（デフォルト: 55）
            min_chop_period: アルファチョピネス期間の最小値（デフォルト: 8）
            chop_weight: チョピネスの重み（0-1の範囲）（デフォルト: 0.6）
                         高いほどチョピネスを重視、低いほど効率比を重視
            er_weight: 効率比の重み（0-1の範囲）（デフォルト: 0.4）
                       高いほど効率比を重視、低いほどチョピネスを重視
            max_threshold: しきい値の最大値（デフォルト: 0.55）
            min_threshold: しきい値の最小値（デフォルト: 0.45）
            max_adx_period: 未使用（後方互換性のため）
            min_adx_period: 未使用（後方互換性のため）
        """
        super().__init__(f"AlphaFilter({er_period})")
        self.er_period = er_period
        
        # 各サブインジケーターの設定
        self.max_chop_period = max_chop_period
        self.min_chop_period = min_chop_period
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        
        # 重みの設定と正規化
        self.chop_weight = max(0.0, min(1.0, chop_weight))
        self.er_weight = max(0.0, min(1.0, er_weight))
        
        # 重みの合計が1になるように正規化
        total_weight = self.chop_weight + self.er_weight
        if total_weight > 0:
            self.chop_weight /= total_weight
            self.er_weight /= total_weight
        else:
            # デフォルト値
            self.chop_weight = 0.6
            self.er_weight = 0.4
        
        # サブインジケーターのインスタンス化
        self.alpha_chop = AlphaChoppiness(
            er_period=er_period,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
        )
        
        self.er_indicator = EfficiencyRatio(er_period)
        
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファフィルターを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            アルファフィルターの値（0-1の範囲）
            - 1に近いほど質の高いトレンド
            - 0に近いほどレンジ相場
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
                close = data['close'].values
            else:
                if data.ndim == 2:
                    close = data[:, 3]  # close
                else:
                    close = data
            
            # 効率比（ER）の計算
            er_values = self.er_indicator.calculate(data)
            
            # アルファチョピネスの計算
            alpha_chop_values = self.alpha_chop.calculate(data)
            
            # 動的な期間の取得（アルファチョピネスから）
            dynamic_period = self.alpha_chop.get_dynamic_period()
            
            # 動的なしきい値の計算
            threshold = calculate_dynamic_threshold(
                er_values,
                self.max_threshold,
                self.min_threshold
            )
            
            # フィルター値の計算（高速化版）
            length = len(close)
            result_values = calculate_alpha_filter(
                alpha_chop_values, 
                er_values, 
                self.chop_weight, 
                self.er_weight, 
                length
            )
            
            self._values = result_values
            
            self._result = AlphaFilterResult(
                values=result_values,
                er=er_values,
                alpha_chop=alpha_chop_values,
                dynamic_period=dynamic_period,
                threshold=threshold
            )
            
            return self._values
            
        except Exception as e:
            self.logger.error(f"AlphaFilter計算中にエラー: {str(e)}")
            return None
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.er
    
    def get_alpha_choppiness(self) -> np.ndarray:
        """
        アルファチョピネスの値を取得する
        
        Returns:
            np.ndarray: アルファチョピネスの値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.alpha_chop
    
    def get_dynamic_period(self) -> np.ndarray:
        """
        動的期間の値を取得する
        
        Returns:
            np.ndarray: 動的期間の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.dynamic_period
    
    def get_threshold(self) -> np.ndarray:
        """
        動的しきい値を取得する
        
        Returns:
            np.ndarray: しきい値の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.threshold 