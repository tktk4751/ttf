#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.alpha_filter import AlphaFilter


@jit(nopython=True)
def generate_alpha_filter_signals(
    filter_values: np.ndarray,
    threshold_values: np.ndarray
) -> np.ndarray:
    """
    アルファフィルター値からシグナルを生成する（高速化版）
    
    Args:
        filter_values: アルファフィルター値の配列
        threshold_values: 動的しきい値の配列
    
    Returns:
        シグナルの配列 (1: トレンド相場, -1: レンジ相場)
    """
    length = len(filter_values)
    signals = np.zeros(length, dtype=np.int32)
    
    for i in range(length):
        if np.isnan(filter_values[i]) or np.isnan(threshold_values[i]):
            signals[i] = 0  # データ不足の場合はニュートラル
        elif filter_values[i] >= threshold_values[i]:
            signals[i] = 1  # トレンド相場
        else:
            signals[i] = -1  # レンジ相場
    
    return signals


class AlphaFilterSignal(BaseSignal, IFilterSignal):
    """
    アルファフィルターを使用したフィルターシグナル
    
    アルファフィルターは、AlphaChoppiness、AlphaADX、EfficiencyRatioを
    組み合わせた高度な市場状態フィルタリング指標です。
    
    - フィルター値 >= 動的しきい値: トレンド相場 (1)
    - フィルター値 < 動的しきい値: レンジ相場 (-1)
    
    動的しきい値は効率比（ER）に基づいて自動調整されます：
    - ERが高い（トレンドが強い）時：しきい値は高くなる（より確実なトレンド判定）
    - ERが低い（トレンドが弱い）時：しきい値は低くなる（より早いトレンド検出）
    
    推奨しきい値範囲：
    - max_threshold: 0.55（トレンド相場での高いしきい値）
    - min_threshold: 0.45（レンジ相場での低いしきい値）
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        max_threshold: float = 0.55,
        min_threshold: float = 0.45,
        solid: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_chop_period: アルファチョピネス期間の最大値（デフォルト: 55）
            min_chop_period: アルファチョピネス期間の最小値（デフォルト: 8）
            max_adx_period: アルファADX期間の最大値（デフォルト: 21）
            min_adx_period: アルファADX期間の最小値（デフォルト: 5）
            max_threshold: しきい値の最大値（デフォルト: 0.55）
            min_threshold: しきい値の最小値（デフォルト: 0.45）
            solid: パラメータ辞書（オプション）
        """
        params = {
            'er_period': er_period,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold,
            'solid': solid or {}
        }
        super().__init__(f"AlphaFilter({er_period})", params)
        
        # アルファフィルターのインスタンス化
        self._alpha_filter = AlphaFilter(
            er_period=er_period,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_threshold=max_threshold,
            min_threshold=min_threshold
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: トレンド相場, -1: レンジ相場)
        """
        # アルファフィルター値の計算
        filter_values = self._alpha_filter.calculate(data)
        
        # 動的しきい値の取得
        threshold_values = self._alpha_filter.get_threshold()
        
        # シグナルの生成（高速化版）
        signals = generate_alpha_filter_signals(filter_values, threshold_values)
        
        return signals
    
    def get_filter_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルファフィルター値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
        
        Returns:
            np.ndarray: アルファフィルター値
        """
        if data is not None:
            self.generate(data)
            
        if self._alpha_filter._values is None:
            raise RuntimeError("generate()を先に呼び出してください")
        return self._alpha_filter._values
    
    def get_threshold_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的しきい値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
        
        Returns:
            np.ndarray: 動的しきい値
        """
        if data is not None:
            self.generate(data)
            
        return self._alpha_filter.get_threshold()
    
    def get_component_values(self, data: Union[pd.DataFrame, np.ndarray] = None):
        """
        アルファフィルターのコンポーネント値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
        
        Returns:
            dict: コンポーネント値の辞書
                - er: 効率比
                - alpha_chop: アルファチョピネス
                - alpha_adx: アルファADX
                - dynamic_period: 動的期間
                - threshold: 動的しきい値
        """
        if data is not None:
            self.generate(data)
            
        if self._alpha_filter._result is None:
            raise RuntimeError("generate()を先に呼び出してください")
        
        return {
            'er': self._alpha_filter.get_efficiency_ratio(),
            'alpha_chop': self._alpha_filter.get_alpha_choppiness(),
            'alpha_adx': self._alpha_filter.get_alpha_adx(),
            'dynamic_period': self._alpha_filter.get_dynamic_period(),
            'threshold': self._alpha_filter.get_threshold()
        } 