#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.guardian_angel import GuardianAngel


@jit(nopython=True)
def generate_signals_numba(
    chop_values: np.ndarray,
    threshold_values: np.ndarray
) -> np.ndarray:
    """
    シグナルを生成する（高速化版）
    
    Args:
        chop_values: チョピネス値の配列
        threshold_values: 動的しきい値の配列
    
    Returns:
        シグナルの配列 (1: トレンド相場, -1: レンジ相場)
    """
    length = len(chop_values)
    signals = np.ones(length)  # デフォルトはトレンド相場
    
    for i in range(length):
        if np.isnan(chop_values[i]) or np.isnan(threshold_values[i]):
            signals[i] = np.nan
        elif chop_values[i] >= threshold_values[i]:
            signals[i] = -1  # レンジ相場
    
    return signals


class GuardianAngelFilterSignal(BaseSignal, IFilterSignal):
    """
    ガーディアンエンジェルを使用したフィルターシグナル
    - ERが高い（トレンドが強い）時：
        - 期間は短くなる（より敏感に反応）
        - しきい値は低くなる（トレンド判定が容易に）
    - ERが低い（トレンドが弱い）時：
        - 期間は長くなる（ノイズを軽減）
        - しきい値は高くなる（レンジ判定が容易に）
    """
    
    def __init__(
        self,
        er_period: int = 10,
        max_period: int = 30,
        min_period: int = 10,
        solid: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 10）
            max_period: チョピネス期間の最大値（デフォルト: 30）
            min_period: チョピネス期間の最小値（デフォルト: 10）
            solid: パラメータ辞書
                - max_threshold: しきい値の最大値（デフォルト: 61.8）
                - min_threshold: しきい値の最小値（デフォルト: 38.2）
        """
        # デフォルトのパラメータ
        default_solid = {
            'max_threshold': 61.8,
            'min_threshold': 38.2
        }
        
        # パラメータの設定
        params = {
            'er_period': er_period,
            'max_period': max_period,
            'min_period': min_period,
            'solid': solid or default_solid
        }
        
        super().__init__(
            f"GuardianAngelFilter({er_period}, {max_period}, {min_period})",
            params
        )
        
        # インジケーターの初期化
        solid_params = self._params['solid']
        self._guardian = GuardianAngel(
            er_period=er_period,
            max_period=max_period,
            min_period=min_period,
            max_threshold=solid_params['max_threshold'],
            min_threshold=solid_params['min_threshold']
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: トレンド相場, -1: レンジ相場)
        """
        # チョピネス値と動的しきい値を計算
        chop_values = self._guardian.calculate(data)
        threshold_values = self._guardian.get_threshold()
        
        # シグナルの生成（高速化版）
        signals = generate_signals_numba(chop_values, threshold_values)
        
        return signals 