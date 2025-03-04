#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.trend_alpha import TrendAlpha


@jit(nopython=True)
def calculate_breakout_signals(
    close: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    half_upper: np.ndarray,
    half_lower: np.ndarray,
    er: np.ndarray,
    dynamic_period: np.ndarray,
    max_period: int
) -> np.ndarray:
    """
    ブレイクアウトシグナルを計算する（高速化版）
    
    Args:
        close: 終値の配列
        upper: アッパーバンドの配列
        lower: ロワーバンドの配列
        half_upper: 中間アッパーバンドの配列
        half_lower: 中間ロワーバンドの配列
        er: 効率比の配列
        dynamic_period: 動的ATR期間の配列
        max_period: 最大期間
    
    Returns:
        シグナルの配列
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初の期間はシグナルなし
    signals[:max_period] = 0
    
    # ブレイクアウトの判定
    for i in range(max_period, length):
        # ロングエントリー: 終値がバンドを上回る
        if close[i] > upper[i-1]:
            signals[i] = 1
        # ショートエントリー: 終値がバンドを下回る
        elif close[i] < lower[i-1]:
            signals[i] = -1
    
    return signals


class TrendAlphaBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    TrendAlphaのブレイクアウトによるエントリーシグナル
    
    効率比（ER）と動的ATR期間に基づいて動的にバンドを選択：
    - ERが高い（トレンドが強い）時：
        - 中間バンドに近づく
        - ATR期間が短くなり、より敏感に反応
    - ERが低い（トレンドが弱い）時：
        - 外側のバンドに近づく
        - ATR期間が長くなり、ノイズを軽減
    
    エントリー条件：
    - 終値が動的に調整されたバンドを上回る/下回る
    """
    
    def __init__(
        self,
        kama_period: int = 10,
        kama_fast: int = 2,
        kama_slow: int = 30,
        max_atr_period: int = 120,
        min_atr_period: int = 5,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0
    ):
        """
        コンストラクタ
        
        Args:
            kama_period: KAMAの効率比の計算期間（デフォルト: 10）
            kama_fast: KAMAの速い移動平均の期間（デフォルト: 2）
            kama_slow: KAMAの遅い移動平均の期間（デフォルト: 30）
            max_atr_period: ATR期間の最大値（デフォルト: 120）
            min_atr_period: ATR期間の最小値（デフォルト: 5）
            max_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（デフォルト: 1.0）
        """
        super().__init__(
            f"TrendAlphaBreakout({kama_period}, {kama_fast}, {kama_slow}, "
            f"{max_atr_period}, {min_atr_period}, {max_multiplier}, {min_multiplier})"
        )
        
        # パラメータの設定
        self._params = {
            'kama_period': kama_period,
            'kama_fast': kama_fast,
            'kama_slow': kama_slow,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
        }
        
        # インジケーターの初期化
        self._trend_alpha = TrendAlpha(
            kama_period=kama_period,
            kama_fast=kama_fast,
            kama_slow=kama_slow,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # TrendAlphaの計算
        result = self._trend_alpha.calculate(data)
        if result is None:
            return np.zeros(len(data))
        
        # 終値の取得
        close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
        
        # バンド、効率比、動的期間の取得
        _, upper, lower, half_upper, half_lower = self._trend_alpha.get_bands()
        er = self._trend_alpha.get_efficiency_ratio()
        dynamic_period = self._trend_alpha.get_dynamic_period()
        
        # ブレイクアウトシグナルの計算（高速化版）
        return calculate_breakout_signals(
            close,
            upper,
            lower,
            half_upper,
            half_lower,
            er,
            dynamic_period,
            self._params['max_atr_period']
        ) 