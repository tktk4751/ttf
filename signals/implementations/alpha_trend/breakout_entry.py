#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.alpha_trend import AlphaTrend


@jit(nopython=True)
def calculate_breakout_signals(
    close: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    half_upper: np.ndarray,
    half_lower: np.ndarray,
    er: np.ndarray,
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
        max_period: 最大期間
    
    Returns:
        シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初の期間はシグナルなし
    signals[:max_period] = 0
    
    # ブレイクアウトの判定
    for i in range(max_period, length):
        if np.isnan(er[i]):
            continue
            
        # ERが高い場合は中間バンドを使用、低い場合は外側バンドを使用
        upper_band = half_upper[i-1] if er[i] >= 0.5 else upper[i-1]
        lower_band = half_lower[i-1] if er[i] >= 0.5 else lower[i-1]
        
        # ロングエントリー: 終値がバンドを上回る
        if close[i] > upper_band:
            signals[i] = 1
        # ショートエントリー: 終値がバンドを下回る
        elif close[i] < lower_band:
            signals[i] = -1
    
    return signals


class AlphaTrendBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    AlphaTrendのブレイクアウトによるエントリーシグナル
    
    効率比（ER）に基づいて動的にバンドを選択：
    - ERが高い（トレンドが強い）時：
        - 中間バンドを使用（より早いエントリー）
        - ATR期間が短くなり、より敏感に反応
        - KAMAのfast/slow期間が短くなり、より敏感に反応
    - ERが低い（トレンドが弱い）時：
        - 外側のバンドを使用（フェイクを回避）
        - ATR期間が長くなり、ノイズを軽減
        - KAMAのfast/slow期間が長くなり、ノイズを軽減
    
    エントリー条件：
    - 終値が動的に調整されたバンドを上回る/下回る
    """
    
    def __init__(
        self,
        period: int = 10,
        max_kama_slow: int = 55,
        min_kama_slow: int = 30,
        max_kama_fast: int = 13,
        min_kama_fast: int = 2,
        max_atr_period: int = 120,
        min_atr_period: int = 5,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0
    ):
        """
        コンストラクタ
        
        Args:
            period: KAMAの効率比の計算期間（デフォルト: 10）
            max_kama_slow: KAMAの遅い移動平均の最大期間（デフォルト: 55）
            min_kama_slow: KAMAの遅い移動平均の最小期間（デフォルト: 30）
            max_kama_fast: KAMAの速い移動平均の最大期間（デフォルト: 13）
            min_kama_fast: KAMAの速い移動平均の最小期間（デフォルト: 2）
            max_atr_period: ATR期間の最大値（デフォルト: 120）
            min_atr_period: ATR期間の最小値（デフォルト: 5）
            max_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（デフォルト: 1.0）
        """
        super().__init__(
            f"AlphaTrendBreakout({period}, {max_kama_slow}, {min_kama_slow}, "
            f"{max_kama_fast}, {min_kama_fast}, {max_atr_period}, {min_atr_period}, "
            f"{max_multiplier}, {min_multiplier})"
        )
        
        # パラメータの設定
        self._params = {
            'period': period,
            'max_kama_slow': max_kama_slow,
            'min_kama_slow': min_kama_slow,
            'max_kama_fast': max_kama_fast,
            'min_kama_fast': min_kama_fast,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
        }
        
        # インジケーターの初期化
        self._alpha_trend = AlphaTrend(
            period=period,
            max_kama_slow=max_kama_slow,
            min_kama_slow=min_kama_slow,
            max_kama_fast=max_kama_fast,
            min_kama_fast=min_kama_fast,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # AlphaTrendの計算
            result = self._alpha_trend.calculate(data)
            if result is None:
                return np.zeros(len(data))
            
            # 終値の取得
            close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
            
            # バンド、効率比の取得
            _, upper, lower, half_upper, half_lower = self._alpha_trend.get_bands()
            er = self._alpha_trend.get_efficiency_ratio()
            
            # ブレイクアウトシグナルの計算（高速化版）
            return calculate_breakout_signals(
                close,
                upper,
                lower,
                half_upper,
                half_lower,
                er,
                self._params['max_atr_period']
            )
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return np.zeros(len(data)) 