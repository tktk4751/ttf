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
    max_atr_period: int
) -> np.ndarray:
    """
    ブレイクアウトシグナルを計算する（高速化版）
    
    Args:
        close: 終値の配列
        upper: 上限バンドの配列
        lower: 下限バンドの配列
        half_upper: 中間上限バンドの配列
        half_lower: 中間下限バンドの配列
        er: 効率比の配列
        max_atr_period: ATRの最大期間（初期化のため）
    
    Returns:
        シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初のmax_atr_period分はシグナルなし（初期化）
    for i in range(max(max_atr_period, 1), length):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(half_upper[i]) or np.isnan(half_lower[i]):
            continue
            
        # 効率比が有効かどうかの確認
        valid_er = not np.isnan(er[i]) if er is not None else True
        
        # 上昇ブレイクアウトシグナル: 終値が上限バンドを上回る
        if valid_er and close[i] > upper[i] and close[i-1] <= upper[i-1]:
            signals[i] = 1
        
        # 下降ブレイクアウトシグナル: 終値が下限バンドを下回る
        elif valid_er and close[i] < lower[i] and close[i-1] >= lower[i-1]:
            signals[i] = -1
        
        # バンド内の反転: 終値が中間バンドを上から下へクロス
        elif close[i] < half_lower[i] and close[i-1] >= half_lower[i-1]:
            signals[i] = -1
        
        # バンド内の反転: 終値が中間バンドを下から上へクロス
        elif close[i] > half_upper[i] and close[i-1] <= half_upper[i-1]:
            signals[i] = 1
    
    return signals


class AlphaTrendBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    アルファトレンドのブレイクアウトによるエントリーシグナル
    
    特徴：
    - 効率比（ER）に基づく動的パラメータ最適化
    - アルファトレンドのバンドブレイクアウトシグナル
    - Numba JITによる高速化
    
    エントリー条件：
    - ロング: 価格が上限バンドを上抜けるか、中間上限バンドを下から上へ抜ける
    - ショート: 価格が下限バンドを下抜けるか、中間下限バンドを上から下へ抜ける
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
            period: 効率比の計算期間
            max_kama_slow: KAMAの最大遅い期間
            min_kama_slow: KAMAの最小遅い期間
            max_kama_fast: KAMAの最大速い期間
            min_kama_fast: KAMAの最小速い期間
            max_atr_period: ATRの最大期間
            min_atr_period: ATRの最小期間
            max_multiplier: ATR乗数の最大値
            min_multiplier: ATR乗数の最小値
        """
        params = {
            'period': period,
            'max_kama_slow': max_kama_slow,
            'min_kama_slow': min_kama_slow,
            'max_kama_fast': max_kama_fast,
            'min_kama_fast': min_kama_fast,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier
        }
        super().__init__(f"AlphaTrendBreakout({period})", params)
        
        # インジケーターの初期化
        self._alpha_trend = AlphaTrend(
            er_period=period,
            max_percentile_length=max_kama_slow,
            min_percentile_length=min_kama_slow,
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
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
                # データはそのまま渡す
                processed_data = data
                close = data['close'].values
            else:
                # NumPy配列の場合、DataFrameに変換
                if data.ndim != 2 or data.shape[1] < 3:
                    raise ValueError("ndarrayは[high, low, close]を含む2次元配列である必要があります")
                processed_data = pd.DataFrame({
                    'high': data[:, 0],
                    'low': data[:, 1],
                    'close': data[:, 2]
                })
                close = data[:, 2]  # 終値は3列目と仮定
            
            # AlphaTrendの計算 - 単一のデータ引数を渡す
            trend = self._alpha_trend.calculate(processed_data)
            if trend is None:
                return np.zeros(len(data))
            
            # バンドを取得（上限バンドと下限バンドのみを返す）
            upper_band, lower_band = self._alpha_trend.get_bands()
            
            # 中間バンドの計算
            half_upper = (upper_band + close) / 2
            half_lower = (lower_band + close) / 2
            
            # 効率比の取得
            er = self._alpha_trend.get_efficiency_ratio()
            
            # ブレイクアウトシグナルの計算（高速化版）
            return calculate_breakout_signals(
                close,
                upper_band,
                lower_band,
                half_upper,
                half_lower,
                er,
                self._params['max_atr_period']
            )
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return np.zeros(len(data)) 