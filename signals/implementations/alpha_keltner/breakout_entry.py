#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.alpha_keltner_channel import AlphaKeltnerChannel


@jit(nopython=True)
def calculate_breakout_signals(close: np.ndarray, upper: np.ndarray, lower: np.ndarray, lookback: int) -> np.ndarray:
    """
    ブレイクアウトシグナルを計算する（高速化版）
    
    Args:
        close: 終値の配列
        upper: アッパーバンドの配列
        lower: ロワーバンドの配列
        lookback: 過去のバンドを参照する期間
    
    Returns:
        シグナルの配列
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初の期間はシグナルなし
    signals[:lookback] = 0
    
    # ブレイクアウトの判定
    for i in range(lookback, length):
        # ロングエントリー: 終値がアッパーバンドを上回る
        if close[i] > upper[i-lookback]:
            signals[i] = 1
        # ショートエントリー: 終値がロワーバンドを下回る
        elif close[i] < lower[i-lookback]:
            signals[i] = -1
    
    return signals


class AlphaKeltnerBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    アルファケルトナーチャネルのブレイクアウトによるエントリーシグナル
    
    特徴:
    - 効率比（ER）に基づく動的な適応性
    - RSXの3段階平滑化アルゴリズムを使用したAlphaATR
    - ボラティリティに応じた最適なバンド幅
    - トレンドの強さに合わせた自動調整
    
    シグナル条件:
    - 現在の終値が指定期間前のアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値が指定期間前のロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_kama_period: int = 89,
        min_kama_period: int = 13,
        max_atr_period: int = 89,
        min_atr_period: int = 13,
        max_multiplier: float = 3.0,
        min_multiplier: float = 0.5,
        lookback: int = 1
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_kama_period: AlphaMAのKAMA最大期間（デフォルト: 89）
            min_kama_period: AlphaMAのKAMA最小期間（デフォルト: 13）
            max_atr_period: AlphaATRの最大期間（デフォルト: 89）
            min_atr_period: AlphaATRの最小期間（デフォルト: 13）
            max_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（デフォルト: 0.5）
            lookback: 過去のバンドを参照する期間（デフォルト: 1）
        """
        params = {
            'er_period': er_period,
            'max_kama_period': max_kama_period,
            'min_kama_period': min_kama_period,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'lookback': lookback
        }
        super().__init__(
            f"AlphaKeltnerBreakout({er_period}, {max_multiplier}, {min_multiplier}, {lookback})",
            params
        )
        
        # アルファケルトナーチャネルのインスタンス化
        self._alpha_keltner = AlphaKeltnerChannel(
            er_period=er_period,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period,
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
        try:
            # アルファケルトナーチャネルの計算
            result = self._alpha_keltner.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if result is None:
                return np.zeros(len(data), dtype=np.int8)
            
            # 終値の取得
            close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
            
            # バンドの取得
            _, upper, lower = self._alpha_keltner.get_bands()
            
            # ブレイクアウトシグナルの計算（高速化版）
            lookback = self._params['lookback']
            signals = calculate_breakout_signals(
                close,
                upper,
                lower,
                lookback
            )
            
            return signals
            
        except Exception as e:
            # エラーが発生した場合はゼロシグナルを返す
            self.logger.error(f"AlphaKeltnerBreakoutEntrySignal計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        ケルトナーチャネルのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            self.generate(data)
            
        _, upper, lower = self._alpha_keltner.get_bands()
        return upper, lower
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 効率比の値
        """
        if data is not None:
            self.generate(data)
            
        return self._alpha_keltner.get_efficiency_ratio()
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        if data is not None:
            self.generate(data)
            
        return self._alpha_keltner.get_dynamic_multiplier()
    
    def get_alpha_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        AlphaATRの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: AlphaATRの値
        """
        if data is not None:
            self.generate(data)
            
        return self._alpha_keltner.get_alpha_atr() 