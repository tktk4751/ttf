#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.alpha_ma_v2_keltner import AlphaMAV2KeltnerChannel


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


class AlphaMAV2KeltnerBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    アルファMAV2ケルトナーチャネルのブレイクアウトによるエントリーシグナル
    
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
        ma_er_period: int = 10,
        ma_max_period: int = 34,
        ma_min_period: int = 5,
        atr_er_period: int = 21,
        atr_max_period: int = 89,
        atr_min_period: int = 13,
        upper_multiplier: float = 2.0,
        lower_multiplier: float = 2.0,
        lookback: int = 1
    ):
        """
        コンストラクタ
        
        Args:
            ma_er_period: AlphaMAV2の効率比計算期間（デフォルト: 10）
            ma_max_period: AlphaMAV2の最大期間（デフォルト: 34）
            ma_min_period: AlphaMAV2の最小期間（デフォルト: 5）
            atr_er_period: AlphaATRの効率比計算期間（デフォルト: 21）
            atr_max_period: AlphaATRの最大期間（デフォルト: 89）
            atr_min_period: AlphaATRの最小期間（デフォルト: 13）
            upper_multiplier: 上限バンドのATR乗数（デフォルト: 2.0）
            lower_multiplier: 下限バンドのATR乗数（デフォルト: 2.0）
            lookback: 過去のバンドを参照する期間（デフォルト: 1）
        """
        params = {
            'ma_er_period': ma_er_period,
            'ma_max_period': ma_max_period,
            'ma_min_period': ma_min_period,
            'atr_er_period': atr_er_period,
            'atr_max_period': atr_max_period,
            'atr_min_period': atr_min_period,
            'upper_multiplier': upper_multiplier,
            'lower_multiplier': lower_multiplier,
            'lookback': lookback
        }
        super().__init__(
            f"AlphaMAV2KeltnerBreakout({ma_er_period}, {ma_max_period}, {ma_min_period}, {upper_multiplier}, {lower_multiplier}, {lookback})",
            params
        )
        
        # アルファMAV2ケルトナーチャネルのインスタンス化
        self._alpha_mav2_keltner = AlphaMAV2KeltnerChannel(
            ma_er_period=ma_er_period,
            ma_max_period=ma_max_period,
            ma_min_period=ma_min_period,
            atr_er_period=atr_er_period,
            atr_max_period=atr_max_period,
            atr_min_period=atr_min_period,
            upper_multiplier=upper_multiplier,
            lower_multiplier=lower_multiplier
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
            # アルファMAV2ケルトナーチャネルの計算
            result = self._alpha_mav2_keltner.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if result is None:
                return np.zeros(len(data), dtype=np.int8)
            
            # 終値の取得
            close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
            
            # バンドの取得
            middle, upper, lower, _, _ = self._alpha_mav2_keltner.get_bands()
            
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
            self.logger.error(f"AlphaMAV2KeltnerBreakoutEntrySignal計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        ケルトナーチャネルのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限バンド, 下限バンド, 中間上限バンド, 中間下限バンド)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self._alpha_mav2_keltner.get_bands()
    
    def get_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        AlphaATRの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: AlphaATRの値
        """
        if data is not None:
            self.generate(data)
            
        return self._alpha_mav2_keltner.get_atr()
    
    def get_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        動的期間の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (AlphaMAV2の動的期間, AlphaATRの動的期間)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self._alpha_mav2_keltner.get_dynamic_periods()
    
    def get_channel_width(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        チャネル幅の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: チャネル幅の値
        """
        if data is not None:
            self.generate(data)
            
        return self._alpha_mav2_keltner.get_channel_width()
    
    def get_breakout_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        ブレイクアウトシグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (上方ブレイクアウト, 下方ブレイクアウト)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self._alpha_mav2_keltner.get_breakout_signals(data) 