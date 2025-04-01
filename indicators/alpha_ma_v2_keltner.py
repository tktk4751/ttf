#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .alpha_atr import AlphaATR
from .alpha_ma_v2 import AlphaMAV2


@dataclass
class AlphaMAV2KeltnerChannelResult:
    """AlphaMAV2ケルトナーチャネルの計算結果"""
    middle: np.ndarray  # 中心線（AlphaMAV2）
    upper: np.ndarray   # 上限線（AlphaMAV2 + upper_multiplier * AlphaATR）
    lower: np.ndarray   # 下限線（AlphaMAV2 - lower_multiplier * AlphaATR）
    half_upper: np.ndarray  # 中間上限線（AlphaMAV2 + upper_multiplier * 0.5 * AlphaATR）
    half_lower: np.ndarray  # 中間下限線（AlphaMAV2 - lower_multiplier * 0.5 * AlphaATR）
    atr: np.ndarray     # AlphaATRの値
    ma_dynamic_period: np.ndarray  # AlphaMAV2の動的期間
    atr_dynamic_period: np.ndarray  # AlphaATRの動的期間


@jit(nopython=True)
def calculate_keltner_bands(
    ma: np.ndarray, 
    atr: np.ndarray, 
    upper_multiplier: float, 
    lower_multiplier: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ケルトナーチャネルのバンドを計算する（高速化版）
    
    Args:
        ma: 移動平均の配列
        atr: ATRの配列
        upper_multiplier: 上限バンドの乗数
        lower_multiplier: 下限バンドの乗数
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            (上限バンド, 下限バンド, 中間上限バンド, 中間下限バンド)の配列
    """
    length = len(ma)
    upper = np.zeros(length)
    lower = np.zeros(length)
    half_upper = np.zeros(length)
    half_lower = np.zeros(length)
    
    for i in range(length):
        if np.isnan(ma[i]) or np.isnan(atr[i]):
            upper[i] = np.nan
            lower[i] = np.nan
            half_upper[i] = np.nan
            half_lower[i] = np.nan
        else:
            upper[i] = ma[i] + (upper_multiplier * atr[i])
            lower[i] = ma[i] - (lower_multiplier * atr[i])
            half_upper[i] = ma[i] + (upper_multiplier * 0.5 * atr[i])
            half_lower[i] = ma[i] - (lower_multiplier * 0.5 * atr[i])
    
    return upper, lower, half_upper, half_lower


class AlphaMAV2KeltnerChannel(Indicator):
    """
    AlphaMAV2ケルトナーチャネル インジケーター
    
    特徴:
    - 中心線にAlphaMAV2（RSXの3段階平滑化を使用）を使用
    - バンド幅の計算にAlphaATR（動的適応型、RSX平滑化使用）を使用
    - 両方とも効率比（ER）に基づいて動的に調整
    
    使用方法:
    - 中心線: AlphaMAV2
    - 上限線: AlphaMAV2 + upper_multiplier * AlphaATR
    - 下限線: AlphaMAV2 - lower_multiplier * AlphaATR
    - 中間上限線: AlphaMAV2 + upper_multiplier * 0.5 * AlphaATR
    - 中間下限線: AlphaMAV2 - lower_multiplier * 0.5 * AlphaATR
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
        lower_multiplier: float = 2.0
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
        """
        super().__init__(f"AlphaMAV2Keltner({ma_er_period}, {ma_max_period}, {ma_min_period}, {upper_multiplier}, {lower_multiplier})")
        self.ma_er_period = ma_er_period
        self.ma_max_period = ma_max_period
        self.ma_min_period = ma_min_period
        self.atr_er_period = atr_er_period
        self.atr_max_period = atr_max_period
        self.atr_min_period = atr_min_period
        self.upper_multiplier = upper_multiplier
        self.lower_multiplier = lower_multiplier
        
        # AlphaMAV2とAlphaATRのインスタンス化
        self.alpha_ma_v2 = AlphaMAV2(
            er_period=ma_er_period,
            max_period=ma_max_period,
            min_period=ma_min_period
        )
        
        self.alpha_atr = AlphaATR(
            er_period=atr_er_period,
            max_atr_period=atr_max_period,
            min_atr_period=atr_min_period
        )
        
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        AlphaMAV2ケルトナーチャネルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            AlphaMAV2の値（中心線）
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high'と'low'と'close'カラムが必要です")
            
            # AlphaMAV2の計算
            alpha_ma_v2_values = self.alpha_ma_v2.calculate(data)
            if alpha_ma_v2_values is None:
                raise ValueError("AlphaMAV2の計算に失敗しました")
            
            # AlphaATRの計算
            alpha_atr_values = self.alpha_atr.calculate(data)
            if alpha_atr_values is None:
                raise ValueError("AlphaATRの計算に失敗しました")
            
            # ケルトナーバンドの計算
            upper, lower, half_upper, half_lower = calculate_keltner_bands(
                alpha_ma_v2_values,
                alpha_atr_values,
                self.upper_multiplier,
                self.lower_multiplier
            )
            
            # AlphaMAV2の動的期間を取得
            ma_dynamic_period = self.alpha_ma_v2.get_dynamic_period()
            
            # AlphaATRの動的期間を取得
            atr_dynamic_period = self.alpha_atr.get_dynamic_period()
            
            # 結果の保存
            self._result = AlphaMAV2KeltnerChannelResult(
                middle=alpha_ma_v2_values,
                upper=upper,
                lower=lower,
                half_upper=half_upper,
                half_lower=half_lower,
                atr=alpha_atr_values,
                ma_dynamic_period=ma_dynamic_period,
                atr_dynamic_period=atr_dynamic_period
            )
            
            self._values = alpha_ma_v2_values  # 基底クラスの要件を満たすため
            return alpha_ma_v2_values
            
        except Exception as e:
            self.logger.error(f"AlphaMAV2KeltnerChannel計算中にエラー: {str(e)}")
            
            # エラー時でも最低限の結果を返す
            if 'data' in locals():
                length = len(data)
                empty_array = np.full(length, np.nan)
                self._result = AlphaMAV2KeltnerChannelResult(
                    middle=empty_array,
                    upper=empty_array,
                    lower=empty_array,
                    half_upper=empty_array,
                    half_lower=empty_array,
                    atr=empty_array,
                    ma_dynamic_period=empty_array,
                    atr_dynamic_period=empty_array
                )
                self._values = empty_array
                return empty_array
            return None
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        すべてのバンドの値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                (中心線, 上限線, 下限線, 中間上限線, 中間下限線)の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return (
            self._result.middle,
            self._result.upper,
            self._result.lower,
            self._result.half_upper,
            self._result.half_lower
        )
    
    def get_atr(self) -> np.ndarray:
        """
        ATRの値を取得する
        
        Returns:
            np.ndarray: ATRの値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.atr
    
    def get_dynamic_periods(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        動的期間を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                (AlphaMAV2の動的期間, AlphaATRの動的期間)のタプル
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return (self._result.ma_dynamic_period, self._result.atr_dynamic_period)
    
    def get_ma_dynamic_period(self) -> np.ndarray:
        """
        AlphaMAV2の動的期間を取得する
        
        Returns:
            np.ndarray: AlphaMAV2の動的期間
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.ma_dynamic_period
    
    def get_atr_dynamic_period(self) -> np.ndarray:
        """
        AlphaATRの動的期間を取得する
        
        Returns:
            np.ndarray: AlphaATRの動的期間
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.atr_dynamic_period
    
    def get_channel_width(self) -> np.ndarray:
        """
        チャネルの幅を取得する
        
        Returns:
            np.ndarray: チャネルの幅（上限線と下限線の差）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.upper - self._result.lower
    
    def get_position_in_channel(self, price: np.ndarray) -> np.ndarray:
        """
        チャネル内の価格の相対位置を取得する
        
        Args:
            price: 価格の配列
        
        Returns:
            np.ndarray: チャネル内の相対位置（0.0-1.0の範囲、0.5が中心線）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        length = len(price)
        position = np.full(length, np.nan)
        
        for i in range(length):
            if (np.isnan(self._result.upper[i]) or np.isnan(self._result.lower[i]) 
                or np.isnan(price[i])):
                position[i] = np.nan
            else:
                channel_width = self._result.upper[i] - self._result.lower[i]
                if channel_width <= 0:
                    position[i] = 0.5  # チャネル幅がゼロの場合は中央とする
                else:
                    position[i] = (price[i] - self._result.lower[i]) / channel_width
        
        return position
    
    def get_breakout_signals(self, data: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        ブレイクアウトシグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                (上方ブレイクアウト, 下方ブレイクアウト)のタプル
        """
        if data is not None:
            self.calculate(data)
        
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        # 価格データの取得
        close = data['close'].values if data is not None else None
        
        if close is None:
            raise ValueError("価格データが必要です")
        
        length = len(close)
        upper_breakout = np.zeros(length, dtype=np.int8)
        lower_breakout = np.zeros(length, dtype=np.int8)
        
        for i in range(1, length):
            # 上方ブレイクアウト: 前の終値がアッパーバンド以下で、現在の終値がアッパーバンド以上
            if (close[i-1] <= self._result.upper[i-1] and 
                close[i] > self._result.upper[i]):
                upper_breakout[i] = 1
            
            # 下方ブレイクアウト: 前の終値がロワーバンド以上で、現在の終値がロワーバンド以下
            if (close[i-1] >= self._result.lower[i-1] and 
                close[i] < self._result.lower[i]):
                lower_breakout[i] = 1
        
        return upper_breakout, lower_breakout
    
    def get_middle_band_signals(self, data: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        中心線クロスシグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                (上方クロス, 下方クロス)のタプル
        """
        if data is not None:
            self.calculate(data)
        
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        # 価格データの取得
        close = data['close'].values if data is not None else None
        
        if close is None:
            raise ValueError("価格データが必要です")
        
        length = len(close)
        upward_cross = np.zeros(length, dtype=np.int8)
        downward_cross = np.zeros(length, dtype=np.int8)
        
        for i in range(1, length):
            # 上方クロス: 前の終値が中心線以下で、現在の終値が中心線以上
            if (close[i-1] <= self._result.middle[i-1] and 
                close[i] > self._result.middle[i]):
                upward_cross[i] = 1
            
            # 下方クロス: 前の終値が中心線以上で、現在の終値が中心線以下
            if (close[i-1] >= self._result.middle[i-1] and 
                close[i] < self._result.middle[i]):
                downward_cross[i] = 1
        
        return upward_cross, downward_cross 