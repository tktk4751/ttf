#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Tuple
from .indicator import Indicator
from .moving_average import MovingAverage
from .atr import ATR


class SqueezeMomentum(Indicator):
    """
    LazyBearのSqueeze Momentum Indicator
    
    Bollinger BandsとKeltner Channelsを使用して価格のスクイーズ状態を検出し、
    モメンタムの方向と強さを計算します。
    """
    
    def __init__(
        self,
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5,
        use_tr: bool = True
    ):
        """
        コンストラクタ
        
        Args:
            bb_length: Bollinger Bands の期間
            bb_mult: Bollinger Bands の乗数
            kc_length: Keltner Channels の期間
            kc_mult: Keltner Channels の乗数
            use_tr: True Range を使用するかどうか
        """
        super().__init__("Squeeze Momentum")
        
        # パラメータの保存
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        
        # インディケーターの初期化
        self.ma = MovingAverage(period=bb_length, ma_type="sma")
        self.atr = ATR(period=kc_length)
        
        # 状態を保持する変数
        self._sqz_on: np.ndarray = None
        self._sqz_off: np.ndarray = None
        self._no_sqz: np.ndarray = None
        
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bollinger Bandsを計算
        
        Args:
            prices: 価格データ
            
        Returns:
            (upper, middle, lower) のタプル
        """
        # 中心線（単純移動平均）
        middle = self.ma.calculate(prices.values)
        
        # 標準偏差
        std = prices.rolling(window=self.bb_length).std().values
        
        # 上下のバンド
        upper = middle + self.bb_mult * std
        lower = middle - self.bb_mult * std
        
        return upper, middle, lower
    
    def _calculate_keltner_channels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Keltner Channelsを計算
        
        Args:
            df: 価格データ
            
        Returns:
            (upper, middle, lower) のタプル
        """
        # 中心線（単純移動平均）
        middle = self.ma.calculate(df['close'].values)
        
        # ATR
        atr = self.atr.calculate(df)
        
        # 上下のバンド
        upper = middle + self.kc_mult * atr
        lower = middle - self.kc_mult * atr
        
        return upper, middle, lower
    
    def _calculate_linear_regression(self, x: np.ndarray, length: int) -> np.ndarray:
        """
        線形回帰を計算
        
        Args:
            x: 入力配列
            length: 期間
            
        Returns:
            線形回帰の結果
        """
        result = np.zeros_like(x)
        for i in range(length - 1, len(x)):
            y = x[i-length+1:i+1]
            x_range = np.arange(length)
            slope, intercept = np.polyfit(x_range, y, 1)
            result[i] = slope * (length - 1) + intercept
        return result
    
    def calculate(self, df: pd.DataFrame) -> np.ndarray:
        """
        Squeeze Momentumインディケーターを計算
        
        Args:
            df: 価格データ（OHLC）を含むDataFrame
            
        Returns:
            モメンタム値の配列
        """
        # Bollinger Bands の計算
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'])
        
        # Keltner Channels の計算
        kc_upper, kc_middle, kc_lower = self._calculate_keltner_channels(df)
        
        # スクイーズ状態の判定
        self._sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        self._sqz_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)
        self._no_sqz = (~self._sqz_on) & (~self._sqz_off)
        
        # モメンタム値の計算
        highest_high = pd.Series(df['high']).rolling(window=self.kc_length).max().values
        lowest_low = pd.Series(df['low']).rolling(window=self.kc_length).min().values
        avg_hl = (highest_high + lowest_low) / 2
        avg_hlc = (avg_hl + kc_middle) / 2
        
        self._values = self._calculate_linear_regression(df['close'].values - avg_hlc, self.kc_length)
        
        return self._values
    
    def get_squeeze_states(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        スクイーズ状態を取得
        
        Returns:
            (sqz_on, sqz_off, no_sqz) のタプル
            - sqz_on: スクイーズオン状態
            - sqz_off: スクイーズオフ状態
            - no_sqz: スクイーズなし状態
        """
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._sqz_on, self._sqz_off, self._no_sqz 