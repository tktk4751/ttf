#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Protocol
import pandas as pd
import numpy as np
from logger import get_logger

class IStrategy(Protocol):
    """戦略のインターフェース"""
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """売買シグナルを生成する"""
        ...

class MovingAverageStrategy(IStrategy):
    """移動平均クロス戦略"""
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        """
        Args:
            short_window: 短期移動平均の期間
            long_window: 長期移動平均の期間
        """
        self.short_window = short_window
        self.long_window = long_window
        self.logger = get_logger(__name__)
    
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """売買シグナルを生成する
        
        Args:
            data: 価格データ
                必要なカラム: ['close']
        
        Returns:
            np.ndarray: 売買シグナル
                1: ロング
                0: ニュートラル
                -1: ショート
        """
        if len(data) < self.long_window:
            self.logger.warning(
                f"データ長が不足しています: {len(data)} < {self.long_window}"
            )
            return np.zeros(len(data))
        
        # 移動平均の計算
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        
        # シグナルの生成
        signals = np.zeros(len(data))
        
        # クロスオーバーの検出
        signals[short_ma > long_ma] = 1    # ゴールデンクロス
        signals[short_ma < long_ma] = -1   # デッドクロス
        
        # 最初のlong_window期間はシグナルなし
        signals[:self.long_window] = 0
        
        self.logger.info(
            f"シグナルを生成しました: "
            f"LONG: {np.sum(signals == 1)}, "
            f"SHORT: {np.sum(signals == -1)}, "
            f"NEUTRAL: {np.sum(signals == 0)}"
        )
        
        return signals 