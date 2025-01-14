#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.alma import ALMA

class ALMACrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    ALMAクロスオーバーを使用したエントリーシグナル
    - 短期ALMA > 長期ALMA: ロングエントリー (1)
    - 短期ALMA < 長期ALMA: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        short_period: int = 9,
        long_period: int = 21,
        sigma: float = 6.0,
        offset: float = 0.85
    ):
        """
        コンストラクタ
        
        Args:
            short_period: 短期ALMAの期間
            long_period: 長期ALMAの期間
            sigma: ガウス分布の標準偏差
            offset: 重みの中心位置（0-1）
        """
        params = {
            'short_period': short_period,
            'long_period': long_period,
            'sigma': sigma,
            'offset': offset
        }
        super().__init__(f"ALMACrossover({short_period}, {long_period})", params)
        
        # ALMAインジケーターの初期化
        self._short_alma = ALMA(short_period, sigma, offset)
        self._long_alma = ALMA(long_period, sigma, offset)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # ALMAの計算
        short_alma = self._short_alma.calculate(data)
        long_alma = self._long_alma.calculate(data)
        
        # シグナルの初期化
        signals = np.zeros(len(data))
        
        # クロスオーバーの検出
        # 前日のクロス状態と当日のクロス状態を比較
        prev_short = np.roll(short_alma, 1)
        prev_long = np.roll(long_alma, 1)
        
        # ゴールデンクロス（短期が長期を上抜け）
        golden_cross = (prev_short <= prev_long) & (short_alma > long_alma)
        
        # デッドクロス（短期が長期を下抜け）
        dead_cross = (prev_short >= prev_long) & (short_alma < long_alma)
        
        # シグナルの設定
        signals = np.where(golden_cross, 1, signals)  # ロングエントリー
        signals = np.where(dead_cross, -1, signals)   # ショートエントリー
        
        # 最初の要素はクロスの判定ができないのでシグナルなし
        signals[0] = 0
        
        return signals 