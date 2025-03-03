#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.kama import KaufmanAdaptiveMA

class KAMACrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    KAMAクロスオーバーを使用したエントリーシグナル
    - 短期KAMA > 長期KAMA: ロングエントリー (1)
    - 短期KAMA < 長期KAMA: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        short_period: int = 9,
        long_period: int = 21,
        fastest: int = 2,
        slowest: int = 30
    ):
        """
        コンストラクタ
        
        Args:
            short_period: 短期KAMAの期間
            long_period: 長期KAMAの期間
            period: 効率比率の計算期間
            fastest: 最速のスムージング定数の期間
            slowest: 最遅のスムージング定数の期間
        """
        params = {
            'short_period': short_period,
            'long_period': long_period,
            'fastest': fastest,
            'slowest': slowest
        }
        super().__init__(f"KAMACrossover({short_period}, {long_period})", params)
        
        # KAMAインジケーターの初期化
        self._short_kama = KaufmanAdaptiveMA(short_period, fastest, slowest)
        self._long_kama = KaufmanAdaptiveMA(long_period, fastest, slowest)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # KAMAの計算
        short_kama = self._short_kama.calculate(data)
        long_kama = self._long_kama.calculate(data)
        
        # シグナルの初期化
        signals = np.zeros(len(data))
        
        # クロスオーバーの検出
        # 前日のクロス状態と当日のクロス状態を比較
        prev_short = np.roll(short_kama, 1)
        prev_long = np.roll(long_kama, 1)
        
        # ゴールデンクロス（短期が長期を上抜け）
        golden_cross = (prev_short <= prev_long) & (short_kama > long_kama)
        
        # デッドクロス（短期が長期を下抜け）
        dead_cross = (prev_short >= prev_long) & (short_kama < long_kama)
        
        # シグナルの設定
        signals = np.where(golden_cross, 1, signals)  # ロングエントリー
        signals = np.where(dead_cross, -1, signals)   # ショートエントリー
        
        # 最初の要素はクロスの判定ができないのでシグナルなし
        signals[0] = 0
        
        return signals 