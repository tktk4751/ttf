#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal


class EngulfingEntrySignal(BaseSignal, IEntrySignal):
    """
    包足（エンガルフィング）パターンのエントリーシグナル
    
    エントリー条件:
    - ロング: 陰線の実体を次の陽線の実体が完全に包み込む
    - ショート: 陽線の実体を次の陰線の実体が完全に包み込む
    """
    
    def __init__(self):
        """コンストラクタ"""
        super().__init__("Engulfing")
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            シグナル配列（1: ロング, -1: ショート, 0: シグナルなし）
        """
        # データフレームの作成
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        
        # 実体の計算
        body_high = np.maximum(df['open'].values, df['close'].values)
        body_low = np.minimum(df['open'].values, df['close'].values)
        
        # 陽線・陰線の判定
        is_bullish = df['close'].values > df['open'].values
        
        # シグナルの初期化
        signals = np.zeros(len(data))
        
        # 2番目の要素からシグナルを計算
        for i in range(1, len(data)):
            # 前日と当日の実体の高値・安値
            prev_body_high = body_high[i-1]
            prev_body_low = body_low[i-1]
            curr_body_high = body_high[i]
            curr_body_low = body_low[i]
            
            # 前日が陰線で当日が陽線の場合（ブル・エンガルフィング）
            if not is_bullish[i-1] and is_bullish[i]:
                if curr_body_high > prev_body_high and curr_body_low < prev_body_low:
                    signals[i] = 1
            
            # 前日が陽線で当日が陰線の場合（ベア・エンガルフィング）
            elif is_bullish[i-1] and not is_bullish[i]:
                if curr_body_high > prev_body_high and curr_body_low < prev_body_low:
                    signals[i] = -1
        
        return signals 