#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal


class PinbarEntrySignal(BaseSignal, IEntrySignal):
    """
    ピンバー（ハンマー/シューティングスター）パターンのエントリーシグナル
    
    エントリー条件:
    - ロング（ハンマー）: 下ヒゲが実体の2倍以上で、上ヒゲが実体の0.5倍以下
    - ショート（シューティングスター）: 上ヒゲが実体の2倍以上で、下ヒゲが実体の0.5倍以下
    """
    
    def __init__(
        self,
        body_ratio: float = 0.25,  # ヒゲと実体の比率
        shadow_ratio: float = 3.0,  # 長いヒゲと短いヒゲの比率
    ):
        """
        コンストラクタ
        
        Args:
            body_ratio: 実体とローソク足全体の比率の閾値
            shadow_ratio: 長いヒゲと短いヒゲの比率の閾値
        """
        super().__init__("Pinbar")
        self.body_ratio = body_ratio
        self.shadow_ratio = shadow_ratio
    
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
        body_size = body_high - body_low
        
        # ヒゲの計算
        upper_shadow = df['high'].values - body_high
        lower_shadow = body_low - df['low'].values
        
        # ローソク足の全体の大きさ
        candle_size = df['high'].values - df['low'].values
        
        # シグナルの初期化
        signals = np.zeros(len(data))
        
        for i in range(len(data)):
            # 実体が小さすぎる場合はスキップ
            if body_size[i] == 0:
                continue
            
            # 実体の比率が閾値以下であることを確認
            if body_size[i] / candle_size[i] <= self.body_ratio:
                # ハンマー（ロング）パターン
                if (lower_shadow[i] >= self.shadow_ratio * body_size[i] and 
                    upper_shadow[i] <= 0.5 * body_size[i]):
                    signals[i] = 1
                
                # シューティングスター（ショート）パターン
                elif (upper_shadow[i] >= self.shadow_ratio * body_size[i] and 
                      lower_shadow[i] <= 0.5 * body_size[i]):
                    signals[i] = -1
        
        return signals 