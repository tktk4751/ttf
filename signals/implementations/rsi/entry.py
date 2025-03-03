#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.rsi import RSI


class RSIEntrySignal(BaseSignal, IEntrySignal):
    """
    RSIを使用したエントリーシグナル
    - RSI <= rsi_long_entry: ロングエントリー (1)
    - RSI >= rsi_short_entry: ショートエントリー (-1)
    """
    
    def __init__(self, period: int = 2, solid: Dict[str, Any] = None):
        """
        コンストラクタ
        
        Args:
            period: RSIの期間
            solid: パラメータ辞書
                - rsi_long_entry: ロングエントリーのRSIしきい値
                - rsi_short_entry: ショートエントリーのRSIしきい値
        """
        params = {
            'period': period,
            'solid': solid or {
                'rsi_long_entry': 20,
                'rsi_short_entry': 80
            }
        }
        super().__init__(f"RSIEntry({period})", params)
        self._rsi = RSI(period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        rsi_values = self._rsi.calculate(data)
        
        # シグナルの初期化
        signals = np.zeros(len(rsi_values))
        
        # エントリーシグナルの生成
        solid = self._params['solid']
        signals = np.where(rsi_values <= solid['rsi_long_entry'], 1, signals)  # ロングエントリー
        signals = np.where(rsi_values >= solid['rsi_short_entry'], -1, signals)  # ショートエントリー
        
        return signals 


class RSICounterTrendEntrySignal(BaseSignal, IEntrySignal):
    """
    RSIを使用した逆張りエントリーシグナル
    
    エントリー条件:
    - ロングエントリー: RSI(t-1) <= entry_solid かつ RSI(t) >= entry_solid
      （RSIが売られすぎの水準から上抜けた場合）
    - ショートエントリー: RSI(t-1) >= entry_solid かつ RSI(t) <= entry_solid
      （RSIが買われすぎの水準から下抜けた場合）
    """
    
    def __init__(
        self,
        period: int = 14,
        solid: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            period: RSIの期間
            solid: パラメータ辞書
                - rsi_long_entry_solid: ロングエントリーのRSIしきい値
                - rsi_short_entry_solid: ショートエントリーのRSIしきい値
        """
        params = {
            'period': period,
            'solid': solid or {
                'rsi_long_entry_solid': 14,  # エグジットの逆の値を使用
                'rsi_short_entry_solid': 86  # エグジットの逆の値を使用
            }
        }
        super().__init__(f"RSICounterTrendEntry({period})", params)
        
        # パラメータの設定
        self.period = period
        self.solid = params['solid']
        
        # インジケーターの初期化
        self._rsi = RSI(period)
    
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
        
        # RSIの計算
        rsi_values = self._rsi.calculate(data)
        
        # シグナルの初期化
        signals = np.zeros(len(data))
        
        # 2番目の要素からシグナルを計算
        for i in range(1, len(data)):
            # ロングエントリー（RSIが売られすぎ水準から上抜けた場合）
            if (rsi_values[i-1] <= self.solid['rsi_long_entry_solid'] and 
                rsi_values[i] >= self.solid['rsi_long_entry_solid']):
                signals[i] = 1
            
            # ショートエントリー（RSIが買われすぎ水準から下抜けた場合）
            elif (rsi_values[i-1] >= self.solid['rsi_short_entry_solid'] and 
                  rsi_values[i] <= self.solid['rsi_short_entry_solid']):
                signals[i] = -1
        
        return signals 