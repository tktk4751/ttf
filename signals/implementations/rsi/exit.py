#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.exit import IExitSignal
from indicators.rsi import RSI

class RSIExitSignal(BaseSignal, IExitSignal):
    """
    RSIを使用したエグジットシグナル
    - ロングエグジット: RSI(t-1) >= exit_solid かつ RSI(t) <= exit_solid
    - ショートエグジット: RSI(t-1) <= exit_solid かつ RSI(t) >= exit_solid
    """
    
    def __init__(self, period: int = 14, solid: Dict[str, Any] = None):
        """
        コンストラクタ
        
        Args:
            period: RSIの期間
            solid: パラメータ辞書
                - rsi_long_exit_solid: ロングエグジットのRSIしきい値
                - rsi_short_exit_solid: ショートエグジットのRSIしきい値
        """
        params = {
            'period': period,
            'solid': solid or {
                'rsi_long_exit_solid': 86,
                'rsi_short_exit_solid': 14
            }
        }
        super().__init__(f"RSIExit({period})", params)
        self._rsi = RSI(period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロングエグジット, -1: ショートエグジット, 0: シグナルなし)
        """
        rsi_values = pd.Series(self._rsi.calculate(data))
        
        # シグナルの初期化
        signals = np.zeros(len(rsi_values))
        
        # エグジットシグナルの生成
        solid = self._params['solid']
        
        # ロングポジションのエグジット条件
        long_exit = (rsi_values.shift(1) >= solid['rsi_long_exit_solid']) & \
                   (rsi_values <= solid['rsi_long_exit_solid'])
        
        # ショートポジションのエグジット条件
        short_exit = (rsi_values.shift(1) <= solid['rsi_short_exit_solid']) & \
                    (rsi_values >= solid['rsi_short_exit_solid'])
        
        # シグナルの設定
        signals = np.where(long_exit, 1, signals)
        signals = np.where(short_exit, -1, signals)
        
        return signals 
    


class RSIExit2Signal(BaseSignal, IExitSignal):
    """
    RSIを使用したエグジットシグナル
    - ロングエグジット: RSI <= 30 (1を出力)
    - ショートエグジット: RSI >= 70 (-1を出力)
    """
    
    def __init__(self, period: int = 14, solid: Dict[str, Any] = None):
        """
        コンストラクタ
        
        Args:
            period: RSIの期間
            solid: パラメータ辞書
                - rsi_long_exit_solid: ロングエグジットのRSIしきい値
                - rsi_short_exit_solid: ショートエグジットのRSIしきい値
        """
        params = {
            'period': period,
            'solid': solid or {
                'rsi_long_exit_solid': 20,
                'rsi_short_exit_solid': 80
            }
        }
        super().__init__(f"RSIExit2({period})", params)
        self._rsi = RSI(period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロングエグジット, -1: ショートエグジット, 0: シグナルなし)
        """
        rsi_values = pd.Series(self._rsi.calculate(data))
        
        # シグナルの初期化
        signals = np.zeros(len(rsi_values))
        
        # エグジットシグナルの生成
        solid = self._params['solid']
        
        # ロングポジションのエグジット条件
        long_exit = rsi_values <= solid['rsi_long_exit_solid']
        
        # ショートポジションのエグジット条件
        short_exit = rsi_values >= solid['rsi_short_exit_solid']
        
        # シグナルの設定
        signals = np.where(long_exit, 1, signals)
        signals = np.where(short_exit, -1, signals)
        
        return signals 