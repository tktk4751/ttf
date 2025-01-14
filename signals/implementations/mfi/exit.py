#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.exit import IExitSignal
from indicators.mfi import MFI

class MFIExitSignal(BaseSignal, IExitSignal):
    """
    MFIを使用したエグジットシグナル
    - ロングエグジット: MFI(t-1) >= exit_solid かつ MFI(t) <= exit_solid
    - ショートエグジット: MFI(t-1) <= exit_solid かつ MFI(t) >= exit_solid
    """
    
    def __init__(self, period: int = 14, solid: Dict[str, Any] = None):
        """
        コンストラクタ
        
        Args:
            period: MFIの期間
            solid: パラメータ辞書
                - mfi_long_exit_solid: ロングエグジットのMFIしきい値
                - mfi_short_exit_solid: ショートエグジットのMFIしきい値
        """
        params = {
            'period': period,
            'solid': solid or {
                'mfi_long_exit_solid': 85,
                'mfi_short_exit_solid': 15
            }
        }
        super().__init__(f"MFIExit({period})", params)
        self._mfi = MFI(period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロングエグジット, -1: ショートエグジット, 0: シグナルなし)
        """
        mfi_values = pd.Series(self._mfi.calculate(data))
        
        # シグナルの初期化
        signals = np.zeros(len(mfi_values))
        
        # エグジットシグナルの生成
        solid = self._params['solid']
        
        # ロングポジションのエグジット条件
        long_exit = (mfi_values.shift(1) >= solid['mfi_long_exit_solid']) & \
                   (mfi_values <= solid['mfi_long_exit_solid'])
        
        # ショートポジションのエグジット条件
        short_exit = (mfi_values.shift(1) <= solid['mfi_short_exit_solid']) & \
                    (mfi_values >= solid['mfi_short_exit_solid'])
        
        # シグナルの設定
        signals = np.where(long_exit, 1, signals)
        signals = np.where(short_exit, -1, signals)
        
        return signals 