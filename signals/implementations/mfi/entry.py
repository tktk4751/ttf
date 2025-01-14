#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.mfi import MFI

class MFIEntrySignal(BaseSignal, IEntrySignal):
    """
    MFIを使用したエントリーシグナル
    - MFI <= mfi_long_entry: ロングエントリー (1)
    - MFI >= mfi_short_entry: ショートエントリー (-1)
    """
    
    def __init__(self, period: int = 14, solid: Dict[str, Any] = None):
        """
        コンストラクタ
        
        Args:
            period: MFIの期間
            solid: パラメータ辞書
                - mfi_long_entry: ロングエントリーのMFIしきい値
                - mfi_short_entry: ショートエントリーのMFIしきい値
        """
        params = {
            'period': period,
            'solid': solid or {
                'mfi_long_entry': 20,
                'mfi_short_entry': 80
            }
        }
        super().__init__(f"MFIEntry({period})", params)
        self._mfi = MFI(period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        mfi_values = self._mfi.calculate(data)
        
        # シグナルの初期化
        signals = np.zeros(len(mfi_values))
        
        # エントリーシグナルの生成
        solid = self._params['solid']
        signals = np.where(mfi_values <= solid['mfi_long_entry'], 1, signals)  # ロングエントリー
        signals = np.where(mfi_values >= solid['mfi_short_entry'], -1, signals)  # ショートエントリー
        
        return signals 