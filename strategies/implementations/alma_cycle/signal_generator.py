#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any
import numpy as np
import pandas as pd

from signals.implementations.alma.direction import ALMACirculationSignal
from strategies.base.signal_generator import BaseSignalGenerator

class ALMACycleSignalGenerator(BaseSignalGenerator):
    """ALMAサイクルストラテジーのシグナルジェネレーター"""
    
    def __init__(self, sigma: float = 6.0, offset: float = 0.85, params: Dict[str, Any] = None):
        """
        コンストラクタ
        
        Args:
            sigma: ガウス分布の標準偏差
            offset: 重みの中心位置（0-1）
            params: パラメータ辞書
                - short_period: 短期ALMAの期間
                - middle_period: 中期ALMAの期間
                - long_period: 長期ALMAの期間
        """
        # デフォルトのパラメータ設定
        default_params = {
            'short_period': 9,
            'middle_period': 21,
            'long_period': 55
        }
        
        # パラメータのマージ
        self._params = params or default_params
        
        # 名前の生成
        name = f"ALMACycle({self._params['short_period']}, {self._params['middle_period']}, {self._params['long_period']})"
        
        super().__init__(name)
        
        # ALMACirculationSignalのパラメータを設定
        alma_params = {
            'short_period': self._params['short_period'],
            'middle_period': self._params['middle_period'],
            'long_period': self._params['long_period']
        }
        
        self._alma_signal = ALMACirculationSignal(
            sigma=sigma,
            offset=offset,
            params=alma_params
        )
        
    def calculate_signals(self, data: pd.DataFrame) -> None:
        """
        全てのシグナルを計算してキャッシュする
        
        Args:
            data: 価格データ
        """
        self._signals = self._alma_signal.generate(data)
    
    def get_entry_signals(self, data: pd.DataFrame, position: int = 0) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート、0: なし）
            
        Returns:
            エントリーシグナル配列 (1: ロング、-1: ショート、0: エントリーなし)
        """
        signals = np.zeros(len(data))
        
        # ステージ6でロングエントリー（ショートポジションがない場合）
        long_entry = (self._signals == 6) & (position != -1)
        signals = np.where(long_entry, 1, signals)
        
        # ステージ3でショートエントリー（ロングポジションがない場合）
        short_entry = (self._signals == 3) & (position != 1)
        signals = np.where(short_entry, -1, signals)
        
        return signals
    
    def get_exit_signals(self, data: pd.DataFrame, position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            エグジットすべきかどうか
        """
        if index == -1:
            index = len(data) - 1
            
        current_stage = self._signals[index]
        
        # ロングポジションのエグジット条件
        if position == 1:
            # ステージ3でエグジット、またはステージが4,5に逆行した場合
            return current_stage == 3 or current_stage in [4, 5]
        
        # ショートポジションのエグジット条件
        elif position == -1:
            # ステージ6でエグジット、またはステージが1,2に逆行した場合
            return current_stage == 6 or current_stage in [1, 2]
        
        return False 