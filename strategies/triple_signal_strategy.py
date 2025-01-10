#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any

import numpy as np
import pandas as pd

from .strategy import Strategy
from signals.direction_signal import SupertrendDirectionSignal
from signals.entry_signal import RSIEntrySignal
from signals.filter_signal import ChopFilterSignal
from signals.exit_signal import RSIExitSignal

class TripleSignalStrategy(Strategy):
    """
    3つのシグナルを組み合わせた戦略
    
    エントリー条件:
    - スーパートレンド、RSIエントリー、チョピネスフィルターが全て1の場合: ロングエントリー
    - スーパートレンド、RSIエントリー、チョピネスフィルターが全て-1の場合: ショートエントリー
    
    エグジット条件:
    - ロングポジション: スーパートレンドが-1に切り替わる or RSIエグジットが1になる
    - ショートポジション: スーパートレンドが1に切り替わる
    """
    
    def __init__(
        self,
        supertrend_params: Dict[str, Any] = None,
        rsi_entry_params: Dict[str, Any] = None,
        rsi_exit_params: Dict[str, Any] = None,
        chop_params: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            supertrend_params: スーパートレンドのパラメータ
                - period: ATRの期間
                - multiplier: ATRの乗数
            rsi_entry_params: RSIエントリーのパラメータ
                - period: RSIの期間
                - solid: RSIのしきい値設定
            rsi_exit_params: RSIエグジットのパラメータ
                - period: RSIの期間
                - solid: RSIのしきい値設定
            chop_params: チョピネスインデックスのパラメータ
                - period: チョピネスの期間
                - solid: チョピネスのしきい値設定
        """
        super().__init__("TripleSignalStrategy")
        
        # デフォルトパラメータの設定
        self.supertrend_params = supertrend_params or {
            'period': 10,
            'multiplier': 3.0
        }
        self.rsi_entry_params = rsi_entry_params or {
            'period': 2,
            'solid': {
                'rsi_long_entry': 20,
                'rsi_short_entry': 80
            }
        }
        self.rsi_exit_params = rsi_exit_params or {
            'period': 14,
            'solid': {
                'rsi_long_exit_solid': 70,
                'rsi_short_exit_solid': 30
            }
        }
        self.chop_params = chop_params or {
            'period': 14,
            'solid': {
                'chop_solid': 50
            }
        }
        
        # シグナルの初期化
        self.supertrend = SupertrendDirectionSignal(
            period=self.supertrend_params['period'],
            multiplier=self.supertrend_params['multiplier']
        )
        self.rsi_entry = RSIEntrySignal(
            period=self.rsi_entry_params['period'],
            solid=self.rsi_entry_params['solid']
        )
        self.rsi_exit = RSIExitSignal(
            period=self.rsi_exit_params['period'],
            solid=self.rsi_exit_params['solid']
        )
        self.chop = ChopFilterSignal(
            period=self.chop_params['period'],
            solid=self.chop_params['solid']
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: ニュートラル)
        """
        # 各シグナルを生成
        supertrend_signals = self.supertrend.generate(data)
        rsi_signals = self.rsi_entry.generate(data)
        chop_signals = self.chop.generate(data)
        
        # シグナルの初期化
        signals = np.zeros(len(supertrend_signals))
        
        # ロングエントリー条件: 全てのシグナルが1
        long_condition = (
            (supertrend_signals == 1) &
            (rsi_signals == 1) &
            (chop_signals == 1)
        )
        
        # ショートエントリー条件: 全てのシグナルが-1
        short_condition = (
            (supertrend_signals == -1) &
            (rsi_signals == -1) &
            (chop_signals == -1)
        )
        
        # シグナルの生成
        signals = np.where(long_condition, 1, signals)
        signals = np.where(short_condition, -1, signals)
        
        return signals
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション (1: ロング, -1: ショート, 0: ニュートラル)
        
        Returns:
            True: エグジット, False: ホールド
        """
        if position == 0:
            return False
        
        # 各シグナルを生成
        supertrend_signals = self.supertrend.generate(data)
        rsi_exit_signals = self.rsi_exit.generate(data)
        
        # 最新のシグナルを取得
        current_supertrend = supertrend_signals[-1]
        current_rsi_exit = rsi_exit_signals[-1]
        
        # ロングポジションのエグジット条件
        if position == 1:
            return (current_supertrend == -1) or (current_rsi_exit == 1)
        
        # ショートポジションのエグジット条件
        if position == -1:
            return current_supertrend == 1 or (current_rsi_exit == -1)
        
        return False 