#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple

import numpy as np
import pandas as pd

from .strategy import Strategy
from signals.direction_signal import SupertrendDirectionSignal
from signals.entry_signal import RSIEntrySignal
from signals.filter_signal import ChopFilterSignal
from signals.exit_signal import RSIExitSignal

class SupertrendRsiChopStrategy(Strategy):
    """
    3つのシグナルを組み合わせた戦略
    
    エントリー条件:
    - スーパートレンド、RSIエントリー、チョピネスフィルターが全て1の場合: ロングエントリー
    - スーパートレンド、RSIエントリー、チョピネスフィルターが全て-1の場合: ショートエントリー
    
    エグジット条件:
    - ロングポジション: スーパートレンドが-1に切り替わる or RSIエグジットが1になる
    - ショートポジション: スーパートレンドが1に切り替わる or RSIエグジットが-1になる
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
        
        # シグナルのキャッシュ
        self._supertrend_signals = None
        self._rsi_exit_signals = None
        self._rsi_entry_signals = None
        self._chop_signals = None

    @staticmethod
    def create_optimization_params(trial) -> Dict[str, Any]:
        """最適化用のパラメータを生成する

        Args:
            trial: Optunaのtrialオブジェクト

        Returns:
            Dict[str, Any]: 戦略パラメータの辞書
        """
        return {
            'supertrend_params': {
                'period': trial.suggest_int('supertrend_period', 3, 100, step=1),
                'multiplier': trial.suggest_float('supertrend_multiplier', 1.5, 8.0, step=0.5)
            },
            'rsi_entry_params': {
                'period': 2,
                'solid': {
                    'rsi_long_entry': 20,
                    'rsi_short_entry': 80
                }
            },
            'rsi_exit_params': {
                'period': trial.suggest_int('rsi_exit_period', 7, 34, step=1),
                'solid': {
                    'rsi_long_exit_solid': 86,
                    'rsi_short_exit_solid': 14
                }
            },
            'chop_params': {
                'period': trial.suggest_int('chop_period', 3, 100, step=1),
                'solid': {
                    'chop_solid': 50
                }
            }
        }

    def get_parameters(self) -> Dict[str, Any]:
        """現在の戦略パラメータを取得する

        Returns:
            Dict[str, Any]: パラメータ名と値の辞書
        """
        return {
            'supertrend_params': self.supertrend_params,
            'rsi_entry_params': self.rsi_entry_params,
            'rsi_exit_params': self.rsi_exit_params,
            'chop_params': self.chop_params
        }
    
    def _calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """全てのシグナルを計算してキャッシュする"""
        if self._supertrend_signals is None:
            self._supertrend_signals = self.supertrend.generate(data)
        if self._rsi_exit_signals is None:
            self._rsi_exit_signals = self.rsi_exit.generate(data)
        if self._rsi_entry_signals is None:
            self._rsi_entry_signals = self.rsi_entry.generate(data)
        if self._chop_signals is None:
            self._chop_signals = self.chop.generate(data)
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: ニュートラル)
        """
        # シグナルの計算とキャッシュ
        self._calculate_signals(data)
        
        # シグナルの初期化
        signals = np.zeros(len(self._supertrend_signals))
        
        # ロングエントリー条件: 全てのシグナルが1
        long_condition = (
            (self._supertrend_signals == 1) &
            (self._rsi_entry_signals == 1) &
            (self._chop_signals == 1)
        )
        
        # ショートエントリー条件: 全てのシグナルが-1
        short_condition = (
            (self._supertrend_signals == -1) &
            (self._rsi_entry_signals == -1) &
            (self._chop_signals == -1)
        )
        
        # シグナルの生成
        signals = np.where(long_condition, 1, signals)
        signals = np.where(short_condition, -1, signals)
        
        return signals
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション (1: ロング, -1: ショート, 0: ニュートラル)
            index: データのインデックス（デフォルトは最新のデータ）
        
        Returns:
            True: エグジット, False: ホールド
        """
        if position == 0:
            return False
        
        # シグナルの計算とキャッシュ
        self._calculate_signals(data)
        
        # 指定されたインデックスのシグナルを取得
        current_supertrend = self._supertrend_signals[index]
        current_rsi_exit = self._rsi_exit_signals[index]
        
        # ロングポジションのエグジット条件
        if position == 1:
            # スーパートレンドが-1に切り替わる、またはRSIエグジットが1になる
            return (current_supertrend == -1) or (current_rsi_exit == 1)
        
        # ショートポジションのエグジット条件
        if position == -1:
            # スーパートレンドが1に切り替わる、またはRSIエグジットが-1になる
            return (current_supertrend == 1) or (current_rsi_exit == -1)
        
        return False 