#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any
import optuna
import pandas as pd
import numpy as np

from strategies.base.strategy import BaseStrategy
from .signal_generator import ALMACycleSignalGenerator

class ALMACycleStrategy(BaseStrategy):
    """ALMAサイクルストラテジー"""
    
    def __init__(
        self,
        short_period: int = 9,
        middle_period: int = 21,
        long_period: int = 55,
        sigma: float = 6.0,
        offset: float = 0.85,
        name: str = "ALMACycle"
    ):
        """
        コンストラクタ
        
        Args:
            short_period: 短期ALMAの期間
            middle_period: 中期ALMAの期間
            long_period: 長期ALMAの期間
            sigma: ガウス分布の標準偏差
            offset: 重みの中心位置（0-1）
            name: 戦略名
        """
        super().__init__(name)
        
        # パラメータの設定
        self._parameters = {
            'short_period': short_period,
            'middle_period': middle_period,
            'long_period': long_period,
            'sigma': sigma,
            'offset': offset
        }
        
        # シグナルジェネレーターの作成
        self._signal_generator = ALMACycleSignalGenerator(
            sigma=sigma,
            offset=offset,
            params=self._parameters
        )
        
        # 現在のポジション（0: なし、1: ロング、-1: ショート）
        self._position = 0
    
    def generate_entry(self, data: pd.DataFrame) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            エントリーシグナル配列 (1: ロング、-1: ショート、0: エントリーなし)
        """
        # シグナルの計算
        self._signal_generator.calculate_signals(data)
        
        # エントリーシグナルを生成
        signals = self._signal_generator.get_entry_signals(data, self._position)
        
        # 最後のシグナルに基づいてポジションを更新
        if signals[-1] != 0:
            self._position = signals[-1]
        
        return signals
    
    def generate_exit(self, data: pd.DataFrame, position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            エグジットすべきかどうか
        """
        # エグジットシグナルを生成
        should_exit = self._signal_generator.get_exit_signals(data, position, index)
        
        # エグジットする場合はポジションをクリア
        if should_exit:
            self._position = 0
            
        return should_exit
    
    @staticmethod
    def create_optimization_params(trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化用のパラメータを生成する
        
        Args:
            trial: Optunaのトライアル
            
        Returns:
            最適化パラメータ
        """
        return {
            'short_period': trial.suggest_int('short_period', 5, 40, step=1),
            'middle_period': trial.suggest_int('middle_period', 40, 120, step=1),
            'long_period': trial.suggest_int('long_period', 120, 400, step=1),
            'sigma': 6,
            'offset': 0.85
        }
    
    @staticmethod
    def convert_params_to_strategy_format(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略クラスの形式に変換
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            戦略クラスのパラメータ
        """
        return {
            'short_period': params['short_period'],
            'middle_period': params['middle_period'],
            'long_period': params['long_period'],
            'sigma': params.get('sigma', 6.0),
            'offset': params.get('offset', 0.85)
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        現在の戦略パラメータを取得する
        
        Returns:
            戦略パラメータ
        """
        return self._parameters 