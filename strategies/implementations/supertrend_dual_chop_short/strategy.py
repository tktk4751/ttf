#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import SupertrendDualChopShortSignalGenerator

class SupertrendDualChopShortStrategy(BaseStrategy):
    """デュアルスーパートレンド+チョピネスフィルターの売り専用戦略"""
    
    def __init__(
        self,
        period: int = 10,
        fast_multiplier: float = 2.0,
        slow_multiplier: float = 4.0,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
    ):
        """
        初期化
        
        Args:
            period: スーパートレンドの期間
            fast_multiplier: ファストスーパートレンドの乗数
            slow_multiplier: スロウスーパートレンドの乗数
            chop_period: チョピネスインデックスの期間
            chop_threshold: チョピネスインデックスのしきい値
        """
        super().__init__("SupertrendDualChopShort")
        
        # パラメータの保存
        self._parameters = {
            'period': period,
            'fast_multiplier': fast_multiplier,
            'slow_multiplier': slow_multiplier,
            'chop_period': chop_period,
            'chop_threshold': chop_threshold,
        }
        
        # シグナル生成器の初期化
        self._signal_generator = SupertrendDualChopShortSignalGenerator(
            period=period,
            fast_multiplier=fast_multiplier,
            slow_multiplier=slow_multiplier,
            chop_period=chop_period,
            chop_threshold=chop_threshold,
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを生成"""
        return self._signal_generator.get_entry_signals(data)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成"""
        return self._signal_generator.get_exit_signals(data, position, index)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """最適化パラメータを生成"""
        return {
            'period': trial.suggest_int('period', 3, 120),
            'fast_multiplier': trial.suggest_float('fast_multiplier', 1.0, 3.0, step=0.1),
            'slow_multiplier': trial.suggest_float('slow_multiplier', 3.1, 7.0, step=0.1),
            'chop_period': 55,
            'chop_threshold': 50,
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'period': int(params['period']),
            'fast_multiplier': float(params['fast_multiplier']),
            'slow_multiplier': float(params['slow_multiplier']),
            'chop_period': 55,
            'chop_threshold': 50,
        }

    def get_entry_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """
        エントリー価格を取得する
        
        Args:
            data: 価格データ
            position: ポジション方向 (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1 = 最新のデータ）
            
        Returns:
            float: エントリー価格
        """
        if position != -1:  # ショートポジションのみ対応
            raise ValueError("This strategy only supports short positions")
            
        if isinstance(data, pd.DataFrame):
            return data['close'].iloc[index]
        return data[index, 3]  # close価格のインデックス

    def get_stop_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """
        ストップロス価格を取得する
        
        Args:
            data: 価格データ
            position: ポジション方向 (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1 = 最新のデータ）
            
        Returns:
            float: ストップロス価格（スロウスーパートレンドの上限値）
        """
        if position != -1:  # ショートポジションのみ対応
            raise ValueError("This strategy only supports short positions")
            
        # シグナル生成器からスーパートレンドの上限値を取得
        return self._signal_generator.get_stop_price(data, index) 