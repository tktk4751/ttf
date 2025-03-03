#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import SpanModelChopLongSignalGenerator

class SpanModelChopLongStrategy(BaseStrategy):
    """スパンモデル+チョピネスフィルターの買い専用戦略"""
    
    def __init__(
        self,
        conversion_period: int = 9,
        base_period: int = 26,
        span_b_period: int = 52,
        displacement: int = 26,
        chop_period: int = 14,
        chop_threshold: float = 50.0,
    ):
        """
        初期化
        
        Args:
            conversion_period: 転換線期間
            base_period: 基準線期間
            span_b_period: スパンB期間
            displacement: 先行スパン移動期間
            chop_period: チョピネスインデックスの期間
            chop_threshold: チョピネスインデックスのしきい値
        """
        super().__init__("SpanModelChopLong")
        
        # パラメータの保存
        self._parameters = {
            'conversion_period': conversion_period,
            'base_period': base_period,
            'span_b_period': span_b_period,
            'displacement': displacement,
            'chop_period': chop_period,
            'chop_threshold': chop_threshold,
        }
        
        # シグナル生成器の初期化
        self._signal_generator = SpanModelChopLongSignalGenerator(
            conversion_period=conversion_period,
            base_period=base_period,
            span_b_period=span_b_period,
            displacement=displacement,
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
            'conversion_period': trial.suggest_int('conversion_period', 3, 150),
            'base_period': trial.suggest_int('base_period', 8, 400),
            'span_b_period': trial.suggest_int('span_b_period', 20, 400),
            'displacement': 26,  # 固定
            'chop_period':55,
            'chop_threshold': 50,
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換"""
        return {
            'conversion_period': int(params['conversion_period']),
            'base_period': int(params['base_period']),
            'span_b_period': int(params['span_b_period']),
            'displacement': 26,  # 固定
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
        if position != 1:  # ロングポジションのみ対応
            raise ValueError("This strategy only supports long positions")
            
        if isinstance(data, pd.DataFrame):
            return data['close'].iloc[index]
        return data[index, 3]  # close価格のインデックス 