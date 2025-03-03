#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import SellCounterPredicterSignalGenerator


class SellCounterPredicter(BaseStrategy):
    """
    複数のシグナルをポイント制で評価する逆張り売り戦略
    
    エントリー条件:
    - 各シグナルのポイントの合計がエントリーしきい値を超えた場合に売りシグナル
    
    エグジット条件:
    - 各シグナルのポイントの合計がエグジットしきい値を超えた場合に買いシグナル
    """
    
    def __init__(
        self,
        entry_threshold: float = 10.0,
        exit_threshold: float = 10.0,
        roc_div_entry_weight: int = 4,
        macd_div_entry_weight: int = 2,
        rsi_filter_weight: int = 2,
        adx_weight: int = 4,
        bollinger_entry_weight: int = 4,
        rsi_counter_entry_weight: int = 4,
        donchian_entry_weight: int = 4,
        alma_entry_weight: int = 6,
        pinbar_exit_weight: int = 6,
        roc_div_exit_weight: int = 3,
        macd_div_exit_weight: int = 3,
        bollinger_exit_weight: int = 6,
        rsi_counter_exit_weight: int = 6,
        donchian_exit_weight: int = 6,
        alma_exit_weight: int = 7,
    ):
        """
        初期化
        
        Args:
            entry_threshold: エントリーのしきい値
            exit_threshold: エグジットのしきい値
            各シグナルの重み付けパラメータ
        """
        super().__init__("SellCounterPredicter")
        
        # パラメータの設定
        self._parameters = {
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'roc_div_entry_weight': roc_div_entry_weight,
            'macd_div_entry_weight': macd_div_entry_weight,
            'rsi_filter_weight': rsi_filter_weight,
            'adx_weight': adx_weight,
            'bollinger_entry_weight': bollinger_entry_weight,
            'rsi_counter_entry_weight': rsi_counter_entry_weight,
            'donchian_entry_weight': donchian_entry_weight,
            'alma_entry_weight': alma_entry_weight,
            'pinbar_exit_weight': pinbar_exit_weight,
            'roc_div_exit_weight': roc_div_exit_weight,
            'macd_div_exit_weight': macd_div_exit_weight,
            'bollinger_exit_weight': bollinger_exit_weight,
            'rsi_counter_exit_weight': rsi_counter_exit_weight,
            'donchian_exit_weight': donchian_exit_weight,
            'alma_exit_weight': alma_exit_weight,
        }
        
        # エントリーポイントの設定
        entry_points = np.array([
            roc_div_entry_weight,       # ROCDivergenceSignal
            macd_div_entry_weight,      # MACDDivergenceSignal
            rsi_filter_weight,          # RSIFilterSignal
            adx_weight,                 # ADX
            bollinger_entry_weight,     # Bollinger Counter Trend
            rsi_counter_entry_weight,   # RSICounterTrendEntrySignal
            donchian_entry_weight,      # DonchianBreakoutEntrySignal
            alma_entry_weight,          # ALMACirculationSignal
        ])
        
        # エグジットポイントの設定
        exit_points = np.array([
            pinbar_exit_weight,         # Pinbar
            roc_div_exit_weight,        # ROCDivergenceSignal
            macd_div_exit_weight,       # MACDDivergenceSignal
            bollinger_exit_weight,      # BollingerBreakoutExitSignal
            rsi_counter_exit_weight,    # RSICounterTrendEntrySignal
            donchian_exit_weight,       # DonchianBreakoutEntrySignal
            alma_exit_weight,           # ALMACirculationSignal
        ])
        
        # シグナル生成器の初期化
        self.signal_generator = SellCounterPredicterSignalGenerator(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            entry_points=entry_points,
            exit_points=exit_points,
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル
        """
        return self.signal_generator.get_entry_signals(data)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        return self.signal_generator.get_exit_signals(data, position, index)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成
        
        Args:
            trial: Optunaのトライアル
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        params = {
            'roc_div_entry_weight': trial.suggest_int('roc_div_entry_weight', 1, 8),
            'macd_div_entry_weight': trial.suggest_int('macd_div_entry_weight', 1, 8),
            'rsi_filter_weight': trial.suggest_int('rsi_filter_weight', 1, 8),
            'adx_weight': trial.suggest_int('adx_weight', 1, 8),
            'bollinger_entry_weight': trial.suggest_int('bollinger_entry_weight', 1, 8),
            'rsi_counter_entry_weight': trial.suggest_int('rsi_counter_entry_weight', 1, 8),
            'donchian_entry_weight': trial.suggest_int('donchian_entry_weight', 1, 8),
            'alma_entry_weight': trial.suggest_int('alma_entry_weight', 1, 8),
            'pinbar_exit_weight': trial.suggest_int('pinbar_exit_weight', 1, 8),
            'roc_div_exit_weight': trial.suggest_int('roc_div_exit_weight', 1, 8),
            'macd_div_exit_weight': trial.suggest_int('macd_div_exit_weight', 1, 8),
            'bollinger_exit_weight': trial.suggest_int('bollinger_exit_weight', 1, 8),
            'rsi_counter_exit_weight': trial.suggest_int('rsi_counter_exit_weight', 1, 8),
            'donchian_exit_weight': trial.suggest_int('donchian_exit_weight', 1, 8),
            'alma_exit_weight': trial.suggest_int('alma_exit_weight', 1, 8),
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            Dict[str, Any]: 戦略パラメータ
        """
        return {
            'entry_threshold': 10.0,  # デフォルト値を使用
            'exit_threshold': 10.0,   # デフォルト値を使用
            'roc_div_entry_weight': int(params['roc_div_entry_weight']),
            'macd_div_entry_weight': int(params['macd_div_entry_weight']),
            'rsi_filter_weight': int(params['rsi_filter_weight']),
            'adx_weight': int(params['adx_weight']),
            'bollinger_entry_weight': int(params['bollinger_entry_weight']),
            'rsi_counter_entry_weight': int(params['rsi_counter_entry_weight']),
            'donchian_entry_weight': int(params['donchian_entry_weight']),
            'alma_entry_weight': int(params['alma_entry_weight']),
            'pinbar_exit_weight': int(params['pinbar_exit_weight']),
            'roc_div_exit_weight': int(params['roc_div_exit_weight']),
            'macd_div_exit_weight': int(params['macd_div_exit_weight']),
            'bollinger_exit_weight': int(params['bollinger_exit_weight']),
            'rsi_counter_exit_weight': int(params['rsi_counter_exit_weight']),
            'donchian_exit_weight': int(params['donchian_exit_weight']),
            'alma_exit_weight': int(params['alma_exit_weight']),
        } 