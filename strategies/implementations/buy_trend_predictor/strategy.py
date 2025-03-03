#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import BuyTrendpredictorSignalGenerator


class BuyTrendpredictor(BaseStrategy):
    """
    複数のシグナルをポイント制で評価する戦略（買い専用）
    
    エントリー条件:
    - 各シグナルのポイントの合計がエントリーしきい値を超えた場合に買いシグナル
    
    エグジット条件:
    - 各シグナルのポイントの合計がエグジットしきい値を超えた場合に売りシグナル
    """
    
    def __init__(
        self,
        entry_threshold: float = 10.0,
        exit_threshold: float = 10.0,
        supertrend_entry_weight: int = 2,
        supertrend2_entry_weight: int = 2,
        keltner_entry_weight: int = 2,
        keltner2_entry_weight: int = 2,
        donchian_entry_weight: int = 2,
        chop_entry_weight: int = 2,
        adx_entry_weight: int = 2,
        kama_trend_entry_weight: int = 2,
        roc_entry_weight: int = 2,
        squeeze_entry_weight: int = 2,
        squeeze2_entry_weight: int = 2,
        alma_crossover_entry_weight: int = 2,
        alma_crossover2_entry_weight: int = 2,
        rsi_entry_weight: int = 2,
        supertrend_exit_weight: int = 4,
        supertrend2_exit_weight: int = 4,
        keltner_exit_weight: int = 4,
        keltner2_exit_weight: int = 4,
        donchian_exit_weight: int = 4,
        chop_exit_weight: int = 4,
        adx_exit_weight: int = 4,
        kama_trend_exit_weight: int = 4,
        roc_exit_weight: int = 4,
        squeeze_exit_weight: int = 4,
        squeeze2_exit_weight: int = 4,
        roc_div_exit_weight: int = 4,
        bollinger_exit_weight: int = 4,
        rsi_exit_weight: int = 4,
    ):
        """
        初期化
        
        Args:
            entry_threshold: エントリーのしきい値
            exit_threshold: エグジットのしきい値
            各シグナルの重み
        """
        super().__init__("BuyTrendpredictor")
        
        # パラメータの設定
        self._parameters = {
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'supertrend_entry_weight': supertrend_entry_weight,
            'supertrend2_entry_weight': supertrend2_entry_weight,
            'keltner_entry_weight': keltner_entry_weight,
            'keltner2_entry_weight': keltner2_entry_weight,
            'donchian_entry_weight': donchian_entry_weight,
            'chop_entry_weight': chop_entry_weight,
            'adx_entry_weight': adx_entry_weight,
            'kama_trend_entry_weight': kama_trend_entry_weight,
            'roc_entry_weight': roc_entry_weight,
            'squeeze_entry_weight': squeeze_entry_weight,
            'squeeze2_entry_weight': squeeze2_entry_weight,
            'alma_crossover_entry_weight': alma_crossover_entry_weight,
            'alma_crossover2_entry_weight': alma_crossover2_entry_weight,
            'rsi_entry_weight': rsi_entry_weight,
            'supertrend_exit_weight': supertrend_exit_weight,
            'supertrend2_exit_weight': supertrend2_exit_weight,
            'keltner_exit_weight': keltner_exit_weight,
            'keltner2_exit_weight': keltner2_exit_weight,
            'donchian_exit_weight': donchian_exit_weight,
            'chop_exit_weight': chop_exit_weight,
            'adx_exit_weight': adx_exit_weight,
            'kama_trend_exit_weight': kama_trend_exit_weight,
            'roc_exit_weight': roc_exit_weight,
            'squeeze_exit_weight': squeeze_exit_weight,
            'squeeze2_exit_weight': squeeze2_exit_weight,
            'roc_div_exit_weight': roc_div_exit_weight,
            'bollinger_exit_weight': bollinger_exit_weight,
            'rsi_exit_weight': rsi_exit_weight,
        }
        
        # エントリーポイントの設定
        entry_points = np.array([
            supertrend_entry_weight,
            supertrend2_entry_weight,
            keltner_entry_weight,
            keltner2_entry_weight,
            donchian_entry_weight,
            chop_entry_weight,
            adx_entry_weight,
            kama_trend_entry_weight,
            roc_entry_weight,
            squeeze_entry_weight,
            squeeze2_entry_weight,
            alma_crossover_entry_weight,
            alma_crossover2_entry_weight,
            rsi_entry_weight,
        ])
        
        # エグジットポイントの設定
        exit_points = np.array([
            supertrend_exit_weight,
            supertrend2_exit_weight,
            keltner_exit_weight,
            keltner2_exit_weight,
            donchian_exit_weight,
            chop_exit_weight,
            adx_exit_weight,
            kama_trend_exit_weight,
            roc_exit_weight,
            squeeze_exit_weight,
            squeeze2_exit_weight,
            roc_div_exit_weight,
            bollinger_exit_weight,
            rsi_exit_weight,
        ])
        
        # シグナル生成器の初期化
        self.signal_generator = BuyTrendpredictorSignalGenerator(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            entry_points=entry_points,
            exit_points=exit_points
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
            'entry_threshold': 10,
            'exit_threshold': 10,
            'supertrend_entry_weight': trial.suggest_int('supertrend_entry_weight', 1, 8),
            'supertrend2_entry_weight': trial.suggest_int('supertrend2_entry_weight', 1, 8),
            'keltner_entry_weight': trial.suggest_int('keltner_entry_weight', 1, 8),
            'keltner2_entry_weight': trial.suggest_int('keltner2_entry_weight', 1, 8),
            'donchian_entry_weight': trial.suggest_int('donchian_entry_weight', 1, 8),
            'chop_entry_weight': trial.suggest_int('chop_entry_weight', 1, 8),
            'adx_entry_weight': trial.suggest_int('adx_entry_weight', 1, 8),
            'kama_trend_entry_weight': trial.suggest_int('kama_trend_entry_weight', 1, 8),
            'roc_entry_weight': trial.suggest_int('roc_entry_weight', 1, 8),
            'squeeze_entry_weight': trial.suggest_int('squeeze_entry_weight', 1, 8),
            'squeeze2_entry_weight': trial.suggest_int('squeeze2_entry_weight', 1, 8),
            'alma_crossover_entry_weight': trial.suggest_int('alma_crossover_entry_weight', 1, 8),
            'alma_crossover2_entry_weight': trial.suggest_int('alma_crossover2_entry_weight', 1, 8),
            'rsi_entry_weight': trial.suggest_int('rsi_entry_weight', 1, 8),
            'supertrend_exit_weight': trial.suggest_int('supertrend_exit_weight', 1, 8),
            'supertrend2_exit_weight': trial.suggest_int('supertrend2_exit_weight', 1, 8),
            'keltner_exit_weight': trial.suggest_int('keltner_exit_weight', 1, 8),
            'keltner2_exit_weight': trial.suggest_int('keltner2_exit_weight', 1, 8),
            'donchian_exit_weight': trial.suggest_int('donchian_exit_weight', 1, 8),
            'chop_exit_weight': trial.suggest_int('chop_exit_weight', 1, 8),
            'adx_exit_weight': trial.suggest_int('adx_exit_weight', 1, 8),
            'kama_trend_exit_weight': trial.suggest_int('kama_trend_exit_weight', 1, 8),
            'roc_exit_weight': trial.suggest_int('roc_exit_weight', 1, 8),
            'squeeze_exit_weight': trial.suggest_int('squeeze_exit_weight', 1, 8),
            'squeeze2_exit_weight': trial.suggest_int('squeeze2_exit_weight', 1, 8),
            'roc_div_exit_weight': trial.suggest_int('roc_div_exit_weight', 1, 8),
            'bollinger_exit_weight': trial.suggest_int('bollinger_exit_weight', 1, 8),
            'rsi_exit_weight': trial.suggest_int('rsi_exit_weight', 1, 8),
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
        return {k: float(v) if k.endswith('threshold') else int(v) for k, v in params.items()} 