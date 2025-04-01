#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import TrueMomentumSignalGenerator


class TrueMomentumStrategy(BaseStrategy):
    """
    トゥルーモメンタム+ガーディアンエンジェルフィルター戦略
    
    エントリー条件:
    - ロング: トゥルーモメンタムの買いシグナル + ガーディアンエンジェルがトレンド相場
    - ショート: トゥルーモメンタムの売りシグナル + ガーディアンエンジェルがトレンド相場
    
    エグジット条件:
    - ロング: トゥルーモメンタムの方向シグナルが売り
    - ショート: トゥルーモメンタムの方向シグナルが買い
    """
    
    def __init__(
        self,
        period: int = 11,
        max_std_mult: float = 2.0,
        min_std_mult: float = 1.0,
        max_kama_slow: int = 55,
        min_kama_slow: int = 30,
        max_kama_fast: int = 13,
        min_kama_fast: int = 2,
        max_atr_period: int = 120,
        min_atr_period: int = 13,
        max_atr_mult: float = 3.0,
        min_atr_mult: float = 1.0,
        max_momentum_period: int = 100,
        min_momentum_period: int = 20,
        momentum_threshold: float = 0.0,
        max_ga_period: int = 100,
        min_ga_period: int = 20,
        max_ga_threshold: float = 55,
        min_ga_threshold: float = 45
    ):
        """
        初期化
        
        Args:
            period: 計算期間（KAMA、ボリンジャーバンド、トレンドアルファ、ERに共通）
            max_std_mult: 標準偏差乗数の最大値
            min_std_mult: 標準偏差乗数の最小値
            max_kama_slow: KAMAの遅い移動平均の最大期間
            min_kama_slow: KAMAの遅い移動平均の最小期間
            max_kama_fast: KAMAの速い移動平均の最大期間
            min_kama_fast: KAMAの速い移動平均の最小期間
            max_atr_period: ATR期間の最大値
            min_atr_period: ATR期間の最小値
            max_atr_mult: ATR乗数の最大値
            min_atr_mult: ATR乗数の最小値
            max_momentum_period: モメンタム計算の最大期間
            min_momentum_period: モメンタム計算の最小期間
            momentum_threshold: モメンタムの閾値
            max_ga_period: ガーディアンエンジェルのチョピネス期間の最大値
            min_ga_period: ガーディアンエンジェルのチョピネス期間の最小値
            max_ga_threshold: ガーディアンエンジェルのしきい値の最大値
            min_ga_threshold: ガーディアンエンジェルのしきい値の最小値
        """
        super().__init__("TrueMomentum")
        
        # パラメータの設定
        self._parameters = {
            'period': period,
            'max_std_mult': max_std_mult,
            'min_std_mult': min_std_mult,
            'max_kama_slow': max_kama_slow,
            'min_kama_slow': min_kama_slow,
            'max_kama_fast': max_kama_fast,
            'min_kama_fast': min_kama_fast,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_atr_mult': max_atr_mult,
            'min_atr_mult': min_atr_mult,
            'max_momentum_period': max_momentum_period,
            'min_momentum_period': min_momentum_period,
            'momentum_threshold': momentum_threshold,
            'max_ga_period': max_ga_period,
            'min_ga_period': min_ga_period,
            'max_ga_threshold': max_ga_threshold,
            'min_ga_threshold': min_ga_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = TrueMomentumSignalGenerator(
            period=period,
            max_std_mult=max_std_mult,
            min_std_mult=min_std_mult,
            max_kama_slow=max_kama_slow,
            min_kama_slow=min_kama_slow,
            max_kama_fast=max_kama_fast,
            min_kama_fast=min_kama_fast,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_atr_mult=max_atr_mult,
            min_atr_mult=min_atr_mult,
            max_momentum_period=max_momentum_period,
            min_momentum_period=min_momentum_period,
            momentum_threshold=momentum_threshold,
            max_ga_period=max_ga_period,
            min_ga_period=min_ga_period,
            max_ga_threshold=max_ga_threshold,
            min_ga_threshold=min_ga_threshold
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
            'period': trial.suggest_int('period', 5, 100),
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
        # 全てのパラメータをそのまま使用
        return {
            'period': int(params['period']),
            'max_std_mult': 2.0,
            'min_std_mult': 1.0,
            'max_kama_slow': 55,
            'min_kama_slow': 30,
            'max_kama_fast': 13,
            'min_kama_fast': 2,
            'max_atr_period': 120,
            'min_atr_period': 13,
            'max_atr_mult': 3.0,
            'min_atr_mult': 1.0,
            'max_momentum_period': 100,
            'min_momentum_period': 20,
            'momentum_threshold': 0.0,
            'max_ga_period': 100,
            'min_ga_period': 20,
            'max_ga_threshold': 55,
            'min_ga_threshold': 45
        } 