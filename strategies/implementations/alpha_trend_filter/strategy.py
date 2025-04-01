#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna
import logging

from ...base.strategy import BaseStrategy
from .signal_generator import AlphaTrendFilterSignalGenerator


logger = logging.getLogger(__name__)

class AlphaTrendFilterStrategy(BaseStrategy):
    """
    アルファトレンド+アルファフィルター戦略
    
    特徴:
    - 効率比（ER）に基づく動的パラメータ最適化
    - アルファトレンドによる高精度なトレンド検出
    - アルファフィルターによる市場状態フィルタリング
    
    エントリー条件:
    - ロング: アルファトレンドの上昇シグナル + アルファフィルターがトレンド相場
    - ショート: アルファトレンドの下降シグナル + アルファフィルターがトレンド相場
    
    エグジット条件:
    - ロング: アルファトレンドの下降シグナル
    - ショート: アルファトレンドの上昇シグナル
    """
    
    def __init__(
        self,
        er_period: int = 21,
        # アルファトレンド用パラメータ
        max_percentile_length: int = 55,
        min_percentile_length: int = 13,
        max_atr_period: int = 89,
        min_atr_period: int = 13,
        max_trend_multiplier: float = 3.0,
        min_trend_multiplier: float = 1.0,
        # アルファフィルター用パラメータ
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        max_adx_period: int = 21,
        min_adx_period: int = 5,
        filter_threshold: float = 0.5
    ):
        """
        初期化
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_percentile_length: パーセンタイル計算の最大期間（デフォルト: 55）
            min_percentile_length: パーセンタイル計算の最小期間（デフォルト: 13）
            max_atr_period: Alpha ATR期間の最大値（デフォルト: 89）
            min_atr_period: Alpha ATR期間の最小値（デフォルト: 13）
            max_trend_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_trend_multiplier: ATR乗数の最小値（デフォルト: 1.0）
            max_chop_period: アルファチョピネスの最大期間（デフォルト: 55）
            min_chop_period: アルファチョピネスの最小期間（デフォルト: 8）
            max_adx_period: アルファADXの最大期間（デフォルト: 21）
            min_adx_period: アルファADXの最小期間（デフォルト: 5）
            filter_threshold: アルファフィルターのしきい値（デフォルト: 0.5）
        """
        super().__init__("AlphaTrendFilter")
        
        # ロガーの設定
        self.logger = logger
        
        # パラメータの設定
        self._parameters = {
            'er_period': er_period,
            'max_percentile_length': max_percentile_length,
            'min_percentile_length': min_percentile_length,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_trend_multiplier': max_trend_multiplier,
            'min_trend_multiplier': min_trend_multiplier,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'filter_threshold': filter_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = AlphaTrendFilterSignalGenerator(
            er_period=er_period,
            max_percentile_length=max_percentile_length,
            min_percentile_length=min_percentile_length,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_trend_multiplier=max_trend_multiplier,
            min_trend_multiplier=min_trend_multiplier,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=max_adx_period,
            min_adx_period=min_adx_period,
            filter_threshold=filter_threshold
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル
        """
        try:
            return self.signal_generator.get_entry_signals(data)
        except Exception as e:
            self.logger.error(f"エントリーシグナル生成中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
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
        try:
            return self.signal_generator.get_exit_signals(data, position, index)
        except Exception as e:
            self.logger.error(f"エグジットシグナル生成中にエラー: {str(e)}")
            return False
    
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
            'er_period': trial.suggest_int('er_period', 5, 200),
            'max_percentile_length': trial.suggest_int('max_percentile_length', 21, 233),
            'min_percentile_length': trial.suggest_int('min_percentile_length', 3, 20),
            'max_atr_period': trial.suggest_int('max_atr_period', 21, 233),
            'min_atr_period': trial.suggest_int('min_atr_period', 5, 20),
            'max_trend_multiplier': trial.suggest_float('max_trend_multiplier', 1.0, 3.0, step=0.1),
            'min_trend_multiplier': trial.suggest_float('min_trend_multiplier', 0.1, 1.0, step=0.1),
            'max_chop_period': trial.suggest_int('max_chop_period', 56, 144),
            'min_chop_period': trial.suggest_int('min_chop_period', 13, 55),
            'max_adx_period': trial.suggest_int('max_adx_period', 5, 34),
            'min_adx_period': trial.suggest_int('min_adx_period', 2, 13),
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
            'er_period': int(params['er_period']),
            'max_percentile_length': int(params['max_percentile_length']),
            'min_percentile_length': int(params['min_percentile_length']),
            'max_atr_period': int(params['max_atr_period']),
            'min_atr_period': int(params['min_atr_period']),
            'max_trend_multiplier': float(params['max_trend_multiplier']),
            'min_trend_multiplier': float(params['min_trend_multiplier']),
            'max_chop_period': int(params['max_chop_period']),
            'min_chop_period': int(params['min_chop_period']),
            'max_adx_period': int(params['max_adx_period']),
            'min_adx_period': int(params['min_adx_period']),
            'filter_threshold': 0.5
        } 