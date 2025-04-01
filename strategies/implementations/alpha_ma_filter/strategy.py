#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna
import logging

from ...base.strategy import BaseStrategy
from .signal_generator import AlphaMAFilterSignalGenerator


logger = logging.getLogger(__name__)

class AlphaMAFilterStrategy(BaseStrategy):
    """
    AlphaMA+アルファフィルター戦略
    
    特徴:
    - 効率比（ER）に基づく動的パラメータ最適化
    - AlphaMAによる高精度なトレンド検出
    - アルファフィルターによる市場状態フィルタリング
    
    エントリー条件:
    - ロング: AlphaMATrendFollowingSignalが上昇トレンド(1) + アルファフィルターがトレンド相場(1)
    - ショート: AlphaMATrendFollowingSignalが下降トレンド(-1) + アルファフィルターがトレンド相場(1)
    
    エグジット条件:
    - ロング: AlphaMATrendFollowingSignalが下降トレンド(-1)
    - ショート: AlphaMATrendFollowingSignalが上昇トレンド(1)
    """
    
    def __init__(
        self,
        # 共通パラメータ
        er_period: int = 21,
        # AlphaMA用パラメータ

        short_max_kama_period: int = 89,
        short_min_kama_period: int = 8,
        long_max_kama_period: int = 144,
        long_min_kama_period: int = 21,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2,
        # アルファフィルター用パラメータ
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        max_adx_period: int = 21,
        min_adx_period: int = 5,
        alma_offset: float = 0.85,
        alma_sigma: float = 6,
        filter_threshold: float = 0.5
    ):
        """
        初期化
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）

            short_max_kama_period: 短期AlphaMAのKAMAピリオドの最大値（デフォルト: 89）
            short_min_kama_period: 短期AlphaMAのKAMAピリオドの最小値（デフォルト: 8）
            long_max_kama_period: 長期AlphaMAのKAMAピリオドの最大値（デフォルト: 144）
            long_min_kama_period: 長期AlphaMAのKAMAピリオドの最小値（デフォルト: 21）
            max_slow_period: 遅い移動平均の最大期間（デフォルト: 89）
            min_slow_period: 遅い移動平均の最小期間（デフォルト: 30）
            max_fast_period: 速い移動平均の最大期間（デフォルト: 13）
            min_fast_period: 速い移動平均の最小期間（デフォルト: 2）
            max_chop_period: アルファチョピネスの最大期間（デフォルト: 55）
            min_chop_period: アルファチョピネスの最小期間（デフォルト: 8）
            max_adx_period: アルファADXの最大期間（デフォルト: 21）
            min_adx_period: アルファADXの最小期間（デフォルト: 5）
            alma_offset: ALMAのオフセット（デフォルト: 0.85）
            alma_sigma: ALMAのシグマ（デフォルト: 6）
            filter_threshold: アルファフィルターのしきい値（デフォルト: 0.5）
        """
        super().__init__("AlphaMAFilter")
        
        # ロガーの設定
        self.logger = logger
        
        # パラメータの設定
        self._parameters = {
            'er_period': er_period,
            'short_er_period': er_period,
            'long_er_period': er_period,
            'short_max_kama_period': short_max_kama_period,
            'short_min_kama_period': short_min_kama_period,
            'long_max_kama_period': long_max_kama_period,
            'long_min_kama_period': long_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'filter_threshold': filter_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = AlphaMAFilterSignalGenerator(
            er_period=er_period,
            short_max_kama_period=short_max_kama_period,
            short_min_kama_period=short_min_kama_period,
            long_max_kama_period=long_max_kama_period,
            long_min_kama_period=long_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=max_adx_period,
            min_adx_period=min_adx_period,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma,
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
            'short_max_kama_period': trial.suggest_int('short_max_kama_period', 34, 89),
            'short_min_kama_period': trial.suggest_int('short_min_kama_period', 3, 21),\

                
            'long_max_kama_period': trial.suggest_int('long_max_kama_period', 89, 377),
            'long_min_kama_period': trial.suggest_int('long_min_kama_period', 21, 144),
            'max_slow_period': trial.suggest_int('max_slow_period', 30, 89),
            'min_slow_period': trial.suggest_int('min_slow_period', 2, 30),
            'max_fast_period': trial.suggest_int('max_fast_period', 30, 89),
            'min_fast_period': trial.suggest_int('min_fast_period', 2, 30),
            'max_chop_period': trial.suggest_int('max_chop_period', 55, 144),
            'min_chop_period': trial.suggest_int('min_chop_period', 13, 55),
            'max_adx_period': trial.suggest_int('max_adx_period', 13, 34),
            'min_adx_period': trial.suggest_int('min_adx_period', 2, 13),
            'alma_offset': trial.suggest_float('alma_offset', 0.3, 0.9, step=0.1),
            'alma_sigma': trial.suggest_int('alma_sigma', 2, 9),
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
            'short_max_kama_period': int(params['short_max_kama_period']),
            'short_min_kama_period': int(params['short_min_kama_period']),
            'long_max_kama_period': int(params['long_max_kama_period']),
            'long_min_kama_period': int(params['long_min_kama_period']),
            'max_slow_period': int(params['max_slow_period']),
            'min_slow_period': int(params['min_slow_period']),
            'max_fast_period': int(params['max_fast_period']),
            'min_fast_period': int(params['min_fast_period']),
            'max_chop_period': int(params['max_chop_period']),
            'min_chop_period': int(params['min_chop_period']),
            'max_adx_period': int(params['max_adx_period']),
            'min_adx_period': int(params['min_adx_period']),
            'alma_offset': float(params['alma_offset']),
            'alma_sigma': int(params['alma_sigma']),
        } 