#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna
import logging

from ...base.strategy import BaseStrategy
from strategies.implementations.alpha_donchian_filter.signal_generator import AlphaDonchianFilterSignalGenerator


logger = logging.getLogger(__name__)

class AlphaDonchianFilterStrategy(BaseStrategy):
    """
    アルファドンチャン+アルファフィルター戦略
    
    特徴:
    - 効率比（ER）に基づく動的パラメータ最適化
    - アルファドンチャンによる高精度な市場構造検出
    - アルファフィルターによる市場状態フィルタリング
    
    エントリー条件:
    - ロング: アルファドンチャンの上昇ブレイクアウト + アルファフィルターがトレンド相場
    - ショート: アルファドンチャンの下降ブレイクアウト + アルファフィルターがトレンド相場
    
    エグジット条件:
    - ロング: アルファドンチャンの下降ブレイクアウト
    - ショート: アルファドンチャンの上昇ブレイクアウト
    """
    
    def __init__(
        self,
        # 共通パラメータ
        er_period: int = 27,
        # アルファドンチャン用パラメータ
        max_donchian_period: int = 65,
        min_donchian_period: int = 12,
        # アルファフィルター用パラメータ
        max_chop_period: int = 67,
        min_chop_period: int = 20,
        max_adx_period: int = 29,
        min_adx_period: int = 9,
        filter_threshold: float = 0.5
    ):
        """
        初期化
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_donchian_period: ドンチャン期間の最大値（デフォルト: 55）
            min_donchian_period: ドンチャン期間の最小値（デフォルト: 13）
            max_chop_period: アルファチョピネスの最大期間（デフォルト: 55）
            min_chop_period: アルファチョピネスの最小期間（デフォルト: 8）
            max_adx_period: アルファADXの最大期間（デフォルト: 21）
            min_adx_period: アルファADXの最小期間（デフォルト: 5）
            filter_threshold: アルファフィルターのしきい値（デフォルト: 0.5）
        """
        super().__init__("AlphaDonchianFilterStrategy")
        
        # 戦略パラメータの保存
        self.params = {
            "er_period": er_period,
            "max_donchian_period": max_donchian_period,
            "min_donchian_period": min_donchian_period,
            "max_chop_period": max_chop_period,
            "min_chop_period": min_chop_period,
            "max_adx_period": max_adx_period,
            "min_adx_period": min_adx_period,
            "filter_threshold": filter_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = AlphaDonchianFilterSignalGenerator(
            er_period=er_period,
            max_donchian_period=max_donchian_period,
            min_donchian_period=min_donchian_period,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=max_adx_period,
            min_adx_period=min_adx_period,
            filter_threshold=filter_threshold
        )
        
        self.name = f"AlphaDonchianFilterStrategy({er_period}, {max_donchian_period}, {min_donchian_period}, {filter_threshold})"
    
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
        最適化パラメータを生成する
        
        Args:
            trial: Optunaのトライアルオブジェクト
            
        Returns:
            最適化されたパラメータ辞書
        """
        # 共通パラメータ
        er_period = trial.suggest_int("er_period", 10, 34, step=1)
        
        # アルファドンチャン用パラメータ
        max_donchian_period = trial.suggest_int("max_donchian_period", 34, 89, step=1)
        min_donchian_period = trial.suggest_int("min_donchian_period", 5, 21, step=1)
        
        # アルファフィルター用パラメータ
        max_chop_period = trial.suggest_int("max_chop_period", 34, 89, step=1)
        min_chop_period = trial.suggest_int("min_chop_period", 5, 21, step=1)
        max_adx_period = trial.suggest_int("max_adx_period", 13, 34, step=1)
        min_adx_period = trial.suggest_int("min_adx_period", 3, 13, step=1)
        filter_threshold = trial.suggest_float("filter_threshold", 0.3, 0.7, step=0.05)
        
        return {
            "er_period": er_period,
            "max_donchian_period": max_donchian_period,
            "min_donchian_period": min_donchian_period,
            "max_chop_period": max_chop_period,
            "min_chop_period": min_chop_period,
            "max_adx_period": max_adx_period,
            "min_adx_period": min_adx_period,
            "filter_threshold": filter_threshold
        }
    
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
            'max_donchian_period': int(params['max_donchian_period']),
            'min_donchian_period': int(params['min_donchian_period']),
            'max_chop_period': int(params['max_chop_period']),
            'min_chop_period': int(params['min_chop_period']),
            'max_adx_period': int(params['max_adx_period']),
            'min_adx_period': int(params['min_adx_period']),
            'filter_threshold': float(params['filter_threshold'])
        } 