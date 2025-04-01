#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZDonchianTrendSignalGenerator


class ZDonchianTrendStrategy(BaseStrategy):
    """
    Zドンチャン+Zトレンドフィルター戦略
    
    特徴:
    - サイクル効率比（CER）に基づく動的パラメータ最適化
    - Zドンチャンチャネルによる高精度なエントリーポイント検出
    - Zトレンドフィルターによる市場状態フィルタリング
    
    エントリー条件:
    - ロング: Zドンチャンチャネルの買いシグナル + Zトレンドフィルターがトレンド相場
    - ショート: Zドンチャンチャネルの売りシグナル + Zトレンドフィルターがトレンド相場
    
    エグジット条件:
    - ロング: Zドンチャンチャネルの売りシグナル
    - ショート: Zドンチャンチャネルの買いシグナル
    """
    
    def __init__(
        self,
        # 共通パラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 10,
        hp_period: int = 233,
        cycle_part: float = 0.8,
        src_type: str = 'hlc3',
        
        # Zドンチャンチャネル用パラメータ
        # 最大期間用パラメータ
        max_dc_cycle_part: float = 0.8,
        max_dc_max_cycle: int = 144,
        max_dc_min_cycle: int = 5,
        max_dc_max_output: int = 89,
        max_dc_min_output: int = 34,
        
        # 最小期間用パラメータ
        min_dc_cycle_part: float = 0.4,
        min_dc_max_cycle: int = 89,
        min_dc_min_cycle: int = 5,
        min_dc_max_output: int = 55,
        min_dc_min_output: int = 21,
        
        # ブレイクアウトパラメータ
        lookback: int = 1,
        
        # Zトレンドフィルター用パラメータ
        max_stddev_period: int = 13,
        min_stddev_period: int = 5,
        max_lookback_period: int = 13,
        min_lookback_period: int = 5,
        max_rms_window: int = 13,
        min_rms_window: int = 5,
        max_threshold: float = 0.75,
        min_threshold: float = 0.55,
        combination_weight: float = 0.6,
        zadx_weight: float = 0.4,
        combination_method: str = "sigmoid",
        
        # Zトレンドインデックスの追加パラメータ
        max_chop_dc_cycle_part: float = 0.8,
        max_chop_dc_max_cycle: int = 144,
        max_chop_dc_min_cycle: int = 10,
        max_chop_dc_max_output: int = 89,
        max_chop_dc_min_output: int = 34,
        min_chop_dc_cycle_part: float = 0.4,
        min_chop_dc_max_cycle: int = 55,
        min_chop_dc_min_cycle: int = 5,
        min_chop_dc_max_output: int = 34,
        min_chop_dc_min_output: int = 13
    ):
        """初期化"""
        super().__init__("ZDonchianTrend")
        
        # パラメータの設定
        self._parameters = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'src_type': src_type,
            'max_dc_cycle_part': max_dc_cycle_part,
            'max_dc_max_cycle': max_dc_max_cycle,
            'max_dc_min_cycle': max_dc_min_cycle,
            'max_dc_max_output': max_dc_max_output,
            'max_dc_min_output': max_dc_min_output,
            'min_dc_cycle_part': min_dc_cycle_part,
            'min_dc_max_cycle': min_dc_max_cycle,
            'min_dc_min_cycle': min_dc_min_cycle,
            'min_dc_max_output': min_dc_max_output,
            'min_dc_min_output': min_dc_min_output,
            'lookback': lookback,
            'max_stddev_period': max_stddev_period,
            'min_stddev_period': min_stddev_period,
            'max_lookback_period': max_lookback_period,
            'min_lookback_period': min_lookback_period,
            'max_rms_window': max_rms_window,
            'min_rms_window': min_rms_window,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold,
            'combination_weight': combination_weight,
            'zadx_weight': zadx_weight,
            'combination_method': combination_method,
            'max_chop_dc_cycle_part': max_chop_dc_cycle_part,
            'max_chop_dc_max_cycle': max_chop_dc_max_cycle,
            'max_chop_dc_min_cycle': max_chop_dc_min_cycle,
            'max_chop_dc_max_output': max_chop_dc_max_output,
            'max_chop_dc_min_output': max_chop_dc_min_output,
            'min_chop_dc_cycle_part': min_chop_dc_cycle_part,
            'min_chop_dc_max_cycle': min_chop_dc_max_cycle,
            'min_chop_dc_min_cycle': min_chop_dc_min_cycle,
            'min_chop_dc_max_output': min_chop_dc_max_output,
            'min_chop_dc_min_output': min_chop_dc_min_output
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZDonchianTrendSignalGenerator(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            src_type=src_type,
            max_dc_cycle_part=max_dc_cycle_part,
            max_dc_max_cycle=max_dc_max_cycle,
            max_dc_min_cycle=max_dc_min_cycle,
            max_dc_max_output=max_dc_max_output,
            max_dc_min_output=max_dc_min_output,
            min_dc_cycle_part=min_dc_cycle_part,
            min_dc_max_cycle=min_dc_max_cycle,
            min_dc_min_cycle=min_dc_min_cycle,
            min_dc_max_output=min_dc_max_output,
            min_dc_min_output=min_dc_min_output,
            lookback=lookback,
            max_stddev_period=max_stddev_period,
            min_stddev_period=min_stddev_period,
            max_lookback_period=max_lookback_period,
            min_lookback_period=min_lookback_period,
            max_rms_window=max_rms_window,
            min_rms_window=min_rms_window,
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            combination_weight=combination_weight,
            zadx_weight=zadx_weight,
            combination_method=combination_method,
            max_chop_dc_cycle_part=max_chop_dc_cycle_part,
            max_chop_dc_max_cycle=max_chop_dc_max_cycle,
            max_chop_dc_min_cycle=max_chop_dc_min_cycle,
            max_chop_dc_max_output=max_chop_dc_max_output,
            max_chop_dc_min_output=max_chop_dc_min_output,
            min_chop_dc_cycle_part=min_chop_dc_cycle_part,
            min_chop_dc_max_cycle=min_chop_dc_max_cycle,
            min_chop_dc_min_cycle=min_chop_dc_min_cycle,
            min_chop_dc_max_output=min_chop_dc_max_output,
            min_chop_dc_min_output=min_chop_dc_min_output
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
            # 共通パラメータ
            'cycle_detector_type': trial.suggest_categorical('cycle_detector_type', ['hody_dc', 'dudi_dc', 'phac_dc']),
            'lp_period': trial.suggest_int('lp_period', 3, 21),
            'hp_period': trial.suggest_int('hp_period', 62, 233),
            'cycle_part': trial.suggest_float('cycle_part', 0.3, 0.8, step=0.1),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # Zドンチャンチャネル用パラメータ
            # 最大期間用パラメータ
            'max_dc_cycle_part': trial.suggest_float('max_dc_cycle_part', 0.3, 0.7, step=0.1),
            'max_dc_max_cycle': trial.suggest_int('max_dc_max_cycle', 55, 233),
            'max_dc_min_cycle': trial.suggest_int('max_dc_min_cycle', 5, 15),
            'max_dc_max_output': trial.suggest_int('max_dc_max_output', 55, 144),
            'max_dc_min_output': trial.suggest_int('max_dc_min_output', 13, 34),
            
            # 最小期間用パラメータ
            'min_dc_cycle_part': trial.suggest_float('min_dc_cycle_part', 0.15, 0.4, step=0.05),
            'min_dc_max_cycle': trial.suggest_int('min_dc_max_cycle', 35, 75),
            'min_dc_min_cycle': trial.suggest_int('min_dc_min_cycle', 3, 10),
            'min_dc_max_output': trial.suggest_int('min_dc_max_output', 13, 34),
            'min_dc_min_output': trial.suggest_int('min_dc_min_output', 3, 13),
            
            # ブレイクアウトパラメータ
            'lookback': trial.suggest_int('lookback', 1, 5),
            
            # Zトレンドフィルター用パラメータ
            'max_stddev_period': trial.suggest_int('max_stddev_period', 8, 21),
            'min_stddev_period': trial.suggest_int('min_stddev_period', 3, 8),
            'max_lookback_period': trial.suggest_int('max_lookback_period', 8, 21),
            'min_lookback_period': trial.suggest_int('min_lookback_period', 3, 8),
            'max_rms_window': trial.suggest_int('max_rms_window', 8, 21),
            'min_rms_window': trial.suggest_int('min_rms_window', 3, 8),
            'max_threshold': trial.suggest_float('max_threshold', 0.6, 0.85, step=0.05),
            'min_threshold': trial.suggest_float('min_threshold', 0.45, 0.6, step=0.05),
            'combination_weight': trial.suggest_float('combination_weight', 0.4, 0.8, step=0.1),
            'zadx_weight': trial.suggest_float('zadx_weight', 0.3, 0.7, step=0.1),
            'combination_method': trial.suggest_categorical('combination_method', ['sigmoid', 'rms', 'simple']),
            
            # Zトレンドインデックスの追加パラメータ
            'max_chop_dc_cycle_part': trial.suggest_float('max_chop_dc_cycle_part', 0.3, 0.7, step=0.1),
            'max_chop_dc_max_cycle': trial.suggest_int('max_chop_dc_max_cycle', 55, 233),
            'max_chop_dc_min_cycle': trial.suggest_int('max_chop_dc_min_cycle', 5, 15),
            'max_chop_dc_max_output': trial.suggest_int('max_chop_dc_max_output', 20, 50),
            'max_chop_dc_min_output': trial.suggest_int('max_chop_dc_min_output', 8, 20),
            'min_chop_dc_cycle_part': trial.suggest_float('min_chop_dc_cycle_part', 0.15, 0.4, step=0.05),
            'min_chop_dc_max_cycle': trial.suggest_int('min_chop_dc_max_cycle', 35, 75),
            'min_chop_dc_min_cycle': trial.suggest_int('min_chop_dc_min_cycle', 3, 10),
            'min_chop_dc_max_output': trial.suggest_int('min_chop_dc_max_output', 8, 20),
            'min_chop_dc_min_output': trial.suggest_int('min_chop_dc_min_output', 3, 10),
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
        strategy_params = {
            'cycle_detector_type': params['cycle_detector_type'],
            'lp_period': int(params['lp_period']),
            'hp_period': int(params['hp_period']),
            'cycle_part': float(params['cycle_part']),
            'src_type': params['src_type'],
            'max_dc_cycle_part': float(params['max_dc_cycle_part']),
            'max_dc_max_cycle': int(params['max_dc_max_cycle']),
            'max_dc_min_cycle': int(params['max_dc_min_cycle']),
            'max_dc_max_output': int(params['max_dc_max_output']),
            'max_dc_min_output': int(params['max_dc_min_output']),
            'min_dc_cycle_part': float(params['min_dc_cycle_part']),
            'min_dc_max_cycle': int(params['min_dc_max_cycle']),
            'min_dc_min_cycle': int(params['min_dc_min_cycle']),
            'min_dc_max_output': int(params['min_dc_max_output']),
            'min_dc_min_output': int(params['min_dc_min_output']),
            'lookback': int(params['lookback']),
            'max_stddev_period': int(params['max_stddev_period']),
            'min_stddev_period': int(params['min_stddev_period']),
            'max_lookback_period': int(params['max_lookback_period']),
            'min_lookback_period': int(params['min_lookback_period']),
            'max_rms_window': int(params['max_rms_window']),
            'min_rms_window': int(params['min_rms_window']),
            'max_threshold': float(params['max_threshold']),
            'min_threshold': float(params['min_threshold']),
            'combination_weight': float(params['combination_weight']),
            'zadx_weight': float(params['zadx_weight']),
            'combination_method': params['combination_method'],
            'max_chop_dc_cycle_part': float(params['max_chop_dc_cycle_part']),
            'max_chop_dc_max_cycle': int(params['max_chop_dc_max_cycle']),
            'max_chop_dc_min_cycle': int(params['max_chop_dc_min_cycle']),
            'max_chop_dc_max_output': int(params['max_chop_dc_max_output']),
            'max_chop_dc_min_output': int(params['max_chop_dc_min_output']),
            'min_chop_dc_cycle_part': float(params['min_chop_dc_cycle_part']),
            'min_chop_dc_max_cycle': int(params['min_chop_dc_max_cycle']),
            'min_chop_dc_min_cycle': int(params['min_chop_dc_min_cycle']),
            'min_chop_dc_max_output': int(params['min_chop_dc_max_output']),
            'min_chop_dc_min_output': int(params['min_chop_dc_min_output']),
        }
        return strategy_params 