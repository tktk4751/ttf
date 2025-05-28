#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZASimpleSignalGenerator


class ZASimpleStrategy(BaseStrategy):
    """
    ZAシンプルストラテジー
    
    特徴:
    - サイクル効率比（CER）に基づく動的パラメータ最適化
    - ZAdaptiveChannelによる高精度なエントリーポイント検出
    
    エントリー条件:
    - ロング: ZAdaptiveChannelの買いシグナル
    - ショート: ZAdaptiveChannelの売りシグナル
    
    エグジット条件:
    - ロング: ZAdaptiveChannelの売りシグナル
    - ショート: ZAdaptiveChannelの買いシグナル
    """
    
    def __init__(
        self,
        # 基本パラメータ
        band_lookback: int = 1,
        src_type: str = 'hlc3',
        # 動的乗数の範囲パラメータ
        max_max_multiplier: float = 9.0,    # 最大乗数の最大値
        min_max_multiplier: float = 2.0,    # 最大乗数の最小値
        max_min_multiplier: float = 4.0,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        
        # CERパラメータ
        detector_type: str = 'phac_e',     # CER用ドミナントサイクル検出器タイプ
        cycle_part: float = 0.5,           # CER用サイクル部分
        lp_period: int = 5,               # CER用ローパスフィルター期間
        hp_period: int = 100,              # CER用ハイパスフィルター期間
        max_cycle: int = 120,              # CER用最大サイクル期間
        min_cycle: int = 5,               # CER用最小サイクル期間
        max_output: int = 89,             # CER用最大出力値
        min_output: int = 5,              # CER用最小出力値
        use_kalman_filter: bool = False,   # CER用カルマンフィルター使用有無
        
        # ZAdaptiveMA用パラメータ
        fast_period: int = 2,             # 速い移動平均の期間（固定値）
        slow_period: int = 144            # 遅い移動平均の期間（固定値）
    ):
        """
        初期化
        
        Args:
            band_lookback: 過去バンド参照期間（デフォルト: 1）
            src_type: 価格ソースタイプ（デフォルト: 'hlc3'）
            
            # 動的乗数の範囲パラメータ
            max_max_multiplier: 最大乗数の最大値（デフォルト: 8.0）
            min_max_multiplier: 最大乗数の最小値（デフォルト: 3.0）
            max_min_multiplier: 最小乗数の最大値（デフォルト: 1.5）
            min_min_multiplier: 最小乗数の最小値（デフォルト: 0.5）
            
            # CERパラメータ
            detector_type: CER用ドミナントサイクル検出器タイプ（デフォルト: 'phac_e'）
            cycle_part: CER用サイクル部分（デフォルト: 0.5）
            lp_period: CER用ローパスフィルター期間（デフォルト: 5）
            hp_period: CER用ハイパスフィルター期間（デフォルト: 55）
            max_cycle: CER用最大サイクル期間（デフォルト: 55）
            min_cycle: CER用最小サイクル期間（デフォルト: 5）
            max_output: CER用最大出力値（デフォルト: 34）
            min_output: CER用最小出力値（デフォルト: 5）
            use_kalman_filter: CER用カルマンフィルター使用有無（デフォルト: False）
            
            # ZAdaptiveMA用パラメータ
            fast_period: 速い移動平均の期間（デフォルト: 2）
            slow_period: 遅い移動平均の期間（デフォルト: 30）
        """
        super().__init__("ZASimple")
        
        # パラメータの設定
        self._parameters = {
            # 基本パラメータ
            'band_lookback': band_lookback,
            'src_type': src_type,
            
            # 動的乗数の範囲パラメータ
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            
            # CERパラメータ
            'detector_type': detector_type,
            'cycle_part': cycle_part,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            'use_kalman_filter': use_kalman_filter,
            
            # ZAdaptiveMA用パラメータ
            'fast_period': fast_period,
            'slow_period': slow_period
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZASimpleSignalGenerator(
            # 基本パラメータ
            band_lookback=band_lookback,
            src_type=src_type,
            
            # 動的乗数の範囲パラメータ
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            
            # CERパラメータ
            detector_type=detector_type,
            cycle_part=cycle_part,
            lp_period=lp_period,
            hp_period=hp_period,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            use_kalman_filter=use_kalman_filter,
            
            # ZAdaptiveMA用パラメータ
            fast_period=fast_period,
            slow_period=slow_period
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
            # 基本パラメータ
            'band_lookback': trial.suggest_int('band_lookback', 1, 5),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # 動的乗数の範囲パラメータ
            'max_max_multiplier': trial.suggest_float('max_max_multiplier', 5.0, 10.0, step=0.5),
            'min_max_multiplier': trial.suggest_float('min_max_multiplier', 2.0, 5.0, step=0.5),
            'max_min_multiplier': trial.suggest_float('max_min_multiplier', 1.0, 2.0, step=0.1),
            'min_min_multiplier': trial.suggest_float('min_min_multiplier', 0.0, 1.0, step=0.1),
            
            # CERパラメータ
            'detector_type': trial.suggest_categorical('detector_type', 
                                                      ['dudi', 'hody', 'phac', 'dudi_e', 'hody_e', 'phac_e']),
            'cycle_part': trial.suggest_float('cycle_part', 0.2, 0.9, step=0.1),
            'lp_period': trial.suggest_int('lp_period', 3, 21),
            'hp_period': trial.suggest_int('hp_period', 34, 233),
            'max_cycle': trial.suggest_int('max_cycle', 34, 144),
            'min_cycle': trial.suggest_int('min_cycle', 3, 13),
            'max_output': trial.suggest_int('max_output', 21, 144),
            'min_output': trial.suggest_int('min_output', 3, 13),
            
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
            # 基本パラメータ
            'band_lookback': int(params['band_lookback']),
            'src_type': params['src_type'],
            
            # 動的乗数の範囲パラメータ
            'max_max_multiplier': float(params['max_max_multiplier']),
            'min_max_multiplier': float(params['min_max_multiplier']),
            'max_min_multiplier': float(params['max_min_multiplier']),
            'min_min_multiplier': float(params['min_min_multiplier']),
            
            # CERパラメータ
            'detector_type': params['detector_type'],
            'cycle_part': float(params['cycle_part']),
            'lp_period': int(params['lp_period']),
            'hp_period': int(params['hp_period']),
            'max_cycle': int(params['max_cycle']),
            'min_cycle': int(params['min_cycle']),
            'max_output': int(params['max_output']),
            'min_output': int(params['min_output']),
            'use_kalman_filter': False,
            
            # ZAdaptiveMA用パラメータ
            'fast_period': 2,
            'slow_period': 30
        }
        return strategy_params 