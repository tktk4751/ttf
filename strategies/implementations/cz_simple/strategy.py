#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd
import optuna

from ...base import BaseStrategy
from signals.implementations.cz_channel.signal_generator import CZSimpleSignalGenerator


class CZSimpleStrategy(BaseStrategy):
    """
    CZチャネルのシンプルストラテジー
    
    特徴:
    - CZチャネルのブレイクアウトでエントリー
    - 反対方向のブレイクアウトでエグジット
    - トレンドフィルターなし
    - サイクル効率比（CER）に基づく動的な適応性
    - CATRによる安定したボラティリティ測定
    """
    
    def __init__(
        self,
        # CZチャネルのパラメータ
        detector_type: str = 'phac_e',
        cer_detector_type: str = 'phac_e',  # CER用の検出器タイプ（デフォルトではdetector_typeと同じ）
        lp_period: int = 5,
        hp_period: int = 55,
        cycle_part: float = 0.7,
        smoother_type: str = 'alma',
        src_type: str = 'hlc3',
        band_lookback: int = 1,
        # 動的乗数の範囲パラメータ
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 3.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        
        # ZMA用パラメータ
        zma_max_dc_cycle_part: float = 0.618,     # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_cycle: int = 144,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_cycle: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_output: int = 89,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_output: int = 34,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_lp_period: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用LPピリオド
        zma_max_dc_hp_period: int = 55,         # ZMA: 最大期間用ドミナントサイクル計算用HPピリオド
        
        zma_min_dc_cycle_part: float = 0.382,    # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_cycle: int = 55,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_cycle: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_output: int = 21,        # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_output: int = 8,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_lp_period: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用LPピリオド
        zma_min_dc_hp_period: int = 34,         # ZMA: 最小期間用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Slow最大用パラメータ
        zma_slow_max_dc_cycle_part: float = 0.618,
        zma_slow_max_dc_max_cycle: int = 144,
        zma_slow_max_dc_min_cycle: int = 5,
        zma_slow_max_dc_max_output: int = 34,
        zma_slow_max_dc_min_output: int = 21,
        zma_slow_max_dc_lp_period: int = 5,      # ZMA: Slow最大用ドミナントサイクル計算用LPピリオド
        zma_slow_max_dc_hp_period: int = 55,     # ZMA: Slow最大用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Slow最小用パラメータ
        zma_slow_min_dc_cycle_part: float = 0.382,
        zma_slow_min_dc_max_cycle: int = 89,
        zma_slow_min_dc_min_cycle: int = 3,
        zma_slow_min_dc_max_output: int = 13,
        zma_slow_min_dc_min_output: int = 5,
        zma_slow_min_dc_lp_period: int = 5,      # ZMA: Slow最小用ドミナントサイクル計算用LPピリオド
        zma_slow_min_dc_hp_period: int = 34,     # ZMA: Slow最小用ドミナントサイクル計算用HPピリオド
        
        # ZMA動的Fast最大用パラメータ
        zma_fast_max_dc_cycle_part: float = 0.25,
        zma_fast_max_dc_max_cycle: int = 75,
        zma_fast_max_dc_min_cycle: int = 5,
        zma_fast_max_dc_max_output: int = 15,
        zma_fast_max_dc_min_output: int = 8,
        zma_fast_max_dc_lp_period: int = 5,      # ZMA: Fast最大用ドミナントサイクル計算用LPピリオド
        zma_fast_max_dc_hp_period: int = 21,     # ZMA: Fast最大用ドミナントサイクル計算用HPピリオド
        
        zma_min_fast_period: int = 2,           # ZMA: 速い移動平均の最小期間（常に2で固定）
        zma_hyper_smooth_period: int = 0,       # ZMA: ハイパースムーサーの平滑化期間
        
        # CATR用パラメータ
        catr_detector_type: str = 'hody',
        catr_cycle_part: float = 0.5,
        catr_lp_period: int = 5,
        catr_hp_period: int = 55,
        catr_max_cycle: int = 55,
        catr_min_cycle: int = 5,
        catr_max_output: int = 34,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'alma'
    ):
        """初期化"""
        super().__init__("CZSimpleStrategy")
        
        # パラメータの保存
        self._params = {
            'detector_type': detector_type,
            'cer_detector_type': cer_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            
            # ZMA用パラメータ
            'zma_max_dc_cycle_part': zma_max_dc_cycle_part,
            'zma_max_dc_max_cycle': zma_max_dc_max_cycle,
            'zma_max_dc_min_cycle': zma_max_dc_min_cycle,
            'zma_max_dc_max_output': zma_max_dc_max_output,
            'zma_max_dc_min_output': zma_max_dc_min_output,
            'zma_max_dc_lp_period': zma_max_dc_lp_period,
            'zma_max_dc_hp_period': zma_max_dc_hp_period,
            'zma_min_dc_cycle_part': zma_min_dc_cycle_part,
            'zma_min_dc_max_cycle': zma_min_dc_max_cycle,
            'zma_min_dc_min_cycle': zma_min_dc_min_cycle,
            'zma_min_dc_max_output': zma_min_dc_max_output,
            'zma_min_dc_min_output': zma_min_dc_min_output,
            'zma_min_dc_lp_period': zma_min_dc_lp_period,
            'zma_min_dc_hp_period': zma_min_dc_hp_period,
            'zma_slow_max_dc_cycle_part': zma_slow_max_dc_cycle_part,
            'zma_slow_max_dc_max_cycle': zma_slow_max_dc_max_cycle,
            'zma_slow_max_dc_min_cycle': zma_slow_max_dc_min_cycle,
            'zma_slow_max_dc_max_output': zma_slow_max_dc_max_output,
            'zma_slow_max_dc_min_output': zma_slow_max_dc_min_output,
            'zma_slow_max_dc_lp_period': zma_slow_max_dc_lp_period,
            'zma_slow_max_dc_hp_period': zma_slow_max_dc_hp_period,
            'zma_slow_min_dc_cycle_part': zma_slow_min_dc_cycle_part,
            'zma_slow_min_dc_max_cycle': zma_slow_min_dc_max_cycle,
            'zma_slow_min_dc_min_cycle': zma_slow_min_dc_min_cycle,
            'zma_slow_min_dc_max_output': zma_slow_min_dc_max_output,
            'zma_slow_min_dc_min_output': zma_slow_min_dc_min_output,
            'zma_slow_min_dc_lp_period': zma_slow_min_dc_lp_period,
            'zma_slow_min_dc_hp_period': zma_slow_min_dc_hp_period,
            'zma_fast_max_dc_cycle_part': zma_fast_max_dc_cycle_part,
            'zma_fast_max_dc_max_cycle': zma_fast_max_dc_max_cycle,
            'zma_fast_max_dc_min_cycle': zma_fast_max_dc_min_cycle,
            'zma_fast_max_dc_max_output': zma_fast_max_dc_max_output,
            'zma_fast_max_dc_min_output': zma_fast_max_dc_min_output,
            'zma_fast_max_dc_lp_period': zma_fast_max_dc_lp_period,
            'zma_fast_max_dc_hp_period': zma_fast_max_dc_hp_period,
            'zma_min_fast_period': zma_min_fast_period,
            'zma_hyper_smooth_period': zma_hyper_smooth_period,
            
            # CATR用パラメータ
            'catr_detector_type': catr_detector_type,
            'catr_cycle_part': catr_cycle_part,
            'catr_lp_period': catr_lp_period,
            'catr_hp_period': catr_hp_period,
            'catr_max_cycle': catr_max_cycle,
            'catr_min_cycle': catr_min_cycle,
            'catr_max_output': catr_max_output,
            'catr_min_output': catr_min_output,
            'catr_smoother_type': catr_smoother_type
        }
        
        # シグナルジェネレーターの初期化
        self._init_signal()
    
    def _init_signal(self) -> None:
        """シグナルジェネレーターの初期化"""
        self.signal_generator = CZSimpleSignalGenerator(
            # 基本パラメータ
            detector_type=self._params['detector_type'],
            cer_detector_type=self._params['cer_detector_type'],
            lp_period=self._params['lp_period'],
            hp_period=self._params['hp_period'],
            cycle_part=self._params['cycle_part'],
            smoother_type=self._params['smoother_type'],
            src_type=self._params['src_type'],
            band_lookback=self._params['band_lookback'],
            
            # 動的乗数の範囲パラメータ
            max_max_multiplier=self._params['max_max_multiplier'],
            min_max_multiplier=self._params['min_max_multiplier'],
            max_min_multiplier=self._params['max_min_multiplier'],
            min_min_multiplier=self._params['min_min_multiplier'],
            
            # ZMA基本パラメータ
            zma_max_dc_cycle_part=self._params['zma_max_dc_cycle_part'],
            zma_max_dc_max_cycle=self._params['zma_max_dc_max_cycle'],
            zma_max_dc_min_cycle=self._params['zma_max_dc_min_cycle'],
            zma_max_dc_max_output=self._params['zma_max_dc_max_output'],
            zma_max_dc_min_output=self._params['zma_max_dc_min_output'],
            zma_max_dc_lp_period=self._params['zma_max_dc_lp_period'],
            zma_max_dc_hp_period=self._params['zma_max_dc_hp_period'],
            
            zma_min_dc_cycle_part=self._params['zma_min_dc_cycle_part'],
            zma_min_dc_max_cycle=self._params['zma_min_dc_max_cycle'],
            zma_min_dc_min_cycle=self._params['zma_min_dc_min_cycle'],
            zma_min_dc_max_output=self._params['zma_min_dc_max_output'],
            zma_min_dc_min_output=self._params['zma_min_dc_min_output'],
            zma_min_dc_lp_period=self._params['zma_min_dc_lp_period'],
            zma_min_dc_hp_period=self._params['zma_min_dc_hp_period'],
            
            # ZMA動的Slow最大用パラメータ
            zma_slow_max_dc_cycle_part=self._params['zma_slow_max_dc_cycle_part'],
            zma_slow_max_dc_max_cycle=self._params['zma_slow_max_dc_max_cycle'],
            zma_slow_max_dc_min_cycle=self._params['zma_slow_max_dc_min_cycle'],
            zma_slow_max_dc_max_output=self._params['zma_slow_max_dc_max_output'],
            zma_slow_max_dc_min_output=self._params['zma_slow_max_dc_min_output'],
            zma_slow_max_dc_lp_period=self._params['zma_slow_max_dc_lp_period'],
            zma_slow_max_dc_hp_period=self._params['zma_slow_max_dc_hp_period'],
            
            # ZMA動的Slow最小用パラメータ
            zma_slow_min_dc_cycle_part=self._params['zma_slow_min_dc_cycle_part'],
            zma_slow_min_dc_max_cycle=self._params['zma_slow_min_dc_max_cycle'],
            zma_slow_min_dc_min_cycle=self._params['zma_slow_min_dc_min_cycle'],
            zma_slow_min_dc_max_output=self._params['zma_slow_min_dc_max_output'],
            zma_slow_min_dc_min_output=self._params['zma_slow_min_dc_min_output'],
            zma_slow_min_dc_lp_period=self._params['zma_slow_min_dc_lp_period'],
            zma_slow_min_dc_hp_period=self._params['zma_slow_min_dc_hp_period'],
            
            # ZMA動的Fast最大用パラメータ
            zma_fast_max_dc_cycle_part=self._params['zma_fast_max_dc_cycle_part'],
            zma_fast_max_dc_max_cycle=self._params['zma_fast_max_dc_max_cycle'],
            zma_fast_max_dc_min_cycle=self._params['zma_fast_max_dc_min_cycle'],
            zma_fast_max_dc_max_output=self._params['zma_fast_max_dc_max_output'],
            zma_fast_max_dc_min_output=self._params['zma_fast_max_dc_min_output'],
            zma_fast_max_dc_lp_period=self._params['zma_fast_max_dc_lp_period'],
            zma_fast_max_dc_hp_period=self._params['zma_fast_max_dc_hp_period'],
            
            # ZMA追加パラメータ
            zma_min_fast_period=self._params['zma_min_fast_period'],
            zma_hyper_smooth_period=self._params['zma_hyper_smooth_period'],
            
            # CATR基本パラメータ
            catr_detector_type=self._params['catr_detector_type'],
            catr_cycle_part=self._params['catr_cycle_part'],
            catr_lp_period=self._params['catr_lp_period'],
            catr_hp_period=self._params['catr_hp_period'],
            catr_max_cycle=self._params['catr_max_cycle'],
            catr_min_cycle=self._params['catr_min_cycle'],
            catr_max_output=self._params['catr_max_output'],
            catr_min_output=self._params['catr_min_output'],
            catr_smoother_type=self._params['catr_smoother_type']
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル生成"""
        return self.signal_generator.get_entry_signals(data)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成"""
        return self.signal_generator.get_exit_signals(data, position, index)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """最適化パラメータの作成"""
        # フィボナッチ数列を使用してパラメータを設定
        def round_to_fibonacci(value: float) -> int:
            """最も近いフィボナッチ数を返す"""
            fib = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
            return min(fib, key=lambda x: abs(x - value))
        
        # 基本パラメータ
        params = {
            'detector_type': trial.suggest_categorical('detector_type', ['hody', 'phac', 'dudi', 'dudi_e', 'hody_e', 'phac_e']),
            'cer_detector_type': trial.suggest_categorical('cer_detector_type', ['hody', 'phac', 'dudi', 'dudi_e', 'hody_e', 'phac_e']),
            'lp_period': round_to_fibonacci(trial.suggest_int('lp_period', 3, 8)),
            'hp_period': round_to_fibonacci(trial.suggest_int('hp_period', 34, 144)),
            'cycle_part': trial.suggest_float('cycle_part', 0.382, 0.618),
            'smoother_type': trial.suggest_categorical('smoother_type', ['alma', 'hyper']),
            'src_type': trial.suggest_categorical('src_type', ['hlc3', 'ohlc4']),
            'band_lookback': trial.suggest_int('band_lookback', 1, 3),
            
            # 動的乗数の範囲パラメータ
            'max_max_multiplier': trial.suggest_float('max_max_multiplier', 6.0, 8.0),
            'min_max_multiplier': trial.suggest_float('min_max_multiplier', 3.0, 5.0),
            'max_min_multiplier': trial.suggest_float('max_min_multiplier', 1.0, 2.0),
            'min_min_multiplier': trial.suggest_float('min_min_multiplier', 0.3, 0.7),
            
            # ZMA用パラメータ
            'zma_max_dc_cycle_part': trial.suggest_float('zma_max_dc_cycle_part', 0.382, 0.618),
            'zma_max_dc_max_cycle': round_to_fibonacci(trial.suggest_int('zma_max_dc_max_cycle', 89, 144)),
            'zma_max_dc_min_cycle': round_to_fibonacci(trial.suggest_int('zma_max_dc_min_cycle', 3, 8)),
            'zma_max_dc_max_output': round_to_fibonacci(trial.suggest_int('zma_max_dc_max_output', 55, 89)),
            'zma_max_dc_min_output': round_to_fibonacci(trial.suggest_int('zma_max_dc_min_output', 13, 34)),
            
            'zma_min_dc_cycle_part': trial.suggest_float('zma_min_dc_cycle_part', 0.236, 0.382),
            'zma_min_dc_max_cycle': round_to_fibonacci(trial.suggest_int('zma_min_dc_max_cycle', 34, 55)),
            'zma_min_dc_min_cycle': round_to_fibonacci(trial.suggest_int('zma_min_dc_min_cycle', 3, 8)),
            'zma_min_dc_max_output': round_to_fibonacci(trial.suggest_int('zma_min_dc_max_output', 8, 21)),
            'zma_min_dc_min_output': round_to_fibonacci(trial.suggest_int('zma_min_dc_min_output', 3, 8)),
            
            # CATR用パラメータ
            'catr_detector_type': trial.suggest_categorical('catr_detector_type', ['hody', 'phac', 'dudi', 'dudi_e', 'hody_e', 'phac_e']),
            'catr_cycle_part': trial.suggest_float('catr_cycle_part', 0.382, 0.618),
            'catr_max_cycle': round_to_fibonacci(trial.suggest_int('catr_max_cycle', 34, 55)),
            'catr_min_cycle': round_to_fibonacci(trial.suggest_int('catr_min_cycle', 3, 8)),
            'catr_max_output': round_to_fibonacci(trial.suggest_int('catr_max_output', 21, 34)),
            'catr_min_output': round_to_fibonacci(trial.suggest_int('catr_min_output', 3, 8)),
            'catr_smoother_type': trial.suggest_categorical('catr_smoother_type', ['alma', 'hyper'])
        }
        
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータをストラテジーのパラメータ形式に変換"""
        return params 