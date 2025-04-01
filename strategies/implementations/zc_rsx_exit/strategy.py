#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZCRSXExitSignalGenerator


class ZCRSXExitStrategy(BaseStrategy):
    """
    Zチャネル&RSXエグジット戦略
    
    特徴:
    - Zチャネルによる両方向のエントリーシグナル生成
    - ロングポジション：Zチャネルの売りシグナルによる通常エグジット
    - ショートポジション：
      1. Zチャネルの買いシグナルによるエグジット（従来ロジック）
      2. RSXトリガーの買いシグナルによるエグジット（追加ロジック）
    - サイクル効率比（CER）に基づく動的パラメータ最適化
    
    エントリー条件:
    - ロング: Zチャネルの買いシグナル
    - ショート: Zチャネルの売りシグナル
    
    エグジット条件:
    - ロング: Zチャネルの売りシグナル
    - ショート: Zチャネルの買いシグナル または RSXトリガーの買いシグナル
    """
    
    def __init__(
        self,
        # Zチャネルのパラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_multiplier: float = 5.0,
        min_multiplier: float = 0.5,
        smoother_type: str = 'alma',
        src_type: str = 'hlc3',
        band_lookback: int = 1,
        
        # RSXトリガーのパラメータ
        rsx_cycle_detector_type: str = 'hody_dc',
        rsx_lp_period: int = 13,
        rsx_hp_period: int = 144,
        rsx_cycle_part: float = 0.5,
        rsx_er_period: int = 10,
        
        # 最大ドミナントサイクル計算パラメータ
        max_dc_cycle_part: float = 0.5,
        max_dc_max_cycle: int = 55,
        max_dc_min_cycle: int = 5,
        max_dc_max_output: int = 21,
        max_dc_min_output: int = 13,
        
        # 最小ドミナントサイクル計算パラメータ
        min_dc_cycle_part: float = 0.25,
        min_dc_max_cycle: int = 34,
        min_dc_min_cycle: int = 3,
        min_dc_max_output: int = 8,
        min_dc_min_output: int = 3,
        
        # 買われすぎ/売られすぎレベルパラメータ
        min_high_level: float = 85.0,
        max_high_level: float = 95.0,
        min_low_level: float = 10.0,
        max_low_level: float = 20.0
    ):
        """
        初期化
        
        Args:
            cycle_detector_type: サイクル検出器の種類（デフォルト: 'hody_dc'）
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_multiplier: ケルトナーチャネルの最大乗数（デフォルト: 3.0）
            min_multiplier: ケルトナーチャネルの最小乗数（デフォルト: 1.5）
            smoother_type: 平滑化アルゴリズム（デフォルト: 'alma'）
            src_type: 価格ソースタイプ（デフォルト: 'hlc3'）
            band_lookback: 過去バンド参照期間（デフォルト: 1）
            
            rsx_cycle_detector_type: RSXサイクル検出器の種類（デフォルト: 'hody_dc'）
            rsx_lp_period: RSX用ローパスフィルターの期間（デフォルト: 13）
            rsx_hp_period: RSX用ハイパスフィルターの期間（デフォルト: 144）
            rsx_cycle_part: RSX用サイクル部分の倍率（デフォルト: 0.5）
            rsx_er_period: RSX用効率比の計算期間（デフォルト: 10）
            
            max_dc_cycle_part: 最大ドミナントサイクル計算の倍率（デフォルト: 0.5）
            max_dc_max_cycle: 最大ドミナントサイクル検出の最大期間（デフォルト: 55）
            max_dc_min_cycle: 最大ドミナントサイクル検出の最小期間（デフォルト: 5）
            max_dc_max_output: 最大ドミナントサイクル出力の最大値（デフォルト: 34）
            max_dc_min_output: 最大ドミナントサイクル出力の最小値（デフォルト: 14）
            
            min_dc_cycle_part: 最小ドミナントサイクル計算の倍率（デフォルト: 0.25）
            min_dc_max_cycle: 最小ドミナントサイクル検出の最大期間（デフォルト: 34）
            min_dc_min_cycle: 最小ドミナントサイクル検出の最小期間（デフォルト: 3）
            min_dc_max_output: 最小ドミナントサイクル出力の最大値（デフォルト: 13）
            min_dc_min_output: 最小ドミナントサイクル出力の最小値（デフォルト: 3）
            
            min_high_level: 最小買われすぎレベル（デフォルト: 75.0）
            max_high_level: 最大買われすぎレベル（デフォルト: 85.0）
            min_low_level: 最小売られすぎレベル（デフォルト: 25.0）
            max_low_level: 最大売られすぎレベル（デフォルト: 15.0）
        """
        super().__init__("ZCRSXExit")
        
        # パラメータの設定
        self._parameters = {
            # Zチャネルのパラメータ
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            
            # RSXトリガーのパラメータ
            'rsx_cycle_detector_type': rsx_cycle_detector_type,
            'rsx_lp_period': rsx_lp_period,
            'rsx_hp_period': rsx_hp_period,
            'rsx_cycle_part': rsx_cycle_part,
            'rsx_er_period': rsx_er_period,
            
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
            
            'min_high_level': min_high_level,
            'max_high_level': max_high_level,
            'min_low_level': min_low_level,
            'max_low_level': max_low_level
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZCRSXExitSignalGenerator(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            smoother_type=smoother_type,
            src_type=src_type,
            band_lookback=band_lookback,
            
            rsx_cycle_detector_type=rsx_cycle_detector_type,
            rsx_lp_period=rsx_lp_period,
            rsx_hp_period=rsx_hp_period,
            rsx_cycle_part=rsx_cycle_part,
            rsx_er_period=rsx_er_period,
            
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
            
            min_high_level=min_high_level,
            max_high_level=max_high_level,
            min_low_level=min_low_level,
            max_low_level=max_low_level
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
            # Zチャネルのパラメータ
            'hp_period': trial.suggest_int('hp_period', 62, 233),
            'max_multiplier': trial.suggest_float('max_multiplier', 2.0, 4.0, step=0.1),
            'min_multiplier': trial.suggest_float('min_multiplier', 1.0, 2.0, step=0.1),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # RSXトリガーのパラメータ
            'rsx_hp_period': trial.suggest_int('rsx_hp_period', 62, 233),
            'rsx_er_period': trial.suggest_int('rsx_er_period', 5, 21),
            
            # 最大ドミナントサイクル計算パラメータ
            'max_dc_max_output': trial.suggest_int('max_dc_max_output', 21, 55),
            
            # 最小ドミナントサイクル計算パラメータ
            'min_dc_max_output': trial.suggest_int('min_dc_max_output', 8, 21),
            
            # 買われすぎ/売られすぎレベルパラメータ
            'min_high_level': trial.suggest_float('min_high_level', 65.0, 85.0, step=5.0),
            'max_high_level': trial.suggest_float('max_high_level', 75.0, 95.0, step=5.0),
            'min_low_level': trial.suggest_float('min_low_level', 15.0, 35.0, step=5.0),
            'max_low_level': trial.suggest_float('max_low_level', 5.0, 25.0, step=5.0)
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
            # Zチャネルのパラメータ
            'cycle_detector_type': 'hody_dc',
            'lp_period': 5,
            'hp_period': int(params['hp_period']),
            'cycle_part': 0.5,
            'max_multiplier': float(params['max_multiplier']),
            'min_multiplier': float(params['min_multiplier']),
            'smoother_type': 'alma',
            'src_type': params['src_type'],
            'band_lookback': 1,
            
            # RSXトリガーのパラメータ
            'rsx_cycle_detector_type': 'hody_dc',
            'rsx_lp_period': 13,
            'rsx_hp_period': int(params['rsx_hp_period']),
            'rsx_cycle_part': 0.5,
            'rsx_er_period': int(params['rsx_er_period']),
            
            # 最大ドミナントサイクル計算パラメータ
            'max_dc_cycle_part': 0.5,
            'max_dc_max_cycle': 55,
            'max_dc_min_cycle': 5,
            'max_dc_max_output': int(params['max_dc_max_output']),
            'max_dc_min_output': 14,
            
            # 最小ドミナントサイクル計算パラメータ
            'min_dc_cycle_part': 0.25,
            'min_dc_max_cycle': 34,
            'min_dc_min_cycle': 3,
            'min_dc_max_output': int(params['min_dc_max_output']),
            'min_dc_min_output': 3,
            
            # 買われすぎ/売られすぎレベルパラメータ
            'min_high_level': float(params['min_high_level']),
            'max_high_level': float(params['max_high_level']),
            'min_low_level': float(params['min_low_level']),
            'max_low_level': float(params['max_low_level'])
        }
        
        return strategy_params 