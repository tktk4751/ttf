#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import SimpleZTrendSignalGenerator


class SimpleZTrendStrategy(BaseStrategy):
    """
    シンプルなZトレンド戦略 - ZTrendFilterなし
    
    特徴:
    - サイクル効率比（CER）に基づく動的パラメータ最適化
    - 25-75パーセンタイルによるレベル計算（動的期間）
    - ZATRによる洗練されたボラティリティ測定
    - ZTrendDirectionSignalのみに基づいたシンプルなエントリー/エグジット条件
    
    エントリー条件:
    - ロング: ZTrend方向が上昇トレンド(1)
    - ショート: ZTrend方向が下降トレンド(-1)
    
    エグジット条件:
    - ロング: ZTrend方向が下降トレンド(-1)
    - ショート: ZTrend方向が上昇トレンド(1)
    """
    
    def __init__(
        self,
        # ZTrendDirectionSignalのパラメータ
        cycle_detector_type: str = 'dudi_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        
        # CERのドミナントサイクル検出器用パラメータ
        cer_max_cycle: int = 233,
        cer_min_cycle: int = 13,
        cer_max_output: int = 144,
        cer_min_output: int = 21,
        
        # 最大パーセンタイル期間用（長期）ドミナントサイクル検出器のパラメータ
        max_percentile_dc_cycle_part: float = 0.5,
        max_percentile_dc_max_cycle: int = 233,
        max_percentile_dc_min_cycle: int = 13,
        max_percentile_dc_max_output: int = 144,
        max_percentile_dc_min_output: int = 21,
        
        # 最小パーセンタイル期間用（短期）ドミナントサイクル検出器のパラメータ
        min_percentile_dc_cycle_part: float = 0.5,
        min_percentile_dc_max_cycle: int = 55,
        min_percentile_dc_min_cycle: int = 5,
        min_percentile_dc_max_output: int = 34,
        min_percentile_dc_min_output: int = 8,
        
        # ZATR用ドミナントサイクル検出器のパラメータ
        zatr_max_dc_cycle_part: float = 0.5,
        zatr_max_dc_max_cycle: int = 55,
        zatr_max_dc_min_cycle: int = 5,
        zatr_max_dc_max_output: int = 55,
        zatr_max_dc_min_output: int = 5,
        zatr_min_dc_cycle_part: float = 0.25,
        zatr_min_dc_max_cycle: int = 34,
        zatr_min_dc_min_cycle: int = 3,
        zatr_min_dc_max_output: int = 13,
        zatr_min_dc_min_output: int = 3,
        
        # パーセンタイル乗数
        max_percentile_cycle_mult: float = 0.5,  # 最大パーセンタイル期間のサイクル乗数
        min_percentile_cycle_mult: float = 0.25,  # 最小パーセンタイル期間のサイクル乗数
        
        # ATR乗数
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0,
        
        # 動的乗数の範囲
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 3.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.3,    # 最小乗数の最小値
        
        # その他の設定
        smoother_type: str = 'alma',   # 平滑化アルゴリズム（'alma'または'hyper'）
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: サイクル検出器の種類
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分
            cer_max_cycle: CER用最大サイクル
            cer_min_cycle: CER用最小サイクル
            cer_max_output: CER用最大出力
            cer_min_output: CER用最小出力
            max_percentile_dc_cycle_part: 最大パーセンタイル期間用DCの周期部分
            max_percentile_dc_max_cycle: 最大パーセンタイル期間用DCの最大周期
            max_percentile_dc_min_cycle: 最大パーセンタイル期間用DCの最小周期
            max_percentile_dc_max_output: 最大パーセンタイル期間用DCの最大出力
            max_percentile_dc_min_output: 最大パーセンタイル期間用DCの最小出力
            min_percentile_dc_cycle_part: 最小パーセンタイル期間用DCの周期部分
            min_percentile_dc_max_cycle: 最小パーセンタイル期間用DCの最大周期
            min_percentile_dc_min_cycle: 最小パーセンタイル期間用DCの最小周期
            min_percentile_dc_max_output: 最小パーセンタイル期間用DCの最大出力
            min_percentile_dc_min_output: 最小パーセンタイル期間用DCの最小出力
            zatr_max_dc_cycle_part: ZATR最大DCの周期部分
            zatr_max_dc_max_cycle: ZATR最大DCの最大周期
            zatr_max_dc_min_cycle: ZATR最大DCの最小周期
            zatr_max_dc_max_output: ZATR最大DCの最大出力
            zatr_max_dc_min_output: ZATR最大DCの最小出力
            zatr_min_dc_cycle_part: ZATR最小DCの周期部分
            zatr_min_dc_max_cycle: ZATR最小DCの最大周期
            zatr_min_dc_min_cycle: ZATR最小DCの最小周期
            zatr_min_dc_max_output: ZATR最小DCの最大出力
            zatr_min_dc_min_output: ZATR最小DCの最小出力
            max_percentile_cycle_mult: 最大パーセンタイル期間の周期乗数
            min_percentile_cycle_mult: 最小パーセンタイル期間の周期乗数
            max_multiplier: ATR乗数の最大値（レガシーパラメータ）
            min_multiplier: ATR乗数の最小値（レガシーパラメータ）
            max_max_multiplier: 最大乗数の最大値
            min_max_multiplier: 最大乗数の最小値
            max_min_multiplier: 最小乗数の最大値
            min_min_multiplier: 最小乗数の最小値
            smoother_type: スムーザーの種類
            src_type: ソースタイプ
        """
        super().__init__("SimpleZTrend")
        
        # パラメータの設定
        self._parameters = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'cer_max_cycle': cer_max_cycle,
            'cer_min_cycle': cer_min_cycle,
            'cer_max_output': cer_max_output,
            'cer_min_output': cer_min_output,
            'max_percentile_dc_cycle_part': max_percentile_dc_cycle_part,
            'max_percentile_dc_max_cycle': max_percentile_dc_max_cycle,
            'max_percentile_dc_min_cycle': max_percentile_dc_min_cycle,
            'max_percentile_dc_max_output': max_percentile_dc_max_output,
            'max_percentile_dc_min_output': max_percentile_dc_min_output,
            'min_percentile_dc_cycle_part': min_percentile_dc_cycle_part,
            'min_percentile_dc_max_cycle': min_percentile_dc_max_cycle,
            'min_percentile_dc_min_cycle': min_percentile_dc_min_cycle,
            'min_percentile_dc_max_output': min_percentile_dc_max_output,
            'min_percentile_dc_min_output': min_percentile_dc_min_output,
            'zatr_max_dc_cycle_part': zatr_max_dc_cycle_part,
            'zatr_max_dc_max_cycle': zatr_max_dc_max_cycle,
            'zatr_max_dc_min_cycle': zatr_max_dc_min_cycle,
            'zatr_max_dc_max_output': zatr_max_dc_max_output,
            'zatr_max_dc_min_output': zatr_max_dc_min_output,
            'zatr_min_dc_cycle_part': zatr_min_dc_cycle_part,
            'zatr_min_dc_max_cycle': zatr_min_dc_max_cycle,
            'zatr_min_dc_min_cycle': zatr_min_dc_min_cycle,
            'zatr_min_dc_max_output': zatr_min_dc_max_output,
            'zatr_min_dc_min_output': zatr_min_dc_min_output,
            'max_percentile_cycle_mult': max_percentile_cycle_mult,
            'min_percentile_cycle_mult': min_percentile_cycle_mult,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            'smoother_type': smoother_type,
            'src_type': src_type
        }
        
        # シグナル生成器の初期化
        self.signal_generator = SimpleZTrendSignalGenerator(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            cer_max_cycle=cer_max_cycle,
            cer_min_cycle=cer_min_cycle,
            cer_max_output=cer_max_output,
            cer_min_output=cer_min_output,
            max_percentile_dc_cycle_part=max_percentile_dc_cycle_part,
            max_percentile_dc_max_cycle=max_percentile_dc_max_cycle,
            max_percentile_dc_min_cycle=max_percentile_dc_min_cycle,
            max_percentile_dc_max_output=max_percentile_dc_max_output,
            max_percentile_dc_min_output=max_percentile_dc_min_output,
            min_percentile_dc_cycle_part=min_percentile_dc_cycle_part,
            min_percentile_dc_max_cycle=min_percentile_dc_max_cycle,
            min_percentile_dc_min_cycle=min_percentile_dc_min_cycle,
            min_percentile_dc_max_output=min_percentile_dc_max_output,
            min_percentile_dc_min_output=min_percentile_dc_min_output,
            zatr_max_dc_cycle_part=zatr_max_dc_cycle_part,
            zatr_max_dc_max_cycle=zatr_max_dc_max_cycle,
            zatr_max_dc_min_cycle=zatr_max_dc_min_cycle,
            zatr_max_dc_max_output=zatr_max_dc_max_output,
            zatr_max_dc_min_output=zatr_max_dc_min_output,
            zatr_min_dc_cycle_part=zatr_min_dc_cycle_part,
            zatr_min_dc_max_cycle=zatr_min_dc_max_cycle,
            zatr_min_dc_min_cycle=zatr_min_dc_min_cycle,
            zatr_min_dc_max_output=zatr_min_dc_max_output,
            zatr_min_dc_min_output=zatr_min_dc_min_output,
            max_percentile_cycle_mult=max_percentile_cycle_mult,
            min_percentile_cycle_mult=min_percentile_cycle_mult,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            smoother_type=smoother_type,
            src_type=src_type
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
            'cycle_detector_type': trial.suggest_categorical('cycle_detector_type', ['dudi_dc', 'hody_dc', 'phac_dc']),
            'lp_period': trial.suggest_int('lp_period', 3, 21),
            'hp_period': trial.suggest_int('hp_period', 62, 233),
            'cycle_part': trial.suggest_float('cycle_part', 0.3, 0.8, step=0.1),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # CERパラメータ
            'cer_max_cycle': trial.suggest_int('cer_max_cycle', 100, 300),
            'cer_min_cycle': trial.suggest_int('cer_min_cycle', 5, 20),
            'cer_max_output': trial.suggest_int('cer_max_output', 100, 200),
            'cer_min_output': trial.suggest_int('cer_min_output', 10, 40),
            
            # 最大パーセンタイル期間用（長期）ドミナントサイクル検出器のパラメータ
            'max_percentile_dc_cycle_part': trial.suggest_float('max_percentile_dc_cycle_part', 0.3, 0.8, step=0.1),
            'max_percentile_dc_max_cycle': trial.suggest_int('max_percentile_dc_max_cycle', 100, 250),
            'max_percentile_dc_min_cycle': trial.suggest_int('max_percentile_dc_min_cycle', 5, 20),
            'max_percentile_dc_max_output': trial.suggest_int('max_percentile_dc_max_output', 80, 200),
            'max_percentile_dc_min_output': trial.suggest_int('max_percentile_dc_min_output', 10, 30),
            
            # 最小パーセンタイル期間用（短期）ドミナントサイクル検出器のパラメータ
            'min_percentile_dc_cycle_part': trial.suggest_float('min_percentile_dc_cycle_part', 0.2, 0.6, step=0.1),
            'min_percentile_dc_max_cycle': trial.suggest_int('min_percentile_dc_max_cycle', 30, 80),
            'min_percentile_dc_min_cycle': trial.suggest_int('min_percentile_dc_min_cycle', 3, 15),
            'min_percentile_dc_max_output': trial.suggest_int('min_percentile_dc_max_output', 13, 50),
            'min_percentile_dc_min_output': trial.suggest_int('min_percentile_dc_min_output', 3, 15),
            
            # ZATR用ドミナントサイクル検出器のパラメータ
            'zatr_max_dc_cycle_part': trial.suggest_float('zatr_max_dc_cycle_part', 0.3, 0.8, step=0.1),
            'zatr_max_dc_max_cycle': trial.suggest_int('zatr_max_dc_max_cycle', 30, 80),
            'zatr_max_dc_min_cycle': trial.suggest_int('zatr_max_dc_min_cycle', 3, 8),
            'zatr_max_dc_max_output': trial.suggest_int('zatr_max_dc_max_output', 20, 80),
            'zatr_max_dc_min_output': trial.suggest_int('zatr_max_dc_min_output', 3, 10),
            'zatr_min_dc_cycle_part': trial.suggest_float('zatr_min_dc_cycle_part', 0.1, 0.4, step=0.05),
            'zatr_min_dc_max_cycle': trial.suggest_int('zatr_min_dc_max_cycle', 13, 50),
            'zatr_min_dc_min_cycle': trial.suggest_int('zatr_min_dc_min_cycle', 3, 8),
            'zatr_min_dc_max_output': trial.suggest_int('zatr_min_dc_max_output', 8, 20),
            'zatr_min_dc_min_output': trial.suggest_int('zatr_min_dc_min_output', 3, 8),
            
            # パーセンタイル乗数
            'max_percentile_cycle_mult': trial.suggest_float('max_percentile_cycle_mult', 0.3, 0.7, step=0.1),
            'min_percentile_cycle_mult': trial.suggest_float('min_percentile_cycle_mult', 0.2, 0.4, step=0.05),
            
            # ATR乗数と動的乗数範囲
            'max_multiplier': trial.suggest_float('max_multiplier', 2.0, 4.0, step=0.1),
            'min_multiplier': trial.suggest_float('min_multiplier', 0.8, 2.0, step=0.1),
            'max_max_multiplier': trial.suggest_float('max_max_multiplier', 6.0, 10.0, step=0.5),
            'min_max_multiplier': trial.suggest_float('min_max_multiplier', 2.0, 4.0, step=0.1),
            'max_min_multiplier': trial.suggest_float('max_min_multiplier', 1.0, 2.0, step=0.1),
            'min_min_multiplier': trial.suggest_float('min_min_multiplier', 0.2, 0.6, step=0.1),
            
            # その他の設定
            'smoother_type': trial.suggest_categorical('smoother_type', ['alma', 'hyper'])
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
            'cer_max_cycle': int(params['cer_max_cycle']),
            'cer_min_cycle': int(params['cer_min_cycle']),
            'cer_max_output': int(params['cer_max_output']),
            'cer_min_output': int(params['cer_min_output']),
            'max_percentile_dc_cycle_part': float(params['max_percentile_dc_cycle_part']),
            'max_percentile_dc_max_cycle': int(params['max_percentile_dc_max_cycle']),
            'max_percentile_dc_min_cycle': int(params['max_percentile_dc_min_cycle']),
            'max_percentile_dc_max_output': int(params['max_percentile_dc_max_output']),
            'max_percentile_dc_min_output': int(params['max_percentile_dc_min_output']),
            'min_percentile_dc_cycle_part': float(params['min_percentile_dc_cycle_part']),
            'min_percentile_dc_max_cycle': int(params['min_percentile_dc_max_cycle']),
            'min_percentile_dc_min_cycle': int(params['min_percentile_dc_min_cycle']),
            'min_percentile_dc_max_output': int(params['min_percentile_dc_max_output']),
            'min_percentile_dc_min_output': int(params['min_percentile_dc_min_output']),
            'zatr_max_dc_cycle_part': float(params['zatr_max_dc_cycle_part']),
            'zatr_max_dc_max_cycle': int(params['zatr_max_dc_max_cycle']),
            'zatr_max_dc_min_cycle': int(params['zatr_max_dc_min_cycle']),
            'zatr_max_dc_max_output': int(params['zatr_max_dc_max_output']),
            'zatr_max_dc_min_output': int(params['zatr_max_dc_min_output']),
            'zatr_min_dc_cycle_part': float(params['zatr_min_dc_cycle_part']),
            'zatr_min_dc_max_cycle': int(params['zatr_min_dc_max_cycle']),
            'zatr_min_dc_min_cycle': int(params['zatr_min_dc_min_cycle']),
            'zatr_min_dc_max_output': int(params['zatr_min_dc_max_output']),
            'zatr_min_dc_min_output': int(params['zatr_min_dc_min_output']),
            'max_percentile_cycle_mult': float(params['max_percentile_cycle_mult']),
            'min_percentile_cycle_mult': float(params['min_percentile_cycle_mult']),
            'max_multiplier': float(params['max_multiplier']),
            'min_multiplier': float(params['min_multiplier']),
            'max_max_multiplier': float(params['max_max_multiplier']),
            'min_max_multiplier': float(params['min_max_multiplier']),
            'max_min_multiplier': float(params['max_min_multiplier']),
            'min_min_multiplier': float(params['min_min_multiplier']),
            'smoother_type': params['smoother_type']
        }
        return strategy_params 