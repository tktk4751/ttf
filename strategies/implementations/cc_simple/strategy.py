#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
# 対応するシグナル生成器をインポート
from .signal_generator import CCSimpleSignalGenerator


class CCSimpleStrategy(BaseStrategy): # クラス名を変更
    """
    CCチャネル戦略（シンプル版）
    
    特徴:
    - CCチャネルによるエントリーポイント検出
    - CCチャネルとその内部インジケーターのパラメータは外部から設定・最適化可能
    
    エントリー条件:
    - ロング: CCチャネルの買いシグナル
    - ショート: CCチャネルの売りシグナル
    
    エグジット条件:
    - ロング: CCチャネルの売りシグナル
    - ショート: CCチャネルの買いシグナル
    """
    
    def __init__( # 引数を band_lookback のみに変更 -> 全パラメータを受け取るように変更
        self,
        band_lookback: int = 1, 
        # CCSimpleSignalGenerator に渡すパラメータを追加
        multiplier_method: str = 'new', # multiplier_method を追加
        new_method_er_source: str = 'cver', # new_method_er_source を追加
        cc_max_max_multiplier: float = 7.0, # デフォルト値を修正
        cc_min_max_multiplier: float = 5.5,
        cc_max_min_multiplier: float = 1.5,
        cc_min_min_multiplier: float = 0.3,
        cma_detector_type: str = 'phac_e', 
        cma_cycle_part: float = 0.5, 
        cma_lp_period: int = 5, 
        cma_hp_period: int = 144, 
        cma_max_cycle: int = 100, 
        cma_min_cycle: int = 10, 
        cma_max_output: int = 55, 
        cma_min_output: int = 13, 
        cma_fast_period: int = 3, 
        cma_slow_period: int = 34, 
        cma_src_type: str = 'hlc3',
        catr_detector_type: str = 'phac_e', 
        catr_cycle_part: float = 0.5, 
        catr_lp_period: int = 5, 
        catr_hp_period: int = 62, 
        catr_max_cycle: int = 55, 
        catr_min_cycle: int = 5, 
        catr_max_output: int = 34, 
        catr_min_output: int = 5, 
        catr_smoother_type: str = 'alma',
        cver_detector_type: str = 'dudi_e', 
        cver_lp_period: int = 5, 
        cver_hp_period: int = 62, 
        cver_cycle_part: float = 0.382, 
        cver_max_cycle: int = 55, 
        cver_min_cycle: int = 5, 
        cver_max_output: int = 34, 
        cver_min_output: int = 5, 
        cver_src_type: str = 'hlc3',
        cer_detector_type: str = 'phac', 
        cer_lp_period: int = 5,
        cer_hp_period: int = 62,
        cer_cycle_part: float = 0.382,
        cer_max_cycle: int = 55,
        cer_min_cycle: int = 5,
        cer_max_output: int = 34,
        cer_min_output: int = 5,
        cer_src_type: str = 'hlc3'
    ):
        """
        初期化
        
        Args:
            band_lookback: 過去バンド参照期間（デフォルト: 1）
            **kwargs: CCSimpleSignalGenerator の初期化パラメータ
        """
        super().__init__("CCSimple") # 戦略名を更新
        
        # パラメータの設定 (全パラメータを保存)
        self._parameters = {
            'band_lookback': band_lookback,
            'multiplier_method': multiplier_method, # multiplier_method を追加
            'new_method_er_source': new_method_er_source, # new_method_er_source を追加
            'cc_max_max_multiplier': cc_max_max_multiplier,
            'cc_min_max_multiplier': cc_min_max_multiplier,
            'cc_max_min_multiplier': cc_max_min_multiplier,
            'cc_min_min_multiplier': cc_min_min_multiplier,
            'cma_detector_type': cma_detector_type, 
            'cma_cycle_part': cma_cycle_part, 
            'cma_lp_period': cma_lp_period, 
            'cma_hp_period': cma_hp_period, 
            'cma_max_cycle': cma_max_cycle, 
            'cma_min_cycle': cma_min_cycle, 
            'cma_max_output': cma_max_output, 
            'cma_min_output': cma_min_output, 
            'cma_fast_period': cma_fast_period, 
            'cma_slow_period': cma_slow_period, 
            'cma_src_type': cma_src_type,
            'catr_detector_type': catr_detector_type, 
            'catr_cycle_part': catr_cycle_part, 
            'catr_lp_period': catr_lp_period, 
            'catr_hp_period': catr_hp_period, 
            'catr_max_cycle': catr_max_cycle, 
            'catr_min_cycle': catr_min_cycle, 
            'catr_max_output': catr_max_output, 
            'catr_min_output': catr_min_output, 
            'catr_smoother_type': catr_smoother_type,
            'cver_detector_type': cver_detector_type, 
            'cver_lp_period': cver_lp_period, 
            'cver_hp_period': cver_hp_period, 
            'cver_cycle_part': cver_cycle_part, 
            'cver_max_cycle': cver_max_cycle, 
            'cver_min_cycle': cver_min_cycle, 
            'cver_max_output': cver_max_output, 
            'cver_min_output': cver_min_output, 
            'cver_src_type': cver_src_type,
            'cer_detector_type': cer_detector_type, 
            'cer_lp_period': cer_lp_period,
            'cer_hp_period': cer_hp_period,
            'cer_cycle_part': cer_cycle_part,
            'cer_max_cycle': cer_max_cycle,
            'cer_min_cycle': cer_min_cycle,
            'cer_max_output': cer_max_output,
            'cer_min_output': cer_min_output,
            'cer_src_type': cer_src_type
        }
        
        # シグナル生成器の初期化 (全パラメータを渡す)
        self.signal_generator = CCSimpleSignalGenerator(
            band_lookback=band_lookback,
            multiplier_method=multiplier_method, # multiplier_method を渡す
            new_method_er_source=new_method_er_source, # new_method_er_source を渡す
            cc_max_max_multiplier=cc_max_max_multiplier,
            cc_min_max_multiplier=cc_min_max_multiplier,
            cc_max_min_multiplier=cc_max_min_multiplier,
            cc_min_min_multiplier=cc_min_min_multiplier,
            cma_detector_type=cma_detector_type, 
            cma_cycle_part=cma_cycle_part, 
            cma_lp_period=cma_lp_period, 
            cma_hp_period=cma_hp_period, 
            cma_max_cycle=cma_max_cycle, 
            cma_min_cycle=cma_min_cycle, 
            cma_max_output=cma_max_output, 
            cma_min_output=cma_min_output, 
            cma_fast_period=cma_fast_period, 
            cma_slow_period=cma_slow_period, 
            cma_src_type=cma_src_type,
            catr_detector_type=catr_detector_type, 
            catr_cycle_part=catr_cycle_part, 
            catr_lp_period=catr_lp_period, 
            catr_hp_period=catr_hp_period, 
            catr_max_cycle=catr_max_cycle, 
            catr_min_cycle=catr_min_cycle, 
            catr_max_output=catr_max_output, 
            catr_min_output=catr_min_output, 
            catr_smoother_type=catr_smoother_type,
            cver_detector_type=cver_detector_type, 
            cver_lp_period=cver_lp_period, 
            cver_hp_period=cver_hp_period, 
            cver_cycle_part=cver_cycle_part, 
            cver_max_cycle=cver_max_cycle, 
            cver_min_cycle=cver_min_cycle, 
            cver_max_output=cver_max_output, 
            cver_min_output=cver_min_output, 
            cver_src_type=cver_src_type,
            cer_detector_type=cer_detector_type, 
            cer_lp_period=cer_lp_period,
            cer_hp_period=cer_hp_period,
            cer_cycle_part=cer_cycle_part,
            cer_max_cycle=cer_max_cycle,
            cer_min_cycle=cer_min_cycle,
            cer_max_output=cer_max_output,
            cer_min_output=cer_min_output,
            cer_src_type=cer_src_type
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
            Dict[str, Any]: 最適化パラメータ (band_lookback のみ) -> 全パラメータに変更
        """
        # 検出器タイプとソースタイプを定義
        detector_types = ['dudi', 'hody', 'phac', 'dudi_e', 'hody_e', 'phac_e']
        src_types = ['close', 'hlc3', 'hl2', 'ohlc4']
        smoother_types = ['alma', 'hyper']

        params = {
            'band_lookback': 1,
            'multiplier_method': trial.suggest_categorical('multiplier_method', ['adaptive', 'new']),
            'new_method_er_source': trial.suggest_categorical('new_method_er_source', ['cver', 'cer']), # 追加
            
            # CCチャネル固有パラメータ (adaptiveメソッドの場合のみ影響)
            'cc_max_max_multiplier': trial.suggest_float('cc_max_max_multiplier', 5.0, 10.0, step=0.5),
            'cc_min_max_multiplier': trial.suggest_float('cc_min_max_multiplier', 2.0, 6.0, step=0.5), 
            'cc_max_min_multiplier': trial.suggest_float('cc_max_min_multiplier', 1.0, 3.0, step=0.1), 
            'cc_min_min_multiplier': trial.suggest_float('cc_min_min_multiplier', 0.1, 1.5, step=0.1), 
            
            # CMAパラメータ
            'cma_detector_type': trial.suggest_categorical('cma_detector_type', detector_types),
            'cma_cycle_part': trial.suggest_float('cma_cycle_part', 0.3, 0.8, step=0.1),
            'cma_lp_period': trial.suggest_int('cma_lp_period', 3, 15), # 範囲調整
            'cma_hp_period': trial.suggest_int('cma_hp_period', 34, 144), # 範囲調整
            'cma_max_cycle': trial.suggest_int('cma_max_cycle', 34, 144), # 範囲調整
            'cma_min_cycle': trial.suggest_int('cma_min_cycle', 3, 21), # 範囲調整
            'cma_max_output': trial.suggest_int('cma_max_output', 21, 89), # 範囲調整
            'cma_min_output': trial.suggest_int('cma_min_output', 5, 21), # 範囲調整
            'cma_fast_period': trial.suggest_int('cma_fast_period', 2, 5), # 少し自由度を与える
            'cma_slow_period': trial.suggest_int('cma_slow_period', 20, 50), # 少し自由度を与える
            'cma_src_type': trial.suggest_categorical('cma_src_type', src_types),
            
            # CATRパラメータ
            'catr_detector_type': trial.suggest_categorical('catr_detector_type', detector_types),
            'catr_cycle_part': trial.suggest_float('catr_cycle_part', 0.3, 0.8, step=0.1),
            'catr_lp_period': trial.suggest_int('catr_lp_period', 3, 15),
            'catr_hp_period': trial.suggest_int('catr_hp_period', 34, 144),
            'catr_max_cycle': trial.suggest_int('catr_max_cycle', 34, 144),
            'catr_min_cycle': trial.suggest_int('catr_min_cycle', 3, 21),
            'catr_max_output': trial.suggest_int('catr_max_output', 21, 89),
            'catr_min_output': trial.suggest_int('catr_min_output', 3, 21),
            'catr_smoother_type': trial.suggest_categorical('catr_smoother_type', smoother_types),

            # CVERパラメータ
            'cver_detector_type': trial.suggest_categorical('cver_detector_type', detector_types),
            'cver_lp_period': trial.suggest_int('cver_lp_period', 3, 15),
            'cver_hp_period': trial.suggest_int('cver_hp_period', 55, 233), # より長い期間も試す
            'cver_cycle_part': trial.suggest_float('cver_cycle_part', 0.3, 0.8, step=0.1),
            'cver_max_cycle': trial.suggest_int('cver_max_cycle', 55, 233),
            'cver_min_cycle': trial.suggest_int('cver_min_cycle', 3, 34),
            'cver_max_output': trial.suggest_int('cver_max_output', 34, 144),
            'cver_min_output': trial.suggest_int('cver_min_output', 3, 34),
            'cver_src_type': trial.suggest_categorical('cver_src_type', src_types),
            
            # CERパラメータ (CMA/CATR用)
            'cer_detector_type': trial.suggest_categorical('cer_detector_type', detector_types),
            'cer_lp_period': trial.suggest_int('cer_lp_period', 3, 15),
            'cer_hp_period': trial.suggest_int('cer_hp_period', 55, 233),
            'cer_cycle_part': trial.suggest_float('cer_cycle_part', 0.3, 0.8, step=0.1),
            'cer_max_cycle': trial.suggest_int('cer_max_cycle', 55, 233),
            'cer_min_cycle': trial.suggest_int('cer_min_cycle', 3, 34),
            'cer_max_output': trial.suggest_int('cer_max_output', 34, 144),
            'cer_min_output': trial.suggest_int('cer_min_output', 3, 34),
            'cer_src_type': trial.suggest_categorical('cer_src_type', src_types),
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            Dict[str, Any]: 戦略パラメータ (band_lookback のみ) -> 全パラメータに変更
        """
        strategy_params = {
            'band_lookback': 1,
            'multiplier_method': params['multiplier_method'],
            'new_method_er_source': params['new_method_er_source'], # 追加
            # CCチャネル固有パラメータ
            'cc_max_max_multiplier': float(params['cc_max_max_multiplier']),
            'cc_min_max_multiplier': float(params['cc_min_max_multiplier']),
            'cc_max_min_multiplier': float(params['cc_max_min_multiplier']),
            'cc_min_min_multiplier': float(params['cc_min_min_multiplier']),
            
            # CMAパラメータ
            'cma_detector_type': params['cma_detector_type'],
            'cma_cycle_part': float(params['cma_cycle_part']),
            'cma_lp_period': int(params['cma_lp_period']),
            'cma_hp_period': int(params['cma_hp_period']),
            'cma_max_cycle': int(params['cma_max_cycle']),
            'cma_min_cycle': int(params['cma_min_cycle']),
            'cma_max_output': int(params['cma_max_output']),
            'cma_min_output': int(params['cma_min_output']),
            'cma_fast_period': int(params['cma_fast_period']),
            'cma_slow_period': int(params['cma_slow_period']),
            'cma_src_type': params['cma_src_type'],
            
            # CATRパラメータ
            'catr_detector_type': params['catr_detector_type'],
            'catr_cycle_part': float(params['catr_cycle_part']),
            'catr_lp_period': int(params['catr_lp_period']),
            'catr_hp_period': int(params['catr_hp_period']),
            'catr_max_cycle': int(params['catr_max_cycle']),
            'catr_min_cycle': int(params['catr_min_cycle']),
            'catr_max_output': int(params['catr_max_output']),
            'catr_min_output': int(params['catr_min_output']),
            'catr_smoother_type': params['catr_smoother_type'],

            # CVERパラメータ
            'cver_detector_type': params['cver_detector_type'],
            'cver_lp_period': int(params['cver_lp_period']),
            'cver_hp_period': int(params['cver_hp_period']),
            'cver_cycle_part': float(params['cver_cycle_part']),
            'cver_max_cycle': int(params['cver_max_cycle']),
            'cver_min_cycle': int(params['cver_min_cycle']),
            'cver_max_output': int(params['cver_max_output']),
            'cver_min_output': int(params['cver_min_output']),
            'cver_src_type': params['cver_src_type'],
            
            # CERパラメータ (CMA/CATR用)
            'cer_detector_type': params['cer_detector_type'],
            'cer_lp_period': int(params['cer_lp_period']),
            'cer_hp_period': int(params['cer_hp_period']),
            'cer_cycle_part': float(params['cer_cycle_part']),
            'cer_max_cycle': int(params['cer_max_cycle']),
            'cer_min_cycle': int(params['cer_min_cycle']),
            'cer_max_output': int(params['cer_max_output']),
            'cer_min_output': int(params['cer_min_output']),
            'cer_src_type': params['cer_src_type'],
        }
        return strategy_params 