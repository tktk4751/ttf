#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZMACrossSignalGenerator


class ZMACrossStrategy(BaseStrategy):
    """
    ZMA交差戦略
    
    特徴:
    - 短期ZMAと長期ZMAの交差を利用した明確なエントリー/エグジットシグナル
    - 効率比（CER）に基づいて動的に調整される移動平均線
    - Numbaによる高速計算
    
    エントリー条件:
    - ロング: ZMA交差のゴールデンクロス(短期ZMAが長期ZMAを下から上に抜ける)
    - ショート: ZMA交差のデッドクロス(短期ZMAが長期ZMAを上から下に抜ける)
    
    エグジット条件:
    - ロング: ZMA交差のデッドクロス
    - ショート: ZMA交差のゴールデンクロス
    """
    
    def __init__(
        self,
        # ドミナントサイクル・効率比（CER）の基本パラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 13,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        
        # 短期ZMA用パラメータ
        fast_max_dc_cycle_part: float = 0.25,
        fast_max_dc_max_cycle: int = 55,
        fast_max_dc_min_cycle: int = 5,
        fast_max_dc_max_output: int = 89,
        fast_max_dc_min_output: int = 21,
        
        fast_min_dc_cycle_part: float = 0.25,
        fast_min_dc_max_cycle: int = 21,
        fast_min_dc_min_cycle: int = 5,
        fast_min_dc_max_output: int = 55,
        fast_min_dc_min_output: int = 5,
        
        fast_max_slow_period: int = 34,
        fast_min_slow_period: int = 13,
        fast_max_fast_period: int = 8,
        fast_min_fast_period: int = 3,
        fast_hyper_smooth_period: int = 0,
        
        # 長期ZMA用パラメータ
        slow_max_dc_cycle_part: float = 0.5,
        slow_max_dc_max_cycle: int = 144,
        slow_max_dc_min_cycle: int = 13,
        slow_max_dc_max_output: int = 233,
        slow_max_dc_min_output: int = 55,
        
        slow_min_dc_cycle_part: float = 0.25,
        slow_min_dc_max_cycle: int = 55,
        slow_min_dc_min_cycle: int = 5,
        slow_min_dc_max_output: int = 89,
        slow_min_dc_min_output: int = 21,
        
        slow_max_slow_period: int = 34,
        slow_min_slow_period: int = 13,
        slow_max_fast_period: int = 8,
        slow_min_fast_period: int = 3,
        slow_hyper_smooth_period: int = 0,
        
        # ソースタイプ
        src_type: str = 'hlc3'
    ):
        """
        初期化
        
        Args:
            cycle_detector_type: サイクル検出器の種類
            lp_period: ローパスフィルターの期間
            hp_period: ハイパスフィルターの期間
            cycle_part: サイクル部分の割合
            
            fast_max_dc_*: 短期ZMAの最大期間用ドミナントサイクルパラメータ
            fast_min_dc_*: 短期ZMAの最小期間用ドミナントサイクルパラメータ
            fast_max_slow_period: 短期ZMAの最大スロー期間
            fast_min_slow_period: 短期ZMAの最小スロー期間
            fast_max_fast_period: 短期ZMAの最大ファスト期間
            fast_min_fast_period: 短期ZMAの最小ファスト期間
            fast_hyper_smooth_period: 短期ZMAのハイパースムーザー期間
            
            slow_max_dc_*: 長期ZMAの最大期間用ドミナントサイクルパラメータ
            slow_min_dc_*: 長期ZMAの最小期間用ドミナントサイクルパラメータ
            slow_max_slow_period: 長期ZMAの最大スロー期間
            slow_min_slow_period: 長期ZMAの最小スロー期間
            slow_max_fast_period: 長期ZMAの最大ファスト期間
            slow_min_fast_period: 長期ZMAの最小ファスト期間
            slow_hyper_smooth_period: 長期ZMAのハイパースムーザー期間
            
            src_type: 価格計算の元となる価格タイプ
        """
        super().__init__("ZMACross")
        
        # パラメータの設定
        self._parameters = {
            # ドミナントサイクル・効率比（CER）の基本パラメータ
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            
            # 短期ZMA用パラメータ
            'fast_max_dc_cycle_part': fast_max_dc_cycle_part,
            'fast_max_dc_max_cycle': fast_max_dc_max_cycle,
            'fast_max_dc_min_cycle': fast_max_dc_min_cycle,
            'fast_max_dc_max_output': fast_max_dc_max_output,
            'fast_max_dc_min_output': fast_max_dc_min_output,
            
            'fast_min_dc_cycle_part': fast_min_dc_cycle_part,
            'fast_min_dc_max_cycle': fast_min_dc_max_cycle,
            'fast_min_dc_min_cycle': fast_min_dc_min_cycle,
            'fast_min_dc_max_output': fast_min_dc_max_output,
            'fast_min_dc_min_output': fast_min_dc_min_output,
            
            'fast_max_slow_period': fast_max_slow_period,
            'fast_min_slow_period': fast_min_slow_period,
            'fast_max_fast_period': fast_max_fast_period,
            'fast_min_fast_period': fast_min_fast_period,
            'fast_hyper_smooth_period': fast_hyper_smooth_period,
            
            # 長期ZMA用パラメータ
            'slow_max_dc_cycle_part': slow_max_dc_cycle_part,
            'slow_max_dc_max_cycle': slow_max_dc_max_cycle,
            'slow_max_dc_min_cycle': slow_max_dc_min_cycle,
            'slow_max_dc_max_output': slow_max_dc_max_output,
            'slow_max_dc_min_output': slow_max_dc_min_output,
            
            'slow_min_dc_cycle_part': slow_min_dc_cycle_part,
            'slow_min_dc_max_cycle': slow_min_dc_max_cycle,
            'slow_min_dc_min_cycle': slow_min_dc_min_cycle,
            'slow_min_dc_max_output': slow_min_dc_max_output,
            'slow_min_dc_min_output': slow_min_dc_min_output,
            
            'slow_max_slow_period': slow_max_slow_period,
            'slow_min_slow_period': slow_min_slow_period,
            'slow_max_fast_period': slow_max_fast_period,
            'slow_min_fast_period': slow_min_fast_period,
            'slow_hyper_smooth_period': slow_hyper_smooth_period,
            
            'src_type': src_type
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZMACrossSignalGenerator(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            
            fast_max_dc_cycle_part=fast_max_dc_cycle_part,
            fast_max_dc_max_cycle=fast_max_dc_max_cycle,
            fast_max_dc_min_cycle=fast_max_dc_min_cycle,
            fast_max_dc_max_output=fast_max_dc_max_output,
            fast_max_dc_min_output=fast_max_dc_min_output,
            
            fast_min_dc_cycle_part=fast_min_dc_cycle_part,
            fast_min_dc_max_cycle=fast_min_dc_max_cycle,
            fast_min_dc_min_cycle=fast_min_dc_min_cycle,
            fast_min_dc_max_output=fast_min_dc_max_output,
            fast_min_dc_min_output=fast_min_dc_min_output,
            
            fast_max_slow_period=fast_max_slow_period,
            fast_min_slow_period=fast_min_slow_period,
            fast_max_fast_period=fast_max_fast_period,
            fast_min_fast_period=fast_min_fast_period,
            fast_hyper_smooth_period=fast_hyper_smooth_period,
            
            slow_max_dc_cycle_part=slow_max_dc_cycle_part,
            slow_max_dc_max_cycle=slow_max_dc_max_cycle,
            slow_max_dc_min_cycle=slow_max_dc_min_cycle,
            slow_max_dc_max_output=slow_max_dc_max_output,
            slow_max_dc_min_output=slow_max_dc_min_output,
            
            slow_min_dc_cycle_part=slow_min_dc_cycle_part,
            slow_min_dc_max_cycle=slow_min_dc_max_cycle,
            slow_min_dc_min_cycle=slow_min_dc_min_cycle,
            slow_min_dc_max_output=slow_min_dc_max_output,
            slow_min_dc_min_output=slow_min_dc_min_output,
            
            slow_max_slow_period=slow_max_slow_period,
            slow_min_slow_period=slow_min_slow_period,
            slow_max_fast_period=slow_max_fast_period,
            slow_min_fast_period=slow_min_fast_period,
            slow_hyper_smooth_period=slow_hyper_smooth_period,
            
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
            # シグナル生成器からシグナルを取得
            signals = self.signal_generator.get_entry_signals(data)
            
            # シグナルの検証と修正
            # np.int8型のシグナル配列を作成し、値は1（ロング）、-1（ショート）、0（ニュートラル）のみ
            length = len(signals)
            validated_signals = np.zeros(length, dtype=np.int8)
            
            for i in range(length):
                if signals[i] == 1:  # ゴールデンクロス
                    validated_signals[i] = 1  # ロングエントリー
                elif signals[i] == -1:  # デッドクロス
                    validated_signals[i] = -1  # ショートエントリー
            
            # デバッグログ
            if length > 0:
                self.logger.debug(f"ZMACrossStrategy - エントリーシグナル最終値: {validated_signals[-1]}")
                
            return validated_signals
            
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
            result = False
            
            # エグジット判定
            if position == 1:  # ロングポジション
                # デッドクロスでロングをエグジット
                result = self.signal_generator.get_exit_signals(data, position, index)
                self.logger.debug(f"ZMACrossStrategy - ロングエグジット判定: {result}")
            elif position == -1:  # ショートポジション
                # ゴールデンクロスでショートをエグジット
                result = self.signal_generator.get_exit_signals(data, position, index)
                self.logger.debug(f"ZMACrossStrategy - ショートエグジット判定: {result}")
            
            return result
            
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
            'hp_period': trial.suggest_int('hp_period', 89, 233),
            'lp_period': trial.suggest_int('lp_period', 5, 21),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # 短期ZMA用パラメータ
            'fast_max_dc_cycle_part': trial.suggest_float('fast_max_dc_cycle_part', 0.1, 0.5, step=0.05),
            'fast_max_dc_max_cycle': trial.suggest_int('fast_max_dc_max_cycle', 34, 89),
            'fast_max_dc_min_cycle': trial.suggest_int('fast_max_dc_min_cycle', 5, 21),
            'fast_max_dc_max_output': trial.suggest_int('fast_max_dc_max_output', 21, 55),
            'fast_max_dc_min_output': trial.suggest_int('fast_max_dc_min_output', 8, 21),
            
            # 長期ZMA用パラメータ
            'slow_max_dc_cycle_part': trial.suggest_float('slow_max_dc_cycle_part', 0.25, 0.75, step=0.05),
            'slow_max_dc_max_cycle': trial.suggest_int('slow_max_dc_max_cycle', 89, 233),
            'slow_max_dc_min_cycle': trial.suggest_int('slow_max_dc_min_cycle', 13, 55),
            'slow_max_dc_max_output': trial.suggest_int('slow_max_dc_max_output', 55, 144),
            'slow_max_dc_min_output': trial.suggest_int('slow_max_dc_min_output', 21, 55)
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
            'cycle_detector_type': 'hody_dc',
            'lp_period': int(params['lp_period']),
            'hp_period': int(params['hp_period']),
            'cycle_part': 0.5,
            'src_type': params['src_type'],
            
            # 短期ZMA用パラメータ
            'fast_max_dc_cycle_part': float(params['fast_max_dc_cycle_part']),
            'fast_max_dc_max_cycle': int(params['fast_max_dc_max_cycle']),
            'fast_max_dc_min_cycle': int(params['fast_max_dc_min_cycle']),
            'fast_max_dc_max_output': int(params['fast_max_dc_max_output']),
            'fast_max_dc_min_output': int(params['fast_max_dc_min_output']),
            
            'fast_min_dc_cycle_part': 0.25,
            'fast_min_dc_max_cycle': 21,
            'fast_min_dc_min_cycle': 5,
            'fast_min_dc_max_output': 8,
            'fast_min_dc_min_output': 3,
            
            'fast_max_slow_period': 21,
            'fast_min_slow_period': 8,
            'fast_max_fast_period': 5,
            'fast_min_fast_period': 2,
            'fast_hyper_smooth_period': 0,
            
            # 長期ZMA用パラメータ
            'slow_max_dc_cycle_part': float(params['slow_max_dc_cycle_part']),
            'slow_max_dc_max_cycle': int(params['slow_max_dc_max_cycle']),
            'slow_max_dc_min_cycle': int(params['slow_max_dc_min_cycle']),
            'slow_max_dc_max_output': int(params['slow_max_dc_max_output']),
            'slow_max_dc_min_output': int(params['slow_max_dc_min_output']),
            
            'slow_min_dc_cycle_part': 0.25,
            'slow_min_dc_max_cycle': 55,
            'slow_min_dc_min_cycle': 5,
            'slow_min_dc_max_output': 13,
            'slow_min_dc_min_output': 5,
            
            'slow_max_slow_period': 34,
            'slow_min_slow_period': 13,
            'slow_max_fast_period': 8,
            'slow_min_fast_period': 3,
            'slow_hyper_smooth_period': 0
        }
        
        return strategy_params 