#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZMACDBreakoutSignalGenerator


class ZMACDBreakoutStrategy(BaseStrategy):
    """
    ZMACDブレイクアウトストラテジー
    
    特徴:
    - ZMACDダイバージェンスシグナルをエントリーに使用
    - ZBBブレイクアウトシグナルをエグジットに使用
    - サイクル効率比（CER）に基づく動的なパラメータ調整
    
    エントリー条件:
    - ロング: ZMACDダイバージェンスシグナルが1のとき
    - ショート: ZMACDダイバージェンスシグナルが-1のとき
    
    エグジット条件:
    - ロング決済: ZBBブレイクアウトシグナルが-1のとき
    - ショート決済: ZBBブレイクアウトシグナルが1のとき
    """
    
    def __init__(
        self,
        # 共通パラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        src_type: str = 'hlc3',
        
        # ZMACDダイバージェンス用パラメータ
        er_period: int = 21,
        # 短期線用パラメータ
        fast_max_dc_max_output: int = 21,
        fast_max_dc_min_output: int = 5,
        # 長期線用パラメータ
        slow_max_dc_max_output: int = 55,
        slow_max_dc_min_output: int = 13,
        # シグナル線用パラメータ
        signal_max_dc_max_output: int = 21,
        signal_max_dc_min_output: int = 5,
        max_slow_period: int = 34,
        min_slow_period: int = 13,
        max_fast_period: int = 8,
        min_fast_period: int = 2,
        lookback: int = 60,
        
        # ZBB用パラメータ
        bb_max_multiplier: float = 2.5,
        bb_min_multiplier: float = 1.0,
        
        # ZBB標準偏差最大期間用パラメータ
        bb_max_cycle_part: float = 0.5,
        bb_max_max_cycle: int = 144,
        bb_max_min_cycle: int = 10,
        bb_max_max_output: int = 89,
        bb_max_min_output: int = 13,
        
        # ZBB標準偏差最小期間用パラメータ
        bb_min_cycle_part: float = 0.25,
        bb_min_max_cycle: int = 55,
        bb_min_min_cycle: int = 5,
        bb_min_max_output: int = 21,
        bb_min_min_output: int = 5,
        
        # ブレイクアウトパラメータ
        bb_lookback: int = 1
    ):
        """初期化"""
        super().__init__("ZMACDBreakout")
        
        # パラメータの設定
        self._parameters = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'src_type': src_type,
            
            'er_period': er_period,
            'fast_max_dc_max_output': fast_max_dc_max_output,
            'fast_max_dc_min_output': fast_max_dc_min_output,
            'slow_max_dc_max_output': slow_max_dc_max_output,
            'slow_max_dc_min_output': slow_max_dc_min_output,
            'signal_max_dc_max_output': signal_max_dc_max_output,
            'signal_max_dc_min_output': signal_max_dc_min_output,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period,
            'lookback': lookback,
            
            'bb_max_multiplier': bb_max_multiplier,
            'bb_min_multiplier': bb_min_multiplier,
            'bb_max_cycle_part': bb_max_cycle_part,
            'bb_max_max_cycle': bb_max_max_cycle,
            'bb_max_min_cycle': bb_max_min_cycle,
            'bb_max_max_output': bb_max_max_output,
            'bb_max_min_output': bb_max_min_output,
            'bb_min_cycle_part': bb_min_cycle_part,
            'bb_min_max_cycle': bb_min_max_cycle,
            'bb_min_min_cycle': bb_min_min_cycle,
            'bb_min_max_output': bb_min_max_output,
            'bb_min_min_output': bb_min_min_output,
            'bb_lookback': bb_lookback
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZMACDBreakoutSignalGenerator(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            src_type=src_type,
            
            er_period=er_period,
            fast_max_dc_max_output=fast_max_dc_max_output,
            fast_max_dc_min_output=fast_max_dc_min_output,
            slow_max_dc_max_output=slow_max_dc_max_output,
            slow_max_dc_min_output=slow_max_dc_min_output,
            signal_max_dc_max_output=signal_max_dc_max_output,
            signal_max_dc_min_output=signal_max_dc_min_output,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period,
            lookback=lookback,
            
            bb_max_multiplier=bb_max_multiplier,
            bb_min_multiplier=bb_min_multiplier,
            bb_max_cycle_part=bb_max_cycle_part,
            bb_max_max_cycle=bb_max_max_cycle,
            bb_max_min_cycle=bb_max_min_cycle,
            bb_max_max_output=bb_max_max_output,
            bb_max_min_output=bb_max_min_output,
            bb_min_cycle_part=bb_min_cycle_part,
            bb_min_max_cycle=bb_min_max_cycle,
            bb_min_min_cycle=bb_min_min_cycle,
            bb_min_max_output=bb_min_max_output,
            bb_min_min_output=bb_min_min_output,
            bb_lookback=bb_lookback
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
            
            # ZMACDダイバージェンス用パラメータ
            'er_period': trial.suggest_int('er_period', 10, 34),
            
            # 短期線用パラメータ
            'fast_max_dc_max_output': trial.suggest_int('fast_max_dc_max_output', 13, 34),
            'fast_max_dc_min_output': trial.suggest_int('fast_max_dc_min_output', 2, 8),
            
            # 長期線用パラメータ
            'slow_max_dc_max_output': trial.suggest_int('slow_max_dc_max_output', 34, 89),
            'slow_max_dc_min_output': trial.suggest_int('slow_max_dc_min_output', 8, 21),
            
            # シグナル線用パラメータ
            'signal_max_dc_max_output': trial.suggest_int('signal_max_dc_max_output', 13, 34),
            'signal_max_dc_min_output': trial.suggest_int('signal_max_dc_min_output', 2, 8),
            
            'max_slow_period': trial.suggest_int('max_slow_period', 21, 55),
            'min_slow_period': trial.suggest_int('min_slow_period', 8, 21),
            'max_fast_period': trial.suggest_int('max_fast_period', 5, 13),
            'min_fast_period': trial.suggest_int('min_fast_period', 2, 5),
            'lookback': trial.suggest_int('lookback', 20, 60),
            
            # ZBB用パラメータ
            'bb_max_multiplier': trial.suggest_float('bb_max_multiplier', 2.0, 3.5, step=0.1),
            'bb_min_multiplier': trial.suggest_float('bb_min_multiplier', 0.5, 2.0, step=0.1),
            
            # ZBB標準偏差最大期間用パラメータ
            'bb_max_cycle_part': trial.suggest_float('bb_max_cycle_part', 0.3, 0.7, step=0.1),
            'bb_max_max_cycle': trial.suggest_int('bb_max_max_cycle', 89, 233),
            'bb_max_min_cycle': trial.suggest_int('bb_max_min_cycle', 5, 21),
            'bb_max_max_output': trial.suggest_int('bb_max_max_output', 55, 144),
            'bb_max_min_output': trial.suggest_int('bb_max_min_output', 8, 21),
            
            # ZBB標準偏差最小期間用パラメータ
            'bb_min_cycle_part': trial.suggest_float('bb_min_cycle_part', 0.15, 0.5, step=0.05),
            'bb_min_max_cycle': trial.suggest_int('bb_min_max_cycle', 34, 89),
            'bb_min_min_cycle': trial.suggest_int('bb_min_min_cycle', 3, 8),
            'bb_min_max_output': trial.suggest_int('bb_min_max_output', 13, 34),
            'bb_min_min_output': trial.suggest_int('bb_min_min_output', 3, 13),
            
            # ブレイクアウトパラメータ
            'bb_lookback': trial.suggest_int('bb_lookback', 1, 3)
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
            # 共通パラメータ
            'cycle_detector_type': params['cycle_detector_type'],
            'lp_period': int(params['lp_period']),
            'hp_period': int(params['hp_period']),
            'cycle_part': float(params['cycle_part']),
            'src_type': params['src_type'],
            
            # ZMACDダイバージェンス用パラメータ
            'er_period': int(params['er_period']),
            
            # 短期線用パラメータ
            'fast_max_dc_max_output': int(params['fast_max_dc_max_output']),
            'fast_max_dc_min_output': int(params['fast_max_dc_min_output']),
            
            # 長期線用パラメータ
            'slow_max_dc_max_output': int(params['slow_max_dc_max_output']),
            'slow_max_dc_min_output': int(params['slow_max_dc_min_output']),
            
            # シグナル線用パラメータ
            'signal_max_dc_max_output': int(params['signal_max_dc_max_output']),
            'signal_max_dc_min_output': int(params['signal_max_dc_min_output']),
            
            'max_slow_period': int(params['max_slow_period']),
            'min_slow_period': int(params['min_slow_period']),
            'max_fast_period': int(params['max_fast_period']),
            'min_fast_period': int(params['min_fast_period']),
            'lookback': int(params['lookback']),
            
            # ZBB用パラメータ
            'bb_max_multiplier': float(params['bb_max_multiplier']),
            'bb_min_multiplier': float(params['bb_min_multiplier']),
            
            # ZBB標準偏差最大期間用パラメータ
            'bb_max_cycle_part': float(params['bb_max_cycle_part']),
            'bb_max_max_cycle': int(params['bb_max_max_cycle']),
            'bb_max_min_cycle': int(params['bb_max_min_cycle']),
            'bb_max_max_output': int(params['bb_max_max_output']),
            'bb_max_min_output': int(params['bb_max_min_output']),
            
            # ZBB標準偏差最小期間用パラメータ
            'bb_min_cycle_part': float(params['bb_min_cycle_part']),
            'bb_min_max_cycle': int(params['bb_min_max_cycle']),
            'bb_min_min_cycle': int(params['bb_min_min_cycle']),
            'bb_min_max_output': int(params['bb_min_max_output']),
            'bb_min_min_output': int(params['bb_min_min_output']),
            
            # ブレイクアウトパラメータ
            'bb_lookback': int(params['bb_lookback'])
        }
        return strategy_params 