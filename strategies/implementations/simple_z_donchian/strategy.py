#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import SimpleZDonchianSignalGenerator


class SimpleZDonchianStrategy(BaseStrategy):
    """
    シンプルなZドンチャン戦略 - ZTrendFilterなし
    
    特徴:
    - サイクル効率比（CER）に基づく動的パラメータ最適化
    - 動的なAVZモード
    - 動的なATR乗数
    - Zドンチャンブレイクアウトシグナルのみに基づいたシンプルなエントリー/エグジット条件
    
    エントリー条件:
    - ロング: 上側バンドブレイクアウト（価格が上側バンドを超える）
    - ショート: 下側バンドブレイクアウト（価格が下側バンドを下回る）
    
    エグジット条件:
    - ロング: 下側バンドブレイクアウト（価格が下側バンドを下回る）
    - ショート: 上側バンドブレイクアウト（価格が上側バンドを超える）
    """
    
    def __init__(
        self,
        # Zドンチャンチャネルのパラメータ
        donchian_length: int = 55,
        # ドミナントサイクル検出器用パラメータ
        cycle_detector_type: str = 'dudi_dc',
        cycle_part: float = 0.5,
        # CERと長期サイクル検出器用パラメータ
        max_cycle: int = 233,
        min_cycle: int = 13,
        max_output: int = 144,
        min_output: int = 21,
        # 短期サイクル検出器用パラメータ
        short_max_cycle: int = 55,
        short_min_cycle: int = 5,
        short_max_output: int = 34,
        short_min_output: int = 5,
        # AVZオプション
        enable_dynamic_avz: bool = True,
        avz_length: int = 5,
        short_length: int = 3,
        # ATR乗数オプション
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0,
        # 動的ATR乗数用パラメータ
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 3.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.3,    # 最小乗数の最小値
        # その他のパラメータ
        price_mode: str = 'close',
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            donchian_length: ドンチャンチャネルの長さ
            cycle_detector_type: サイクル検出器の種類
            cycle_part: サイクル部分
            max_cycle: 最大サイクル
            min_cycle: 最小サイクル
            max_output: 最大出力
            min_output: 最小出力
            short_max_cycle: 短期最大サイクル
            short_min_cycle: 短期最小サイクル
            short_max_output: 短期最大出力
            short_min_output: 短期最小出力
            enable_dynamic_avz: 動的AVZを有効にするかどうか
            avz_length: AVZ長さ
            short_length: 短期長さ
            max_multiplier: 最大乗数
            min_multiplier: 最小乗数
            max_max_multiplier: 最大乗数の最大値
            min_max_multiplier: 最大乗数の最小値
            max_min_multiplier: 最小乗数の最大値
            min_min_multiplier: 最小乗数の最小値
            price_mode: 価格モード
            src_type: ソースタイプ
        """
        super().__init__("SimpleZDonchian")
        
        # パラメータの設定
        self._parameters = {
            'donchian_length': donchian_length,
            'cycle_detector_type': cycle_detector_type,
            'cycle_part': cycle_part,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            'short_max_cycle': short_max_cycle,
            'short_min_cycle': short_min_cycle,
            'short_max_output': short_max_output,
            'short_min_output': short_min_output,
            'enable_dynamic_avz': enable_dynamic_avz,
            'avz_length': avz_length,
            'short_length': short_length,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            'price_mode': price_mode,
            'src_type': src_type
        }
        
        # シグナル生成器の初期化
        self.signal_generator = SimpleZDonchianSignalGenerator(
            donchian_length=donchian_length,
            cycle_detector_type=cycle_detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            short_max_cycle=short_max_cycle,
            short_min_cycle=short_min_cycle,
            short_max_output=short_max_output,
            short_min_output=short_min_output,
            enable_dynamic_avz=enable_dynamic_avz,
            avz_length=avz_length,
            short_length=short_length,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            price_mode=price_mode,
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
            'donchian_length': trial.suggest_int('donchian_length', 21, 89),
            'cycle_detector_type': trial.suggest_categorical('cycle_detector_type', ['dudi_dc', 'hody_dc', 'phac_dc']),
            'cycle_part': trial.suggest_float('cycle_part', 0.3, 0.8, step=0.1),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # CERと長期サイクル検出器用パラメータ
            'max_cycle': trial.suggest_int('max_cycle', 100, 300),
            'min_cycle': trial.suggest_int('min_cycle', 5, 20),
            'max_output': trial.suggest_int('max_output', 80, 200),
            'min_output': trial.suggest_int('min_output', 10, 30),
            
            # 短期サイクル検出器用パラメータ
            'short_max_cycle': trial.suggest_int('short_max_cycle', 30, 80),
            'short_min_cycle': trial.suggest_int('short_min_cycle', 3, 8),
            'short_max_output': trial.suggest_int('short_max_output', 20, 55),
            'short_min_output': trial.suggest_int('short_min_output', 3, 8),
            
            # AVZオプション
            'enable_dynamic_avz': trial.suggest_categorical('enable_dynamic_avz', [True, False]),
            'avz_length': trial.suggest_int('avz_length', 3, 10),
            'short_length': trial.suggest_int('short_length', 2, 5),
            
            # ATR乗数オプション
            'max_multiplier': trial.suggest_float('max_multiplier', 2.0, 4.0, step=0.1),
            'min_multiplier': trial.suggest_float('min_multiplier', 0.8, 2.0, step=0.1),
            'max_max_multiplier': trial.suggest_float('max_max_multiplier', 6.0, 10.0, step=0.5),
            'min_max_multiplier': trial.suggest_float('min_max_multiplier', 2.0, 4.0, step=0.1),
            'max_min_multiplier': trial.suggest_float('max_min_multiplier', 1.0, 2.0, step=0.1),
            'min_min_multiplier': trial.suggest_float('min_min_multiplier', 0.2, 0.6, step=0.1),
            
            # その他のパラメータ
            'price_mode': trial.suggest_categorical('price_mode', ['close', 'high_low'])
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
            'donchian_length': int(params['donchian_length']),
            'cycle_detector_type': params['cycle_detector_type'],
            'cycle_part': float(params['cycle_part']),
            'src_type': params.get('src_type', 'hlc3'),
            'max_cycle': int(params['max_cycle']),
            'min_cycle': int(params['min_cycle']),
            'max_output': int(params['max_output']),
            'min_output': int(params['min_output']),
            'short_max_cycle': int(params['short_max_cycle']),
            'short_min_cycle': int(params['short_min_cycle']),
            'short_max_output': int(params['short_max_output']),
            'short_min_output': int(params['short_min_output']),
            'enable_dynamic_avz': bool(params['enable_dynamic_avz']),
            'avz_length': int(params['avz_length']),
            'short_length': int(params['short_length']),
            'max_multiplier': float(params['max_multiplier']),
            'min_multiplier': float(params['min_multiplier']),
            'max_max_multiplier': float(params['max_max_multiplier']),
            'min_max_multiplier': float(params['min_max_multiplier']),
            'max_min_multiplier': float(params['max_min_multiplier']),
            'min_min_multiplier': float(params['min_min_multiplier']),
            'price_mode': params['price_mode']
        }
        return strategy_params 