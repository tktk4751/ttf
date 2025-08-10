#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import MESAFRAMASignalGenerator


class MESAFRAMAStrategy(BaseStrategy):
    """
    MESA_FRAMAデュアルクロスオーバーストラテジー
    
    特徴:
    - 2本のMESA_FRAMA (短期線・長期線のクロスオーバー戦略)
    - MESA適応期間を使用したFractal Adaptive Moving Average
    - MAMAの期間決定アルゴリズムとFRAMAのフラクタル適応性を組み合わせ
    - 市場サイクルに応じた動的期間調整
    - フラクタル次元ベースの応答性制御
    - カルマンフィルターとゼロラグ処理の統合（オプション）
    
    エントリー条件:
    1. クロスオーバーモード:
       - ロング: 短期MESA_FRAMAが長期MESA_FRAMAを上抜け
       - ショート: 短期MESA_FRAMAが長期MESA_FRAMAを下抜け
    2. 位置関係モード:
       - ロング: 短期MESA_FRAMA > 長期MESA_FRAMA
       - ショート: 短期MESA_FRAMA < 長期MESA_FRAMA
    
    エグジット条件:
    - ロング: 短期MESA_FRAMAが長期MESA_FRAMAを下抜け
    - ショート: 短期MESA_FRAMAが長期MESA_FRAMAを上抜け
    """
    
    def __init__(
        self,
        # 短期MESA_FRAMAパラメータ
        fast_base_period: int = 12,               # 短期基本期間
        fast_src_type: str = 'oc2',              # 短期ソースタイプ
        fast_fc: int = 1,                        # 短期Fast Constant
        fast_sc: int = 198,                      # 短期Slow Constant
        fast_mesa_fast_limit: float = 0.5,      # 短期MESA高速制限値
        fast_mesa_slow_limit: float = 0.05,      # 短期MESA低速制限値
        # 長期MESA_FRAMAパラメータ
        slow_base_period: int = 32,              # 長期基本期間
        slow_src_type: str = 'oc2',              # 長期ソースタイプ
        slow_fc: int = 1,                        # 長期Fast Constant
        slow_sc: int = 198,                      # 長期Slow Constant
        slow_mesa_fast_limit: float = 0.3,      # 長期MESA高速制限値
        slow_mesa_slow_limit: float = 0.01,     # 長期MESA低速制限値
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,         # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented',   # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,      # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True,               # ゼロラグ処理を使用するか
        # シグナルパラメータ
        signal_mode: str = 'position',          # シグナルモード ('crossover' または 'position')
        crossover_threshold: float = 0.0        # クロスオーバー閾値
    ):
        """
        初期化
        
        Args:
            fast_base_period: 短期基本期間（偶数である必要がある、デフォルト: 8）
            fast_src_type: 短期ソースタイプ（デフォルト: 'hl2'）
            fast_fc: 短期Fast Constant（デフォルト: 1）
            fast_sc: 短期Slow Constant（デフォルト: 198）
            fast_mesa_fast_limit: 短期MESA高速制限値（デフォルト: 0.7）
            fast_mesa_slow_limit: 短期MESA低速制限値（デフォルト: 0.1）
            slow_base_period: 長期基本期間（偶数である必要がある、デフォルト: 32）
            slow_src_type: 長期ソースタイプ（デフォルト: 'hl2'）
            slow_fc: 長期Fast Constant（デフォルト: 1）
            slow_sc: 長期Slow Constant（デフォルト: 198）
            slow_mesa_fast_limit: 長期MESA高速制限値（デフォルト: 0.3）
            slow_mesa_slow_limit: 長期MESA低速制限値（デフォルト: 0.02）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
            signal_mode: シグナルモード（デフォルト: 'crossover'）
            crossover_threshold: クロスオーバー閾値（デフォルト: 0.0）
        """
        super().__init__("MESA_FRAMA_DUAL")
        
        # パラメータの設定
        self._parameters = {
            'fast_base_period': fast_base_period,
            'fast_src_type': fast_src_type,
            'fast_fc': fast_fc,
            'fast_sc': fast_sc,
            'fast_mesa_fast_limit': fast_mesa_fast_limit,
            'fast_mesa_slow_limit': fast_mesa_slow_limit,
            'slow_base_period': slow_base_period,
            'slow_src_type': slow_src_type,
            'slow_fc': slow_fc,
            'slow_sc': slow_sc,
            'slow_mesa_fast_limit': slow_mesa_fast_limit,
            'slow_mesa_slow_limit': slow_mesa_slow_limit,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'signal_mode': signal_mode,
            'crossover_threshold': crossover_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = MESAFRAMASignalGenerator(
            fast_base_period=fast_base_period,
            fast_src_type=fast_src_type,
            fast_fc=fast_fc,
            fast_sc=fast_sc,
            fast_mesa_fast_limit=fast_mesa_fast_limit,
            fast_mesa_slow_limit=fast_mesa_slow_limit,
            slow_base_period=slow_base_period,
            slow_src_type=slow_src_type,
            slow_fc=slow_fc,
            slow_sc=slow_sc,
            slow_mesa_fast_limit=slow_mesa_fast_limit,
            slow_mesa_slow_limit=slow_mesa_slow_limit,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag,
            signal_mode=signal_mode,
            crossover_threshold=crossover_threshold
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
            # 短期MESA_FRAMAパラメータ
            'fast_base_period': trial.suggest_int('fast_base_period', 4, 16, step=2),  # 偶数のみ
            'fast_fc': trial.suggest_int('fast_fc', 1, 5),
            'fast_sc': trial.suggest_int('fast_sc', 100, 300, step=25),
            'fast_src_type': trial.suggest_categorical('fast_src_type', ['hl2', 'hlc3', 'close']),
            'fast_mesa_fast_limit': trial.suggest_float('fast_mesa_fast_limit', 0.5, 0.9, step=0.1),
            'fast_mesa_slow_limit': trial.suggest_float('fast_mesa_slow_limit', 0.05, 0.2, step=0.05),
            
            # 長期MESA_FRAMAパラメータ
            'slow_base_period': trial.suggest_int('slow_base_period', 20, 48, step=2),  # 偶数のみ
            'slow_fc': trial.suggest_int('slow_fc', 1, 5),
            'slow_sc': trial.suggest_int('slow_sc', 150, 350, step=25),
            'slow_src_type': trial.suggest_categorical('slow_src_type', ['hl2', 'hlc3', 'close']),
            'slow_mesa_fast_limit': trial.suggest_float('slow_mesa_fast_limit', 0.1, 0.5, step=0.1),
            'slow_mesa_slow_limit': trial.suggest_float('slow_mesa_slow_limit', 0.01, 0.1, step=0.01),
            
            # シグナルパラメータ
            'signal_mode': trial.suggest_categorical('signal_mode', ['crossover', 'position']),
            'crossover_threshold': trial.suggest_float('crossover_threshold', 0.0, 0.02, step=0.005),
            
            # オプション機能（基本的には無効で最適化）
            'use_zero_lag': trial.suggest_categorical('use_zero_lag', [True, False]),
            'use_kalman_filter': trial.suggest_categorical('use_kalman_filter', [False])  # 基本は無効
        }
        
        # カルマンフィルターが有効な場合のパラメータ
        if params['use_kalman_filter']:
            params.update({
                'kalman_filter_type': trial.suggest_categorical('kalman_filter_type', ['unscented', 'extended']),
                'kalman_process_noise': trial.suggest_float('kalman_process_noise', 0.001, 0.1, log=True),
                'kalman_observation_noise': trial.suggest_float('kalman_observation_noise', 0.0001, 0.01, log=True)
            })
        
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
            # 短期MESA_FRAMAパラメータ
            'fast_base_period': int(params['fast_base_period']),
            'fast_src_type': params['fast_src_type'],
            'fast_fc': int(params['fast_fc']),
            'fast_sc': int(params['fast_sc']),
            'fast_mesa_fast_limit': float(params['fast_mesa_fast_limit']),
            'fast_mesa_slow_limit': float(params['fast_mesa_slow_limit']),
            
            # 長期MESA_FRAMAパラメータ
            'slow_base_period': int(params['slow_base_period']),
            'slow_src_type': params['slow_src_type'],
            'slow_fc': int(params['slow_fc']),
            'slow_sc': int(params['slow_sc']),
            'slow_mesa_fast_limit': float(params['slow_mesa_fast_limit']),
            'slow_mesa_slow_limit': float(params['slow_mesa_slow_limit']),
            
            # シグナルパラメータ
            'signal_mode': params['signal_mode'],
            'crossover_threshold': float(params['crossover_threshold']),
            'use_zero_lag': bool(params['use_zero_lag']),
            'use_kalman_filter': bool(params.get('use_kalman_filter', False))
        }
        
        # カルマンフィルター関連パラメータ
        if strategy_params['use_kalman_filter']:
            strategy_params.update({
                'kalman_filter_type': params.get('kalman_filter_type', 'unscented'),
                'kalman_process_noise': float(params.get('kalman_process_noise', 0.01)),
                'kalman_observation_noise': float(params.get('kalman_observation_noise', 0.001))
            })
        
        return strategy_params
    
    def get_mesa_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        両方のMESA_FRAMA値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            tuple: (短期MESA_FRAMA値, 長期MESA_FRAMA値)
        """
        return self.signal_generator.get_mesa_frama_values(data)
    
    def get_fractal_dimension(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        フラクタル次元を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            tuple: (短期フラクタル次元, 長期フラクタル次元)
        """
        return self.signal_generator.get_fractal_dimension(data)
    
    def get_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        MESA動的期間を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            tuple: (短期MESA動的期間, 長期MESA動的期間)
        """
        return self.signal_generator.get_dynamic_periods(data)
    
    def get_mesa_phase(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        MESA位相を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            tuple: (短期MESA位相, 長期MESA位相)
        """
        return self.signal_generator.get_mesa_phase(data)
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        フラクタルアルファ値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            tuple: (短期フラクタルアルファ値, 長期フラクタルアルファ値)
        """
        return self.signal_generator.get_alpha_values(data)
    
    def get_filtered_price(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        カルマンフィルター後の価格を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            tuple: (短期フィルタリングされた価格, 長期フィルタリングされた価格)
        """
        return self.signal_generator.get_filtered_price(data)