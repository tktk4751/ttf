#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import XMAMASignalGenerator


class XMAMAStrategy(BaseStrategy):
    """
    X_MAMAストラテジー
    
    特徴:
    - X_MAMA (eXtended Mother of Adaptive Moving Average) / X_FAMA (eXtended Following Adaptive Moving Average)
    - 市場のサイクルに応じて自動的に期間を調整する適応型移動平均線
    - カルマンフィルターとゼロラグ処理統合版
    - Ehlers's MESA (Maximum Entropy Spectrum Analysis) アルゴリズムベース
    - トレンド強度に応じて応答速度を調整
    
    エントリー条件:
    - クロスオーバーモード: ゴールデンクロス（X_MAMA > X_FAMAに転換）でロング、デッドクロス（X_MAMA < X_FAMAに転換）でショート
    - 位置関係モード: X_MAMA > X_FAMAでロング、X_MAMA < X_FAMAでショート
    
    エグジット条件:
    - ロング: X_MAMA < X_FAMAに転換
    - ショート: X_MAMA > X_FAMAに転換
    """
    
    def __init__(
        self,
        # X_MAMAパラメータ
        fast_limit: float = 0.3,               # 高速制限値
        slow_limit: float = 0.01,              # 低速制限値
        src_type: str = 'oc2',                # ソースタイプ
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,       # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,    # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True,             # ゼロラグ処理を使用するか
        # シグナル設定
        position_mode: bool = True            # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
    ):
        """
        初期化
        
        Args:
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        """
        signal_type = "Position" if position_mode else "Crossover"
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(f"X_MAMA{signal_type}{kalman_str}{zero_lag_str}")
        
        # パラメータの設定
        self._parameters = {
            'fast_limit': fast_limit,
            'slow_limit': slow_limit,
            'src_type': src_type,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'position_mode': position_mode
        }
        
        # シグナル生成器の初期化
        self.signal_generator = XMAMASignalGenerator(
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag,
            position_mode=position_mode
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
    
    def get_mama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_MAMA値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: X_MAMA値
        """
        try:
            return self.signal_generator.get_mama_values(data)
        except Exception as e:
            self.logger.error(f"X_MAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_FAMA値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: X_FAMA値
        """
        try:
            return self.signal_generator.get_fama_values(data)
        except Exception as e:
            self.logger.error(f"X_FAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        全ての高度なメトリクスを取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Dict[str, np.ndarray]: 全メトリクス
        """
        try:
            return self.signal_generator.get_advanced_metrics(data)
        except Exception as e:
            self.logger.error(f"高度なメトリクス取得中にエラー: {str(e)}")
            return {}
    
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
            # X_MAMAパラメータ
            'fast_limit': trial.suggest_float('fast_limit', 0.1, 0.9, step=0.05),
            'slow_limit': trial.suggest_float('slow_limit', 0.01, 0.1, step=0.005),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
            
            # カルマンフィルターパラメータ
            'use_kalman_filter': trial.suggest_categorical('use_kalman_filter', [True, False]),
            'kalman_filter_type': trial.suggest_categorical('kalman_filter_type', ['adaptive', 'quantum_adaptive', 'simple', 'unscented', 'unscented_v2']),

            # ゼロラグ処理パラメータ
            'use_zero_lag': trial.suggest_categorical('use_zero_lag', [True, False]),
            
            # シグナル設定
            # 'position_mode': trial.suggest_categorical('position_mode', [True, False])
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
            'fast_limit': float(params['fast_limit']),
            'slow_limit': float(params['slow_limit']),
            'src_type': params['src_type'],
            'use_kalman_filter': bool(params['use_kalman_filter']),
            'kalman_filter_type': params['kalman_filter_type'],

            'use_zero_lag': bool(params['use_zero_lag']),
            # 'position_mode': bool(params['position_mode'])
        }
        return strategy_params
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        ストラテジー情報を取得
        
        Returns:
            Dict[str, Any]: ストラテジー情報
        """
        return {
            'name': 'X_MAMA Strategy',
            'description': 'eXtended Mother of Adaptive Moving Average with Kalman Filter and Zero-Lag Processing',
            'parameters': self._parameters.copy(),
            'features': [
                'Adaptive Moving Average based on market cycles',
                'Kalman Filter integration for noise reduction',
                'Zero-lag processing for faster response',
                'Configurable crossover or position-based signals',
                'Optimized with Numba JIT compilation'
            ]
        }
    
    def reset(self) -> None:
        """
        ストラテジーの状態をリセット
        """
        super().reset()
        if hasattr(self.signal_generator, 'reset'):
            self.signal_generator.reset()