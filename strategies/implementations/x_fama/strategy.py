#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import XFAMASignalGenerator


class XFAMAStrategy(BaseStrategy):
    """
    X_FAMAストラテジー
    
    特徴:
    - X_FAMA (eXtended Fractal Adaptive Moving Average) + Fast X_FAMA
    - フラクタル次元に基づく適応型移動平均線
    - カルマンフィルターとゼロラグ処理統合版
    - 通常線と高速線の2本による精密なシグナル生成
    - フラクタル次元とアルファ値によるトレンド強度判定
    
    エントリー条件:
    - クロスオーバーモード: ゴールデンクロス（Fast X_FAMA > X_FAMAに転換）でロング、デッドクロス（Fast X_FAMA < X_FAMAに転換）でショート
    - 位置関係モード: Fast X_FAMA > X_FAMAでロング、Fast X_FAMA < X_FAMAでショート
    - フラクタルモード: フラクタル次元とアルファ値によるトレンド強度でフィルタリング
    
    エグジット条件:
    - ロング: Fast X_FAMA < X_FAMAに転換
    - ショート: Fast X_FAMA > X_FAMAに転換
    """
    
    def __init__(
        self,
        # X_FAMAパラメータ
        period: int = 16,                      # 期間（偶数である必要がある）
        src_type: str = 'hl2',                 # ソースタイプ
        fc: int = 1,                           # Fast Constant
        sc: int = 198,                         # Slow Constant
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,       # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,    # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True,             # ゼロラグ処理を使用するか
        # シグナル設定
        position_mode: bool = True,            # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        fractal_mode: bool = False,            # フラクタル次元ベースのシグナル追加
        trend_threshold: float = 1.5,          # フラクタル次元のトレンド閾値
        alpha_threshold: float = 0.5           # アルファ値の閾値
    ):
        """
        初期化
        
        Args:
            period: 期間（偶数である必要がある、デフォルト: 16）
            src_type: ソースタイプ（デフォルト: 'hl2'）
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
            fc: Fast Constant（デフォルト: 1）
            sc: Slow Constant（デフォルト: 198）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
            fractal_mode: フラクタル次元ベースのシグナル追加
            trend_threshold: フラクタル次元のトレンド閾値
            alpha_threshold: アルファ値の閾値
        """
        signal_type = "Position" if position_mode else "Crossover"
        fractal_str = "_fractal" if fractal_mode else ""
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(f"X_FAMA{signal_type}{fractal_str}{kalman_str}{zero_lag_str}")
        
        # パラメータの設定
        self._parameters = {
            'period': period,
            'src_type': src_type,
            'fc': fc,
            'sc': sc,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'position_mode': position_mode,
            'fractal_mode': fractal_mode,
            'trend_threshold': trend_threshold,
            'alpha_threshold': alpha_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = XFAMASignalGenerator(
            period=period,
            src_type=src_type,
            fc=fc,
            sc=sc,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag,
            position_mode=position_mode,
            fractal_mode=fractal_mode,
            trend_threshold=trend_threshold,
            alpha_threshold=alpha_threshold
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
    
    def get_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_FRAMA値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: X_FRAMA値
        """
        try:
            return self.signal_generator.get_frama_values(data)
        except Exception as e:
            self.logger.error(f"X_FRAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fast_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Fast X_FRAMA値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: Fast X_FRAMA値
        """
        try:
            return self.signal_generator.get_fast_fama_values(data)
        except Exception as e:
            self.logger.error(f"Fast X_FRAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fractal_dimension(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        フラクタル次元を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: フラクタル次元
        """
        try:
            return self.signal_generator.get_fractal_dimension(data)
        except Exception as e:
            self.logger.error(f"フラクタル次元取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルファ値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: アルファ値
        """
        try:
            return self.signal_generator.get_alpha_values(data)
        except Exception as e:
            self.logger.error(f"アルファ値取得中にエラー: {str(e)}")
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
            # X_FAMAパラメータ
            'period': trial.suggest_int('period', 8, 32, step=2),  # 偶数のみ
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
            'fc': trial.suggest_int('fc', 1, 5),
            'sc': trial.suggest_int('sc', 50, 300, step=10),
            
            # カルマンフィルターパラメータ
            'use_kalman_filter': trial.suggest_categorical('use_kalman_filter', [True, False]),
            'kalman_filter_type': trial.suggest_categorical('kalman_filter_type', ['adaptive', 'quantum_adaptive', 'simple', 'unscented', 'unscented_v2']),
            'kalman_process_noise': trial.suggest_float('kalman_process_noise', 0.001, 0.1, log=True),
            'kalman_observation_noise': trial.suggest_float('kalman_observation_noise', 0.0001, 0.01, log=True),

            # ゼロラグ処理パラメータ
            'use_zero_lag': trial.suggest_categorical('use_zero_lag', [True, False]),
            
            # シグナル設定
            'position_mode': trial.suggest_categorical('position_mode', [True, False]),
            'fractal_mode': trial.suggest_categorical('fractal_mode', [True, False]),
            'trend_threshold': trial.suggest_float('trend_threshold', 1.2, 1.8, step=0.1),
            'alpha_threshold': trial.suggest_float('alpha_threshold', 0.2, 0.8, step=0.1)
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
            'period': int(params['period']),
            'src_type': params['src_type'],
            'fc': int(params['fc']),
            'sc': int(params['sc']),
            'use_kalman_filter': bool(params['use_kalman_filter']),
            'kalman_filter_type': params['kalman_filter_type'],
            'kalman_process_noise': float(params['kalman_process_noise']),
            'kalman_observation_noise': float(params['kalman_observation_noise']),
            'use_zero_lag': bool(params['use_zero_lag']),
            'position_mode': bool(params['position_mode']),
            'fractal_mode': bool(params['fractal_mode']),
            'trend_threshold': float(params['trend_threshold']),
            'alpha_threshold': float(params['alpha_threshold'])
        }
        return strategy_params
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        ストラテジー情報を取得
        
        Returns:
            Dict[str, Any]: ストラテジー情報
        """
        return {
            'name': 'X_FAMA Strategy',
            'description': 'eXtended Fractal Adaptive Moving Average with Kalman Filter and Zero-Lag Processing',
            'parameters': self._parameters.copy(),
            'features': [
                'Fractal dimension-based adaptive moving average',
                'Dual-line system (FRAMA + Fast FRAMA)',
                'Kalman Filter integration for noise reduction',
                'Zero-lag processing for faster response',
                'Fractal dimension and alpha value trend filtering',
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