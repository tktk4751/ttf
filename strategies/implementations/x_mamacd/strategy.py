#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import XMAMACDSignalGenerator


class XMAMACDStrategy(BaseStrategy):
    """
    X_MAMACDストラテジー
    
    特徴:
    - X_MAMACD (X_MAMAベースMACD) を使用
    - MAMACD、シグナルライン、ヒストグラムの3つの要素を統合
    - 市場のサイクルに応じて自動的に期間を調整する適応型MACD
    - カルマンフィルターとゼロラグ処理統合版
    - 複数のシグナルモード（クロスオーバー、ゼロライン、トレンドフォロー）
    
    エントリー条件:
    - クロスオーバーモード: MAMACD > Signal に転換でロング、MAMACD < Signal に転換でショート
    - ゼロラインモード: MAMACD > 0 に転換でロング、MAMACD < 0 に転換でショート
    - トレンドフォローモード: 複合条件でトレンド継続シグナル
    
    エグジット条件:
    - ロング: 逆方向のシグナルが発生
    - ショート: 逆方向のシグナルが発生
    """
    
    def __init__(
        self,
        # X_MAMACDパラメータ
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.05,              # 低速制限値
        src_type: str = 'hlc3',                # ソースタイプ
        signal_period: int = 9,                # シグナルライン期間
        use_adaptive_signal: bool = True,      # 適応型シグナルラインを使用するか
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,       # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,    # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True,             # ゼロラグ処理を使用するか
        # シグナル設定
        signal_mode: str = 'crossover',        # シグナルモード ('crossover', 'zero_line', 'trend_follow')
        trend_threshold: float = 0.0,          # トレンドフォロー閾値
        momentum_mode: bool = False,           # モメンタムモードを使用するか
        momentum_lookback: int = 3             # モメンタム計算の振り返り期間
    ):
        """
        初期化
        
        Args:
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
            signal_period: シグナルライン期間（デフォルト: 9）
            use_adaptive_signal: 適応型シグナルラインを使用するか（デフォルト: True）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
            signal_mode: シグナルモード（デフォルト: 'crossover'）
            trend_threshold: トレンドフォロー閾値（デフォルト: 0.0）
            momentum_mode: モメンタムモードを使用するか（デフォルト: False）
            momentum_lookback: モメンタム計算の振り返り期間（デフォルト: 3）
        """
        adaptive_str = "_adaptive" if use_adaptive_signal else ""
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        momentum_str = f"_momentum({momentum_lookback})" if momentum_mode else ""
        
        super().__init__(f"X_MAMACD{signal_mode.title()}{adaptive_str}{kalman_str}{zero_lag_str}{momentum_str}")
        
        # パラメータの設定
        self._parameters = {
            'fast_limit': fast_limit,
            'slow_limit': slow_limit,
            'src_type': src_type,
            'signal_period': signal_period,
            'use_adaptive_signal': use_adaptive_signal,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'signal_mode': signal_mode,
            'trend_threshold': trend_threshold,
            'momentum_mode': momentum_mode,
            'momentum_lookback': momentum_lookback
        }
        
        # シグナル生成器の初期化
        self.signal_generator = XMAMACDSignalGenerator(
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            signal_period=signal_period,
            use_adaptive_signal=use_adaptive_signal,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag,
            signal_mode=signal_mode,
            trend_threshold=trend_threshold,
            momentum_mode=momentum_mode,
            momentum_lookback=momentum_lookback
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
    
    def get_mamacd_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        MAMACD値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: MAMACD値
        """
        try:
            return self.signal_generator.get_mamacd_values(data)
        except Exception as e:
            self.logger.error(f"MAMACD値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_signal_line_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        シグナルライン値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: シグナルライン値
        """
        try:
            return self.signal_generator.get_signal_line_values(data)
        except Exception as e:
            self.logger.error(f"シグナルライン値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_histogram_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ヒストグラム値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: ヒストグラム値
        """
        try:
            return self.signal_generator.get_histogram_values(data)
        except Exception as e:
            self.logger.error(f"ヒストグラム値取得中にエラー: {str(e)}")
            return np.array([])
    
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
    
    def get_all_mamacd_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        すべてのMAMACD関連値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Dict[str, np.ndarray]: MAMACD関連の全ての値
        """
        try:
            return {
                'mamacd': self.get_mamacd_values(data),
                'signal': self.get_signal_line_values(data),
                'histogram': self.get_histogram_values(data),
                'mama': self.get_mama_values(data),
                'fama': self.get_fama_values(data)
            }
        except Exception as e:
            self.logger.error(f"MAMACD全値取得中にエラー: {str(e)}")
            return {}
    
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
            # X_MAMACDパラメータ
            'fast_limit': trial.suggest_float('fast_limit', 0.1, 0.9, step=0.05),
            'slow_limit': trial.suggest_float('slow_limit', 0.01, 0.1, step=0.005),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
            'signal_period': trial.suggest_int('signal_period', 6, 15),
            'use_adaptive_signal': trial.suggest_categorical('use_adaptive_signal', [True, False]),
            
            # カルマンフィルターパラメータ
            'use_kalman_filter': trial.suggest_categorical('use_kalman_filter', [True, False]),
            'kalman_filter_type': trial.suggest_categorical('kalman_filter_type', ['adaptive', 'quantum_adaptive', 'simple', 'unscented', 'unscented_v2']),
            
            # ゼロラグ処理パラメータ
            'use_zero_lag': trial.suggest_categorical('use_zero_lag', [True, False]),
            
            # シグナル設定
            'signal_mode': trial.suggest_categorical('signal_mode', ['crossover', 'zero_line', 'trend_follow']),
            'trend_threshold': trial.suggest_float('trend_threshold', -0.5, 0.5, step=0.1),
            'momentum_mode': trial.suggest_categorical('momentum_mode', [True, False]),
            'momentum_lookback': trial.suggest_int('momentum_lookback', 2, 6)
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
            'signal_period': int(params['signal_period']),
            'use_adaptive_signal': bool(params['use_adaptive_signal']),
            'use_kalman_filter': bool(params['use_kalman_filter']),
            'kalman_filter_type': params['kalman_filter_type'],
            'use_zero_lag': bool(params['use_zero_lag']),
            'signal_mode': params['signal_mode'],
            'trend_threshold': float(params['trend_threshold']),
            'momentum_mode': bool(params['momentum_mode']),
            'momentum_lookback': int(params['momentum_lookback'])
        }
        return strategy_params
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        ストラテジー情報を取得
        
        Returns:
            Dict[str, Any]: ストラテジー情報
        """
        return {
            'name': 'X_MAMACD Strategy',
            'description': 'eXtended MAMA-based MACD with multiple signal modes and advanced features',
            'parameters': self._parameters.copy(),
            'features': [
                'Adaptive MACD based on X_MAMA/X_FAMA',
                'Multiple signal modes (crossover, zero-line, trend-follow)',
                'Adaptive signal line with X_MAMA alpha values',
                'Kalman Filter integration for noise reduction',
                'Zero-lag processing for faster response',
                'Momentum-based trend continuation detection',
                'Optimized with Numba JIT compilation'
            ],
            'signal_modes': {
                'crossover': 'MAMACD vs Signal line crossover signals',
                'zero_line': 'MAMACD zero line cross signals',
                'trend_follow': 'Complex trend following with momentum'
            }
        }
    
    def reset(self) -> None:
        """
        ストラテジーの状態をリセット
        """
        super().reset()
        if hasattr(self.signal_generator, 'reset'):
            self.signal_generator.reset()