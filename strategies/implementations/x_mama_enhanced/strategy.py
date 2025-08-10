#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import XMAMAEnhancedSignalGenerator, FilterType






class XMAMAEnhancedStrategy(BaseStrategy):
    """
    X_MAMA Enhanced ストラテジー
    
    特徴:
    - X_MAMA (eXtended Mother of Adaptive Moving Average) ベースの高度なトレードシステム
    - 5つの高度なフィルターから選択可能:
      1. Phasor Trend Filter - フェーザー分析による高精度トレンド判定
      2. Correlation Cycle Filter - 相関サイクル分析によるトレンド・サイクル判定
      3. Correlation Trend Filter - 相関トレンド分析による線形トレンド検出
      4. Unified Trend Cycle Filter - 3つのEhlersアルゴリズム統合による超高精度判定
      5. X-Choppiness Filter - STR基盤改良型チョピネス指標によるトレンド・レンジ判定
    - フィルターなしオプションも提供
    
    エントリー条件:
    - ロング: X_MAMAシグナル=1 かつ フィルターシグナル=1（フィルター有効時）
    - ショート: X_MAMAシグナル=-1 かつ フィルターシグナル=-1（フィルター有効時）
    - フィルターシグナル=0または逆方向の場合はスルー
    
    エグジット条件:
    - ロング: X_MAMAシグナル=-1
    - ショート: X_MAMAシグナル=1
    
    革新的な優位性:
    - 複数のEhlersアルゴリズムの統合による高精度判定
    - 適応的フィルタリングによる誤判定の大幅削減
    - 市場状態に応じた自動フィルター調整
    - Numba JIT最適化による高速処理
    """
    
    def __init__(
        self,
        # X_MAMAパラメータ
        fast_limit: float = 0.2,               # 高速制限値
        slow_limit: float = 0.02,              # 低速制限値
        src_type: str = 'oc2',                # ソースタイプ
        # カルマンフィルターパラメータ（X_MAMA用）
        use_kalman_filter: bool = True,       # カルマンフィルターを使用するか
        kalman_filter_type: str = 'quantum_adaptive', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,    # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ（X_MAMA用）
        use_zero_lag: bool = True,             # ゼロラグ処理を使用するか
        # シグナル設定（X_MAMA用）
        position_mode: bool = True,            # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        # フィルター選択
        filter_type: FilterType = FilterType.PHASOR_TREND,  # フィルタータイプ
        # Phasor Trend Filterパラメータ
        phasor_period: int = 34,               # フェーザー分析周期
        phasor_trend_threshold: float = 6.0,   # フェーザートレンド判定閾値
        phasor_use_kalman: bool = False,       # フェーザーフィルター用カルマンフィルター
        # Correlation Cycle Filterパラメータ
        correlation_cycle_period: int = 20,    # 相関サイクル計算期間
        correlation_cycle_threshold: float = 9.0, # サイクルトレンド判定閾値
        # Correlation Trend Filterパラメータ
        correlation_trend_length: int = 20,    # 相関トレンド計算期間
        correlation_trend_threshold: float = 0.3, # トレンド判定閾値
        correlation_trend_smoothing: bool = False, # 平滑化有効
        # Unified Trend Cycle Filterパラメータ
        unified_period: int = 55,              # 統合フィルター基本周期
        unified_trend_length: int = 55,        # 統合トレンド分析長
        unified_trend_threshold: float = 0.5,  # 統合トレンド判定閾値
        unified_adaptability: float = 0.7,     # 適応性係数
        unified_consensus: bool = True,         # コンセンサスフィルター有効
        # X-Choppiness Filterパラメータ
        x_choppiness_detector_type: str = 'hody_e',  # サイクル検出器タイプ
        x_choppiness_lp_period: int = 12,      # ローパスフィルター期間
        x_choppiness_hp_period: int = 124,     # ハイパスフィルター期間
        x_choppiness_cycle_part: float = 0.5,  # サイクル部分
        x_choppiness_max_cycle: int = 124,     # 最大サイクル期間
        x_choppiness_min_cycle: int = 12,      # 最小サイクル期間
        x_choppiness_max_output: int = 89,     # 最大出力値
        x_choppiness_min_output: int = 5       # 最小出力値
    ):
        """
        初期化
        
        Args:
            fast_limit: X_MAMA高速制限値（デフォルト: 0.3）
            slow_limit: X_MAMA低速制限値（デフォルト: 0.01）
            src_type: ソースタイプ（デフォルト: 'oc2'）
            use_kalman_filter: X_MAMA用カルマンフィルター使用（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理使用（デフォルト: True）
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
            filter_type: フィルタータイプ（デフォルト: FilterType.UNIFIED_TREND_CYCLE）
            その他: 各フィルターのパラメータ
        """
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        signal_type = "Position" if position_mode else "Crossover"
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        zero_lag_str = "_zero_lag" if use_zero_lag else ""
        
        super().__init__(f"X_MAMA_Enhanced_{signal_type}_{filter_name}{kalman_str}{zero_lag_str}")
        
        # パラメータの設定
        self._parameters = {
            # X_MAMAパラメータ
            'fast_limit': fast_limit,
            'slow_limit': slow_limit,
            'src_type': src_type,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'position_mode': position_mode,
            # フィルター設定
            'filter_type': filter_type,
            # Phasor Trend Filterパラメータ
            'phasor_period': phasor_period,
            'phasor_trend_threshold': phasor_trend_threshold,
            'phasor_use_kalman': phasor_use_kalman,
            # Correlation Cycle Filterパラメータ
            'correlation_cycle_period': correlation_cycle_period,
            'correlation_cycle_threshold': correlation_cycle_threshold,
            # Correlation Trend Filterパラメータ
            'correlation_trend_length': correlation_trend_length,
            'correlation_trend_threshold': correlation_trend_threshold,
            'correlation_trend_smoothing': correlation_trend_smoothing,
            # Unified Trend Cycle Filterパラメータ
            'unified_period': unified_period,
            'unified_trend_length': unified_trend_length,
            'unified_trend_threshold': unified_trend_threshold,
            'unified_adaptability': unified_adaptability,
            'unified_consensus': unified_consensus,
            # X-Choppiness Filterパラメータ
            'x_choppiness_detector_type': x_choppiness_detector_type,
            'x_choppiness_lp_period': x_choppiness_lp_period,
            'x_choppiness_hp_period': x_choppiness_hp_period,
            'x_choppiness_cycle_part': x_choppiness_cycle_part,
            'x_choppiness_max_cycle': x_choppiness_max_cycle,
            'x_choppiness_min_cycle': x_choppiness_min_cycle,
            'x_choppiness_max_output': x_choppiness_max_output,
            'x_choppiness_min_output': x_choppiness_min_output
        }
        
        # シグナル生成器の初期化
        self.signal_generator = XMAMAEnhancedSignalGenerator(
            # X_MAMAパラメータ
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag,
            position_mode=position_mode,
            # フィルター設定
            filter_type=filter_type,
            # Phasor Trend Filterパラメータ
            phasor_period=phasor_period,
            phasor_trend_threshold=phasor_trend_threshold,
            phasor_use_kalman=phasor_use_kalman,
            # Correlation Cycle Filterパラメータ
            correlation_cycle_period=correlation_cycle_period,
            correlation_cycle_threshold=correlation_cycle_threshold,
            # Correlation Trend Filterパラメータ
            correlation_trend_length=correlation_trend_length,
            correlation_trend_threshold=correlation_trend_threshold,
            correlation_trend_smoothing=correlation_trend_smoothing,
            # Unified Trend Cycle Filterパラメータ
            unified_period=unified_period,
            unified_trend_length=unified_trend_length,
            unified_trend_threshold=unified_trend_threshold,
            unified_adaptability=unified_adaptability,
            unified_consensus=unified_consensus,
            # X-Choppiness Filterパラメータ
            x_choppiness_detector_type=x_choppiness_detector_type,
            x_choppiness_lp_period=x_choppiness_lp_period,
            x_choppiness_hp_period=x_choppiness_hp_period,
            x_choppiness_cycle_part=x_choppiness_cycle_part,
            x_choppiness_max_cycle=x_choppiness_max_cycle,
            x_choppiness_min_cycle=x_choppiness_min_cycle,
            x_choppiness_max_output=x_choppiness_max_output,
            x_choppiness_min_output=x_choppiness_min_output
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル（ロング=1、ショート=-1、なし=0）
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
        """X_MAMA値を取得"""
        try:
            return self.signal_generator.get_mama_values(data)
        except Exception as e:
            self.logger.error(f"X_MAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """X_FAMA値を取得"""
        try:
            return self.signal_generator.get_fama_values(data)
        except Exception as e:
            self.logger.error(f"X_FAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_long_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ロングエントリーシグナル取得"""
        try:
            return self.signal_generator.get_long_signals(data)
        except Exception as e:
            self.logger.error(f"ロングシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_short_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ショートエントリーシグナル取得"""
        try:
            return self.signal_generator.get_short_signals(data)
        except Exception as e:
            self.logger.error(f"ショートシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_x_mama_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """X_MAMAシグナル取得"""
        try:
            return self.signal_generator.get_x_mama_signals(data)
        except Exception as e:
            self.logger.error(f"X_MAMAシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_filter_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """フィルターシグナル取得"""
        try:
            return self.signal_generator.get_filter_signals(data)
        except Exception as e:
            self.logger.error(f"フィルターシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_filter_details(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """フィルター詳細情報を取得"""
        try:
            return self.signal_generator.get_filter_details(data)
        except Exception as e:
            self.logger.error(f"フィルター詳細取得中にエラー: {str(e)}")
            return {}
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """全ての高度なメトリクスを取得"""
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
        # フィルタータイプの選択
        filter_type = trial.suggest_categorical('filter_type', [
            FilterType.NONE.value,
            # FilterType.PHASOR_TREND.value,
            FilterType.X_CHOPPINESS.value,
            # FilterType.CORRELATION_CYCLE.value,
            # FilterType.CORRELATION_TREND.value,
            # FilterType.UNIFIED_TREND_CYCLE.value
        ])
        
        params = {
            # X_MAMAパラメータ
            'fast_limit': trial.suggest_float('fast_limit', 0.1, 0.9, step=0.05),
            'slow_limit': trial.suggest_float('slow_limit', 0.01, 0.1, step=0.005),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
            
            # カルマンフィルターパラメータ
            'use_kalman_filter': trial.suggest_categorical('use_kalman_filter', [True, False]),
            'kalman_filter_type': trial.suggest_categorical('kalman_filter_type', ['adaptive', 'quantum_adaptive', 'simple', 'unscented']),
            
            # ゼロラグ処理パラメータ
            'use_zero_lag': trial.suggest_categorical('use_zero_lag', [True, False]),
            
            # フィルター設定
            'filter_type': filter_type,
        }
        
        # フィルタータイプに応じたパラメータ
        if filter_type == FilterType.PHASOR_TREND.value:
            params.update({
                'phasor_period': trial.suggest_int('phasor_period', 10, 50),
                'phasor_trend_threshold': trial.suggest_float('phasor_trend_threshold', 3.0, 12.0, step=0.5),
                'phasor_use_kalman': trial.suggest_categorical('phasor_use_kalman', [True, False])
            })
        elif filter_type == FilterType.CORRELATION_CYCLE.value:
            params.update({
                'correlation_cycle_period': trial.suggest_int('correlation_cycle_period', 10, 50),
                'correlation_cycle_threshold': trial.suggest_float('correlation_cycle_threshold', 5.0, 15.0, step=0.5)
            })
        elif filter_type == FilterType.CORRELATION_TREND.value:
            params.update({
                'correlation_trend_length': trial.suggest_int('correlation_trend_length', 10, 50),
                'correlation_trend_threshold': trial.suggest_float('correlation_trend_threshold', 0.1, 0.7, step=0.05),
                'correlation_trend_smoothing': trial.suggest_categorical('correlation_trend_smoothing', [True, False])
            })
        elif filter_type == FilterType.UNIFIED_TREND_CYCLE.value:
            params.update({
                'unified_period': trial.suggest_int('unified_period', 10, 50),
                'unified_trend_length': trial.suggest_int('unified_trend_length', 10, 50),
                'unified_trend_threshold': trial.suggest_float('unified_trend_threshold', 0.3, 0.8, step=0.05),
                'unified_adaptability': trial.suggest_float('unified_adaptability', 0.5, 0.9, step=0.05),
                'unified_consensus': trial.suggest_categorical('unified_consensus', [True, False])
            })
        elif filter_type == FilterType.X_CHOPPINESS.value:
            params.update({
                'x_choppiness_detector_type': trial.suggest_categorical('x_choppiness_detector_type', ['hody_e', 'phac_e', 'dudi_e', 'absolute_ultimate']),
                'x_choppiness_lp_period': trial.suggest_int('x_choppiness_lp_period', 8, 20),
                'x_choppiness_hp_period': trial.suggest_int('x_choppiness_hp_period', 80, 200),
                'x_choppiness_max_cycle': trial.suggest_int('x_choppiness_max_cycle', 80, 200),
                'x_choppiness_min_cycle': trial.suggest_int('x_choppiness_min_cycle', 8, 20)
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
            'fast_limit': float(params['fast_limit']),
            'slow_limit': float(params['slow_limit']),
            'src_type': params['src_type'],
            'use_kalman_filter': bool(params['use_kalman_filter']),
            'kalman_filter_type': params['kalman_filter_type'],
            'use_zero_lag': bool(params['use_zero_lag']),
            'filter_type': FilterType(params['filter_type'])
        }
        
        # フィルタータイプに応じたパラメータの追加
        filter_type = params['filter_type']
        if filter_type == FilterType.PHASOR_TREND.value:
            strategy_params.update({
                'phasor_period': int(params.get('phasor_period', 20)),
                'phasor_trend_threshold': float(params.get('phasor_trend_threshold', 6.0)),
                'phasor_use_kalman': bool(params.get('phasor_use_kalman', False))
            })
        elif filter_type == FilterType.CORRELATION_CYCLE.value:
            strategy_params.update({
                'correlation_cycle_period': int(params.get('correlation_cycle_period', 20)),
                'correlation_cycle_threshold': float(params.get('correlation_cycle_threshold', 9.0))
            })
        elif filter_type == FilterType.CORRELATION_TREND.value:
            strategy_params.update({
                'correlation_trend_length': int(params.get('correlation_trend_length', 20)),
                'correlation_trend_threshold': float(params.get('correlation_trend_threshold', 0.3)),
                'correlation_trend_smoothing': bool(params.get('correlation_trend_smoothing', False))
            })
        elif filter_type == FilterType.UNIFIED_TREND_CYCLE.value:
            strategy_params.update({
                'unified_period': int(params.get('unified_period', 20)),
                'unified_trend_length': int(params.get('unified_trend_length', 20)),
                'unified_trend_threshold': float(params.get('unified_trend_threshold', 0.5)),
                'unified_adaptability': float(params.get('unified_adaptability', 0.7)),
                'unified_consensus': bool(params.get('unified_consensus', True))
            })
        elif filter_type == FilterType.X_CHOPPINESS.value:
            strategy_params.update({
                'x_choppiness_detector_type': params.get('x_choppiness_detector_type', 'hody_e'),
                'x_choppiness_lp_period': int(params.get('x_choppiness_lp_period', 12)),
                'x_choppiness_hp_period': int(params.get('x_choppiness_hp_period', 124)),
                'x_choppiness_cycle_part': float(params.get('x_choppiness_cycle_part', 0.5)),
                'x_choppiness_max_cycle': int(params.get('x_choppiness_max_cycle', 124)),
                'x_choppiness_min_cycle': int(params.get('x_choppiness_min_cycle', 12)),
                'x_choppiness_max_output': int(params.get('x_choppiness_max_output', 89)),
                'x_choppiness_min_output': int(params.get('x_choppiness_min_output', 5))
            })
        
        return strategy_params
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """ストラテジー情報を取得"""
        filter_type = self._parameters.get('filter_type', FilterType.NONE)
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        
        return {
            'name': 'X_MAMA Enhanced Strategy',
            'description': f'eXtended Mother of Adaptive Moving Average with {filter_name} Filter Integration',
            'parameters': self._parameters.copy(),
            'features': [
                'Adaptive Moving Average based on market cycles',
                'Multiple Ehlers algorithm integration',
                f'Advanced {filter_name} filtering',
                'Kalman Filter integration for noise reduction',
                'Zero-lag processing for faster response',
                'Configurable crossover or position-based signals',
                'Optimized with Numba JIT compilation',
                'High-precision trend and cycle detection'
            ],
            'filter_capabilities': {
                'phasor_trend': 'Phasor analysis for trend/range detection',
                'correlation_cycle': 'Correlation cycle analysis for trend/cycle detection',
                'correlation_trend': 'Linear trend correlation analysis',
                'unified_trend_cycle': '3-algorithm consensus for ultra-high precision',
                'x_choppiness': 'STR-based improved choppiness index for trend/range detection',
                'none': 'Pure X_MAMA signals without filtering'
            }
        }
    
    def reset(self) -> None:
        """ストラテジーの状態をリセット"""
        super().reset()
        if hasattr(self.signal_generator, 'reset'):
            self.signal_generator.reset()