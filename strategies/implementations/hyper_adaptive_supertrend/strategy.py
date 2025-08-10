#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import HyperAdaptiveSupertrendSignalGenerator


class HyperAdaptiveSupertrendStrategy(BaseStrategy):
    """
    ハイパーアダプティブスーパートレンドストラテジー
    
    特徴:
    - 最強のスーパートレンドインジケーター（unified_smoother + unscented_kalman_filter + x_atr）
    - 統合スムーサーによる高精度ミッドライン計算
    - カルマンフィルターによるノイズ除去（オプション）
    - X_ATRによる拡張ボラティリティ測定
    - 動的期間調整対応
    
    エントリー条件:
    - トレンド変化モード: トレンド転換によるシグナル
      * 下降→上昇トレンド転換: ロングシグナル
      * 上昇→下降トレンド転換: ショートシグナル
    - 位置関係モード: 価格とスーパートレンドラインの位置関係 + トレンド方向
      * 上昇トレンドかつ価格 > スーパートレンドライン: ロングシグナル
      * 下降トレンドかつ価格 < スーパートレンドライン: ショートシグナル
    
    エグジット条件:
    - ロング: 下降トレンドに転換
    - ショート: 上昇トレンドに転換
    """
    
    def __init__(
        self,
        # ハイパーアダプティブスーパートレンドのパラメータ
        atr_period: int = 13,                     # X_ATR期間
        multiplier: float = 3.0,                  # ATR乗数
        atr_method: str = 'str',                  # X_ATRの計算方法
        atr_smoother_type: str = 'super_smoother',           # X_ATRのスムーサータイプ
        midline_smoother_type: str = 'FRAMA',     # ミッドラインスムーサータイプ
        midline_period: int = 21,                 # ミッドライン期間
        src_type: str = 'hlc3',                   # ソースタイプ
        # カルマンフィルターパラメータ
        enable_kalman: bool = True,               # カルマンフィルター使用フラグ
        kalman_alpha: float = 1.0,                # UKFアルファパラメータ
        kalman_beta: float = 2.0,                 # UKFベータパラメータ
        kalman_kappa: float = 0.0,                # UKFカッパパラメータ
        kalman_process_noise: float = 0.01,       # UKFプロセスノイズ
        # 動的期間パラメータ
        use_dynamic_period: bool = True,          # 動的期間を使用するか
        cycle_part: float = 0.5,                  # サイクル部分の倍率
        detector_type: str = 'hody_e',            # 検出器タイプ
        max_cycle: int = 124,                     # 最大サイクル期間
        min_cycle: int = 13,                      # 最小サイクル期間
        max_output: int = 124,                    # 最大出力値
        min_output: int = 13,                     # 最小出力値
        lp_period: int = 13,                      # ローパスフィルター期間
        hp_period: int = 124,                     # ハイパスフィルター期間
        # シグナル設定
        trend_change_mode: bool = False,           # トレンド変化シグナル(True)または位置関係シグナル(False)
        # フィルターシグナル設定
        enable_filter_signals: bool = True,       # フィルターシグナルを有効にするか
        phasor_filter_period: int = 20,           # PhasorTrendFilterの期間
        phasor_filter_threshold: float = 6.0,     # PhasorTrendFilterの閾値
        correlation_cycle_period: int = 20,       # CorrelationCycleFilterの期間
        correlation_cycle_threshold: float = 9.0, # CorrelationCycleFilterの閾値
        correlation_trend_length: int = 20,       # CorrelationTrendFilterの長さ
        correlation_trend_threshold: float = 0.3, # CorrelationTrendFilterの閾値
        unified_trend_cycle_period: int = 20,     # UnifiedTrendCycleFilterの期間
        unified_trend_cycle_threshold: float = 0.5, # UnifiedTrendCycleFilterの閾値
        filter_consensus_mode: bool = True        # フィルター合意モード（True=全フィルター合意、False=多数決）
    ):
        """
        初期化
        
        Args:
            atr_period: X_ATR期間（デフォルト: 14）
            multiplier: ATR乗数（デフォルト: 3.0）
            atr_method: X_ATRの計算方法（'atr' または 'str'、デフォルト: 'str'）
            atr_smoother_type: X_ATRのスムーサータイプ（デフォルト: 'sma'）
            midline_smoother_type: ミッドラインスムーサータイプ（デフォルト: 'frama'）
            midline_period: ミッドライン期間（デフォルト: 21）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
            enable_kalman: カルマンフィルター使用フラグ（デフォルト: True）
            kalman_alpha: UKFアルファパラメータ（デフォルト: 1.0）
            kalman_beta: UKFベータパラメータ（デフォルト: 2.0）
            kalman_kappa: UKFカッパパラメータ（デフォルト: 0.0）
            kalman_process_noise: UKFプロセスノイズ（デフォルト: 0.01）
            use_dynamic_period: 動的期間を使用するか（デフォルト: True）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            detector_type: 検出器タイプ（デフォルト: 'hody_e'）
            max_cycle: 最大サイクル期間（デフォルト: 124）
            min_cycle: 最小サイクル期間（デフォルト: 13）
            max_output: 最大出力値（デフォルト: 124）
            min_output: 最小出力値（デフォルト: 13）
            lp_period: ローパスフィルター期間（デフォルト: 13）
            hp_period: ハイパスフィルター期間（デフォルト: 124）
            trend_change_mode: トレンド変化シグナル(True)または位置関係シグナル(False)
            enable_filter_signals: フィルターシグナルを有効にするか（デフォルト: True）
            phasor_filter_period: PhasorTrendFilterの期間（デフォルト: 20）
            phasor_filter_threshold: PhasorTrendFilterの閾値（デフォルト: 6.0）
            correlation_cycle_period: CorrelationCycleFilterの期間（デフォルト: 20）
            correlation_cycle_threshold: CorrelationCycleFilterの閾値（デフォルト: 9.0）
            correlation_trend_length: CorrelationTrendFilterの長さ（デフォルト: 20）
            correlation_trend_threshold: CorrelationTrendFilterの閾値（デフォルト: 0.3）
            unified_trend_cycle_period: UnifiedTrendCycleFilterの期間（デフォルト: 20）
            unified_trend_cycle_threshold: UnifiedTrendCycleFilterの閾値（デフォルト: 0.5）
            filter_consensus_mode: フィルター合意モード（True=全フィルター合意、False=多数決）
        """
        signal_type = "TrendChange" if trend_change_mode else "Position"
        kalman_str = f"_kalman({kalman_alpha},{kalman_beta},{kalman_kappa})" if enable_kalman else ""
        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_period else ""
        filter_str = "_filters" if enable_filter_signals else ""
        
        super().__init__(
            f"HyperAdaptiveSupertrend{signal_type}("
            f"atr={atr_period}×{multiplier}_{atr_method}_{atr_smoother_type}, "
            f"mid={midline_period}_{midline_smoother_type}, "
            f"{src_type}{kalman_str}{dynamic_str}{filter_str})"
        )
        
        # パラメータの設定
        self._parameters = {
            'atr_period': atr_period,
            'multiplier': multiplier,
            'atr_method': atr_method,
            'atr_smoother_type': atr_smoother_type,
            'midline_smoother_type': midline_smoother_type,
            'midline_period': midline_period,
            'src_type': src_type,
            'enable_kalman': enable_kalman,
            'kalman_alpha': kalman_alpha,
            'kalman_beta': kalman_beta,
            'kalman_kappa': kalman_kappa,
            'kalman_process_noise': kalman_process_noise,
            'use_dynamic_period': use_dynamic_period,
            'cycle_part': cycle_part,
            'detector_type': detector_type,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'trend_change_mode': trend_change_mode,
            'enable_filter_signals': enable_filter_signals,
            'phasor_filter_period': phasor_filter_period,
            'phasor_filter_threshold': phasor_filter_threshold,
            'correlation_cycle_period': correlation_cycle_period,
            'correlation_cycle_threshold': correlation_cycle_threshold,
            'correlation_trend_length': correlation_trend_length,
            'correlation_trend_threshold': correlation_trend_threshold,
            'unified_trend_cycle_period': unified_trend_cycle_period,
            'unified_trend_cycle_threshold': unified_trend_cycle_threshold,
            'filter_consensus_mode': filter_consensus_mode
        }
        
        # シグナル生成器の初期化
        self.signal_generator = HyperAdaptiveSupertrendSignalGenerator(
            atr_period=atr_period,
            multiplier=multiplier,
            atr_method=atr_method,
            atr_smoother_type=atr_smoother_type,
            midline_smoother_type=midline_smoother_type,
            midline_period=midline_period,
            src_type=src_type,
            enable_kalman=enable_kalman,
            kalman_alpha=kalman_alpha,
            kalman_beta=kalman_beta,
            kalman_kappa=kalman_kappa,
            kalman_process_noise=kalman_process_noise,
            use_dynamic_period=use_dynamic_period,
            cycle_part=cycle_part,
            detector_type=detector_type,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            lp_period=lp_period,
            hp_period=hp_period,
            trend_change_mode=trend_change_mode,
            enable_filter_signals=enable_filter_signals,
            phasor_filter_period=phasor_filter_period,
            phasor_filter_threshold=phasor_filter_threshold,
            correlation_cycle_period=correlation_cycle_period,
            correlation_cycle_threshold=correlation_cycle_threshold,
            correlation_trend_length=correlation_trend_length,
            correlation_trend_threshold=correlation_trend_threshold,
            unified_trend_cycle_period=unified_trend_cycle_period,
            unified_trend_cycle_threshold=unified_trend_cycle_threshold,
            filter_consensus_mode=filter_consensus_mode
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
    
    def get_supertrend_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        スーパートレンドライン値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: スーパートレンドライン値
        """
        try:
            return self.signal_generator.get_supertrend_values(data)
        except Exception as e:
            self.logger.error(f"スーパートレンドライン値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_upper_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        上側バンド値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: 上側バンド値
        """
        try:
            return self.signal_generator.get_upper_band(data)
        except Exception as e:
            self.logger.error(f"上側バンド値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_lower_band(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        下側バンド値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: 下側バンド値
        """
        try:
            return self.signal_generator.get_lower_band(data)
        except Exception as e:
            self.logger.error(f"下側バンド値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド方向値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: トレンド方向値（1=上昇、-1=下降）
        """
        try:
            return self.signal_generator.get_trend_values(data)
        except Exception as e:
            self.logger.error(f"トレンド方向値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_midline_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ミッドライン値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: ミッドライン値（統合スムーサー結果）
        """
        try:
            return self.signal_generator.get_midline_values(data)
        except Exception as e:
            self.logger.error(f"ミッドライン値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_atr_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_ATR値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: X_ATR値
        """
        try:
            return self.signal_generator.get_atr_values(data)
        except Exception as e:
            self.logger.error(f"X_ATR値取得中にエラー: {str(e)}")
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
            # ハイパーアダプティブスーパートレンドパラメータ
            'atr_period': trial.suggest_int('atr_period', 10, 30, step=2),
            'multiplier': trial.suggest_float('multiplier', 1.5, 5.0, step=0.25),
            'atr_method': trial.suggest_categorical('atr_method', ['atr', 'str']),
            'atr_smoother_type': trial.suggest_categorical('atr_smoother_type', ['super_smoother', 'ultimate_smoother', 'frama', 'alma']),
            'midline_smoother_type': trial.suggest_categorical('midline_smoother_type', ['FRAMA', 'alma', 'ema', 'sma', 'hma']),
            'midline_period': trial.suggest_int('midline_period', 14, 50, step=3),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # カルマンフィルターパラメータ
            'enable_kalman': trial.suggest_categorical('enable_kalman', [True, False]),
            'kalman_alpha': trial.suggest_float('kalman_alpha', 0.5, 2.0, step=0.1),
            'kalman_beta': trial.suggest_float('kalman_beta', 1.0, 3.0, step=0.2),
            'kalman_process_noise': trial.suggest_float('kalman_process_noise', 0.001, 0.1, step=0.005),
            
            # 動的期間パラメータ
            'use_dynamic_period': trial.suggest_categorical('use_dynamic_period', [True, False]),
            'cycle_part': trial.suggest_float('cycle_part', 0.3, 0.8, step=0.1),
            'detector_type': trial.suggest_categorical('detector_type', ['hody_e', 'phac_e', 'dudi_e']),
            
            # シグナル設定
            'trend_change_mode': trial.suggest_categorical('trend_change_mode', [True, False]),
            
            # フィルターシグナル設定
            'enable_filter_signals': trial.suggest_categorical('enable_filter_signals', [True, False]),
            'phasor_filter_period': trial.suggest_int('phasor_filter_period', 15, 30, step=2),
            'phasor_filter_threshold': trial.suggest_float('phasor_filter_threshold', 4.0, 10.0, step=1.0),
            'correlation_cycle_period': trial.suggest_int('correlation_cycle_period', 15, 30, step=2),
            'correlation_cycle_threshold': trial.suggest_float('correlation_cycle_threshold', 6.0, 12.0, step=1.0),
            'correlation_trend_length': trial.suggest_int('correlation_trend_length', 15, 30, step=2),
            'correlation_trend_threshold': trial.suggest_float('correlation_trend_threshold', 0.2, 0.5, step=0.05),
            'unified_trend_cycle_period': trial.suggest_int('unified_trend_cycle_period', 15, 30, step=2),
            'unified_trend_cycle_threshold': trial.suggest_float('unified_trend_cycle_threshold', 0.3, 0.7, step=0.1),
            'filter_consensus_mode': trial.suggest_categorical('filter_consensus_mode', [True, False])
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
            'atr_period': int(params['atr_period']),
            'multiplier': float(params['multiplier']),
            'atr_method': params['atr_method'],
            'atr_smoother_type': params['atr_smoother_type'],
            'midline_smoother_type': params['midline_smoother_type'],
            'midline_period': int(params['midline_period']),
            'src_type': params['src_type'],
            'enable_kalman': bool(params['enable_kalman']),
            'kalman_alpha': float(params['kalman_alpha']),
            'kalman_beta': float(params['kalman_beta']),
            'kalman_process_noise': float(params['kalman_process_noise']),
            'use_dynamic_period': bool(params['use_dynamic_period']),
            'cycle_part': float(params['cycle_part']),
            'detector_type': params['detector_type'],
            'trend_change_mode': bool(params['trend_change_mode']),
            'enable_filter_signals': bool(params.get('enable_filter_signals', True)),
            'phasor_filter_period': int(params.get('phasor_filter_period', 20)),
            'phasor_filter_threshold': float(params.get('phasor_filter_threshold', 6.0)),
            'correlation_cycle_period': int(params.get('correlation_cycle_period', 20)),
            'correlation_cycle_threshold': float(params.get('correlation_cycle_threshold', 9.0)),
            'correlation_trend_length': int(params.get('correlation_trend_length', 20)),
            'correlation_trend_threshold': float(params.get('correlation_trend_threshold', 0.3)),
            'unified_trend_cycle_period': int(params.get('unified_trend_cycle_period', 20)),
            'unified_trend_cycle_threshold': float(params.get('unified_trend_cycle_threshold', 0.5)),
            'filter_consensus_mode': bool(params.get('filter_consensus_mode', True))
        }
        return strategy_params
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        ストラテジー情報を取得
        
        Returns:
            Dict[str, Any]: ストラテジー情報
        """
        return {
            'name': 'Hyper Adaptive Supertrend Strategy',
            'description': 'Ultimate Supertrend with Unified Smoother, Unscented Kalman Filter, and X_ATR',
            'parameters': self._parameters.copy(),
            'features': [
                'Unified Smoother for high-precision midline calculation',
                'Unscented Kalman Filter for noise reduction',
                'X_ATR for extended volatility measurement',
                'Dynamic period adjustment support',
                'Configurable trend change or position-based signals',
                'Optimized with Numba JIT compilation',
                'Advanced filter signals integration',
                'PhasorTrendFilter for phasor analysis',
                'CorrelationCycleFilter for cycle detection',
                'CorrelationTrendFilter for trend analysis',
                'UnifiedTrendCycleFilter for consensus analysis',
                'Consensus filtering with configurable modes'
            ]
        }
    
    def reset(self) -> None:
        """
        ストラテジーの状態をリセット
        """
        super().reset()
        if hasattr(self.signal_generator, 'reset'):
            self.signal_generator.reset()