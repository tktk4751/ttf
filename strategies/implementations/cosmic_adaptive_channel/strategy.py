#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 Cosmic Adaptive Channel Strategy - 宇宙最強トレンドフォロー戦略 🌌

ZASimpleStrategyを参考にした、Cosmic Adaptive Channelによる
革命的なブレイクアウト・トレンドフォロー戦略システム

特徴:
- 8層ハイブリッドシステムによる超高精度エントリー検出
- 量子コヒーレンス + 神経適応学習による偽シグナル完全防御
- Optuna最適化対応
- 宇宙知能による動的パラメータ調整
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import CosmicAdaptiveChannelSignalGenerator


class CosmicAdaptiveChannelStrategy(BaseStrategy):
    """
    🌌 Cosmic Adaptive Channel Strategy - 宇宙最強トレンドフォロー戦略
    
    特徴:
    - Cosmic Adaptive Channelによる超高精度ブレイクアウト検出
    - 8層ハイブリッドシステム（量子統計フュージョン、ヒルベルト位相解析等）
    - 宇宙知能による偽シグナル完全防御
    - Optuna最適化による動的パラメータ調整
    
    エントリー条件:
    - ロング: Cosmic Adaptive Channelの上抜けブレイクアウト + 上昇トレンド + 高信頼度
    - ショート: Cosmic Adaptive Channelの下抜けブレイクアウト + 下降トレンド + 高信頼度
    
    エグジット条件:
    - ロング: 1のときロングエントリー&ショートエグジット
    - ショート: -1のときショートエントリー&ロングエグジット
    """
    
    def __init__(
        self,
        # Cosmic Adaptive Channelパラメータ
        atr_period: int = 34,
        base_multiplier: float = 1.75,
        quantum_window: int = 20,
        neural_window: int = 100,
        volatility_window: int = 30,
        src_type: str = 'hlc3',
        
        # シグナルフィルタリングパラメータ
        min_confidence: float = 0.2,
        min_trend_strength: float = 0.2,
        min_quantum_coherence: float = 0.5,
        
        # 決済パラメータ
        enable_exit_signals: bool = True,
        exit_trend_threshold: float = 0.0,
        exit_confidence_threshold: float = 0.0,
        exit_quantum_threshold: float = 0.0,
        
        # 宇宙知能強化パラメータ
        enable_cosmic_enhancement: bool = True,
        cosmic_enhancement_threshold: float = 0.3,
        require_strong_signals: bool = False,
        
        # 高度設定パラメータ
        adaptive_threshold_mode: bool = True,
        volatility_adjustment: bool = True,
        neural_learning_rate: float = 0.01,
        quantum_coherence_sensitivity: float = 1.0
    ):
        """
        🌌 宇宙最強トレンドフォロー戦略初期化
        
        Args:
            # Cosmic Adaptive Channelパラメータ
            atr_period: ATR計算期間（デフォルト: 21）
            base_multiplier: 基本チャネル幅倍率（デフォルト: 2.5）
            quantum_window: 量子解析ウィンドウ（デフォルト: 50）
            neural_window: 神経学習ウィンドウ（デフォルト: 100）
            volatility_window: ボラティリティ解析ウィンドウ（デフォルト: 30）
            src_type: 価格ソースタイプ（デフォルト: 'hlc3'）
            
            # シグナルフィルタリングパラメータ
            min_confidence: 最小信頼度しきい値（デフォルト: 0.5）
            min_trend_strength: 最小トレンド強度しきい値（デフォルト: 0.3）
            min_quantum_coherence: 最小量子コヒーレンスしきい値（デフォルト: 0.4）
            
            # 決済パラメータ
            enable_exit_signals: 決済シグナルを有効にするか（デフォルト: True）
            exit_trend_threshold: 決済トレンド強度しきい値（デフォルト: 0.2）
            exit_confidence_threshold: 決済信頼度しきい値（デフォルト: 0.3）
            exit_quantum_threshold: 決済量子コヒーレンスしきい値（デフォルト: 0.3）
            
            # 宇宙知能強化パラメータ
            enable_cosmic_enhancement: 宇宙知能強化を有効にするか（デフォルト: True）
            cosmic_enhancement_threshold: 宇宙強化しきい値（デフォルト: 0.4）
            require_strong_signals: 強いシグナルのみを要求するか（デフォルト: False）
            
            # 高度設定パラメータ
            adaptive_threshold_mode: 適応的しきい値モード（デフォルト: True）
            volatility_adjustment: ボラティリティ調整（デフォルト: True）
            neural_learning_rate: 神経学習レート（デフォルト: 0.01）
            quantum_coherence_sensitivity: 量子コヒーレンス感度（デフォルト: 1.0）
        """
        super().__init__("CosmicAdaptiveChannel")
        
        # パラメータの設定
        self._parameters = {
            # Cosmic Adaptive Channelパラメータ
            'atr_period': atr_period,
            'base_multiplier': base_multiplier,
            'quantum_window': quantum_window,
            'neural_window': neural_window,
            'volatility_window': volatility_window,
            'src_type': src_type,
            
            # シグナルフィルタリングパラメータ
            'min_confidence': min_confidence,
            'min_trend_strength': min_trend_strength,
            'min_quantum_coherence': min_quantum_coherence,
            
            # 決済パラメータ
            'enable_exit_signals': enable_exit_signals,
            'exit_trend_threshold': exit_trend_threshold,
            'exit_confidence_threshold': exit_confidence_threshold,
            'exit_quantum_threshold': exit_quantum_threshold,
            
            # 宇宙知能強化パラメータ
            'enable_cosmic_enhancement': enable_cosmic_enhancement,
            'cosmic_enhancement_threshold': cosmic_enhancement_threshold,
            'require_strong_signals': require_strong_signals,
            
            # 高度設定パラメータ
            'adaptive_threshold_mode': adaptive_threshold_mode,
            'volatility_adjustment': volatility_adjustment,
            'neural_learning_rate': neural_learning_rate,
            'quantum_coherence_sensitivity': quantum_coherence_sensitivity
        }
        
        # シグナル生成器の初期化
        self.signal_generator = CosmicAdaptiveChannelSignalGenerator(
            # Cosmic Adaptive Channelパラメータ
            atr_period=atr_period,
            base_multiplier=base_multiplier,
            quantum_window=quantum_window,
            neural_window=neural_window,
            volatility_window=volatility_window,
            src_type=src_type,
            
            # シグナルフィルタリングパラメータ
            min_confidence=min_confidence,
            min_trend_strength=min_trend_strength,
            min_quantum_coherence=min_quantum_coherence,
            
            # 決済パラメータ
            enable_exit_signals=enable_exit_signals,
            exit_trend_threshold=exit_trend_threshold,
            exit_confidence_threshold=exit_confidence_threshold,
            exit_quantum_threshold=exit_quantum_threshold,
            
            # 宇宙知能強化パラメータ
            enable_cosmic_enhancement=enable_cosmic_enhancement,
            cosmic_enhancement_threshold=cosmic_enhancement_threshold,
            require_strong_signals=require_strong_signals,
            
            # 高度設定パラメータ
            adaptive_threshold_mode=adaptive_threshold_mode,
            volatility_adjustment=volatility_adjustment,
            neural_learning_rate=neural_learning_rate,
            quantum_coherence_sensitivity=quantum_coherence_sensitivity
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        🌌 宇宙最強エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル（1=ロング、-1=ショート、0=なし）
        """
        try:
            return self.signal_generator.get_entry_signals(data)
        except Exception as e:
            self.logger.error(f"🌌 エントリーシグナル生成中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        🌌 宇宙最強エグジットシグナルを生成する（修正版）
        
        エグジット条件:
        - ロングポジション時: ショートシグナル(-1)でエグジット
        - ショートポジション時: ロングシグナル(1)でエグジット
        - 同方向シグナルの場合はエグジットしない
        
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
            self.logger.error(f"🌌 エグジットシグナル生成中にエラー: {str(e)}")
            return False
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        🌌 宇宙最強最適化パラメータを生成
        
        Args:
            trial: Optunaのトライアル
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        params = {
            # Cosmic Adaptive Channelパラメータ
            'atr_period': trial.suggest_int('atr_period', 10, 50, step=5),
            'base_multiplier': trial.suggest_float('base_multiplier', 1.5, 4.0, step=0.25),
            'quantum_window': trial.suggest_int('quantum_window', 20, 100, step=10),
            'neural_window': trial.suggest_int('neural_window', 50, 200, step=25),
            'volatility_window': trial.suggest_int('volatility_window', 10, 50, step=5),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # シグナルフィルタリングパラメータ
            'min_confidence': trial.suggest_float('min_confidence', 0.2, 0.8, step=0.1),
            'min_trend_strength': trial.suggest_float('min_trend_strength', 0.1, 0.6, step=0.1),
            'min_quantum_coherence': trial.suggest_float('min_quantum_coherence', 0.2, 0.7, step=0.1),
            
            # 決済パラメータ
            'exit_trend_threshold': trial.suggest_float('exit_trend_threshold', 0.1, 0.4, step=0.05),
            'exit_confidence_threshold': trial.suggest_float('exit_confidence_threshold', 0.1, 0.5, step=0.05),
            'exit_quantum_threshold': trial.suggest_float('exit_quantum_threshold', 0.1, 0.5, step=0.05),
            
            # 宇宙知能強化パラメータ
            'enable_cosmic_enhancement': trial.suggest_categorical('enable_cosmic_enhancement', [True, False]),
            'cosmic_enhancement_threshold': trial.suggest_float('cosmic_enhancement_threshold', 0.2, 0.6, step=0.1),
            'require_strong_signals': trial.suggest_categorical('require_strong_signals', [True, False]),
            
            # 高度設定パラメータ
            'adaptive_threshold_mode': trial.suggest_categorical('adaptive_threshold_mode', [True, False]),
            'volatility_adjustment': trial.suggest_categorical('volatility_adjustment', [True, False]),
            'neural_learning_rate': trial.suggest_float('neural_learning_rate', 0.005, 0.05, step=0.005),
            'quantum_coherence_sensitivity': trial.suggest_float('quantum_coherence_sensitivity', 0.5, 2.0, step=0.25)
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        🌌 最適化パラメータを戦略パラメータに変換
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            Dict[str, Any]: 戦略パラメータ
        """
        strategy_params = {
            # Cosmic Adaptive Channelパラメータ
            'atr_period': int(params['atr_period']),
            'base_multiplier': float(params['base_multiplier']),
            'quantum_window': int(params['quantum_window']),
            'neural_window': int(params['neural_window']),
            'volatility_window': int(params['volatility_window']),
            'src_type': params['src_type'],
            
            # シグナルフィルタリングパラメータ
            'min_confidence': float(params['min_confidence']),
            'min_trend_strength': float(params['min_trend_strength']),
            'min_quantum_coherence': float(params['min_quantum_coherence']),
            
            # 決済パラメータ
            'enable_exit_signals': True,  # 常に有効
            'exit_trend_threshold': float(params['exit_trend_threshold']),
            'exit_confidence_threshold': float(params['exit_confidence_threshold']),
            'exit_quantum_threshold': float(params['exit_quantum_threshold']),
            
            # 宇宙知能強化パラメータ
            'enable_cosmic_enhancement': bool(params['enable_cosmic_enhancement']),
            'cosmic_enhancement_threshold': float(params['cosmic_enhancement_threshold']),
            'require_strong_signals': bool(params['require_strong_signals']),
            
            # 高度設定パラメータ
            'adaptive_threshold_mode': bool(params['adaptive_threshold_mode']),
            'volatility_adjustment': bool(params['volatility_adjustment']),
            'neural_learning_rate': float(params['neural_learning_rate']),
            'quantum_coherence_sensitivity': float(params['quantum_coherence_sensitivity'])
        }
        return strategy_params
    
    def get_cosmic_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        🌌 Cosmic Adaptive Channelのバンド値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Dict[str, np.ndarray]: バンド値の辞書
        """
        try:
            center_line, upper_channel, lower_channel = self.signal_generator.get_cosmic_band_values(data)
            return {
                'center_line': center_line,
                'upper_channel': upper_channel,
                'lower_channel': lower_channel
            }
        except Exception as e:
            self.logger.error(f"🌌 Cosmicバンド値取得中にエラー: {str(e)}")
            return {
                'center_line': np.array([]),
                'upper_channel': np.array([]),
                'lower_channel': np.array([])
            }
    
    def get_cosmic_indicators(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        🌌 Cosmic Adaptive Channelの指標を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Dict[str, np.ndarray]: 宇宙指標の辞書
        """
        try:
            return {
                'breakout_confidence': self.signal_generator.get_breakout_confidence(data),
                'trend_strength': self.signal_generator.get_trend_strength(data),
                'quantum_coherence': self.signal_generator.get_quantum_coherence(data),
                'channel_efficiency': self.signal_generator.get_channel_efficiency(data),
                'neural_adaptation_score': self.signal_generator.get_neural_adaptation_score(data),
                'volatility_regime': self.signal_generator.get_volatility_regime(data)
            }
        except Exception as e:
            self.logger.error(f"🌌 Cosmic指標取得中にエラー: {str(e)}")
            return {
                'breakout_confidence': np.array([]),
                'trend_strength': np.array([]),
                'quantum_coherence': np.array([]),
                'channel_efficiency': np.array([]),
                'neural_adaptation_score': np.array([]),
                'volatility_regime': np.array([])
            }
    
    def get_cosmic_intelligence_report(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, Any]:
        """
        🌌 宇宙知能レポートを取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Dict[str, Any]: 宇宙知能レポート
        """
        try:
            return self.signal_generator.get_cosmic_intelligence_report(data)
        except Exception as e:
            self.logger.error(f"🌌 宇宙知能レポート取得中にエラー: {str(e)}")
            return {
                'cosmic_intelligence_score': 0.0,
                'current_trend_phase': 'unknown',
                'current_volatility_regime': 'unknown',
                'current_breakout_probability': 0.0,
                'current_quantum_coherence': 0.0,
                'current_neural_adaptation': 0.0,
                'false_signal_rate': 1.0,
                'current_channel_efficiency': 0.0
            }
    
    def get_current_cosmic_state(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, Any]:
        """
        🌌 現在の宇宙状態情報を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Dict[str, Any]: 現在の宇宙状態情報
        """
        try:
            return self.signal_generator.get_current_cosmic_state(data)
        except Exception as e:
            self.logger.error(f"🌌 宇宙状態取得中にエラー: {str(e)}")
            return {
                'latest_breakout_signal': 0,
                'latest_confidence': 0.0,
                'latest_trend_strength': 0.0,
                'latest_quantum_coherence': 0.0,
                'latest_channel_efficiency': 0.0,
                'latest_neural_adaptation': 0.0,
                'volatility_regime': 3
            }
    
    def get_strategy_summary(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, Any]:
        """
        🌌 宇宙戦略サマリーを取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Dict[str, Any]: 戦略サマリー
        """
        try:
            # 基本情報
            summary = {
                'strategy_name': 'Cosmic Adaptive Channel',
                'strategy_version': '1.0.0',
                'strategy_type': 'Trend Following Breakout',
                'parameters': self._parameters.copy()
            }
            
            # 現在の宇宙状態
            cosmic_state = self.get_current_cosmic_state(data)
            summary['current_cosmic_state'] = cosmic_state
            
            # 宇宙知能レポート
            intelligence_report = self.get_cosmic_intelligence_report(data)
            summary['cosmic_intelligence'] = intelligence_report
            
            # パフォーマンス指標
            if data is not None:
                entry_signals = self.generate_entry(data)
                total_signals = np.sum(np.abs(entry_signals))
                long_signals = np.sum(entry_signals == 1)
                short_signals = np.sum(entry_signals == -1)
                
                summary['signal_statistics'] = {
                    'total_signals': int(total_signals),
                    'long_signals': int(long_signals),
                    'short_signals': int(short_signals),
                    'signal_density': float(total_signals / len(data)) if len(data) > 0 else 0.0,
                    'long_short_ratio': float(long_signals / short_signals) if short_signals > 0 else float('inf')
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"🌌 戦略サマリー取得中にエラー: {str(e)}")
            return {
                'strategy_name': 'Cosmic Adaptive Channel',
                'strategy_version': '1.0.0',
                'error': str(e)
            }