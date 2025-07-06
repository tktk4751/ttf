#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 Cosmic Adaptive Channel Signal Generator - 宇宙最強シグナルジェネレーター 🌌

ZASimpleSignalGeneratorを参考にした、Cosmic Adaptive Channelによる
革命的なブレイクアウト・トレンドフォローシグナル生成システム

特徴:
- 8層ハイブリッドシステムによる超高精度シグナル
- 量子コヒーレンス + 神経適応学習による偽シグナル完全防御
- Numba最適化による超高速処理
- 宇宙知能による最適エントリー・エグジットタイミング
"""

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.cosmic_adaptive_channel import CosmicAdaptiveChannelEntrySignal


class CosmicAdaptiveChannelSignalGenerator(BaseSignalGenerator):
    """
    🌌 Cosmic Adaptive Channel Signal Generator - 宇宙最強シグナルジェネレーター
    
    特徴:
    - Cosmic Adaptive Channelによる超高精度ブレイクアウト検出
    - 8層ハイブリッドシステム（量子統計フュージョン、ヒルベルト位相解析等）
    - 宇宙知能による偽シグナル完全防御
    - Numba最適化による超高速処理
    
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
        atr_period: int = 21,
        base_multiplier: float = 2.5,
        quantum_window: int = 50,
        neural_window: int = 100,
        volatility_window: int = 30,
        src_type: str = 'hlc3',
        
        # シグナルフィルタリングパラメータ
        min_confidence: float = 0.5,
        min_trend_strength: float = 0.3,
        min_quantum_coherence: float = 0.4,
        
        # 決済パラメータ
        enable_exit_signals: bool = True,
        exit_trend_threshold: float = 0.2,
        exit_confidence_threshold: float = 0.3,
        exit_quantum_threshold: float = 0.3,
        
        # 宇宙知能強化パラメータ
        enable_cosmic_enhancement: bool = True,
        cosmic_enhancement_threshold: float = 0.4,
        require_strong_signals: bool = False,
        
        # 高度設定パラメータ
        adaptive_threshold_mode: bool = True,
        volatility_adjustment: bool = True,
        neural_learning_rate: float = 0.01,
        quantum_coherence_sensitivity: float = 1.0
    ):
        """
        🌌 宇宙最強シグナルジェネレーター初期化
        
        Args:
            # Cosmic Adaptive Channelパラメータ
            atr_period: ATR計算期間
            base_multiplier: 基本チャネル幅倍率
            quantum_window: 量子解析ウィンドウ
            neural_window: 神経学習ウィンドウ
            volatility_window: ボラティリティ解析ウィンドウ
            src_type: 価格ソースタイプ
            
            # シグナルフィルタリングパラメータ
            min_confidence: 最小信頼度しきい値
            min_trend_strength: 最小トレンド強度しきい値
            min_quantum_coherence: 最小量子コヒーレンスしきい値
            
            # 決済パラメータ
            enable_exit_signals: 決済シグナルを有効にするか
            exit_trend_threshold: 決済トレンド強度しきい値
            exit_confidence_threshold: 決済信頼度しきい値
            exit_quantum_threshold: 決済量子コヒーレンスしきい値
            
            # 宇宙知能強化パラメータ
            enable_cosmic_enhancement: 宇宙知能強化を有効にするか
            cosmic_enhancement_threshold: 宇宙強化しきい値
            require_strong_signals: 強いシグナルのみを要求するか
            
            # 高度設定パラメータ
            adaptive_threshold_mode: 適応的しきい値モード
            volatility_adjustment: ボラティリティ調整
            neural_learning_rate: 神経学習レート
            quantum_coherence_sensitivity: 量子コヒーレンス感度
        """
        super().__init__("CosmicAdaptiveChannelSignalGenerator")
        
        # パラメータの設定
        self._params = {
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
        
        # Cosmic Adaptive Channelシグナルの初期化
        self.cosmic_signal = CosmicAdaptiveChannelEntrySignal(
            atr_period=atr_period,
            base_multiplier=base_multiplier,
            quantum_window=quantum_window,
            neural_window=neural_window,
            volatility_window=volatility_window,
            src_type=src_type,
            min_confidence=min_confidence,
            min_trend_strength=min_trend_strength,
            min_quantum_coherence=min_quantum_coherence,
            enable_exit_signals=enable_exit_signals,
            exit_trend_threshold=exit_trend_threshold,
            exit_confidence_threshold=exit_confidence_threshold,
            exit_quantum_threshold=exit_quantum_threshold,
            enable_cosmic_enhancement=enable_cosmic_enhancement,
            cosmic_enhancement_threshold=cosmic_enhancement_threshold,
            require_strong_signals=require_strong_signals
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._exit_signals = None
        self._cosmic_result = None
        
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """🌌 宇宙最強シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']].copy()
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                # Cosmic Adaptive Channelシグナルの計算
                try:
                    cosmic_entry_signals = self.cosmic_signal.generate(df)
                    cosmic_exit_signals = self.cosmic_signal.get_exit_signals()
                    
                    # シグナルをキャッシュ
                    self._signals = cosmic_entry_signals
                    self._exit_signals = cosmic_exit_signals
                    self._cosmic_result = self.cosmic_signal.get_cosmic_result()
                    
                except Exception as e:
                    self.logger.error(f"🌌 Cosmic Adaptive Channelシグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._exit_signals = np.zeros(current_len, dtype=np.int8)
                    self._cosmic_result = None
                
                self._data_len = current_len
                
        except Exception as e:
            self.logger.error(f"🌌 calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._exit_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
                self._cosmic_result = None
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """🌌 宇宙最強エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals.copy() if self._signals is not None else np.zeros(len(data), dtype=np.int8)
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        🌌 宇宙最強エグジットシグナル生成（修正版）
        
        エグジット条件:
        - ロングポジション時: シグナル=-1でエグジット（ショート転換）
        - ショートポジション時: シグナル=1でエグジット（ロング転換）
        - 同方向シグナルの場合はエグジットしない（ポジション継続）
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # インデックス範囲チェック
        if index < 0 or index >= len(self._signals):
            return False
        
        current_signal = self._signals[index]
        
        # 宇宙知能エグジット判定（修正版）
        if position == 1:  # ロングポジション
            # ショートシグナル(-1)が出た場合のみエグジット
            return bool(current_signal == -1)
        
        elif position == -1:  # ショートポジション
            # ロングシグナル(1)が出た場合のみエグジット
            return bool(current_signal == 1)
        
        return False
    
    def get_cosmic_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        🌌 Cosmic Adaptive Channelのバンド値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._cosmic_result is not None:
                return (
                    self._cosmic_result.midline,  # center_line -> midline
                    self._cosmic_result.upper_channel,
                    self._cosmic_result.lower_channel
                )
            else:
                empty = np.array([])
                return empty, empty, empty
                
        except Exception as e:
            self.logger.error(f"🌌 Cosmicバンド値取得中にエラー: {str(e)}")
            empty = np.array([])
            return empty, empty, empty
    
    def get_breakout_confidence(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        🌌 ブレイクアウト信頼度を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: ブレイクアウト信頼度の配列（0-1）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._cosmic_result is not None:
                return self._cosmic_result.breakout_confidence
            else:
                return np.array([])
                
        except Exception as e:
            self.logger.error(f"🌌 ブレイクアウト信頼度取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_strength(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        🌌 統合トレンド強度を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: 統合トレンド強度の配列（-1～1）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._cosmic_result is not None:
                return self._cosmic_result.trend_strength
            else:
                return np.array([])
                
        except Exception as e:
            self.logger.error(f"🌌 トレンド強度取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_quantum_coherence(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        🌌 量子コヒーレンス指数を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: 量子コヒーレンス指数の配列（0-1）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._cosmic_result is not None:
                return self._cosmic_result.quantum_coherence
            else:
                return np.array([])
                
        except Exception as e:
            self.logger.error(f"🌌 量子コヒーレンス取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_channel_efficiency(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        🌌 チャネル効率度を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: チャネル効率度の配列（0-1）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._cosmic_result is not None:
                return self._cosmic_result.channel_efficiency
            else:
                return np.array([])
                
        except Exception as e:
            self.logger.error(f"🌌 チャネル効率度取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_neural_adaptation_score(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        🌌 神経適応スコアを取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: 神経適応スコアの配列（0-1）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._cosmic_result is not None:
                return self._cosmic_result.adaptation_score
            else:
                return np.array([])
                
        except Exception as e:
            self.logger.error(f"🌌 神経適応スコア取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_volatility_regime(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        🌌 ボラティリティレジームを取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: ボラティリティレジーム（1-5段階）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._cosmic_result is not None:
                return self._cosmic_result.volatility_regime
            else:
                return np.array([])
                
        except Exception as e:
            self.logger.error(f"🌌 ボラティリティレジーム取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_cosmic_intelligence_report(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, Any]:
        """
        🌌 宇宙知能レポートを取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Dict[str, Any]: 宇宙知能レポート
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.cosmic_signal.get_cosmic_intelligence_report()
            
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
            if data is not None:
                self.calculate_signals(data)
            
            return self.cosmic_signal.get_current_state()
            
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