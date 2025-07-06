#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.ultra_quantum_adaptive_channel.ultra_quantum_adaptive_channel_entry import UltraQuantumAdaptiveChannelEntrySignal


@njit(fastmath=True, parallel=True)
def calculate_uqavc_exit_signals_numba(breakout_signals: np.ndarray,
                                      exit_signals: np.ndarray,
                                      current_position: int,
                                      current_index: int,
                                      tunnel_probability: np.ndarray,
                                      tunnel_threshold: float) -> bool:
    """
    UQAVC決済シグナル計算（Numba高速化版）
    
    Args:
        breakout_signals: ブレイクアウトシグナル配列
        exit_signals: UQAVCの決済シグナル配列
        current_position: 現在のポジション（1=ロング、-1=ショート）
        current_index: 現在のインデックス
        tunnel_probability: 量子トンネル確率配列
        tunnel_threshold: トンネル効果による早期決済の閾値
    
    Returns:
        bool: 決済すべきかどうか
    """
    if current_index >= len(breakout_signals):
        return False
    
    # NaN値チェック
    if np.isnan(breakout_signals[current_index]):
        return False
    
    # UQAVCの決済シグナルがある場合はそれを優先
    if exit_signals is not None and current_index < len(exit_signals):
        if not np.isnan(exit_signals[current_index]):
            if current_position == 1 and exit_signals[current_index] == 1:
                return True  # ロング決済
            elif current_position == -1 and exit_signals[current_index] == -1:
                return True  # ショート決済
    
    # 量子トンネル効果による早期決済
    if tunnel_probability is not None and current_index < len(tunnel_probability):
        if not np.isnan(tunnel_probability[current_index]):
            if tunnel_probability[current_index] > tunnel_threshold:
                return True  # 早期決済
    
    # ブレイクアウトシグナルの反転による決済
    if current_position == 1:  # ロングポジション
        if breakout_signals[current_index] == -1:  # 下抜けシグナル発生
            return True  # ロング決済
    elif current_position == -1:  # ショートポジション
        if breakout_signals[current_index] == 1:  # 上抜けシグナル発生
            return True  # ショート決済
    
    return False


@njit(fastmath=True)
def calculate_confidence_filter_numba(entry_signals: np.ndarray,
                                     entry_confidence: np.ndarray,
                                     confidence_threshold: float) -> np.ndarray:
    """
    信頼度フィルタ計算（Numba高速化版）
    
    Args:
        entry_signals: エントリーシグナル配列
        entry_confidence: エントリー信頼度配列
        confidence_threshold: 信頼度閾値
    
    Returns:
        np.ndarray: フィルタ済みエントリーシグナル
    """
    length = len(entry_signals)
    filtered_signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        if entry_confidence is not None and i < len(entry_confidence):
            if not np.isnan(entry_confidence[i]) and entry_confidence[i] >= confidence_threshold:
                filtered_signals[i] = entry_signals[i]
            else:
                filtered_signals[i] = 0
        else:
            filtered_signals[i] = entry_signals[i]
    
    return filtered_signals


class UltraQuantumAdaptiveChannelSignalGenerator(BaseSignalGenerator):
    """
    Ultra Quantum Adaptive Channelのシグナル生成クラス（両方向・高速化版）
    
    特徴:
    - 15層量子フィルタリングシステムとウェーブレット多時間軸解析
    - 量子コヒーレンス理論による市場の量子もつれ状態検出
    - ブレイクアウトシグナルの変化によるエントリー検出
    - エントリー信頼度による高精度フィルタリング
    - 量子トンネル効果による早期決済システム
    - Numbaによる高速化処理
    
    エントリー条件:
    - ロング: breakout_signals が 0 から 1 に変化 かつ entry_confidence >= threshold
    - ショート: breakout_signals が 0 から -1 に変化 かつ entry_confidence >= threshold
    
    エグジット条件:
    - UQAVCの決済シグナルによる決済
    - 量子トンネル効果による早期決済（tunnel_probability > threshold）
    - ブレイクアウトシグナルの反転による決済
    """
    
    def __init__(
        self,
        # Ultra Quantum Adaptive Volatility Channelパラメータ
        volatility_period: int = 21,
        base_multiplier: float = 2.0,
        quantum_window: int = 50,
        neural_window: int = 100,
        src_type: str = 'hlc3',
        # シグナル生成パラメータ
        confidence_threshold: float = 0.3,
        enable_exit_signals: bool = True,
        tunnel_threshold: float = 0.8,
        use_neural_adaptation: bool = True
    ):
        """初期化"""
        super().__init__("UltraQuantumAdaptiveChannelSignalGenerator")
        
        # パラメータの設定
        self._params = {
            'volatility_period': volatility_period,
            'base_multiplier': base_multiplier,
            'quantum_window': quantum_window,
            'neural_window': neural_window,
            'src_type': src_type,
            'confidence_threshold': confidence_threshold,
            'enable_exit_signals': enable_exit_signals,
            'tunnel_threshold': tunnel_threshold,
            'use_neural_adaptation': use_neural_adaptation
        }
        
        # Ultra Quantum Adaptive Channelエントリーシグナルの初期化
        self.uqavc_signal = UltraQuantumAdaptiveChannelEntrySignal(
            volatility_period=volatility_period,
            base_multiplier=base_multiplier,
            quantum_window=quantum_window,
            neural_window=neural_window,
            src_type=src_type,
            confidence_threshold=confidence_threshold,
            enable_exit_signals=enable_exit_signals,
            tunnel_threshold=tunnel_threshold,
            use_neural_adaptation=use_neural_adaptation
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._breakout_signals = None
        self._exit_signals = None
        self._uqavc_result = None
        self._current_position = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                # Ultra Quantum Adaptive Channelシグナルの計算
                try:
                    # エントリーシグナルの計算
                    uqavc_entry_signals = self.uqavc_signal.generate(df)
                    
                    # UQAVCの結果を取得
                    self._uqavc_result = self.uqavc_signal.get_uqavc_result(df)
                    
                    # ブレイクアウトシグナルと決済シグナルを取得
                    self._breakout_signals = self.uqavc_signal.get_breakout_signals(df)
                    self._exit_signals = self.uqavc_signal.get_exit_signals(df)
                    self._current_position = self.uqavc_signal.get_current_position(df)
                    
                    # エントリーシグナルを設定
                    self._signals = uqavc_entry_signals
                    
                except Exception as e:
                    self.logger.error(f"Ultra Quantum Adaptive Channelシグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._breakout_signals = np.zeros(current_len, dtype=np.int8)
                    self._exit_signals = np.zeros(current_len, dtype=np.int8)
                    self._current_position = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._breakout_signals = np.zeros(len(data), dtype=np.int8)
                self._exit_signals = np.zeros(len(data), dtype=np.int8)
                self._current_position = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # Numba高速化された決済シグナル計算
        if self._breakout_signals is not None and self._uqavc_result is not None:
            return calculate_uqavc_exit_signals_numba(
                self._breakout_signals,
                self._exit_signals,
                position,
                index,
                self._uqavc_result.tunnel_probability if hasattr(self._uqavc_result, 'tunnel_probability') else None,
                self._params['tunnel_threshold']
            )
        
        return False
    
    def get_breakout_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ブレイクアウトシグナルを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ブレイクアウトシグナルの配列（1=上抜け、-1=下抜け、0=なし）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._breakout_signals is not None:
                return self._breakout_signals.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"ブレイクアウトシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_entry_confidence(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        エントリー信頼度を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: エントリー信頼度の配列（0-1の範囲）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.uqavc_signal.get_entry_confidence(data)
        except Exception as e:
            self.logger.error(f"エントリー信頼度取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_quantum_analysis(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        量子解析データを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: 量子解析データ
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.uqavc_signal.get_quantum_analysis(data)
        except Exception as e:
            self.logger.error(f"量子解析データ取得中にエラー: {str(e)}")
            return {}
    
    def get_neural_analysis(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        神経回路網解析データを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: 神経回路網解析データ
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.uqavc_signal.get_neural_analysis(data)
        except Exception as e:
            self.logger.error(f"神経回路網解析データ取得中にエラー: {str(e)}")
            return {}
    
    def get_channel_data(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        チャネルデータを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: チャネルデータ
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.uqavc_signal.get_channel_data(data)
        except Exception as e:
            self.logger.error(f"チャネルデータ取得中にエラー: {str(e)}")
            return {}
    
    def get_current_position(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        現在のポジション状態を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ポジション配列（1=ロング、-1=ショート、0=ポジションなし）
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._current_position is not None:
                return self._current_position.copy()
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"現在のポジション取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_market_intelligence_report(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        市場知能レポートを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: 市場知能レポート
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            return self.uqavc_signal.get_market_intelligence_report(data)
        except Exception as e:
            self.logger.error(f"市場知能レポート取得中にエラー: {str(e)}")
            return {}
    
    def get_all_uqavc_stages(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        UQAVCの全段階の結果を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: 全段階の結果
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._uqavc_result is not None:
                return {
                    'upper_channel': self._uqavc_result.upper_channel.copy(),
                    'lower_channel': self._uqavc_result.lower_channel.copy(),
                    'midline': self._uqavc_result.midline.copy(), 
                    'dynamic_width': self._uqavc_result.dynamic_width.copy(),
                    'breakout_signals': self._uqavc_result.breakout_signals.copy(),
                    'entry_confidence': self._uqavc_result.entry_confidence.copy(),
                    'exit_signals': self._uqavc_result.exit_signals.copy(),
                    'quantum_coherence': self._uqavc_result.quantum_coherence.copy(),
                    'entanglement_strength': self._uqavc_result.entanglement_strength.copy(),
                    'tunnel_probability': self._uqavc_result.tunnel_probability.copy(),
                    'wave_interference': self._uqavc_result.wave_interference.copy(),
                    'neural_weight': self._uqavc_result.neural_weight.copy(),
                    'learning_rate': self._uqavc_result.learning_rate.copy(),
                    'adaptation_score': self._uqavc_result.adaptation_score.copy(),
                    'memory_state': self._uqavc_result.memory_state.copy(),
                    'current_phase': self._uqavc_result.current_phase,
                    'current_coherence': self._uqavc_result.current_coherence,
                    'current_flow_state': self._uqavc_result.current_flow_state,
                    'market_intelligence': self._uqavc_result.market_intelligence
                }
            else:
                return {}
        except Exception as e:
            self.logger.error(f"UQAVC全段階結果取得中にエラー: {str(e)}")
            return {}
    
    def get_confidence_filtered_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        信頼度フィルタ済みシグナルを取得（高速化版）
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 信頼度フィルタ済みエントリーシグナル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
            
            if self._signals is not None and self._uqavc_result is not None:
                return calculate_confidence_filter_numba(
                    self._signals,
                    self._uqavc_result.entry_confidence,
                    self._params['confidence_threshold']
                )
            else:
                return np.array([])
        except Exception as e:
            self.logger.error(f"信頼度フィルタ済みシグナル取得中にエラー: {str(e)}")
            return np.array([]) 