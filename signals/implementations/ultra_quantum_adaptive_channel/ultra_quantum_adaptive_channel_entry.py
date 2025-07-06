#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.ultra_quantum_adaptive_channel import UltraQuantumAdaptiveVolatilityChannel


@njit(fastmath=True, parallel=True)
def calculate_uqavc_entry_signals(breakout_signals: np.ndarray, 
                                 confidence_threshold: float = 0.3,
                                 entry_confidence: np.ndarray = None) -> np.ndarray:
    """
    Ultra Quantum Adaptive Channelのエントリーシグナルを計算する（高速化版）
    
    Args:
        breakout_signals: ブレイクアウトシグナル配列（1=上抜け、-1=下抜け、0=なし）
        confidence_threshold: 信頼度閾値（デフォルト: 0.3）
        entry_confidence: エントリー信頼度配列（0-1の範囲）
    
    Returns:
        シグナルの配列（1=ロング、-1=ショート、0=シグナルなし）
    """
    length = len(breakout_signals)
    signals = np.zeros(length, dtype=np.int8)
    
    # エントリーシグナル判定（並列処理化）
    for i in prange(1, length):  # i=0は前の値がないためスキップ
        # NaN値チェック
        if np.isnan(breakout_signals[i]) or np.isnan(breakout_signals[i-1]):
            signals[i] = 0
            continue
        
        # 信頼度チェック（指定されている場合）
        confidence_ok = True
        if entry_confidence is not None:
            if np.isnan(entry_confidence[i]) or entry_confidence[i] < confidence_threshold:
                confidence_ok = False
        
        if confidence_ok:
            # ロングエントリー: breakout_signals が 0 から 1 に変化
            if breakout_signals[i-1] == 0 and breakout_signals[i] == 1:
                signals[i] = 1
            # ショートエントリー: breakout_signals が 0 から -1 に変化
            elif breakout_signals[i-1] == 0 and breakout_signals[i] == -1:
                signals[i] = -1
            else:
                signals[i] = 0
        else:
            signals[i] = 0
    
    return signals


@njit(fastmath=True, parallel=True)
def calculate_uqavc_exit_signals(breakout_signals: np.ndarray,
                                exit_signals: np.ndarray = None,
                                current_position: np.ndarray = None,
                                tunnel_probability: np.ndarray = None,
                                tunnel_threshold: float = 0.8) -> np.ndarray:
    """
    Ultra Quantum Adaptive Channelの決済シグナルを計算する（高速化版）
    
    Args:
        breakout_signals: ブレイクアウトシグナル配列
        exit_signals: UQAVCの決済シグナル配列
        current_position: 現在のポジション（1=ロング、-1=ショート、0=ポジションなし）
        tunnel_probability: 量子トンネル確率配列
        tunnel_threshold: トンネル効果による早期決済の閾値（デフォルト: 0.8）
    
    Returns:
        決済シグナルの配列（1=ロング決済、-1=ショート決済、0=決済なし）
    """
    length = len(breakout_signals)
    exit_signals_out = np.zeros(length, dtype=np.int8)
    
    if current_position is None:
        return exit_signals_out
    
    # 決済シグナル判定（並列処理化）
    for i in prange(length):
        # NaN値チェック
        if np.isnan(breakout_signals[i]):
            continue
        
        # UQAVCの決済シグナルがある場合はそれを優先
        if exit_signals is not None and not np.isnan(exit_signals[i]):
            if current_position[i] == 1 and exit_signals[i] == 1:
                exit_signals_out[i] = 1  # ロング決済
            elif current_position[i] == -1 and exit_signals[i] == -1:
                exit_signals_out[i] = -1  # ショート決済
            continue
        
        # 量子トンネル効果による早期決済
        if tunnel_probability is not None and not np.isnan(tunnel_probability[i]):
            if tunnel_probability[i] > tunnel_threshold:
                if current_position[i] == 1:
                    exit_signals_out[i] = 1  # ロング決済
                elif current_position[i] == -1:
                    exit_signals_out[i] = -1  # ショート決済
                continue
        
        # ブレイクアウトシグナルの反転による決済
        if current_position[i] == 1:  # ロングポジション
            if breakout_signals[i] == -1:  # 下抜けシグナル発生
                exit_signals_out[i] = 1  # ロング決済
        elif current_position[i] == -1:  # ショートポジション
            if breakout_signals[i] == 1:  # 上抜けシグナル発生
                exit_signals_out[i] = -1  # ショート決済
    
    return exit_signals_out


@njit(fastmath=True)
def track_position_with_signals(entry_signals: np.ndarray, exit_signals: np.ndarray) -> np.ndarray:
    """
    エントリーシグナルと決済シグナルからポジションを追跡する（高速化版）
    
    Args:
        entry_signals: エントリーシグナル（1=ロング、-1=ショート、0=なし）
        exit_signals: 決済シグナル（1=ロング決済、-1=ショート決済、0=決済なし）
    
    Returns:
        ポジション配列（1=ロング、-1=ショート、0=ポジションなし）
    """
    length = len(entry_signals)
    position = np.zeros(length, dtype=np.int8)
    current_pos = 0
    
    for i in range(length):
        # 決済シグナルチェック
        if exit_signals[i] == 1 and current_pos == 1:  # ロング決済
            current_pos = 0
        elif exit_signals[i] == -1 and current_pos == -1:  # ショート決済
            current_pos = 0
        
        # エントリーシグナルチェック
        if entry_signals[i] == 1:  # ロングエントリー
            current_pos = 1
        elif entry_signals[i] == -1:  # ショートエントリー
            current_pos = -1
        
        position[i] = current_pos
    
    return position


class UltraQuantumAdaptiveChannelEntrySignal(BaseSignal, IEntrySignal):
    """
    Ultra Quantum Adaptive Volatility Channelによるエントリーシグナル
    
    特徴:
    - 15層量子フィルタリングシステムとウェーブレット多時間軸解析
    - 量子コヒーレンス理論による市場の量子もつれ状態検出
    - ブレイクアウトシグナルの変化（0→1、0→-1）によるエントリー検出
    - エントリー信頼度による高精度フィルタリング
    - 量子トンネル効果による早期決済システム
    - Numbaによる高速化処理
    
    エントリー条件:
    - ロング: breakout_signals が 0 から 1 に変化 かつ entry_confidence >= threshold
    - ショート: breakout_signals が 0 から -1 に変化 かつ entry_confidence >= threshold
    
    決済条件:
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
        """
        コンストラクタ
        
        Args:
            # Ultra Quantum Adaptive Volatility Channelパラメータ
            volatility_period: ボラティリティ計算期間（デフォルト: 21）
            base_multiplier: 基本チャネル幅倍率（デフォルト: 2.0）
            quantum_window: 量子解析ウィンドウ（デフォルト: 50）
            neural_window: 神経回路網ウィンドウ（デフォルト: 100）
            src_type: 価格ソースタイプ（デフォルト: 'hlc3'）
            
            # シグナル生成パラメータ
            confidence_threshold: エントリー信頼度閾値（デフォルト: 0.3）
            enable_exit_signals: 決済シグナルを有効にするか（デフォルト: True）
            tunnel_threshold: 量子トンネル効果による早期決済の閾値（デフォルト: 0.8）
            use_neural_adaptation: 神経回路網適応を使用するか（デフォルト: True）
        """
        params = {
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
        
        neural_desc = "Neural" if use_neural_adaptation else "Base"
        super().__init__(
            f"UQAVCEntry(vol={volatility_period}, mult={base_multiplier}, conf={confidence_threshold}, {neural_desc})",
            params
        )
        
        # Ultra Quantum Adaptive Volatility Channelのインスタンス化
        self._uqavc = UltraQuantumAdaptiveVolatilityChannel(
            volatility_period=volatility_period,
            base_multiplier=base_multiplier,
            quantum_window=quantum_window,
            neural_window=neural_window,
            src_type=src_type
        )
        
        # 結果キャッシュ
        self._entry_signals = None
        self._exit_signals = None
        self._data_hash = None
        self._current_position = None
        self._uqavc_result = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラム（OHLC）のハッシュ
            required_cols = ['open', 'high', 'low', 'close']
            available_cols = [col for col in required_cols if col in data.columns]
            if available_cols:
                data_hash = hash(tuple(map(tuple, data[available_cols].values)))
            else:
                # フォールバック
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                # OHLCデータの場合
                data_hash = hash(tuple(map(tuple, data)))
            else:
                # それ以外は全体をハッシュ
                data_hash = hash(tuple(map(tuple, data)) if data.ndim == 2 else tuple(data))
        
        return f"{data_hash}_{hash(frozenset(self._params.items()))}"
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ（OHLC必須）
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._entry_signals is not None:
                return self._entry_signals
                
            self._data_hash = data_hash
            
            # Ultra Quantum Adaptive Volatility Channelの計算
            uqavc_result = self._uqavc.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if uqavc_result is None:
                self._entry_signals = np.zeros(len(data), dtype=np.int8)
                return self._entry_signals
            
            self._uqavc_result = uqavc_result
            
            # ブレイクアウトシグナルとエントリー信頼度の取得
            breakout_signals = uqavc_result.breakout_signals.astype(np.int8)
            entry_confidence = uqavc_result.entry_confidence if self._params['use_neural_adaptation'] else None
            
            # エントリーシグナルの計算（高速化版）
            entry_signals = calculate_uqavc_entry_signals(
                breakout_signals,
                self._params['confidence_threshold'],
                entry_confidence
            )
            
            # 結果をキャッシュ
            self._entry_signals = entry_signals
            
            # 決済シグナルも計算（有効な場合）
            if self._params['enable_exit_signals']:
                # ポジション追跡用の決済シグナル
                exit_signals_temp = np.zeros(len(entry_signals), dtype=np.int8)
                current_position = track_position_with_signals(entry_signals, exit_signals_temp)
                
                # 実際の決済シグナル計算
                self._exit_signals = calculate_uqavc_exit_signals(
                    breakout_signals,
                    uqavc_result.exit_signals.astype(np.int8),
                    current_position,
                    uqavc_result.tunnel_probability,
                    self._params['tunnel_threshold']
                )
                
                # 最終的なポジション追跡
                self._current_position = track_position_with_signals(entry_signals, self._exit_signals)
            
            return entry_signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"UltraQuantumAdaptiveChannelEntrySignal計算中にエラー: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self._entry_signals = np.zeros(len(data), dtype=np.int8)
            return self._entry_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        決済シグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 決済シグナルの配列（1=ロング決済、-1=ショート決済、0=決済なし）
        """
        if data is not None:
            self.generate(data)
        
        if self._exit_signals is not None:
            return self._exit_signals.copy()
        else:
            return np.zeros(len(data) if data is not None else 0, dtype=np.int8)
    
    def get_current_position(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        現在のポジション状態を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ポジション配列（1=ロング、-1=ショート、0=ポジションなし）
        """
        if data is not None:
            self.generate(data)
        
        if self._current_position is not None:
            return self._current_position.copy()
        else:
            return np.zeros(len(data) if data is not None else 0, dtype=np.int8)
    
    def get_uqavc_result(self, data: Union[pd.DataFrame, np.ndarray] = None) -> object:
        """
        Ultra Quantum Adaptive Volatility Channelの計算結果を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            UQAVCResult: UQAVCの計算結果
        """
        if data is not None:
            self.generate(data)
            
        return self._uqavc_result
    
    def get_breakout_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ブレイクアウトシグナルを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: ブレイクアウトシグナルの配列（1=上抜け、-1=下抜け、0=なし）
        """
        if data is not None:
            self.generate(data)
        
        if self._uqavc_result is not None:
            return self._uqavc_result.breakout_signals.copy()
        else:
            return np.array([])
    
    def get_entry_confidence(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        エントリー信頼度を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: エントリー信頼度の配列（0-1の範囲）
        """
        if data is not None:
            self.generate(data)
        
        if self._uqavc_result is not None:
            return self._uqavc_result.entry_confidence.copy()
        else:
            return np.array([])
    
    def get_quantum_analysis(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        量子解析データを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: 量子解析データ
        """
        if data is not None:
            self.generate(data)
        
        if self._uqavc_result is not None:
            return {
                'quantum_coherence': self._uqavc_result.quantum_coherence.copy(),
                'entanglement_strength': self._uqavc_result.entanglement_strength.copy(),
                'tunnel_probability': self._uqavc_result.tunnel_probability.copy(),
                'wave_interference': self._uqavc_result.wave_interference.copy()
            }
        else:
            return {}
    
    def get_neural_analysis(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        神経回路網解析データを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: 神経回路網解析データ
        """
        if data is not None:
            self.generate(data)
        
        if self._uqavc_result is not None:
            return {
                'neural_weight': self._uqavc_result.neural_weight.copy(),
                'learning_rate': self._uqavc_result.learning_rate.copy(),
                'adaptation_score': self._uqavc_result.adaptation_score.copy(),
                'memory_state': self._uqavc_result.memory_state.copy()
            }
        else:
            return {}
    
    def get_channel_data(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        チャネルデータを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: チャネルデータ
        """
        if data is not None:
            self.generate(data)
        
        if self._uqavc_result is not None:
            return {
                'upper_channel': self._uqavc_result.upper_channel.copy(),
                'lower_channel': self._uqavc_result.lower_channel.copy(),
                'midline': self._uqavc_result.midline.copy(),
                'dynamic_width': self._uqavc_result.dynamic_width.copy()
            }
        else:
            return {}
    
    def get_market_intelligence_report(self, data: Union[pd.DataFrame, np.ndarray] = None) -> dict:
        """
        市場知能レポートを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            dict: 市場知能レポート
        """
        if data is not None:
            self.generate(data)
        
        if self._uqavc_result is not None:
            # UQAVCの市場知能レポートを取得
            base_report = {
                'current_phase': self._uqavc_result.current_phase,
                'current_coherence': self._uqavc_result.current_coherence,
                'current_flow_state': self._uqavc_result.current_flow_state,
                'market_intelligence': self._uqavc_result.market_intelligence
            }
            
            # シグナル統計を追加
            if self._entry_signals is not None:
                total_entry_signals = int(np.sum(np.abs(self._entry_signals)))
                long_signals = int(np.sum(self._entry_signals == 1))
                short_signals = int(np.sum(self._entry_signals == -1))
                
                base_report.update({
                    'total_entry_signals': total_entry_signals,
                    'long_entry_signals': long_signals,
                    'short_entry_signals': short_signals,
                    'signal_balance': (long_signals - short_signals) / max(1, total_entry_signals)
                })
            
            return base_report
        else:
            return {}
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self._uqavc.reset() if hasattr(self._uqavc, 'reset') else None
        self._entry_signals = None
        self._exit_signals = None
        self._current_position = None
        self._data_hash = None
        self._uqavc_result = None 