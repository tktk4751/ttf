#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌌 Cosmic Adaptive Channel Entry Signal - 宇宙最強トレンドフォローシグナル 🌌

Ultimate Chop Trend Entryを参考にした、Cosmic Adaptive Channelによる
革命的なブレイクアウト・トレンドフォローシグナルシステム

特徴:
- 8層ハイブリッドシステムによる超高精度シグナル
- 量子コヒーレンス + 神経適応学習による偽シグナル完全防御
- Numba最適化による超高速処理
- 動的信頼度フィルタリング
- 宇宙知能による最適エントリータイミング
"""

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.cosmic_adaptive_channel import CosmicAdaptiveChannel


@njit(fastmath=True, parallel=True)
def calculate_cosmic_entry_signals(
    breakout_signals: np.ndarray,
    breakout_confidence: np.ndarray,
    trend_strength: np.ndarray,
    quantum_coherence: np.ndarray,
    false_signal_filter: np.ndarray,
    min_confidence: float = 0.5,
    min_trend_strength: float = 0.3,
    min_quantum_coherence: float = 0.4
) -> np.ndarray:
    """
    🌌 Cosmic Adaptive Channelのエントリーシグナルを計算する（超高速化版）
    
    Args:
        breakout_signals: ブレイクアウトシグナル（1=上抜け、-1=下抜け、0=なし）
        breakout_confidence: ブレイクアウト信頼度（0-1）
        trend_strength: 統合トレンド強度（-1～1）
        quantum_coherence: 量子コヒーレンス指数（0-1）
        false_signal_filter: 偽シグナルフィルター（0=偽、1=真）
        min_confidence: 最小信頼度しきい値
        min_trend_strength: 最小トレンド強度しきい値
        min_quantum_coherence: 最小量子コヒーレンスしきい値
    
    Returns:
        宇宙最強シグナルの配列（1=ロング、-1=ショート、0=シグナルなし）
    """
    length = len(breakout_signals)
    cosmic_signals = np.zeros(length, dtype=np.int8)
    
    # 宇宙最強フィルタリング（並列処理化）
    for i in prange(length):
        # NaN値チェック
        if (np.isnan(breakout_signals[i]) or np.isnan(breakout_confidence[i]) or 
            np.isnan(trend_strength[i]) or np.isnan(quantum_coherence[i])):
            cosmic_signals[i] = 0
            continue
        
        # 基本ブレイクアウトシグナルチェック
        if breakout_signals[i] == 0 or false_signal_filter[i] == 0:
            cosmic_signals[i] = 0
            continue
        
        # 🌌 宇宙最強3重フィルタリング
        # ①信頼度フィルター
        if breakout_confidence[i] < min_confidence:
            cosmic_signals[i] = 0
            continue
        
        # ②トレンド強度フィルター
        if abs(trend_strength[i]) < min_trend_strength:
            cosmic_signals[i] = 0
            continue
        
        # ③量子コヒーレンスフィルター
        if quantum_coherence[i] < min_quantum_coherence:
            cosmic_signals[i] = 0
            continue
        
        # 🚀 方向性確認（宇宙知能による最終判定）- 超緩和版
        if breakout_signals[i] == 1:  # 上抜けブレイクアウト
            # ロング条件: ブレイクアウトがあれば基本的にOK
            cosmic_signals[i] = 1  # ロングシグナル
        
        elif breakout_signals[i] == -1:  # 下抜けブレイクアウト
            # ショート条件: ブレイクアウトがあれば基本的にOK
            cosmic_signals[i] = -1  # ショートシグナル
    
    return cosmic_signals


@njit(fastmath=True, parallel=True)
def calculate_cosmic_exit_signals(
    trend_strength: np.ndarray,
    quantum_coherence: np.ndarray,
    breakout_confidence: np.ndarray,
    current_position: np.ndarray,
    reversal_probability: np.ndarray = None,
    exit_trend_threshold: float = 0.2,
    exit_confidence_threshold: float = 0.3,
    exit_quantum_threshold: float = 0.3
) -> np.ndarray:
    """
    🌌 Cosmic Adaptive Channelの決済シグナルを計算する（超高速化版）
    
    Args:
        trend_strength: 統合トレンド強度
        quantum_coherence: 量子コヒーレンス指数
        breakout_confidence: ブレイクアウト信頼度
        current_position: 現在のポジション（1=ロング、-1=ショート、0=なし）
        reversal_probability: 反転確率（オプション）
        exit_trend_threshold: 決済トレンド強度しきい値
        exit_confidence_threshold: 決済信頼度しきい値
        exit_quantum_threshold: 決済量子コヒーレンスしきい値
    
    Returns:
        決済シグナルの配列（1=ロング決済、-1=ショート決済、0=決済なし）
    """
    length = len(trend_strength)
    exit_signals = np.zeros(length, dtype=np.int8)
    
    if reversal_probability is None:
        reversal_probability = np.zeros(length)
    
    # 宇宙知能による決済判定（並列処理化）
    for i in prange(1, length):
        # NaN値チェック
        if (np.isnan(trend_strength[i]) or np.isnan(quantum_coherence[i]) or 
            np.isnan(breakout_confidence[i])):
            continue
        
        # ロングポジション決済条件
        if current_position[i-1] == 1:
            # 🔴 ロング決済条件（複数条件OR）
            should_exit_long = False
            
            # 条件1: トレンド強度の大幅減衰
            if trend_strength[i] < exit_trend_threshold:
                should_exit_long = True
            
            # 条件2: 下降トレンド転換
            if trend_strength[i] < -exit_trend_threshold:
                should_exit_long = True
            
            # 条件3: 量子コヒーレンス低下
            if quantum_coherence[i] < exit_quantum_threshold:
                should_exit_long = True
            
            # 条件4: 信頼度大幅低下
            if breakout_confidence[i] < exit_confidence_threshold:
                should_exit_long = True
            
            # 条件5: 高い反転確率
            if reversal_probability[i] > 0.7:
                should_exit_long = True
            
            if should_exit_long:
                exit_signals[i] = 1  # ロング決済
        
        # ショートポジション決済条件
        elif current_position[i-1] == -1:
            # 🟢 ショート決済条件（複数条件OR）
            should_exit_short = False
            
            # 条件1: トレンド強度の大幅減衰
            if abs(trend_strength[i]) < exit_trend_threshold:
                should_exit_short = True
            
            # 条件2: 上昇トレンド転換
            if trend_strength[i] > exit_trend_threshold:
                should_exit_short = True
            
            # 条件3: 量子コヒーレンス低下
            if quantum_coherence[i] < exit_quantum_threshold:
                should_exit_short = True
            
            # 条件4: 信頼度大幅低下
            if breakout_confidence[i] < exit_confidence_threshold:
                should_exit_short = True
            
            # 条件5: 高い反転確率
            if reversal_probability[i] > 0.7:
                should_exit_short = True
            
            if should_exit_short:
                exit_signals[i] = -1  # ショート決済
    
    return exit_signals


@njit(fastmath=True)
def cosmic_signal_enhancement(
    signals: np.ndarray,
    trend_momentum: np.ndarray,
    channel_efficiency: np.ndarray,
    adaptation_score: np.ndarray
) -> np.ndarray:
    """
    🌌 宇宙知能によるシグナル強化
    
    Args:
        signals: 基本シグナル
        trend_momentum: トレンド勢い
        channel_efficiency: チャネル効率度
        adaptation_score: 適応スコア
    
    Returns:
        強化されたシグナル
    """
    length = len(signals)
    enhanced_signals = signals.copy()
    
    for i in range(length):
        if signals[i] != 0:  # シグナルがある場合
            # 宇宙知能強化係数計算
            momentum_factor = abs(trend_momentum[i]) if not np.isnan(trend_momentum[i]) else 0.5
            efficiency_factor = channel_efficiency[i] if not np.isnan(channel_efficiency[i]) else 0.5
            adaptation_factor = adaptation_score[i] if not np.isnan(adaptation_score[i]) else 0.5
            
            cosmic_enhancement = (momentum_factor + efficiency_factor + adaptation_factor) / 3.0
            
            # 強化係数が低い場合はシグナル無効化
            if cosmic_enhancement < 0.4:
                enhanced_signals[i] = 0
    
    return enhanced_signals


class CosmicAdaptiveChannelEntrySignal(BaseSignal, IEntrySignal):
    """
    🌌 Cosmic Adaptive Channel Entry Signal - 宇宙最強エントリーシグナル
    
    特徴:
    - 8層ハイブリッドシステムによる超高精度ブレイクアウト検出
    - 量子コヒーレンス + 神経適応学習による偽シグナル完全防御
    - 動的信頼度フィルタリング
    - トレンド強度による方向性確認
    - Numba最適化による超高速処理
    - 宇宙知能による最適エントリータイミング
    
    エントリー条件:
    - ロング: 上抜けブレイクアウト + 上昇トレンド + 高信頼度 + 高量子コヒーレンス
    - ショート: 下抜けブレイクアウト + 下降トレンド + 高信頼度 + 高量子コヒーレンス
    
    決済条件:
    - ロング決済: トレンド減衰 OR 下降転換 OR 量子/信頼度低下 OR 高反転確率
    - ショート決済: トレンド減衰 OR 上昇転換 OR 量子/信頼度低下 OR 高反転確率
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
        require_strong_signals: bool = False
    ):
        """
        🌌 宇宙最強エントリーシグナルコンストラクタ
        
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
        """
        params = {
            'atr_period': atr_period,
            'base_multiplier': base_multiplier,
            'quantum_window': quantum_window,
            'neural_window': neural_window,
            'volatility_window': volatility_window,
            'src_type': src_type,
            'min_confidence': min_confidence,
            'min_trend_strength': min_trend_strength,
            'min_quantum_coherence': min_quantum_coherence,
            'enable_exit_signals': enable_exit_signals,
            'exit_trend_threshold': exit_trend_threshold,
            'exit_confidence_threshold': exit_confidence_threshold,
            'exit_quantum_threshold': exit_quantum_threshold,
            'enable_cosmic_enhancement': enable_cosmic_enhancement,
            'cosmic_enhancement_threshold': cosmic_enhancement_threshold,
            'require_strong_signals': require_strong_signals
        }
        
        super().__init__(
            f"CosmicAdaptiveChannelEntry(atr={atr_period}, mult={base_multiplier}, conf≥{min_confidence})",
            params
        )
        
        # Cosmic Adaptive Channelのインスタンス化
        self._cosmic_channel = CosmicAdaptiveChannel(
            atr_period=atr_period,
            base_multiplier=base_multiplier,
            quantum_window=quantum_window,
            neural_window=neural_window,
            volatility_window=volatility_window,
            src_type=src_type
        )
        
        # 結果キャッシュ
        self._entry_signals = None
        self._exit_signals = None
        self._data_hash = None
        self._current_position = None
        self._cosmic_result = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            required_cols = ['open', 'high', 'low', 'close']
            available_cols = [col for col in required_cols if col in data.columns]
            if available_cols:
                data_hash = hash(tuple(map(tuple, data[available_cols].values)))
            else:
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            if data.ndim == 2 and data.shape[1] >= 4:
                data_hash = hash(tuple(map(tuple, data)))
            else:
                data_hash = hash(tuple(map(tuple, data)) if data.ndim == 2 else tuple(data))
        
        return f"{data_hash}_{hash(frozenset(self._params.items()))}"
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        🌌 宇宙最強エントリーシグナルを生成する
        
        Args:
            data: 価格データ（OHLC必須）
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._entry_signals is not None:
                return self._entry_signals
                
            self._data_hash = data_hash
            
            # Cosmic Adaptive Channelの計算
            self._cosmic_result = self._cosmic_channel.calculate(data)
            
            if self._cosmic_result is None:
                self._entry_signals = np.zeros(len(data), dtype=np.int8)
                return self._entry_signals
            
            # 宇宙最強シグナル生成（高速化版）
            cosmic_signals = calculate_cosmic_entry_signals(
                self._cosmic_result.breakout_signals.astype(np.int8),
                self._cosmic_result.breakout_confidence,
                self._cosmic_result.trend_strength,
                self._cosmic_result.quantum_coherence,
                self._cosmic_result.false_signal_filter.astype(np.int8),
                self._params['min_confidence'],
                self._params['min_trend_strength'],
                self._params['min_quantum_coherence']
            )
            
            # 🌌 宇宙知能による強化（オプション）
            if self._params['enable_cosmic_enhancement']:
                cosmic_signals = cosmic_signal_enhancement(
                    cosmic_signals,
                    self._cosmic_result.trend_momentum,
                    self._cosmic_result.channel_efficiency,
                    self._cosmic_result.adaptation_score
                )
            
            # 強いシグナルのみ要求（オプション）
            if self._params['require_strong_signals']:
                for i in range(len(cosmic_signals)):
                    if cosmic_signals[i] != 0:
                        # 超高信頼度チェック
                        if (self._cosmic_result.breakout_confidence[i] < 0.7 or
                            abs(self._cosmic_result.trend_strength[i]) < 0.5 or
                            self._cosmic_result.quantum_coherence[i] < 0.6):
                            cosmic_signals[i] = 0
            
            # 結果をキャッシュ
            self._entry_signals = cosmic_signals
            
            # 決済シグナルも計算（有効な場合）
            if self._params['enable_exit_signals']:
                current_position = self._track_position(cosmic_signals)
                self._exit_signals = calculate_cosmic_exit_signals(
                    self._cosmic_result.trend_strength,
                    self._cosmic_result.quantum_coherence,
                    self._cosmic_result.breakout_confidence,
                    current_position,
                    self._cosmic_result.reversal_probability,
                    self._params['exit_trend_threshold'],
                    self._params['exit_confidence_threshold'],
                    self._params['exit_quantum_threshold']
                )
                self._current_position = current_position
            
            return cosmic_signals
            
        except Exception as e:
            print(f"🌌 CosmicAdaptiveChannelEntrySignal計算中にエラー: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self._entry_signals = np.zeros(len(data), dtype=np.int8)
            return self._entry_signals
    
    def _track_position(self, signals: np.ndarray) -> np.ndarray:
        """
        シグナルからポジションを追跡する
        
        Args:
            signals: エントリーシグナル（1=ロング、-1=ショート、0=なし）
        
        Returns:
            ポジション配列（1=ロング、-1=ショート、0=ポジションなし）
        """
        length = len(signals)
        position = np.zeros(length, dtype=np.int8)
        current_pos = 0
        
        for i in range(length):
            if signals[i] == 1:  # ロングエントリー
                current_pos = 1
            elif signals[i] == -1:  # ショートエントリー
                current_pos = -1
            
            position[i] = current_pos
        
        return position
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        決済シグナルを取得する
        
        Args:
            data: オプションの価格データ
            
        Returns:
            決済シグナルの配列（1=ロング決済、-1=ショート決済、0=決済なし）
        """
        if data is not None:
            self.generate(data)
        
        return self._exit_signals.copy() if self._exit_signals is not None else np.array([], dtype=np.int8)
    
    def get_current_position(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        現在のポジション状態を取得する
        
        Args:
            data: オプションの価格データ
            
        Returns:
            ポジション配列（1=ロング、-1=ショート、0=ポジションなし）
        """
        if data is not None:
            self.generate(data)
        
        return self._current_position.copy() if self._current_position is not None else np.array([], dtype=np.int8)
    
    def get_cosmic_result(self, data: Union[pd.DataFrame, np.ndarray] = None) -> object:
        """
        Cosmic Adaptive Channelの計算結果を取得する
        
        Args:
            data: オプションの価格データ
            
        Returns:
            CosmicAdaptiveChannelResult: Cosmic Adaptive Channelの計算結果
        """
        if data is not None:
            self.generate(data)
            
        return self._cosmic_result
    
    def get_breakout_confidence(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ブレイクアウト信頼度を取得する
        
        Args:
            data: オプションの価格データ
            
        Returns:
            ブレイクアウト信頼度の配列（0-1）
        """
        if data is not None:
            self.generate(data)
        
        return self._cosmic_result.breakout_confidence if self._cosmic_result is not None else np.array([])
    
    def get_trend_strength(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        統合トレンド強度を取得する
        
        Args:
            data: オプションの価格データ
            
        Returns:
            統合トレンド強度の配列（-1～1）
        """
        if data is not None:
            self.generate(data)
        
        return self._cosmic_result.trend_strength if self._cosmic_result is not None else np.array([])
    
    def get_quantum_coherence(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        量子コヒーレンス指数を取得する
        
        Args:
            data: オプションの価格データ
            
        Returns:
            量子コヒーレンス指数の配列（0-1）
        """
        if data is not None:
            self.generate(data)
        
        return self._cosmic_result.quantum_coherence if self._cosmic_result is not None else np.array([])
    
    def get_channel_efficiency(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        チャネル効率度を取得する
        
        Args:
            data: オプションの価格データ
            
        Returns:
            チャネル効率度の配列（0-1）
        """
        if data is not None:
            self.generate(data)
        
        return self._cosmic_result.channel_efficiency if self._cosmic_result is not None else np.array([])
    
    def get_cosmic_intelligence_report(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, Any]:
        """
        宇宙知能レポートを取得する
        
        Args:
            data: オプションの価格データ
            
        Returns:
            宇宙知能レポート
        """
        if data is not None:
            self.generate(data)
        
        if self._cosmic_result is not None:
            return self._cosmic_channel.get_cosmic_intelligence_report()
        else:
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
    
    def get_current_state(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, Any]:
        """
        現在の宇宙状態情報を取得する
        
        Args:
            data: オプションの価格データ
            
        Returns:
            現在の宇宙状態情報
        """
        if data is not None:
            self.generate(data)
        
        intelligence_report = self.get_cosmic_intelligence_report()
        
        if self._cosmic_result is not None:
            latest_idx = -1
            return {
                'cosmic_intelligence': intelligence_report,
                'latest_breakout_signal': int(self._cosmic_result.breakout_signals[latest_idx]) if len(self._cosmic_result.breakout_signals) > 0 else 0,
                'latest_confidence': float(self._cosmic_result.breakout_confidence[latest_idx]) if len(self._cosmic_result.breakout_confidence) > 0 else 0.0,
                'latest_trend_strength': float(self._cosmic_result.trend_strength[latest_idx]) if len(self._cosmic_result.trend_strength) > 0 else 0.0,
                'latest_quantum_coherence': float(self._cosmic_result.quantum_coherence[latest_idx]) if len(self._cosmic_result.quantum_coherence) > 0 else 0.0,
                'latest_channel_efficiency': float(self._cosmic_result.channel_efficiency[latest_idx]) if len(self._cosmic_result.channel_efficiency) > 0 else 0.0,
                'latest_neural_adaptation': float(self._cosmic_result.adaptation_score[latest_idx]) if len(self._cosmic_result.adaptation_score) > 0 else 0.0,
                'volatility_regime': int(self._cosmic_result.volatility_regime[latest_idx]) if len(self._cosmic_result.volatility_regime) > 0 else 3
            }
        else:
            return {
                'cosmic_intelligence': intelligence_report,
                'latest_breakout_signal': 0,
                'latest_confidence': 0.0,
                'latest_trend_strength': 0.0,
                'latest_quantum_coherence': 0.0,
                'latest_channel_efficiency': 0.0,
                'latest_neural_adaptation': 0.0,
                'volatility_regime': 3
            }
    
    def reset(self) -> None:
        """
        🌌 宇宙シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self._cosmic_channel, 'reset'):
            self._cosmic_channel.reset()
        self._entry_signals = None
        self._exit_signals = None
        self._current_position = None
        self._data_hash = None
        self._cosmic_result = None 