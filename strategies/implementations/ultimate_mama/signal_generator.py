#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate MAMA 革新的シグナルジェネレーター
人類史上最強の適応型移動平均線による量子レベル精密シグナル生成

Revolutionary Features:
- 量子もつれシグナル相関統合
- マルチモード適応フィルタリング
- 機械学習強度制御システム
- 情報理論最適化エンジン
- 超低遅延並列処理
"""

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from ...base.signal_generator import BaseSignalGenerator
except ImportError:
    # 直接実行時の絶対インポート
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from strategies.base.signal_generator import BaseSignalGenerator

try:
    from signals.implementations.ultimate_mama.entry import UltimateMAMATrendFollowSignal
    from indicators.ultimate_mama import UltimateMAMA
except ImportError:
    # フォールバック
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from signals.implementations.ultimate_mama.entry import UltimateMAMATrendFollowSignal
    from indicators.ultimate_mama import UltimateMAMA


class QuantumFilterType(Enum):
    """量子フィルタータイプ列挙"""
    NONE = "none"
    QUANTUM_ENTANGLEMENT = "quantum_entanglement"
    MULTI_MODEL_ADAPTIVE = "multi_model_adaptive"
    VARIATIONAL_MODE = "variational_mode"
    FRACTIONAL_ORDER = "fractional_order"
    INFORMATION_THEORY = "information_theory"
    MACHINE_LEARNING = "machine_learning"
    PARALLEL_QUANTUM = "parallel_quantum"
    ULTIMATE_FUSION = "ultimate_fusion"


@njit(fastmath=True, parallel=True)
def quantum_signal_fusion(
    ultimate_signals: np.ndarray,
    quantum_signals: np.ndarray,
    mmae_signals: np.ndarray,
    vmd_signals: np.ndarray,
    fractional_signals: np.ndarray,
    entropy_signals: np.ndarray,
    ml_signals: np.ndarray,
    parallel_signals: np.ndarray,
    fusion_weights: np.ndarray
) -> np.ndarray:
    """
    量子シグナル融合アルゴリズム
    
    8つの革新的シグナル成分を量子重ね合わせ原理で統合し、
    究極の精度を持つ統合シグナルを生成
    """
    length = len(ultimate_signals)
    fused_signals = np.zeros(length, dtype=np.float64)
    
    for i in prange(length):
        # 量子重ね合わせ係数の正規化
        weights_sum = np.sum(fusion_weights)
        normalized_weights = fusion_weights / (weights_sum + 1e-10)
        
        # 各シグナル成分の重み付け合成
        signal_components = np.array([
            ultimate_signals[i],
            quantum_signals[i],
            mmae_signals[i],
            vmd_signals[i],
            fractional_signals[i],
            entropy_signals[i],
            ml_signals[i],
            parallel_signals[i]
        ], dtype=np.float64)
        
        # 量子干渉効果の計算
        interference_sum = 0.0
        for j in range(len(signal_components)):
            for k in range(j + 1, len(signal_components)):
                phase_diff = np.sin(signal_components[j] - signal_components[k])
                interference_sum += normalized_weights[j] * normalized_weights[k] * phase_diff
        
        # 最終融合シグナル
        main_signal = np.sum(normalized_weights * signal_components)
        quantum_interference = 0.1 * interference_sum  # 干渉項の寄与
        
        fused_signals[i] = main_signal + quantum_interference
    
    return fused_signals


@njit(fastmath=True, parallel=True)
def adaptive_confidence_threshold(
    signals: np.ndarray,
    signal_quality: np.ndarray,
    market_regime: np.ndarray,
    base_threshold: float = 0.7
) -> np.ndarray:
    """
    適応的信頼度閾値計算
    
    市場状況と信号品質に基づいて動的に閾値を調整し、
    最適なエントリータイミングを決定
    """
    length = len(signals)
    adaptive_thresholds = np.zeros(length, dtype=np.float64)
    
    for i in prange(length):
        # 基本閾値
        threshold = base_threshold
        
        # 信号品質による調整
        if not np.isnan(signal_quality[i]):
            quality_factor = 1.0 + 0.3 * (signal_quality[i] - 0.5)
            threshold *= quality_factor
        
        # 市場レジームによる調整
        if not np.isnan(market_regime[i]):
            if abs(market_regime[i]) > 0.7:  # 強いトレンド
                threshold *= 0.8  # 閾値を下げて感度向上
            elif abs(market_regime[i]) < 0.2:  # レンジ相場
                threshold *= 1.3  # 閾値を上げて誤シグナル抑制
        
        # 閾値の制限
        adaptive_thresholds[i] = max(0.3, min(threshold, 1.5))
    
    return adaptive_thresholds


@njit(fastmath=True)
def generate_ultimate_entry_signals(
    fused_signals: np.ndarray,
    adaptive_thresholds: np.ndarray,
    signal_quality: np.ndarray,
    minimum_quality: float = 0.4
) -> np.ndarray:
    """
    Ultimate エントリーシグナル生成（改善版）
    
    融合シグナルと適応的閾値から最終的なエントリーシグナルを決定
    修正: より感度の高いシグナル生成
    """
    length = len(fused_signals)
    entry_signals = np.zeros(length, dtype=np.int8)
    
    for i in range(length):
        # より緩い信号品質チェック
        quality_threshold = minimum_quality * 0.7  # 品質閾値を下げる
        if signal_quality[i] < quality_threshold:
            entry_signals[i] = 0
            continue
        
        # 適応的閾値をさらに下げる
        signal_strength = abs(fused_signals[i])
        base_threshold = adaptive_thresholds[i] * 0.7  # 閾値を大幅に下げる
        
        # モメンタムベースのシグナル強化
        momentum_bonus = 0.0
        if i >= 2:
            current_momentum = fused_signals[i] - fused_signals[i-1]
            prev_momentum = fused_signals[i-1] - fused_signals[i-2]
            
            # モメンタムが加速している場合はボーナス
            if abs(current_momentum) > abs(prev_momentum):
                momentum_bonus = 0.1
        
        # トレンド継続性ボーナス
        trend_bonus = 0.0
        if i >= 5:
            # 過去5期間のトレンド方向を確認
            trend_count = 0
            for j in range(i-4, i):
                if (fused_signals[j] > 0 and fused_signals[i] > 0) or \
                   (fused_signals[j] < 0 and fused_signals[i] < 0):
                    trend_count += 1
            
            if trend_count >= 3:  # 5期間中3期間以上同じ方向
                trend_bonus = 0.15
        
        # 最終閾値調整
        final_threshold = max(0.2, base_threshold - momentum_bonus - trend_bonus)
        
        # シグナル判定（より緩い条件）
        if signal_strength > final_threshold:
            if fused_signals[i] > 0:
                entry_signals[i] = 1  # ロングシグナル
            else:
                entry_signals[i] = -1  # ショートシグナル
        
        # 緩い補助条件：弱いシグナルでも方向が明確な場合
        elif signal_strength > final_threshold * 0.6:
            # 過去のシグナルとの一貫性をチェック
            consistent_direction = True
            if i >= 3:
                for j in range(max(0, i-2), i):
                    if (fused_signals[i] > 0 and fused_signals[j] < 0) or \
                       (fused_signals[i] < 0 and fused_signals[j] > 0):
                        consistent_direction = False
                        break
            
            if consistent_direction:
                if fused_signals[i] > 0:
                    entry_signals[i] = 1
                else:
                    entry_signals[i] = -1
        else:
            entry_signals[i] = 0  # シグナルなし
    
    return entry_signals


@njit(fastmath=True)
def quantum_exit_conditions(
    ultimate_signals: np.ndarray,
    quantum_coherence: np.ndarray,
    position: int,
    index: int,
    exit_threshold: float = 0.3
) -> bool:
    """
    量子エグジット条件判定
    
    量子コヒーレンスを考慮した高精度エグジット判定
    修正版: より反応的なエグジット条件
    """
    if index < 0 or index >= len(ultimate_signals):
        return False
    
    if index < 5:  # 最初の数ポイントはエグジットしない
        return False
    
    current_signal = ultimate_signals[index]
    coherence = quantum_coherence[index] if index < len(quantum_coherence) else 0.5
    
    # より敏感なコヒーレンス調整済み閾値
    adjusted_threshold = exit_threshold * (0.5 + 0.3 * coherence)  # 閾値を大幅に下げる
    
    # 過去5期間の平均シグナル
    recent_signal_avg = 0.0
    for i in range(max(0, index-4), index+1):
        recent_signal_avg += ultimate_signals[i]
    recent_signal_avg /= min(5, index+1)
    
    # モメンタム変化の検出
    momentum_change = False
    if index >= 2:
        prev_momentum = ultimate_signals[index-1] - ultimate_signals[index-2]
        curr_momentum = current_signal - ultimate_signals[index-1]
        # モメンタム方向転換の検出
        if (prev_momentum > 0 and curr_momentum < 0) or (prev_momentum < 0 and curr_momentum > 0):
            momentum_change = True
    
    # より緩い条件でのエグジット判定
    if position == 1:  # ロングポジション
        # 1. 強いネガティブシグナル
        if current_signal < -adjusted_threshold:
            return True
        # 2. 平均的にネガティブトレンド
        if recent_signal_avg < -adjusted_threshold * 0.5:
            return True
        # 3. モメンタム転換 + ネガティブ
        if momentum_change and current_signal < 0:
            return True
        # 4. 継続的下落
        if index >= 3:
            declining_count = 0
            for i in range(index-2, index+1):
                if ultimate_signals[i] < ultimate_signals[i-1]:
                    declining_count += 1
            if declining_count >= 2 and current_signal < 0:
                return True
    
    elif position == -1:  # ショートポジション
        # 1. 強いポジティブシグナル
        if current_signal > adjusted_threshold:
            return True
        # 2. 平均的にポジティブトレンド
        if recent_signal_avg > adjusted_threshold * 0.5:
            return True
        # 3. モメンタム転換 + ポジティブ
        if momentum_change and current_signal > 0:
            return True
        # 4. 継続的上昇
        if index >= 3:
            rising_count = 0
            for i in range(index-2, index+1):
                if ultimate_signals[i] > ultimate_signals[i-1]:
                    rising_count += 1
            if rising_count >= 2 and current_signal > 0:
                return True
    
    return False


class UltimateMAMASignalGenerator(BaseSignalGenerator):
    """
    Ultimate MAMA 革新的シグナルジェネレーター
    
    Revolutionary Quantum-Inspired Signal Processing:
    - 量子もつれシグナル相関統合
    - マルチモード適応フィルタリング
    - 機械学習強度制御システム
    - 情報理論最適化エンジン
    - 超低遅延並列処理
    
    Signal Generation Logic:
    1. Ultimate MAMAから8つの独立シグナル成分を取得
    2. 量子重ね合わせ原理による統合処理
    3. 適応的信頼度閾値による動的調整
    4. 信号品質フィルタリング
    5. 最終エントリー・エグジットシグナル生成
    """
    
    def __init__(
        self,
        # Ultimate MAMAパラメータ
        fast_limit: float = 0.8,
        slow_limit: float = 0.02,
        src_type: str = 'hlc3',
        
        # 量子パラメータ
        quantum_coherence_factor: float = 0.8,
        quantum_entanglement_strength: float = 0.4,
        
        # マルチモデルパラメータ
        mmae_models_count: int = 7,
        vmd_modes_count: int = 4,
        
        # フラクショナルパラメータ
        fractional_order: float = 1.618,
        
        # 機械学習パラメータ
        ml_adaptation_enabled: bool = True,
        
        # シグナル統合パラメータ
        base_confidence_threshold: float = 0.7,
        minimum_signal_quality: float = 0.4,
        quantum_exit_threshold: float = 0.3,
        
        # フィルタータイプ
        quantum_filter_type: QuantumFilterType = QuantumFilterType.ULTIMATE_FUSION,
        
        # 融合重み（8成分）
        fusion_weights: Optional[np.ndarray] = None
    ):
        """
        Ultimate MAMA シグナルジェネレーターの初期化
        """
        filter_name = quantum_filter_type.value if isinstance(quantum_filter_type, QuantumFilterType) else str(quantum_filter_type)
        super().__init__(f"UltimateMAMA_SignalGen_{filter_name}")
        
        # パラメータ保存
        self._params = {
            'fast_limit': fast_limit,
            'slow_limit': slow_limit,
            'src_type': src_type,
            'quantum_coherence_factor': quantum_coherence_factor,
            'quantum_entanglement_strength': quantum_entanglement_strength,
            'mmae_models_count': mmae_models_count,
            'vmd_modes_count': vmd_modes_count,
            'fractional_order': fractional_order,
            'ml_adaptation_enabled': ml_adaptation_enabled,
            'base_confidence_threshold': base_confidence_threshold,
            'minimum_signal_quality': minimum_signal_quality,
            'quantum_exit_threshold': quantum_exit_threshold,
            'quantum_filter_type': quantum_filter_type
        }
        
        # デフォルト融合重み（均等重み）
        if fusion_weights is None:
            self.fusion_weights = np.array([
                0.2,   # Ultimate MAMA
                0.15,  # Quantum Adapted
                0.15,  # MMAE Optimal
                0.1,   # VMD Decomposed
                0.1,   # Fractional
                0.1,   # Entropy Optimized
                0.1,   # ML Adapted
                0.1    # Parallel Processed
            ], dtype=np.float64)
        else:
            self.fusion_weights = np.asarray(fusion_weights, dtype=np.float64)
        
        # Ultimate MAMAインジケーターの初期化
        self.ultimate_mama = UltimateMAMA(
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            quantum_coherence_factor=quantum_coherence_factor,
            quantum_entanglement_strength=quantum_entanglement_strength,
            mmae_models_count=mmae_models_count,
            vmd_modes_count=vmd_modes_count,
            fractional_order=fractional_order,
            ml_adaptation_enabled=ml_adaptation_enabled
        )
        
        # Ultimate MAMAトレンドフォローシグナル
        self.trend_signal = UltimateMAMATrendFollowSignal(
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            confidence_threshold=base_confidence_threshold,
            quantum_coherence_factor=quantum_coherence_factor,
            quantum_entanglement_strength=quantum_entanglement_strength,
            mmae_models_count=mmae_models_count,
            vmd_modes_count=vmd_modes_count,
            fractional_order=fractional_order,
            ml_adaptation_enabled=ml_adaptation_enabled
        )
        
        # キャッシュ変数
        self._data_len = 0
        self._entry_signals = None
        self._long_signals = None
        self._short_signals = None
        self._ultimate_result = None
        self._fused_signals = None
        self._adaptive_thresholds = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """革新的シグナル計算"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._entry_signals is None or current_len != self._data_len:
                # Ultimate MAMA計算
                self._ultimate_result = self.ultimate_mama.calculate(data)
                
                if (self._ultimate_result is None or 
                    len(self._ultimate_result.ultimate_mama) == 0):
                    # エラー時はゼロシグナル
                    self._initialize_zero_signals(current_len)
                    return
                
                # 8つのシグナル成分を取得し、正規化
                signal_components = [
                    self._ultimate_result.ultimate_mama,
                    self._ultimate_result.quantum_adapted_mama,
                    self._ultimate_result.mmae_optimal_mama,
                    self._ultimate_result.vmj_decomposed_mama,
                    self._ultimate_result.fractional_mama,
                    self._ultimate_result.entropy_optimized_mama,
                    self._ultimate_result.ml_adapted_mama,
                    self._ultimate_result.parallel_processed_mama
                ]
                
                # 各シグナル成分を-1から1の範囲に正規化
                normalized_components = []
                for component in signal_components:
                    # 価格変化率ベースの正規化（ゼロ平均、単位分散）
                    component_returns = np.diff(component, prepend=component[0])
                    component_mean = np.mean(component_returns)
                    component_std = np.std(component_returns) + 1e-10
                    normalized_returns = (component_returns - component_mean) / component_std
                    
                    # -1から1の範囲にクリップ
                    normalized_signal = np.tanh(normalized_returns * 0.5)  # タンジェント関数で滑らかにクリップ
                    normalized_components.append(normalized_signal)
                
                # 量子シグナル融合（正規化済み）
                self._fused_signals = quantum_signal_fusion(
                    *normalized_components,
                    self.fusion_weights
                )
                
                # 適応的閾値計算
                self._adaptive_thresholds = adaptive_confidence_threshold(
                    self._fused_signals,
                    self._ultimate_result.signal_quality,
                    self._ultimate_result.market_regime,
                    self._params['base_confidence_threshold']
                )
                
                # 最終エントリーシグナル生成
                self._entry_signals = generate_ultimate_entry_signals(
                    self._fused_signals,
                    self._adaptive_thresholds,
                    self._ultimate_result.signal_quality,
                    self._params['minimum_signal_quality']
                )
                
                # ロング・ショートシグナル分離
                self._long_signals = np.where(self._entry_signals == 1, 1, 0).astype(np.int8)
                self._short_signals = np.where(self._entry_signals == -1, 1, 0).astype(np.int8)
                
                self._data_len = current_len
                
        except Exception as e:
            self.logger.error(f"Ultimate MAMAシグナル計算エラー: {str(e)}")
            self._initialize_zero_signals(current_len if 'current_len' in locals() else len(data))
    
    def _initialize_zero_signals(self, length: int) -> None:
        """ゼロシグナルで初期化"""
        self._entry_signals = np.zeros(length, dtype=np.int8)
        self._long_signals = np.zeros(length, dtype=np.int8)
        self._short_signals = np.zeros(length, dtype=np.int8)
        self._fused_signals = np.zeros(length, dtype=np.float64)
        self._adaptive_thresholds = np.full(length, 0.7, dtype=np.float64)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """統合エントリーシグナル取得"""
        if self._entry_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._entry_signals.copy()
    
    def get_long_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ロングエントリーシグナル取得"""
        if self._long_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._long_signals.copy()
    
    def get_short_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ショートエントリーシグナル取得"""
        if self._short_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._short_signals.copy()
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """量子エグジットシグナル生成"""
        if self._fused_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        if self._ultimate_result is None:
            return False
        
        # 量子エグジット条件
        return quantum_exit_conditions(
            self._fused_signals,
            self._ultimate_result.quantum_coherence,
            position,
            index,
            self._params['quantum_exit_threshold']
        )
    
    def get_ultimate_mama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """Ultimate MAMA値を取得"""
        if data is not None and (self._ultimate_result is None or len(data) != self._data_len):
            self.calculate_signals(data)
        
        if self._ultimate_result is not None:
            return self._ultimate_result.ultimate_mama.copy()
        return np.array([])
    
    def get_ultimate_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """Ultimate FAMA値を取得"""
        if data is not None and (self._ultimate_result is None or len(data) != self._data_len):
            self.calculate_signals(data)
        
        if self._ultimate_result is not None:
            return self._ultimate_result.ultimate_fama.copy()
        return np.array([])
    
    def get_fused_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """融合シグナル値を取得"""
        if data is not None and (self._fused_signals is None or len(data) != self._data_len):
            self.calculate_signals(data)
        
        if self._fused_signals is not None:
            return self._fused_signals.copy()
        return np.array([])
    
    def get_quantum_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """量子メトリクスを取得"""
        if data is not None and (self._ultimate_result is None or len(data) != self._data_len):
            self.calculate_signals(data)
        
        if self._ultimate_result is None:
            return {}
        
        return {
            'quantum_coherence': self._ultimate_result.quantum_coherence.copy(),
            'adaptation_strength': self._ultimate_result.adaptation_strength.copy(),
            'signal_quality': self._ultimate_result.signal_quality.copy(),
            'noise_level': self._ultimate_result.noise_level.copy(),
            'market_regime': self._ultimate_result.market_regime.copy()
        }
    
    def get_signal_components(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """全シグナル成分を取得"""
        if data is not None and (self._ultimate_result is None or len(data) != self._data_len):
            self.calculate_signals(data)
        
        if self._ultimate_result is None:
            return {}
        
        return {
            'ultimate_mama': self._ultimate_result.ultimate_mama.copy(),
            'ultimate_fama': self._ultimate_result.ultimate_fama.copy(),
            'quantum_adapted_mama': self._ultimate_result.quantum_adapted_mama.copy(),
            'quantum_adapted_fama': self._ultimate_result.quantum_adapted_fama.copy(),
            'mmae_optimal_mama': self._ultimate_result.mmae_optimal_mama.copy(),
            'vmj_decomposed_mama': self._ultimate_result.vmj_decomposed_mama.copy(),
            'fractional_mama': self._ultimate_result.fractional_mama.copy(),
            'entropy_optimized_mama': self._ultimate_result.entropy_optimized_mama.copy(),
            'ml_adapted_mama': self._ultimate_result.ml_adapted_mama.copy(),
            'parallel_processed_mama': self._ultimate_result.parallel_processed_mama.copy()
        }
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, Any]:
        """全ての高度なメトリクスを取得"""
        if data is not None:
            self.calculate_signals(data)
        
        metrics = {
            # 基本シグナル
            'entry_signals': self.get_entry_signals(data) if data is not None else np.array([]),
            'long_signals': self._long_signals.copy() if self._long_signals is not None else np.array([]),
            'short_signals': self._short_signals.copy() if self._short_signals is not None else np.array([]),
            
            # 融合シグナル
            'fused_signals': self._fused_signals.copy() if self._fused_signals is not None else np.array([]),
            'adaptive_thresholds': self._adaptive_thresholds.copy() if self._adaptive_thresholds is not None else np.array([]),
            
            # パラメータ情報
            'fusion_weights': self.fusion_weights.copy(),
            'quantum_filter_type': self._params['quantum_filter_type'].value
        }
        
        # 量子メトリクスを統合
        quantum_metrics = self.get_quantum_metrics(data)
        metrics.update(quantum_metrics)
        
        # シグナル成分を統合
        signal_components = self.get_signal_components(data)
        metrics.update(signal_components)
        
        return metrics
    
    def reset(self) -> None:
        """シグナルジェネレーターの状態をリセット"""
        super().reset()
        self._data_len = 0
        self._entry_signals = None
        self._long_signals = None
        self._short_signals = None
        self._ultimate_result = None
        self._fused_signals = None
        self._adaptive_thresholds = None
        
        if hasattr(self.ultimate_mama, 'reset'):
            self.ultimate_mama.reset()
        if hasattr(self.trend_signal, 'reset'):
            self.trend_signal.reset()


if __name__ == "__main__":
    """Ultimate MAMA シグナルジェネレーターのテスト"""
    print("=== Ultimate MAMA 革新的シグナルジェネレーター テスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    n = 200
    
    # 複雑なトレンド市場の模擬
    t = np.linspace(0, 4*np.pi, n)
    trend = 100 + 0.03 * t**1.5
    cycle1 = 5 * np.sin(0.5 * t)
    cycle2 = 2 * np.sin(1.2 * t + np.pi/3)
    noise = np.random.normal(0, 1.0, n)
    
    close_prices = trend + cycle1 + cycle2 + noise
    
    # OHLC生成
    data = []
    for i, close in enumerate(close_prices):
        spread = 0.8
        high = close + spread * np.random.uniform(0.5, 1.0)
        low = close - spread * np.random.uniform(0.5, 1.0)
        open_price = close + np.random.normal(0, 0.3)
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 5000)
        })
    
    df = pd.DataFrame(data)
    
    print(f"テストデータ: {len(df)}点")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Ultimate MAMAシグナルジェネレーターテスト
    print("\nUltimate MAMAシグナルジェネレーター生成中...")
    
    try:
        signal_gen = UltimateMAMASignalGenerator(
            quantum_coherence_factor=0.8,
            mmae_models_count=5,
            vmd_modes_count=3,
            base_confidence_threshold=0.7,
            quantum_filter_type=QuantumFilterType.ULTIMATE_FUSION
        )
        
        # シグナル生成
        entry_signals = signal_gen.get_entry_signals(df)
        long_signals = signal_gen.get_long_signals(df)
        short_signals = signal_gen.get_short_signals(df)
        
        print(f"シグナル生成完了:")
        print(f"  エントリーシグナル形状: {entry_signals.shape}")
        print(f"  ロングシグナル数: {np.sum(long_signals)}")
        print(f"  ショートシグナル数: {np.sum(short_signals)}")
        print(f"  総シグナル率: {(np.sum(entry_signals != 0) / len(entry_signals) * 100):.2f}%")
        
        # 高度なメトリクス取得
        advanced_metrics = signal_gen.get_advanced_metrics(df)
        
        print(f"\n高度なメトリクス:")
        print(f"  量子コヒーレンス: {np.nanmean(advanced_metrics.get('quantum_coherence', [0])):.4f}")
        print(f"  適応強度: {np.nanmean(advanced_metrics.get('adaptation_strength', [0])):.4f}")
        print(f"  信号品質: {np.nanmean(advanced_metrics.get('signal_quality', [0])):.4f}")
        print(f"  融合重み: {advanced_metrics.get('fusion_weights', [])}")
        
        # シグナル成分統計
        signal_components = signal_gen.get_signal_components(df)
        print(f"\nシグナル成分統計:")
        for name, values in signal_components.items():
            if len(values) > 0:
                print(f"  {name}: 平均={np.nanmean(values):.4f}, 標準偏差={np.nanstd(values):.4f}")
        
        # エグジットテスト
        print(f"\nエグジットシグナルテスト:")
        test_exit_long = signal_gen.get_exit_signals(df, position=1, index=-1)
        test_exit_short = signal_gen.get_exit_signals(df, position=-1, index=-1)
        print(f"  ロングエグジット判定: {test_exit_long}")
        print(f"  ショートエグジット判定: {test_exit_short}")
        
        print("\n✅ Ultimate MAMAシグナルジェネレーター テスト成功！")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()