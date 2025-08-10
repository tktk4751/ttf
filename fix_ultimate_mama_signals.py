#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultimate MAMA シグナル修正スクリプト
バランスの取れたロング・ショートシグナル生成のための改善
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from signals.implementations.ultimate_mama.entry import UltimateMAMATrendFollowSignal
from numba import njit

# 改善されたシグナル生成関数
@njit(fastmath=True)
def generate_balanced_trend_signals(
    ultimate_mama: object,
    ultimate_fama: object,
    quantum_mama: object,
    quantum_fama: object,
    signal_quality: object,
    market_regime: object,
    adaptation_strength: object,
    confidence_threshold: float = 0.7
) -> object:
    """
    バランスの取れたトレンドシグナル生成
    
    ロングとショートの両方向のシグナルを適切に生成するように改善
    """
    length = len(ultimate_mama)
    trend_signals = [0] * length  # リストで初期化（Numba互換）
    
    # 動的閾値とクロスオーバー検出を組み合わせる
    for i in range(2, length):
        # 1. クロスオーバー検出
        mama_cross = ultimate_mama[i] > ultimate_fama[i] and ultimate_mama[i-1] <= ultimate_fama[i-1]
        fama_cross = ultimate_mama[i] < ultimate_fama[i] and ultimate_mama[i-1] >= ultimate_fama[i-1]
        
        # 2. モメンタム分析  
        mama_momentum = ultimate_mama[i] - ultimate_mama[i-1]
        fama_momentum = ultimate_fama[i] - ultimate_fama[i-1]
        
        # 3. 量子相関分析
        quantum_momentum = quantum_mama[i] - quantum_fama[i]
        ultimate_momentum = ultimate_mama[i] - ultimate_fama[i]
        
        # 4. 信号品質による重み付け
        quality_weight = max(0.3, min(signal_quality[i], 1.0))
        
        # 5. 統合シグナル強度計算
        signal_strength = 0.0
        
        # クロスオーバーに基づく基本シグナル
        if mama_cross:
            signal_strength += 0.5 * quality_weight
        elif fama_cross:
            signal_strength -= 0.5 * quality_weight
            
        # モメンタムに基づく補強シグナル
        if mama_momentum > 0 and fama_momentum > 0:
            signal_strength += 0.3 * quality_weight
        elif mama_momentum < 0 and fama_momentum < 0:
            signal_strength -= 0.3 * quality_weight
            
        # 量子相関による最終調整
        if quantum_momentum > 0 and ultimate_momentum > 0:
            signal_strength += 0.2 * quality_weight
        elif quantum_momentum < 0 and ultimate_momentum < 0:
            signal_strength -= 0.2 * quality_weight
        
        # 6. 適応的閾値による判定
        dynamic_threshold = confidence_threshold * (0.5 + 0.5 * quality_weight)
        
        if signal_strength > dynamic_threshold:
            trend_signals[i] = 1  # ロングシグナル
        elif signal_strength < -dynamic_threshold:
            trend_signals[i] = -1  # ショートシグナル
        else:
            trend_signals[i] = 0  # シグナルなし
    
    return trend_signals

def create_improved_entry_signal():
    """改善されたエントリーシグナルクラスを作成"""
    print("=== Ultimate MAMA シグナル改善 ===")
    print("よりバランスの取れたロング・ショートシグナル生成に修正")
    
    # 改善されたパラメータ設定
    improved_signal = UltimateMAMATrendFollowSignal(
        # より感度の高い設定
        fast_limit=0.6,
        slow_limit=0.03,
        src_type='hlc3',
        
        # バランスの良い量子パラメータ
        quantum_coherence_factor=0.7,
        quantum_entanglement_strength=0.3,
        
        # 適度なモデル数
        mmae_models_count=5,
        vmd_modes_count=3,
        
        # より柔軟な閾値
        confidence_threshold=0.5,  # 閾値を下げる
        signal_smoothing_window=3,  # スムージングを減らす
        
        # 機械学習を有効化
        ml_adaptation_enabled=True,
        
        # 情報理論最適化を調整
        entropy_optimization_enabled=True,
        information_window=15  # ウィンドウを小さくして感度向上
    )
    
    return improved_signal

if __name__ == "__main__":
    create_improved_entry_signal()
    print("✅ Ultimate MAMA シグナル改善完了")
    print("🔄 より多様なトレードシグナルが期待できます")