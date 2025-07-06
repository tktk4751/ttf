#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultra Quantum Adaptive Channel Strategy Implementation

15層量子フィルタリングシステムとウェーブレット多時間軸解析による
高度な市場予測ストラテジー実装モジュール

特徴:
- 15層量子フィルタリングシステムによる高精度なシグナル検出
- ウェーブレット多時間軸解析による市場周期の捕捉
- 量子コヒーレンス理論による市場の量子もつれ状態検出
- エントリー信頼度による高精度フィルタリング
- 量子トンネル効果による早期決済システム
- 神経回路網適応による市場環境への動的対応
- Numbaによる高速化処理
"""

from .strategy import UltraQuantumAdaptiveChannelStrategy
from .signal_generator import UltraQuantumAdaptiveChannelSignalGenerator

__all__ = [
    'UltraQuantumAdaptiveChannelStrategy',
    'UltraQuantumAdaptiveChannelSignalGenerator'
]

# ストラテジー情報
STRATEGY_INFO = {
    'name': 'UltraQuantumAdaptiveChannel',
    'description': '量子フィルタリングとウェーブレット解析による高度トレーディングストラテジー',
    'version': '1.0.0',
    'author': 'TTF Strategy Team',
    'requires': ['numba', 'numpy', 'pandas'],
    'features': [
        '15層量子フィルタリングシステム',
        'ウェーブレット多時間軸解析',
        '量子コヒーレンス理論による市場分析',
        'エントリー信頼度フィルタリング',
        '量子トンネル効果による早期決済',
        '神経回路網適応システム',
        'Numba高速化処理'
    ],
    'parameters': {
        'volatility_period': {'type': 'int', 'default': 21, 'range': (10, 50), 'description': 'ボラティリティ計算期間'},
        'base_multiplier': {'type': 'float', 'default': 2.0, 'range': (1.0, 5.0), 'description': '基本チャネル幅倍率'},
        'quantum_window': {'type': 'int', 'default': 50, 'range': (20, 100), 'description': '量子解析ウィンドウ'},
        'neural_window': {'type': 'int', 'default': 100, 'range': (50, 200), 'description': '神経回路網ウィンドウ'},
        'src_type': {'type': 'str', 'default': 'hlc3', 'options': ['hlc3', 'ohlc4', 'close', 'hl2'], 'description': '価格ソースタイプ'},
        'confidence_threshold': {'type': 'float', 'default': 0.3, 'range': (0.1, 0.8), 'description': 'エントリー信頼度閾値'},
        'tunnel_threshold': {'type': 'float', 'default': 0.8, 'range': (0.5, 0.95), 'description': '量子トンネル効果閾値'},
        'use_neural_adaptation': {'type': 'bool', 'default': True, 'description': '神経回路網適応を使用'}
    }
} 