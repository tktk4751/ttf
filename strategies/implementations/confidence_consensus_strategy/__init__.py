#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
信頼度ベース・コンセンサス戦略モジュール

階層的適応型コンセンサス法による高度な信頼度算出戦略。
複数のインジケーターシグナルを重み付きで組み合わせ、
市場状況に応じて動的に調整する洗練されたアプローチ。
"""

from .strategy import (
    ConfidenceConsensusStrategy,
    ConfidenceCalculationResult,
    create_confidence_consensus_strategy
)

__all__ = [
    'ConfidenceConsensusStrategy',
    'ConfidenceCalculationResult', 
    'create_confidence_consensus_strategy'
]