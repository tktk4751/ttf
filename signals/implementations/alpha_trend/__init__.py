"""
アルファトレンドシグナルパッケージ

AlphaTrendを使用した高度な方向シグナルとエントリーシグナルの実装。
効率比（ER）に基づいた動的パラメータ最適化とNumba JITコンパイルによる高速化。
"""

from .direction import AlphaTrendDirectionSignal
from .breakout_entry import AlphaTrendBreakoutEntrySignal

__all__ = [
    'AlphaTrendDirectionSignal',
    'AlphaTrendBreakoutEntrySignal'
] 