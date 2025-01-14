#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any

from ...interfaces.parameters import IParameters

class SupertrendRsiChopParameters(IParameters):
    """スーパートレンド、RSI、Chopのパラメータ管理クラス"""
    
    def __init__(self):
        """コンストラクタ"""
        self._parameters = {
            'supertrend': {
                'period': 10,
                'multiplier': 3.0
            },
            'rsi_entry': {
                'period': 2,
                'solid': {
                    'rsi_long_entry': 20,
                    'rsi_short_entry': 80
                }
            },
            'rsi_exit': {
                'period': 14,
                'solid': {
                    'rsi_long_exit_solid': 70,
                    'rsi_short_exit_solid': 30
                }
            },
            'chop': {
                'period': 14,
                'solid': {
                    'chop_solid': 50
                }
            }
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """現在のパラメータを取得する"""
        return self._parameters.copy()
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """パラメータを設定する"""
        if self.validate_parameters(params):
            self._parameters.update(params)
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        パラメータの妥当性を検証する
        
        Args:
            params: 検証するパラメータ
        
        Returns:
            bool: パラメータが有効な場合はTrue
        
        Raises:
            ValueError: パラメータが無効な場合
        """
        required_keys = {'supertrend', 'rsi_entry', 'rsi_exit', 'chop'}
        if not all(key in params for key in required_keys):
            raise ValueError(f"Required parameters missing. Required: {required_keys}")
        
        # スーパートレンドのパラメータ検証
        supertrend = params.get('supertrend', {})
        if not (isinstance(supertrend.get('period', 0), (int, float)) and supertrend.get('period', 0) > 0):
            raise ValueError("Supertrend period must be a positive number")
        if not (isinstance(supertrend.get('multiplier', 0), (int, float)) and supertrend.get('multiplier', 0) > 0):
            raise ValueError("Supertrend multiplier must be a positive number")
        
        # RSIエントリーのパラメータ検証
        rsi_entry = params.get('rsi_entry', {})
        if not (isinstance(rsi_entry.get('period', 0), (int, float)) and rsi_entry.get('period', 0) > 0):
            raise ValueError("RSI entry period must be a positive number")
        
        solid_entry = rsi_entry.get('solid', {})
        if not (0 <= solid_entry.get('rsi_long_entry', -1) <= 100):
            raise ValueError("RSI long entry must be between 0 and 100")
        if not (0 <= solid_entry.get('rsi_short_entry', -1) <= 100):
            raise ValueError("RSI short entry must be between 0 and 100")
        
        # RSIエグジットのパラメータ検証
        rsi_exit = params.get('rsi_exit', {})
        if not (isinstance(rsi_exit.get('period', 0), (int, float)) and rsi_exit.get('period', 0) > 0):
            raise ValueError("RSI exit period must be a positive number")
        
        solid_exit = rsi_exit.get('solid', {})
        if not (0 <= solid_exit.get('rsi_long_exit_solid', -1) <= 100):
            raise ValueError("RSI long exit must be between 0 and 100")
        if not (0 <= solid_exit.get('rsi_short_exit_solid', -1) <= 100):
            raise ValueError("RSI short exit must be between 0 and 100")
        
        # Chopのパラメータ検証
        chop = params.get('chop', {})
        if not (isinstance(chop.get('period', 0), (int, float)) and chop.get('period', 0) > 0):
            raise ValueError("Chop period must be a positive number")
        
        solid_chop = chop.get('solid', {})
        if not (0 <= solid_chop.get('chop_solid', -1) <= 100):
            raise ValueError("Chop solid must be between 0 and 100")
        
        return True 