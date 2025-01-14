#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Protocol, Dict, Any

class IParameters(Protocol):
    """パラメータ管理のインターフェース"""
    
    def get_parameters(self) -> Dict[str, Any]:
        """現在のパラメータを取得する"""
        ...
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """パラメータを設定する"""
        ...
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """パラメータの妥当性を検証する"""
        ... 