#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Protocol, Dict, Any, ClassVar
import optuna

class IOptimization(Protocol):
    """最適化のインターフェース"""
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """最適化パラメータを生成する"""
        ...
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換する"""
        ... 