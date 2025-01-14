#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Protocol, Dict, Any
import optuna

class IStrategyParameters(Protocol):
    """戦略パラメータのインターフェース"""
    @staticmethod
    def create_optimization_params(trial: optuna.Trial) -> Dict[str, Any]:
        """最適化用のパラメータを生成する"""
        ...
    
    @staticmethod
    def convert_params_to_strategy_format(params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略クラスの形式に変換"""
        ...
    
    def get_parameters(self) -> Dict[str, Any]:
        """現在の戦略パラメータを取得する"""
        ...

class ISignalGenerator(Protocol):
    """シグナル生成のインターフェース"""
    def calculate_signals(self, data: Any) -> None:
        """全てのシグナルを計算してキャッシュする"""
        ...

class IEntryStrategy(Protocol):
    """エントリー戦略のインターフェース"""
    def generate_entry(self, data: Any) -> Any:
        """エントリーシグナルを生成する"""
        ...

class IExitStrategy(Protocol):
    """エグジット戦略のインターフェース"""
    def generate_exit(self, data: Any, position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成する"""
        ... 