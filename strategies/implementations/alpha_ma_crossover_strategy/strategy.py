#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, List, Optional, Tuple
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import AlphaMACrossoverSignalGenerator


class AlphaMACrossoverStrategy(BaseStrategy):
    """
    アルファMAクロスオーバー戦略
    
    AlphaMACrossoverEntrySignalを使用した適応型移動平均クロスオーバー戦略です。
    
    特徴:
    - 効率比（ER）に基づいて動的に調整されるAlphaMAを使用
    - トレンドが強い時は短いピリオドで速く反応
    - レンジ相場時は長いピリオドでノイズを除去
    
    エントリー条件:
    - AlphaMACrossoverEntrySignalが1のときにロング
    - AlphaMACrossoverEntrySignalが-1のときにショート
    
    エグジット条件:
    - AlphaMACrossoverEntrySignalが-1でロング決済
    - AlphaMACrossoverEntrySignalが1になったらショート決済
    """
    
    def __init__(
        self,
        # 基本パラメータ
        name: str = "AlphaMACrossoverStrategy",
        
        # AlphaMACrossoverEntrySignalのパラメータ
        er_period: int = 21,
        short_max_kama_period: int = 89,
        short_min_kama_period: int = 5,
        long_max_kama_period: int = 233,
        long_min_kama_period: int = 21,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2,
        
        # リスク管理パラメータ
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
        trailing_stop_pct: float = 1.5,
        
        # その他のパラメータ
        **kwargs
    ):
        """初期化"""
        super().__init__(name)
        
        # シグナル生成器の初期化
        self.signal_generator = AlphaMACrossoverSignalGenerator(
            # AlphaMACrossoverEntrySignalのパラメータ
            er_period=er_period,
            short_max_kama_period=short_max_kama_period,
            short_min_kama_period=short_min_kama_period,
            long_max_kama_period=long_max_kama_period,
            long_min_kama_period=long_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        
        # リスク管理パラメータの設定
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        
        # パラメータの保存
        self._params = {
            # 基本パラメータ
            'name': name,
            
            # AlphaMACrossoverEntrySignalのパラメータ
            'er_period': er_period,
            'short_max_kama_period': short_max_kama_period,
            'short_min_kama_period': short_min_kama_period,
            'long_max_kama_period': long_max_kama_period,
            'long_min_kama_period': long_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period,
            
            # リスク管理パラメータ
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'trailing_stop_pct': trailing_stop_pct
        }
        
        # その他のパラメータを追加
        for key, value in kwargs.items():
            self._params[key] = value
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル配列
        """
        return self.signal_generator.get_entry_signals(data)
    
    def should_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットすべきかどうかを判断
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: 判断する時点のインデックス（デフォルトは最新）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        return self.signal_generator.get_exit_signals(data, position, index)
    
    def calculate_stop_loss(self, data: Union[pd.DataFrame, np.ndarray], position: int, entry_price: float) -> float:
        """
        ストップロス価格を計算
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            entry_price: エントリー価格
            
        Returns:
            float: ストップロス価格
        """
        if position == 1:  # ロングポジション
            return entry_price * (1 - self.stop_loss_pct / 100)
        elif position == -1:  # ショートポジション
            return entry_price * (1 + self.stop_loss_pct / 100)
        return 0.0
    
    def calculate_take_profit(self, data: Union[pd.DataFrame, np.ndarray], position: int, entry_price: float) -> float:
        """
        利益確定価格を計算
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            entry_price: エントリー価格
            
        Returns:
            float: 利益確定価格
        """
        if position == 1:  # ロングポジション
            return entry_price * (1 + self.take_profit_pct / 100)
        elif position == -1:  # ショートポジション
            return entry_price * (1 - self.take_profit_pct / 100)
        return 0.0
    
    def calculate_trailing_stop(self, data: Union[pd.DataFrame, np.ndarray], position: int, highest_lowest_price: float) -> float:
        """
        トレーリングストップ価格を計算
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            highest_lowest_price: ポジション保有中の最高値（ロング）または最安値（ショート）
            
        Returns:
            float: トレーリングストップ価格
        """
        if position == 1:  # ロングポジション
            return highest_lowest_price * (1 - self.trailing_stop_pct / 100)
        elif position == -1:  # ショートポジション
            return highest_lowest_price * (1 + self.trailing_stop_pct / 100)
        return 0.0
    
    def get_ma_crossover_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        MAクロスオーバーシグナルを取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: MAクロスオーバーシグナル配列
        """
        return self.signal_generator.get_ma_crossover_signals(data)
    
    def get_params(self) -> Dict[str, Any]:
        """
        戦略のパラメータを取得
        
        Returns:
            Dict[str, Any]: パラメータの辞書
        """
        return self._params.copy()
    
    def set_params(self, **params) -> None:
        """
        戦略のパラメータを設定
        
        Args:
            **params: 設定するパラメータ
        """
        # パラメータの更新
        for key, value in params.items():
            if key in self._params:
                self._params[key] = value
        
        # シグナル生成器の再初期化
        self.signal_generator = AlphaMACrossoverSignalGenerator(
            # AlphaMACrossoverEntrySignalのパラメータ
            er_period=self._params['er_period'],
            short_max_kama_period=self._params['short_max_kama_period'],
            short_min_kama_period=self._params['short_min_kama_period'],
            long_max_kama_period=self._params['long_max_kama_period'],
            long_min_kama_period=self._params['long_min_kama_period'],
            max_slow_period=self._params['max_slow_period'],
            min_slow_period=self._params['min_slow_period'],
            max_fast_period=self._params['max_fast_period'],
            min_fast_period=self._params['min_fast_period']
        )
        
        # リスク管理パラメータの更新
        self.stop_loss_pct = self._params['stop_loss_pct']
        self.take_profit_pct = self._params['take_profit_pct']
        self.trailing_stop_pct = self._params['trailing_stop_pct']
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル
        """
        try:
            return self.get_entry_signals(data)
        except Exception as e:
            self.logger.error(f"エントリーシグナル生成中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        try:
            return self.should_exit(data, position, index)
        except Exception as e:
            self.logger.error(f"エグジットシグナル生成中にエラー: {str(e)}")
            return False
    
    def get_stop_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """
        ストップロス価格を取得する
        
        Args:
            data: 価格データ
            position: ポジション方向 (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1 = 最新のデータ）
            
        Returns:
            float: ストップロス価格
        """
        entry_price = self.get_entry_price(data, position, index)
        return self.calculate_stop_loss(data, position, entry_price)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成する
        
        Args:
            trial: Optunaのトライアル
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        params = {
            # AlphaMACrossoverEntrySignalのパラメータ
            'er_period': trial.suggest_int('er_period', 5, 200),
            'short_max_kama_period': trial.suggest_int('short_max_kama_period', 21, 89),
            'short_min_kama_period': trial.suggest_int('short_min_kama_period', 2, 13),
            'long_max_kama_period': trial.suggest_int('long_max_kama_period', 144, 377),
            'long_min_kama_period': trial.suggest_int('long_min_kama_period', 21, 89),
            'max_slow_period': trial.suggest_int('max_slow_period', 55, 144),
            'min_slow_period': trial.suggest_int('min_slow_period', 13, 55),
            'max_fast_period': trial.suggest_int('max_fast_period', 8, 21),
            'min_fast_period': trial.suggest_int('min_fast_period', 2, 8),
            
            # リスク管理パラメータ
            'stop_loss_pct': trial.suggest_float('stop_loss_pct', 1.0, 5.0, step=0.5),
            'take_profit_pct': trial.suggest_float('take_profit_pct', 2.0, 10.0, step=0.5),
            'trailing_stop_pct': trial.suggest_float('trailing_stop_pct', 0.5, 3.0, step=0.5)
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換する
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            Dict[str, Any]: 戦略パラメータ
        """
        # パラメータの型変換
        strategy_params = {}
        
        # 整数パラメータ
        int_params = [
            'er_period', 'short_max_kama_period', 'short_min_kama_period',
            'long_max_kama_period', 'long_min_kama_period', 'max_slow_period',
            'min_slow_period', 'max_fast_period', 'min_fast_period'
        ]
        
        # 浮動小数点パラメータ
        float_params = [
            'stop_loss_pct', 'take_profit_pct', 'trailing_stop_pct'
        ]
        
        # 整数パラメータの変換
        for param in int_params:
            if param in params:
                strategy_params[param] = int(params[param])
        
        # 浮動小数点パラメータの変換
        for param in float_params:
            if param in params:
                strategy_params[param] = float(params[param])
        
        return strategy_params 