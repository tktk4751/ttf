#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, List, Optional, Tuple
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import AlphaCirculationSignalGenerator


class AlphaCirculationStrategy(BaseStrategy):
    """
    アルファ循環戦略
    
    AlphaMACirculationSignal、AlphaFilterSignal、AlphaMACDDivergenceSignalを組み合わせた
    市場の循環に基づくトレード戦略です。
    
    特徴:
    - 市場の6つのステージを識別し、各ステージに最適な取引を行います
    - 適応型フィルターで市場状態を判断し、トレンド相場と揉み合い相場を区別します
    - ダイバージェンス検出で転換点を捉えます
    - 各指標が適応型で、市場の状態に応じて自動的にパラメータを調整します
    
    エントリー条件:
    - ステージ1: 短期 > 中期 > 長期（安定上昇相場）+ AlphaFilterSignal が1でロング
    - ステージ2: 中期 > 短期 > 長期（上昇相場の終焉）+ AlphaMACDDivergenceSignal が-1でショート
    - ステージ4: 長期 > 中期 > 短期（安定下降相場）+ AlphaFilterSignal が-1でショート
    - ステージ6: 短期 > 長期 > 中期（上昇相場の入口）+ AlphaMACDDivergenceSignal が1でロング
    
    エグジット条件:
    - ステージ3: 中期 > 長期 > 短期（下降相場の入口）でロングエグジット
    - ステージ5: 長期 > 短期 > 中期（下降相場の終焉）でショートエグジット
    """
    
    def __init__(
        self,
        # 基本パラメータ
        name: str = "AlphaCirculationStrategy",
        
        # AlphaMACirculationSignalのパラメータ
        er_period: int = 21,
        short_max_kama_period: int = 55,
        short_min_kama_period: int = 3,
        middle_max_kama_period: int = 144,
        middle_min_kama_period: int = 21,
        long_max_kama_period: int = 377,
        long_min_kama_period: int = 55,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2,
        
        # AlphaFilterSignalのパラメータ
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        max_adx_period: int = 21,
        min_adx_period: int = 5,
        alma_period: int = 10,
        alma_offset: float = 0.85,
        alma_sigma: float = 6,
        filter_threshold: float = 0.5,
        
        # AlphaMACDDivergenceSignalのパラメータ
        fast_max_kama_period: int = 89,
        fast_min_kama_period: int = 8,
        slow_max_kama_period: int = 144,
        slow_min_kama_period: int = 21,
        signal_max_kama_period: int = 55,
        signal_min_kama_period: int = 5,
        lookback: int = 30,
        
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
        self.signal_generator = AlphaCirculationSignalGenerator(
            # AlphaMACirculationSignalのパラメータ
            er_period=er_period,
            short_max_kama_period=short_max_kama_period,
            short_min_kama_period=short_min_kama_period,
            middle_max_kama_period=middle_max_kama_period,
            middle_min_kama_period=middle_min_kama_period,
            long_max_kama_period=long_max_kama_period,
            long_min_kama_period=long_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period,
            
            # AlphaFilterSignalのパラメータ
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=max_adx_period,
            min_adx_period=min_adx_period,
            alma_period=alma_period,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma,
            filter_threshold=filter_threshold,
            
            # AlphaMACDDivergenceSignalのパラメータ
            fast_max_kama_period=fast_max_kama_period,
            fast_min_kama_period=fast_min_kama_period,
            slow_max_kama_period=slow_max_kama_period,
            slow_min_kama_period=slow_min_kama_period,
            signal_max_kama_period=signal_max_kama_period,
            signal_min_kama_period=signal_min_kama_period,
            lookback=lookback
        )
        
        # リスク管理パラメータの設定
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        
        # パラメータの保存
        self._params = {
            # 基本パラメータ
            'name': name,
            
            # AlphaMACirculationSignalのパラメータ
            'er_period': er_period,
            'short_max_kama_period': short_max_kama_period,
            'short_min_kama_period': short_min_kama_period,
            'middle_max_kama_period': middle_max_kama_period,
            'middle_min_kama_period': middle_min_kama_period,
            'long_max_kama_period': long_max_kama_period,
            'long_min_kama_period': long_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period,
            
            # AlphaFilterSignalのパラメータ
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'alma_period': alma_period,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'filter_threshold': filter_threshold,
            
            # AlphaMACDDivergenceSignalのパラメータ
            'fast_max_kama_period': fast_max_kama_period,
            'fast_min_kama_period': fast_min_kama_period,
            'slow_max_kama_period': slow_max_kama_period,
            'slow_min_kama_period': slow_min_kama_period,
            'signal_max_kama_period': signal_max_kama_period,
            'signal_min_kama_period': signal_min_kama_period,
            'lookback': lookback,
            
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
    
    def get_stages(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        市場ステージを取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: ステージ配列
        """
        return self.signal_generator.get_stages(data)
    
    def get_filter_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        フィルター値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: フィルター値配列
        """
        return self.signal_generator.get_filter_values(data)
    
    def get_divergence_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        ダイバージェンス値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Dict[str, np.ndarray]: ダイバージェンス値の辞書
        """
        return self.signal_generator.get_divergence_values(data)
    
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
        self.signal_generator = AlphaCirculationSignalGenerator(
            # AlphaMACirculationSignalのパラメータ
            er_period=self._params['er_period'],
            short_max_kama_period=self._params['short_max_kama_period'],
            short_min_kama_period=self._params['short_min_kama_period'],
            middle_max_kama_period=self._params['middle_max_kama_period'],
            middle_min_kama_period=self._params['middle_min_kama_period'],
            long_max_kama_period=self._params['long_max_kama_period'],
            long_min_kama_period=self._params['long_min_kama_period'],
            max_slow_period=self._params['max_slow_period'],
            min_slow_period=self._params['min_slow_period'],
            max_fast_period=self._params['max_fast_period'],
            min_fast_period=self._params['min_fast_period'],
            
            # AlphaFilterSignalのパラメータ
            max_chop_period=self._params['max_chop_period'],
            min_chop_period=self._params['min_chop_period'],
            max_adx_period=self._params['max_adx_period'],
            min_adx_period=self._params['min_adx_period'],
            alma_period=self._params['alma_period'],
            alma_offset=self._params['alma_offset'],
            alma_sigma=self._params['alma_sigma'],
            filter_threshold=self._params['filter_threshold'],
            
            # AlphaMACDDivergenceSignalのパラメータ
            fast_max_kama_period=self._params['fast_max_kama_period'],
            fast_min_kama_period=self._params['fast_min_kama_period'],
            slow_max_kama_period=self._params['slow_max_kama_period'],
            slow_min_kama_period=self._params['slow_min_kama_period'],
            signal_max_kama_period=self._params['signal_max_kama_period'],
            signal_min_kama_period=self._params['signal_min_kama_period'],
            lookback=self._params['lookback']
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
            # AlphaMACirculationSignalのパラメータ
            'er_period': trial.suggest_int('er_period', 5, 55),
            'short_max_kama_period': trial.suggest_int('short_max_kama_period', 13, 55),
            'short_min_kama_period': trial.suggest_int('short_min_kama_period', 2, 13),
            'middle_max_kama_period': trial.suggest_int('middle_max_kama_period', 55, 144),
            'middle_min_kama_period': trial.suggest_int('middle_min_kama_period', 13, 55),
            'long_max_kama_period': trial.suggest_int('long_max_kama_period', 144, 377),
            'long_min_kama_period': trial.suggest_int('long_min_kama_period', 55, 144),
            'max_slow_period': trial.suggest_int('max_slow_period', 55, 144),
            'min_slow_period': trial.suggest_int('min_slow_period', 13, 55),
            'max_fast_period': trial.suggest_int('max_fast_period', 8, 21),
            'min_fast_period': trial.suggest_int('min_fast_period', 2, 8),
            
            # AlphaFilterSignalのパラメータ
            'max_chop_period': trial.suggest_int('max_chop_period', 34, 89),
            'min_chop_period': trial.suggest_int('min_chop_period', 5, 21),
            'max_adx_period': trial.suggest_int('max_adx_period', 13, 34),
            'min_adx_period': trial.suggest_int('min_adx_period', 3, 13),
            'alma_period': trial.suggest_int('alma_period', 5, 21),
            'alma_offset': trial.suggest_float('alma_offset', 0.5, 0.95, step=0.05),
            'alma_sigma': trial.suggest_float('alma_sigma', 2.0, 9.0, step=0.5),
            'filter_threshold': trial.suggest_float('filter_threshold', 0.3, 0.7, step=0.05),
            
            # AlphaMACDDivergenceSignalのパラメータ
            'fast_max_kama_period': trial.suggest_int('fast_max_kama_period', 21, 89),
            'fast_min_kama_period': trial.suggest_int('fast_min_kama_period', 5, 21),
            'slow_max_kama_period': trial.suggest_int('slow_max_kama_period', 55, 144),
            'slow_min_kama_period': trial.suggest_int('slow_min_kama_period', 13, 55),
            'signal_max_kama_period': trial.suggest_int('signal_max_kama_period', 21, 55),
            'signal_min_kama_period': trial.suggest_int('signal_min_kama_period', 3, 13),
            'lookback': trial.suggest_int('lookback', 20, 60),
            
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
            'middle_max_kama_period', 'middle_min_kama_period', 'long_max_kama_period',
            'long_min_kama_period', 'max_slow_period', 'min_slow_period',
            'max_fast_period', 'min_fast_period', 'max_chop_period',
            'min_chop_period', 'max_adx_period', 'min_adx_period',
            'alma_period', 'fast_max_kama_period', 'fast_min_kama_period',
            'slow_max_kama_period', 'slow_min_kama_period', 'signal_max_kama_period',
            'signal_min_kama_period', 'lookback'
        ]
        
        # 浮動小数点パラメータ
        float_params = [
            'alma_offset', 'alma_sigma', 'filter_threshold',
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
        