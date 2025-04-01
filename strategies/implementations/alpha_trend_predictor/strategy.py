#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, List, Optional, Tuple
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import AlphaTrendPredictorGenerator


class AlphaTrendPredictorStrategy(BaseStrategy):
    """
    Alphaトレンドプレディクター戦略
    
    複数のAlphaシグナルを重み付けして組み合わせ、スコアリングする戦略です。
    
    特徴:
    - 9種類のAlphaシグナルを重み付けして組み合わせる
    - スコアを-20から20の範囲に正規化
    - スコアが10以上でロングエントリー、-10以下でショートエントリー
    
    使用するシグナル:
    - AlphaFilterSignal: フィルターシグナル
    - AlphaKeltnerBreakoutEntrySignal: エントリーシグナル
    - AlphaMACirculationSignal: ディレクションシグナル
    - AlphaTrendDirectionSignal: ディレクションシグナル
    - AlphaSqueezeEntrySignal: エントリーシグナル
    - AlphaMACDDivergenceSignal: エントリーシグナル
    - AlphaMACDHiddenDivergenceSignal: エントリーシグナル
    - AlphaROCDivergenceSignal: エントリーシグナル
    - AlphaROCHiddenDivergenceSignal: エントリーシグナル
    
    エントリー条件:
    - 合成スコアが10以上でロングエントリー
    - 合成スコアが-10以下でショートエントリー
    
    エグジット条件:
    - ロングポジションの場合、スコアが0未満になったらエグジット
    - ショートポジションの場合、スコアが0より大きくなったらエグジット
    """
    
    def __init__(
        self,
        # 基本パラメータ
        name: str = "AlphaTrendPredictorStrategy",
        
        # 各シグナルの重み
        alpha_filter_weight: float = 2.0,
        alpha_keltner_weight: float = 2.5,
        alpha_ma_circulation_weight: float = 3.0,
        alpha_trend_weight: float = 3.0,
        alpha_squeeze_weight: float = 2.0,
        alpha_macd_divergence_weight: float = 2.5,
        alpha_macd_hidden_divergence_weight: float = 2.5,
        alpha_roc_divergence_weight: float = 2.5,
        alpha_roc_hidden_divergence_weight: float = 2.5,
        
        # エントリーしきい値
        entry_threshold: float = 10.0,
        
        # 各シグナルのパラメータ
        alpha_filter_period: int = 14,
        alpha_keltner_period: int = 20,
        alpha_keltner_atr_period: int = 10,
        alpha_keltner_atr_multiplier: float = 2.0,
        alpha_ma_circulation_er_period: int = 21,
        alpha_trend_period: int = 14,
        alpha_squeeze_bb_period: int = 20,
        alpha_squeeze_bb_std_dev: float = 2.0,
        alpha_squeeze_kc_period: int = 20,
        alpha_squeeze_kc_atr_period: int = 10,
        alpha_squeeze_kc_atr_multiplier: float = 1.5,
        alpha_macd_fast_period: int = 12,
        alpha_macd_slow_period: int = 26,
        alpha_macd_signal_period: int = 9,
        alpha_roc_period: int = 14,
        alpha_roc_smooth_period: int = 3,
        divergence_lookback: int = 10,
        
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
        self.signal_generator = AlphaTrendPredictorGenerator(
            # 各シグナルの重み
            alpha_filter_weight=alpha_filter_weight,
            alpha_keltner_weight=alpha_keltner_weight,
            alpha_ma_circulation_weight=alpha_ma_circulation_weight,
            alpha_trend_weight=alpha_trend_weight,
            alpha_squeeze_weight=alpha_squeeze_weight,
            alpha_macd_divergence_weight=alpha_macd_divergence_weight,
            alpha_macd_hidden_divergence_weight=alpha_macd_hidden_divergence_weight,
            alpha_roc_divergence_weight=alpha_roc_divergence_weight,
            alpha_roc_hidden_divergence_weight=alpha_roc_hidden_divergence_weight,
            
            # エントリーしきい値
            entry_threshold=entry_threshold,
            
            # 各シグナルのパラメータ
            alpha_filter_period=alpha_filter_period,
            alpha_keltner_period=alpha_keltner_period,
            alpha_keltner_atr_period=alpha_keltner_atr_period,
            alpha_keltner_atr_multiplier=alpha_keltner_atr_multiplier,
            alpha_ma_circulation_er_period=alpha_ma_circulation_er_period,
            alpha_trend_period=alpha_trend_period,
            alpha_squeeze_bb_period=alpha_squeeze_bb_period,
            alpha_squeeze_bb_std_dev=alpha_squeeze_bb_std_dev,
            alpha_squeeze_kc_period=alpha_squeeze_kc_period,
            alpha_squeeze_kc_atr_period=alpha_squeeze_kc_atr_period,
            alpha_squeeze_kc_atr_multiplier=alpha_squeeze_kc_atr_multiplier,
            alpha_macd_fast_period=alpha_macd_fast_period,
            alpha_macd_slow_period=alpha_macd_slow_period,
            alpha_macd_signal_period=alpha_macd_signal_period,
            alpha_roc_period=alpha_roc_period,
            alpha_roc_smooth_period=alpha_roc_smooth_period,
            divergence_lookback=divergence_lookback
        )
        
        # リスク管理パラメータの設定
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        
        # パラメータの保存
        self._params = {
            # 基本パラメータ
            'name': name,
            
            # 各シグナルの重み
            'alpha_filter_weight': alpha_filter_weight,
            'alpha_keltner_weight': alpha_keltner_weight,
            'alpha_ma_circulation_weight': alpha_ma_circulation_weight,
            'alpha_trend_weight': alpha_trend_weight,
            'alpha_squeeze_weight': alpha_squeeze_weight,
            'alpha_macd_divergence_weight': alpha_macd_divergence_weight,
            'alpha_macd_hidden_divergence_weight': alpha_macd_hidden_divergence_weight,
            'alpha_roc_divergence_weight': alpha_roc_divergence_weight,
            'alpha_roc_hidden_divergence_weight': alpha_roc_hidden_divergence_weight,
            
            # エントリーしきい値
            'entry_threshold': entry_threshold,
            
            # 各シグナルのパラメータ
            'alpha_filter_period': alpha_filter_period,
            'alpha_keltner_period': alpha_keltner_period,
            'alpha_keltner_atr_period': alpha_keltner_atr_period,
            'alpha_keltner_atr_multiplier': alpha_keltner_atr_multiplier,
            'alpha_ma_circulation_er_period': alpha_ma_circulation_er_period,
            'alpha_trend_period': alpha_trend_period,
            'alpha_squeeze_bb_period': alpha_squeeze_bb_period,
            'alpha_squeeze_bb_std_dev': alpha_squeeze_bb_std_dev,
            'alpha_squeeze_kc_period': alpha_squeeze_kc_period,
            'alpha_squeeze_kc_atr_period': alpha_squeeze_kc_atr_period,
            'alpha_squeeze_kc_atr_multiplier': alpha_squeeze_kc_atr_multiplier,
            'alpha_macd_fast_period': alpha_macd_fast_period,
            'alpha_macd_slow_period': alpha_macd_slow_period,
            'alpha_macd_signal_period': alpha_macd_signal_period,
            'alpha_roc_period': alpha_roc_period,
            'alpha_roc_smooth_period': alpha_roc_smooth_period,
            'divergence_lookback': divergence_lookback,
            
            # リスク管理パラメータ
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'trailing_stop_pct': trailing_stop_pct
        }
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを取得する
        
        Args:
            data: 価格データ
            
        Returns:
            エントリーシグナル配列 (1: ロング、-1: ショート、0: エントリーなし)
        """
        return self.signal_generator.get_entry_signals(data)
    
    def should_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットすべきかどうかを判定する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: チェックするインデックス（デフォルトは最新）
            
        Returns:
            エグジットすべきかどうか
        """
        return self.signal_generator.get_exit_signals(data, position, index)
    
    def calculate_stop_loss(self, data: Union[pd.DataFrame, np.ndarray], position: int, entry_price: float) -> float:
        """
        ストップロス価格を計算する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            entry_price: エントリー価格
            
        Returns:
            ストップロス価格
        """
        if position == 1:  # ロングポジション
            return entry_price * (1 - self.stop_loss_pct / 100)
        elif position == -1:  # ショートポジション
            return entry_price * (1 + self.stop_loss_pct / 100)
        return 0.0
    
    def calculate_take_profit(self, data: Union[pd.DataFrame, np.ndarray], position: int, entry_price: float) -> float:
        """
        利確価格を計算する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            entry_price: エントリー価格
            
        Returns:
            利確価格
        """
        if position == 1:  # ロングポジション
            return entry_price * (1 + self.take_profit_pct / 100)
        elif position == -1:  # ショートポジション
            return entry_price * (1 - self.take_profit_pct / 100)
        return 0.0
    
    def calculate_trailing_stop(self, data: Union[pd.DataFrame, np.ndarray], position: int, highest_lowest_price: float) -> float:
        """
        トレーリングストップ価格を計算する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            highest_lowest_price: ポジション保有中の最高値（ロング）または最安値（ショート）
            
        Returns:
            トレーリングストップ価格
        """
        if position == 1:  # ロングポジション
            return highest_lowest_price * (1 - self.trailing_stop_pct / 100)
        elif position == -1:  # ショートポジション
            return highest_lowest_price * (1 + self.trailing_stop_pct / 100)
        return 0.0
    
    def get_score(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        スコアを取得する
        
        Args:
            data: 価格データ（Noneの場合はキャッシュを使用）
            
        Returns:
            スコア配列
        """
        return self.signal_generator.get_score(data)
    
    def get_params(self) -> Dict[str, Any]:
        """
        現在のパラメータを取得する
        
        Returns:
            パラメータ辞書
        """
        return self._params.copy()
    
    def set_params(self, **params) -> None:
        """
        パラメータを設定する
        
        Args:
            **params: 設定するパラメータ
        """
        # パラメータの更新
        self._params.update(params)
        
        # シグナル生成器のパラメータを更新
        signal_params = {}
        for key, value in params.items():
            if key in self.signal_generator._params:
                signal_params[key] = value
        
        if signal_params:
            self.signal_generator.set_parameters(signal_params)
        
        # リスク管理パラメータの更新
        if 'stop_loss_pct' in params:
            self.stop_loss_pct = params['stop_loss_pct']
        if 'take_profit_pct' in params:
            self.take_profit_pct = params['take_profit_pct']
        if 'trailing_stop_pct' in params:
            self.trailing_stop_pct = params['trailing_stop_pct']
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            エントリーシグナル配列 (1: ロング、-1: ショート、0: エントリーなし)
        """
        return self.get_entry_signals(data)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: チェックするインデックス（デフォルトは最新）
            
        Returns:
            エグジットすべきかどうか
        """
        return self.should_exit(data, position, index)
    
    def get_stop_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """
        ストップ価格を取得する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: チェックするインデックス（デフォルトは最新）
            
        Returns:
            ストップ価格
        """
        # 現在の価格を取得
        if isinstance(data, pd.DataFrame):
            current_price = data.iloc[index]['close']
        else:
            current_price = data[index, 3]  # 3はcloseカラムのインデックス
        
        return self.calculate_stop_loss(data, position, current_price)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化用のパラメータを生成する
        
        Args:
            trial: optunaのtrialオブジェクト
            
        Returns:
            最適化用のパラメータ辞書
        """
        params = {
            # 各シグナルの重み
            'alpha_filter_weight': trial.suggest_float('alpha_filter_weight', 1.0, 4.0),
            'alpha_keltner_weight': trial.suggest_float('alpha_keltner_weight', 1.0, 4.0),
            'alpha_ma_circulation_weight': trial.suggest_float('alpha_ma_circulation_weight', 1.0, 5.0),
            'alpha_trend_weight': trial.suggest_float('alpha_trend_weight', 1.0, 5.0),
            'alpha_squeeze_weight': trial.suggest_float('alpha_squeeze_weight', 1.0, 4.0),
            'alpha_macd_divergence_weight': trial.suggest_float('alpha_macd_divergence_weight', 1.0, 4.0),
            'alpha_macd_hidden_divergence_weight': trial.suggest_float('alpha_macd_hidden_divergence_weight', 1.0, 4.0),
            'alpha_roc_divergence_weight': trial.suggest_float('alpha_roc_divergence_weight', 1.0, 4.0),
            'alpha_roc_hidden_divergence_weight': trial.suggest_float('alpha_roc_hidden_divergence_weight', 1.0, 4.0),
            
            # エントリーしきい値
            'entry_threshold': trial.suggest_float('entry_threshold', 8.0, 12.0),
            
            # 各シグナルのパラメータ
            'alpha_filter_period': trial.suggest_int('alpha_filter_period', 7, 21),
            'alpha_keltner_period': trial.suggest_int('alpha_keltner_period', 10, 30),
            'alpha_keltner_atr_period': trial.suggest_int('alpha_keltner_atr_period', 5, 15),
            'alpha_keltner_atr_multiplier': trial.suggest_float('alpha_keltner_atr_multiplier', 1.5, 3.0),
            'alpha_ma_circulation_er_period': trial.suggest_int('alpha_ma_circulation_er_period', 10, 30),
            'alpha_trend_period': trial.suggest_int('alpha_trend_period', 7, 21),
            'alpha_squeeze_bb_period': trial.suggest_int('alpha_squeeze_bb_period', 10, 30),
            'alpha_squeeze_bb_std_dev': trial.suggest_float('alpha_squeeze_bb_std_dev', 1.5, 3.0),
            'alpha_squeeze_kc_period': trial.suggest_int('alpha_squeeze_kc_period', 10, 30),
            'alpha_squeeze_kc_atr_period': trial.suggest_int('alpha_squeeze_kc_atr_period', 5, 15),
            'alpha_squeeze_kc_atr_multiplier': trial.suggest_float('alpha_squeeze_kc_atr_multiplier', 1.0, 2.5),
            'alpha_macd_fast_period': trial.suggest_int('alpha_macd_fast_period', 8, 16),
            'alpha_macd_slow_period': trial.suggest_int('alpha_macd_slow_period', 20, 32),
            'alpha_macd_signal_period': trial.suggest_int('alpha_macd_signal_period', 6, 12),
            'alpha_roc_period': trial.suggest_int('alpha_roc_period', 7, 21),
            'alpha_roc_smooth_period': trial.suggest_int('alpha_roc_smooth_period', 2, 5),
            'divergence_lookback': trial.suggest_int('divergence_lookback', 5, 15),
            
            # リスク管理パラメータ
            'stop_loss_pct': trial.suggest_float('stop_loss_pct', 1.0, 3.0),
            'take_profit_pct': trial.suggest_float('take_profit_pct', 2.0, 6.0),
            'trailing_stop_pct': trial.suggest_float('trailing_stop_pct', 0.5, 2.5)
        }
        
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータをストラテジーフォーマットに変換する
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            ストラテジーフォーマットのパラメータ
        """
        # パラメータをそのまま返す（必要に応じて変換処理を追加）
        return params.copy() 