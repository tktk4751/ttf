#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna
import logging

from ...base.strategy import BaseStrategy
from .signal_generator import AlphaTrendMACDHiddenDivergenceSignalGenerator


logger = logging.getLogger(__name__)

class AlphaTrendMACDHiddenDivergenceStrategy(BaseStrategy):
    """
    AlphaTrend + AlphaMACDヒドゥンダイバージェンス戦略
    
    特徴:
    - 効率比（ER）に基づく動的パラメータ最適化
    - AlphaTrendによる高精度なトレンド検出
    - AlphaMACDヒドゥンダイバージェンスによるトレンド継続の確認
    - Numbaによる高速化
    
    エントリー条件:
    - ロング: AlphaTrendDirectionSignalが1 + AlphaMACDHiddenDivergenceSignalが1
    - ショート: AlphaTrendDirectionSignalが-1 + AlphaMACDHiddenDivergenceSignalが-1
    
    エグジット条件:
    - ロング: AlphaTrendDirectionSignalが-1
    - ショート: AlphaTrendDirectionSignalが1
    """
    
    def __init__(
        self,
        # 共通パラメータ
        er_period: int = 21,
        # AlphaTrend用パラメータ
        max_percentile_length: int = 233,
        min_percentile_length: int = 13,
        max_atr_period: int = 89,
        min_atr_period: int = 13,
        max_trend_multiplier: float = 3.0,
        min_trend_multiplier: float = 1.0,
        alma_offset: float = 0.85,
        alma_sigma: int = 6,
        # AlphaMACDHiddenDivergence用パラメータ
        fast_max_kama_period: int = 34,
        fast_min_kama_period: int = 5,
        slow_max_kama_period: int = 89,
        slow_min_kama_period: int = 21,
        signal_max_kama_period: int = 34,
        signal_min_kama_period: int = 5,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2,
        lookback: int = 30
    ):
        """
        初期化
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_percentile_length: パーセンタイル計算の最大期間（デフォルト: 55）
            min_percentile_length: パーセンタイル計算の最小期間（デフォルト: 13）
            max_atr_period: Alpha ATR期間の最大値（デフォルト: 89）
            min_atr_period: Alpha ATR期間の最小値（デフォルト: 13）
            max_trend_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_trend_multiplier: ATR乗数の最小値（デフォルト: 1.0）
            alma_offset: ALMAのオフセット（デフォルト: 0.85）
            alma_sigma: ALMAのシグマ（デフォルト: 6）
            fast_max_kama_period: 短期AlphaMAのKAMAピリオドの最大値（デフォルト: 89）
            fast_min_kama_period: 短期AlphaMAのKAMAピリオドの最小値（デフォルト: 8）
            slow_max_kama_period: 長期AlphaMAのKAMAピリオドの最大値（デフォルト: 144）
            slow_min_kama_period: 長期AlphaMAのKAMAピリオドの最小値（デフォルト: 21）
            signal_max_kama_period: シグナルAlphaMAのKAMAピリオドの最大値（デフォルト: 55）
            signal_min_kama_period: シグナルAlphaMAのKAMAピリオドの最小値（デフォルト: 5）
            max_slow_period: 遅い移動平均の最大期間（デフォルト: 89）
            min_slow_period: 遅い移動平均の最小期間（デフォルト: 30）
            max_fast_period: 速い移動平均の最大期間（デフォルト: 13）
            min_fast_period: 速い移動平均の最小期間（デフォルト: 2）
            lookback: ヒドゥンダイバージェンス検出のルックバック期間（デフォルト: 30）
        """
        super().__init__("AlphaTrendMACDHiddenDivergence")
        
        # ロガーの設定
        self.logger = logger
        
        # パラメータの設定
        self._parameters = {
            'er_period': er_period,
            'max_percentile_length': max_percentile_length,
            'min_percentile_length': min_percentile_length,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_trend_multiplier': max_trend_multiplier,
            'min_trend_multiplier': min_trend_multiplier,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'fast_max_kama_period': fast_max_kama_period,
            'fast_min_kama_period': fast_min_kama_period,
            'slow_max_kama_period': slow_max_kama_period,
            'slow_min_kama_period': slow_min_kama_period,
            'signal_max_kama_period': signal_max_kama_period,
            'signal_min_kama_period': signal_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period,
            'lookback': lookback
        }
        
        # シグナル生成器の初期化
        self.signal_generator = AlphaTrendMACDHiddenDivergenceSignalGenerator(
            er_period=er_period,
            max_percentile_length=max_percentile_length,
            min_percentile_length=min_percentile_length,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_trend_multiplier=max_trend_multiplier,
            min_trend_multiplier=min_trend_multiplier,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma,
            fast_max_kama_period=fast_max_kama_period,
            fast_min_kama_period=fast_min_kama_period,
            slow_max_kama_period=slow_max_kama_period,
            slow_min_kama_period=slow_min_kama_period,
            signal_max_kama_period=signal_max_kama_period,
            signal_min_kama_period=signal_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period,
            lookback=lookback
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル
        """
        try:
            return self.signal_generator.get_entry_signals(data)
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
            return self.signal_generator.get_exit_signals(data, position, index)
        except Exception as e:
            self.logger.error(f"エグジットシグナル生成中にエラー: {str(e)}")
            return False
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成
        
        Args:
            trial: Optunaのトライアル
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        params = {
            'er_period': trial.suggest_int('er_period', 5, 200),
            'max_percentile_length': trial.suggest_int('max_percentile_length', 34, 233),
            'min_percentile_length': trial.suggest_int('min_percentile_length', 3, 33),
            'max_atr_period': trial.suggest_int('max_atr_period', 22, 144),
            'min_atr_period': trial.suggest_int('min_atr_period', 5, 21),
            'max_trend_multiplier': trial.suggest_float('max_trend_multiplier', 1.6, 3.0, step=0.1),
            'min_trend_multiplier': trial.suggest_float('min_trend_multiplier', 0.1, 1.5, step=0.1),
            'alma_offset': trial.suggest_float('alma_offset', 0.3, 0.9, step=0.1),
            'alma_sigma': trial.suggest_int('alma_sigma', 2, 9),
            'fast_max_kama_period': trial.suggest_int('fast_max_kama_period', 55, 144),
            'fast_min_kama_period': trial.suggest_int('fast_min_kama_period', 5, 21),
            'slow_max_kama_period': trial.suggest_int('slow_max_kama_period', 89, 233),
            'slow_min_kama_period': trial.suggest_int('slow_min_kama_period', 13, 55),
            'signal_max_kama_period': trial.suggest_int('signal_max_kama_period', 34, 89),
            'signal_min_kama_period': trial.suggest_int('signal_min_kama_period', 3, 13),
            'max_slow_period': trial.suggest_int('max_slow_period', 55, 144),
            'min_slow_period': trial.suggest_int('min_slow_period', 13, 55),
            'max_fast_period': trial.suggest_int('max_fast_period', 8, 21),
            'min_fast_period': trial.suggest_int('min_fast_period', 2, 8),
            'lookback': trial.suggest_int('lookback', 10, 50, step=5)
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            Dict[str, Any]: 戦略パラメータ
        """
        return {
            'er_period': int(params['er_period']),
            'max_percentile_length': int(params['max_percentile_length']),
            'min_percentile_length': int(params['min_percentile_length']),
            'max_atr_period': int(params['max_atr_period']),
            'min_atr_period': int(params['min_atr_period']),
            'max_trend_multiplier': float(params['max_trend_multiplier']),
            'min_trend_multiplier': float(params['min_trend_multiplier']),
            'alma_offset': float(params['alma_offset']),
            'alma_sigma': int(params['alma_sigma']),
            'fast_max_kama_period': int(params['fast_max_kama_period']),
            'fast_min_kama_period': int(params['fast_min_kama_period']),
            'slow_max_kama_period': int(params['slow_max_kama_period']),
            'slow_min_kama_period': int(params['slow_min_kama_period']),
            'signal_max_kama_period': int(params['signal_max_kama_period']),
            'signal_min_kama_period': int(params['signal_min_kama_period']),
            'max_slow_period': int(params['max_slow_period']),
            'min_slow_period': int(params['min_slow_period']),
            'max_fast_period': int(params['max_fast_period']),
            'min_fast_period': int(params['min_fast_period']),
            'lookback': int(params['lookback'])
        } 