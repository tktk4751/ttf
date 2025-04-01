#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import AlphaKeltnerFilterSignalGenerator


class AlphaKeltnerFilterStrategy(BaseStrategy):
    """
    アルファケルトナーチャネル+アルファフィルター戦略
    
    特徴:
    - 効率比（ER）に基づく動的パラメータ最適化
    - アルファケルトナーチャネルによる高精度なエントリーポイント検出
    - アルファフィルターによる市場状態フィルタリング
    
    エントリー条件:
    - ロング: アルファケルトナーチャネルの買いシグナル + アルファフィルターがトレンド相場
    - ショート: アルファケルトナーチャネルの売りシグナル + アルファフィルターがトレンド相場
    
    エグジット条件:
    - ロング: アルファケルトナーチャネルの売りシグナル
    - ショート: アルファケルトナーチャネルの買いシグナル
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_kama_period: int = 144,
        min_kama_period: int = 5,
        max_atr_period: int = 233,
        min_atr_period: int = 5,
        max_keltner_multiplier: float = 3.0,
        min_keltner_multiplier: float = 1.0,
        lookback: int = 1,
        max_chop_period: int = 89,
        min_chop_period: int = 13,
        filter_threshold: float = 0.5
    ):
        """
        初期化
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_kama_period: AlphaMAの最大期間（デフォルト: 89）
            min_kama_period: AlphaMAの最小期間（デフォルト: 13）
            max_atr_period: AlphaATRの最大期間（デフォルト: 89）
            min_atr_period: AlphaATRの最小期間（デフォルト: 13）
            max_keltner_multiplier: ケルトナーチャネルの最大乗数（デフォルト: 3.0）
            min_keltner_multiplier: ケルトナーチャネルの最小乗数（デフォルト: 0.5）
            alma_offset: ALMAのオフセット（デフォルト: 0.9）
            alma_sigma: ALMAのシグマ（デフォルト: 4）
            lookback: 過去バンド参照期間（デフォルト: 1）
            max_adx_period: アルファADXの最大期間（デフォルト: 21）
            min_adx_period: アルファADXの最小期間（デフォルト: 5）
            filter_threshold: アルファフィルターのしきい値（デフォルト: 0.5）
        """
        super().__init__("AlphaKeltnerFilter")
        
        # パラメータの設定
        self._parameters = {
            'er_period': er_period,
            'max_kama_period': max_kama_period,
            'min_kama_period': min_kama_period,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_keltner_multiplier': max_keltner_multiplier,
            'min_keltner_multiplier': min_keltner_multiplier,
            'lookback': lookback,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'filter_threshold': filter_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = AlphaKeltnerFilterSignalGenerator(
            er_period=er_period,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_keltner_multiplier=max_keltner_multiplier,
            min_keltner_multiplier=min_keltner_multiplier,
            lookback=lookback,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            filter_threshold=filter_threshold
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
            'max_kama_period': trial.suggest_int('max_kama_period', 55, 233),
            'min_kama_period': trial.suggest_int('min_kama_period', 3, 54),
            'max_atr_period': trial.suggest_int('max_atr_period', 22, 233),
            'min_atr_period': trial.suggest_int('min_atr_period', 5, 21),
            'max_keltner_multiplier': trial.suggest_float('max_keltner_multiplier', 2.0, 4.0,step=0.1),
            'min_keltner_multiplier': trial.suggest_float('min_keltner_multiplier', 1.0, 2.0,step=0.1),
            'max_chop_period': trial.suggest_int('max_chop_period', 55, 144),
            'min_chop_period': trial.suggest_int('min_chop_period', 13, 54),
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
            'max_kama_period': int(params['max_kama_period']),
            'min_kama_period': int(params['min_kama_period']),
            'max_atr_period': int(params['max_atr_period']),
            'min_atr_period': int(params['min_atr_period']),
            'max_keltner_multiplier': int(params['max_keltner_multiplier']), 
            'min_keltner_multiplier': int(params['min_keltner_multiplier']),
            'lookback':1,
            'max_chop_period': int(params['max_chop_period']),
            'min_chop_period': int(params['min_chop_period']),
            'filter_threshold': 0.5
        } 