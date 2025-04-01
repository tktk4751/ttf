#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import AlphaMAV2KeltnerFilterSignalGenerator


class AlphaMAV2KeltnerFilterStrategy(BaseStrategy):
    """
    アルファMAV2ケルトナーチャネル+アルファフィルター戦略
    
    特徴:
    - RSXの3段階平滑化を採用したAlphaMAV2を中心線として使用
    - 効率比（ER）に基づく動的パラメータ最適化
    - アルファMAV2ケルトナーチャネルによる高精度なエントリーポイント検出
    - アルファフィルターによる市場状態フィルタリング
    
    エントリー条件:
    - ロング: アルファMAV2ケルトナーチャネルの買いシグナル + アルファフィルターがトレンド相場
    - ショート: アルファMAV2ケルトナーチャネルの売りシグナル + アルファフィルターがトレンド相場
    
    エグジット条件:
    - ロング: アルファMAV2ケルトナーチャネルの売りシグナル
    - ショート: アルファMAV2ケルトナーチャネルの買いシグナル
    """
    
    def __init__(
        self,
        ma_er_period: int = 10,
        ma_max_period: int = 34,
        ma_min_period: int = 5,
        atr_er_period: int = 21,
        atr_max_period: int = 89,
        atr_min_period: int = 13,
        max_keltner_multiplier: float = 2.0,
        min_keltner_multiplier: float = 2.0,
        lookback: int = 1,
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        max_adx_period: int = 21,
        min_adx_period: int = 5,
        filter_threshold: float = 0.5
    ):
        """
        初期化
        
        Args:
            ma_er_period: AlphaMAV2の効率比計算期間（デフォルト: 10）
            ma_max_period: AlphaMAV2の最大期間（デフォルト: 34）
            ma_min_period: AlphaMAV2の最小期間（デフォルト: 5）
            atr_er_period: AlphaATRの効率比計算期間（デフォルト: 21）
            atr_max_period: AlphaATRの最大期間（デフォルト: 89）
            atr_min_period: AlphaATRの最小期間（デフォルト: 13）
            max_keltner_multiplier: ケルトナーチャネルの最大乗数（デフォルト: 2.0）
            min_keltner_multiplier: ケルトナーチャネルの最小乗数（デフォルト: 2.0）
            lookback: 過去バンド参照期間（デフォルト: 1）
            max_chop_period: アルファチョピネスの最大期間（デフォルト: 55）
            min_chop_period: アルファチョピネスの最小期間（デフォルト: 8）
            max_adx_period: アルファADXの最大期間（デフォルト: 21）
            min_adx_period: アルファADXの最小期間（デフォルト: 5）
            filter_threshold: アルファフィルターのしきい値（デフォルト: 0.5）
        """
        super().__init__("AlphaMAV2KeltnerFilter")
        
        # パラメータの設定
        self._parameters = {
            'ma_er_period': ma_er_period,
            'ma_max_period': ma_max_period,
            'ma_min_period': ma_min_period,
            'atr_er_period': atr_er_period,
            'atr_max_period': atr_max_period,
            'atr_min_period': atr_min_period,
            'max_keltner_multiplier': max_keltner_multiplier,
            'min_keltner_multiplier': min_keltner_multiplier,
            'lookback': lookback,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'filter_threshold': filter_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = AlphaMAV2KeltnerFilterSignalGenerator(
            ma_er_period=ma_er_period,
            ma_max_period=ma_max_period,
            ma_min_period=ma_min_period,
            atr_er_period=atr_er_period,
            atr_max_period=atr_max_period,
            atr_min_period=atr_min_period,
            max_keltner_multiplier=max_keltner_multiplier,
            min_keltner_multiplier=min_keltner_multiplier,
            lookback=lookback,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=max_adx_period,
            min_adx_period=min_adx_period,
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
            'ma_er_period': trial.suggest_int('ma_er_period', 5, 200),
            'ma_max_period': trial.suggest_int('ma_max_period', 21, 233),
            'ma_min_period': trial.suggest_int('ma_min_period', 3, 21),
            'atr_er_period': trial.suggest_int('atr_er_period', 5, 55),
            'atr_max_period': trial.suggest_int('atr_max_period', 55, 144),
            'atr_min_period': trial.suggest_int('atr_min_period', 8, 34),
            'max_keltner_multiplier': trial.suggest_float('max_keltner_multiplier', 1.5, 3.0, step=0.1),
            'min_keltner_multiplier': trial.suggest_float('min_keltner_multiplier', 1.5, 3.0, step=0.1),
            'max_chop_period': trial.suggest_int('max_chop_period', 34, 144),
            'min_chop_period': trial.suggest_int('min_chop_period', 8, 34),
            'max_adx_period': trial.suggest_int('max_adx_period', 13, 55),
            'min_adx_period': trial.suggest_int('min_adx_period', 3, 13),
            'filter_threshold': trial.suggest_float('filter_threshold', 0.3, 0.7, step=0.05),
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
            'ma_er_period': int(params['ma_er_period']),
            'ma_max_period': int(params['ma_max_period']),
            'ma_min_period': int(params['ma_min_period']),
            'atr_er_period': int(params['atr_er_period']),
            'atr_max_period': int(params['atr_max_period']),
            'atr_min_period': int(params['atr_min_period']),
            'max_keltner_multiplier': float(params['max_keltner_multiplier']),
            'min_keltner_multiplier': float(params['min_keltner_multiplier']),
            'lookback': 1,
            'max_chop_period': int(params['max_chop_period']),
            'min_chop_period': int(params['min_chop_period']),
            'max_adx_period': int(params['max_adx_period']),
            'min_adx_period': int(params['min_adx_period']),
            'filter_threshold': float(params['filter_threshold'])
        } 