#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna
import logging

from ...base.strategy import BaseStrategy
from .signal_generator import AlphaROCDivergenceSignalGenerator


logger = logging.getLogger(__name__)

class AlphaROCDivergenceStrategy(BaseStrategy):
    """
    AlphaROCダイバージェンス戦略
    
    特徴:
    - 効率比（ER）に基づく動的パラメータ最適化
    - AlphaMADirectionSignal2による方向性の確認
    - AlphaROCDivergenceSignalによるダイバージェンス検出
    - AlphaFilterSignalによる市場状態フィルタリング
    - BollingerBreakoutExitSignalによる利益確定
    
    エントリー条件:
    - ロング: AlphaMADirectionSignal2が-1 + AlphaROCDivergenceSignalが1 + AlphaFilterSignalが1
    - ショート: AlphaMADirectionSignal2が1 + AlphaROCDivergenceSignalが-1 + AlphaFilterSignalが1
    
    エグジット条件:
    - ロング: BollingerBreakoutExitSignalが1 または AlphaROCDivergenceSignalが-1
    - ショート: BollingerBreakoutExitSignalが-1 または AlphaROCDivergenceSignalが1
    """
    
    def __init__(
        self,
        # 共通パラメータ
        er_period: int = 21,
        # AlphaMA用パラメータ
        max_ma_period: int = 200,
        min_ma_period: int = 20,
        alma_offset: float = 0.85,
        alma_sigma: float = 6,
        # AlphaROC用パラメータ
        max_roc_period: int = 50,
        min_roc_period: int = 5,
        lookback: int = 30,
        # アルファフィルター用パラメータ
        max_chop_period: int = 55,
        min_chop_period: int = 8,
        max_adx_period: int = 21,
        min_adx_period: int = 5,
        filter_threshold: float = 0.5,
        # ボリンジャーバンド用パラメータ
        bb_period: int = 20,
        bb_std_dev: float = 2.0
    ):
        """
        初期化
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_ma_period: AlphaMAの最大期間（デフォルト: 200）
            min_ma_period: AlphaMAの最小期間（デフォルト: 20）
            alma_offset: ALMAのオフセット（デフォルト: 0.85）
            alma_sigma: ALMAのシグマ（デフォルト: 6）
            max_roc_period: AlphaROCの最大期間（デフォルト: 50）
            min_roc_period: AlphaROCの最小期間（デフォルト: 5）
            lookback: ダイバージェンス検出のルックバック期間（デフォルト: 30）
            max_chop_period: アルファチョピネスの最大期間（デフォルト: 55）
            min_chop_period: アルファチョピネスの最小期間（デフォルト: 8）
            max_adx_period: アルファADXの最大期間（デフォルト: 21）
            min_adx_period: アルファADXの最小期間（デフォルト: 5）
            filter_threshold: アルファフィルターのしきい値（デフォルト: 0.5）
            bb_period: ボリンジャーバンドの期間（デフォルト: 20）
            bb_std_dev: ボリンジャーバンドの標準偏差（デフォルト: 2.0）
        """
        super().__init__("AlphaROCDivergence")
        
        # ロガーの設定
        self.logger = logger
        
        # パラメータの設定
        self._parameters = {
            'er_period': er_period,
            'max_ma_period': max_ma_period,
            'min_ma_period': min_ma_period,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'max_roc_period': max_roc_period,
            'min_roc_period': min_roc_period,
            'lookback': lookback,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'filter_threshold': filter_threshold,
            'bb_period': bb_period,
            'bb_std_dev': bb_std_dev
        }
        
        # シグナル生成器の初期化
        self.signal_generator = AlphaROCDivergenceSignalGenerator(
            er_period=er_period,
            max_ma_period=max_ma_period,
            min_ma_period=min_ma_period,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma,
            max_roc_period=max_roc_period,
            min_roc_period=min_roc_period,
            lookback=lookback,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=max_adx_period,
            min_adx_period=min_adx_period,
            filter_threshold=filter_threshold,
            bb_period=bb_period,
            bb_std_dev=bb_std_dev
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
            'max_ma_period': trial.suggest_int('max_ma_period', 100, 300),
            'min_ma_period': trial.suggest_int('min_ma_period', 5, 50),
            'alma_offset': trial.suggest_float('alma_offset', 0.3, 0.9, step=0.1),
            'alma_sigma': trial.suggest_int('alma_sigma', 2, 9),
            'max_roc_period': trial.suggest_int('max_roc_period', 30, 100),
            'min_roc_period': trial.suggest_int('min_roc_period', 2, 20),
            'lookback': trial.suggest_int('lookback', 10, 50),
            'max_chop_period': trial.suggest_int('max_chop_period', 56, 144),
            'min_chop_period': trial.suggest_int('min_chop_period', 13, 55),
            'max_adx_period': trial.suggest_int('max_adx_period', 5, 34),
            'min_adx_period': trial.suggest_int('min_adx_period', 2, 13),
            'filter_threshold': trial.suggest_float('filter_threshold', 0.3, 0.7, step=0.1),
            'bb_period': trial.suggest_int('bb_period', 10, 50),
            'bb_std_dev': trial.suggest_float('bb_std_dev', 1.0, 3.0, step=0.1)
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
            'max_ma_period': int(params['max_ma_period']),
            'min_ma_period': int(params['min_ma_period']),
            'alma_offset': float(params['alma_offset']),
            'alma_sigma': int(params['alma_sigma']),
            'max_roc_period': int(params['max_roc_period']),
            'min_roc_period': int(params['min_roc_period']),
            'lookback': int(params['lookback']),
            'max_chop_period': int(params['max_chop_period']),
            'min_chop_period': int(params['min_chop_period']),
            'max_adx_period': int(params['max_adx_period']),
            'min_adx_period': int(params['min_adx_period']),
            'filter_threshold': float(params['filter_threshold']),
            'bb_period': int(params['bb_period']),
            'bb_std_dev': float(params['bb_std_dev'])
        } 