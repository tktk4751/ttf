#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import AlphaMomentumFilterSignalGenerator


class AlphaMomentumFilterStrategy(BaseStrategy):
    """
    アルファモメンタム+アルファフィルター戦略
    
    特徴:
    - 効率比（ER）に基づく動的パラメータ最適化
    - アルファモメンタムによる高精度なスクイーズモメンタム検出
    - アルファフィルターによる市場状態フィルタリング
    
    エントリー条件:
    - ロング: アルファモメンタムの買いシグナル + アルファフィルターがトレンド相場
    - ショート: アルファモメンタムの売りシグナル + アルファフィルターがトレンド相場
    
    エグジット条件:
    - ロング: アルファモメンタムの売りシグナル
    - ショート: アルファモメンタムの買いシグナル
    """
    
    def __init__(
        self,
        er_period: int = 25,
        # アルファモメンタム用パラメータ
        bb_max_period: int = 21,
        bb_min_period: int = 13,
        kc_max_period: int = 21,
        kc_min_period: int = 13,
        bb_max_mult: float = 2.0,
        bb_min_mult: float = 1,
        kc_max_mult: float = 3.0,
        kc_min_mult: float = 1,
        max_length: int = 100,
        min_length: int = 10,
        momentum_threshold: float = 0.0,
        # アルファフィルター用パラメータ
        max_chop_period: int = 144,
        min_chop_period: int = 21,
        max_adx_period: int = 21,
        min_adx_period: int = 5,
        alma_offset: float = 0.9,
        alma_sigma: float = 4,
        filter_threshold: float = 0.5
    ):
        """
        初期化
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            bb_max_period: ボリンジャーバンドの最大期間（デフォルト: 55）
            bb_min_period: ボリンジャーバンドの最小期間（デフォルト: 13）
            kc_max_period: ケルトナーチャネルの最大期間（デフォルト: 55）
            kc_min_period: ケルトナーチャネルの最小期間（デフォルト: 13）
            bb_max_mult: ボリンジャーバンドの最大乗数（デフォルト: 2.0）
            bb_min_mult: ボリンジャーバンドの最小乗数（デフォルト: 1.0）
            kc_max_mult: ケルトナーチャネルの最大乗数（デフォルト: 3.0）
            kc_min_mult: ケルトナーチャネルの最小乗数（デフォルト: 1.0）
            max_length: モメンタム計算の最大期間（デフォルト: 34）
            min_length: モメンタム計算の最小期間（デフォルト: 8）
            momentum_threshold: モメンタムの閾値（デフォルト: 0.0）
            max_chop_period: アルファチョピネスの最大期間（デフォルト: 55）
            min_chop_period: アルファチョピネスの最小期間（デフォルト: 8）
            max_adx_period: アルファADXの最大期間（デフォルト: 21）
            min_adx_period: アルファADXの最小期間（デフォルト: 5）
            alma_offset: ALMAのオフセット（デフォルト: 0.85）
            alma_sigma: ALMAのシグマ（デフォルト: 6）
            filter_threshold: アルファフィルターのしきい値（デフォルト: 0.5）
        """
        super().__init__("AlphaMomentumFilter")
        
        # パラメータの設定
        self._parameters = {
            'er_period': er_period,
            'bb_max_period': bb_max_period,
            'bb_min_period': bb_min_period,
            'kc_max_period': kc_max_period,
            'kc_min_period': kc_min_period,
            'bb_max_mult': bb_max_mult,
            'bb_min_mult': bb_min_mult,
            'kc_max_mult': kc_max_mult,
            'kc_min_mult': kc_min_mult,
            'max_length': max_length,
            'min_length': min_length,
            'momentum_threshold': momentum_threshold,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'filter_threshold': filter_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = AlphaMomentumFilterSignalGenerator(
            er_period=er_period,
            bb_max_period=bb_max_period,
            bb_min_period=bb_min_period,
            kc_max_period=kc_max_period,
            kc_min_period=kc_min_period,
            bb_max_mult=bb_max_mult,
            bb_min_mult=bb_min_mult,
            kc_max_mult=kc_max_mult,
            kc_min_mult=kc_min_mult,
            max_length=max_length,
            min_length=min_length,
            momentum_threshold=momentum_threshold,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_adx_period=max_adx_period,
            min_adx_period=min_adx_period,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma,
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
            'bb_max_period': trial.suggest_int('bb_max_period', 14, 55),
            'bb_min_period': trial.suggest_int('bb_min_period', 3, 13),
            'kc_max_period': trial.suggest_int('kc_max_period', 14, 55),
            'kc_min_period': trial.suggest_int('kc_min_period', 3, 13),
            'bb_max_mult': trial.suggest_float('bb_max_mult', 1.6, 3.0, step=0.1),
            'bb_min_mult': trial.suggest_float('bb_min_mult', 0.1, 1.5, step=0.1),
            'kc_max_mult': trial.suggest_float('kc_max_mult', 1.6, 3.0, step=0.1),
            'kc_min_mult': trial.suggest_float('kc_min_mult', 0.1, 1.5, step=0.1),
            'max_length': trial.suggest_int('max_length', 35, 100),
            'min_length': trial.suggest_int('min_length', 3, 34),
            'max_chop_period': trial.suggest_int('max_chop_period', 56, 144),
            'min_chop_period': trial.suggest_int('min_chop_period', 13, 55),
            'max_adx_period': trial.suggest_int('max_adx_period', 5, 34),
            'min_adx_period': trial.suggest_int('min_adx_period', 2, 13),
            'alma_offset': trial.suggest_float('alma_offset', 0.3, 0.9, step=0.1),
            'alma_sigma': trial.suggest_int('alma_sigma', 2, 9),
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
            'bb_max_period': int(params['bb_max_period']),
            'bb_min_period': int(params['bb_min_period']),
            'kc_max_period': int(params['kc_max_period']),
            'kc_min_period': int(params['kc_min_period']),
            'bb_max_mult': float(params['bb_max_mult']),
            'bb_min_mult': float(params['bb_min_mult']),
            'kc_max_mult': float(params['kc_max_mult']),
            'kc_min_mult': float(params['kc_min_mult']),
            'max_length': int(params['max_length']),
            'min_length': int(params['min_length']),
            'momentum_threshold': 0.0,
            'max_chop_period': int(params['max_chop_period']),
            'min_chop_period': int(params['min_chop_period']),
            'max_adx_period': int(params['max_adx_period']),
            'min_adx_period': int(params['min_adx_period']),
            'alma_offset': float(params['alma_offset']),
            'alma_sigma': int(params['alma_sigma']),
            'filter_threshold': 0.5
        } 