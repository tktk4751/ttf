#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import KernelMAFilterSignalGenerator


class KernelMAFilterStrategy(BaseStrategy):
    """
    カーネルMA+アルファフィルター戦略
    
    特徴:
    - 効率比（ER）に基づく動的バンド幅調整
    - カーネル回帰法による非パラメトリックな平滑化
    - アルファフィルターによる市場状態フィルタリング
    
    エントリー条件:
    - ロング: カーネルMAが上昇トレンド + アルファフィルターがトレンド相場
    - ショート: カーネルMAが下降トレンド + アルファフィルターがトレンド相場
    
    エグジット条件:
    - ロング: カーネルMAが下降トレンドまたは横ばい
    - ショート: カーネルMAが上昇トレンドまたは横ばい
    """
    
    def __init__(
        self,
        er_period: int = 144,
        # カーネルMA用パラメータ
        max_bandwidth: float = 20.0,
        min_bandwidth: float = 1.5,
        kernel_type: str = 'gaussian',
        confidence_level: float = 0.95,
        slope_period: int = 6,
        slope_threshold: float = 0.0009,
        # アルファフィルター用パラメータ
        max_chop_period: int = 56,
        min_chop_period: int = 29,
        max_adx_period: int = 20,
        min_adx_period: int = 13,
        alma_offset: float = 0.9,
        alma_sigma: float = 6,
        filter_threshold: float = 0.5
    ):
        """
        初期化
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_bandwidth: バンド幅の最大値（デフォルト: 10.0）
            min_bandwidth: バンド幅の最小値（デフォルト: 2.0）
            kernel_type: カーネルの種類（'gaussian'または'epanechnikov'）（デフォルト: 'gaussian'）
            confidence_level: 信頼区間のレベル（デフォルト: 0.95）
            slope_period: 傾きを計算する期間（デフォルト: 5）
            slope_threshold: トレンド判定の傾き閾値（デフォルト: 0.0001）
            max_chop_period: アルファチョピネスの最大期間（デフォルト: 55）
            min_chop_period: アルファチョピネスの最小期間（デフォルト: 8）
            max_adx_period: アルファADXの最大期間（デフォルト: 21）
            min_adx_period: アルファADXの最小期間（デフォルト: 5）
            alma_offset: ALMAのオフセット（デフォルト: 0.85）
            alma_sigma: ALMAのシグマ（デフォルト: 6）
            filter_threshold: アルファフィルターのしきい値（デフォルト: 0.5）
        """
        super().__init__("KernelMAFilter")
        
        # パラメータの設定
        self._parameters = {
            'er_period': er_period,
            'max_bandwidth': max_bandwidth,
            'min_bandwidth': min_bandwidth,
            'kernel_type': kernel_type,
            'confidence_level': confidence_level,
            'slope_period': slope_period,
            'slope_threshold': slope_threshold,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_adx_period': max_adx_period,
            'min_adx_period': min_adx_period,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'filter_threshold': filter_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = KernelMAFilterSignalGenerator(
            er_period=er_period,
            max_bandwidth=max_bandwidth,
            min_bandwidth=min_bandwidth,
            kernel_type=kernel_type,
            confidence_level=confidence_level,
            slope_period=slope_period,
            slope_threshold=slope_threshold,
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
            'max_bandwidth': trial.suggest_float('max_bandwidth', 5.0, 20.0, step=1.0),
            'min_bandwidth': trial.suggest_float('min_bandwidth', 1.0, 5.0, step=0.5),
            'kernel_type': trial.suggest_categorical('kernel_type', ['gaussian', 'epanechnikov']),
            'slope_period': trial.suggest_int('slope_period', 3, 10),
            'slope_threshold': trial.suggest_float('slope_threshold', 0.00005, 0.001, log=True),
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
            'max_bandwidth': float(params['max_bandwidth']),
            'min_bandwidth': float(params['min_bandwidth']),
            'kernel_type': str(params['kernel_type']),
            'confidence_level': 0.95,  # 固定値
            'slope_period': int(params['slope_period']),
            'slope_threshold': float(params['slope_threshold']),
            'max_chop_period': int(params['max_chop_period']),
            'min_chop_period': int(params['min_chop_period']),
            'max_adx_period': int(params['max_adx_period']),
            'min_adx_period': int(params['min_adx_period']),
            'alma_offset': float(params['alma_offset']),
            'alma_sigma': int(params['alma_sigma']),
            'filter_threshold': 0.5  # 固定値
        } 