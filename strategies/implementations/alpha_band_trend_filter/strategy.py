#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import AlphaBandTrendFilterSignalGenerator


class AlphaBandTrendFilterStrategy(BaseStrategy):
    """
    アルファバンド+アルファトレンドフィルター戦略
    
    特徴:
    - サイクル効率比（CER）に基づく動的パラメータ最適化
    - アルファバンドによる高精度なエントリーポイント検出
    - アルファトレンドフィルターによる市場状態フィルタリング
    
    エントリー条件:
    - ロング: アルファバンドの買いシグナル + アルファトレンドフィルターがトレンド相場
    - ショート: アルファバンドの売りシグナル + アルファトレンドフィルターがトレンド相場
    
    エグジット条件:
    - ロング: アルファバンドの売りシグナル
    - ショート: アルファバンドの買いシグナル
    """
    
    def __init__(
        self,
        # アルファバンドのパラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_kama_period: int = 34,
        min_kama_period: int = 5,
        max_atr_period: int = 34,
        min_atr_period: int = 5,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.0,
        smoother_type: str = 'alma',
        band_lookback: int = 1,
        # アルファトレンドフィルターのパラメータ
        max_chop_period: int = 34,
        min_chop_period: int = 5,
        max_filter_atr_period: int = 34,
        min_filter_atr_period: int = 5,
        max_stddev_period: int = 13,
        min_stddev_period: int = 5,
        max_lookback_period: int = 13,
        min_lookback_period: int = 5,
        max_rms_window: int = 13,
        min_rms_window: int = 5,
        max_threshold: float = 0.8,
        min_threshold: float = 0.6,
        combination_weight: float = 0.6,
        combination_method: str = "sigmoid"
    ):
        """
        初期化
        
        Args:
            cycle_detector_type: サイクル検出器の種類（デフォルト: 'hody_dc'）
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_kama_period: AlphaMAの最大期間（デフォルト: 55）
            min_kama_period: AlphaMAの最小期間（デフォルト: 8）
            max_atr_period: AlphaATRの最大期間（デフォルト: 55）
            min_atr_period: AlphaATRの最小期間（デフォルト: 8）
            max_multiplier: ケルトナーチャネルの最大乗数（デフォルト: 3.0）
            min_multiplier: ケルトナーチャネルの最小乗数（デフォルト: 1.5）
            smoother_type: 平滑化アルゴリズム（デフォルト: 'alma'）
            band_lookback: 過去バンド参照期間（デフォルト: 1）
            max_chop_period: チョピネス期間の最大値（デフォルト: 89）
            min_chop_period: チョピネス期間の最小値（デフォルト: 21）
            max_filter_atr_period: フィルターATR期間の最大値（デフォルト: 89）
            min_filter_atr_period: フィルターATR期間の最小値（デフォルト: 13）
            max_stddev_period: 標準偏差期間の最大値（デフォルト: 13）
            min_stddev_period: 標準偏差期間の最小値（デフォルト: 5）
            max_lookback_period: ルックバック期間の最大値（デフォルト: 13）
            min_lookback_period: ルックバック期間の最小値（デフォルト: 5）
            max_rms_window: RMSウィンドウの最大値（デフォルト: 13）
            min_rms_window: RMSウィンドウの最小値（デフォルト: 5）
            max_threshold: フィルターしきい値の最大値（デフォルト: 0.75）
            min_threshold: フィルターしきい値の最小値（デフォルト: 0.55）
            combination_weight: 組み合わせの重み（デフォルト: 0.6）
            combination_method: 組み合わせメソッド（デフォルト: "sigmoid"）
        """
        super().__init__("AlphaBandTrendFilter")
        
        # パラメータの設定
        self._parameters = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_kama_period': max_kama_period,
            'min_kama_period': min_kama_period,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'smoother_type': smoother_type,
            'band_lookback': band_lookback,
            'max_chop_period': max_chop_period,
            'min_chop_period': min_chop_period,
            'max_filter_atr_period': max_filter_atr_period,
            'min_filter_atr_period': min_filter_atr_period,
            'max_stddev_period': max_stddev_period,
            'min_stddev_period': min_stddev_period,
            'max_lookback_period': max_lookback_period,
            'min_lookback_period': min_lookback_period,
            'max_rms_window': max_rms_window,
            'min_rms_window': min_rms_window,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold,
            'combination_weight': combination_weight,
            'combination_method': combination_method
        }
        
        # シグナル生成器の初期化
        self.signal_generator = AlphaBandTrendFilterSignalGenerator(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            smoother_type=smoother_type,
            band_lookback=band_lookback,
            max_chop_period=max_chop_period,
            min_chop_period=min_chop_period,
            max_filter_atr_period=max_filter_atr_period,
            min_filter_atr_period=min_filter_atr_period,
            max_stddev_period=max_stddev_period,
            min_stddev_period=min_stddev_period,
            max_lookback_period=max_lookback_period,
            min_lookback_period=min_lookback_period,
            max_rms_window=max_rms_window,
            min_rms_window=min_rms_window,
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            combination_weight=combination_weight,
            combination_method=combination_method
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
            # アルファバンドのパラメータ
            'hp_period': trial.suggest_int('hp_period', 62, 233),
            'max_kama_period': trial.suggest_int('max_kama_period', 34, 144),
            'min_kama_period': trial.suggest_int('min_kama_period', 5, 21),
            'max_atr_period': trial.suggest_int('max_atr_period', 34, 144),
            'min_atr_period': trial.suggest_int('min_atr_period', 5, 21),
            'max_multiplier': trial.suggest_float('max_multiplier', 2.0, 4.0, step=0.1),
            'min_multiplier': trial.suggest_float('min_multiplier', 1.0, 2.0, step=0.1),
            # アルファトレンドフィルターのパラメータ
            'max_chop_period': trial.suggest_int('max_chop_period', 55, 144),
            'min_chop_period': trial.suggest_int('min_chop_period', 13, 54),
            'max_filter_atr_period': trial.suggest_int('max_filter_atr_period', 55, 144),
            'min_filter_atr_period': trial.suggest_int('min_filter_atr_period', 13, 54),
            'max_threshold': trial.suggest_float('max_threshold', 0.6, 0.85, step=0.05),
            'min_threshold': trial.suggest_float('min_threshold', 0.45, 0.6, step=0.05),
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
        strategy_params = {
            'cycle_detector_type': 'hody_dc',
            'lp_period': 5,
            'hp_period': int(params['hp_period']),
            'cycle_part': 0.5,
            'max_kama_period': int(params['max_kama_period']),
            'min_kama_period': int(params['min_kama_period']),
            'max_atr_period': int(params['max_atr_period']),
            'min_atr_period': int(params['min_atr_period']),
            'max_multiplier': float(params['max_multiplier']),
            'min_multiplier': float(params['min_multiplier']),
            'smoother_type': 'alma',
            'band_lookback': 1,
            'max_chop_period': int(params['max_chop_period']),
            'min_chop_period': int(params['min_chop_period']),
            'max_filter_atr_period': int(params['max_filter_atr_period']),
            'min_filter_atr_period': int(params['min_filter_atr_period']),
            'max_stddev_period': 13,
            'min_stddev_period': 5,
            'max_lookback_period': 13,
            'min_lookback_period': 5,
            'max_rms_window': 13,
            'min_rms_window': 5,
            'max_threshold': float(params['max_threshold']),
            'min_threshold': float(params['min_threshold']),
            'combination_weight': 0.6,
            'combination_method': 'sigmoid'
        }
        return strategy_params 