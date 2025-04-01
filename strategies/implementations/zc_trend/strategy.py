#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZCTrendSignalGenerator


class ZCTrendStrategy(BaseStrategy):
    """
    Zチャネル&トレンドインデックス戦略
    
    特徴:
    - Zチャネルのブレイクアウトシグナルによる高精度なエントリーポイント検出
    - Zトレンドインデックスによるトレンド/レンジ相場の判別
    - トレンド相場のみでエントリーし、レンジ相場ではエントリーしない
    
    エントリー条件:
    - ロング: Zチャネルの買いシグナル(1) かつ Zトレンドインデックスがトレンド相場(1)
    - ショート: Zチャネルの売りシグナル(-1) かつ Zトレンドインデックスがトレンド相場(1)
    
    エグジット条件:
    - ロング: Zチャネルの売りシグナル(-1)
    - ショート: Zチャネルの買いシグナル(1)
    """
    
    def __init__(
        self,
        # Zチャネルのパラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.5,
        smoother_type: str = 'alma',
        src_type: str = 'hlc3',
        band_lookback: int = 1,
        
        # Zトレンドインデックスのパラメータ
        max_chop_dc_cycle_part: float = 0.8,
        max_chop_dc_max_cycle: int = 233,
        max_chop_dc_min_cycle: int = 34,
        max_chop_dc_max_output: int = 144,
        max_chop_dc_min_output: int = 55,
        min_chop_dc_cycle_part: float = 0.5,
        min_chop_dc_max_cycle: int = 144,
        min_chop_dc_min_cycle: int = 55,
        min_chop_dc_max_output: int = 55,
        min_chop_dc_min_output: int = 21,
        max_stddev_period: int = 21,
        min_stddev_period: int = 13,
        max_lookback_period: int = 13,
        min_lookback_period: int = 10,
        max_threshold: float = 0.8,
        min_threshold: float = 0.5
    ):
        """
        初期化
        
        Args:
            cycle_detector_type: サイクル検出器の種類（デフォルト: 'hody_dc'）
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_multiplier: ケルトナーチャネルの最大乗数（デフォルト: 3.0）
            min_multiplier: ケルトナーチャネルの最小乗数（デフォルト: 1.5）
            smoother_type: 平滑化アルゴリズム（デフォルト: 'alma'）
            src_type: 価格ソースタイプ（デフォルト: 'hlc3'）
            band_lookback: 過去バンド参照期間（デフォルト: 1）
            max_chop_dc_cycle_part: 最大チョピネス期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            max_chop_dc_max_cycle: 最大チョピネス期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 144）
            max_chop_dc_min_cycle: 最大チョピネス期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 10）
            max_chop_dc_max_output: 最大チョピネス期間用ドミナントサイクル計算用の最大出力値（デフォルト: 34）
            max_chop_dc_min_output: 最大チョピネス期間用ドミナントサイクル計算用の最小出力値（デフォルト: 13）
            min_chop_dc_cycle_part: 最小チョピネス期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.25）
            min_chop_dc_max_cycle: 最小チョピネス期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            min_chop_dc_min_cycle: 最小チョピネス期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            min_chop_dc_max_output: 最小チョピネス期間用ドミナントサイクル計算用の最大出力値（デフォルト: 13）
            min_chop_dc_min_output: 最小チョピネス期間用ドミナントサイクル計算用の最小出力値（デフォルト: 5）
            max_stddev_period: 標準偏差期間の最大値（デフォルト: 21）
            min_stddev_period: 標準偏差期間の最小値（デフォルト: 14）
            max_lookback_period: 標準偏差の最小値を探す期間の最大値（デフォルト: 14）
            min_lookback_period: 標準偏差の最小値を探す期間の最小値（デフォルト: 7）
            max_threshold: しきい値の最大値（デフォルト: 0.75）
            min_threshold: しきい値の最小値（デフォルト: 0.55）
        """
        super().__init__("ZCTrend")
        
        # パラメータの設定
        self._parameters = {
            # Zチャネルのパラメータ
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            
            # Zトレンドインデックスのパラメータ
            'max_chop_dc_cycle_part': max_chop_dc_cycle_part,
            'max_chop_dc_max_cycle': max_chop_dc_max_cycle,
            'max_chop_dc_min_cycle': max_chop_dc_min_cycle,
            'max_chop_dc_max_output': max_chop_dc_max_output,
            'max_chop_dc_min_output': max_chop_dc_min_output,
            'min_chop_dc_cycle_part': min_chop_dc_cycle_part,
            'min_chop_dc_max_cycle': min_chop_dc_max_cycle,
            'min_chop_dc_min_cycle': min_chop_dc_min_cycle,
            'min_chop_dc_max_output': min_chop_dc_max_output,
            'min_chop_dc_min_output': min_chop_dc_min_output,
            'max_stddev_period': max_stddev_period,
            'min_stddev_period': min_stddev_period,
            'max_lookback_period': max_lookback_period,
            'min_lookback_period': min_lookback_period,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZCTrendSignalGenerator(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            smoother_type=smoother_type,
            src_type=src_type,
            band_lookback=band_lookback,
            max_chop_dc_cycle_part=max_chop_dc_cycle_part,
            max_chop_dc_max_cycle=max_chop_dc_max_cycle,
            max_chop_dc_min_cycle=max_chop_dc_min_cycle,
            max_chop_dc_max_output=max_chop_dc_max_output,
            max_chop_dc_min_output=max_chop_dc_min_output,
            min_chop_dc_cycle_part=min_chop_dc_cycle_part,
            min_chop_dc_max_cycle=min_chop_dc_max_cycle,
            min_chop_dc_min_cycle=min_chop_dc_min_cycle,
            min_chop_dc_max_output=min_chop_dc_max_output,
            min_chop_dc_min_output=min_chop_dc_min_output,
            max_stddev_period=max_stddev_period,
            min_stddev_period=min_stddev_period,
            max_lookback_period=max_lookback_period,
            min_lookback_period=min_lookback_period,
            max_threshold=max_threshold,
            min_threshold=min_threshold
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
            # Zチャネルのパラメータ
            'hp_period': trial.suggest_int('hp_period', 62, 233),
            'max_multiplier': trial.suggest_float('max_multiplier', 2.0, 4.0, step=0.1),
            'min_multiplier': trial.suggest_float('min_multiplier', 0.0, 2.0, step=0.1),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # Zトレンドインデックスのパラメータ
            'max_chop_dc_cycle_part': trial.suggest_float('max_chop_dc_cycle_part', 0.25, 0.9, step=0.05),
            'max_chop_dc_max_cycle': trial.suggest_int('max_chop_dc_max_cycle', 10, 377),
            'max_chop_dc_min_cycle': trial.suggest_int('max_chop_dc_min_cycle', 5, 89),
            'max_chop_dc_max_output': trial.suggest_int('max_chop_dc_max_output', 13, 233),
            'max_chop_dc_min_output': trial.suggest_int('max_chop_dc_min_output', 5, 89),
            'min_chop_dc_cycle_part': trial.suggest_float('min_chop_dc_cycle_part', 0.25, 0.75, step=0.05),
            'min_chop_dc_max_cycle': trial.suggest_int('min_chop_dc_max_cycle', 2, 233),
            'min_chop_dc_min_cycle': trial.suggest_int('min_chop_dc_min_cycle', 5, 89),
            'min_chop_dc_max_output': trial.suggest_int('min_chop_dc_max_output', 13, 89),
            'min_chop_dc_min_output': trial.suggest_int('min_chop_dc_min_output', 5, 34),
            'max_stddev_period': trial.suggest_int('max_stddev_period', 14, 21),
            'min_stddev_period': trial.suggest_int('min_stddev_period', 7, 14),
            'max_lookback_period': trial.suggest_int('max_lookback_period', 7, 14),
            'min_lookback_period': trial.suggest_int('min_lookback_period', 7, 14),
            'max_threshold': trial.suggest_float('max_threshold', 0.65, 0.9, step=0.05),
            'min_threshold': trial.suggest_float('min_threshold', 0.45, 0.8, step=0.05),
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
            # Zチャネルのパラメータ
            'cycle_detector_type': 'hody_dc',
            'lp_period': 5,
            'hp_period': int(params['hp_period']),
            'cycle_part': 0.5,
            'max_multiplier': float(params['max_multiplier']),
            'min_multiplier': float(params['min_multiplier']),
            'smoother_type': 'alma',
            'src_type': params['src_type'],
            'band_lookback': 1,
            
            # Zトレンドインデックスのパラメータ
            'max_chop_dc_cycle_part': 0.5,
            'max_chop_dc_max_cycle': 144,
            'max_chop_dc_min_cycle': 10,
            'max_chop_dc_max_output': 34,
            'max_chop_dc_min_output': 13,
            'min_chop_dc_cycle_part': 0.25,
            'min_chop_dc_max_cycle': 55,
            'min_chop_dc_min_cycle': 5,
            'min_chop_dc_max_output': 13,
            'min_chop_dc_min_output': 5,
            'max_stddev_period': 21,
            'min_stddev_period': 14,
            'max_lookback_period': 14,
            'min_lookback_period': 7,
            'max_threshold': float(params['max_threshold']),
            'min_threshold': float(params['min_threshold'])
        }
        
        return strategy_params 