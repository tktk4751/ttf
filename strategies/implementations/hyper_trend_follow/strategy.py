#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HyperTrendFollow戦略

HyperFRAMA位置関係とHyperFRAMAChannelブレイクアウトを組み合わせたトレンドフォロー戦略
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import HyperTrendFollowSignalGenerator


class HyperTrendFollowStrategy(BaseStrategy):
    """
    HyperTrendFollow戦略
    
    特徴:
    - HyperFRAMA位置関係シグナルとHyperFRAMAChannelブレイクアウトシグナルの組み合わせ
    - トレンドフォロー型戦略
    - 包括的なパラメータサポート
    
    エントリー条件:
    - ロング: 位置関係シグナル=1 かつ ブレイクアウトシグナル=1
    - ショート: 位置関係シグナル=-1 かつ ブレイクアウトシグナル=-1
    
    エグジット条件:
    - ロング: ブレイクアウトシグナル=-1
    - ショート: ブレイクアウトシグナル=1
    """
    
    def __init__(
        self,
        # HyperFRAMAパラメータ
        hyper_frama_period: int = 6,
        hyper_frama_src_type: str = 'oc2',
        hyper_frama_fc: int = 4,
        hyper_frama_sc: int = 120,
        hyper_frama_alpha_multiplier: float = 0.12,
        # 動的期間パラメータ
        hyper_frama_period_mode: str = 'fixed',
        hyper_frama_cycle_detector_type: str = 'dft_dominant',
        hyper_frama_lp_period: int = 13,
        hyper_frama_hp_period: int = 124,
        hyper_frama_cycle_part: float = 0.5,
        hyper_frama_max_cycle: int = 89,
        hyper_frama_min_cycle: int = 8,
        hyper_frama_max_output: int = 124,
        hyper_frama_min_output: int = 8,
        
        # HyperFRAMAChannelブレイクアウトシグナルパラメータ
        # 基本パラメータ
        channel_band_lookback: int = 4,
        channel_exit_mode: int = 1,
        channel_src_type: str = 'oc2',
        channel_period: int = 14,
        channel_multiplier_mode: str = "fixed",
        channel_fixed_multiplier: float = 1.2,
        channel_hyper_frama_channel_src_type: str = "oc2",
        
        # HyperFRAMA パラメータ（チャネル用）
        channel_hyper_frama_period: int = 4,
        channel_hyper_frama_src_type: str = 'oc2',
        channel_hyper_frama_fc: int = 1,
        channel_hyper_frama_sc: int = 198,
        channel_hyper_frama_alpha_multiplier: float = 0.5,
        # 動的期間パラメータ（チャネル用）
        channel_hyper_frama_period_mode: str = 'fixed',
        channel_hyper_frama_cycle_detector_type: str = 'dft_dominant',
        channel_hyper_frama_lp_period: int = 13,
        channel_hyper_frama_hp_period: int = 124,
        channel_hyper_frama_cycle_part: float = 0.5,
        channel_hyper_frama_max_cycle: int = 89,
        channel_hyper_frama_min_cycle: int = 8,
        channel_hyper_frama_max_output: int = 124,
        channel_hyper_frama_min_output: int = 8,
        
        # X_ATR パラメータ（チャネル用）
        channel_x_atr_period: float = 12.0,
        channel_x_atr_tr_method: str = 'atr',
        channel_x_atr_smoother_type: str = 'frama',
        channel_x_atr_src_type: str = 'close',
        channel_x_atr_enable_kalman: bool = True,
        channel_x_atr_kalman_type: str = 'unscented',
        
        # HyperER パラメータ（チャネル用）
        channel_hyper_er_period: int = 8,
        channel_hyper_er_midline_period: int = 100,
        channel_hyper_er_er_period: int = 13,
        channel_hyper_er_er_src_type: str = 'oc2',
        channel_hyper_er_use_kalman_filter: bool = True,
        channel_hyper_er_kalman_filter_type: str = 'unscented',
        channel_hyper_er_kalman_process_noise: float = 1e-5,
        channel_hyper_er_kalman_min_observation_noise: float = 1e-6,
        channel_hyper_er_kalman_adaptation_window: int = 5,
        channel_hyper_er_use_roofing_filter: bool = True,
        channel_hyper_er_roofing_hp_cutoff: float = 55.0,
        channel_hyper_er_roofing_ss_band_edge: float = 10.0,
        channel_hyper_er_use_smoothing: bool = True,
        channel_hyper_er_smoother_type: str = 'laguerre',
        channel_hyper_er_smoother_period: int = 16
    ):
        """
        初期化
        
        Args:
            hyper_frama_*: HyperFRAMA位置関係シグナル用パラメータ
            channel_*: HyperFRAMAChannelブレイクアウトシグナル用パラメータ
        """
        
        super().__init__(f"HyperTrendFollow(frama={hyper_frama_period}, channel={channel_period})")
        
        # パラメータの設定
        self._parameters = {
            # HyperFRAMA位置関係シグナルパラメータ
            'hyper_frama_period': hyper_frama_period,
            'hyper_frama_src_type': hyper_frama_src_type,
            'hyper_frama_fc': hyper_frama_fc,
            'hyper_frama_sc': hyper_frama_sc,
            'hyper_frama_alpha_multiplier': hyper_frama_alpha_multiplier,
            'hyper_frama_period_mode': hyper_frama_period_mode,
            'hyper_frama_cycle_detector_type': hyper_frama_cycle_detector_type,
            'hyper_frama_lp_period': hyper_frama_lp_period,
            'hyper_frama_hp_period': hyper_frama_hp_period,
            'hyper_frama_cycle_part': hyper_frama_cycle_part,
            'hyper_frama_max_cycle': hyper_frama_max_cycle,
            'hyper_frama_min_cycle': hyper_frama_min_cycle,
            'hyper_frama_max_output': hyper_frama_max_output,
            'hyper_frama_min_output': hyper_frama_min_output,
            
            # HyperFRAMAChannelブレイクアウトシグナルパラメータ
            'channel_band_lookback': channel_band_lookback,
            'channel_exit_mode': channel_exit_mode,
            'channel_src_type': channel_src_type,
            'channel_period': channel_period,
            'channel_multiplier_mode': channel_multiplier_mode,
            'channel_fixed_multiplier': channel_fixed_multiplier,
            'channel_hyper_frama_channel_src_type': channel_hyper_frama_channel_src_type,
            'channel_hyper_frama_period': channel_hyper_frama_period,
            'channel_hyper_frama_src_type': channel_hyper_frama_src_type,
            'channel_hyper_frama_fc': channel_hyper_frama_fc,
            'channel_hyper_frama_sc': channel_hyper_frama_sc,
            'channel_hyper_frama_alpha_multiplier': channel_hyper_frama_alpha_multiplier,
            'channel_hyper_frama_period_mode': channel_hyper_frama_period_mode,
            'channel_hyper_frama_cycle_detector_type': channel_hyper_frama_cycle_detector_type,
            'channel_hyper_frama_lp_period': channel_hyper_frama_lp_period,
            'channel_hyper_frama_hp_period': channel_hyper_frama_hp_period,
            'channel_hyper_frama_cycle_part': channel_hyper_frama_cycle_part,
            'channel_hyper_frama_max_cycle': channel_hyper_frama_max_cycle,
            'channel_hyper_frama_min_cycle': channel_hyper_frama_min_cycle,
            'channel_hyper_frama_max_output': channel_hyper_frama_max_output,
            'channel_hyper_frama_min_output': channel_hyper_frama_min_output,
            
            # X_ATRパラメータ（チャネル用）
            'channel_x_atr_period': channel_x_atr_period,
            'channel_x_atr_tr_method': channel_x_atr_tr_method,
            'channel_x_atr_smoother_type': channel_x_atr_smoother_type,
            'channel_x_atr_src_type': channel_x_atr_src_type,
            'channel_x_atr_enable_kalman': channel_x_atr_enable_kalman,
            'channel_x_atr_kalman_type': channel_x_atr_kalman_type,
            
            # HyperERパラメータ（チャネル用）
            'channel_hyper_er_period': channel_hyper_er_period,
            'channel_hyper_er_midline_period': channel_hyper_er_midline_period,
            'channel_hyper_er_er_period': channel_hyper_er_er_period,
            'channel_hyper_er_er_src_type': channel_hyper_er_er_src_type,
            'channel_hyper_er_use_kalman_filter': channel_hyper_er_use_kalman_filter,
            'channel_hyper_er_kalman_filter_type': channel_hyper_er_kalman_filter_type,
            'channel_hyper_er_kalman_process_noise': channel_hyper_er_kalman_process_noise,
            'channel_hyper_er_kalman_min_observation_noise': channel_hyper_er_kalman_min_observation_noise,
            'channel_hyper_er_kalman_adaptation_window': channel_hyper_er_kalman_adaptation_window,
            'channel_hyper_er_use_roofing_filter': channel_hyper_er_use_roofing_filter,
            'channel_hyper_er_roofing_hp_cutoff': channel_hyper_er_roofing_hp_cutoff,
            'channel_hyper_er_roofing_ss_band_edge': channel_hyper_er_roofing_ss_band_edge,
            'channel_hyper_er_use_smoothing': channel_hyper_er_use_smoothing,
            'channel_hyper_er_smoother_type': channel_hyper_er_smoother_type,
            'channel_hyper_er_smoother_period': channel_hyper_er_smoother_period,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = HyperTrendFollowSignalGenerator(**self._parameters)
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル（ロング=1、ショート=-1、なし=0）
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
    
    def get_trend_follow_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """HyperTrendFollowシグナル取得"""
        try:
            return self.signal_generator.get_trend_follow_signals(data)
        except Exception as e:
            self.logger.error(f"HyperTrendFollowシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成
        
        Args:
            trial: Optunaのトライアル
            
        Returns:
            Dict[str, Any]: 最適化パラメータ（指定された10個のパラメータのみ）
        """
        params = {
            # ユーザー指定の最適化対象パラメータのみ
            'channel_src_type': trial.suggest_categorical('channel_src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
            'channel_multiplier_mode': trial.suggest_categorical('channel_multiplier_mode', ['fixed', 'dynamic']),
            'channel_fixed_multiplier': trial.suggest_float('channel_fixed_multiplier', 1.0, 4.0, step=0.1),
            'channel_hyper_frama_alpha_multiplier': trial.suggest_float('channel_hyper_frama_alpha_multiplier', 0.05, 0.8, step=0.01),
            'channel_x_atr_tr_method': trial.suggest_categorical('channel_x_atr_tr_method', ['atr', 'str']),
            'channel_x_atr_smoother_type': trial.suggest_categorical('channel_x_atr_smoother_type', 
                ['frama', 'ultimate_smoother', 'super_smoother', 'alma', 'laguerre']),
            'channel_hyper_er_er_src_type': trial.suggest_categorical('channel_hyper_er_er_src_type', 
                ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
            'channel_hyper_er_use_kalman_filter': trial.suggest_categorical('channel_hyper_er_use_kalman_filter', [True, False]),
            'channel_hyper_er_use_roofing_filter': trial.suggest_categorical('channel_hyper_er_use_roofing_filter', [True, False]),
            'channel_hyper_er_smoother_type': trial.suggest_categorical('channel_hyper_er_smoother_type', 
                ['frama', 'ultimate_smoother', 'super_smoother', 'alma', 'laguerre']),
        }
        
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換
        
        Args:
            params: 最適化パラメータ（指定された10個のパラメータのみ）
            
        Returns:
            Dict[str, Any]: 戦略パラメータ（最適化対象パラメータのみ更新）
        """
        strategy_params = {
            # 最適化対象パラメータのみ
            'channel_src_type': params.get('channel_src_type', 'oc2'),
            'channel_multiplier_mode': params.get('channel_multiplier_mode', 'dynamic'),
            'channel_fixed_multiplier': float(params.get('channel_fixed_multiplier', 1.5)),
            'channel_hyper_frama_alpha_multiplier': float(params.get('channel_hyper_frama_alpha_multiplier', 0.5)),
            'channel_x_atr_tr_method': params.get('channel_x_atr_tr_method', 'atr'),
            'channel_x_atr_smoother_type': params.get('channel_x_atr_smoother_type', 'frama'),
            'channel_hyper_er_er_src_type': params.get('channel_hyper_er_er_src_type', 'oc2'),
            'channel_hyper_er_use_kalman_filter': bool(params.get('channel_hyper_er_use_kalman_filter', True)),
            'channel_hyper_er_use_roofing_filter': bool(params.get('channel_hyper_er_use_roofing_filter', True)),
            'channel_hyper_er_smoother_type': params.get('channel_hyper_er_smoother_type', 'ultimate_smoother'),
        }
        
        return strategy_params
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """ストラテジー情報を取得"""
        return {
            'name': 'HyperTrendFollow Strategy',
            'description': 'HyperFRAMA位置関係とHyperFRAMAChannelブレイクアウトを組み合わせたトレンドフォロー戦略',
            'parameters': self._parameters.copy(),
            'features': [
                'HyperFRAMA位置関係シグナル',
                'HyperFRAMAChannelブレイクアウト',
                '組み合わせエントリー/エグジットロジック',
                'Numba最適化による高速処理',
                'Optuna最適化サポート'
            ],
            'entry_conditions': {
                'long': '位置関係シグナル=1 かつ ブレイクアウトシグナル=1',
                'short': '位置関係シグナル=-1 かつ ブレイクアウトシグナル=-1'
            },
            'exit_conditions': {
                'long': 'ブレイクアウトシグナル=-1',
                'short': 'ブレイクアウトシグナル=1'
            }
        }
    
    def reset(self) -> None:
        """ストラテジーの状態をリセット"""
        super().reset()
        if hasattr(self.signal_generator, 'reset'):
            self.signal_generator.reset()