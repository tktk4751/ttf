#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import HyperDonchianSignalGenerator, FilterType


class HyperDonchianStrategy(BaseStrategy):
    """Hyperドンチャンブレイクアウトストラテジー
    
    Hyperドンチャンチャネル（Min/Max範囲内80-20%位置版）のブレイクアウトでエントリーし、
    複数のトレンドフィルターでシグナルを調整する戦略。
    従来のドンチャンより外れ値に堅牢で安定性が向上。
    """
    
    def __init__(
        self,
        # Hyperドンチャンブレイクアウトパラメータ
        hyper_donchian_period: int = 20,
        hyper_donchian_src_type: str = 'close',
        
        # HyperER動的適応パラメータ
        enable_hyper_er_adaptation: bool = True,  # HyperER動的適応を有効にするか
        hyper_er_period: int = 14,                 # HyperER計算期間
        hyper_er_midline_period: int = 100,        # HyperERミッドライン期間
        
        # Hyperドンチャン HyperER動的適応パラメータ
        hyper_donchian_period_min: float = 40,         # Hyperドンチャン最小期間（ER高い時）
        hyper_donchian_period_max: float = 240,        # Hyperドンチャン最大期間（ER低い時）
        
        # フィルタータイプ
        filter_type: str = 'none',  # 'none', 'hyper_er', 'hyper_trend_index', 'hyper_adx', 'consensus'
        
        # HyperTrendIndexパラメータ  
        hyper_trend_index_period: int = 14,
        hyper_trend_index_midline_period: int = 100,
        
        # HyperADXパラメータ
        hyper_adx_period: int = 14,
        hyper_adx_midline_period: int = 100,
        
        # HyperERフィルター詳細パラメータ
        filter_hyper_er_src_type: str = 'oc2',
        filter_hyper_er_use_roofing_filter: bool = True,
        filter_hyper_er_roofing_hp_cutoff: float = 55.0,
        filter_hyper_er_roofing_ss_band_edge: float = 10.0,
        filter_hyper_er_use_laguerre_filter: bool = False,
        filter_hyper_er_laguerre_gamma: float = 0.5,
        filter_hyper_er_smoother_type: str = 'frama',
        filter_hyper_er_smoother_period: int = 16,
        filter_hyper_er_detector_type: str = 'dft_dominant',
        filter_hyper_er_lp_period: int = 13,
        filter_hyper_er_hp_period: int = 124,
        filter_hyper_er_cycle_part: float = 0.4,
        filter_hyper_er_max_cycle: int = 124,
        filter_hyper_er_min_cycle: int = 13,
        filter_hyper_er_max_output: int = 89,
        filter_hyper_er_min_output: int = 5,
        
        # HyperTrendIndexフィルター詳細パラメータ
        filter_hyper_trend_index_src_type: str = 'oc2',
        filter_hyper_trend_index_use_kalman_filter: bool = False,
        filter_hyper_trend_index_kalman_filter_type: str = 'unscented',
        filter_hyper_trend_index_use_dynamic_period: bool = True,
        filter_hyper_trend_index_detector_type: str = 'dft_dominant',
        filter_hyper_trend_index_use_roofing_filter: bool = True,
        filter_hyper_trend_index_roofing_hp_cutoff: float = 55.0,
        filter_hyper_trend_index_roofing_ss_band_edge: float = 10.0,
        
        # HyperADXフィルター詳細パラメータ
        filter_hyper_adx_use_kalman_filter: bool = False,
        filter_hyper_adx_kalman_filter_type: str = 'unsectend',
        filter_hyper_adx_use_dynamic_period: bool = True,
        filter_hyper_adx_detector_type: str = 'dft_dominant',
        filter_hyper_adx_use_roofing_filter: bool = True,
        filter_hyper_adx_roofing_hp_cutoff: float = 55.0,
        filter_hyper_adx_roofing_ss_band_edge: float = 10.0,

        # 将来拡張用のkwargs受け入れ
        **kwargs
    ):
        super().__init__(f"HyperDonchianBreakout_{filter_type}")
        
        # パラメータ設定
        self._parameters = {
            'hyper_donchian_period': hyper_donchian_period,
            'hyper_donchian_src_type': hyper_donchian_src_type,
            'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
            'hyper_er_period': hyper_er_period,
            'hyper_er_midline_period': hyper_er_midline_period,
            'hyper_donchian_period_min': hyper_donchian_period_min,
            'hyper_donchian_period_max': hyper_donchian_period_max,
            'filter_type': filter_type,
            'hyper_trend_index_period': hyper_trend_index_period,
            'hyper_trend_index_midline_period': hyper_trend_index_midline_period,
            'hyper_adx_period': hyper_adx_period,
            'hyper_adx_midline_period': hyper_adx_midline_period,
            
            # HyperERフィルター詳細パラメータ
            'filter_hyper_er_src_type': filter_hyper_er_src_type,
            'filter_hyper_er_use_roofing_filter': filter_hyper_er_use_roofing_filter,
            'filter_hyper_er_roofing_hp_cutoff': filter_hyper_er_roofing_hp_cutoff,
            'filter_hyper_er_roofing_ss_band_edge': filter_hyper_er_roofing_ss_band_edge,
            'filter_hyper_er_use_laguerre_filter': filter_hyper_er_use_laguerre_filter,
            'filter_hyper_er_laguerre_gamma': filter_hyper_er_laguerre_gamma,
            'filter_hyper_er_smoother_type': filter_hyper_er_smoother_type,
            'filter_hyper_er_smoother_period': filter_hyper_er_smoother_period,
            'filter_hyper_er_detector_type': filter_hyper_er_detector_type,
            'filter_hyper_er_lp_period': filter_hyper_er_lp_period,
            'filter_hyper_er_hp_period': filter_hyper_er_hp_period,
            'filter_hyper_er_cycle_part': filter_hyper_er_cycle_part,
            'filter_hyper_er_max_cycle': filter_hyper_er_max_cycle,
            'filter_hyper_er_min_cycle': filter_hyper_er_min_cycle,
            'filter_hyper_er_max_output': filter_hyper_er_max_output,
            'filter_hyper_er_min_output': filter_hyper_er_min_output,
            
            # HyperTrendIndexフィルター詳細パラメータ
            'filter_hyper_trend_index_src_type': filter_hyper_trend_index_src_type,
            'filter_hyper_trend_index_use_kalman_filter': filter_hyper_trend_index_use_kalman_filter,
            'filter_hyper_trend_index_kalman_filter_type': filter_hyper_trend_index_kalman_filter_type,
            'filter_hyper_trend_index_use_dynamic_period': filter_hyper_trend_index_use_dynamic_period,
            'filter_hyper_trend_index_detector_type': filter_hyper_trend_index_detector_type,
            'filter_hyper_trend_index_use_roofing_filter': filter_hyper_trend_index_use_roofing_filter,
            'filter_hyper_trend_index_roofing_hp_cutoff': filter_hyper_trend_index_roofing_hp_cutoff,
            'filter_hyper_trend_index_roofing_ss_band_edge': filter_hyper_trend_index_roofing_ss_band_edge,
            
            # HyperADXフィルター詳細パラメータ
            'filter_hyper_adx_use_kalman_filter': filter_hyper_adx_use_kalman_filter,
            'filter_hyper_adx_kalman_filter_type': filter_hyper_adx_kalman_filter_type,
            'filter_hyper_adx_use_dynamic_period': filter_hyper_adx_use_dynamic_period,
            'filter_hyper_adx_detector_type': filter_hyper_adx_detector_type,
            'filter_hyper_adx_use_roofing_filter': filter_hyper_adx_use_roofing_filter,
            'filter_hyper_adx_roofing_hp_cutoff': filter_hyper_adx_roofing_hp_cutoff,
            'filter_hyper_adx_roofing_ss_band_edge': filter_hyper_adx_roofing_ss_band_edge
        }
        
        self.filter_type = FilterType(filter_type)
        
        # シグナルジェネレーター用のパラメータ構築
        signal_params = {
            'filter_type': filter_type,
            'entry': {
                'hyper_donchian_period': hyper_donchian_period,
                'hyper_donchian_src_type': hyper_donchian_src_type,
                # HyperER動的適応パラメータを追加
                'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
                'hyper_er_period': hyper_er_period,
                'hyper_er_midline_period': hyper_er_midline_period,
                'hyper_donchian_period_min': hyper_donchian_period_min,
                'hyper_donchian_period_max': hyper_donchian_period_max
            },
            'hyper_er': {
                'period': hyper_er_period,
                'midline_period': hyper_er_midline_period,
                # HyperERフィルター詳細パラメータ
                'src_type': filter_hyper_er_src_type,
                'use_roofing_filter': filter_hyper_er_use_roofing_filter,
                'roofing_hp_cutoff': filter_hyper_er_roofing_hp_cutoff,
                'roofing_ss_band_edge': filter_hyper_er_roofing_ss_band_edge,
                'use_laguerre_filter': filter_hyper_er_use_laguerre_filter,
                'laguerre_gamma': filter_hyper_er_laguerre_gamma,
                'smoother_type': filter_hyper_er_smoother_type,
                'smoother_period': filter_hyper_er_smoother_period,
                'detector_type': filter_hyper_er_detector_type,
                'lp_period': filter_hyper_er_lp_period,
                'hp_period': filter_hyper_er_hp_period,
                'cycle_part': filter_hyper_er_cycle_part,
                'max_cycle': filter_hyper_er_max_cycle,
                'min_cycle': filter_hyper_er_min_cycle,
                'max_output': filter_hyper_er_max_output,
                'min_output': filter_hyper_er_min_output
            },
            'hyper_trend_index': {
                'period': hyper_trend_index_period,
                'midline_period': hyper_trend_index_midline_period,
                # HyperTrendIndexフィルター詳細パラメータ
                'src_type': filter_hyper_trend_index_src_type,
                'use_kalman_filter': filter_hyper_trend_index_use_kalman_filter,
                'kalman_filter_type': filter_hyper_trend_index_kalman_filter_type,
                'use_dynamic_period': filter_hyper_trend_index_use_dynamic_period,
                'detector_type': filter_hyper_trend_index_detector_type,
                'use_roofing_filter': filter_hyper_trend_index_use_roofing_filter,
                'roofing_hp_cutoff': filter_hyper_trend_index_roofing_hp_cutoff,
                'roofing_ss_band_edge': filter_hyper_trend_index_roofing_ss_band_edge
            },
            'hyper_adx': {
                'period': hyper_adx_period,
                'midline_period': hyper_adx_midline_period,
                # HyperADXフィルター詳細パラメータ
                'use_kalman_filter': filter_hyper_adx_use_kalman_filter,
                'kalman_filter_type': filter_hyper_adx_kalman_filter_type,
                'use_dynamic_period': filter_hyper_adx_use_dynamic_period,
                'detector_type': filter_hyper_adx_detector_type,
                'use_roofing_filter': filter_hyper_adx_use_roofing_filter,
                'roofing_hp_cutoff': filter_hyper_adx_roofing_hp_cutoff,
                'roofing_ss_band_edge': filter_hyper_adx_roofing_ss_band_edge
            }
        }
        
        # シグナルジェネレーター
        self.signal_generator = HyperDonchianSignalGenerator(signal_params)
    
    def generate_entry(self, data: pd.DataFrame) -> np.ndarray:
        """エントリーシグナルを生成"""
        try:
            return self.signal_generator.generate_entry_signals(data)
        except Exception as e:
            self.logger.error(f"エントリーシグナル生成中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def generate_exit(self, data: pd.DataFrame, position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成"""
        try:
            return self.signal_generator.generate_exit_signals(data, position, index)
        except Exception as e:
            self.logger.error(f"エグジットシグナル生成中にエラー: {str(e)}")
            return False
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """シグナル生成"""
        entry_signals = self.signal_generator.generate_entry_signals(data)
        exit_signals = self.signal_generator.generate_exit_signals(data)
        
        return {
            'entry': entry_signals,
            'exit': exit_signals,
            'direction': entry_signals,  # エントリーシグナルと同じ
            'filter': self.signal_generator._get_filter_signals(data) if self.filter_type != FilterType.NONE else np.ones(len(data))
        }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """戦略情報取得"""
        return {
            'name': 'HyperDonchian Breakout Strategy',
            'description': 'HyperDonchian (Min/Max範囲内80-20%位置版) breakout trading strategy with trend filtering',
            'parameters': self._parameters.copy(),
            'filter_type': self.filter_type.value,
            'features': [
                'HyperDonchian channel breakout (Min/Max範囲内80-20%位置版)',
                'Outlier-robust breakout detection',
                'Multiple trend filter options (HyperER, HyperTrendIndex, HyperADX)',
                'Consensus filtering (2 out of 3 agreement)',
                'HyperER dynamic adaptation support',
                'Enhanced stability compared to traditional Donchian',
                'Pure signal-based trading without stop loss/take profit'
            ]
        }
    
    def reset_state(self):
        """状態リセット"""
        if hasattr(self.signal_generator, 'reset'):
            self.signal_generator.reset()
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成
        
        Args:
            trial: Optunaのトライアル
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        # フィルタータイプの選択
        filter_type = trial.suggest_categorical('filter_type', [
            'none',
            'hyper_er',
            'hyper_trend_index',
            'hyper_adx',
            'consensus'
        ])
        
        # HyperER動的適応の有効無効
        enable_hyper_er_adaptation = trial.suggest_categorical('enable_hyper_er_adaptation', [True, False])
        
        params = {
            # 基本パラメータ
            'hyper_donchian_period': trial.suggest_int('hyper_donchian_period', 10, 50),
            'hyper_donchian_src_type': trial.suggest_categorical('hyper_donchian_src_type', ['close', 'high', 'low', 'hl2', 'hlc3', 'ohlc4']),
            
            # HyperER動的適応パラメータ
            'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
            'hyper_er_period': trial.suggest_int('hyper_er_period', 10, 30),
            'hyper_er_midline_period': trial.suggest_int('hyper_er_midline_period', 50, 200),
            
            # フィルター設定
            'filter_type': filter_type,
            
            # HyperTrendIndexパラメータ
            'hyper_trend_index_period': trial.suggest_int('hyper_trend_index_period', 10, 30),
            'hyper_trend_index_midline_period': trial.suggest_int('hyper_trend_index_midline_period', 50, 200),
            
            # HyperADXパラメータ
            'hyper_adx_period': trial.suggest_int('hyper_adx_period', 10, 30),
            'hyper_adx_midline_period': trial.suggest_int('hyper_adx_midline_period', 50, 200)
        }
        
        # HyperER動的適応が有効な場合の追加パラメータ
        if enable_hyper_er_adaptation:
            params.update({
                # Hyperドンチャン HyperER動的適応パラメータ
                'hyper_donchian_period_min': trial.suggest_float('hyper_donchian_period_min', 10.0, 30.0),
                'hyper_donchian_period_max': trial.suggest_float('hyper_donchian_period_max', 40.0, 80.0)
            })
        else:
            # HyperER動的適応が無効な場合のデフォルト値
            params.update({
                'hyper_donchian_period_min': 15.0,
                'hyper_donchian_period_max': 60.0
            })
        
        # 全フィルタータイプの詳細パラメータを常に最適化対象に含める（Optuna制約対応）
        
        # HyperERフィルターの詳細パラメータ
        params.update({
            'filter_hyper_er_src_type': trial.suggest_categorical('filter_hyper_er_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'filter_hyper_er_use_roofing_filter': trial.suggest_categorical('filter_hyper_er_use_roofing_filter', [True, False]),
            'filter_hyper_er_roofing_hp_cutoff': trial.suggest_float('filter_hyper_er_roofing_hp_cutoff', 20.0, 100.0, step=4.0),
            'filter_hyper_er_roofing_ss_band_edge': trial.suggest_float('filter_hyper_er_roofing_ss_band_edge', 5.0, 19.0, step=1.0),
            'filter_hyper_er_use_laguerre_filter': trial.suggest_categorical('filter_hyper_er_use_laguerre_filter', [True, False]),
            'filter_hyper_er_laguerre_gamma': trial.suggest_float('filter_hyper_er_laguerre_gamma', 0.1, 0.9, step=0.1),
            'filter_hyper_er_smoother_type': trial.suggest_categorical('filter_hyper_er_smoother_type', ['super_smoother', 'laguerre', 'ultimate_smoother','alma','hma','zlema']),
            'filter_hyper_er_smoother_period': trial.suggest_int('filter_hyper_er_smoother_period', 5, 25),
            'filter_hyper_er_detector_type': trial.suggest_categorical('filter_hyper_er_detector_type', [
                'hody', 'phac', 'dudi', 'dudi_e', 'hody_e', 'phac_e',
                'cycle_period', 'cycle_period2', 'bandpass_zero', 'autocorr_perio', 
                'dft_dominant', 'multi_bandpass', 'absolute_ultimate'
            ]),
            'filter_hyper_er_lp_period': trial.suggest_int('filter_hyper_er_lp_period', 5, 20),
            'filter_hyper_er_hp_period': trial.suggest_int('filter_hyper_er_hp_period', 40, 150),
            'filter_hyper_er_cycle_part': trial.suggest_float('filter_hyper_er_cycle_part', 0.2, 1.0, step=0.1),
            'filter_hyper_er_max_cycle': trial.suggest_int('filter_hyper_er_max_cycle', 40, 150),
            'filter_hyper_er_min_cycle': trial.suggest_int('filter_hyper_er_min_cycle', 5, 20),
            'filter_hyper_er_max_output': trial.suggest_int('filter_hyper_er_max_output', 30, 100),
            'filter_hyper_er_min_output': trial.suggest_int('filter_hyper_er_min_output', 3, 8)
        })
        
        # HyperTrendIndexフィルターの詳細パラメータ
        params.update({
            'filter_hyper_trend_index_src_type': trial.suggest_categorical('filter_hyper_trend_index_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'filter_hyper_trend_index_use_kalman_filter': trial.suggest_categorical('filter_hyper_trend_index_use_kalman_filter', [True, False]),
            'filter_hyper_trend_index_kalman_filter_type': trial.suggest_categorical('filter_hyper_trend_index_kalman_filter_type', ['simple', 'unscented', 'quantum_adaptive']),
            'filter_hyper_trend_index_use_dynamic_period': trial.suggest_categorical('filter_hyper_trend_index_use_dynamic_period', [True, False]),
            'filter_hyper_trend_index_detector_type': trial.suggest_categorical('filter_hyper_trend_index_detector_type', [
                'dft_dominant', 'hody', 'phac', 'dudi', 'cycle_period', 'bandpass_zero'
            ]),
            'filter_hyper_trend_index_use_roofing_filter': trial.suggest_categorical('filter_hyper_trend_index_use_roofing_filter', [True, False]),
            'filter_hyper_trend_index_roofing_hp_cutoff': trial.suggest_float('filter_hyper_trend_index_roofing_hp_cutoff', 20.0, 100.0, step=5.0),
            'filter_hyper_trend_index_roofing_ss_band_edge': trial.suggest_float('filter_hyper_trend_index_roofing_ss_band_edge', 5.0, 20.0, step=1.0)
        })
        
        # HyperADXフィルターの詳細パラメータ
        params.update({
            'filter_hyper_adx_use_kalman_filter': trial.suggest_categorical('filter_hyper_adx_use_kalman_filter', [True, False]),
            'filter_hyper_adx_kalman_filter_type': trial.suggest_categorical('filter_hyper_adx_kalman_filter_type', ['simple', 'unscented', 'quantum_adaptive']),
            'filter_hyper_adx_use_dynamic_period': trial.suggest_categorical('filter_hyper_adx_use_dynamic_period', [True, False]),
            'filter_hyper_adx_detector_type': trial.suggest_categorical('filter_hyper_adx_detector_type', [
                'dft_dominant', 'hody', 'phac', 'dudi', 'cycle_period', 'bandpass_zero'
            ]),
            'filter_hyper_adx_use_roofing_filter': trial.suggest_categorical('filter_hyper_adx_use_roofing_filter', [True, False]),
            'filter_hyper_adx_roofing_hp_cutoff': trial.suggest_float('filter_hyper_adx_roofing_hp_cutoff', 20.0, 100.0, step=5.0),
            'filter_hyper_adx_roofing_ss_band_edge': trial.suggest_float('filter_hyper_adx_roofing_ss_band_edge', 5.0, 20.0, step=1.0)
        })
        
        return params
    
    def get_stop_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """
        ストップロス価格を取得する
        
        Args:
            data: 価格データ
            position: ポジション方向 (1: ロング, -1: ショート)
            index: データのインデックス【デフォルト: -1 = 最新のデータ】
            
        Returns:
            float: ストップロス価格
        """
        # シンプルなストップロスは実装しない（シグナルベースのため）
        if isinstance(data, pd.DataFrame):
            return data['close'].iloc[index]
        return data[index, 3]
    
    def get_exit_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """
        エグジット価格を取得する
        
        Args:
            data: 価格データ
            position: ポジション方向 (1: ロング, -1: ショート)
            index: データのインデックス【デフォルト: -1 = 最新のデータ】
            
        Returns:
            float: エグジット価格
        """
        if isinstance(data, pd.DataFrame):
            return data['close'].iloc[index]
        return data[index, 3]

