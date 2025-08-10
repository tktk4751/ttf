#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import HyperDonchianFRAMASignalGenerator, FilterType


class HyperDonchianFRAMAStrategy(BaseStrategy):
    """HyperドンチャンFRAMAストラテジー
    
    Hyperドンチャンチャネル（80-20パーセンタイル版）とFRAMAの位置関係でエントリーし、
    複数のトレンドフィルターでシグナルを調整する戦略。
    従来のドンチャンより外れ値に堅牢で安定性が向上。
    """
    
    def __init__(
        self,
        # HyperドンチャンFRAMAパラメータ
        hyper_donchian_period: int = 200,
        hyper_donchian_src_type: str = 'hlc3',
        frama_period: int = 8,
        frama_src_type: str = 'oc2',
        frama_fc: int = 2,
        frama_sc: int = 198,
        frama_period_mode: str = 'fixed',
        signal_mode: str = 'position',  # 'position' または 'crossover'
        
        # HyperER動的適応パラメータ
        enable_hyper_er_adaptation: bool = True,  
        
        # HyperER動的適応を有効にするか
        hyper_er_period: int = 14,                 # HyperER計算期間
        hyper_er_midline_period: int = 100,        # HyperERミッドライン期間
        
        # FRAMA HyperER動的適応パラメータ
        frama_fc_min: float = 1.0,                 # FRAMA FC最小値（ER高い時）
        frama_fc_max: float = 13.0,                # FRAMA FC最大値（ER低い時）
        frama_sc_min: float = 60.0,                # FRAMA SC最小値（ER高い時）
        frama_sc_max: float = 250.0,               # FRAMA SC最大値（ER低い時）
        
        # HyperドンチャンHyperER動的適応パラメータ
        hyper_donchian_period_min: float = 55.0,         # Hyperドンチャン最小期間（ER高い時）
        hyper_donchian_period_max: float = 250.0,        # Hyperドンチャン最大期間（ER低い時）
        
        # フィルタータイプ
        filter_type: str = 'consensus',  # 'none', 'hyper_er', 'hyper_trend_index', 'hyper_adx', 'consensus'
        
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
        super().__init__(f"HyperDonchianFRAMA_{signal_mode}_{filter_type}")
        
        # パラメータ設定
        self._parameters = {
            'hyper_donchian_period': hyper_donchian_period,
            'hyper_donchian_src_type': hyper_donchian_src_type,
            'frama_period': frama_period,
            'frama_src_type': frama_src_type,
            'frama_fc': frama_fc,
            'frama_sc': frama_sc,
            'frama_period_mode': frama_period_mode,
            'signal_mode': signal_mode,
            'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
            'hyper_er_period': hyper_er_period,
            'hyper_er_midline_period': hyper_er_midline_period,
            'frama_fc_min': frama_fc_min,
            'frama_fc_max': frama_fc_max,
            'frama_sc_min': frama_sc_min,
            'frama_sc_max': frama_sc_max,
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
                'frama_period': frama_period,
                'frama_src_type': frama_src_type,
                'frama_fc': frama_fc,
                'frama_sc': frama_sc,
                'frama_period_mode': frama_period_mode,
                'signal_mode': signal_mode,
                # HyperER動的適応パラメータを追加
                'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
                'hyper_er_period': hyper_er_period,
                'hyper_er_midline_period': hyper_er_midline_period,
                'frama_fc_min': frama_fc_min,
                'frama_fc_max': frama_fc_max,
                'frama_sc_min': frama_sc_min,
                'frama_sc_max': frama_sc_max,
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
        self.signal_generator = HyperDonchianFRAMASignalGenerator(signal_params)
    
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
            'name': 'HyperDonchian FRAMA Strategy',
            'description': 'HyperDonchian (80-20 percentile) and FRAMA position-based trading strategy with trend filtering',
            'parameters': self._parameters.copy(),
            'filter_type': self.filter_type.value,
            'features': [
                'HyperDonchian channel (80-20 percentile, outlier-robust)',
                'FRAMA (Fractal Adaptive Moving Average)',
                'Position relationship signals (FRAMA vs HyperDonchian)',
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
            'frama_period': trial.suggest_int('frama_period', 8, 32, step=2),  # 偶数のみ
            'frama_fc': trial.suggest_int('frama_fc', 1, 5),
            'frama_sc': trial.suggest_int('frama_sc', 50, 300),
            'signal_mode': trial.suggest_categorical('signal_mode', ['position', 'crossover']),
            
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
                # FRAMA HyperER動的適応パラメータ
                'frama_fc_min': trial.suggest_float('frama_fc_min', 1.0, 3.0),
                'frama_fc_max': trial.suggest_float('frama_fc_max', 10.0, 20.0),
                'frama_sc_min': trial.suggest_float('frama_sc_min', 30.0, 100.0),
                'frama_sc_max': trial.suggest_float('frama_sc_max', 200.0, 350.0),
                
                # HyperドンチャンHyperER動的適応パラメータ
                'hyper_donchian_period_min': trial.suggest_float('hyper_donchian_period_min', 20.0, 100.0),
                'hyper_donchian_period_max': trial.suggest_float('hyper_donchian_period_max', 150.0, 350.0)
            })
        else:
            # HyperER動的適応が無効な場合のデフォルト値
            params.update({
                'frama_fc_min': 1.0,
                'frama_fc_max': 13.0,
                'frama_sc_min': 60.0,
                'frama_sc_max': 250.0,
                'hyper_donchian_period_min': 55.0,
                'hyper_donchian_period_max': 250.0
            })
        
        # 全フィルタータイプの詳細パラメータを常に最適化対象に含める（Optuna制約対応）
        
        # HyperERフィルターの詳細パラメータ
        params.update({
            'filter_hyper_er_src_type': trial.suggest_categorical('filter_hyper_er_src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
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
            # 基本パラメータ
            'hyper_donchian_period': int(params.get('hyper_donchian_period', 20)),
            'frama_period': int(params.get('frama_period', 16)),
            'frama_fc': int(params.get('frama_fc', 2)),
            'frama_sc': int(params.get('frama_sc', 198)),
            'signal_mode': params.get('signal_mode', 'position'),
            
            # HyperER動的適応パラメータ
            'enable_hyper_er_adaptation': bool(params.get('enable_hyper_er_adaptation', False)),
            'hyper_er_period': int(params.get('hyper_er_period', 14)),
            'hyper_er_midline_period': int(params.get('hyper_er_midline_period', 100)),
            
            # フィルター設定
            'filter_type': params.get('filter_type', 'consensus'),
            
            # HyperTrendIndexパラメータ
            'hyper_trend_index_period': int(params.get('hyper_trend_index_period', 14)),
            'hyper_trend_index_midline_period': int(params.get('hyper_trend_index_midline_period', 100)),
            
            # HyperADXパラメータ
            'hyper_adx_period': int(params.get('hyper_adx_period', 14)),
            'hyper_adx_midline_period': int(params.get('hyper_adx_midline_period', 100))
        }
        
        # HyperER動的適応が有効な場合の追加パラメータ
        if params.get('enable_hyper_er_adaptation', False):
            strategy_params.update({
                # FRAMA HyperER動的適応パラメータ
                'frama_fc_min': float(params.get('frama_fc_min', 1.0)),
                'frama_fc_max': float(params.get('frama_fc_max', 13.0)),
                'frama_sc_min': float(params.get('frama_sc_min', 60.0)),
                'frama_sc_max': float(params.get('frama_sc_max', 250.0)),
                
                # HyperドンチャンHyperER動的適応パラメータ
                'hyper_donchian_period_min': float(params.get('hyper_donchian_period_min', 55.0)),
                'hyper_donchian_period_max': float(params.get('hyper_donchian_period_max', 250.0))
            })
        else:
            # HyperER動的適応が無効な場合のデフォルト値
            strategy_params.update({
                'frama_fc_min': 1.0,
                'frama_fc_max': 13.0,
                'frama_sc_min': 60.0,
                'frama_sc_max': 250.0,
                'hyper_donchian_period_min': 55.0,
                'hyper_donchian_period_max': 250.0
            })
        
        # 全フィルター詳細パラメータを常に変換（デフォルト値使用）
        
        # HyperERフィルター詳細パラメータ
        strategy_params.update({
            'filter_hyper_er_src_type': params.get('filter_hyper_er_src_type', 'close'),
            'filter_hyper_er_use_roofing_filter': bool(params.get('filter_hyper_er_use_roofing_filter', False)),
            'filter_hyper_er_roofing_hp_cutoff': float(params.get('filter_hyper_er_roofing_hp_cutoff', 48.0)),
            'filter_hyper_er_roofing_ss_band_edge': float(params.get('filter_hyper_er_roofing_ss_band_edge', 10.0)),
            'filter_hyper_er_use_laguerre_filter': bool(params.get('filter_hyper_er_use_laguerre_filter', False)),
            'filter_hyper_er_laguerre_gamma': float(params.get('filter_hyper_er_laguerre_gamma', 0.5)),
            'filter_hyper_er_smoother_type': params.get('filter_hyper_er_smoother_type', 'super_smoother'),
            'filter_hyper_er_smoother_period': int(params.get('filter_hyper_er_smoother_period', 10)),
            'filter_hyper_er_detector_type': params.get('filter_hyper_er_detector_type', 'hody'),
            'filter_hyper_er_lp_period': int(params.get('filter_hyper_er_lp_period', 10)),
            'filter_hyper_er_hp_period': int(params.get('filter_hyper_er_hp_period', 60)),
            'filter_hyper_er_cycle_part': float(params.get('filter_hyper_er_cycle_part', 0.5)),
            'filter_hyper_er_max_cycle': int(params.get('filter_hyper_er_max_cycle', 60)),
            'filter_hyper_er_min_cycle': int(params.get('filter_hyper_er_min_cycle', 10)),
            'filter_hyper_er_max_output': int(params.get('filter_hyper_er_max_output', 50)),
            'filter_hyper_er_min_output': int(params.get('filter_hyper_er_min_output', 5))
        })
        
        # HyperTrendIndexフィルター詳細パラメータ
        strategy_params.update({
            'filter_hyper_trend_index_src_type': params.get('filter_hyper_trend_index_src_type', 'close'),
            'filter_hyper_trend_index_use_kalman_filter': bool(params.get('filter_hyper_trend_index_use_kalman_filter', False)),
            'filter_hyper_trend_index_kalman_filter_type': params.get('filter_hyper_trend_index_kalman_filter_type', 'simple'),
            'filter_hyper_trend_index_use_dynamic_period': bool(params.get('filter_hyper_trend_index_use_dynamic_period', False)),
            'filter_hyper_trend_index_detector_type': params.get('filter_hyper_trend_index_detector_type', 'dft_dominant'),
            'filter_hyper_trend_index_use_roofing_filter': bool(params.get('filter_hyper_trend_index_use_roofing_filter', False)),
            'filter_hyper_trend_index_roofing_hp_cutoff': float(params.get('filter_hyper_trend_index_roofing_hp_cutoff', 48.0)),
            'filter_hyper_trend_index_roofing_ss_band_edge': float(params.get('filter_hyper_trend_index_roofing_ss_band_edge', 10.0))
        })
        
        # HyperADXフィルター詳細パラメータ
        strategy_params.update({
            'filter_hyper_adx_use_kalman_filter': bool(params.get('filter_hyper_adx_use_kalman_filter', False)),
            'filter_hyper_adx_kalman_filter_type': params.get('filter_hyper_adx_kalman_filter_type', 'simple'),
            'filter_hyper_adx_use_dynamic_period': bool(params.get('filter_hyper_adx_use_dynamic_period', False)),
            'filter_hyper_adx_detector_type': params.get('filter_hyper_adx_detector_type', 'dft_dominant'),
            'filter_hyper_adx_use_roofing_filter': bool(params.get('filter_hyper_adx_use_roofing_filter', False)),
            'filter_hyper_adx_roofing_hp_cutoff': float(params.get('filter_hyper_adx_roofing_hp_cutoff', 48.0)),
            'filter_hyper_adx_roofing_ss_band_edge': float(params.get('filter_hyper_adx_roofing_ss_band_edge', 10.0))
        })
        
        return strategy_params
    
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



if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== HyperドンチャンFRAMAストラテジーのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 400
    base_price = 100.0
    
    prices = [base_price]
    for i in range(1, length):
        if i < 100:
            change = 0.003 + np.random.normal(0, 0.008)
        elif i < 200:
            change = np.random.normal(0, 0.012)
        elif i < 300:
            change = 0.005 + np.random.normal(0, 0.006)
        else:
            change = -0.003 + np.random.normal(0, 0.008)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.01))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.005)
            open_price = prices[i-1] + gap
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # ストラテジーのテスト
    strategy = HyperDonchianFRAMAStrategy(
        hyper_donchian_period=20,
        frama_period=16,
        filter_type='none'
    )
    
    # シグナル生成
    signals = strategy.generate_signals(df)
    
    entry_signals = signals['entry']
    long_entries = np.sum(entry_signals == 1)
    short_entries = np.sum(entry_signals == -1)
    neutral = np.sum(entry_signals == 0)
    
    print(f"\nシグナル統計:")
    print(f"  ロングエントリー: {long_entries} ({long_entries/len(df)*100:.1f}%)")
    print(f"  ショートエントリー: {short_entries} ({short_entries/len(df)*100:.1f}%)")
    print(f"  中立: {neutral} ({neutral/len(df)*100:.1f}%)")
    
    # 戦略情報
    strategy_info = strategy.get_strategy_info()
    print(f"\n戦略名: {strategy_info['name']}")
    print(f"説明: {strategy_info['description']}")
    print(f"フィルタータイプ: {strategy_info['filter_type']}")
    
    # コンセンサスフィルターのテスト
    print("\nコンセンサスフィルターのテスト...")
    consensus_strategy = HyperDonchianFRAMAStrategy(
        hyper_donchian_period=20,
        frama_period=16,
        filter_type='consensus'
    )
    
    consensus_signals = consensus_strategy.generate_signals(df)
    consensus_entry = consensus_signals['entry']
    consensus_long = np.sum(consensus_entry == 1)
    consensus_short = np.sum(consensus_entry == -1)
    consensus_neutral = np.sum(consensus_entry == 0)
    
    print(f"  コンセンサスロング: {consensus_long} ({consensus_long/len(df)*100:.1f}%)")
    print(f"  コンセンサスショート: {consensus_short} ({consensus_short/len(df)*100:.1f}%)")
    print(f"  コンセンサス中立: {consensus_neutral} ({consensus_neutral/len(df)*100:.1f}%)")
    
    # 従来版との比較
    print("\n従来のドンチャンFRAMAとの比較...")
    try:
        from ..donchian_frama.strategy import DonchianFRAMAStrategy
        
        traditional_strategy = DonchianFRAMAStrategy(
            donchian_period=20,
            frama_period=16,
            filter_type='none'
        )
        
        traditional_signals = traditional_strategy.generate_signals(df)
        trad_entry = traditional_signals['entry']
        trad_long = np.sum(trad_entry == 1)
        trad_short = np.sum(trad_entry == -1)
        trad_neutral = np.sum(trad_entry == 0)
        
        print(f"  従来版ロング: {trad_long} ({trad_long/len(df)*100:.1f}%)")
        print(f"  従来版ショート: {trad_short} ({trad_short/len(df)*100:.1f}%)")
        print(f"  従来版中立: {trad_neutral} ({trad_neutral/len(df)*100:.1f}%)")
        
        # シグナル安定性比較
        hyper_changes = np.sum(np.diff(entry_signals) != 0)
        trad_changes = np.sum(np.diff(trad_entry) != 0)
        
        print(f"\nシグナル安定性比較:")
        print(f"  Hyperドンチャン版変化回数: {hyper_changes}")
        print(f"  従来版変化回数: {trad_changes}")
        print(f"  安定性改善: {((trad_changes - hyper_changes) / trad_changes * 100):.1f}%")
        
    except ImportError:
        print("  従来のドンチャンFRAMAストラテジーが見つかりませんでした")
    
    print("\n=== テスト完了 ===")