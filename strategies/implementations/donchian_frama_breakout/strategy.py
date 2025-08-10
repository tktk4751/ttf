#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import DonchianFRAMABreakoutSignalGenerator, FilterType


class DonchianFRAMABreakoutStrategy(BaseStrategy):
    """ドンチャンFRAMAブレイクアウトストラテジー
    
    3つのインジケーターを使用する高度なブレイクアウト戦略:
    1. ドンチャンミッドライン（期間200、フィルター用）
    2. FRAMA（フラクタル適応移動平均、トレンド方向判定用）
    3. トリガー用ドンチャンチャネル（期間60、ブレイクアウト検出とエグジット用）
    
    エントリーロジック:
    - 3つのフィルター（HyperER、HyperTrendIndex、HyperADX）通過後
    - ドンチャンFRAMA位置関係シグナル=1時、トリガードンチャン前回上部突破でロング
    - ドンチャンFRAMA位置関係シグナル=-1時、トリガードンチャン前回下部突破でショート
    
    エグジットロジック:
    - ロングポジション: トリガードンチャン前回下部を終値が下回ったら決済
    - ショートポジション: トリガードンチャン前回上部を終値が上回ったら決済
    """
    
    def __init__(
        self,
        # ドンチャンミッドラインパラメータ（フィルター用）
        donchian_midline_period: int = 200,
        donchian_midline_src_type: str = 'hlc3',
        
        # FRAMAパラメータ（トレンド方向判定用）
        frama_period: int = 16,
        frama_src_type: str = 'hlc3',
        frama_fc: int = 2,
        frama_sc: int = 198,
        frama_period_mode: str = 'fixed',
        
        # トリガー用ドンチャンチャネルパラメータ
        trigger_donchian_period: int = 60,
        trigger_donchian_src_type: str = 'hlc3',
        
        # HyperER動的適応パラメータ
        enable_hyper_er_adaptation: bool = True,
        hyper_er_period: int = 14,
        hyper_er_midline_period: int = 100,
        
        # FRAMA HyperER動的適応パラメータ
        frama_fc_min: float = 1.0,
        frama_fc_max: float = 13.0,
        frama_sc_min: float = 60.0,
        frama_sc_max: float = 250.0,
        
        # ドンチャンミッドライン HyperER動的適応パラメータ
        donchian_midline_period_min: float = 55.0,
        donchian_midline_period_max: float = 250.0,
        
        # トリガー用ドンチャン HyperER動的適応パラメータ
        trigger_donchian_period_min: float = 20.0,
        trigger_donchian_period_max: float = 100.0,
        
        # フィルタータイプ
        filter_type: str = 'consensus',  # 'none', 'hyper_er', 'hyper_trend_index', 'hyper_adx', 'consensus'
        
        # HyperTrendIndexパラメータ
        hyper_trend_index_period: int = 14,
        hyper_trend_index_midline_period: int = 100,
        
        # HyperADXパラメータ
        hyper_adx_period: int = 14,
        hyper_adx_midline_period: int = 100,
        
        # HyperERフィルター詳細パラメータ
        filter_hyper_er_src_type: str = 'close',
        filter_hyper_er_use_roofing_filter: bool = False,
        filter_hyper_er_roofing_hp_cutoff: float = 48.0,
        filter_hyper_er_roofing_ss_band_edge: float = 10.0,
        filter_hyper_er_use_laguerre_filter: bool = False,
        filter_hyper_er_laguerre_gamma: float = 0.5,
        filter_hyper_er_smoother_type: str = 'super_smoother',
        filter_hyper_er_smoother_period: int = 10,
        filter_hyper_er_detector_type: str = 'hody',
        filter_hyper_er_lp_period: int = 10,
        filter_hyper_er_hp_period: int = 60,
        filter_hyper_er_cycle_part: float = 0.5,
        filter_hyper_er_max_cycle: int = 60,
        filter_hyper_er_min_cycle: int = 10,
        filter_hyper_er_max_output: int = 50,
        filter_hyper_er_min_output: int = 5,
        
        # HyperTrendIndexフィルター詳細パラメータ
        filter_hyper_trend_index_src_type: str = 'close',
        filter_hyper_trend_index_use_kalman_filter: bool = False,
        filter_hyper_trend_index_kalman_filter_type: str = 'simple',
        filter_hyper_trend_index_use_dynamic_period: bool = False,
        filter_hyper_trend_index_detector_type: str = 'dft_dominant',
        filter_hyper_trend_index_use_roofing_filter: bool = False,
        filter_hyper_trend_index_roofing_hp_cutoff: float = 48.0,
        filter_hyper_trend_index_roofing_ss_band_edge: float = 10.0,
        
        # HyperADXフィルター詳細パラメータ
        filter_hyper_adx_use_kalman_filter: bool = False,
        filter_hyper_adx_kalman_filter_type: str = 'simple',
        filter_hyper_adx_use_dynamic_period: bool = False,
        filter_hyper_adx_detector_type: str = 'dft_dominant',
        filter_hyper_adx_use_roofing_filter: bool = False,
        filter_hyper_adx_roofing_hp_cutoff: float = 48.0,
        filter_hyper_adx_roofing_ss_band_edge: float = 10.0,
        
        # シグナル設定
        signal_mode: str = 'position',  # 'position' または 'crossover'
        
        # 将来拡張用のkwargs受け入れ
        **kwargs
    ):
        super().__init__(f"DonchianFRAMABreakout_{signal_mode}_{filter_type}")
        
        # パラメータ設定
        self._parameters = {
            # ドンチャンミッドラインパラメータ（フィルター用）
            'donchian_midline_period': donchian_midline_period,
            'donchian_midline_src_type': donchian_midline_src_type,
            
            # FRAMAパラメータ（トレンド方向判定用）
            'frama_period': frama_period,
            'frama_src_type': frama_src_type,
            'frama_fc': frama_fc,
            'frama_sc': frama_sc,
            'frama_period_mode': frama_period_mode,
            
            # トリガー用ドンチャンチャネルパラメータ
            'trigger_donchian_period': trigger_donchian_period,
            'trigger_donchian_src_type': trigger_donchian_src_type,
            
            # HyperER動的適応パラメータ
            'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
            'hyper_er_period': hyper_er_period,
            'hyper_er_midline_period': hyper_er_midline_period,
            
            # FRAMA HyperER動的適応パラメータ
            'frama_fc_min': frama_fc_min,
            'frama_fc_max': frama_fc_max,
            'frama_sc_min': frama_sc_min,
            'frama_sc_max': frama_sc_max,
            
            # ドンチャンミッドライン HyperER動的適応パラメータ
            'donchian_midline_period_min': donchian_midline_period_min,
            'donchian_midline_period_max': donchian_midline_period_max,
            
            # トリガー用ドンチャン HyperER動的適応パラメータ
            'trigger_donchian_period_min': trigger_donchian_period_min,
            'trigger_donchian_period_max': trigger_donchian_period_max,
            
            # フィルター設定
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
            'filter_hyper_adx_roofing_ss_band_edge': filter_hyper_adx_roofing_ss_band_edge,
            
            # シグナル設定
            'signal_mode': signal_mode
        }
        
        self.filter_type = FilterType(filter_type)
        
        # シグナルジェネレーター用のパラメータ構築
        signal_params = {
            'filter_type': filter_type,
            'entry': {
                # ドンチャンミッドラインパラメータ（フィルター用）
                'donchian_midline_period': donchian_midline_period,
                'donchian_midline_src_type': donchian_midline_src_type,
                
                # FRAMAパラメータ
                'frama_period': frama_period,
                'frama_src_type': frama_src_type,
                'frama_fc': frama_fc,
                'frama_sc': frama_sc,
                'frama_period_mode': frama_period_mode,
                
                # トリガー用ドンチャンチャネルパラメータ
                'trigger_donchian_period': trigger_donchian_period,
                'trigger_donchian_src_type': trigger_donchian_src_type,
                
                # HyperER動的適応パラメータ
                'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
                'hyper_er_period': hyper_er_period,
                'hyper_er_midline_period': hyper_er_midline_period,
                
                # FRAMA HyperER動的適応パラメータ
                'frama_fc_min': frama_fc_min,
                'frama_fc_max': frama_fc_max,
                'frama_sc_min': frama_sc_min,
                'frama_sc_max': frama_sc_max,
                
                # ドンチャンミッドライン HyperER動的適応パラメータ
                'donchian_midline_period_min': donchian_midline_period_min,
                'donchian_midline_period_max': donchian_midline_period_max,
                
                # トリガー用ドンチャン HyperER動的適応パラメータ
                'trigger_donchian_period_min': trigger_donchian_period_min,
                'trigger_donchian_period_max': trigger_donchian_period_max,
                
                'signal_mode': signal_mode
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
        self.signal_generator = DonchianFRAMABreakoutSignalGenerator(signal_params)
    
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
            'name': 'Donchian FRAMA Breakout Strategy',
            'description': 'Advanced breakout strategy using 3 Donchian indicators with trend filtering',
            'parameters': self._parameters.copy(),
            'filter_type': self.filter_type.value,
            'features': [
                'Donchian Midline (period 200) for trend filtering',
                'FRAMA for trend direction detection',
                'Trigger Donchian Channel (period 60) for breakout entry/exit',
                'Multiple advanced trend filter options (HyperER, HyperTrendIndex, HyperADX)',
                'Consensus filtering (2 out of 3 agreement)',
                'HyperER dynamic adaptation support',
                'Pure signal-based trading with channel-based exits'
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
            'donchian_midline_period': trial.suggest_int('donchian_midline_period', 100, 300),
            'frama_period': trial.suggest_int('frama_period', 8, 32, step=2),  # 偶数のみ
            'frama_fc': trial.suggest_int('frama_fc', 1, 5),
            'frama_sc': trial.suggest_int('frama_sc', 50, 300),
            'trigger_donchian_period': trial.suggest_int('trigger_donchian_period', 20, 100),
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
                
                # ドンチャンミッドライン HyperER動的適応パラメータ
                'donchian_midline_period_min': trial.suggest_float('donchian_midline_period_min', 30.0, 100.0),
                'donchian_midline_period_max': trial.suggest_float('donchian_midline_period_max', 150.0, 350.0),
                
                # トリガー用ドンチャン HyperER動的適応パラメータ
                'trigger_donchian_period_min': trial.suggest_float('trigger_donchian_period_min', 10.0, 50.0),
                'trigger_donchian_period_max': trial.suggest_float('trigger_donchian_period_max', 60.0, 150.0)
            })
        else:
            # HyperER動的適応が無効な場合のデフォルト値
            params.update({
                'frama_fc_min': 1.0,
                'frama_fc_max': 13.0,
                'frama_sc_min': 60.0,
                'frama_sc_max': 250.0,
                'donchian_midline_period_min': 55.0,
                'donchian_midline_period_max': 250.0,
                'trigger_donchian_period_min': 20.0,
                'trigger_donchian_period_max': 100.0
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
            'donchian_midline_period': int(params.get('donchian_midline_period', 200)),
            'frama_period': int(params.get('frama_period', 16)),
            'frama_fc': int(params.get('frama_fc', 2)),
            'frama_sc': int(params.get('frama_sc', 198)),
            'trigger_donchian_period': int(params.get('trigger_donchian_period', 60)),
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
                
                # ドンチャンミッドライン HyperER動的適応パラメータ
                'donchian_midline_period_min': float(params.get('donchian_midline_period_min', 55.0)),
                'donchian_midline_period_max': float(params.get('donchian_midline_period_max', 250.0)),
                
                # トリガー用ドンチャン HyperER動的適応パラメータ
                'trigger_donchian_period_min': float(params.get('trigger_donchian_period_min', 20.0)),
                'trigger_donchian_period_max': float(params.get('trigger_donchian_period_max', 100.0))
            })
        else:
            # HyperER動的適応が無効な場合のデフォルト値
            strategy_params.update({
                'frama_fc_min': 1.0,
                'frama_fc_max': 13.0,
                'frama_sc_min': 60.0,
                'frama_sc_max': 250.0,
                'donchian_midline_period_min': 55.0,
                'donchian_midline_period_max': 250.0,
                'trigger_donchian_period_min': 20.0,
                'trigger_donchian_period_max': 100.0
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
        # シンプルなストップロスは実装しない（チャネルベースのエグジットのため）
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
    
    print("=== ドンチャンFRAMAブレイクアウトストラテジーのテスト ===")
    
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
    strategy = DonchianFRAMABreakoutStrategy(
        donchian_midline_period=200,
        frama_period=16,
        trigger_donchian_period=60,
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
    
    print("\n=== テスト完了 ===")