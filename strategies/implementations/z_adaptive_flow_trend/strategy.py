#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZAdaptiveFlowTrendSignalGenerator


class ZAdaptiveFlowTrendStrategy(BaseStrategy):
    """
    Z Adaptive Flowトレンドチェンジ戦略
    
    特徴:
    - Z Adaptive Flowのトレンド状態変化に基づくエントリー・エグジット
    - 固定乗数による安定したシグナル生成
    - 全てのZ Adaptive Flowパラメータをサポート
    
    エントリー条件:
    - ロング: Z Adaptive Flowのトレンドチェンジシグナルが1（-1から1への変化）
    - ショート: Z Adaptive Flowのトレンドチェンジシグナルが-1（1から-1への変化）
    
    エグジット条件:
    - ロング: Z Adaptive Flowのトレンドチェンジシグナルが-1（1から-1への変化）
    - ショート: Z Adaptive Flowのトレンドチェンジシグナルが1（-1から1への変化）
    """
    
    def __init__(
        self,
        # 基本パラメータ
        length: int = 10,
        smooth_length: int = 14,
        src_type: str = 'hlc3',
        
        # MAタイプ選択
        ma_type: str = 'hma',
        
        # MA共通パラメータ
        ma_period: int = None,
        ma_use_dynamic_period: bool = True,
        ma_slope_index: int = 1,
        ma_range_threshold: float = 0.005,
        
        # ボラティリティタイプ選択
        volatility_type: str = 'atr',
        
        # 固定乗数パラメータ
        slow_period_multiplier: float = 5.0,
        volatility_multiplier: float = 3.0,
        
        # 共通サイクル検出器パラメータ
        detector_type: str = 'absolute_ultimate',
        cycle_part: float = 1.0,
        lp_period: int = 10,
        hp_period: int = 48,
        max_cycle: int = 233,
        min_cycle: int = 13,
        max_output: int = 144,
        min_output: int = 13,
        
        # ALMA固有パラメータ
        alma_offset: float = 0.85,
        alma_sigma: float = 6,
        
        # ZAdaptiveMA固有パラメータ
        z_adaptive_ma_fast_period: int = 2,
        z_adaptive_ma_slow_period: int = 144,
        
        # EfficiencyRatio固有パラメータ
        efficiency_ratio_period: int = 5,
        efficiency_ratio_smoothing_method: str = 'hma',
        efficiency_ratio_use_dynamic_period: bool = True,
        efficiency_ratio_slope_index: int = 3,
        efficiency_ratio_range_threshold: float = 0.005,
        efficiency_ratio_smoother_period: int = 13,
        
        # ATR固有パラメータ
        atr_period: int = None,
        atr_smoothing_method: str = 'alma',
        atr_use_dynamic_period: bool = False,
        atr_slope_index: int = 1,
        atr_range_threshold: float = 0.005,
        
        # volatility固有パラメータ
        volatility_period_mode: str = 'fixed',
        volatility_fixed_period: int = None,
        volatility_calculation_mode: str = 'return',
        volatility_return_type: str = 'log',
        volatility_ddof: int = 1,
        volatility_smoother_type: str = 'hma',
        volatility_smoother_period: int = None,
        
        # 以下のパラメータは互換性のため残すが使用しない
        adaptive_trigger: str = 'chop_trend',
        adaptive_power: float = 1.0,
        adaptive_invert: bool = False,
        adaptive_reverse_mapping: bool = False,
        **trigger_params
    ):
        """
        初期化
        
        Args:
            # 基本パラメータ
            length: メイン期間（デフォルト: 10）
            smooth_length: ボラティリティ平滑化期間（デフォルト: 14）
            src_type: 価格ソースタイプ（デフォルト: 'hlc3'）
            
            # MAタイプ選択
            ma_type: MAタイプ（デフォルト: 'zlema'）
            
            # MA共通パラメータ
            ma_period: MA用期間（デフォルト: None = lengthを使用）
            ma_use_dynamic_period: 動的期間を使用するかどうか（デフォルト: True）
            ma_slope_index: スロープインデックス（デフォルト: 1）
            ma_range_threshold: range閾値（デフォルト: 0.005）
            
            # ボラティリティタイプ選択
            volatility_type: ボラティリティタイプ（デフォルト: 'volatility'）
            
            # 固定乗数パラメータ
            slow_period_multiplier: スロー期間乗数（デフォルト: 4.0）
            volatility_multiplier: ボラティリティ乗数（デフォルト: 2.0）
            
            # 共通サイクル検出器パラメータ
            detector_type: サイクル検出器タイプ（デフォルト: 'cycle_period2'）
            cycle_part: サイクル部分（デフォルト: 1.0）
            lp_period: 低周波期間（デフォルト: 10）
            hp_period: 高周波期間（デフォルト: 48）
            max_cycle: 最大サイクル（デフォルト: 233）
            min_cycle: 最小サイクル（デフォルト: 13）
            max_output: 最大出力（デフォルト: 144）
            min_output: 最小出力（デフォルト: 13）
            
            # ALMA固有パラメータ
            alma_offset: ALMAオフセット（デフォルト: 0.85）
            alma_sigma: ALMAシグマ（デフォルト: 6）
            
            # ZAdaptiveMA固有パラメータ
            z_adaptive_ma_fast_period: ZAdaptiveMAファスト期間（デフォルト: 2）
            z_adaptive_ma_slow_period: ZAdaptiveMAスロー期間（デフォルト: 144）
            
            # EfficiencyRatio固有パラメータ
            efficiency_ratio_period: EfficiencyRatio期間（デフォルト: 5）
            efficiency_ratio_smoothing_method: EfficiencyRatio平滑化方法（デフォルト: 'hma'）
            efficiency_ratio_use_dynamic_period: EfficiencyRatio動的期間使用（デフォルト: True）
            efficiency_ratio_slope_index: EfficiencyRatioスロープインデックス（デフォルト: 3）
            efficiency_ratio_range_threshold: EfficiencyRatio range閾値（デフォルト: 0.005）
            efficiency_ratio_smoother_period: EfficiencyRatio平滑化期間（デフォルト: 13）
            
            # ATR固有パラメータ
            atr_period: ATR期間（デフォルト: None = lengthを使用）
            atr_smoothing_method: ATR平滑化方法（デフォルト: 'alma'）
            atr_use_dynamic_period: ATR動的期間使用（デフォルト: False）
            atr_slope_index: ATRスロープインデックス（デフォルト: 1）
            atr_range_threshold: ATR range閾値（デフォルト: 0.005）
            
            # volatility固有パラメータ
            volatility_period_mode: ボラティリティ期間モード（デフォルト: 'fixed'）
            volatility_fixed_period: ボラティリティ固定期間（デフォルト: None = lengthを使用）
            volatility_calculation_mode: ボラティリティ計算モード（デフォルト: 'return'）
            volatility_return_type: ボラティリティリターンタイプ（デフォルト: 'log'）
            volatility_ddof: ボラティリティ自由度調整（デフォルト: 1）
            volatility_smoother_type: ボラティリティ平滑化タイプ（デフォルト: 'hma'）
            volatility_smoother_period: ボラティリティ平滑化期間（デフォルト: None = smooth_lengthを使用）
            
            # 互換性のためのパラメータ（未使用）
            adaptive_trigger: 旧AdaptivePeriod用トリガー（デフォルト: 'chop_trend'）
            adaptive_power: 旧AdaptivePeriod用べき乗値（デフォルト: 1.0）
            adaptive_invert: 旧AdaptivePeriod用反転フラグ（デフォルト: False）
            adaptive_reverse_mapping: 旧AdaptivePeriod用逆マッピング（デフォルト: False）
            **trigger_params: 旧トリガー用追加パラメータ
        """
        super().__init__("ZAdaptiveFlowTrend")
        
        # パラメータの設定
        self._parameters = {
            # 基本パラメータ
            'length': length,
            'smooth_length': smooth_length,
            'src_type': src_type,
            
            # MAタイプ選択
            'ma_type': ma_type,
            
            # MA共通パラメータ
            'ma_period': ma_period,
            'ma_use_dynamic_period': ma_use_dynamic_period,
            'ma_slope_index': ma_slope_index,
            'ma_range_threshold': ma_range_threshold,
            
            # ボラティリティタイプ選択
            'volatility_type': volatility_type,
            
            # 固定乗数パラメータ
            'slow_period_multiplier': slow_period_multiplier,
            'volatility_multiplier': volatility_multiplier,
            
            # 共通サイクル検出器パラメータ
            'detector_type': detector_type,
            'cycle_part': cycle_part,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            
            # ALMA固有パラメータ
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            
            # ZAdaptiveMA固有パラメータ
            'z_adaptive_ma_fast_period': z_adaptive_ma_fast_period,
            'z_adaptive_ma_slow_period': z_adaptive_ma_slow_period,
            
            # EfficiencyRatio固有パラメータ
            'efficiency_ratio_period': efficiency_ratio_period,
            'efficiency_ratio_smoothing_method': efficiency_ratio_smoothing_method,
            'efficiency_ratio_use_dynamic_period': efficiency_ratio_use_dynamic_period,
            'efficiency_ratio_slope_index': efficiency_ratio_slope_index,
            'efficiency_ratio_range_threshold': efficiency_ratio_range_threshold,
            'efficiency_ratio_smoother_period': efficiency_ratio_smoother_period,
            
            # ATR固有パラメータ
            'atr_period': atr_period,
            'atr_smoothing_method': atr_smoothing_method,
            'atr_use_dynamic_period': atr_use_dynamic_period,
            'atr_slope_index': atr_slope_index,
            'atr_range_threshold': atr_range_threshold,
            
            # volatility固有パラメータ
            'volatility_period_mode': volatility_period_mode,
            'volatility_fixed_period': volatility_fixed_period,
            'volatility_calculation_mode': volatility_calculation_mode,
            'volatility_return_type': volatility_return_type,
            'volatility_ddof': volatility_ddof,
            'volatility_smoother_type': volatility_smoother_type,
            'volatility_smoother_period': volatility_smoother_period,
            
            # 互換性のためのパラメータ
            'adaptive_trigger': adaptive_trigger,
            'adaptive_power': adaptive_power,
            'adaptive_invert': adaptive_invert,
            'adaptive_reverse_mapping': adaptive_reverse_mapping,
            **trigger_params
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZAdaptiveFlowTrendSignalGenerator(**self._parameters)
        
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # シグナル生成器からエントリーシグナルを取得
            return self.signal_generator.get_entry_signals(data)
            
        except Exception as e:
            self.logger.error(f"エントリーシグナル生成中にエラーが発生しました: {str(e)}")
            # エラー時は空のシグナル配列を返す
            data_len = len(data) if hasattr(data, '__len__') else 0
            return np.zeros(data_len, dtype=np.int8)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する
        
        Args:
            data: 価格データ
            position: 現在のポジション (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1 = 最新のデータ）
            
        Returns:
            bool: エグジットシグナル（True: 決済, False: 保持）
        """
        try:
            # シグナル生成器からエグジットシグナルを取得
            return self.signal_generator.get_exit_signals(data, position, index)
            
        except Exception as e:
            self.logger.error(f"エグジットシグナル生成中にエラーが発生しました: {str(e)}")
            return False
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成する
        
        Args:
            trial: Optunaトライアルオブジェクト
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        return {
            # 基本パラメータ
            'length': trial.suggest_int('length', 5, 50),
            'smooth_length': trial.suggest_int('smooth_length', 5, 30),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'ohlc4']),
            
            # MAタイプ選択
            'ma_type': trial.suggest_categorical('ma_type', ['hma', 'alma', 'z_adaptive_ma', 'zlema']),
            
            # MA共通パラメータ
            'ma_use_dynamic_period': True,
            'ma_slope_index': trial.suggest_int('ma_slope_index', 1, 5),
            
            # ボラティリティタイプ選択
            'volatility_type': trial.suggest_categorical('volatility_type', ['volatility', 'atr']),
            
            # 固定乗数パラメータ
            'slow_period_multiplier': trial.suggest_float('slow_period_multiplier', 2.0, 8.0,step=0.1),
            'volatility_multiplier': trial.suggest_float('volatility_multiplier', 1.0, 5.0,step=0.1),
            
            # 共通サイクル検出器パラメータ
            'detector_type': trial.suggest_categorical('detector_type', 
                ['hody', 'phac', 'dudi', 'dudi_e', 'hody_e', 'phac_e', 'cycle_period', 'cycle_period2', 'bandpass_zero', 'autocorr_perio', 'dft_dominant', 'multi_bandpass']),
            'cycle_part': trial.suggest_float('cycle_part', 0.5, 2.0),
            'lp_period': trial.suggest_int('lp_period', 5, 20),
            'hp_period': trial.suggest_int('hp_period', 30, 100),
            'max_cycle': trial.suggest_int('max_cycle', 100, 300),
            'min_cycle': trial.suggest_int('min_cycle', 5, 20),
            'max_output': trial.suggest_int('max_output', 50, 200),
            'min_output': trial.suggest_int('min_output', 5, 20),
            
            
            # ZAdaptiveMA固有パラメータ
            'z_adaptive_ma_fast_period': trial.suggest_int('z_adaptive_ma_fast_period', 2, 20),
            'z_adaptive_ma_slow_period': trial.suggest_int('z_adaptive_ma_slow_period', 20, 150),
            
            # EfficiencyRatio固有パラメータ
            'efficiency_ratio_period': trial.suggest_int('efficiency_ratio_period', 3, 15),
            'efficiency_ratio_smoothing_method': trial.suggest_categorical(
                'efficiency_ratio_smoothing_method', ['hma', 'alma', 'zlema']),
            'efficiency_ratio_use_dynamic_period': True,
            'efficiency_ratio_slope_index': trial.suggest_int('efficiency_ratio_slope_index', 1, 5),
            'efficiency_ratio_smoother_period': trial.suggest_int('efficiency_ratio_smoother_period', 5, 20),
            
            # ATR固有パラメータ
            'atr_smoothing_method': trial.suggest_categorical('atr_smoothing_method', ['alma', 'hma', 'zlema']),
            'atr_use_dynamic_period': trial.suggest_categorical('atr_use_dynamic_period', [True, False]),
            'atr_slope_index': trial.suggest_int('atr_slope_index', 1, 5),
            
            # volatility固有パラメータ
            'volatility_period_mode': trial.suggest_categorical('volatility_period_mode', ['fixed', 'adaptive']),
            'volatility_calculation_mode': trial.suggest_categorical('volatility_calculation_mode', ['return', 'price']),
            'volatility_return_type': trial.suggest_categorical('volatility_return_type', ['log', 'simple']),
            'volatility_ddof': trial.suggest_int('volatility_ddof', 0, 2),
            'volatility_smoother_type': trial.suggest_categorical(
                'volatility_smoother_type', ['hma', 'alma', 'zlema', 'none']),
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換する
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            Dict[str, Any]: 戦略パラメータ
        """
        # パラメータをそのまま使用（追加の変換は不要）
        return params.copy()
    
    def get_trend_state(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Z Adaptive Flowのトレンド状態を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: トレンド状態の値 (1: bullish, -1: bearish)
        """
        return self.signal_generator.get_trend_state(data)
    
    def get_detailed_result(self, data: Union[pd.DataFrame, np.ndarray] = None):
        """
        Z Adaptive Flowの詳細な計算結果を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            ZAdaptiveFlowResult: 詳細な計算結果
        """
        return self.signal_generator.get_detailed_result(data)
    
    def get_trend_lines(self, data: Union[pd.DataFrame, np.ndarray] = None):
        """
        メインのトレンドライン（basis, level）を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            basis, level のタプル
        """
        return self.signal_generator.get_trend_lines(data)
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None):
        """
        バンド（upper, lower）を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            upper, lower のタプル
        """
        return self.signal_generator.get_bands(data) 