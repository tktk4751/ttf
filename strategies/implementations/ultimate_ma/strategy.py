#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import UltimateMASignalGenerator


class UltimateMAStrategy(BaseStrategy):
    """
    Ultimate MAストラテジー
    
    特徴:
    - 6段階革新的フィルタリングシステムによる高精度なシグナル生成
    - 適応的カルマンフィルター、スーパースムーザー、ゼロラグEMAを組み合わせ
    - ヒルベルト変換による位相遅延ゼロの実現
    - リアルタイムトレンド検出とトレンド判定の組み合わせ
    - ノイズ除去された超低遅延シグナル
    
    エントリー条件:
    - ロング: realtime_trends > 0 かつ trend_signals == 1
    - ショート: realtime_trends < 0 かつ trend_signals == -1
    
    エグジット条件:
    - ロング決済: realtime_trends <= 0 または trend_signals == -1
    - ショート決済: realtime_trends >= 0 または trend_signals == 1
    """
    
    def __init__(
        self,
        # Ultimate MAパラメータ
        super_smooth_period: int = 10,
        zero_lag_period: int = 21,
        realtime_window: int = 13,
        src_type: str = 'hlc3',
        slope_index: int = 4,
        range_threshold: float = 0.002,
        enable_exit_signals: bool = True,
        # 動的適応パラメータ
        zero_lag_period_mode: str = 'dynamic',
        realtime_window_mode: str = 'dynamic',
        # ゼロラグ用サイクル検出器パラメータ
        zl_cycle_detector_type: str = 'absolute_ultimate',
        zl_cycle_detector_cycle_part: float = 1.0,
        zl_cycle_detector_max_cycle: int = 120,
        zl_cycle_detector_min_cycle: int = 5,
        zl_cycle_period_multiplier: float = 1.0,
        # リアルタイムウィンドウ用サイクル検出器パラメータ
        rt_cycle_detector_type: str = 'absolute_ultimate',
        rt_cycle_detector_cycle_part: float = 0.5,
        rt_cycle_detector_max_cycle: int = 120,
        rt_cycle_detector_min_cycle: int = 5,
        rt_cycle_period_multiplier: float = 0.5,
        # period_rangeパラメータ
        zl_cycle_detector_period_range: Tuple[int, int] = (5, 120),
        rt_cycle_detector_period_range: Tuple[int, int] = (5, 120)
    ):
        """
        初期化
        
        Args:
            super_smooth_period: スーパースムーザーフィルター期間（デフォルト: 10）
            zero_lag_period: ゼロラグEMA期間（デフォルト: 21）
            realtime_window: リアルタイムトレンド検出ウィンドウ（デフォルト: 13）
            src_type: 価格ソース（'close', 'hlc3', 'hl2', 'ohlc4'など）
            slope_index: トレンド判定期間（デフォルト: 1）
            range_threshold: range判定の基本閾値（デフォルト: 0.005 = 0.5%）
            enable_exit_signals: 決済シグナルを有効にするか（デフォルト: True）
            
            # 動的適応パラメータ
            zero_lag_period_mode: ゼロラグ期間モード ('fixed' or 'dynamic')
            realtime_window_mode: リアルタイムウィンドウモード ('fixed' or 'dynamic')
            
            # ゼロラグ用サイクル検出器パラメータ
            zl_cycle_detector_type: ゼロラグ用サイクル検出器タイプ
            zl_cycle_detector_cycle_part: ゼロラグ用サイクル検出器のサイクル部分倍率
            zl_cycle_detector_max_cycle: ゼロラグ用サイクル検出器の最大サイクル期間
            zl_cycle_detector_min_cycle: ゼロラグ用サイクル検出器の最小サイクル期間
            zl_cycle_period_multiplier: ゼロラグ用サイクル期間の乗数
            
            # リアルタイムウィンドウ用サイクル検出器パラメータ
            rt_cycle_detector_type: リアルタイム用サイクル検出器タイプ
            rt_cycle_detector_cycle_part: リアルタイム用サイクル検出器のサイクル部分倍率
            rt_cycle_detector_max_cycle: リアルタイム用サイクル検出器の最大サイクル期間
            rt_cycle_detector_min_cycle: リアルタイム用サイクル検出器の最小サイクル期間
            rt_cycle_period_multiplier: リアルタイム用サイクル期間の乗数
            
            # period_rangeパラメータ
            zl_cycle_detector_period_range: ゼロラグ用サイクル検出器の周期範囲
            rt_cycle_detector_period_range: リアルタイム用サイクル検出器の周期範囲
        """
        super().__init__("UltimateMA")
        
        # パラメータの設定
        self._parameters = {
            'super_smooth_period': super_smooth_period,
            'zero_lag_period': zero_lag_period,
            'realtime_window': realtime_window,
            'src_type': src_type,
            'slope_index': slope_index,
            'range_threshold': range_threshold,
            'enable_exit_signals': enable_exit_signals,
            'zero_lag_period_mode': zero_lag_period_mode,
            'realtime_window_mode': realtime_window_mode,
            'zl_cycle_detector_type': zl_cycle_detector_type,
            'zl_cycle_detector_cycle_part': zl_cycle_detector_cycle_part,
            'zl_cycle_detector_max_cycle': zl_cycle_detector_max_cycle,
            'zl_cycle_detector_min_cycle': zl_cycle_detector_min_cycle,
            'zl_cycle_period_multiplier': zl_cycle_period_multiplier,
            'rt_cycle_detector_type': rt_cycle_detector_type,
            'rt_cycle_detector_cycle_part': rt_cycle_detector_cycle_part,
            'rt_cycle_detector_max_cycle': rt_cycle_detector_max_cycle,
            'rt_cycle_detector_min_cycle': rt_cycle_detector_min_cycle,
            'rt_cycle_period_multiplier': rt_cycle_period_multiplier,
            'zl_cycle_detector_period_range': zl_cycle_detector_period_range,
            'rt_cycle_detector_period_range': rt_cycle_detector_period_range
        }
        
        # シグナル生成器の初期化
        self.signal_generator = UltimateMASignalGenerator(
            super_smooth_period=super_smooth_period,
            zero_lag_period=zero_lag_period,
            realtime_window=realtime_window,
            src_type=src_type,
            slope_index=slope_index,
            range_threshold=range_threshold,
            enable_exit_signals=enable_exit_signals,
            zero_lag_period_mode=zero_lag_period_mode,
            realtime_window_mode=realtime_window_mode,
            zl_cycle_detector_type=zl_cycle_detector_type,
            zl_cycle_detector_cycle_part=zl_cycle_detector_cycle_part,
            zl_cycle_detector_max_cycle=zl_cycle_detector_max_cycle,
            zl_cycle_detector_min_cycle=zl_cycle_detector_min_cycle,
            zl_cycle_period_multiplier=zl_cycle_period_multiplier,
            rt_cycle_detector_type=rt_cycle_detector_type,
            rt_cycle_detector_cycle_part=rt_cycle_detector_cycle_part,
            rt_cycle_detector_max_cycle=rt_cycle_detector_max_cycle,
            rt_cycle_detector_min_cycle=rt_cycle_detector_min_cycle,
            rt_cycle_period_multiplier=rt_cycle_period_multiplier,
            zl_cycle_detector_period_range=zl_cycle_detector_period_range,
            rt_cycle_detector_period_range=rt_cycle_detector_period_range
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル（1=ロング、-1=ショート、0=シグナルなし）
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
    
    def get_ultimate_ma_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Ultimate MAの最終フィルター済み値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: Ultimate MAの最終フィルター済み値
        """
        try:
            return self.signal_generator.get_ultimate_ma_values(data)
        except Exception as e:
            self.logger.error(f"Ultimate MA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_realtime_trends(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        リアルタイムトレンド信号を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: リアルタイムトレンド信号の配列
        """
        try:
            return self.signal_generator.get_realtime_trends(data)
        except Exception as e:
            self.logger.error(f"リアルタイムトレンド取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        トレンドシグナルを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: トレンドシグナルの配列（1=上昇、-1=下降、0=range）
        """
        try:
            return self.signal_generator.get_trend_signals(data)
        except Exception as e:
            self.logger.error(f"トレンドシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_noise_reduction_stats(self, data: Union[pd.DataFrame, np.ndarray]) -> dict:
        """
        ノイズ除去統計を取得
        
        Args:
            data: 価格データ
            
        Returns:
            dict: ノイズ除去統計
        """
        try:
            return self.signal_generator.get_noise_reduction_stats(data)
        except Exception as e:
            self.logger.error(f"ノイズ除去統計取得中にエラー: {str(e)}")
            return {}
    
    def get_all_ultimate_ma_stages(self, data: Union[pd.DataFrame, np.ndarray]) -> dict:
        """
        Ultimate MAの全段階の結果を取得
        
        Args:
            data: 価格データ
            
        Returns:
            dict: 全段階の結果（生値からフィルター適用後まで）
        """
        try:
            return self.signal_generator.get_all_ultimate_ma_stages(data)
        except Exception as e:
            self.logger.error(f"Ultimate MA全段階結果取得中にエラー: {str(e)}")
            return {}
    
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
            # Ultimate MAパラメータ
            'super_smooth_period': trial.suggest_int('super_smooth_period', 5, 20),
            'zero_lag_period': trial.suggest_int('zero_lag_period', 8, 233),
            'realtime_window': trial.suggest_int('realtime_window', 5, 89),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'slope_index': trial.suggest_int('slope_index', 1, 8),
            'range_threshold': trial.suggest_float('range_threshold', 0.001, 0.02, step=0.001),
        
            
            # ゼロラグ用サイクル検出器パラメータ
            'zl_cycle_detector_type': trial.suggest_categorical('zl_cycle_detector_type', 
                ['hody', 'phac', 'dudi', 'dudi_e', 'hody_e', 'phac_e', 'cycle_period', 'cycle_period2', 
                 'bandpass_zero', 'autocorr_perio', 'dft_dominant', 'multi_bandpass', 'absolute_ultimate', 'ultra_supreme_stability']),
            'zl_cycle_detector_cycle_part': trial.suggest_float('zl_cycle_detector_cycle_part', 0.3, 2.0, step=0.1),
            'zl_cycle_detector_max_cycle': trial.suggest_int('zl_cycle_detector_max_cycle', 30, 200),
            'zl_cycle_detector_min_cycle': trial.suggest_int('zl_cycle_detector_min_cycle', 3, 15),
            'zl_cycle_period_multiplier': trial.suggest_float('zl_cycle_period_multiplier', 0.5, 2.0),
            
            # リアルタイムウィンドウ用サイクル検出器パラメータ
            'rt_cycle_detector_type': trial.suggest_categorical('rt_cycle_detector_type', 
                ['hody', 'phac', 'dudi', 'dudi_e', 'hody_e', 'phac_e', 'cycle_period', 'cycle_period2', 
                 'bandpass_zero', 'autocorr_perio', 'dft_dominant', 'multi_bandpass', 'absolute_ultimate', 'ultra_supreme_stability']),
            'rt_cycle_detector_cycle_part': trial.suggest_float('rt_cycle_detector_cycle_part', 0.3, 2.0, step=0.1),
            'rt_cycle_detector_max_cycle': trial.suggest_int('rt_cycle_detector_max_cycle', 30, 200),
            'rt_cycle_detector_min_cycle': trial.suggest_int('rt_cycle_detector_min_cycle', 3, 15),
            'rt_cycle_period_multiplier': trial.suggest_float('rt_cycle_period_multiplier', 0.1, 1.0),
            
            # period_rangeパラメータ（タプルとして生成）
            'zl_cycle_detector_period_range_min': trial.suggest_int('zl_cycle_detector_period_range_min', 3, 15),
            'zl_cycle_detector_period_range_max': trial.suggest_int('zl_cycle_detector_period_range_max', 50, 200),
            'rt_cycle_detector_period_range_min': trial.suggest_int('rt_cycle_detector_period_range_min', 3, 15),
            'rt_cycle_detector_period_range_max': trial.suggest_int('rt_cycle_detector_period_range_max', 50, 200),
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
            'super_smooth_period': int(params['super_smooth_period']),
            'zero_lag_period': int(params['zero_lag_period']),
            'realtime_window': int(params['realtime_window']),
            'src_type': params['src_type'],
            'slope_index': int(params['slope_index']),
            'range_threshold': float(params['range_threshold']),
            
            
            # ゼロラグ用サイクル検出器パラメータ
            'zl_cycle_detector_type': params.get('zl_cycle_detector_type', 'absolute_ultimate'),
            'zl_cycle_detector_cycle_part': float(params.get('zl_cycle_detector_cycle_part', 1.0)),
            'zl_cycle_detector_max_cycle': int(params.get('zl_cycle_detector_max_cycle', 120)),
            'zl_cycle_detector_min_cycle': int(params.get('zl_cycle_detector_min_cycle', 5)),
            'zl_cycle_period_multiplier': float(params.get('zl_cycle_period_multiplier', 1.0)),
            
            # リアルタイムウィンドウ用サイクル検出器パラメータ
            'rt_cycle_detector_type': params.get('rt_cycle_detector_type', 'phac_e'),
            'rt_cycle_detector_cycle_part': float(params.get('rt_cycle_detector_cycle_part', 1.0)),
            'rt_cycle_detector_max_cycle': int(params.get('rt_cycle_detector_max_cycle', 120)),
            'rt_cycle_detector_min_cycle': int(params.get('rt_cycle_detector_min_cycle', 5)),
            'rt_cycle_period_multiplier': float(params.get('rt_cycle_period_multiplier', 0.33)),
            
            # period_rangeパラメータ（タプルに変換）
            'zl_cycle_detector_period_range': (
                int(params.get('zl_cycle_detector_period_range_min', 5)), 
                int(params.get('zl_cycle_detector_period_range_max', 120))
            ),
            'rt_cycle_detector_period_range': (
                int(params.get('rt_cycle_detector_period_range_min', 5)), 
                int(params.get('rt_cycle_detector_period_range_max', 120))
            ),
        }
        return strategy_params 