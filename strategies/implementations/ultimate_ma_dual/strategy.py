#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import UltimateMADualSignalGenerator


class UltimateMADualStrategy(BaseStrategy):
    """
    Ultimate MAデュアルエントリーストラテジー
    
    特徴:
    - 短期・長期Ultimate MAのデュアルエントリーシステム
    - 6段階革新的フィルタリングシステムによる高精度なMA計算
    - 短期MAでエントリータイミング、長期MAでトレンド確認・決済判定
    - 適応的カルマンフィルター、スーパースムーザー、ゼロラグEMAを組み合わせ
    - ヒルベルト変換による位相遅延ゼロの実現
    
    エントリー条件:
    - ロングエントリー: 短期 realtime_trends > 0 かつ 短期 trend_signals == 1 かつ 長期 trend_signals == 1
    - ショートエントリー: 短期 realtime_trends < 0 かつ 短期 trend_signals == -1 かつ 長期 trend_signals == -1
    
    エグジット条件:
    - ロング決済: 長期 realtime_trends <= 0 または 長期 trend_signals == -1
    - ショート決済: 長期 realtime_trends >= 0 または 長期 trend_signals == 1
    """
    
    def __init__(
        self,
        # 短期Ultimate MAパラメータ
        short_super_smooth_period: int = 10,
        short_zero_lag_period: int = 21,
        short_realtime_window: int = 13,
        short_src_type: str = 'hlc3',
        short_slope_index: int = 1,
        short_range_threshold: float = 0.005,
        short_zero_lag_period_mode: str = 'dynamic',
        short_realtime_window_mode: str = 'dynamic',
        short_zl_cycle_detector_type: str = 'absolute_ultimate',
        short_zl_cycle_detector_cycle_part: float = 0.5,
        short_zl_cycle_detector_max_cycle: int = 55,
        short_zl_cycle_detector_min_cycle: int = 5,
        short_zl_cycle_period_multiplier: float = 1.0,
        short_rt_cycle_detector_type: str = 'absolute_ultimate',
        short_rt_cycle_detector_cycle_part: float = 0.5,
        short_rt_cycle_detector_max_cycle: int = 55,
        short_rt_cycle_detector_min_cycle: int = 5,
        short_rt_cycle_period_multiplier: float = 0.5,
        short_zl_cycle_detector_period_range: Tuple[int, int] = (5, 55),
        short_rt_cycle_detector_period_range: Tuple[int, int] = (5, 55),
        
        # 長期Ultimate MAパラメータ
        long_super_smooth_period: int = 20,
        long_zero_lag_period: int = 42,
        long_realtime_window: int = 26,
        long_src_type: str = 'hlc3',
        long_slope_index: int = 1,
        long_range_threshold: float = 0.005,
        long_zero_lag_period_mode: str = 'dynamic',
        long_realtime_window_mode: str = 'dynamic',
        long_zl_cycle_detector_type: str = 'absolute_ultimate',
        long_zl_cycle_detector_cycle_part: float = 1.0,
        long_zl_cycle_detector_max_cycle: int = 150,
        long_zl_cycle_detector_min_cycle: int = 10,
        long_zl_cycle_period_multiplier: float = 2.0,
        long_rt_cycle_detector_type: str = 'absolute_ultimate',
        long_rt_cycle_detector_cycle_part: float = 1.0,
        long_rt_cycle_detector_max_cycle: int = 150,
        long_rt_cycle_detector_min_cycle: int = 10,
        long_rt_cycle_period_multiplier: float = 0.5,
        long_zl_cycle_detector_period_range: Tuple[int, int] = (35, 200),
        long_rt_cycle_detector_period_range: Tuple[int, int] = (35, 200),
        
        # 共通パラメータ
        enable_exit_signals: bool = True
    ):
        """
        初期化
        
        Args:
            # 短期Ultimate MAパラメータ
            short_super_smooth_period: 短期スーパースムーザーフィルター期間（デフォルト: 10）
            short_zero_lag_period: 短期ゼロラグEMA期間（デフォルト: 21）
            short_realtime_window: 短期リアルタイムトレンド検出ウィンドウ（デフォルト: 13）
            short_src_type: 短期価格ソース（'close', 'hlc3', 'hl2', 'ohlc4'など）
            short_slope_index: 短期トレンド判定期間（デフォルト: 1）
            short_range_threshold: 短期range判定の基本閾値（デフォルト: 0.005 = 0.5%）
            short_zero_lag_period_mode: 短期ゼロラグ期間モード ('fixed' or 'dynamic')
            short_realtime_window_mode: 短期リアルタイムウィンドウモード ('fixed' or 'dynamic')
            
            # 短期サイクル検出器パラメータ
            short_zl_cycle_detector_type: 短期ゼロラグ用サイクル検出器タイプ
            short_zl_cycle_detector_cycle_part: 短期ゼロラグ用サイクル検出器のサイクル部分倍率
            short_zl_cycle_detector_max_cycle: 短期ゼロラグ用サイクル検出器の最大サイクル期間
            short_zl_cycle_detector_min_cycle: 短期ゼロラグ用サイクル検出器の最小サイクル期間
            short_zl_cycle_period_multiplier: 短期ゼロラグ用サイクル期間の乗数
            short_rt_cycle_detector_type: 短期リアルタイム用サイクル検出器タイプ
            short_rt_cycle_detector_cycle_part: 短期リアルタイム用サイクル検出器のサイクル部分倍率
            short_rt_cycle_detector_max_cycle: 短期リアルタイム用サイクル検出器の最大サイクル期間
            short_rt_cycle_detector_min_cycle: 短期リアルタイム用サイクル検出器の最小サイクル期間
            short_rt_cycle_period_multiplier: 短期リアルタイム用サイクル期間の乗数
            short_zl_cycle_detector_period_range: 短期ゼロラグ用サイクル検出器の周期範囲
            short_rt_cycle_detector_period_range: 短期リアルタイム用サイクル検出器の周期範囲
            
            # 長期Ultimate MAパラメータ（同様の説明で長期版）
            long_super_smooth_period: 長期スーパースムーザーフィルター期間（デフォルト: 20）
            long_zero_lag_period: 長期ゼロラグEMA期間（デフォルト: 42）
            long_realtime_window: 長期リアルタイムトレンド検出ウィンドウ（デフォルト: 26）
            long_src_type: 長期価格ソース
            long_slope_index: 長期トレンド判定期間
            long_range_threshold: 長期range判定の基本閾値
            long_zero_lag_period_mode: 長期ゼロラグ期間モード
            long_realtime_window_mode: 長期リアルタイムウィンドウモード
            
            # 長期サイクル検出器パラメータ
            long_zl_cycle_detector_type: 長期ゼロラグ用サイクル検出器タイプ
            long_zl_cycle_detector_cycle_part: 長期ゼロラグ用サイクル検出器のサイクル部分倍率
            long_zl_cycle_detector_max_cycle: 長期ゼロラグ用サイクル検出器の最大サイクル期間
            long_zl_cycle_detector_min_cycle: 長期ゼロラグ用サイクル検出器の最小サイクル期間
            long_zl_cycle_period_multiplier: 長期ゼロラグ用サイクル期間の乗数
            long_rt_cycle_detector_type: 長期リアルタイム用サイクル検出器タイプ
            long_rt_cycle_detector_cycle_part: 長期リアルタイム用サイクル検出器のサイクル部分倍率
            long_rt_cycle_detector_max_cycle: 長期リアルタイム用サイクル検出器の最大サイクル期間
            long_rt_cycle_detector_min_cycle: 長期リアルタイム用サイクル検出器の最小サイクル期間
            long_rt_cycle_period_multiplier: 長期リアルタイム用サイクル期間の乗数
            long_zl_cycle_detector_period_range: 長期ゼロラグ用サイクル検出器の周期範囲
            long_rt_cycle_detector_period_range: 長期リアルタイム用サイクル検出器の周期範囲
            
            # 共通パラメータ
            enable_exit_signals: 決済シグナルを有効にするか（デフォルト: True）
        """
        super().__init__("UltimateMADual")
        
        # パラメータの設定
        self._parameters = {
            # 短期パラメータ
            'short_super_smooth_period': short_super_smooth_period,
            'short_zero_lag_period': short_zero_lag_period,
            'short_realtime_window': short_realtime_window,
            'short_src_type': short_src_type,
            'short_slope_index': short_slope_index,
            'short_range_threshold': short_range_threshold,
            'short_zero_lag_period_mode': short_zero_lag_period_mode,
            'short_realtime_window_mode': short_realtime_window_mode,
            'short_zl_cycle_detector_type': short_zl_cycle_detector_type,
            'short_zl_cycle_detector_cycle_part': short_zl_cycle_detector_cycle_part,
            'short_zl_cycle_detector_max_cycle': short_zl_cycle_detector_max_cycle,
            'short_zl_cycle_detector_min_cycle': short_zl_cycle_detector_min_cycle,
            'short_zl_cycle_period_multiplier': short_zl_cycle_period_multiplier,
            'short_rt_cycle_detector_type': short_rt_cycle_detector_type,
            'short_rt_cycle_detector_cycle_part': short_rt_cycle_detector_cycle_part,
            'short_rt_cycle_detector_max_cycle': short_rt_cycle_detector_max_cycle,
            'short_rt_cycle_detector_min_cycle': short_rt_cycle_detector_min_cycle,
            'short_rt_cycle_period_multiplier': short_rt_cycle_period_multiplier,
            'short_zl_cycle_detector_period_range': short_zl_cycle_detector_period_range,
            'short_rt_cycle_detector_period_range': short_rt_cycle_detector_period_range,
            
            # 長期パラメータ
            'long_super_smooth_period': long_super_smooth_period,
            'long_zero_lag_period': long_zero_lag_period,
            'long_realtime_window': long_realtime_window,
            'long_src_type': long_src_type,
            'long_slope_index': long_slope_index,
            'long_range_threshold': long_range_threshold,
            'long_zero_lag_period_mode': long_zero_lag_period_mode,
            'long_realtime_window_mode': long_realtime_window_mode,
            'long_zl_cycle_detector_type': long_zl_cycle_detector_type,
            'long_zl_cycle_detector_cycle_part': long_zl_cycle_detector_cycle_part,
            'long_zl_cycle_detector_max_cycle': long_zl_cycle_detector_max_cycle,
            'long_zl_cycle_detector_min_cycle': long_zl_cycle_detector_min_cycle,
            'long_zl_cycle_period_multiplier': long_zl_cycle_period_multiplier,
            'long_rt_cycle_detector_type': long_rt_cycle_detector_type,
            'long_rt_cycle_detector_cycle_part': long_rt_cycle_detector_cycle_part,
            'long_rt_cycle_detector_max_cycle': long_rt_cycle_detector_max_cycle,
            'long_rt_cycle_detector_min_cycle': long_rt_cycle_detector_min_cycle,
            'long_rt_cycle_period_multiplier': long_rt_cycle_period_multiplier,
            'long_zl_cycle_detector_period_range': long_zl_cycle_detector_period_range,
            'long_rt_cycle_detector_period_range': long_rt_cycle_detector_period_range,
            
            # 共通パラメータ
            'enable_exit_signals': enable_exit_signals
        }
        
        # シグナル生成器の初期化
        self.signal_generator = UltimateMADualSignalGenerator(
            short_super_smooth_period=short_super_smooth_period,
            short_zero_lag_period=short_zero_lag_period,
            short_realtime_window=short_realtime_window,
            short_src_type=short_src_type,
            short_slope_index=short_slope_index,
            short_range_threshold=short_range_threshold,
            short_zero_lag_period_mode=short_zero_lag_period_mode,
            short_realtime_window_mode=short_realtime_window_mode,
            short_zl_cycle_detector_type=short_zl_cycle_detector_type,
            short_zl_cycle_detector_cycle_part=short_zl_cycle_detector_cycle_part,
            short_zl_cycle_detector_max_cycle=short_zl_cycle_detector_max_cycle,
            short_zl_cycle_detector_min_cycle=short_zl_cycle_detector_min_cycle,
            short_zl_cycle_period_multiplier=short_zl_cycle_period_multiplier,
            short_rt_cycle_detector_type=short_rt_cycle_detector_type,
            short_rt_cycle_detector_cycle_part=short_rt_cycle_detector_cycle_part,
            short_rt_cycle_detector_max_cycle=short_rt_cycle_detector_max_cycle,
            short_rt_cycle_detector_min_cycle=short_rt_cycle_detector_min_cycle,
            short_rt_cycle_period_multiplier=short_rt_cycle_period_multiplier,
            short_zl_cycle_detector_period_range=short_zl_cycle_detector_period_range,
            short_rt_cycle_detector_period_range=short_rt_cycle_detector_period_range,
            
            long_super_smooth_period=long_super_smooth_period,
            long_zero_lag_period=long_zero_lag_period,
            long_realtime_window=long_realtime_window,
            long_src_type=long_src_type,
            long_slope_index=long_slope_index,
            long_range_threshold=long_range_threshold,
            long_zero_lag_period_mode=long_zero_lag_period_mode,
            long_realtime_window_mode=long_realtime_window_mode,
            long_zl_cycle_detector_type=long_zl_cycle_detector_type,
            long_zl_cycle_detector_cycle_part=long_zl_cycle_detector_cycle_part,
            long_zl_cycle_detector_max_cycle=long_zl_cycle_detector_max_cycle,
            long_zl_cycle_detector_min_cycle=long_zl_cycle_detector_min_cycle,
            long_zl_cycle_period_multiplier=long_zl_cycle_period_multiplier,
            long_rt_cycle_detector_type=long_rt_cycle_detector_type,
            long_rt_cycle_detector_cycle_part=long_rt_cycle_detector_cycle_part,
            long_rt_cycle_detector_max_cycle=long_rt_cycle_detector_max_cycle,
            long_rt_cycle_detector_min_cycle=long_rt_cycle_detector_min_cycle,
            long_rt_cycle_period_multiplier=long_rt_cycle_period_multiplier,
            long_zl_cycle_detector_period_range=long_zl_cycle_detector_period_range,
            long_rt_cycle_detector_period_range=long_rt_cycle_detector_period_range,
            
            enable_exit_signals=enable_exit_signals
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
    
    def get_short_ma_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        短期Ultimate MAの値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 短期Ultimate MAの値
        """
        try:
            return self.signal_generator.get_short_ma_values(data)
        except Exception as e:
            self.logger.error(f"短期Ultimate MA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_long_ma_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        長期Ultimate MAの値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 長期Ultimate MAの値
        """
        try:
            return self.signal_generator.get_long_ma_values(data)
        except Exception as e:
            self.logger.error(f"長期Ultimate MA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_short_realtime_trends(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        短期Ultimate MAのrealtime_trendsを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 短期Ultimate MAのrealtime_trends
        """
        try:
            return self.signal_generator.get_short_realtime_trends(data)
        except Exception as e:
            self.logger.error(f"短期realtime_trends取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_long_realtime_trends(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        長期Ultimate MAのrealtime_trendsを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 長期Ultimate MAのrealtime_trends
        """
        try:
            return self.signal_generator.get_long_realtime_trends(data)
        except Exception as e:
            self.logger.error(f"長期realtime_trends取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_short_trend_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        短期Ultimate MAのトレンドシグナルを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 短期Ultimate MAのトレンドシグナル
        """
        try:
            return self.signal_generator.get_short_trend_signals(data)
        except Exception as e:
            self.logger.error(f"短期トレンドシグナル取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_long_trend_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        長期Ultimate MAのトレンドシグナルを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 長期Ultimate MAのトレンドシグナル
        """
        try:
            return self.signal_generator.get_long_trend_signals(data)
        except Exception as e:
            self.logger.error(f"長期トレンドシグナル取得中にエラー: {str(e)}")
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
            dict: 全段階の結果（短期・長期それぞれの生値からフィルター適用後まで）
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
            # 短期Ultimate MAパラメータ
            'short_super_smooth_period': trial.suggest_int('short_super_smooth_period', 3, 15),
            'short_zero_lag_period': trial.suggest_int('short_zero_lag_period', 5, 34),
            'short_realtime_window': trial.suggest_int('short_realtime_window', 5, 21),
            'short_src_type': trial.suggest_categorical('short_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'short_slope_index': trial.suggest_int('short_slope_index', 1, 5),
            'short_range_threshold': trial.suggest_float('short_range_threshold', 0.001, 0.02, step=0.001),
            
            # 長期Ultimate MAパラメータ
            'long_super_smooth_period': trial.suggest_int('long_super_smooth_period', 10, 30),
            'long_zero_lag_period': trial.suggest_int('long_zero_lag_period', 21, 89),
            'long_realtime_window': trial.suggest_int('long_realtime_window', 13, 55),
            'long_src_type': trial.suggest_categorical('long_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'long_slope_index': trial.suggest_int('long_slope_index', 1, 8),
            'long_range_threshold': trial.suggest_float('long_range_threshold', 0.001, 0.02, step=0.001),
            
            # 短期ゼロラグ用サイクル検出器パラメータ
            'short_zl_cycle_detector_type': trial.suggest_categorical('short_zl_cycle_detector_type', 
                ['hody', 'phac', 'dudi', 'dudi_e', 'hody_e', 'phac_e', 'cycle_period', 'cycle_period2', 
                 'bandpass_zero', 'autocorr_perio', 'dft_dominant', 'multi_bandpass', 'absolute_ultimate', 'ultra_supreme_stability']),
            'short_zl_cycle_detector_cycle_part': trial.suggest_float('short_zl_cycle_detector_cycle_part', 0.3, 2.0, step=0.1),
            'short_zl_cycle_detector_max_cycle': trial.suggest_int('short_zl_cycle_detector_max_cycle', 30, 200),
            'short_zl_cycle_detector_min_cycle': trial.suggest_int('short_zl_cycle_detector_min_cycle', 3, 15),
            'short_zl_cycle_period_multiplier': trial.suggest_float('short_zl_cycle_period_multiplier', 0.5, 2.0),
            
            # 短期リアルタイムウィンドウ用サイクル検出器パラメータ
            'short_rt_cycle_detector_type': trial.suggest_categorical('short_rt_cycle_detector_type', 
                ['hody', 'phac', 'dudi', 'dudi_e', 'hody_e', 'phac_e', 'cycle_period', 'cycle_period2', 
                 'bandpass_zero', 'autocorr_perio', 'dft_dominant', 'multi_bandpass', 'absolute_ultimate', 'ultra_supreme_stability']),
            'short_rt_cycle_detector_cycle_part': trial.suggest_float('short_rt_cycle_detector_cycle_part', 0.3, 2.0, step=0.1),
            'short_rt_cycle_detector_max_cycle': trial.suggest_int('short_rt_cycle_detector_max_cycle', 30, 200),
            'short_rt_cycle_detector_min_cycle': trial.suggest_int('short_rt_cycle_detector_min_cycle', 3, 15),
            'short_rt_cycle_period_multiplier': trial.suggest_float('short_rt_cycle_period_multiplier', 0.1, 1.0),
            
            # 長期ゼロラグ用サイクル検出器パラメータ
            'long_zl_cycle_detector_type': trial.suggest_categorical('long_zl_cycle_detector_type', 
                ['hody', 'phac', 'dudi', 'dudi_e', 'hody_e', 'phac_e', 'cycle_period', 'cycle_period2', 
                 'bandpass_zero', 'autocorr_perio', 'dft_dominant', 'multi_bandpass', 'absolute_ultimate', 'ultra_supreme_stability']),
            'long_zl_cycle_detector_cycle_part': trial.suggest_float('long_zl_cycle_detector_cycle_part', 0.3, 2.0, step=0.1),
            'long_zl_cycle_detector_max_cycle': trial.suggest_int('long_zl_cycle_detector_max_cycle', 50, 300),
            'long_zl_cycle_detector_min_cycle': trial.suggest_int('long_zl_cycle_detector_min_cycle', 5, 20),
            'long_zl_cycle_period_multiplier': trial.suggest_float('long_zl_cycle_period_multiplier', 0.5, 2.0),
            
            # 長期リアルタイムウィンドウ用サイクル検出器パラメータ
            'long_rt_cycle_detector_type': trial.suggest_categorical('long_rt_cycle_detector_type', 
                ['hody', 'phac', 'dudi', 'dudi_e', 'hody_e', 'phac_e', 'cycle_period', 'cycle_period2', 
                 'bandpass_zero', 'autocorr_perio', 'dft_dominant', 'multi_bandpass', 'absolute_ultimate', 'ultra_supreme_stability']),
            'long_rt_cycle_detector_cycle_part': trial.suggest_float('long_rt_cycle_detector_cycle_part', 0.3, 2.0, step=0.1),
            'long_rt_cycle_detector_max_cycle': trial.suggest_int('long_rt_cycle_detector_max_cycle', 50, 300),
            'long_rt_cycle_detector_min_cycle': trial.suggest_int('long_rt_cycle_detector_min_cycle', 5, 20),
            'long_rt_cycle_period_multiplier': trial.suggest_float('long_rt_cycle_period_multiplier', 0.1, 1.0),
            
            # period_rangeパラメータ（タプルとして生成）
            'short_zl_cycle_detector_period_range_min': trial.suggest_int('short_zl_cycle_detector_period_range_min', 3, 10),
            'short_zl_cycle_detector_period_range_max': trial.suggest_int('short_zl_cycle_detector_period_range_max', 50, 200),
            'short_rt_cycle_detector_period_range_min': trial.suggest_int('short_rt_cycle_detector_period_range_min', 3, 10),
            'short_rt_cycle_detector_period_range_max': trial.suggest_int('short_rt_cycle_detector_period_range_max', 50, 200),
            'long_zl_cycle_detector_period_range_min': trial.suggest_int('long_zl_cycle_detector_period_range_min', 5, 15),
            'long_zl_cycle_detector_period_range_max': trial.suggest_int('long_zl_cycle_detector_period_range_max', 100, 300),
            'long_rt_cycle_detector_period_range_min': trial.suggest_int('long_rt_cycle_detector_period_range_min', 5, 15),
            'long_rt_cycle_detector_period_range_max': trial.suggest_int('long_rt_cycle_detector_period_range_max', 100, 300),
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
            # 短期Ultimate MAパラメータ
            'short_super_smooth_period': int(params['short_super_smooth_period']),
            'short_zero_lag_period': int(params['short_zero_lag_period']),
            'short_realtime_window': int(params['short_realtime_window']),
            'short_src_type': params.get('short_src_type', 'hlc3'),
            'short_slope_index': int(params.get('short_slope_index', 1)),
            'short_range_threshold': float(params.get('short_range_threshold', 0.005)),
            
            # 長期Ultimate MAパラメータ
            'long_super_smooth_period': int(params['long_super_smooth_period']),
            'long_zero_lag_period': int(params['long_zero_lag_period']),
            'long_realtime_window': int(params['long_realtime_window']),
            'long_src_type': params.get('long_src_type', 'hlc3'),
            'long_slope_index': int(params.get('long_slope_index', 1)),
            'long_range_threshold': float(params.get('long_range_threshold', 0.005)),
            
            # 短期ゼロラグ用サイクル検出器パラメータ
            'short_zl_cycle_detector_type': params.get('short_zl_cycle_detector_type', 'absolute_ultimate'),
            'short_zl_cycle_detector_cycle_part': float(params.get('short_zl_cycle_detector_cycle_part', 1.0)),
            'short_zl_cycle_detector_max_cycle': int(params.get('short_zl_cycle_detector_max_cycle', 120)),
            'short_zl_cycle_detector_min_cycle': int(params.get('short_zl_cycle_detector_min_cycle', 5)),
            'short_zl_cycle_period_multiplier': float(params.get('short_zl_cycle_period_multiplier', 1.0)),
            
            # 短期リアルタイムウィンドウ用サイクル検出器パラメータ
            'short_rt_cycle_detector_type': params.get('short_rt_cycle_detector_type', 'phac_e'),
            'short_rt_cycle_detector_cycle_part': float(params.get('short_rt_cycle_detector_cycle_part', 1.0)),
            'short_rt_cycle_detector_max_cycle': int(params.get('short_rt_cycle_detector_max_cycle', 120)),
            'short_rt_cycle_detector_min_cycle': int(params.get('short_rt_cycle_detector_min_cycle', 5)),
            'short_rt_cycle_period_multiplier': float(params.get('short_rt_cycle_period_multiplier', 0.33)),
            
            # 長期ゼロラグ用サイクル検出器パラメータ
            'long_zl_cycle_detector_type': params.get('long_zl_cycle_detector_type', 'absolute_ultimate'),
            'long_zl_cycle_detector_cycle_part': float(params.get('long_zl_cycle_detector_cycle_part', 1.0)),
            'long_zl_cycle_detector_max_cycle': int(params.get('long_zl_cycle_detector_max_cycle', 240)),
            'long_zl_cycle_detector_min_cycle': int(params.get('long_zl_cycle_detector_min_cycle', 10)),
            'long_zl_cycle_period_multiplier': float(params.get('long_zl_cycle_period_multiplier', 1.0)),
            
            # 長期リアルタイムウィンドウ用サイクル検出器パラメータ
            'long_rt_cycle_detector_type': params.get('long_rt_cycle_detector_type', 'phac_e'),
            'long_rt_cycle_detector_cycle_part': float(params.get('long_rt_cycle_detector_cycle_part', 1.0)),
            'long_rt_cycle_detector_max_cycle': int(params.get('long_rt_cycle_detector_max_cycle', 240)),
            'long_rt_cycle_detector_min_cycle': int(params.get('long_rt_cycle_detector_min_cycle', 10)),
            'long_rt_cycle_period_multiplier': float(params.get('long_rt_cycle_period_multiplier', 0.33)),
            
            # period_rangeパラメータ（タプルに変換）
            'short_zl_cycle_detector_period_range': (
                int(params.get('short_zl_cycle_detector_period_range_min', 5)), 
                int(params.get('short_zl_cycle_detector_period_range_max', 120))
            ),
            'short_rt_cycle_detector_period_range': (
                int(params.get('short_rt_cycle_detector_period_range_min', 5)), 
                int(params.get('short_rt_cycle_detector_period_range_max', 120))
            ),
            'long_zl_cycle_detector_period_range': (
                int(params.get('long_zl_cycle_detector_period_range_min', 10)), 
                int(params.get('long_zl_cycle_detector_period_range_max', 240))
            ),
            'long_rt_cycle_detector_period_range': (
                int(params.get('long_rt_cycle_detector_period_range_min', 10)), 
                int(params.get('long_rt_cycle_detector_period_range_max', 240))
            ),
        }
        return strategy_params 