#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import UltimateTrendSignalGenerator


class UltimateTrendStrategy(BaseStrategy):
    """
    Ultimate Trendストラテジー
    
    特徴:
    - Ultimate MAフィルタリングシステムとスーパートレンドロジックの統合
    - ATRベースの動的バンド調整による高精度なトレンド検出
    - アルティメットトレンドのトレンド方向によるシグナル生成
    - オプションでUltimate MAのトレンドシグナルによるフィルタリング
    - Numbaによる高速化処理
    
    エントリー条件:
    - ベース: trend == 1 でロング、trend == -1 でショート
    - フィルター有効時: 上記 + Ultimate MAのtrend_signalsが同方向
    
    エグジット条件:
    - ベース: trendが反転または0になった時
    - フィルター有効時: 上記 + Ultimate MAのtrend_signalsが反転
    """
    
    def __init__(
        self,
        # Ultimate Trendパラメータ
        length: int = 13,
        multiplier: float = 3.0,
        ultimate_smoother_period: int = 8,
        zero_lag_period: int = 21,
        filtering_mode: int = 1,
        # シグナル生成パラメータ
        use_filter: bool = False,
        enable_exit_signals: bool = True,
        # Ultimate MAの動的適応パラメータ
        zero_lag_period_mode: str = 'dynamic',
        # Ultimate MAのゼロラグ用サイクル検出器パラメータ
        zl_cycle_detector_type: str = 'absolute_ultimate',
        zl_cycle_detector_cycle_part: float = 0.5,
        zl_cycle_detector_max_cycle: int = 55,
        zl_cycle_detector_min_cycle: int = 5,
        zl_cycle_period_multiplier: float = 1.0,
        zl_cycle_detector_period_range: Tuple[int, int] = (3, 34)
    ):
        """
        初期化
        
        Args:
            # Ultimate Trendパラメータ
            length: ATR計算期間（デフォルト: 13）
            multiplier: ATR乗数（デフォルト: 2.0）
            ultimate_smoother_period: スーパースムーザー期間（デフォルト: 10）
            zero_lag_period: ゼロラグEMA期間（デフォルト: 21）
            filtering_mode: フィルタリングモード（デフォルト: 1）
            
            # シグナル生成パラメータ
            use_filter: フィルターオプション（True=Ultimate MAのtrend_signalsでフィルタリング）
            enable_exit_signals: 決済シグナルを有効にするか（デフォルト: True）
            
            # Ultimate MAの動的適応パラメータ
            zero_lag_period_mode: ゼロラグEMA期間モード（'dynamic' or 'fixed'）
            realtime_window_mode: リアルタイムウィンドウモード（'dynamic' or 'fixed'）
            
            # Ultimate MAのゼロラグ用サイクル検出器パラメータ
            zl_cycle_detector_type: ゼロラグ用サイクル検出器タイプ
            zl_cycle_detector_cycle_part: ゼロラグ用サイクル部分
            zl_cycle_detector_max_cycle: ゼロラグ用最大サイクル
            zl_cycle_detector_min_cycle: ゼロラグ用最小サイクル
            zl_cycle_period_multiplier: ゼロラグ用サイクル期間乗数
            zl_cycle_detector_period_range: ゼロラグ用period_rangeパラメータ
        """
        super().__init__("UltimateTrend")
        
        # パラメータの設定
        self._parameters = {
            'length': length,
            'multiplier': multiplier,
            'ultimate_smoother_period': ultimate_smoother_period,
            'zero_lag_period': zero_lag_period,
            'filtering_mode': filtering_mode,
            'use_filter': use_filter,
            'enable_exit_signals': enable_exit_signals,
            'zero_lag_period_mode': zero_lag_period_mode,
            'zl_cycle_detector_type': zl_cycle_detector_type,
            'zl_cycle_detector_cycle_part': zl_cycle_detector_cycle_part,
            'zl_cycle_detector_max_cycle': zl_cycle_detector_max_cycle,
            'zl_cycle_detector_min_cycle': zl_cycle_detector_min_cycle,
            'zl_cycle_period_multiplier': zl_cycle_period_multiplier,
            'zl_cycle_detector_period_range': zl_cycle_detector_period_range
        }
        
        # シグナル生成器の初期化
        self.signal_generator = UltimateTrendSignalGenerator(
            length=length,
            multiplier=multiplier,
            ultimate_smoother_period=ultimate_smoother_period,
            zero_lag_period=zero_lag_period,
            filtering_mode=filtering_mode,
            use_filter=use_filter,
            enable_exit_signals=enable_exit_signals,
            zero_lag_period_mode=zero_lag_period_mode,
            zl_cycle_detector_type=zl_cycle_detector_type,
            zl_cycle_detector_cycle_part=zl_cycle_detector_cycle_part,
            zl_cycle_detector_max_cycle=zl_cycle_detector_max_cycle,
            zl_cycle_detector_min_cycle=zl_cycle_detector_min_cycle,
            zl_cycle_period_multiplier=zl_cycle_period_multiplier,
            zl_cycle_detector_period_range=zl_cycle_detector_period_range
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
    
    def get_ultimate_trend_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Ultimate Trendラインの値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: Ultimate Trendラインの値
        """
        try:
            return self.signal_generator.get_ultimate_trend_values(data)
        except Exception as e:
            self.logger.error(f"Ultimate Trend値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルティメットトレンド方向を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: トレンド方向の配列（1=上昇、-1=下降、0=なし）
        """
        try:
            return self.signal_generator.get_trend(data)
        except Exception as e:
            self.logger.error(f"トレンド方向取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trend_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Ultimate MAのトレンドシグナルを取得
        
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
    
    def get_upper_band(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        上側バンドを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 上側バンドの配列
        """
        try:
            return self.signal_generator.get_upper_band(data)
        except Exception as e:
            self.logger.error(f"上側バンド取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_lower_band(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        下側バンドを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 下側バンドの配列
        """
        try:
            return self.signal_generator.get_lower_band(data)
        except Exception as e:
            self.logger.error(f"下側バンド取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_final_upper_band(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        調整済み上側バンドを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 調整済み上側バンドの配列
        """
        try:
            return self.signal_generator.get_final_upper_band(data)
        except Exception as e:
            self.logger.error(f"調整済み上側バンド取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_final_lower_band(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        調整済み下側バンドを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 調整済み下側バンドの配列
        """
        try:
            return self.signal_generator.get_final_lower_band(data)
        except Exception as e:
            self.logger.error(f"調整済み下側バンド取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_filtered_midline(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Ultimate MAフィルタ済みミッドラインを取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: フィルタ済みミッドラインの配列
        """
        try:
            return self.signal_generator.get_filtered_midline(data)
        except Exception as e:
            self.logger.error(f"フィルタ済みミッドライン取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_ukf_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        カルマンフィルター後の値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: カルマンフィルター後の値の配列
        """
        try:
            return self.signal_generator.get_ukf_values(data)
        except Exception as e:
            self.logger.error(f"カルマンフィルター値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_filtering_stats(self, data: Union[pd.DataFrame, np.ndarray]) -> dict:
        """
        フィルタリング統計を取得
        
        Args:
            data: 価格データ
            
        Returns:
            dict: フィルタリング統計
        """
        try:
            return self.signal_generator.get_filtering_stats(data)
        except Exception as e:
            self.logger.error(f"フィルタリング統計取得中にエラー: {str(e)}")
            return {}
    
    def get_all_ultimate_trend_stages(self, data: Union[pd.DataFrame, np.ndarray]) -> dict:
        """
        Ultimate Trendの全段階の結果を取得
        
        Args:
            data: 価格データ
            
        Returns:
            dict: 全段階の結果（バンド、トレンド、フィルタリング結果など）
        """
        try:
            return self.signal_generator.get_all_ultimate_trend_stages(data)
        except Exception as e:
            self.logger.error(f"Ultimate Trend全段階結果取得中にエラー: {str(e)}")
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
            # Ultimate Trendパラメータ
            'length': trial.suggest_int('length', 5, 50),
            'multiplier': trial.suggest_float('multiplier', 1.0, 5.0, step=0.1),
            'ultimate_smoother_period': trial.suggest_int('ultimate_smoother_period', 5, 20),
            'zero_lag_period': trial.suggest_int('zero_lag_period', 8, 233),
            'filtering_mode': trial.suggest_int('filtering_mode', 0, 2),
            
            # シグナル生成パラメータ
            'use_filter': trial.suggest_categorical('use_filter', [True, False]),
            
            # ゼロラグ用サイクル検出器パラメータ
            'zl_cycle_detector_type': trial.suggest_categorical('zl_cycle_detector_type', 
                ['hody', 'phac', 'dudi', 'dudi_e', 'hody_e', 'phac_e', 'cycle_period', 'cycle_period2', 
                 'bandpass_zero', 'autocorr_perio', 'dft_dominant', 'multi_bandpass', 'absolute_ultimate', 'ultra_supreme_stability']),
            'zl_cycle_detector_cycle_part': trial.suggest_float('zl_cycle_detector_cycle_part', 0.3, 2.0, step=0.1),
            'zl_cycle_detector_max_cycle': trial.suggest_int('zl_cycle_detector_max_cycle', 30, 200),
            'zl_cycle_detector_min_cycle': trial.suggest_int('zl_cycle_detector_min_cycle', 3, 15),
            'zl_cycle_period_multiplier': trial.suggest_float('zl_cycle_period_multiplier', 0.5, 2.0),
            
            # period_rangeパラメータ（タプルとして生成）
            'zl_cycle_detector_period_range_min': trial.suggest_int('zl_cycle_detector_period_range_min', 3, 15),
            'zl_cycle_detector_period_range_max': trial.suggest_int('zl_cycle_detector_period_range_max', 50, 200),
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
            'length': int(params['length']),
            'multiplier': float(params['multiplier']),
            'ultimate_smoother_period': int(params['ultimate_smoother_period']),
            'zero_lag_period': int(params['zero_lag_period']),
            'filtering_mode': int(params['filtering_mode']),
            'use_filter': bool(params.get('use_filter', False)),
            
            # ゼロラグ用サイクル検出器パラメータ
            'zl_cycle_detector_type': params.get('zl_cycle_detector_type', 'absolute_ultimate'),
            'zl_cycle_detector_cycle_part': float(params.get('zl_cycle_detector_cycle_part', 1.0)),
            'zl_cycle_detector_max_cycle': int(params.get('zl_cycle_detector_max_cycle', 120)),
            'zl_cycle_detector_min_cycle': int(params.get('zl_cycle_detector_min_cycle', 5)),
            'zl_cycle_period_multiplier': float(params.get('zl_cycle_period_multiplier', 1.0)),
            
            # period_rangeパラメータ（タプルに変換）
            'zl_cycle_detector_period_range': (
                int(params.get('zl_cycle_detector_period_range_min', 5)), 
                int(params.get('zl_cycle_detector_period_range_max', 120))
            ),
        }
        return strategy_params 