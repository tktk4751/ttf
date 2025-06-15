#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZASimpleSignalGenerator


class ZASimpleStrategy(BaseStrategy):
    """
    ZAシンプルストラテジー
    
    特徴:
    - サイクル効率比（CER）に基づく動的パラメータ最適化
    - ZAdaptiveChannelによる高精度なエントリーポイント検出
    
    エントリー条件:
    - ロング: ZAdaptiveChannelの買いシグナル
    - ショート: ZAdaptiveChannelの売りシグナル
    
    エグジット条件:
    - ロング: ZAdaptiveChannelの売りシグナル
    - ショート: ZAdaptiveChannelの買いシグナル
    """
    
    def __init__(
        self,
        # 基本パラメータ
        band_lookback: int = 1,
        src_type: str = 'hlc3',
        
        # 乗数計算方法選択
        multiplier_method: str = 'simple_adjustment',  # 'adaptive', 'simple', 'simple_adjustment'
        
        # トリガーソース選択
        multiplier_source: str = 'cer',  # 'cer', 'x_trend', 'z_trend'
        ma_source: str = 'x_trend',      # ZAdaptiveMAに渡すソース（'cer', 'x_trend'）
        
        # X-Trend Index調整の有効化
        use_x_trend_adjustment: bool = True,


        # CERパラメータ
        detector_type: str = 'absolute_ultimate',     # CER用ドミナントサイクル検出器タイプ
        cycle_part: float = 0.5,           # CER用サイクル部分
        lp_period: int = 5,               # CER用ローパスフィルター期間
        hp_period: int = 100,              # CER用ハイパスフィルター期間
        max_cycle: int = 120,              # CER用最大サイクル期間
        min_cycle: int = 5,               # CER用最小サイクル期間
        max_output: int = 89,             # CER用最大出力値
        min_output: int = 5,              # CER用最小出力値
        
        # ZAdaptiveMA用パラメータ
        fast_period: int = 2,             # 速い移動平均の期間（固定値）
        slow_period: int = 144,           # 遅い移動平均の期間（固定値）
        
        # Xトレンドインデックスパラメータ
        x_detector_type: str = 'cycle_period2',
        x_cycle_part: float = 0.7,
        x_max_cycle: int = 120,
        x_min_cycle: int = 5,
        x_max_output: int = 89,
        x_min_output: int = 5,
        x_smoother_type: str = 'alma',
        
        # 固定しきい値のパラメータ（XTrendIndex用）
        fixed_threshold: float = 0.65,

    ):
        """
        初期化
        
        Args:
            band_lookback: 過去バンド参照期間（デフォルト: 1）
            src_type: 価格ソースタイプ（デフォルト: 'hlc3'）
            
            # 動的乗数の範囲パラメータ
            max_max_multiplier: 最大乗数の最大値（デフォルト: 9.0）
            min_max_multiplier: 最大乗数の最小値（デフォルト: 2.0）
            max_min_multiplier: 最小乗数の最大値（デフォルト: 4.0）
            min_min_multiplier: 最小乗数の最小値（デフォルト: 0.5）
            
            # 乗数計算方法選択
            multiplier_method: 乗数計算方法（デフォルト: 'simple_adjustment'）
            
            # トリガーソース選択
            multiplier_source: 乗数計算に使用するトリガーのソース（デフォルト: 'cer'）
            ma_source: ZAdaptiveMAに渡すソース（デフォルト: 'x_trend'）
            
            # X-Trend Index調整の有効化
            use_x_trend_adjustment: X-Trend Index調整の有効化（デフォルト: True）
            
            # 乗数平滑化オプション
            multiplier_smoothing_method: 乗数平滑化方法（デフォルト: 'none'）
            multiplier_smoothing_period: 平滑化期間（デフォルト: 4）
            alma_offset: ALMA用オフセット（デフォルト: 0.85）
            alma_sigma: ALMA用シグマ（デフォルト: 6）
            
            # CERパラメータ
            detector_type: CER用ドミナントサイクル検出器タイプ（デフォルト: 'phac_e'）
            cycle_part: CER用サイクル部分（デフォルト: 0.5）
            lp_period: CER用ローパスフィルター期間（デフォルト: 5）
            hp_period: CER用ハイパスフィルター期間（デフォルト: 100）
            max_cycle: CER用最大サイクル期間（デフォルト: 120）
            min_cycle: CER用最小サイクル期間（デフォルト: 5）
            max_output: CER用最大出力値（デフォルト: 89）
            min_output: CER用最小出力値（デフォルト: 5）
            use_kalman_filter: CER用カルマンフィルター使用有無（デフォルト: False）
            
            # ZAdaptiveMA用パラメータ
            fast_period: 速い移動平均の期間（デフォルト: 2）
            slow_period: 遅い移動平均の期間（デフォルト: 144）
            
            # Xトレンドインデックスパラメータ
            x_detector_type: Xトレンド用検出器タイプ（デフォルト: 'dudi_e'）
            x_cycle_part: Xトレンド用サイクル部分（デフォルト: 0.7）
            x_max_cycle: Xトレンド用最大サイクル期間（デフォルト: 120）
            x_min_cycle: Xトレンド用最小サイクル期間（デフォルト: 5）
            x_max_output: Xトレンド用最大出力値（デフォルト: 55）
            x_min_output: Xトレンド用最小出力値（デフォルト: 8）
            x_smoother_type: Xトレンド用平滑化タイプ（デフォルト: 'alma'）
            
            # 固定しきい値のパラメータ（XTrendIndex用）
            fixed_threshold: 固定しきい値（デフォルト: 0.65）
            
            # ROC Persistenceパラメータ
            roc_detector_type: ROC Persistence用検出器タイプ（デフォルト: 'hody_e'）
            roc_max_persistence_periods: ROC Persistence用最大持続期間（デフォルト: 89）
            roc_smooth_persistence: ROC Persistence平滑化有無（デフォルト: False）
            roc_persistence_smooth_period: ROC Persistence平滑化期間（デフォルト: 3）
            roc_smooth_roc: ROC Persistence平滑化ROC有無（デフォルト: True）
            roc_alma_period: ROC Persistence用ALMA期間（デフォルト: 5）
            roc_alma_offset: ROC Persistence用ALMAオフセット（デフォルト: 0.85）
            roc_alma_sigma: ROC Persistence用ALMAシグマ（デフォルト: 6）
            roc_signal_threshold: ROC Persistence信号閾値（デフォルト: 0.0）
            
            # Cycle RSXパラメータ
            cycle_rsx_detector_type: サイクルRSX用検出器タイプ（デフォルト: 'dudi_e'）
            cycle_rsx_lp_period: サイクルRSX用ローパスフィルター期間（デフォルト: 5）
            cycle_rsx_hp_period: サイクルRSX用ハイパスフィルター期間（デフォルト: 89）
            cycle_rsx_cycle_part: サイクルRSX用サイクル部分（デフォルト: 0.4）
            cycle_rsx_max_cycle: サイクルRSX用最大サイクル期間（デフォルト: 55）
            cycle_rsx_min_cycle: サイクルRSX用最小サイクル期間（デフォルト: 5）
            cycle_rsx_max_output: サイクルRSX用最大出力値（デフォルト: 34）
            cycle_rsx_min_output: サイクルRSX用最小出力値（デフォルト: 3）
            cycle_rsx_src_type: サイクルRSX用ソースタイプ（デフォルト: 'hlc3'）
        """
        super().__init__("ZASimple")
        
        # パラメータの設定
        self._parameters = {
            # 基本パラメータ
            'band_lookback': band_lookback,
            'src_type': src_type,
            

            
            # 乗数計算方法選択
            'multiplier_method': multiplier_method,
            
            # トリガーソース選択
            'multiplier_source': multiplier_source,
            'ma_source': ma_source,
            
            # X-Trend Index調整の有効化
            'use_x_trend_adjustment': use_x_trend_adjustment,
            

            # CERパラメータ
            'detector_type': detector_type,
            'cycle_part': cycle_part,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            
            # ZAdaptiveMA用パラメータ
            'fast_period': fast_period,
            'slow_period': slow_period,
            
            # Xトレンドインデックスパラメータ
            'x_detector_type': x_detector_type,
            'x_cycle_part': x_cycle_part,
            'x_max_cycle': x_max_cycle,
            'x_min_cycle': x_min_cycle,
            'x_max_output': x_max_output,
            'x_min_output': x_min_output,
            'x_smoother_type': x_smoother_type,
            
            # 固定しきい値のパラメータ（XTrendIndex用）
            'fixed_threshold': fixed_threshold,
            
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZASimpleSignalGenerator(
            # 基本パラメータ
            band_lookback=band_lookback,
            src_type=src_type,

            
            # 乗数計算方法選択
            multiplier_method=multiplier_method,
            
            # トリガーソース選択
            multiplier_source=multiplier_source,
            ma_source=ma_source,
            
            # X-Trend Index調整の有効化
            use_x_trend_adjustment=use_x_trend_adjustment,
            

            # CERパラメータ
            detector_type=detector_type,
            cycle_part=cycle_part,
            lp_period=lp_period,
            hp_period=hp_period,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            
            # ZAdaptiveMA用パラメータ
            fast_period=fast_period,
            slow_period=slow_period,
            
            # Xトレンドインデックスパラメータ
            x_detector_type=x_detector_type,
            x_cycle_part=x_cycle_part,
            x_max_cycle=x_max_cycle,
            x_min_cycle=x_min_cycle,
            x_max_output=x_max_output,
            x_min_output=x_min_output,
            x_smoother_type=x_smoother_type,
            
            # 固定しきい値のパラメータ（XTrendIndex用）
            fixed_threshold=fixed_threshold,

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
            # 基本パラメータ
            'band_lookback': trial.suggest_int('band_lookback', 1, 5),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # トリガーソース選択
            'multiplier_source': trial.suggest_categorical('multiplier_source', 
                                                          ['cer', 'x_trend', 'z_trend']),
            'ma_source': trial.suggest_categorical('ma_source', ['cer', 'x_trend']),
            
            
            # CERパラメータ
            'detector_type': trial.suggest_categorical('detector_type', 
                                                      ['dudi', 'hody', 'phac', 'dudi_e', 'hody_e', 'phac_e']),
            'cycle_part': trial.suggest_float('cycle_part', 0.2, 0.9, step=0.1),

            # ZAdaptiveMA用パラメータ
            'fast_period': trial.suggest_int('fast_period', 1, 5),
            'slow_period': trial.suggest_int('slow_period', 10, 200),
            
            # Xトレンドインデックスパラメータ
            'x_detector_type': trial.suggest_categorical('x_detector_type', 
                                                        ['dudi', 'hody', 'phac', 'dudi_e', 'hody_e', 'phac_e']),
            'x_cycle_part': trial.suggest_float('x_cycle_part', 0.3, 0.9, step=0.1),



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
            # 基本パラメータ
            'band_lookback': int(params['band_lookback']),
            'src_type': params['src_type'],
            
            # トリガーソース選択
            'multiplier_source': params['multiplier_source'],
            'ma_source': params['ma_source'],
            
            # CERパラメータ
            'detector_type': params['detector_type'],
            'cycle_part': float(params['cycle_part']),

            
            # ZAdaptiveMA用パラメータ
            'fast_period': int(params['fast_period']),
            'slow_period': int(params['slow_period']),
            
            # Xトレンドインデックスパラメータ
            'x_detector_type': params['x_detector_type'],
            'x_cycle_part': float(params['x_cycle_part']),

            
            

        }
        return strategy_params 