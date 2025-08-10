#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import HyperFRAMAEnhancedSignalGenerator, FilterType


class HyperFRAMAEnhancedStrategy(BaseStrategy):
    """
    HyperFRAMA Enhanced ストラテジー
    
    特徴:
    - HyperFRAMA (Hyper Fractal Adaptive Moving Average) ベースの高度なトレードシステム
    - フラクタル次元に基づく適応型移動平均線
    - アルファ調整係数による柔軟な設定（0.1〜1.0）
    - 4つの高度なフィルターから選択可能:
      1. HyperER Filter - 効率性比率ベースの高精度トレンド判定
      2. HyperTrendIndex Filter - 高度なトレンドインデックスによる判定
      3. HyperADX Filter - 方向性移動インデックスによる判定
      4. Consensus Filter - 3つのフィルターの合意判定（3つのうち2つが1を出力）
    - フィルターなしオプションも提供
    
    エントリー条件:
    - ロング: HyperFRAMAシグナル=1 かつ フィルターシグナル=1（フィルター有効時）
    - ショート: HyperFRAMAシグナル=-1 かつ フィルターシグナル=-1（フィルター有効時）
    - フィルターシグナル=0または逆方向の場合はスルー
    
    エグジット条件:
    - ロング: HyperFRAMAシグナル=-1
    - ショート: HyperFRAMAシグナル=1
    
    革新的な優位性:
    - フラクタル次元による市場適応制御
    - 複数のEhlersアルゴリズムの統合による高精度判定
    - 適応的フィルタリングによる誤判定の大幅削減
    - 市場状態に応じた自動フィルター調整
    - Numba JIT最適化による高速処理
    """
    
    def __init__(
        self,
        # HyperFRAMAパラメータ
        period: int = 6,
        src_type: str = 'oc2',
        fc: int = 4,
        sc: int = 120,
        alpha_multiplier: float = 0.12,
        # 動的期間パラメータ
        period_mode: str = 'fixed',
        cycle_detector_type: str = 'dft_dominant',
        lp_period: int = 10,
        hp_period: int = 60,
        cycle_part: float = 0.5,
        max_cycle: int = 124,
        min_cycle: int = 13,
        max_output: int = 124,
        min_output: int = 12,
        # シグナル設定
        position_mode: bool = False,              # 位置関係シグナル(True)またはクロスオーバーシグナル(False)
        # フィルター選択
        filter_type: FilterType = FilterType.NONE,  # フィルタータイプ
        # HyperER フィルターパラメータ（2つ目）
        filter_hyper_er_period: int = 14,
        filter_hyper_er_midline_period: int = 100,
        # HyperTrendIndex フィルターパラメータ
        filter_hyper_trend_index_period: int = 14,
        filter_hyper_trend_index_midline_period: int = 100,
        # HyperADX フィルターパラメータ
        filter_hyper_adx_period: int = 14,
        filter_hyper_adx_midline_period: int = 100
    ):
        """
        初期化
        
        Args:
            period: 期間（偶数である必要がある、デフォルト: 16）
            src_type: ソースタイプ（デフォルト: 'hl2'）
            fc: Fast Constant（デフォルト: 1）
            sc: Slow Constant（デフォルト: 198）
            alpha_multiplier: アルファ調整係数（デフォルト: 0.5）
            period_mode: 期間モード ('fixed' または 'dynamic')
            cycle_detector_type: サイクル検出器タイプ
            ... その他の動的期間パラメータ
            position_mode: 位置関係シグナル(True)またはクロスオーバーシグナル(False)
            filter_type: フィルタータイプ（デフォルト: FilterType.PHASOR_TREND）
            その他: 各フィルターのパラメータ
        """
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        signal_type = "Position" if position_mode else "Crossover"
        
        super().__init__(f"HyperFRAMA_Enhanced_{signal_type}_{filter_name}")
        
        # パラメータの設定
        self._parameters = {
            # HyperFRAMAパラメータ
            'period': period,
            'src_type': src_type,
            'fc': fc,
            'sc': sc,
            'alpha_multiplier': alpha_multiplier,
            # 動的期間パラメータ
            'period_mode': period_mode,
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output,
            # シグナル設定
            'position_mode': position_mode,
            # フィルター設定
            'filter_type': filter_type,
            # HyperER フィルターパラメータ
            'filter_hyper_er_period': filter_hyper_er_period,
            'filter_hyper_er_midline_period': filter_hyper_er_midline_period,
            # HyperTrendIndex フィルターパラメータ
            'filter_hyper_trend_index_period': filter_hyper_trend_index_period,
            'filter_hyper_trend_index_midline_period': filter_hyper_trend_index_midline_period,
            # HyperADX フィルターパラメータ
            'filter_hyper_adx_period': filter_hyper_adx_period,
            'filter_hyper_adx_midline_period': filter_hyper_adx_midline_period
        }
        
        # シグナル生成器の初期化
        self.signal_generator = HyperFRAMAEnhancedSignalGenerator(
            # HyperFRAMAパラメータ
            period=period,
            src_type=src_type,
            fc=fc,
            sc=sc,
            alpha_multiplier=alpha_multiplier,
            # 動的期間パラメータ
            period_mode=period_mode,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            # シグナル設定
            position_mode=position_mode,
            # フィルター設定
            filter_type=filter_type,
            # HyperER フィルターパラメータ
            filter_hyper_er_period=filter_hyper_er_period,
            filter_hyper_er_midline_period=filter_hyper_er_midline_period,
            # HyperTrendIndex フィルターパラメータ
            filter_hyper_trend_index_period=filter_hyper_trend_index_period,
            filter_hyper_trend_index_midline_period=filter_hyper_trend_index_midline_period,
            # HyperADX フィルターパラメータ
            filter_hyper_adx_period=filter_hyper_adx_period,
            filter_hyper_adx_midline_period=filter_hyper_adx_midline_period
        )
    
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
    
    def get_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """FRAMA値を取得"""
        try:
            return self.signal_generator.get_frama_values(data)
        except Exception as e:
            self.logger.error(f"FRAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_adjusted_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """Adjusted FRAMA値を取得"""
        try:
            return self.signal_generator.get_adjusted_frama_values(data)
        except Exception as e:
            self.logger.error(f"Adjusted FRAMA値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fractal_dimension(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """フラクタル次元を取得"""
        try:
            return self.signal_generator.get_fractal_dimension(data)
        except Exception as e:
            self.logger.error(f"フラクタル次元取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """Alpha値を取得"""
        try:
            return self.signal_generator.get_alpha_values(data)
        except Exception as e:
            self.logger.error(f"Alpha値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_long_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ロングエントリーシグナル取得"""
        try:
            return self.signal_generator.get_long_signals(data)
        except Exception as e:
            self.logger.error(f"ロングシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_short_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ショートエントリーシグナル取得"""
        try:
            return self.signal_generator.get_short_signals(data)
        except Exception as e:
            self.logger.error(f"ショートシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_hyper_frama_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """HyperFRAMAシグナル取得"""
        try:
            return self.signal_generator.get_hyper_frama_signals(data)
        except Exception as e:
            self.logger.error(f"HyperFRAMAシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_filter_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """フィルターシグナル取得"""
        try:
            return self.signal_generator.get_filter_signals(data)
        except Exception as e:
            self.logger.error(f"フィルターシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_filter_details(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """フィルター詳細情報を取得"""
        try:
            return self.signal_generator.get_filter_details(data)
        except Exception as e:
            self.logger.error(f"フィルター詳細取得中にエラー: {str(e)}")
            return {}
    
    def get_advanced_metrics(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """全ての高度なメトリクスを取得"""
        try:
            return self.signal_generator.get_advanced_metrics(data)
        except Exception as e:
            self.logger.error(f"高度なメトリクス取得中にエラー: {str(e)}")
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
        # フィルタータイプの選択
        filter_type = trial.suggest_categorical('filter_type', [
            FilterType.NONE.value,
            FilterType.HYPER_ER.value,
            FilterType.HYPER_TREND_INDEX.value,
            FilterType.HYPER_ADX.value,
            FilterType.CONSENSUS.value
        ])
        
        params = {
            # HyperFRAMAパラメータ
            'period': trial.suggest_int('period', 2, 34, step=2),  # 偶数のみ
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4','oc2']),
            'fc': trial.suggest_int('fc', 1, 10),
            'sc': trial.suggest_int('sc', 40, 300, step=5),
            'alpha_multiplier': trial.suggest_float('alpha_multiplier', 0.02, 0.8, step=0.01),

            
            # フィルター設定
            'filter_type': filter_type,
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
            # HyperFRAMAパラメータ
            'period': int(params.get('period', 16)),
            'src_type': params.get('src_type', 'hl2'),
            'fc': int(params.get('fc', 1)),
            'sc': int(params.get('sc', 198)),
            'alpha_multiplier': float(params.get('alpha_multiplier', 0.5)),

            # フィルター設定
            'filter_type': FilterType(params['filter_type'])
        }
        

        
        return strategy_params
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """ストラテジー情報を取得"""
        filter_type = self._parameters.get('filter_type', FilterType.NONE)
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        
        return {
            'name': 'HyperFRAMA Enhanced Strategy',
            'description': f'Hyper Fractal Adaptive Moving Average with {filter_name} Filter Integration',
            'parameters': self._parameters.copy(),
            'features': [
                'Fractal dimension-based adaptive control',
                'Configurable alpha multiplier (0.1-1.0)',
                'Market cycle-responsive moving average system',
                'Multiple Ehlers algorithm integration',
                f'Advanced {filter_name} filtering',
                'Configurable crossover or position-based signals',
                'Optimized with Numba JIT compilation',
                'High-precision trend and cycle detection with fractal analysis'
            ],
            'filter_capabilities': {
                'hyper_er': 'HyperER efficiency ratio-based trend filtering',
                'hyper_trend_index': 'HyperTrendIndex advanced trend detection',
                'hyper_adx': 'HyperADX directional movement filtering',
                'consensus': '3-filter consensus (2 out of 3 agreement)',
                'none': 'Pure HyperFRAMA signals without filtering'
            },
            'adaptive_features': {
                'fractal_dimension_control': 'Fractal dimension-based parameter adjustment',
                'alpha_multiplier_scaling': 'Flexible alpha scaling for different market conditions',
                'dynamic_period_adaptation': 'Optional dynamic period based on cycle detection'
            }
        }
    
    def reset(self) -> None:
        """ストラテジーの状態をリセット"""
        super().reset()
        if hasattr(self.signal_generator, 'reset'):
            self.signal_generator.reset()