#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import HyperChannelSignalGenerator, FilterType


class HyperChannelStrategy(BaseStrategy):
    """
    ハイパーチャネル ストラテジー
    
    特徴:
    - ハイパーチャネル (Hyper Channel) ベースの高度なトレードシステム
    - Unified Smootherによる高精度ミッドライン
    - X_ATRを使用したボラティリティベースのバンド幅
    - HyperER/HyperADXによる動的適応乗数
    - 4つの高度なフィルターから選択可能:
      1. HyperER Filter - 効率性比率ベースの高精度トレンド判定
      2. HyperTrendIndex Filter - 高度なトレンドインデックスによる判定
      3. HyperADX Filter - 方向性移動インデックスによる判定
      4. Consensus Filter - 3つのフィルターの合意判定（3つのうち2つが1を出力）
    - フィルターなしオプションも提供
    
    エントリー条件:
    - ロング: チャネル上限ブレイクアウト=1 かつ フィルターシグナル=1（フィルター有効時）
    - ショート: チャネル下限ブレイクアウト=-1 かつ フィルターシグナル=1（フィルター有効時）
    - フィルターシグナル=0または逆方向の場合はスルー
    
    エグジット条件:
    - ロング: 下限ブレイクアウト=-1 またはミッドライン割れ（オプション）
    - ショート: 上限ブレイクアウト=1 またはミッドライン抜け（オプション）
    
    革新的な優位性:
    - Unified Smootherによる市場適応制御
    - X_ATRによるボラティリティベースの動的バンド幅
    - 複数のEhlersアルゴリズムの統合による高精度判定
    - 適応的フィルタリングによる誤判定の大幅削減
    - 市場状態に応じた自動フィルター調整
    - Numba JIT最適化による高速処理
    """
    
    def __init__(
        self,
        # ハイパーチャネルパラメータ
        smoother_type: str = 'frama',
        smoother_period: int = 14,
        smoother_src_type: str = 'hlc3',
        atr_period: float = 14.0,
        atr_tr_method: str = 'str',
        atr_smoother_type: str = 'laguerre',
        atr_src_type: str = 'close',
        adaptation_type: str = 'hyper_er',
        max_multiplier: float = 5.0,
        min_multiplier: float = 0.5,
        
        # ブレイクアウトシグナルパラメータ
        band_lookback: int = 1,
        
        # フィルター選択
        filter_type: FilterType = FilterType.NONE,
        
        # HyperER フィルターパラメータ
        filter_hyper_er_period: int = 14,
        filter_hyper_er_midline_period: int = 100,
        
        # HyperTrendIndex フィルターパラメータ
        filter_hyper_trend_index_period: int = 14,
        filter_hyper_trend_index_midline_period: int = 100,
        
        # HyperADX フィルターパラメータ
        filter_hyper_adx_period: int = 14,
        filter_hyper_adx_midline_period: int = 100,
        
        # エグジット設定
        enable_midline_exit: bool = True,
        
        # その他のハイパーチャネルパラメータ
        **hyper_channel_kwargs
    ):
        """
        初期化
        
        Args:
            smoother_type: 使用するスムーサータイプ
            smoother_period: スムーサー期間
            smoother_src_type: スムーサー用価格ソース
            atr_period: X_ATR期間
            atr_tr_method: X_ATR TR計算方法
            atr_smoother_type: X_ATR スムーサータイプ
            atr_src_type: X_ATR用価格ソース
            adaptation_type: 動的適応タイプ
            max_multiplier: 最大乗数
            min_multiplier: 最小乗数
            band_lookback: 过去バンド参照期間
            filter_type: フィルタータイプ（デフォルト: FilterType.NONE）
            enable_midline_exit: ミッドライン回帰エグジットを有効化
            その他: ハイパーチャネル固有の追加パラメータ
        """
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        
        super().__init__(f"HyperChannel_{smoother_type}_{adaptation_type}_{filter_name}")
        
        # パラメータの設定
        self._parameters = {
            # ハイパーチャネルパラメータ
            'smoother_type': smoother_type,
            'smoother_period': smoother_period,
            'smoother_src_type': smoother_src_type,
            'atr_period': atr_period,
            'atr_tr_method': atr_tr_method,
            'atr_smoother_type': atr_smoother_type,
            'atr_src_type': atr_src_type,
            'adaptation_type': adaptation_type,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            # ブレイクアウトパラメータ
            'band_lookback': band_lookback,
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
            'filter_hyper_adx_midline_period': filter_hyper_adx_midline_period,
            # エグジット設定
            'enable_midline_exit': enable_midline_exit,
            # その他
            **hyper_channel_kwargs
        }
        
        # シグナル生成器の初期化
        self.signal_generator = HyperChannelSignalGenerator(
            # ハイパーチャネルパラメータ
            smoother_type=smoother_type,
            smoother_period=smoother_period,
            smoother_src_type=smoother_src_type,
            atr_period=atr_period,
            atr_tr_method=atr_tr_method,
            atr_smoother_type=atr_smoother_type,
            atr_src_type=atr_src_type,
            adaptation_type=adaptation_type,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            # ブレイクアウトパラメータ
            band_lookback=band_lookback,
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
            filter_hyper_adx_midline_period=filter_hyper_adx_midline_period,
            # エグジット設定
            enable_midline_exit=enable_midline_exit,
            # その他
            **hyper_channel_kwargs
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
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """チャネルバンド値を取得"""
        try:
            return self.signal_generator.get_band_values(data)
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            empty = np.array([])
            return empty, empty, empty
    
    def get_x_atr_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """X_ATR値を取得"""
        try:
            return self.signal_generator.get_x_atr_values(data)
        except Exception as e:
            self.logger.error(f"X_ATR値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """動的乗数を取得"""
        try:
            return self.signal_generator.get_dynamic_multiplier(data)
        except Exception as e:
            self.logger.error(f"動的乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_adaptation_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """動的適応値を取得"""
        try:
            return self.signal_generator.get_adaptation_values(data)
        except Exception as e:
            self.logger.error(f"動的適応値取得中にエラー: {str(e)}")
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
    
    def get_channel_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """ハイパーチャネルブレイクアウトシグナル取得"""
        try:
            return self.signal_generator.get_channel_signals(data)
        except Exception as e:
            self.logger.error(f"チャネルシグナル取得中にエラー: {str(e)}")
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
            # ハイパーチャネルパラメータ
            'smoother_type': trial.suggest_categorical('smoother_type', 
                ['frama', 'super_smoother', 'ultimate_smoother', 'laguerre']),
            'smoother_period': trial.suggest_int('smoother_period', 8, 34, step=2),
            'smoother_src_type': trial.suggest_categorical('smoother_src_type', 
                ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
            'atr_period': trial.suggest_float('atr_period', 8.0, 28.0, step=2.0),
            'atr_tr_method': trial.suggest_categorical('atr_tr_method', ['str', 'atr', 'range']),
            'atr_smoother_type': trial.suggest_categorical('atr_smoother_type', 
                ['laguerre', 'ema', 'super_smoother']),
            'adaptation_type': trial.suggest_categorical('adaptation_type', ['hyper_er', 'hyper_adx']),
            'max_multiplier': trial.suggest_float('max_multiplier', 2.0, 8.0, step=0.5),
            'min_multiplier': trial.suggest_float('min_multiplier', 0.2, 1.5, step=0.1),
            # ブレイクアウトパラメータ
            'band_lookback': trial.suggest_int('band_lookback', 0, 3),
            # フィルター設定
            'filter_type': filter_type,
            # エグジット設定
            'enable_midline_exit': trial.suggest_categorical('enable_midline_exit', [True, False])
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
            # ハイパーチャネルパラメータ
            'smoother_type': params.get('smoother_type', 'frama'),
            'smoother_period': int(params.get('smoother_period', 14)),
            'smoother_src_type': params.get('smoother_src_type', 'oc2'),
            'atr_period': float(params.get('atr_period', 14.0)),
            'atr_tr_method': params.get('atr_tr_method', 'str'),
            'atr_smoother_type': params.get('atr_smoother_type', 'laguerre'),
            'adaptation_type': params.get('adaptation_type', 'hyper_er'),
            'max_multiplier': float(params.get('max_multiplier', 6.0)),
            'min_multiplier': float(params.get('min_multiplier', 0.5)),
            # ブレイクアウトパラメータ
            'band_lookback': int(params.get('band_lookback', 1)),
            # フィルター設定
            'filter_type': FilterType(params['filter_type']),
            # エグジット設定
            'enable_midline_exit': bool(params.get('enable_midline_exit', True))
        }
        
        return strategy_params
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """ストラテジー情報を取得"""
        filter_type = self._parameters.get('filter_type', FilterType.NONE)
        filter_name = filter_type.value if isinstance(filter_type, FilterType) else str(filter_type)
        
        return {
            'name': 'HyperChannel Strategy',
            'description': f'Advanced Hyper Channel with {filter_name} Filter Integration',
            'parameters': self._parameters.copy(),
            'features': [
                'Unified Smoother-based adaptive midline control',
                'X_ATR volatility-based dynamic band width',
                'HyperER/HyperADX dynamic adaptation multiplier',
                'Channel breakout entry signals',
                'Configurable midline regression exit',
                f'Advanced {filter_name} filtering',
                'Kalman and Roofing filter integration',
                'Optimized with Numba JIT compilation',
                'High-precision trend and volatility detection'
            ],
            'filter_capabilities': {
                'hyper_er': 'HyperER efficiency ratio-based trend filtering',
                'hyper_trend_index': 'HyperTrendIndex advanced trend detection',
                'hyper_adx': 'HyperADX directional movement filtering',
                'consensus': '3-filter consensus (2 out of 3 agreement)',
                'none': 'Pure HyperChannel signals without filtering'
            },
            'adaptive_features': {
                'unified_smoother_midline': 'Multi-algorithm adaptive midline',
                'x_atr_band_width': 'Volatility-based dynamic band width',
                'dynamic_multiplier_control': 'HyperER/HyperADX-based multiplier adjustment',
                'breakout_signal_generation': 'Channel breakout detection with lookback',
                'midline_regression_exit': 'Optional midline-based exit strategy'
            }
        }
    
    def reset(self) -> None:
        """ストラテジーの状態をリセット"""
        super().reset()
        if hasattr(self.signal_generator, 'reset'):
            self.signal_generator.reset()