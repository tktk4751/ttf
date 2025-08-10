#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import LaguerreRSITrendReversalSignalGenerator


class LaguerreRSITrendReversalStrategy(BaseStrategy):
    """
    Laguerre RSI トレンドリバーサル ストラテジー
    
    特徴:
    - Laguerre RSI（ラゲール変換ベースRSI）を使用したトレンドリバーサル戦略
    - John Ehlers's Laguerre transform filterによる高感度なRSI指標
    - パインスクリプト仕様準拠のリバーサルシグナル生成
    - 従来のRSIより価格変動に敏感で、短期間のデータでも効果的
    
    エントリー条件:
    - ロング: RSI < buy_band (0.2) - 売られすぎからの反転狙い
    - ショート: RSI > sell_band (0.8) - 買われすぎからの反転狙い
    - position_mode=True: 閾値内では前回ポジション維持
    - position_mode=False: クロスオーバーでのみシグナル発生
    - mean_reversion_mode=True: 平均回帰モード（傾向変化を考慮）
    
    エグジット条件:
    - ロング: シグナル=-1（RSI > sell_band）
    - ショート: シグナル=1（RSI < buy_band）
    
    革新的な優位性:
    - ラゲール変換による平滑化でノイズ除去
    - UltimateSmoother統合によるさらなる平滑化
    - ルーフィングフィルターオプションで高周波ノイズ除去
    - 平均回帰モードで傾向変化の検出
    - Numba JIT最適化による高速処理
    """
    
    def __init__(
        self,
        # Laguerre RSIパラメータ
        gamma: float = 0.5,                      # ガンマパラメータ
        src_type: str = 'oc2',                 # ソースタイプ
        # シグナル閾値（リバーサル用）
        buy_band: float = 0.2,                   # 買い閾値（売られすぎ水準）
        sell_band: float = 0.8,                  # 売り閾値（買われすぎ水準）
        # ルーフィングフィルターパラメータ（オプション）
        use_roofing_filter: bool = True,        # ルーフィングフィルターを使用するか
        roofing_hp_cutoff: float = 48.0,         # ルーフィングフィルターのHighPassカットオフ
        roofing_ss_band_edge: float = 10.0,      # ルーフィングフィルターのSuperSmootherバンドエッジ
        # シグナル設定
        position_mode: bool = True,              # ポジション維持モード(True)またはクロスオーバーモード(False)
        mean_reversion_mode: bool = False        # 平均回帰モード（傾向変化を考慮）
    ):
        """
        初期化
        
        Args:
            gamma: ガンマパラメータ（デフォルト: 0.5、パインスクリプト仕様）
            src_type: ソースタイプ（デフォルト: 'close'）
            buy_band: 買い閾値（リバーサル用、デフォルト: 0.2）
            sell_band: 売り閾値（リバーサル用、デフォルト: 0.8）
            use_roofing_filter: ルーフィングフィルター使用（デフォルト: False）
            roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ（デフォルト: 48.0）
            roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ（デフォルト: 10.0）
            position_mode: ポジション維持モード(True)またはクロスオーバーモード(False)
            mean_reversion_mode: 平均回帰モード（傾向変化を考慮、デフォルト: False）
        """
        if mean_reversion_mode:
            signal_type = "MeanReversion"
        elif position_mode:
            signal_type = "Position"
        else:
            signal_type = "Crossover"
            
        roofing_str = f"_roofing(hp={roofing_hp_cutoff}, ss={roofing_ss_band_edge})" if use_roofing_filter else ""
        
        super().__init__(f"LaguerreRSI_TrendReversal_{signal_type}(gamma={gamma}, {src_type}{roofing_str})")
        
        # パラメータの設定
        self._parameters = {
            'gamma': gamma,
            'src_type': src_type,
            'buy_band': buy_band,
            'sell_band': sell_band,
            'use_roofing_filter': use_roofing_filter,
            'roofing_hp_cutoff': roofing_hp_cutoff,
            'roofing_ss_band_edge': roofing_ss_band_edge,
            'position_mode': position_mode,
            'mean_reversion_mode': mean_reversion_mode
        }
        
        # シグナル生成器の初期化
        self.signal_generator = LaguerreRSITrendReversalSignalGenerator(
            gamma=gamma,
            src_type=src_type,
            buy_band=buy_band,
            sell_band=sell_band,
            use_roofing_filter=use_roofing_filter,
            roofing_hp_cutoff=roofing_hp_cutoff,
            roofing_ss_band_edge=roofing_ss_band_edge,
            position_mode=position_mode,
            mean_reversion_mode=mean_reversion_mode
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
    
    def get_lrsi_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """Laguerre RSI値を取得"""
        try:
            return self.signal_generator.get_lrsi_values(data)
        except Exception as e:
            self.logger.error(f"Laguerre RSI値取得中にエラー: {str(e)}")
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
    
    def get_lrsi_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Laguerre RSIシグナル取得"""
        try:
            return self.signal_generator.get_lrsi_signals(data)
        except Exception as e:
            self.logger.error(f"Laguerre RSIシグナル取得中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_l0_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """L0値を取得"""
        try:
            return self.signal_generator.get_l0_values(data)
        except Exception as e:
            self.logger.error(f"L0値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_l1_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """L1値を取得"""
        try:
            return self.signal_generator.get_l1_values(data)
        except Exception as e:
            self.logger.error(f"L1値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_l2_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """L2値を取得"""
        try:
            return self.signal_generator.get_l2_values(data)
        except Exception as e:
            self.logger.error(f"L2値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_l3_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """L3値を取得"""
        try:
            return self.signal_generator.get_l3_values(data)
        except Exception as e:
            self.logger.error(f"L3値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_cu_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """CU値（上昇累積）を取得"""
        try:
            return self.signal_generator.get_cu_values(data)
        except Exception as e:
            self.logger.error(f"CU値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_cd_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """CD値（下降累積）を取得"""
        try:
            return self.signal_generator.get_cd_values(data)
        except Exception as e:
            self.logger.error(f"CD値取得中にエラー: {str(e)}")
            return np.array([])
    
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
        params = {
            # Laguerre RSIパラメータ
            'gamma': trial.suggest_float('gamma', 0.1, 0.9, step=0.05),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4', 'oc2']),
            
            # シグナル閾値（リバーサル用）
            'buy_band': trial.suggest_float('buy_band', 0.05, 0.4, step=0.05),
            'sell_band': trial.suggest_float('sell_band', 0.6, 0.95, step=0.05),
            
            # ルーフィングフィルター
            'use_roofing_filter': trial.suggest_categorical('use_roofing_filter', [True, False]),
            
            # シグナル設定
            'position_mode': trial.suggest_categorical('position_mode', [True, False]),
            'mean_reversion_mode': trial.suggest_categorical('mean_reversion_mode', [True, False])
        }
        
        # ルーフィングフィルター使用時のパラメータ
        if params['use_roofing_filter']:
            params.update({
                'roofing_hp_cutoff': trial.suggest_float('roofing_hp_cutoff', 20.0, 100.0, step=5.0),
                'roofing_ss_band_edge': trial.suggest_float('roofing_ss_band_edge', 5.0, 20.0, step=2.5)
            })
        
        # buy_band < sell_bandの制約を確保（リバーサル用）
        if params['buy_band'] >= params['sell_band']:
            params['buy_band'] = params['sell_band'] - 0.1
        
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
            'gamma': float(params['gamma']),
            'src_type': params['src_type'],
            'buy_band': float(params['buy_band']),
            'sell_band': float(params['sell_band']),
            'use_roofing_filter': bool(params['use_roofing_filter']),
            'position_mode': bool(params['position_mode']),
            'mean_reversion_mode': bool(params['mean_reversion_mode'])
        }
        
        # ルーフィングフィルターパラメータの追加
        if params.get('use_roofing_filter', False):
            strategy_params.update({
                'roofing_hp_cutoff': float(params.get('roofing_hp_cutoff', 48.0)),
                'roofing_ss_band_edge': float(params.get('roofing_ss_band_edge', 10.0))
            })
        
        return strategy_params
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """ストラテジー情報を取得"""
        position_mode = self._parameters.get('position_mode', True)
        mean_reversion_mode = self._parameters.get('mean_reversion_mode', False)
        roofing_enabled = self._parameters.get('use_roofing_filter', False)
        
        if mean_reversion_mode:
            signal_mode = "Mean Reversion"
        elif position_mode:
            signal_mode = "Position Maintenance"
        else:
            signal_mode = "Crossover"
        
        return {
            'name': 'Laguerre RSI Trend Reversal Strategy',
            'description': f'Laguerre-transformed RSI based trend reversal with {signal_mode} signals',
            'parameters': self._parameters.copy(),
            'features': [
                'Laguerre transform RSI for enhanced sensitivity',
                'John Ehlers\'s Laguerre filter integration',
                'UltimateSmoother for additional smoothing',
                f'{signal_mode} signal generation',
                'Mean reversion trend analysis' if mean_reversion_mode else 'Standard reversal signals',
                'Roofing filter for high-frequency noise reduction' if roofing_enabled else 'Standard price processing',
                'Contrarian/reversal strategy (anti-momentum)',
                'Optimized with Numba JIT compilation',
                'PineScript specification compliance'
            ],
            'signal_conditions': {
                'long_entry': f'RSI < {self._parameters.get("buy_band", 0.2)} (oversold reversal)',
                'short_entry': f'RSI > {self._parameters.get("sell_band", 0.8)} (overbought reversal)',
                'long_exit': 'Signal changes to -1 (RSI rises above sell_band)',
                'short_exit': 'Signal changes to 1 (RSI drops below buy_band)',
                'position_maintenance': position_mode,
                'mean_reversion_mode': mean_reversion_mode
            },
            'advantages': [
                'Higher sensitivity than traditional RSI',
                'Effective with shorter time periods',
                'Reduced lag through Laguerre filtering',
                'Contrarian approach for range-bound markets',
                'Mean reversion detection capabilities',
                'Built-in noise filtering capabilities'
            ]
        }
    
    def reset(self) -> None:
        """ストラテジーの状態をリセット"""
        super().reset()
        if hasattr(self.signal_generator, 'reset'):
            self.signal_generator.reset()