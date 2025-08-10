from typing import Dict, Any
import pandas as pd
import numpy as np
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import DonchianFRAMASignalGenerator, FilterType


class DonchianFRAMAStrategy(BaseStrategy):
    """ドンチャンFRAMAストラテジー
    
    ドンチャンチャネルの中間線とFRAMAの位置関係でエントリーし、
    複数のトレンドフィルターでシグナルを調整する戦略
    """
    
    def __init__(
        self,
        # ドンチャンFRAMAパラメータ
        donchian_period: int = 200,
        frama_period: int = 16,
        frama_fc: int = 2,
        frama_sc: int = 198,
        signal_mode: str = 'position',  # 'position' または 'crossover'
        
        # HyperER動的適応パラメータ
        enable_hyper_er_adaptation: bool = True,  # HyperER動的適応を有効にするか
        hyper_er_period: int = 14,                 # HyperER計算期間
        hyper_er_midline_period: int = 100,        # HyperERミッドライン期間
        
        # FRAMA HyperER動的適応パラメータ
        frama_fc_min: float = 1.0,                 # FRAMA FC最小値（ER高い時）
        frama_fc_max: float = 13.0,                # FRAMA FC最大値（ER低い時）
        frama_sc_min: float = 60.0,                # FRAMA SC最小値（ER高い時）
        frama_sc_max: float = 250.0,               # FRAMA SC最大値（ER低い時）
        
        # ドンチャン HyperER動的適応パラメータ
        donchian_period_min: float = 55.0,         # ドンチャン最小期間（ER高い時）
        donchian_period_max: float = 250.0,        # ドンチャン最大期間（ER低い時）
        
        # フィルタータイプ
        filter_type: str = 'consensus',  # 'none', 'hyper_er', 'hyper_trend_index', 'hyper_adx', 'consensus'
        
        # HyperTrendIndexパラメータ  
        hyper_trend_index_period: int = 14,
        hyper_trend_index_midline_period: int = 100,
        
        # HyperADXパラメータ
        hyper_adx_period: int = 14,
        hyper_adx_midline_period: int = 100
    ):
        super().__init__(f"DonchianFRAMA_{signal_mode}_{filter_type}")
        
        # パラメータ設定
        self._parameters = {
            'donchian_period': donchian_period,
            'frama_period': frama_period,
            'frama_fc': frama_fc,
            'frama_sc': frama_sc,
            'signal_mode': signal_mode,
            'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
            'hyper_er_period': hyper_er_period,
            'hyper_er_midline_period': hyper_er_midline_period,
            'frama_fc_min': frama_fc_min,
            'frama_fc_max': frama_fc_max,
            'frama_sc_min': frama_sc_min,
            'frama_sc_max': frama_sc_max,
            'donchian_period_min': donchian_period_min,
            'donchian_period_max': donchian_period_max,
            'filter_type': filter_type,
            'hyper_trend_index_period': hyper_trend_index_period,
            'hyper_trend_index_midline_period': hyper_trend_index_midline_period,
            'hyper_adx_period': hyper_adx_period,
            'hyper_adx_midline_period': hyper_adx_midline_period
        }
        
        self.filter_type = FilterType(filter_type)
        
        # シグナルジェネレーター用のパラメータ構築
        signal_params = {
            'filter_type': filter_type,
            'entry': {
                'donchian_period': donchian_period,
                'frama_period': frama_period,
                'frama_fc': frama_fc,
                'frama_sc': frama_sc,
                'signal_mode': signal_mode,
                # HyperER動的適応パラメータを追加
                'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
                'hyper_er_period': hyper_er_period,
                'hyper_er_midline_period': hyper_er_midline_period,
                'frama_fc_min': frama_fc_min,
                'frama_fc_max': frama_fc_max,
                'frama_sc_min': frama_sc_min,
                'frama_sc_max': frama_sc_max,
                'donchian_period_min': donchian_period_min,
                'donchian_period_max': donchian_period_max
            },
            'hyper_er': {
                'period': hyper_er_period,
                'midline_period': hyper_er_midline_period
            },
            'hyper_trend_index': {
                'period': hyper_trend_index_period,
                'midline_period': hyper_trend_index_midline_period
            },
            'hyper_adx': {
                'period': hyper_adx_period,
                'midline_period': hyper_adx_midline_period
            }
        }
        
        # シグナルジェネレーター
        self.signal_generator = DonchianFRAMASignalGenerator(signal_params)
    
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
    
    # execute_trade メソッドは削除 - シンプルなシグナルベーストレードのため不要
    
    # テイクプロフィット・ストップロス関連メソッドは削除 - シグナル通りにトレードするため不要
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """戦略情報取得"""
        return {
            'name': 'DonchianFRAMA Strategy',
            'description': 'Donchian Midline and FRAMA position-based trading strategy with trend filtering',
            'parameters': self._parameters.copy(),
            'filter_type': self.filter_type.value,
            'features': [
                'Donchian channel midline calculation',
                'FRAMA (Fractal Adaptive Moving Average)',
                'Position relationship signals (FRAMA vs Midline)',
                'Multiple trend filter options',
                'Consensus filtering (2 out of 3 agreement)',
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
            'donchian_period': trial.suggest_int('donchian_period', 50, 300),
            'frama_period': trial.suggest_int('frama_period', 8, 32, step=2),  # 偶数のみ
            'frama_fc': trial.suggest_int('frama_fc', 1, 5),
            'frama_sc': trial.suggest_int('frama_sc', 50, 300),
            'signal_mode': trial.suggest_categorical('signal_mode', ['position', 'crossover']),
            
            # HyperER動的適応パラメータ
            'enable_hyper_er_adaptation': enable_hyper_er_adaptation,
            'hyper_er_period': trial.suggest_int('hyper_er_period', 10, 30),
            'hyper_er_midline_period': trial.suggest_int('hyper_er_midline_period', 50, 200),
            
            # フィルター設定
            'filter_type': 'consensus',
            
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
                
                # ドンチャン HyperER動的適応パラメータ
                'donchian_period_min': trial.suggest_float('donchian_period_min', 20.0, 100.0),
                'donchian_period_max': trial.suggest_float('donchian_period_max', 150.0, 350.0)
            })
        else:
            # HyperER動的適応が無効な場合のデフォルト値
            params.update({
                'frama_fc_min': 1.0,
                'frama_fc_max': 13.0,
                'frama_sc_min': 60.0,
                'frama_sc_max': 250.0,
                'donchian_period_min': 55.0,
                'donchian_period_max': 250.0
            })
        
        return params