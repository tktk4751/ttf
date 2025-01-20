from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import SupertrendChopMfiSignalGenerator


class SupertrendChopMfiStrategy(BaseStrategy):
    """スーパートレンド、チョピネス、MFIを組み合わせた戦略"""
    
    def __init__(
        self,
        supertrend_params: Dict[str, Any] = None,
        chop_params: Dict[str, Any] = None,
        mfi_params: Dict[str, Any] = None
    ):
        """
        コンストラクタ
        
        Args:
            supertrend_params: スーパートレンドのパラメータ
            chop_params: チョピネスインデックスのパラメータ
            mfi_params: MFIのパラメータ
        """
        super().__init__("SupertrendChopMfi")
        
        # デフォルトパラメータ
        self._parameters = {
            'supertrend': supertrend_params or {
                'period': 10,
                'multiplier': 3.0
            },
            'chop': chop_params or {
                'period': 14,
                'solid': {
                    'chop_solid': 50
                }
            },
            'mfi': mfi_params or {
                'period': 14,
                'solid': {
                    'mfi_long_exit_solid': 90,
                    'mfi_short_exit_solid': 10
                }
            }
        }
        
        # シグナル生成器の初期化
        self.signal_generator = SupertrendChopMfiSignalGenerator(
            supertrend_params=self._parameters['supertrend'],
            chop_params=self._parameters['chop'],
            mfi_params=self._parameters['mfi']
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルを生成する"""
        return self.signal_generator.get_entry_signals(data)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナルを生成する"""
        return self.signal_generator.get_exit_signals(data, position, index)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """最適化パラメータを生成する"""
        params = {
            'supertrend_period': trial.suggest_int('supertrend_period', 3, 100, step=1),
            'supertrend_multiplier': trial.suggest_float('supertrend_multiplier', 1.5, 7.0, step=0.5),
            'chop_period': trial.suggest_int('chop_period', 3, 100, step=1),
            'chop_solid': 50,  # 固定値
            'mfi_period': trial.suggest_int('mfi_period', 3, 21, step=1),
            'mfi_long_exit_solid': 90,
            'mfi_short_exit_solid': 10
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """最適化パラメータを戦略パラメータに変換する"""
        return {
            'supertrend_params': {
                'period': params['supertrend_period'],
                'multiplier': params['supertrend_multiplier']
            },
            'chop_params': {
                'period': params['chop_period'],
                'solid': {
                    'chop_solid': 50  # 固定値を直接指定
                }
            },
            'mfi_params': {
                'period': params['mfi_period'],
                'solid': {
                    'mfi_long_exit_solid': 90,  # 固定値を直接指定
                    'mfi_short_exit_solid': 10  # 固定値を直接指定
                }
            }
        } 