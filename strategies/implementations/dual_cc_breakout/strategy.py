#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from ..dual_cc_breakout.signal_generator import DualCCBreakoutSignalGenerator


class DualCCBreakoutStrategy(BaseStrategy):
    """
    DualCCBreakoutStrategy
    
    このストラテジーはDualCCBreakoutSignalGeneratorを使用して、
    2つの異なる幅のCチャネルを使用したブレイクアウト戦略を実装します。
    
    特徴:
    - 広いバンド幅を持つCチャネルをエントリーに使用
    - 狭いバンド幅を持つCチャネルをエグジットに使用
    
    エントリー条件:
    - ロングエントリー: エントリー用CCBreakoutSignalのシグナルが1のとき
    - ショートエントリー: エントリー用CCBreakoutSignalのシグナルが-1のとき
    
    エグジット条件:
    - ロングエグジット: エグジット用CCBreakoutSignalのシグナルが-1のとき
    - ショートエグジット: エグジット用CCBreakoutSignalのシグナルが1のとき
    """
    
    def __init__(
        self,
        # Cチャネルの基本パラメータ（共通）
        detector_type: str = 'dudi_e',
        cer_detector_type: str = 'phac_e',  # CER用の検出器タイプ
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.618,
        smoother_type: str = 'alma',
        src_type: str = 'hlc3',
        band_lookback: int = 1,
        
        # エントリー用の動的乗数の範囲パラメータ（広め）
        entry_max_max_multiplier: float = 10.0,    # 最大乗数の最大値
        entry_min_max_multiplier: float = 6.0,    # 最大乗数の最小値
        entry_max_min_multiplier: float = 4.0,    # 最小乗数の最大値
        entry_min_min_multiplier: float = 2.0,    # 最小乗数の最小値
        
        # エグジット用の動的乗数の範囲パラメータ（狭め）
        exit_max_max_multiplier: float = 2.5,     # 最大乗数の最大値
        exit_min_max_multiplier: float = 1.5,     # 最大乗数の最小値
        exit_max_min_multiplier: float = 1.0,     # 最小乗数の最大値
        exit_min_min_multiplier: float = 0.3,     # 最小乗数の最小値
        
        # CMA用パラメータ
        cma_detector_type: str = 'dudi_e',
        cma_cycle_part: float = 0.618,
        cma_lp_period: int = 5,
        cma_hp_period: int = 55,
        cma_max_cycle: int = 89,
        cma_min_cycle: int = 5,
        cma_max_output: int = 89,
        cma_min_output: int = 13,
        cma_fast_period: int = 3,
        cma_slow_period: int = 34,
        cma_src_type: str = 'hlc3',
        
        # CATR用パラメータ
        catr_detector_type: str = 'phac_e',
        catr_cycle_part: float = 0.618,
        catr_lp_period: int = 13,
        catr_hp_period: int = 89,
        catr_max_cycle: int = 62,
        catr_min_cycle: int = 5,
        catr_max_output: int = 55,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'hyper',
        
        strategy_name: str = "DualCCBreakoutStrategy"
    ):
        """
        初期化メソッド
        
        Args:
            detector_type: サイクル検出器のタイプ
            cer_detector_type: CER用の検出器タイプ
            lp_period: 低域通過フィルタの期間
            hp_period: 高域通過フィルタの期間
            cycle_part: サイクル部分の比率
            smoother_type: スムーザーのタイプ
            src_type: 元データのタイプ
            band_lookback: バンド計算のルックバック期間
            
            entry_max_max_multiplier: エントリー用最大乗数の最大値
            entry_min_max_multiplier: エントリー用最大乗数の最小値
            entry_max_min_multiplier: エントリー用最小乗数の最大値
            entry_min_min_multiplier: エントリー用最小乗数の最小値
            
            exit_max_max_multiplier: エグジット用最大乗数の最大値
            exit_min_max_multiplier: エグジット用最大乗数の最小値
            exit_max_min_multiplier: エグジット用最小乗数の最大値
            exit_min_min_multiplier: エグジット用最小乗数の最小値
            
            cma_detector_type: CMA用の検出器タイプ
            cma_cycle_part: CMA用のサイクル部分の比率
            cma_lp_period: CMA用の低域通過フィルタの期間
            cma_hp_period: CMA用の高域通過フィルタの期間
            cma_max_cycle: CMA用の最大サイクル
            cma_min_cycle: CMA用の最小サイクル
            cma_max_output: CMA用の最大出力
            cma_min_output: CMA用の最小出力
            cma_fast_period: CMA用の短期期間
            cma_slow_period: CMA用の長期期間
            cma_src_type: CMA用の元データのタイプ
            
            catr_detector_type: CATR用の検出器タイプ
            catr_cycle_part: CATR用のサイクル部分の比率
            catr_lp_period: CATR用の低域通過フィルタの期間
            catr_hp_period: CATR用の高域通過フィルタの期間
            catr_max_cycle: CATR用の最大サイクル
            catr_min_cycle: CATR用の最小サイクル
            catr_max_output: CATR用の最大出力
            catr_min_output: CATR用の最小出力
            catr_smoother_type: CATR用のスムーザーのタイプ
            
            strategy_name: 戦略の名前
        """
        super().__init__(strategy_name)
        
        # シグナル生成器の初期化
        self.signal_generator = DualCCBreakoutSignalGenerator(
            # 基本パラメータ
            detector_type=detector_type,
            cer_detector_type=cer_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            smoother_type=smoother_type,
            src_type=src_type,
            band_lookback=band_lookback,
            
            # エントリー用動的乗数
            entry_max_max_multiplier=entry_max_max_multiplier,
            entry_min_max_multiplier=entry_min_max_multiplier,
            entry_max_min_multiplier=entry_max_min_multiplier,
            entry_min_min_multiplier=entry_min_min_multiplier,
            
            # エグジット用動的乗数
            exit_max_max_multiplier=exit_max_max_multiplier,
            exit_min_max_multiplier=exit_min_max_multiplier,
            exit_max_min_multiplier=exit_max_min_multiplier,
            exit_min_min_multiplier=exit_min_min_multiplier,
            
            # CMA用パラメータ
            cma_detector_type=cma_detector_type,
            cma_cycle_part=cma_cycle_part,
            cma_lp_period=cma_lp_period,
            cma_hp_period=cma_hp_period,
            cma_max_cycle=cma_max_cycle,
            cma_min_cycle=cma_min_cycle,
            cma_max_output=cma_max_output,
            cma_min_output=cma_min_output,
            cma_fast_period=cma_fast_period,
            cma_slow_period=cma_slow_period,
            cma_src_type=cma_src_type,
            
            # CATR用パラメータ
            catr_detector_type=catr_detector_type,
            catr_cycle_part=catr_cycle_part,
            catr_lp_period=catr_lp_period,
            catr_hp_period=catr_hp_period,
            catr_max_cycle=catr_max_cycle,
            catr_min_cycle=catr_min_cycle,
            catr_max_output=catr_max_output,
            catr_min_output=catr_min_output,
            catr_smoother_type=catr_smoother_type
        )
        
        # パラメータの設定
        self._parameters.update({
            # 基本パラメータ
            'detector_type': detector_type,
            'cer_detector_type': cer_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            
            # エントリー用動的乗数
            'entry_max_max_multiplier': entry_max_max_multiplier,
            'entry_min_max_multiplier': entry_min_max_multiplier,
            'entry_max_min_multiplier': entry_max_min_multiplier,
            'entry_min_min_multiplier': entry_min_min_multiplier,
            
            # エグジット用動的乗数
            'exit_max_max_multiplier': exit_max_max_multiplier,
            'exit_min_max_multiplier': exit_min_max_multiplier,
            'exit_max_min_multiplier': exit_max_min_multiplier,
            'exit_min_min_multiplier': exit_min_min_multiplier,
            
            # CMA用パラメータ
            'cma_detector_type': cma_detector_type,
            'cma_cycle_part': cma_cycle_part,
            'cma_lp_period': cma_lp_period,
            'cma_hp_period': cma_hp_period,
            'cma_max_cycle': cma_max_cycle,
            'cma_min_cycle': cma_min_cycle,
            'cma_max_output': cma_max_output,
            'cma_min_output': cma_min_output,
            'cma_fast_period': cma_fast_period,
            'cma_slow_period': cma_slow_period,
            'cma_src_type': cma_src_type,
            
            # CATR用パラメータ
            'catr_detector_type': catr_detector_type,
            'catr_cycle_part': catr_cycle_part,
            'catr_lp_period': catr_lp_period,
            'catr_hp_period': catr_hp_period,
            'catr_max_cycle': catr_max_cycle,
            'catr_min_cycle': catr_min_cycle,
            'catr_max_output': catr_max_output,
            'catr_min_output': catr_min_output,
            'catr_smoother_type': catr_smoother_type
        })
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray], index: int = -1) -> np.ndarray:
        """
        エントリーシグナルの生成
        
        Args:
            data: 価格データ（DataFrameまたはnumpy配列）
            index: 信号を生成する特定のインデックス（デフォルトは最後のデータポイント）
            
        Returns:
            np.ndarray: エントリーシグナル配列 (1: ロング、-1: ショート、0: シグナルなし)
        """
        try:
            # エントリーシグナルの取得
            signals = self.signal_generator.get_entry_signals(data)
            return signals
        except Exception as e:
            import traceback
            print(f"エントリーシグナル生成中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.zeros(len(data), dtype=np.int8)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルの生成
        
        Args:
            data: 価格データ（DataFrameまたはnumpy配列）
            position: 現在のポジション（1: ロング、-1: ショート）
            index: 信号を生成する特定のインデックス（デフォルトは最後のデータポイント）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        try:
            # エグジットシグナルの取得
            return self.signal_generator.get_exit_signals(data, position, index)
        except Exception as e:
            import traceback
            print(f"エグジットシグナル生成中にエラー: {str(e)}\n{traceback.format_exc()}")
            return False
    
    def get_entry_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, Any]:
        """
        エントリー用Cチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Dict: バンド値の辞書
                {
                    'middle': np.ndarray,  # 中心線
                    'upper': np.ndarray,   # 上限バンド
                    'lower': np.ndarray    # 下限バンド
                }
        """
        try:
            middle, upper, lower = self.signal_generator.get_entry_band_values(data)
            
            return {
                'middle': middle,
                'upper': upper,
                'lower': lower
            }
        except Exception as e:
            import traceback
            print(f"エントリーバンド値取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            empty = np.array([])
            return {
                'middle': empty,
                'upper': empty,
                'lower': empty
            }
    
    def get_exit_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict[str, Any]:
        """
        エグジット用Cチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Dict: バンド値の辞書
                {
                    'middle': np.ndarray,  # 中心線
                    'upper': np.ndarray,   # 上限バンド
                    'lower': np.ndarray    # 下限バンド
                }
        """
        try:
            middle, upper, lower = self.signal_generator.get_exit_band_values(data)
            
            return {
                'middle': middle,
                'upper': upper,
                'lower': lower
            }
        except Exception as e:
            import traceback
            print(f"エグジットバンド値取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            empty = np.array([])
            return {
                'middle': empty,
                'upper': empty,
                'lower': empty
            }
    
    def get_entry_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        エントリー用サイクル効率比（CER）の値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        try:
            return self.signal_generator.get_entry_efficiency_ratio(data)
        except Exception as e:
            import traceback
            print(f"エントリー効率比取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def get_exit_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        エグジット用サイクル効率比（CER）の値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        try:
            return self.signal_generator.get_exit_efficiency_ratio(data)
        except Exception as e:
            import traceback
            print(f"エグジット効率比取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def get_entry_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        エントリー用動的乗数の値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        try:
            return self.signal_generator.get_entry_dynamic_multiplier(data)
        except Exception as e:
            import traceback
            print(f"エントリー動的乗数取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def get_exit_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        エグジット用動的乗数の値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        try:
            return self.signal_generator.get_exit_dynamic_multiplier(data)
        except Exception as e:
            import traceback
            print(f"エグジット動的乗数取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def get_c_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CATR値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: CATR値
        """
        try:
            return self.signal_generator.get_c_atr(data)
        except Exception as e:
            import traceback
            print(f"CATR取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
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
            # エントリー用動的乗数のパラメータ
            'entry_max_max_multiplier': trial.suggest_float('entry_max_max_multiplier', 6.0, 10.0, step=0.5),
            'entry_min_max_multiplier': trial.suggest_float('entry_min_max_multiplier', 4.0, 8.0, step=0.5),
            'entry_max_min_multiplier': trial.suggest_float('entry_max_min_multiplier', 1.0, 2.0, step=0.1),
            'entry_min_min_multiplier': trial.suggest_float('entry_min_min_multiplier', 0.3, 1.0, step=0.1),
            
            # エグジット用動的乗数のパラメータ
            'exit_max_max_multiplier': trial.suggest_float('exit_max_max_multiplier', 3.0, 5.0, step=0.5),
            'exit_min_max_multiplier': trial.suggest_float('exit_min_max_multiplier', 2.0, 4.0, step=0.5),
            'exit_max_min_multiplier': trial.suggest_float('exit_max_min_multiplier', 0.8, 1.5, step=0.1),
            'exit_min_min_multiplier': trial.suggest_float('exit_min_min_multiplier', 0.1, 0.5, step=0.1),
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
            # エントリー用動的乗数のパラメータ
            'entry_max_max_multiplier': float(params['entry_max_max_multiplier']),
            'entry_min_max_multiplier': float(params['entry_min_max_multiplier']),
            'entry_max_min_multiplier': float(params['entry_max_min_multiplier']),
            'entry_min_min_multiplier': float(params['entry_min_min_multiplier']),
            
            # エグジット用動的乗数のパラメータ
            'exit_max_max_multiplier': float(params['exit_max_max_multiplier']),
            'exit_min_max_multiplier': float(params['exit_min_max_multiplier']),
            'exit_max_min_multiplier': float(params['exit_max_min_multiplier']),
            'exit_min_min_multiplier': float(params['exit_min_min_multiplier']),
        }
        return strategy_params 