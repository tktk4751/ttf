#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

# --- 依存関係のインポート ---
try:
    from strategies.base.strategy import BaseStrategy
    from .signal_generator import XCSimpleSignalGenerator # XChannel用ジェネレータ
except ImportError:
    # フォールバック
    print("Warning: Could not import from relative path. Assuming base classes/functions are available.")
    class BaseStrategy:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class XCSimpleSignalGenerator: # Dummy
        def __init__(self, **kwargs): pass
        def get_entry_signals(self, data): return np.zeros(len(data))
        def get_exit_signals(self, data, pos, idx): return False


class XCSimpleStrategy(BaseStrategy):
    """
    Xチャネル戦略（シンプル版）

    特徴:
    - XMAとCATRを使用
    - ATR乗数はトリガー値（CER or XTrendIndex）に基づいて非線形に変化

    エントリー条件:
    - ロング: Xチャネルの買いシグナル
    - ショート: Xチャネルの売りシグナル

    エグジット条件:
    - ロング: Xチャネルの売りシグナル
    - ショート: Xチャネルの買いシグナル
    """

    def __init__(
        self,
        # --- XChannel Parameters (passed down) ---
        # XMA Params
        xma_ma_type: str = 'hma',
        xma_src_type: str = 'hlc3',
        xma_use_kalman_filter: bool = False,
        kalman_measurement_noise: float = 1.0,
        kalman_process_noise: float = 0.01,
        kalman_n_states: int = 5,
        xma_alma_offset: float = 0.85,
        xma_alma_sigma: float = 6.0,
        xma_trigger_type: str = 'cer',
        # --- XMA's CER params (全パラメータに対応) ---
        xma_cer_detector_type: str = 'phac_e',
        xma_cer_lp_period: int = 5,
        xma_cer_hp_period: int = 120,
        xma_cer_cycle_part: float = 0.3,
        xma_cer_max_cycle: int = 233,  # 追加: CER最大サイクル
        xma_cer_min_cycle: int = 15,    # 追加: CER最小サイクル
        xma_cer_max_output: int = 89,  # 追加: CER最大出力
        xma_cer_min_output: int = 13,  # 追加: CER最小出力
        xma_cer_src_type: str = 'hlc3',  # 追加: CER用ソースタイプ
        xma_cer_use_kalman_filter: bool = False, # 追加: CER用カルマンフィルター
        # --- XMA's internal XTrend params ---
        xma_xt_dc_detector_type: str = 'phac_e',
        xma_xt_dc_cycle_part: float = 0.382,
        xma_xt_dc_max_cycle: int = 100,
        xma_xt_dc_min_cycle: int = 5,
        xma_xt_dc_max_output: int = 55,
        xma_xt_dc_min_output: int = 5,
        xma_xt_dc_src_type: str = 'hlc3',
        xma_xt_dc_lp_period: int = 5,
        xma_xt_dc_hp_period: int = 55,
        xma_xt_catr_smoother_type: str = 'alma',
        xma_xt_cer_detector_type: str = 'dudi_e',
        xma_xt_cer_lp_period: int = 5,
        xma_xt_cer_hp_period: int = 144,
        xma_xt_cer_cycle_part: float = 0.5,
        xma_xt_max_threshold: float = 0.75,
        xma_xt_min_threshold: float = 0.55,
        # CATR Params
        catr_detector_type: str = 'phac_e',
        catr_cycle_part: float = 0.5,
        catr_lp_period: int = 5,
        catr_hp_period: int = 55,
        catr_max_cycle: int = 55,
        catr_min_cycle: int = 5,
        catr_max_output: int = 55,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'alma',
        catr_src_type: str = 'hlc3',
        catr_use_kalman_filter: bool = False,
        # Multiplier Trigger Params
        multiplier_trigger_type: str = 'cer',
        mult_cer_detector_type: str = 'phac_e',
        mult_cer_lp_period: int = 5,
        mult_cer_hp_period: int = 144,
        mult_cer_cycle_part: float = 0.3,
        mult_cer_max_cycle: int = 120,
        mult_cer_min_cycle: int = 13,
        mult_cer_max_output: int = 89,
        mult_cer_min_output: int = 13,
        mult_cer_src_type: str = 'hlc3',
        mult_cer_use_kalman_filter: bool = False,
        mult_xt_dc_detector_type: str = 'phac_e',
        mult_xt_dc_cycle_part: float = 0.55,
        mult_xt_dc_max_cycle: int = 100,
        mult_xt_dc_min_cycle: int = 3,
        mult_xt_dc_max_output: int = 55,
        mult_xt_dc_min_output: int = 13,
        mult_xt_dc_src_type: str = 'hlc3',
        mult_xt_dc_lp_period: int = 5,
        mult_xt_dc_hp_period: int = 55,
        mult_xt_catr_smoother_type: str = 'alma',
        mult_xt_cer_detector_type: str = 'dudi',
        mult_xt_cer_lp_period: int = 5,
        mult_xt_cer_hp_period: int = 144,
        mult_xt_cer_cycle_part: float = 0.5,
        mult_xt_max_threshold: float = 0.75,
        mult_xt_min_threshold: float = 0.55,

        # --- Signal Specific Params ---
        use_close_confirmation: bool = True
    ):
        """
        初期化
        (パラメータ詳細はXChannelおよびXChannelBreakoutEntrySignalを参照)
        """
        super().__init__("XCSimple")

        # パラメータを辞書にまとめる (XChannelBreakoutEntrySignalが必要とする形式)
        x_channel_params = {
            # XMA Params
            'xma_ma_type': xma_ma_type,
            'xma_src_type': xma_src_type,
            'xma_use_kalman_filter': xma_use_kalman_filter,
            'kalman_measurement_noise': kalman_measurement_noise,
            'kalman_process_noise': kalman_process_noise,
            'kalman_n_states': kalman_n_states,
            'xma_alma_offset': xma_alma_offset,
            'xma_alma_sigma': xma_alma_sigma,
            'xma_trigger_type': xma_trigger_type,
            # XMA's CER params (全パラメータ対応)
            'xma_cer_detector_type': xma_cer_detector_type,
            'xma_cer_lp_period': xma_cer_lp_period,
            'xma_cer_hp_period': xma_cer_hp_period,
            'xma_cer_cycle_part': xma_cer_cycle_part,
            'xma_cer_max_cycle': xma_cer_max_cycle,
            'xma_cer_min_cycle': xma_cer_min_cycle,
            'xma_cer_max_output': xma_cer_max_output,
            'xma_cer_min_output': xma_cer_min_output,
            'xma_cer_src_type': xma_cer_src_type,
            'xma_cer_use_kalman_filter': xma_cer_use_kalman_filter,
            # XTrend params
            'xma_xt_dc_detector_type': xma_xt_dc_detector_type,
            'xma_xt_dc_cycle_part': xma_xt_dc_cycle_part,
            'xma_xt_dc_max_cycle': xma_xt_dc_max_cycle,
            'xma_xt_dc_min_cycle': xma_xt_dc_min_cycle,
            'xma_xt_dc_max_output': xma_xt_dc_max_output,
            'xma_xt_dc_min_output': xma_xt_dc_min_output,
            'xma_xt_dc_src_type': xma_xt_dc_src_type,
            'xma_xt_dc_lp_period': xma_xt_dc_lp_period,
            'xma_xt_dc_hp_period': xma_xt_dc_hp_period,
            'xma_xt_catr_smoother_type': xma_xt_catr_smoother_type,
            'xma_xt_cer_detector_type': xma_xt_cer_detector_type,
            'xma_xt_cer_lp_period': xma_xt_cer_lp_period,
            'xma_xt_cer_hp_period': xma_xt_cer_hp_period,
            'xma_xt_cer_cycle_part': xma_xt_cer_cycle_part,
            'xma_xt_max_threshold': xma_xt_max_threshold,
            'xma_xt_min_threshold': xma_xt_min_threshold,
            # CATR Params
            'catr_detector_type': catr_detector_type,
            'catr_cycle_part': catr_cycle_part,
            'catr_lp_period': catr_lp_period,
            'catr_hp_period': catr_hp_period,
            'catr_max_cycle': catr_max_cycle,
            'catr_min_cycle': catr_min_cycle,
            'catr_max_output': catr_max_output,
            'catr_min_output': catr_min_output,
            'catr_smoother_type': catr_smoother_type,
            'catr_src_type': catr_src_type,
            'catr_use_kalman_filter': catr_use_kalman_filter,
            # Multiplier Trigger Params
            'multiplier_trigger_type': multiplier_trigger_type,
            'mult_cer_detector_type': mult_cer_detector_type,
            'mult_cer_lp_period': mult_cer_lp_period,
            'mult_cer_hp_period': mult_cer_hp_period,
            'mult_cer_cycle_part': mult_cer_cycle_part,
            'mult_cer_max_cycle': mult_cer_max_cycle,
            'mult_cer_min_cycle': mult_cer_min_cycle,
            'mult_cer_max_output': mult_cer_max_output,
            'mult_cer_min_output': mult_cer_min_output,
            'mult_cer_src_type': mult_cer_src_type,
            'mult_cer_use_kalman_filter': mult_cer_use_kalman_filter,
            'mult_xt_dc_detector_type': mult_xt_dc_detector_type,
            'mult_xt_dc_cycle_part': mult_xt_dc_cycle_part,
            'mult_xt_dc_max_cycle': mult_xt_dc_max_cycle,
            'mult_xt_dc_min_cycle': mult_xt_dc_min_cycle,
            'mult_xt_dc_max_output': mult_xt_dc_max_output,
            'mult_xt_dc_min_output': mult_xt_dc_min_output,
            'mult_xt_dc_src_type': mult_xt_dc_src_type,
            'mult_xt_dc_lp_period': mult_xt_dc_lp_period,
            'mult_xt_dc_hp_period': mult_xt_dc_hp_period,
            'mult_xt_catr_smoother_type': mult_xt_catr_smoother_type,
            'mult_xt_cer_detector_type': mult_xt_cer_detector_type,
            'mult_xt_cer_lp_period': mult_xt_cer_lp_period,
            'mult_xt_cer_hp_period': mult_xt_cer_hp_period,
            'mult_xt_cer_cycle_part': mult_xt_cer_cycle_part,
            'mult_xt_max_threshold': mult_xt_max_threshold,
            'mult_xt_min_threshold': mult_xt_min_threshold,
        }

        # 全パラメータを戦略インスタンスに保存（最適化などで使用）
        self._parameters = {
            **x_channel_params, # XChannelパラメータを展開
            'use_close_confirmation': use_close_confirmation
        }

        # シグナル生成器の初期化
        self.signal_generator = XCSimpleSignalGenerator(
            x_channel_params=x_channel_params,
            use_close_confirmation=use_close_confirmation
        )

    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する

        Args:
            data: 価格データ

        Returns:
            np.ndarray: エントリーシグナル (1: 買い, -1: 売り, 0: なし)
        """
        try:
            return self.signal_generator.get_entry_signals(data)
        except Exception as e:
            self.logger.error(f"エントリーシグナル生成中にエラー: {e}", exc_info=True)
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
            self.logger.error(f"エグジットシグナル生成中にエラー: {e}", exc_info=True)
            return False

    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Optunaトライアルから最適化パラメータを生成
        (XChannelの全パラメータを対象とする)
        """
        params = {
            # --- XMA Params ---
            'xma_ma_type': trial.suggest_categorical('xma_ma_type', ['alma', 'hyperma', 'hma']),
            'xma_src_type': trial.suggest_categorical('xma_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
           # XMA Internal CER params (全パラメータ対応)
            'xma_cer_detector_type': trial.suggest_categorical('xma_cer_detector_type', ['hody', 'phac', 'dudi', 'hody_e', 'phac_e']),
            'xma_cer_lp_period': trial.suggest_int('xma_cer_lp_period', 3, 20),
            'xma_cer_hp_period': trial.suggest_int('xma_cer_hp_period', 50, 200),
            'xma_cer_cycle_part': trial.suggest_float('xma_cer_cycle_part', 0.2, 0.7, step=0.1),
            'xma_cer_max_cycle': trial.suggest_int('xma_cer_max_cycle', 30, 200,step=5),
            'xma_cer_min_cycle': trial.suggest_int('xma_cer_min_cycle', 3, 20,step=1),
            'xma_cer_max_output': trial.suggest_int('xma_cer_max_output', 20, 200,step=5),
            'xma_cer_min_output': trial.suggest_int('xma_cer_min_output', 5, 20),
            'xma_cer_src_type': trial.suggest_categorical('xma_cer_src_type', [None, 'close', 'hlc3', 'hl2', 'ohlc4']),
 
            # --- Multiplier Trigger Params ---
            # Multiplier CER params
            'mult_cer_detector_type': trial.suggest_categorical('mult_cer_detector_type', ['hody', 'phac', 'dudi', 'hody_e', 'phac_e']),
            'mult_cer_lp_period': trial.suggest_int('mult_cer_lp_period', 3, 20),
            'mult_cer_hp_period': trial.suggest_int('mult_cer_hp_period', 30, 200,step=5),
            'mult_cer_cycle_part': trial.suggest_float('mult_cer_cycle_part', 0.3, 0.7, step=0.1),
            'mult_cer_max_cycle': trial.suggest_int('mult_cer_max_cycle', 30, 200,step=5),
            'mult_cer_min_cycle': trial.suggest_int('mult_cer_min_cycle', 3, 20,step=1),
            'mult_cer_max_output': trial.suggest_int('mult_cer_max_output', 20, 200,step=5),
            'mult_cer_min_output': trial.suggest_int('mult_cer_min_output', 3, 20),
            'mult_cer_src_type': trial.suggest_categorical('mult_cer_src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
       }


        return params

    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optunaパラメータを戦略の__init__が受け入れる形式に変換する。
        (主に型の整合性を確保)
        """
        strategy_params = params.copy() # 基本的に同じキー名を使う

        # 型変換が必要なパラメータ (例: float -> int)
        int_keys = [
            'xma_cer_lp_period', 'xma_cer_hp_period', 'xma_cer_max_cycle', 'xma_cer_min_cycle',
            'xma_cer_max_output', 'xma_cer_min_output',
            'mult_cer_lp_period', 'mult_cer_hp_period', 'mult_cer_max_cycle',
            'mult_cer_min_cycle', 'mult_cer_max_output', 'mult_cer_min_output',
        ]
        for key in int_keys:
            if key in strategy_params:
                strategy_params[key] = int(strategy_params[key])

        # bool型はOptunaが正しく処理するはずだが念のため
        bool_keys = [
            'catr_use_kalman_filter',  
        ]
        for key in bool_keys:
            if key in strategy_params:
                strategy_params[key] = bool(strategy_params[key])

        return strategy_params 