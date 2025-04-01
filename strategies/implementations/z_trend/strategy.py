#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZTrendSignalGenerator


class ZTrendStrategy(BaseStrategy):
    """
    ZTrendディレクション + ZTrendフィルター戦略
    
    特徴:
    - サイクル効率比（CER）に基づく動的パラメータ最適化
    - ZTrendディレクションシグナルによる高精度なトレンド方向判定
    - ZTrendフィルターによる市場状態フィルタリング
    
    エントリー条件:
    - ロング: ZTrendDirectionSignalが1（上昇トレンド）かつZTrendFilterSignalが1（トレンド相場）
    - ショート: ZTrendDirectionSignalが-1（下降トレンド）かつZTrendFilterSignalが1（トレンド相場）
    
    エグジット条件:
    - ロング: ZTrendDirectionSignalが-1（下降トレンド）に変化
    - ショート: ZTrendDirectionSignalが1（上昇トレンド）に変化
    """
    
    def __init__(
        self,
        # ZTrendDirectionSignalのパラメータ
        cycle_detector_type: str = 'dudi_dc',
        lp_period: int = 10,
        hp_period: int = 195,
        cycle_part: float = 0.618,

        # CERのドミナントサイクル検出器用パラメータ
        cer_max_cycle: int = 300,
        cer_min_cycle: int = 15,
        cer_max_output: int = 162,
        cer_min_output: int = 30,

        # 最大パーセンタイル期間用（長期）ドミナントサイクル検出器のパラメータ
        max_percentile_dc_cycle_part: float = 0.786,
        max_percentile_dc_max_cycle: int = 250,
        max_percentile_dc_min_cycle: int = 15,
        max_percentile_dc_max_output: int = 162,
        max_percentile_dc_min_output: int = 20,

        # 最小パーセンタイル期間用（短期）ドミナントサイクル検出器のパラメータ
        min_percentile_dc_cycle_part: float = 0.34,
        min_percentile_dc_max_cycle: int = 50,
        min_percentile_dc_min_cycle: int = 10,
        min_percentile_dc_max_output: int = 50,
        min_percentile_dc_min_output: int = 8,

        # ZATR用ドミナントサイクル検出器のパラメータ
        zatr_max_dc_cycle_part: float = 0.786,
        zatr_max_dc_max_cycle: int = 55,
        zatr_max_dc_min_cycle: int = 3,
        zatr_max_dc_max_output: int = 60,
        zatr_max_dc_min_output: int = 3,
        zatr_min_dc_cycle_part: float = 0.15,
        zatr_min_dc_max_cycle: int = 35,
        zatr_min_dc_min_cycle: int = 3,
        zatr_min_dc_max_output: int = 20,
        zatr_min_dc_min_output: int = 3,

        # パーセンタイル乗数
        max_percentile_cycle_mult: float = 0.5,  # 最大パーセンタイル期間のサイクル乗数
        min_percentile_cycle_mult: float = 0.35,  # 最小パーセンタイル期間のサイクル乗数

        # ATR乗数
        max_multiplier: float = 3.4,
        min_multiplier: float = 1.3,

        # その他の設定
        smoother_type: str = 'hyper',   # 平滑化アルゴリズム（'alma'または'hyper'）
        src_type: str = 'hlc3',

        # Zトレンドフィルターのパラメータ
        max_stddev_period: int = 20,
        min_stddev_period: int = 5,
        max_lookback_period: int = 15,
        min_lookback_period: int = 5,
        max_rms_window: int = 15,
        min_rms_window: int = 5,
        max_threshold: float = 0.8,
        min_threshold: float = 0.55,
        combination_method: str = "rms",  # "sigmoid", "rms", "simple"

        # Zトレンドインデックスの追加パラメータ
        max_chop_dc_cycle_part: float = 0.55,
        max_chop_dc_max_cycle: int = 125,
        max_chop_dc_min_cycle: int = 13,
        max_chop_dc_max_output: int = 21,
        max_chop_dc_min_output: int = 13,
        min_chop_dc_cycle_part: float = 0.382,
        min_chop_dc_max_cycle: int = 75,
        min_chop_dc_min_cycle: int = 3,
        min_chop_dc_max_output: int = 15,
        min_chop_dc_min_output: int = 5
    ):
        """
        初期化
        
        Args:
            cycle_detector_type: サイクル検出器の種類（デフォルト: 'hody_dc'）
            lp_period: ローパスフィルターの期間（デフォルト: 13）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            cer_max_cycle: CER用ドミナントサイクル検出器の最大サイクル（デフォルト: 233）
            cer_min_cycle: CER用ドミナントサイクル検出器の最小サイクル（デフォルト: 13）
            cer_max_output: CER用ドミナントサイクル検出器の最大出力（デフォルト: 144）
            cer_min_output: CER用ドミナントサイクル検出器の最小出力（デフォルト: 21）
            max_percentile_dc_cycle_part: 最大パーセンタイル期間用DCのサイクル部分（デフォルト: 0.5）
            max_percentile_dc_max_cycle: 最大パーセンタイル期間用DCの最大サイクル（デフォルト: 233）
            max_percentile_dc_min_cycle: 最大パーセンタイル期間用DCの最小サイクル（デフォルト: 13）
            max_percentile_dc_max_output: 最大パーセンタイル期間用DCの最大出力（デフォルト: 144）
            max_percentile_dc_min_output: 最大パーセンタイル期間用DCの最小出力（デフォルト: 21）
            min_percentile_dc_cycle_part: 最小パーセンタイル期間用DCのサイクル部分（デフォルト: 0.5）
            min_percentile_dc_max_cycle: 最小パーセンタイル期間用DCの最大サイクル（デフォルト: 55）
            min_percentile_dc_min_cycle: 最小パーセンタイル期間用DCの最小サイクル（デフォルト: 5）
            min_percentile_dc_max_output: 最小パーセンタイル期間用DCの最大出力（デフォルト: 34）
            min_percentile_dc_min_output: 最小パーセンタイル期間用DCの最小出力（デフォルト: 8）
            zatr_max_dc_cycle_part: ZATR最大DCのサイクル部分（デフォルト: 0.5）
            zatr_max_dc_max_cycle: ZATR最大DCの最大サイクル（デフォルト: 55）
            zatr_max_dc_min_cycle: ZATR最大DCの最小サイクル（デフォルト: 5）
            zatr_max_dc_max_output: ZATR最大DCの最大出力（デフォルト: 55）
            zatr_max_dc_min_output: ZATR最大DCの最小出力（デフォルト: 5）
            zatr_min_dc_cycle_part: ZATR最小DCのサイクル部分（デフォルト: 0.25）
            zatr_min_dc_max_cycle: ZATR最小DCの最大サイクル（デフォルト: 34）
            zatr_min_dc_min_cycle: ZATR最小DCの最小サイクル（デフォルト: 3）
            zatr_min_dc_max_output: ZATR最小DCの最大出力（デフォルト: 13）
            zatr_min_dc_min_output: ZATR最小DCの最小出力（デフォルト: 3）
            max_percentile_cycle_mult: 最大パーセンタイル期間のサイクル乗数（デフォルト: 0.5）
            min_percentile_cycle_mult: 最小パーセンタイル期間のサイクル乗数（デフォルト: 0.25）
            max_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（デフォルト: 1.0）
            smoother_type: 平滑化アルゴリズム（デフォルト: 'alma'）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
            max_stddev_period: 標準偏差期間の最大値（デフォルト: 13）
            min_stddev_period: 標準偏差期間の最小値（デフォルト: 5）
            max_lookback_period: ルックバック期間の最大値（デフォルト: 13）
            min_lookback_period: ルックバック期間の最小値（デフォルト: 5）
            max_rms_window: RMSウィンドウの最大値（デフォルト: 13）
            min_rms_window: RMSウィンドウの最小値（デフォルト: 5）
            max_threshold: フィルターしきい値の最大値（デフォルト: 0.75）
            min_threshold: フィルターしきい値の最小値（デフォルト: 0.55）
            combination_weight: 組み合わせの重み（デフォルト: 0.6）
            combination_method: 組み合わせメソッド（デフォルト: "sigmoid"）
            max_chop_dc_cycle_part: 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_max_cycle: 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_min_cycle: 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_max_output: 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_min_output: 最大チョピネス期間用ドミナントサイクル設定
            min_chop_dc_cycle_part: 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_max_cycle: 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_min_cycle: 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_max_output: 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_min_output: 最小チョピネス期間用ドミナントサイクル設定
        """
        super().__init__("ZTrend")
        
        # パラメータの設定
        self._parameters = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'cer_max_cycle': cer_max_cycle,
            'cer_min_cycle': cer_min_cycle,
            'cer_max_output': cer_max_output,
            'cer_min_output': cer_min_output,
            'max_percentile_dc_cycle_part': max_percentile_dc_cycle_part,
            'max_percentile_dc_max_cycle': max_percentile_dc_max_cycle,
            'max_percentile_dc_min_cycle': max_percentile_dc_min_cycle,
            'max_percentile_dc_max_output': max_percentile_dc_max_output,
            'max_percentile_dc_min_output': max_percentile_dc_min_output,
            'min_percentile_dc_cycle_part': min_percentile_dc_cycle_part,
            'min_percentile_dc_max_cycle': min_percentile_dc_max_cycle,
            'min_percentile_dc_min_cycle': min_percentile_dc_min_cycle,
            'min_percentile_dc_max_output': min_percentile_dc_max_output,
            'min_percentile_dc_min_output': min_percentile_dc_min_output,
            'zatr_max_dc_cycle_part': zatr_max_dc_cycle_part,
            'zatr_max_dc_max_cycle': zatr_max_dc_max_cycle,
            'zatr_max_dc_min_cycle': zatr_max_dc_min_cycle,
            'zatr_max_dc_max_output': zatr_max_dc_max_output,
            'zatr_max_dc_min_output': zatr_max_dc_min_output,
            'zatr_min_dc_cycle_part': zatr_min_dc_cycle_part,
            'zatr_min_dc_max_cycle': zatr_min_dc_max_cycle,
            'zatr_min_dc_min_cycle': zatr_min_dc_min_cycle,
            'zatr_min_dc_max_output': zatr_min_dc_max_output,
            'zatr_min_dc_min_output': zatr_min_dc_min_output,
            'max_percentile_cycle_mult': max_percentile_cycle_mult,
            'min_percentile_cycle_mult': min_percentile_cycle_mult,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'max_stddev_period': max_stddev_period,
            'min_stddev_period': min_stddev_period,
            'max_lookback_period': max_lookback_period,
            'min_lookback_period': min_lookback_period,
            'max_rms_window': max_rms_window,
            'min_rms_window': min_rms_window,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold,
            'combination_method': combination_method,
            'max_chop_dc_cycle_part': max_chop_dc_cycle_part,
            'max_chop_dc_max_cycle': max_chop_dc_max_cycle,
            'max_chop_dc_min_cycle': max_chop_dc_min_cycle,
            'max_chop_dc_max_output': max_chop_dc_max_output,
            'max_chop_dc_min_output': max_chop_dc_min_output,
            'min_chop_dc_cycle_part': min_chop_dc_cycle_part,
            'min_chop_dc_max_cycle': min_chop_dc_max_cycle,
            'min_chop_dc_min_cycle': min_chop_dc_min_cycle,
            'min_chop_dc_max_output': min_chop_dc_max_output,
            'min_chop_dc_min_output': min_chop_dc_min_output
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZTrendSignalGenerator(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            cer_max_cycle=cer_max_cycle,
            cer_min_cycle=cer_min_cycle,
            cer_max_output=cer_max_output,
            cer_min_output=cer_min_output,
            max_percentile_dc_cycle_part=max_percentile_dc_cycle_part,
            max_percentile_dc_max_cycle=max_percentile_dc_max_cycle,
            max_percentile_dc_min_cycle=max_percentile_dc_min_cycle,
            max_percentile_dc_max_output=max_percentile_dc_max_output,
            max_percentile_dc_min_output=max_percentile_dc_min_output,
            min_percentile_dc_cycle_part=min_percentile_dc_cycle_part,
            min_percentile_dc_max_cycle=min_percentile_dc_max_cycle,
            min_percentile_dc_min_cycle=min_percentile_dc_min_cycle,
            min_percentile_dc_max_output=min_percentile_dc_max_output,
            min_percentile_dc_min_output=min_percentile_dc_min_output,
            zatr_max_dc_cycle_part=zatr_max_dc_cycle_part,
            zatr_max_dc_max_cycle=zatr_max_dc_max_cycle,
            zatr_max_dc_min_cycle=zatr_max_dc_min_cycle,
            zatr_max_dc_max_output=zatr_max_dc_max_output,
            zatr_max_dc_min_output=zatr_max_dc_min_output,
            zatr_min_dc_cycle_part=zatr_min_dc_cycle_part,
            zatr_min_dc_max_cycle=zatr_min_dc_max_cycle,
            zatr_min_dc_min_cycle=zatr_min_dc_min_cycle,
            zatr_min_dc_max_output=zatr_min_dc_max_output,
            zatr_min_dc_min_output=zatr_min_dc_min_output,
            max_percentile_cycle_mult=max_percentile_cycle_mult,
            min_percentile_cycle_mult=min_percentile_cycle_mult,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            smoother_type=smoother_type,
            src_type=src_type,
            max_stddev_period=max_stddev_period,
            min_stddev_period=min_stddev_period,
            max_lookback_period=max_lookback_period,
            min_lookback_period=min_lookback_period,
            max_rms_window=max_rms_window,
            min_rms_window=min_rms_window,
            max_threshold=max_threshold,
            min_threshold=min_threshold,
            combination_method=combination_method,
            max_chop_dc_cycle_part=max_chop_dc_cycle_part,
            max_chop_dc_max_cycle=max_chop_dc_max_cycle,
            max_chop_dc_min_cycle=max_chop_dc_min_cycle,
            max_chop_dc_max_output=max_chop_dc_max_output,
            max_chop_dc_min_output=max_chop_dc_min_output,
            min_chop_dc_cycle_part=min_chop_dc_cycle_part,
            min_chop_dc_max_cycle=min_chop_dc_max_cycle,
            min_chop_dc_min_cycle=min_chop_dc_min_cycle,
            min_chop_dc_max_output=min_chop_dc_max_output,
            min_chop_dc_min_output=min_chop_dc_min_output
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
            # 共通パラメータ
            'cycle_detector_type': trial.suggest_categorical('cycle_detector_type', ['hody_dc', 'dudi_dc', 'phac_dc']),
            'lp_period': trial.suggest_int('lp_period', 5, 21),
            'hp_period': trial.suggest_int('hp_period', 89, 233),
            'cycle_part': trial.suggest_float('cycle_part', 0.2, 0.8, step=0.1),
            
            # ZTrendDirectionSignalのパラメータ
            'cer_max_cycle': trial.suggest_int('cer_max_cycle', 180, 300),
            'cer_min_cycle': trial.suggest_int('cer_min_cycle', 8, 20),
            'cer_max_output': trial.suggest_int('cer_max_output', 100, 200),
            'cer_min_output': trial.suggest_int('cer_min_output', 15, 30),
            
            # 最大パーセンタイル期間用パラメータ
            'max_percentile_dc_cycle_part': trial.suggest_float('max_percentile_dc_cycle_part', 0.3, 0.7, step=0.1),
            'max_percentile_dc_max_cycle': trial.suggest_int('max_percentile_dc_max_cycle', 180, 300),
            'max_percentile_dc_min_cycle': trial.suggest_int('max_percentile_dc_min_cycle', 8, 20),
            'max_percentile_dc_max_output': trial.suggest_int('max_percentile_dc_max_output', 100, 200),
            'max_percentile_dc_min_output': trial.suggest_int('max_percentile_dc_min_output', 15, 30),
            
            # 最小パーセンタイル期間用パラメータ
            'min_percentile_dc_cycle_part': trial.suggest_float('min_percentile_dc_cycle_part', 0.3, 0.7, step=0.1),
            'min_percentile_dc_max_cycle': trial.suggest_int('min_percentile_dc_max_cycle', 35, 75),
            'min_percentile_dc_min_cycle': trial.suggest_int('min_percentile_dc_min_cycle', 3, 10),
            'min_percentile_dc_max_output': trial.suggest_int('min_percentile_dc_max_output', 20, 50),
            'min_percentile_dc_min_output': trial.suggest_int('min_percentile_dc_min_output', 5, 15),
            
            # ZATR用パラメータ
            'zatr_max_dc_cycle_part': trial.suggest_float('zatr_max_dc_cycle_part', 0.3, 0.7, step=0.1),
            'zatr_max_dc_max_cycle': trial.suggest_int('zatr_max_dc_max_cycle', 35, 75),
            'zatr_max_dc_min_cycle': trial.suggest_int('zatr_max_dc_min_cycle', 3, 10),
            'zatr_max_dc_max_output': trial.suggest_int('zatr_max_dc_max_output', 35, 75),
            'zatr_max_dc_min_output': trial.suggest_int('zatr_max_dc_min_output', 3, 10),
            'zatr_min_dc_cycle_part': trial.suggest_float('zatr_min_dc_cycle_part', 0.15, 0.5, step=0.05),
            'zatr_min_dc_max_cycle': trial.suggest_int('zatr_min_dc_max_cycle', 20, 50),
            'zatr_min_dc_min_cycle': trial.suggest_int('zatr_min_dc_min_cycle', 2, 7),
            'zatr_min_dc_max_output': trial.suggest_int('zatr_min_dc_max_output', 8, 20),
            'zatr_min_dc_min_output': trial.suggest_int('zatr_min_dc_min_output', 2, 7),
            
            # パーセンタイル乗数
            'max_percentile_cycle_mult': trial.suggest_float('max_percentile_cycle_mult', 0.3, 0.7, step=0.1),
            'min_percentile_cycle_mult': trial.suggest_float('min_percentile_cycle_mult', 0.15, 0.4, step=0.05),
            
            # ATR乗数
            'max_multiplier': trial.suggest_float('max_multiplier', 2.0, 4.0, step=0.1),
            'min_multiplier': trial.suggest_float('min_multiplier', 0.8, 2.0, step=0.1),
            
            # その他の設定
            'smoother_type': trial.suggest_categorical('smoother_type', ['alma', 'hyper']),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            
            # Zトレンドフィルターのパラメータ
            'max_stddev_period': trial.suggest_int('max_stddev_period', 8, 20),
            'min_stddev_period': trial.suggest_int('min_stddev_period', 3, 8),
            'max_lookback_period': trial.suggest_int('max_lookback_period', 8, 20),
            'min_lookback_period': trial.suggest_int('min_lookback_period', 3, 8),
            'max_rms_window': trial.suggest_int('max_rms_window', 8, 20),
            'min_rms_window': trial.suggest_int('min_rms_window', 3, 8),
            'max_threshold': trial.suggest_float('max_threshold', 0.65, 0.85, step=0.05),
            'min_threshold': trial.suggest_float('min_threshold', 0.45, 0.65, step=0.05),
            'combination_method': trial.suggest_categorical('combination_method', ['sigmoid', 'rms', 'simple']),
            
            # Zトレンドインデックスの追加パラメータ
            'max_chop_dc_cycle_part': trial.suggest_float('max_chop_dc_cycle_part', 0.3, 0.7, step=0.1),
            'max_chop_dc_max_cycle': trial.suggest_int('max_chop_dc_max_cycle', 55, 233),
            'max_chop_dc_min_cycle': trial.suggest_int('max_chop_dc_min_cycle', 5, 15),
            'max_chop_dc_max_output': trial.suggest_int('max_chop_dc_max_output', 20, 50),
            'max_chop_dc_min_output': trial.suggest_int('max_chop_dc_min_output', 8, 20),
            'min_chop_dc_cycle_part': trial.suggest_float('min_chop_dc_cycle_part', 0.15, 0.4, step=0.05),
            'min_chop_dc_max_cycle': trial.suggest_int('min_chop_dc_max_cycle', 35, 75),
            'min_chop_dc_min_cycle': trial.suggest_int('min_chop_dc_min_cycle', 3, 10),
            'min_chop_dc_max_output': trial.suggest_int('min_chop_dc_max_output', 8, 20),
            'min_chop_dc_min_output': trial.suggest_int('min_chop_dc_min_output', 3, 10),
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
            'cycle_detector_type': params['cycle_detector_type'],
            'lp_period': int(params['lp_period']),
            'hp_period': int(params['hp_period']),
            'cycle_part': float(params['cycle_part']),
            'cer_max_cycle': int(params['cer_max_cycle']),
            'cer_min_cycle': int(params['cer_min_cycle']),
            'cer_max_output': int(params['cer_max_output']),
            'cer_min_output': int(params['cer_min_output']),
            'max_percentile_dc_cycle_part': float(params['max_percentile_dc_cycle_part']),
            'max_percentile_dc_max_cycle': int(params['max_percentile_dc_max_cycle']),
            'max_percentile_dc_min_cycle': int(params['max_percentile_dc_min_cycle']),
            'max_percentile_dc_max_output': int(params['max_percentile_dc_max_output']),
            'max_percentile_dc_min_output': int(params['max_percentile_dc_min_output']),
            'min_percentile_dc_cycle_part': float(params['min_percentile_dc_cycle_part']),
            'min_percentile_dc_max_cycle': int(params['min_percentile_dc_max_cycle']),
            'min_percentile_dc_min_cycle': int(params['min_percentile_dc_min_cycle']),
            'min_percentile_dc_max_output': int(params['min_percentile_dc_max_output']),
            'min_percentile_dc_min_output': int(params['min_percentile_dc_min_output']),
            'zatr_max_dc_cycle_part': float(params['zatr_max_dc_cycle_part']),
            'zatr_max_dc_max_cycle': int(params['zatr_max_dc_max_cycle']),
            'zatr_max_dc_min_cycle': int(params['zatr_max_dc_min_cycle']),
            'zatr_max_dc_max_output': int(params['zatr_max_dc_max_output']),
            'zatr_max_dc_min_output': int(params['zatr_max_dc_min_output']),
            'zatr_min_dc_cycle_part': float(params['zatr_min_dc_cycle_part']),
            'zatr_min_dc_max_cycle': int(params['zatr_min_dc_max_cycle']),
            'zatr_min_dc_min_cycle': int(params['zatr_min_dc_min_cycle']),
            'zatr_min_dc_max_output': int(params['zatr_min_dc_max_output']),
            'zatr_min_dc_min_output': int(params['zatr_min_dc_min_output']),
            'max_percentile_cycle_mult': float(params['max_percentile_cycle_mult']),
            'min_percentile_cycle_mult': float(params['min_percentile_cycle_mult']),
            'max_multiplier': float(params['max_multiplier']),
            'min_multiplier': float(params['min_multiplier']),
            'smoother_type': params['smoother_type'],
            'src_type': params['src_type'],
            'max_stddev_period': int(params['max_stddev_period']),
            'min_stddev_period': int(params['min_stddev_period']),
            'max_lookback_period': int(params['max_lookback_period']),
            'min_lookback_period': int(params['min_lookback_period']),
            'max_rms_window': int(params['max_rms_window']),
            'min_rms_window': int(params['min_rms_window']),
            'max_threshold': float(params['max_threshold']),
            'min_threshold': float(params['min_threshold']),
            'combination_method': params['combination_method'],
            'max_chop_dc_cycle_part': float(params['max_chop_dc_cycle_part']),
            'max_chop_dc_max_cycle': int(params['max_chop_dc_max_cycle']),
            'max_chop_dc_min_cycle': int(params['max_chop_dc_min_cycle']),
            'max_chop_dc_max_output': int(params['max_chop_dc_max_output']),
            'max_chop_dc_min_output': int(params['max_chop_dc_min_output']),
            'min_chop_dc_cycle_part': float(params['min_chop_dc_cycle_part']),
            'min_chop_dc_max_cycle': int(params['min_chop_dc_max_cycle']),
            'min_chop_dc_min_cycle': int(params['min_chop_dc_min_cycle']),
            'min_chop_dc_max_output': int(params['min_chop_dc_max_output']),
            'min_chop_dc_min_output': int(params['min_chop_dc_min_output']),
        }
        return strategy_params 