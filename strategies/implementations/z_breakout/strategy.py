#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZBreakoutSignalGenerator


class ZBreakoutStrategy(BaseStrategy):
    """
    Zブレイクアウトストラテジー
    
    特徴:
    - サイクル効率比（CER）に基づく動的パラメータ最適化
    - Zチャネルによる高精度なエントリーポイント検出
    - Zハーストエクスポネントによるトレンド/レンジ判定
    
    エントリー条件:
    - ロング: Zチャネルの買いシグナルかつZハーストフィルターがトレンド相場
    - ショート: Zチャネルの売りシグナルかつZハーストフィルターがトレンド相場
    
    エグジット条件:
    - ロング: Zチャネルの売りシグナル
    - ショート: Zチャネルの買いシグナル
    """
    
    def __init__(
        self,
        # Zチャネルのパラメータ
        cycle_detector_type: str = 'phac_dce',
        lp_period: int = 5,
        hp_period: int = 55,
        cycle_part: float = 0.7,
        smoother_type: str = 'alma',
        src_type: str = 'hlc3',
        band_lookback: int = 1,
        # 動的乗数の範囲パラメータ
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 6.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        
        # ZMA用パラメータ
        zma_max_dc_cycle_part: float = 0.5,     # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_cycle: int = 100,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_cycle: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_output: int = 120,       # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_output: int = 22,        # ZMA: 最大期間用ドミナントサイクル計算用
        
        zma_min_dc_cycle_part: float = 0.25,    # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_cycle: int = 55,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_cycle: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_output: int = 13,        # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_output: int = 3,         # ZMA: 最小期間用ドミナントサイクル計算用
        
        zma_max_slow_period: int = 50,          # ZMA: 遅い移動平均の最大期間
        zma_min_slow_period: int = 13,          # ZMA: 遅い移動平均の最小期間
        zma_max_fast_period: int = 8,           # ZMA: 速い移動平均の最大期間
        zma_min_fast_period: int = 2,           # ZMA: 速い移動平均の最小期間
        zma_hyper_smooth_period: int = 0,       # ZMA: ハイパースムーサーの平滑化期間
        
        # ZATR用パラメータ
        zatr_max_dc_cycle_part: float = 0.7,    # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_cycle: int = 77,        # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_cycle: int = 5,         # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_output: int = 35,       # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_output: int = 5,        # ZATR: 最大期間用ドミナントサイクル計算用
        
        zatr_min_dc_cycle_part: float = 0.5,    # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_cycle: int = 34,        # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_cycle: int = 3,         # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_output: int = 13,       # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_output: int = 3,        # ZATR: 最小期間用ドミナントサイクル計算用
        
        # Zハーストエクスポネントフィルターのパラメータ
        max_window_dc_cycle_part: float = 0.75,
        max_window_dc_max_cycle: int = 144,
        max_window_dc_min_cycle: int = 8,
        max_window_dc_max_output: int = 200,
        max_window_dc_min_output: int = 50,
        
        min_window_dc_cycle_part: float = 0.5,
        min_window_dc_max_cycle: int = 55,
        min_window_dc_min_cycle: int = 5,
        min_window_dc_max_output: int = 50,
        min_window_dc_min_output: int = 20,
        
        # ラグパラメータ
        max_lag_ratio: float = 0.5,  # 最大ラグはウィンドウの何%か
        min_lag_ratio: float = 0.1,  # 最小ラグはウィンドウの何%か
        
        # ステップパラメータ
        max_step: int = 10,
        min_step: int = 2,
        
        # 動的しきい値のパラメータ
        max_threshold: float = 0.7,
        min_threshold: float = 0.55
    ):
        """
        初期化
        
        Args:
            cycle_detector_type: サイクル検出器の種類（デフォルト: 'phac_dce'）
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 55）
            cycle_part: サイクル部分の倍率（デフォルト: 0.7）
            smoother_type: 平滑化アルゴリズム（デフォルト: 'alma'）
            src_type: 価格ソースタイプ（デフォルト: 'hlc3'）
            band_lookback: 過去バンド参照期間（デフォルト: 1）
            
            # 動的乗数の範囲パラメータ
            max_max_multiplier: 最大乗数の最大値（デフォルト: 8.0）
            min_max_multiplier: 最大乗数の最小値（デフォルト: 6.0）
            max_min_multiplier: 最小乗数の最大値（デフォルト: 1.5）
            min_min_multiplier: 最小乗数の最小値（デフォルト: 0.5）
            
            # ZMA用パラメータ
            zma_max_dc_cycle_part: ZMA最大期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            zma_max_dc_max_cycle: ZMA最大期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 100）
            zma_max_dc_min_cycle: ZMA最大期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zma_max_dc_max_output: ZMA最大期間用ドミナントサイクル計算用の最大出力値（デフォルト: 120）
            zma_max_dc_min_output: ZMA最大期間用ドミナントサイクル計算用の最小出力値（デフォルト: 22）
            
            zma_min_dc_cycle_part: ZMA最小期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.25）
            zma_min_dc_max_cycle: ZMA最小期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            zma_min_dc_min_cycle: ZMA最小期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zma_min_dc_max_output: ZMA最小期間用ドミナントサイクル計算用の最大出力値（デフォルト: 13）
            zma_min_dc_min_output: ZMA最小期間用ドミナントサイクル計算用の最小出力値（デフォルト: 3）
            
            zma_max_slow_period: ZMA遅い移動平均の最大期間（デフォルト: 50）
            zma_min_slow_period: ZMA遅い移動平均の最小期間（デフォルト: 13）
            zma_max_fast_period: ZMA速い移動平均の最大期間（デフォルト: 8）
            zma_min_fast_period: ZMA速い移動平均の最小期間（デフォルト: 2）
            zma_hyper_smooth_period: ZMAハイパースムーサーの平滑化期間（デフォルト: 0）
            
            # ZATR用パラメータ
            zatr_max_dc_cycle_part: ZATR最大期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.7）
            zatr_max_dc_max_cycle: ZATR最大期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 77）
            zatr_max_dc_min_cycle: ZATR最大期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            zatr_max_dc_max_output: ZATR最大期間用ドミナントサイクル計算用の最大出力値（デフォルト: 35）
            zatr_max_dc_min_output: ZATR最大期間用ドミナントサイクル計算用の最小出力値（デフォルト: 5）
            
            zatr_min_dc_cycle_part: ZATR最小期間用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            zatr_min_dc_max_cycle: ZATR最小期間用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 34）
            zatr_min_dc_min_cycle: ZATR最小期間用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 3）
            zatr_min_dc_max_output: ZATR最小期間用ドミナントサイクル計算用の最大出力値（デフォルト: 13）
            zatr_min_dc_min_output: ZATR最小期間用ドミナントサイクル計算用の最小出力値（デフォルト: 3）
            
            # Zハーストエクスポネントフィルターのパラメータ
            max_window_dc_cycle_part: 最大ウィンドウ用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.75）
            max_window_dc_max_cycle: 最大ウィンドウ用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 144）
            max_window_dc_min_cycle: 最大ウィンドウ用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 8）
            max_window_dc_max_output: 最大ウィンドウ用ドミナントサイクル計算用の最大出力値（デフォルト: 200）
            max_window_dc_min_output: 最大ウィンドウ用ドミナントサイクル計算用の最小出力値（デフォルト: 50）
            
            min_window_dc_cycle_part: 最小ウィンドウ用ドミナントサイクル計算用のサイクル部分（デフォルト: 0.5）
            min_window_dc_max_cycle: 最小ウィンドウ用ドミナントサイクル計算用の最大サイクル期間（デフォルト: 55）
            min_window_dc_min_cycle: 最小ウィンドウ用ドミナントサイクル計算用の最小サイクル期間（デフォルト: 5）
            min_window_dc_max_output: 最小ウィンドウ用ドミナントサイクル計算用の最大出力値（デフォルト: 50）
            min_window_dc_min_output: 最小ウィンドウ用ドミナントサイクル計算用の最小出力値（デフォルト: 20）
            
            max_lag_ratio: 最大ラグとウィンドウの比率（デフォルト: 0.5）
            min_lag_ratio: 最小ラグとウィンドウの比率（デフォルト: 0.1）
            
            max_step: 最大ステップ（デフォルト: 10）
            min_step: 最小ステップ（デフォルト: 2）
            
            max_threshold: 最大しきい値（デフォルト: 0.7）
            min_threshold: 最小しきい値（デフォルト: 0.55）
        """
        super().__init__("ZBreakout")
        
        # パラメータの設定
        self._parameters = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            
            # ZMA用パラメータ
            'zma_max_dc_cycle_part': zma_max_dc_cycle_part,
            'zma_max_dc_max_cycle': zma_max_dc_max_cycle,
            'zma_max_dc_min_cycle': zma_max_dc_min_cycle,
            'zma_max_dc_max_output': zma_max_dc_max_output,
            'zma_max_dc_min_output': zma_max_dc_min_output,
            'zma_min_dc_cycle_part': zma_min_dc_cycle_part,
            'zma_min_dc_max_cycle': zma_min_dc_max_cycle,
            'zma_min_dc_min_cycle': zma_min_dc_min_cycle,
            'zma_min_dc_max_output': zma_min_dc_max_output,
            'zma_min_dc_min_output': zma_min_dc_min_output,
            'zma_max_slow_period': zma_max_slow_period,
            'zma_min_slow_period': zma_min_slow_period,
            'zma_max_fast_period': zma_max_fast_period,
            'zma_min_fast_period': zma_min_fast_period,
            'zma_hyper_smooth_period': zma_hyper_smooth_period,
            
            # ZATR用パラメータ
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
            
            # Zハーストエクスポネントフィルターのパラメータ
            'max_window_dc_cycle_part': max_window_dc_cycle_part,
            'max_window_dc_max_cycle': max_window_dc_max_cycle,
            'max_window_dc_min_cycle': max_window_dc_min_cycle,
            'max_window_dc_max_output': max_window_dc_max_output,
            'max_window_dc_min_output': max_window_dc_min_output,
            'min_window_dc_cycle_part': min_window_dc_cycle_part,
            'min_window_dc_max_cycle': min_window_dc_max_cycle,
            'min_window_dc_min_cycle': min_window_dc_min_cycle,
            'min_window_dc_max_output': min_window_dc_max_output,
            'min_window_dc_min_output': min_window_dc_min_output,
            'max_lag_ratio': max_lag_ratio,
            'min_lag_ratio': min_lag_ratio,
            'max_step': max_step,
            'min_step': min_step,
            'max_threshold': max_threshold,
            'min_threshold': min_threshold
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZBreakoutSignalGenerator(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            smoother_type=smoother_type,
            src_type=src_type,
            band_lookback=band_lookback,
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            
            # ZMA用パラメータ
            zma_max_dc_cycle_part=zma_max_dc_cycle_part,
            zma_max_dc_max_cycle=zma_max_dc_max_cycle,
            zma_max_dc_min_cycle=zma_max_dc_min_cycle,
            zma_max_dc_max_output=zma_max_dc_max_output,
            zma_max_dc_min_output=zma_max_dc_min_output,
            zma_min_dc_cycle_part=zma_min_dc_cycle_part,
            zma_min_dc_max_cycle=zma_min_dc_max_cycle,
            zma_min_dc_min_cycle=zma_min_dc_min_cycle,
            zma_min_dc_max_output=zma_min_dc_max_output,
            zma_min_dc_min_output=zma_min_dc_min_output,
            zma_max_slow_period=zma_max_slow_period,
            zma_min_slow_period=zma_min_slow_period,
            zma_max_fast_period=zma_max_fast_period,
            zma_min_fast_period=zma_min_fast_period,
            zma_hyper_smooth_period=zma_hyper_smooth_period,
            
            # ZATR用パラメータ
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
            
            # Zハーストエクスポネントフィルターのパラメータ
            max_window_dc_cycle_part=max_window_dc_cycle_part,
            max_window_dc_max_cycle=max_window_dc_max_cycle,
            max_window_dc_min_cycle=max_window_dc_min_cycle,
            max_window_dc_max_output=max_window_dc_max_output,
            max_window_dc_min_output=max_window_dc_min_output,
            min_window_dc_cycle_part=min_window_dc_cycle_part,
            min_window_dc_max_cycle=min_window_dc_max_cycle,
            min_window_dc_min_cycle=min_window_dc_min_cycle,
            min_window_dc_max_output=min_window_dc_max_output,
            min_window_dc_min_output=min_window_dc_min_output,
            max_lag_ratio=max_lag_ratio,
            min_lag_ratio=min_lag_ratio,
            max_step=max_step,
            min_step=min_step,
            max_threshold=max_threshold,
            min_threshold=min_threshold
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
            'cycle_detector_type': trial.suggest_categorical('cycle_detector_type', ['dudi_dc', 'hody_dc', 'phac_dc', 'dudi_dce', 'hody_dce', 'phac_dce']),
            'lp_period': trial.suggest_int('lp_period', 3, 21),
            'hp_period': trial.suggest_int('hp_period', 34, 233),
            'cycle_part': trial.suggest_float('cycle_part', 0.2, 0.9, step=0.1),
            'smoother_type': trial.suggest_categorical('smoother_type', ['alma', 'hyper']),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
            'band_lookback': trial.suggest_int('band_lookback', 1, 5),
            
            # 動的乗数の範囲パラメータ
            'max_max_multiplier': trial.suggest_float('max_max_multiplier', 5.0, 10.0, step=0.5),
            'min_max_multiplier': trial.suggest_float('min_max_multiplier', 3.0, 6.0, step=0.5),
            'max_min_multiplier': trial.suggest_float('max_min_multiplier', 1.0, 2.0, step=0.1),
            'min_min_multiplier': trial.suggest_float('min_min_multiplier', 0.0, 1.0, step=0.1),
            
            # Zハーストエクスポネントフィルターのパラメータ
            'max_window_dc_cycle_part': trial.suggest_float('max_window_dc_cycle_part', 0.3, 0.9, step=0.1),
            'max_window_dc_max_cycle': trial.suggest_int('max_window_dc_max_cycle', 89, 233),
            'max_window_dc_min_cycle': trial.suggest_int('max_window_dc_min_cycle', 5, 13),
            'max_window_dc_max_output': trial.suggest_int('max_window_dc_max_output', 100, 300, step=20),
            'max_window_dc_min_output': trial.suggest_int('max_window_dc_min_output', 30, 70, step=5),
            
            'min_window_dc_cycle_part': trial.suggest_float('min_window_dc_cycle_part', 0.2, 0.6, step=0.1),
            'min_window_dc_max_cycle': trial.suggest_int('min_window_dc_max_cycle', 34, 89),
            'min_window_dc_min_cycle': trial.suggest_int('min_window_dc_min_cycle', 3, 8),
            'min_window_dc_max_output': trial.suggest_int('min_window_dc_max_output', 30, 70, step=5),
            'min_window_dc_min_output': trial.suggest_int('min_window_dc_min_output', 10, 30, step=2),
            
            'max_lag_ratio': trial.suggest_float('max_lag_ratio', 0.3, 0.7, step=0.1),
            'min_lag_ratio': trial.suggest_float('min_lag_ratio', 0.05, 0.2, step=0.05),
            
            'max_step': trial.suggest_int('max_step', 5, 15),
            'min_step': trial.suggest_int('min_step', 1, 5),
            
            'max_threshold': trial.suggest_float('max_threshold', 0.6, 0.8, step=0.05),
            'min_threshold': trial.suggest_float('min_threshold', 0.5, 0.6, step=0.05)
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
        strategy_params = {}
        
        # 基本パラメータを変換
        for key, value in params.items():
            if isinstance(value, (int, float, str)):
                strategy_params[key] = value
            elif isinstance(value, np.int64):
                strategy_params[key] = int(value)
            elif isinstance(value, np.float64):
                strategy_params[key] = float(value)
            else:
                strategy_params[key] = value
        
        return strategy_params 