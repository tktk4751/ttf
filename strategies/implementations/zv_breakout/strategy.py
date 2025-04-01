#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZVBreakoutSignalGenerator


class ZVBreakoutStrategy(BaseStrategy):
    """
    ZVチャネルブレイクアウト戦略
    
    ボリンジャーバンドとZチャネルを組み合わせた「ZVチャネル」のブレイクアウトでトレードするシンプルな戦略。
    
    エントリー条件:
    - ロング: ZVチャネルの上限ブレイクアウトで買いシグナル(1)が発生したとき
    - ショート: ZVチャネルの下限ブレイクアウトで売りシグナル(-1)が発生したとき
    
    エグジット条件:
    - ロング: ZVチャネルの売りシグナル(-1)が発生したとき
    - ショート: ZVチャネルの買いシグナル(1)が発生したとき
    """
    
    def __init__(
        self,
        # 基本パラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        
        # ボリンジャーバンドパラメータ
        bb_max_multiplier: float = 2.5,
        bb_min_multiplier: float = 1.0,
        
        # ZBBの標準偏差計算用パラメータ
        bb_max_cycle_part: float = 0.5,    # 標準偏差最大期間用サイクル部分
        bb_max_max_cycle: int = 144,       # 標準偏差最大期間用最大サイクル
        bb_max_min_cycle: int = 10,        # 標準偏差最大期間用最小サイクル
        bb_max_max_output: int = 89,       # 標準偏差最大期間用最大出力値
        bb_max_min_output: int = 13,       # 標準偏差最大期間用最小出力値
        bb_min_cycle_part: float = 0.25,   # 標準偏差最小期間用サイクル部分
        bb_min_max_cycle: int = 55,        # 標準偏差最小期間用最大サイクル
        bb_min_min_cycle: int = 5,         # 標準偏差最小期間用最小サイクル
        bb_min_max_output: int = 21,       # 標準偏差最小期間用最大出力値
        bb_min_min_output: int = 5,        # 標準偏差最小期間用最小出力値
        
        # Zチャネルパラメータ
        kc_max_multiplier: float = 3.0,
        kc_min_multiplier: float = 1.5,
        kc_smoother_type: str = 'alma',
        
        # ZChannel ZMA用パラメータ
        zma_max_dc_cycle_part: float = 0.5,     # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_cycle: int = 144,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_cycle: int = 5,          # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_max_output: int = 89,        # ZMA: 最大期間用ドミナントサイクル計算用
        zma_max_dc_min_output: int = 22,        # ZMA: 最大期間用ドミナントサイクル計算用
        
        zma_min_dc_cycle_part: float = 0.25,    # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_cycle: int = 55,         # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_cycle: int = 5,          # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_max_output: int = 13,        # ZMA: 最小期間用ドミナントサイクル計算用
        zma_min_dc_min_output: int = 3,         # ZMA: 最小期間用ドミナントサイクル計算用
        
        zma_max_slow_period: int = 34,          # ZMA: 遅い移動平均の最大期間
        zma_min_slow_period: int = 9,           # ZMA: 遅い移動平均の最小期間
        zma_max_fast_period: int = 8,           # ZMA: 速い移動平均の最大期間
        zma_min_fast_period: int = 2,           # ZMA: 速い移動平均の最小期間
        zma_hyper_smooth_period: int = 0,       # ZMA: ハイパースムーサーの平滑化期間（0=平滑化しない）
        
        # ZChannel ZATR用パラメータ
        zatr_max_dc_cycle_part: float = 0.5,    # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_cycle: int = 55,        # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_cycle: int = 5,         # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_max_output: int = 55,       # ZATR: 最大期間用ドミナントサイクル計算用
        zatr_max_dc_min_output: int = 5,        # ZATR: 最大期間用ドミナントサイクル計算用
        
        zatr_min_dc_cycle_part: float = 0.25,   # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_cycle: int = 34,        # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_cycle: int = 3,         # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_max_output: int = 13,       # ZATR: 最小期間用ドミナントサイクル計算用
        zatr_min_dc_min_output: int = 3,        # ZATR: 最小期間用ドミナントサイクル計算用
        
        # 共通パラメータ
        src_type: str = 'hlc3',
        band_lookback: int = 1,
        
        # 取引方向設定
        trade_long: bool = True,         # ロング取引を行うか
        trade_short: bool = True         # ショート取引を行うか
    ):
        """初期化"""
        super().__init__("ZVBreakout")
        
        # パラメータの設定
        self._parameters = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            
            'bb_max_multiplier': bb_max_multiplier,
            'bb_min_multiplier': bb_min_multiplier,
            
            'bb_max_cycle_part': bb_max_cycle_part,
            'bb_max_max_cycle': bb_max_max_cycle,
            'bb_max_min_cycle': bb_max_min_cycle,
            'bb_max_max_output': bb_max_max_output,
            'bb_max_min_output': bb_max_min_output,
            'bb_min_cycle_part': bb_min_cycle_part,
            'bb_min_max_cycle': bb_min_max_cycle,
            'bb_min_min_cycle': bb_min_min_cycle,
            'bb_min_max_output': bb_min_max_output,
            'bb_min_min_output': bb_min_min_output,
            
            'kc_max_multiplier': kc_max_multiplier,
            'kc_min_multiplier': kc_min_multiplier,
            'kc_smoother_type': kc_smoother_type,
            
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
            
            'src_type': src_type,
            'band_lookback': band_lookback,
            
            'trade_long': trade_long,
            'trade_short': trade_short
        }
        
        # 取引方向設定
        self._trade_long = trade_long
        self._trade_short = trade_short
        
        # シグナル生成器の初期化
        self.signal_generator = ZVBreakoutSignalGenerator(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            
            bb_max_multiplier=bb_max_multiplier,
            bb_min_multiplier=bb_min_multiplier,
            
            bb_max_cycle_part=bb_max_cycle_part,
            bb_max_max_cycle=bb_max_max_cycle,
            bb_max_min_cycle=bb_max_min_cycle,
            bb_max_max_output=bb_max_max_output,
            bb_max_min_output=bb_max_min_output,
            bb_min_cycle_part=bb_min_cycle_part,
            bb_min_max_cycle=bb_min_max_cycle,
            bb_min_min_cycle=bb_min_min_cycle,
            bb_min_max_output=bb_min_max_output,
            bb_min_min_output=bb_min_min_output,
            
            kc_max_multiplier=kc_max_multiplier,
            kc_min_multiplier=kc_min_multiplier,
            kc_smoother_type=kc_smoother_type,
            
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
            
            src_type=src_type,
            band_lookback=band_lookback
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
            signals = self.signal_generator.get_entry_signals(data)
            
            # 取引方向によるフィルタリング
            if not self._trade_long:
                signals[signals == 1] = 0
            if not self._trade_short:
                signals[signals == -1] = 0
                
            return signals
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
            'hp_period': trial.suggest_int('hp_period', 89, 233),
            'cycle_part': trial.suggest_float('cycle_part', 0.4, 0.6, step=0.05),
            
            # ボリンジャーバンドパラメータ
            'bb_max_multiplier': trial.suggest_float('bb_max_multiplier', 2.0, 3.0, step=0.1),
            'bb_min_multiplier': trial.suggest_float('bb_min_multiplier', 0.8, 1.5, step=0.1),
            
            # Zチャネルパラメータ
            'kc_max_multiplier': trial.suggest_float('kc_max_multiplier', 2.5, 3.5, step=0.1),
            'kc_min_multiplier': trial.suggest_float('kc_min_multiplier', 1.0, 2.0, step=0.1),
            
            # 共通パラメータ
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
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
            'cycle_detector_type': 'hody_dc',
            'lp_period': 5,
            'hp_period': int(params['hp_period']),
            'cycle_part': float(params['cycle_part']),
            
            'bb_max_multiplier': float(params['bb_max_multiplier']),
            'bb_min_multiplier': float(params['bb_min_multiplier']),
            
            'kc_max_multiplier': float(params['kc_max_multiplier']),
            'kc_min_multiplier': float(params['kc_min_multiplier']),
            'kc_smoother_type': 'alma',
            
            'src_type': params['src_type'],
            'band_lookback': 1,
            
            'trade_long': True,
            'trade_short': True
        }
        return strategy_params
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        ZVチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (中心線, 上限バンド, 下限バンド)のタプル
        """
        return self.signal_generator.get_band_values(data)
    
    def get_bb_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        内部ボリンジャーバンドの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (中心線, 上限バンド, 下限バンド)のタプル
        """
        return self.signal_generator.get_bb_band_values(data)
    
    def get_kc_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        内部ケルトナーチャネル（Zチャネル）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (中心線, 上限バンド, 下限バンド)のタプル
        """
        return self.signal_generator.get_kc_band_values(data)
    
    def get_cycle_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        return self.signal_generator.get_cycle_efficiency_ratio(data) 