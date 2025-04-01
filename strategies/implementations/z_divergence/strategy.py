#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import optuna

from strategies.base.strategy import BaseStrategy
from .signal_generator import ZDivergenceSignalGenerator


class ZDivergenceStrategy(BaseStrategy):
    """
    Zダイバージェンス戦略
    
    特徴:
    - ZMACDダイバージェンスシグナルによるトレンド転換タイミングの検出
    - Zリバーサルフィルタにより市場状態と方向性を確認
    - Zドンチャンブレイクアウトによる明確な決済ポイント
    - Numbaによる最適化で高速処理
    
    エントリー条件:
    - ロング: ZMACDダイバージェンスシグナルが1(強気ダイバージェンス) かつ Zリバーサルフィルターが1(ロングリバーサル)
    - ショート: ZMACDダイバージェンスシグナルが-1(弱気ダイバージェンス) かつ Zリバーサルフィルターが-1(ショートリバーサル)
    
    エグジット条件:
    - ロング: Zドンチャンブレイクアウトシグナルが-1(ショートブレイクアウト)に変化
    - ショート: Zドンチャンブレイクアウトシグナルが1(ロングブレイクアウト)に変化
    """
    
    def __init__(
        self,
        # ZMACDダイバージェンス用パラメータ
        er_period: int = 21,
        fast_max_dc_max_output: int = 21,
        fast_max_dc_min_output: int = 5,
        slow_max_dc_max_output: int = 55,
        slow_max_dc_min_output: int = 13,
        signal_max_dc_max_output: int = 21,
        signal_max_dc_min_output: int = 5,
        max_slow_period: int = 34,
        min_slow_period: int = 13,
        max_fast_period: int = 8,
        min_fast_period: int = 2,
        div_lookback: int = 30,
        
        # Zリバーサルフィルター用パラメータ
        # Zロングリバーサル用パラメータ
        long_max_rms_window: int = 13,
        long_min_rms_window: int = 5,
        long_max_threshold: float = 0.9,
        long_min_threshold: float = 0.75,
        
        # Zショートリバーサル用パラメータ
        short_max_rms_window: int = 13,
        short_min_rms_window: int = 5,
        short_max_threshold: float = 0.25,
        short_min_threshold: float = 0.1,
        
        # サイクル効率比(CER)のパラメーター
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 62,
        cycle_part: float = 0.5,
        
        # 組み合わせパラメータ
        zadx_weight: float = 0.4,
        zrsx_weight: float = 0.4,
        combination_method: str = "sigmoid",  # "sigmoid", "rms", "simple"
        
        # ZADX用パラメータ
        zadx_max_dc_cycle_part: float = 0.8,
        zadx_max_dc_max_cycle: int = 55,
        zadx_max_dc_min_cycle: int = 5,
        zadx_max_dc_max_output: int = 34,
        zadx_max_dc_min_output: int = 8,
        zadx_min_dc_cycle_part: float = 0.4,
        zadx_min_dc_max_cycle: int = 35,
        zadx_min_dc_min_cycle: int = 5,
        zadx_min_dc_max_output: int = 13,
        zadx_min_dc_min_output: int = 5,
        zadx_er_period: int = 21,
        
        # ZRSX用パラメータ
        zrsx_max_dc_cycle_part: float = 0.8,
        zrsx_max_dc_max_cycle: int = 55,
        zrsx_max_dc_min_cycle: int = 5,
        zrsx_max_dc_max_output: int = 34,
        zrsx_max_dc_min_output: int = 10,
        zrsx_min_dc_cycle_part: float = 0.4,
        zrsx_min_dc_max_cycle: int = 34,
        zrsx_min_dc_min_cycle: int = 3,
        zrsx_min_dc_max_output: int = 10,
        zrsx_min_dc_min_output: int = 5,
        zrsx_er_period: int = 10,
        
        # Zドンチャン用パラメータ
        # 最大期間用パラメータ
        donchian_max_dc_cycle_part: float = 0.5,
        donchian_max_dc_max_cycle: int = 144,
        donchian_max_dc_min_cycle: int = 13,
        donchian_max_dc_max_output: int = 89,
        donchian_max_dc_min_output: int = 21,
        
        # 最小期間用パラメータ
        donchian_min_dc_cycle_part: float = 0.25,
        donchian_min_dc_max_cycle: int = 55,
        donchian_min_dc_min_cycle: int = 5,
        donchian_min_dc_max_output: int = 21,
        donchian_min_dc_min_output: int = 8,
        
        # ブレイクアウトパラメータ
        band_lookback: int = 1,
        
        # 共通パラメータ
        smoother_type: str = 'alma',  # 'alma'または'hyper'
        src_type: str = 'hlc3'
    ):
        """
        コンストラクタ
        
        Args:
            各コンポーネントごとに必要なパラメータを受け取ります
        """
        super().__init__("ZDivergenceStrategy")
        
        # パラメータの設定
        self._parameters = {
            'er_period': er_period,
            'fast_max_dc_max_output': fast_max_dc_max_output,
            'fast_max_dc_min_output': fast_max_dc_min_output,
            'slow_max_dc_max_output': slow_max_dc_max_output,
            'slow_max_dc_min_output': slow_max_dc_min_output,
            'signal_max_dc_max_output': signal_max_dc_max_output,
            'signal_max_dc_min_output': signal_max_dc_min_output,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period,
            'div_lookback': div_lookback,
            'long_max_rms_window': long_max_rms_window,
            'long_min_rms_window': long_min_rms_window,
            'long_max_threshold': long_max_threshold,
            'long_min_threshold': long_min_threshold,
            'short_max_rms_window': short_max_rms_window,
            'short_min_rms_window': short_min_rms_window,
            'short_max_threshold': short_max_threshold,
            'short_min_threshold': short_min_threshold,
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'zadx_weight': zadx_weight,
            'zrsx_weight': zrsx_weight,
            'combination_method': combination_method,
            'zadx_max_dc_cycle_part': zadx_max_dc_cycle_part,
            'zadx_max_dc_max_cycle': zadx_max_dc_max_cycle,
            'zadx_max_dc_min_cycle': zadx_max_dc_min_cycle,
            'zadx_max_dc_max_output': zadx_max_dc_max_output,
            'zadx_max_dc_min_output': zadx_max_dc_min_output,
            'zadx_min_dc_cycle_part': zadx_min_dc_cycle_part,
            'zadx_min_dc_max_cycle': zadx_min_dc_max_cycle,
            'zadx_min_dc_min_cycle': zadx_min_dc_min_cycle,
            'zadx_min_dc_max_output': zadx_min_dc_max_output,
            'zadx_min_dc_min_output': zadx_min_dc_min_output,
            'zadx_er_period': zadx_er_period,
            'zrsx_max_dc_cycle_part': zrsx_max_dc_cycle_part,
            'zrsx_max_dc_max_cycle': zrsx_max_dc_max_cycle,
            'zrsx_max_dc_min_cycle': zrsx_max_dc_min_cycle,
            'zrsx_max_dc_max_output': zrsx_max_dc_max_output,
            'zrsx_max_dc_min_output': zrsx_max_dc_min_output,
            'zrsx_min_dc_cycle_part': zrsx_min_dc_cycle_part,
            'zrsx_min_dc_max_cycle': zrsx_min_dc_max_cycle,
            'zrsx_min_dc_min_cycle': zrsx_min_dc_min_cycle,
            'zrsx_min_dc_max_output': zrsx_min_dc_max_output,
            'zrsx_min_dc_min_output': zrsx_min_dc_min_output,
            'zrsx_er_period': zrsx_er_period,
            'donchian_max_dc_cycle_part': donchian_max_dc_cycle_part,
            'donchian_max_dc_max_cycle': donchian_max_dc_max_cycle,
            'donchian_max_dc_min_cycle': donchian_max_dc_min_cycle,
            'donchian_max_dc_max_output': donchian_max_dc_max_output,
            'donchian_max_dc_min_output': donchian_max_dc_min_output,
            'donchian_min_dc_cycle_part': donchian_min_dc_cycle_part,
            'donchian_min_dc_max_cycle': donchian_min_dc_max_cycle,
            'donchian_min_dc_min_cycle': donchian_min_dc_min_cycle,
            'donchian_min_dc_max_output': donchian_min_dc_max_output,
            'donchian_min_dc_min_output': donchian_min_dc_min_output,
            'band_lookback': band_lookback,
            'smoother_type': smoother_type,
            'src_type': src_type
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZDivergenceSignalGenerator(
            er_period=er_period,
            fast_max_dc_max_output=fast_max_dc_max_output,
            fast_max_dc_min_output=fast_max_dc_min_output,
            slow_max_dc_max_output=slow_max_dc_max_output,
            slow_max_dc_min_output=slow_max_dc_min_output,
            signal_max_dc_max_output=signal_max_dc_max_output,
            signal_max_dc_min_output=signal_max_dc_min_output,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period,
            div_lookback=div_lookback,
            long_max_rms_window=long_max_rms_window,
            long_min_rms_window=long_min_rms_window,
            long_max_threshold=long_max_threshold,
            long_min_threshold=long_min_threshold,
            short_max_rms_window=short_max_rms_window,
            short_min_rms_window=short_min_rms_window,
            short_max_threshold=short_max_threshold,
            short_min_threshold=short_min_threshold,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            zadx_weight=zadx_weight,
            zrsx_weight=zrsx_weight,
            combination_method=combination_method,
            zadx_max_dc_cycle_part=zadx_max_dc_cycle_part,
            zadx_max_dc_max_cycle=zadx_max_dc_max_cycle,
            zadx_max_dc_min_cycle=zadx_max_dc_min_cycle,
            zadx_max_dc_max_output=zadx_max_dc_max_output,
            zadx_max_dc_min_output=zadx_max_dc_min_output,
            zadx_min_dc_cycle_part=zadx_min_dc_cycle_part,
            zadx_min_dc_max_cycle=zadx_min_dc_max_cycle,
            zadx_min_dc_min_cycle=zadx_min_dc_min_cycle,
            zadx_min_dc_max_output=zadx_min_dc_max_output,
            zadx_min_dc_min_output=zadx_min_dc_min_output,
            zadx_er_period=zadx_er_period,
            zrsx_max_dc_cycle_part=zrsx_max_dc_cycle_part,
            zrsx_max_dc_max_cycle=zrsx_max_dc_max_cycle,
            zrsx_max_dc_min_cycle=zrsx_max_dc_min_cycle,
            zrsx_max_dc_max_output=zrsx_max_dc_max_output,
            zrsx_max_dc_min_output=zrsx_max_dc_min_output,
            zrsx_min_dc_cycle_part=zrsx_min_dc_cycle_part,
            zrsx_min_dc_max_cycle=zrsx_min_dc_max_cycle,
            zrsx_min_dc_min_cycle=zrsx_min_dc_min_cycle,
            zrsx_min_dc_max_output=zrsx_min_dc_max_output,
            zrsx_min_dc_min_output=zrsx_min_dc_min_output,
            zrsx_er_period=zrsx_er_period,
            donchian_max_dc_cycle_part=donchian_max_dc_cycle_part,
            donchian_max_dc_max_cycle=donchian_max_dc_max_cycle,
            donchian_max_dc_min_cycle=donchian_max_dc_min_cycle,
            donchian_max_dc_max_output=donchian_max_dc_max_output,
            donchian_max_dc_min_output=donchian_max_dc_min_output,
            donchian_min_dc_cycle_part=donchian_min_dc_cycle_part,
            donchian_min_dc_max_cycle=donchian_min_dc_max_cycle,
            donchian_min_dc_min_cycle=donchian_min_dc_min_cycle,
            donchian_min_dc_max_output=donchian_min_dc_max_output,
            donchian_min_dc_min_output=donchian_min_dc_min_output,
            band_lookback=band_lookback,
            smoother_type=smoother_type,
            src_type=src_type
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
            # ZMACDダイバージェンス用パラメータ
            'er_period': trial.suggest_int('er_period', 10, 30),
            'fast_max_dc_max_output': trial.suggest_int('fast_max_dc_max_output', 15, 30),
            'fast_max_dc_min_output': trial.suggest_int('fast_max_dc_min_output', 3, 8),
            'slow_max_dc_max_output': trial.suggest_int('slow_max_dc_max_output', 40, 70),
            'slow_max_dc_min_output': trial.suggest_int('slow_max_dc_min_output', 8, 20),
            'signal_max_dc_max_output': trial.suggest_int('signal_max_dc_max_output', 15, 30),
            'signal_max_dc_min_output': trial.suggest_int('signal_max_dc_min_output', 3, 8),
            'max_slow_period': trial.suggest_int('max_slow_period', 25, 50),
            'min_slow_period': trial.suggest_int('min_slow_period', 8, 20),
            'max_fast_period': trial.suggest_int('max_fast_period', 5, 12),
            'min_fast_period': trial.suggest_int('min_fast_period', 1, 3),
            'div_lookback': trial.suggest_int('div_lookback', 20, 50),
            
            # Zリバーサルフィルター用パラメータ
            'long_max_rms_window': trial.suggest_int('long_max_rms_window', 8, 20),
            'long_min_rms_window': trial.suggest_int('long_min_rms_window', 3, 8),
            'long_max_threshold': trial.suggest_float('long_max_threshold', 0.8, 0.95, step=0.05),
            'long_min_threshold': trial.suggest_float('long_min_threshold', 0.65, 0.8, step=0.05),
            'short_max_rms_window': trial.suggest_int('short_max_rms_window', 8, 20),
            'short_min_rms_window': trial.suggest_int('short_min_rms_window', 3, 8),
            'short_max_threshold': trial.suggest_float('short_max_threshold', 0.2, 0.35, step=0.05),
            'short_min_threshold': trial.suggest_float('short_min_threshold', 0.05, 0.2, step=0.05),
            
            # サイクル効率比(CER)のパラメーター
            'cycle_detector_type': trial.suggest_categorical('cycle_detector_type', ['hody_dc', 'dudi_dc', 'phac_dc']),
            'lp_period': trial.suggest_int('lp_period', 3, 8),
            'hp_period': trial.suggest_int('hp_period', 50, 90),
            'cycle_part': trial.suggest_float('cycle_part', 0.3, 0.7, step=0.1),
            
            # 組み合わせパラメータ
            'zadx_weight': trial.suggest_float('zadx_weight', 0.3, 0.6, step=0.1),
            'zrsx_weight': trial.suggest_float('zrsx_weight', 0.3, 0.6, step=0.1),
            'combination_method': trial.suggest_categorical('combination_method', ['sigmoid', 'rms', 'simple']),
            
            # Zドンチャン用パラメータ
            'donchian_max_dc_cycle_part': trial.suggest_float('donchian_max_dc_cycle_part', 0.3, 0.7, step=0.1),
            'donchian_max_dc_max_cycle': trial.suggest_int('donchian_max_dc_max_cycle', 100, 200),
            'donchian_max_dc_min_cycle': trial.suggest_int('donchian_max_dc_min_cycle', 8, 20),
            'donchian_max_dc_max_output': trial.suggest_int('donchian_max_dc_max_output', 70, 120),
            'donchian_max_dc_min_output': trial.suggest_int('donchian_max_dc_min_output', 15, 30),
            'donchian_min_dc_cycle_part': trial.suggest_float('donchian_min_dc_cycle_part', 0.15, 0.4, step=0.05),
            'donchian_min_dc_max_cycle': trial.suggest_int('donchian_min_dc_max_cycle', 40, 70),
            'donchian_min_dc_min_cycle': trial.suggest_int('donchian_min_dc_min_cycle', 3, 8),
            'donchian_min_dc_max_output': trial.suggest_int('donchian_min_dc_max_output', 15, 30),
            'donchian_min_dc_min_output': trial.suggest_int('donchian_min_dc_min_output', 5, 12),
            'band_lookback': trial.suggest_int('band_lookback', 1, 3),
            
            # 共通パラメータ
            'smoother_type': trial.suggest_categorical('smoother_type', ['alma', 'hyper']),
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
        # パラメータの型変換を行う
        strategy_params = {
            'er_period': int(params['er_period']),
            'fast_max_dc_max_output': int(params['fast_max_dc_max_output']),
            'fast_max_dc_min_output': int(params['fast_max_dc_min_output']),
            'slow_max_dc_max_output': int(params['slow_max_dc_max_output']),
            'slow_max_dc_min_output': int(params['slow_max_dc_min_output']),
            'signal_max_dc_max_output': int(params['signal_max_dc_max_output']),
            'signal_max_dc_min_output': int(params['signal_max_dc_min_output']),
            'max_slow_period': int(params['max_slow_period']),
            'min_slow_period': int(params['min_slow_period']),
            'max_fast_period': int(params['max_fast_period']),
            'min_fast_period': int(params['min_fast_period']),
            'div_lookback': int(params['div_lookback']),
            'long_max_rms_window': int(params['long_max_rms_window']),
            'long_min_rms_window': int(params['long_min_rms_window']),
            'long_max_threshold': float(params['long_max_threshold']),
            'long_min_threshold': float(params['long_min_threshold']),
            'short_max_rms_window': int(params['short_max_rms_window']),
            'short_min_rms_window': int(params['short_min_rms_window']),
            'short_max_threshold': float(params['short_max_threshold']),
            'short_min_threshold': float(params['short_min_threshold']),
            'cycle_detector_type': params['cycle_detector_type'],
            'lp_period': int(params['lp_period']),
            'hp_period': int(params['hp_period']),
            'cycle_part': float(params['cycle_part']),
            'zadx_weight': float(params['zadx_weight']),
            'zrsx_weight': float(params['zrsx_weight']),
            'combination_method': params['combination_method'],
            'donchian_max_dc_cycle_part': float(params['donchian_max_dc_cycle_part']),
            'donchian_max_dc_max_cycle': int(params['donchian_max_dc_max_cycle']),
            'donchian_max_dc_min_cycle': int(params['donchian_max_dc_min_cycle']),
            'donchian_max_dc_max_output': int(params['donchian_max_dc_max_output']),
            'donchian_max_dc_min_output': int(params['donchian_max_dc_min_output']),
            'donchian_min_dc_cycle_part': float(params['donchian_min_dc_cycle_part']),
            'donchian_min_dc_max_cycle': int(params['donchian_min_dc_max_cycle']),
            'donchian_min_dc_min_cycle': int(params['donchian_min_dc_min_cycle']),
            'donchian_min_dc_max_output': int(params['donchian_min_dc_max_output']),
            'donchian_min_dc_min_output': int(params['donchian_min_dc_min_output']),
            'band_lookback': int(params['band_lookback']),
            'smoother_type': params['smoother_type'],
            'src_type': params['src_type'],
        }
        return strategy_params 