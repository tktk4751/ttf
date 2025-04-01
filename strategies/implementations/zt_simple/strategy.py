#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZTSimpleSignalGenerator


class ZTSimpleStrategy(BaseStrategy):
    """
    ZTSIMPLEストラテジー
    
    このストラテジーはZトレンドブレイクアウトシグナルを使用したシンプルな取引戦略です。
    
    特徴:
    - Zトレンドのブレイクアウトを利用したエントリーと決済
    - Numbaによる高速化
    - サイクル効率比(CER)を利用した動的パラメータ調整
    - Optunaによる全パラメータの最適化対応
    
    エントリー条件:
    - シグナルが1の場合: ロングエントリー
    - シグナルが-1の場合: ロング決済、ショートエントリー
    
    決済条件:
    - シグナルが-1の場合: ロング決済
    - シグナルが1の場合: ショート決済
    """
    
    def __init__(
        self,
        # Zトレンドのパラメータ
        cycle_detector_type: str = 'dudi_e',
        lp_period: int = 5,
        hp_period: int = 55,
        cycle_part: float = 0.5,
        band_lookback: int = 1,
        
        # CERのドミナントサイクル検出器用パラメータ
        cer_max_cycle: int = 233,
        cer_min_cycle: int = 13,
        cer_max_output: int = 144,
        cer_min_output: int = 21,
        
        # 最大パーセンタイル期間用（長期）ドミナントサイクル検出器のパラメータ
        max_percentile_dc_cycle_part: float = 0.5,
        max_percentile_dc_max_cycle: int = 233,
        max_percentile_dc_min_cycle: int = 13,
        max_percentile_dc_max_output: int = 144,
        max_percentile_dc_min_output: int = 21,
        
        # 最小パーセンタイル期間用（短期）ドミナントサイクル検出器のパラメータ
        min_percentile_dc_cycle_part: float = 0.5,
        min_percentile_dc_max_cycle: int = 55,
        min_percentile_dc_min_cycle: int = 5,
        min_percentile_dc_max_output: int = 34,
        min_percentile_dc_min_output: int = 8,
        
        # ZATR用ドミナントサイクル検出器のパラメータ
        zatr_max_dc_cycle_part: float = 0.5,
        zatr_max_dc_max_cycle: int = 55,
        zatr_max_dc_min_cycle: int = 5,
        zatr_max_dc_max_output: int = 55,
        zatr_max_dc_min_output: int = 5,
        zatr_min_dc_cycle_part: float = 0.25,
        zatr_min_dc_max_cycle: int = 34,
        zatr_min_dc_min_cycle: int = 3,
        zatr_min_dc_max_output: int = 13,
        zatr_min_dc_min_output: int = 3,
        
        # パーセンタイル乗数
        max_percentile_cycle_mult: float = 0.5,  # 最大パーセンタイル期間のサイクル乗数
        min_percentile_cycle_mult: float = 0.25,  # 最小パーセンタイル期間のサイクル乗数
        
        # 動的乗数の範囲
        max_max_multiplier: float = 5.0,    # 最大乗数の最大値
        min_max_multiplier: float = 2.5,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        
        # その他の設定
        smoother_type: str = 'alma',   # 平滑化アルゴリズム（'alma'または'hyper'）
        src_type: str = 'hlc3'
    ):
        """初期化"""
        super().__init__("ZTSimple")
        
        # パラメータの設定
        self._parameters = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'band_lookback': band_lookback,
            
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
            
            'max_max_multiplier': max_max_multiplier,
            'min_max_multiplier': min_max_multiplier,
            'max_min_multiplier': max_min_multiplier,
            'min_min_multiplier': min_min_multiplier,
            
            'smoother_type': smoother_type,
            'src_type': src_type
        }
        
        # シグナルジェネレータの初期化
        self.signal_generator = ZTSimpleSignalGenerator(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            band_lookback=band_lookback,
            
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
            
            max_max_multiplier=max_max_multiplier,
            min_max_multiplier=min_max_multiplier,
            max_min_multiplier=max_min_multiplier,
            min_min_multiplier=min_min_multiplier,
            
            smoother_type=smoother_type,
            src_type=src_type
        )
    
    def generate_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナルの生成"""
        try:
            return self.signal_generator.get_entry_signals(data)
        except Exception as e:
            self.logger.error(f"エントリーシグナル生成中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        BaseStrategyが期待するエントリーシグナル生成メソッド
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル配列
        """
        return self.generate_entry_signals(data)
    
    def should_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジット判定"""
        try:
            return self.signal_generator.get_exit_signals(data, position, index)
        except Exception as e:
            self.logger.error(f"エグジット判定中にエラー: {str(e)}")
            return False
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        BaseStrategyが期待するエグジットシグナル生成メソッド
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: チェックするインデックス（デフォルトは最新のデータポイント）
            
        Returns:
            bool: エグジットすべきかどうか
        """
        return self.should_exit(data, position, index)
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータの取得
        
        Args:
            trial: Optunaのtrialオブジェクト
            
        Returns:
            Dict[str, Any]: パラメータ名とその設定範囲を含む辞書
        """
        params = {
            # 基本パラメータ
            'cycle_detector_type': trial.suggest_categorical('cycle_detector_type', ['dudi_dc', 'hody_dc', 'phac_dc']),
            'lp_period': trial.suggest_int('lp_period', 3, 21),
            'hp_period': trial.suggest_int('hp_period', 34, 233),
            'cycle_part': trial.suggest_float('cycle_part', 0.2, 0.9, step=0.1),
            'band_lookback': trial.suggest_int('band_lookback', 1, 5),
            
            # CERドミナントサイクル検出器用パラメータ
            'cer_max_cycle': trial.suggest_int('cer_max_cycle', 89, 377),
            'cer_min_cycle': trial.suggest_int('cer_min_cycle', 8, 21),
            'cer_max_output': trial.suggest_int('cer_max_output', 89, 233),
            'cer_min_output': trial.suggest_int('cer_min_output', 13, 34),
            
            # 最大パーセンタイル期間用（長期）ドミナントサイクル検出器のパラメータ
            'max_percentile_dc_cycle_part': trial.suggest_float('max_percentile_dc_cycle_part', 0.3, 0.7, step=0.1),
            'max_percentile_dc_max_cycle': trial.suggest_int('max_percentile_dc_max_cycle', 144, 377),
            'max_percentile_dc_min_cycle': trial.suggest_int('max_percentile_dc_min_cycle', 8, 21),
            'max_percentile_dc_max_output': trial.suggest_int('max_percentile_dc_max_output', 89, 233),
            'max_percentile_dc_min_output': trial.suggest_int('max_percentile_dc_min_output', 13, 34),
            
            # 最小パーセンタイル期間用（短期）ドミナントサイクル検出器のパラメータ
            'min_percentile_dc_cycle_part': trial.suggest_float('min_percentile_dc_cycle_part', 0.2, 0.6, step=0.1),
            'min_percentile_dc_max_cycle': trial.suggest_int('min_percentile_dc_max_cycle', 34, 89),
            'min_percentile_dc_min_cycle': trial.suggest_int('min_percentile_dc_min_cycle', 3, 13),
            'min_percentile_dc_max_output': trial.suggest_int('min_percentile_dc_max_output', 21, 55),
            'min_percentile_dc_min_output': trial.suggest_int('min_percentile_dc_min_output', 5, 13),
            
            # ZATR用ドミナントサイクル検出器のパラメータ
            'zatr_max_dc_cycle_part': trial.suggest_float('zatr_max_dc_cycle_part', 0.3, 0.7, step=0.1),
            'zatr_max_dc_max_cycle': trial.suggest_int('zatr_max_dc_max_cycle', 34, 89),
            'zatr_max_dc_min_cycle': trial.suggest_int('zatr_max_dc_min_cycle', 3, 13),
            'zatr_max_dc_max_output': trial.suggest_int('zatr_max_dc_max_output', 34, 55),
            'zatr_max_dc_min_output': trial.suggest_int('zatr_max_dc_min_output', 3, 13),
            
            'zatr_min_dc_cycle_part': trial.suggest_float('zatr_min_dc_cycle_part', 0.1, 0.5, step=0.1),
            'zatr_min_dc_max_cycle': trial.suggest_int('zatr_min_dc_max_cycle', 21, 55),
            'zatr_min_dc_min_cycle': trial.suggest_int('zatr_min_dc_min_cycle', 2, 5),
            'zatr_min_dc_max_output': trial.suggest_int('zatr_min_dc_max_output', 8, 21),
            'zatr_min_dc_min_output': trial.suggest_int('zatr_min_dc_min_output', 2, 5),
            
            # パーセンタイル乗数
            'max_percentile_cycle_mult': trial.suggest_float('max_percentile_cycle_mult', 0.3, 1.0, step=0.1),
            'min_percentile_cycle_mult': trial.suggest_float('min_percentile_cycle_mult', 0.1, 0.5, step=0.1),
            
            # 動的乗数の範囲
            'max_max_multiplier': trial.suggest_float('max_max_multiplier', 3.0, 7.0, step=0.5),
            'min_max_multiplier': trial.suggest_float('min_max_multiplier', 1.5, 3.5, step=0.5),
            'max_min_multiplier': trial.suggest_float('max_min_multiplier', 1.0, 2.0, step=0.1),
            'min_min_multiplier': trial.suggest_float('min_min_multiplier', 0.3, 0.8, step=0.1),
            
            # その他の設定
            'smoother_type': trial.suggest_categorical('smoother_type', ['alma', 'hyper']),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4'])
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化結果からストラテジーパラメータへの変換
        
        Args:
            params: 最適化されたパラメータ
            
        Returns:
            Dict[str, Any]: ストラテジー用パラメータ
        """
        strategy_params = {
            # 基本パラメータ
            'cycle_detector_type': params['cycle_detector_type'],
            'lp_period': int(params['lp_period']),
            'hp_period': int(params['hp_period']),
            'cycle_part': float(params['cycle_part']),
            'band_lookback': int(params['band_lookback']),
            
            # CERドミナントサイクル検出器用パラメータ
            'cer_max_cycle': int(params['cer_max_cycle']),
            'cer_min_cycle': int(params['cer_min_cycle']),
            'cer_max_output': int(params['cer_max_output']),
            'cer_min_output': int(params['cer_min_output']),
            
            # 最大パーセンタイル期間用（長期）ドミナントサイクル検出器のパラメータ
            'max_percentile_dc_cycle_part': float(params['max_percentile_dc_cycle_part']),
            'max_percentile_dc_max_cycle': int(params['max_percentile_dc_max_cycle']),
            'max_percentile_dc_min_cycle': int(params['max_percentile_dc_min_cycle']),
            'max_percentile_dc_max_output': int(params['max_percentile_dc_max_output']),
            'max_percentile_dc_min_output': int(params['max_percentile_dc_min_output']),
            
            # 最小パーセンタイル期間用（短期）ドミナントサイクル検出器のパラメータ
            'min_percentile_dc_cycle_part': float(params['min_percentile_dc_cycle_part']),
            'min_percentile_dc_max_cycle': int(params['min_percentile_dc_max_cycle']),
            'min_percentile_dc_min_cycle': int(params['min_percentile_dc_min_cycle']),
            'min_percentile_dc_max_output': int(params['min_percentile_dc_max_output']),
            'min_percentile_dc_min_output': int(params['min_percentile_dc_min_output']),
            
            # ZATR用ドミナントサイクル検出器のパラメータ
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
            
            # パーセンタイル乗数
            'max_percentile_cycle_mult': float(params['max_percentile_cycle_mult']),
            'min_percentile_cycle_mult': float(params['min_percentile_cycle_mult']),
            
            # 動的乗数の範囲
            'max_max_multiplier': float(params['max_max_multiplier']),
            'min_max_multiplier': float(params['min_max_multiplier']),
            'max_min_multiplier': float(params['max_min_multiplier']),
            'min_min_multiplier': float(params['min_min_multiplier']),
            
            # その他の設定
            'smoother_type': params['smoother_type'],
            'src_type': params['src_type']
        }
        return strategy_params
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        バンド値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (上限バンド, 下限バンド)のタプル
        """
        return self.signal_generator.get_bands(data)
    
    def get_trend(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド方向を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: トレンド方向の配列
        """
        return self.signal_generator.get_trend(data)
    
    def get_percentiles(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        パーセンタイル値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (上側パーセンタイル, 下側パーセンタイル)のタプル
        """
        return self.signal_generator.get_percentiles(data)
    
    def get_cycle_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        return self.signal_generator.get_cycle_er(data)
    
    def get_dynamic_parameters(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        動的パラメータの値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (動的乗数, 動的パーセンタイル期間)のタプル
        """
        return self.signal_generator.get_dynamic_parameters(data)
    
    def get_z_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        ZATRの値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: ZATRの値
        """
        return self.signal_generator.get_z_atr(data)
    
    def get_dominant_cycles(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        ドミナントサイクルの値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (最大パーセンタイル期間用ドミナントサイクル, 最小パーセンタイル期間用ドミナントサイクル)のタプル
        """
        return self.signal_generator.get_dominant_cycles(data) 