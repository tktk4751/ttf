#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import ZBBSimpleSignalGenerator


class ZBBSimpleStrategy(BaseStrategy):
    """
    Zボリンジャーバンド戦略（シンプル版 - トレンドフィルターなし）
    
    特徴:
    - サイクル効率比（CER）に基づく動的標準偏差
    - Zボリンジャーバンドによる高精度なエントリーポイント検出
    
    エントリー条件:
    - ロング: Zボリンジャーバンドの買いシグナル
    - ショート: Zボリンジャーバンドの売りシグナル
    
    エグジット条件:
    - ロング: Zボリンジャーバンドの売りシグナル
    - ショート: Zボリンジャーバンドの買いシグナル
    """
    
    def __init__(
        self,
        # ZBBBreakoutEntrySignalのパラメータ
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_multiplier: float = 2.5,
        min_multiplier: float = 1.0,
        max_cycle_part: float = 0.5,
        max_max_cycle: int = 144,
        max_min_cycle: int = 10,
        max_max_output: int = 89,
        max_min_output: int = 13,
        min_cycle_part: float = 0.25,
        min_max_cycle: int = 55,
        min_min_cycle: int = 5,
        min_max_output: int = 21,
        min_min_output: int = 5,
        src_type: str = 'hlc3',
        lookback: int = 1,
    ):
        """
        初期化
        
        Args:
            cycle_detector_type: ZBBのサイクル検出器の種類
            lp_period: ZBBのローパスフィルター期間
            hp_period: ZBBのハイパスフィルター期間
            cycle_part: ZBBのサイクル部分倍率
            max_multiplier: ZBBの最大標準偏差乗数
            min_multiplier: ZBBの最小標準偏差乗数
            max_cycle_part: ZBBの最大標準偏差サイクル部分
            max_max_cycle: ZBBの最大標準偏差最大サイクル
            max_min_cycle: ZBBの最大標準偏差最小サイクル
            max_max_output: ZBBの最大標準偏差最大出力
            max_min_output: ZBBの最大標準偏差最小出力
            min_cycle_part: ZBBの最小標準偏差サイクル部分
            min_max_cycle: ZBBの最小標準偏差最大サイクル
            min_min_cycle: ZBBの最小標準偏差最小サイクル
            min_max_output: ZBBの最小標準偏差最大出力
            min_min_output: ZBBの最小標準偏差最小出力
            src_type: ZBBの価格ソースタイプ
            lookback: ZBBのルックバック期間
        """
        super().__init__("ZBBSimple")
        
        # パラメータの設定
        self._parameters = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'max_cycle_part': max_cycle_part,
            'max_max_cycle': max_max_cycle,
            'max_min_cycle': max_min_cycle,
            'max_max_output': max_max_output,
            'max_min_output': max_min_output,
            'min_cycle_part': min_cycle_part,
            'min_max_cycle': min_max_cycle,
            'min_min_cycle': min_min_cycle,
            'min_max_output': min_max_output,
            'min_min_output': min_min_output,
            'src_type': src_type,
            'lookback': lookback,
        }
        
        # シグナル生成器の初期化
        self.signal_generator = ZBBSimpleSignalGenerator(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            max_cycle_part=max_cycle_part,
            max_max_cycle=max_max_cycle,
            max_min_cycle=max_min_cycle,
            max_max_output=max_max_output,
            max_min_output=max_min_output,
            min_cycle_part=min_cycle_part,
            min_max_cycle=min_max_cycle,
            min_min_cycle=min_min_cycle,
            min_max_output=min_max_output,
            min_min_output=min_min_output,
            src_type=src_type,
            lookback=lookback
        )
        
        # ATRキャッシュ
        self._atr_cache = None
        self._atr_data_len = 0
        
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
                DataFrameの場合、'open', 'high', 'low', 'close'カラムが必要
                NumPy配列の場合、[open, high, low, close]形式のOHLCデータが必要
            
        Returns:
            np.ndarray: エントリーシグナル (1: ロング, -1: ショート, 0: シグナルなし)
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
            position: 現在のポジション (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            bool: エグジットすべき場合はTrue
        """
        try:
            return self.signal_generator.get_exit_signals(data, position, index)
        except Exception as e:
            self.logger.error(f"エグジットシグナル生成中にエラー: {str(e)}")
            return False
    
    def _calculate_atr(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ATR（Average True Range）を計算する
        
        Args:
            data: 価格データ
                DataFrameの場合、'open', 'high', 'low', 'close'カラムが必要
                NumPy配列の場合、[open, high, low, close]形式のOHLCデータが必要
                
        Returns:
            np.ndarray: ATR値
        """
        try:
            # キャッシュチェック
            if self._atr_cache is not None and len(data) == self._atr_data_len:
                return self._atr_cache
            
            # データの前処理
            if isinstance(data, pd.DataFrame):
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
            else:
                high = data[:, 1]
                low = data[:, 2]
                close = data[:, 3]
            
            # ATRの計算
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            tr = np.insert(tr, 0, tr1[0])  # 最初の要素を追加
            
            # 14期間の単純移動平均ATR
            period = 14
            atr = np.zeros_like(tr)
            atr[:period] = np.nan
            atr[period:] = np.convolve(tr, np.ones(period)/period, mode='valid')
            
            # キャッシュ更新
            self._atr_cache = atr
            self._atr_data_len = len(data)
            
            return atr
            
        except Exception as e:
            self.logger.error(f"ATR計算中にエラー: {str(e)}")
            return np.zeros(len(data))
    
    def get_stop_price(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> float:
        """
        ストップロス価格を計算する
        
        Args:
            data: 価格データ
            position: 現在のポジション (1: ロング, -1: ショート)
            index: データのインデックス（デフォルト: -1）
            
        Returns:
            float: ストップロス価格
        """
        try:
            # データの前処理
            if isinstance(data, pd.DataFrame):
                close = data['close'].iloc[index]
            else:
                close = data[index, 3]
            
            # ATRの計算
            atr = self._calculate_atr(data)
            current_atr = atr[index]
            
            # ATRが無効な場合はNaNを返す
            if np.isnan(current_atr):
                return np.nan
            
            # ストップロス価格を計算（ATRの2倍）
            if position == 1:  # ロング
                stop_price = close - current_atr * 2
            elif position == -1:  # ショート
                stop_price = close + current_atr * 2
            else:
                stop_price = np.nan
                
            return stop_price
            
        except Exception as e:
            self.logger.error(f"ストップロス計算中にエラー: {str(e)}")
            return np.nan
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Optunaによる最適化パラメータを生成する
        
        Args:
            trial: Optunaのトライアル
            
        Returns:
            Dict[str, Any]: 最適化パラメータ
        """
        params = {
            'hp_period': trial.suggest_int('hp_period', 89, 233),
            'max_multiplier': trial.suggest_float('max_multiplier', 2.0, 3.0, step=0.1),
            'min_multiplier': trial.suggest_float('min_multiplier', 0.8, 1.5, step=0.1),
            'max_max_output': trial.suggest_int('max_max_output', 55, 144),
            'max_min_output': trial.suggest_int('max_min_output', 13, 34),
            'min_max_output': trial.suggest_int('min_max_output', 13, 34),
            'min_min_output': trial.suggest_int('min_min_output', 5, 13),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4']),
        }
        return params
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換する
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            Dict[str, Any]: 戦略パラメータ
        """
        strategy_params = {
            'cycle_detector_type': 'hody_dc',
            'lp_period': 5,
            'hp_period': int(params['hp_period']),
            'cycle_part': 0.5,
            'max_multiplier': float(params['max_multiplier']),
            'min_multiplier': float(params['min_multiplier']),
            'max_cycle_part': 0.5,
            'max_max_cycle': 144,
            'max_min_cycle': 10,
            'max_max_output': int(params['max_max_output']),
            'max_min_output': int(params['max_min_output']),
            'min_cycle_part': 0.25,
            'min_max_cycle': 55,
            'min_min_cycle': 5,
            'min_max_output': int(params['min_max_output']),
            'min_min_output': int(params['min_min_output']),
            'src_type': params['src_type'],
            'lookback': 1,
        }
        return strategy_params 