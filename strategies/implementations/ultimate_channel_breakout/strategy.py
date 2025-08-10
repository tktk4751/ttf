#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import UltimateChannelBreakoutSignalGenerator


class UltimateChannelBreakoutStrategy(BaseStrategy):
    """
    Ultimate Channel Breakoutストラテジー
    
    特徴:
    - John Ehlersのアルティメットチャネル技術によるブレイクアウト検出
    - 動的適応乗数対応（UQATRDベース）
    - 線形補間による滑らかなチャネル調整
    - 高精度なブレイクアウト検出
    - 低遅延フィルタリング技術
    
    エントリー条件:
    - ロング: 前回終値が前回の上部チャネル以下かつ現在終値が現在の上部チャネル以上
    - ショート: 前回終値が前回の下部チャネル以上かつ現在終値が現在の下部チャネル以下
    
    エグジット条件:
    - ロング決済: 終値が下部チャネル以下
    - ショート決済: 終値が上部チャネル以上
    """
    
    def __init__(
        self,
        # 基本パラメータ
        channel_lookback: int = 1,
        
        # アルティメットチャネルのパラメータ
        length: float = 55.0,
        num_strs: float = 3,
        multiplier_mode: str = 'dynamic',
        src_type: str = 'hlc3',
        # 追加のアルティメットチャネルパラメータ
        ultimate_channel_params: Dict[str, Any] = None
    ):
        """
        初期化
        
        Args:
            channel_lookback: 過去チャネル参照期間（デフォルト: 1）
            length: アルティメットチャネルの期間（デフォルト: 20.0）
            num_strs: アルティメットチャネルのストランド数（デフォルト: 2.0）
            multiplier_mode: 乗数モード（'fixed' or 'dynamic'、デフォルト: 'fixed'）
            src_type: 価格ソース（'close', 'hlc3', 'hl2', 'ohlc4'など、デフォルト: 'hlc3'）
            ultimate_channel_params: その他のアルティメットチャネルパラメータ（オプション）
        """
        super().__init__("UltimateChannelBreakout")
        
        # パラメータの設定
        self._parameters = {
            'channel_lookback': channel_lookback,
            'length': length,
            'num_strs': num_strs,
            'multiplier_mode': multiplier_mode,
            'src_type': src_type,
            **(ultimate_channel_params or {})
        }
        
        # シグナル生成器の初期化
        # channel_lookbackを除外したパラメータを作成
        filtered_ultimate_channel_params = {k: v for k, v in (ultimate_channel_params or {}).items() if k != 'channel_lookback'}
        self.signal_generator = UltimateChannelBreakoutSignalGenerator(
            channel_lookback=channel_lookback,
            length=length,
            num_strs=num_strs,
            multiplier_mode=multiplier_mode,
            src_type=src_type,
            ultimate_channel_params=filtered_ultimate_channel_params
        )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エントリーシグナル（1=ロング、-1=ショート、0=シグナルなし）
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
    
    def get_channel_values(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        アルティメットチャネルのチャネル値を取得
        
        Args:
            data: 価格データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上部チャネル, 下部チャネル)のタプル
        """
        try:
            return self.signal_generator.get_channel_values(data)
        except Exception as e:
            self.logger.error(f"チャネル値取得中にエラー: {str(e)}")
            return np.array([]), np.array([]), np.array([])
    
    def get_dynamic_multipliers(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        動的乗数の値を取得する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        try:
            return self.signal_generator.get_dynamic_multipliers(data)
        except Exception as e:
            self.logger.error(f"動的乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_uqatrd_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        UQATRD値の値を取得する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: UQATRD値
        """
        try:
            return self.signal_generator.get_uqatrd_values(data)
        except Exception as e:
            self.logger.error(f"UQATRD値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_multiplier_mode(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        乗数モードを取得する
        
        Args:
            data: 価格データ
            
        Returns:
            str: 乗数モード ('fixed' or 'dynamic')
        """
        try:
            return self.signal_generator.get_multiplier_mode(data)
        except Exception as e:
            self.logger.error(f"乗数モード取得中にエラー: {str(e)}")
            return 'fixed'
    
    def get_channel_statistics(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict:
        """
        チャネル統計情報を取得する
        
        Args:
            data: 価格データ
            
        Returns:
            Dict: チャネル統計情報
        """
        try:
            return self.signal_generator.get_channel_statistics(data)
        except Exception as e:
            self.logger.error(f"チャネル統計取得中にエラー: {str(e)}")
            return {"status": "no_data"}
    
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
            'channel_lookback': trial.suggest_int('channel_lookback', 1, 5),
            
            # アルティメットチャネルパラメータ
            'length': trial.suggest_float('length', 10.0, 50.0, step=1.0),
            'num_strs': trial.suggest_float('num_strs', 1.0, 5.0, step=0.5),
            'multiplier_mode': trial.suggest_categorical('multiplier_mode', ['fixed', 'dynamic']),
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
            'channel_lookback': int(params['channel_lookback']),
            'length': float(params['length']),
            'num_strs': float(params['num_strs']),
            'multiplier_mode': params['multiplier_mode'],
            'src_type': params['src_type'],
        }
        return strategy_params 