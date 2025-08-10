#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import optuna

from ...base.strategy import BaseStrategy
from .signal_generator import MAMASignalGenerator


class MAMAStrategy(BaseStrategy):
    """
    MAMAストラテジー
    
    特徴:
    - MAMA (Mother of Adaptive Moving Average) / FAMA (Following Adaptive Moving Average)
    - 市場のサイクルに応じて自動的に期間を調整する適応型移動平均線
    - Ehlers's MESA (Maximum Entropy Spectrum Analysis) アルゴリズムベース
    - トレンド強度に応じて応答速度を調整
    
    エントリー条件:
    - ロング: MAMA > FAMA
    - ショート: MAMA < FAMA
    
    エグジット条件:
    - ロング: MAMA < FAMA
    - ショート: MAMA > FAMA
    """
    
    def __init__(
        self,
        # MAMAパラメータ
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.08,              # 低速制限値
        src_type: str = 'close',                # ソースタイプ
        ukf_params: Optional[Dict] = None      # UKFパラメータ
    ):
        """
        初期化
        
        Args:
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
                UKFソース: 'ukf', 'ukf_close', 'ukf_hlc3', 'ukf_hl2', 'ukf_ohlc4'
            ukf_params: UKFパラメータ（UKFソース使用時のオプション）
        """
        super().__init__("MAMA")
        
        # パラメータの設定
        self._parameters = {
            'fast_limit': fast_limit,
            'slow_limit': slow_limit,
            'src_type': src_type,
            'ukf_params': ukf_params
        }
        
        # シグナル生成器の初期化
        self.signal_generator = MAMASignalGenerator(
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            ukf_params=ukf_params
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
            # MAMAパラメータ
            'fast_limit': trial.suggest_float('fast_limit', 0.1, 0.9, step=0.1),
            'slow_limit': trial.suggest_float('slow_limit', 0.01, 0.1, step=0.01),
            'src_type': trial.suggest_categorical('src_type', ['close', 'hlc3', 'hl2', 'ohlc4'])
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
            'fast_limit': float(params['fast_limit']),
            'slow_limit': float(params['slow_limit']),
            'src_type': params['src_type']
        }
        return strategy_params