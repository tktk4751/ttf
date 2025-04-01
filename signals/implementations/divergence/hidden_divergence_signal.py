#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, List, Callable, Optional
import numpy as np
import pandas as pd
from numba import njit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal


class HiddenDivergenceSignal(BaseSignal, IEntrySignal):
    """
    隠れダイバージェンスシグナルベースクラス

    価格とインジケーターの間の隠れダイバージェンスを検出します。

    隠れダイバージェンスは通常のダイバージェンスとは異なり、トレンドの継続を示唆します：
    
    - 強気の隠れダイバージェンス（ロングエントリー）：
      価格が高値を切り上げているのに対し、インジケーターが高値を切り下げている状態で、
      現在の上昇トレンドの調整局面で発生することが多く、トレンド継続の可能性を示唆します。
      
    - 弱気の隠れダイバージェンス（ショートエントリー）：
      価格が安値を切り下げているのに対し、インジケーターが安値を切り上げている状態で、
      現在の下降トレンドの戻り局面で発生することが多く、トレンド継続の可能性を示唆します。
    """
    
    def __init__(
        self,
        lookback: int = 14,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        コンストラクタ
        
        Args:
            lookback: ダイバージェンス検出のためのルックバック期間
            params: その他の設定パラメータ
        """
        super().__init__("HiddenDivergenceSignal")
        
        self.lookback = lookback
        self.params = params if params is not None else {}
        
        # キャッシュ
        self._cached_data_hash = None
        self._cached_signals = None
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成します
        
        Args:
            data: 価格データ（OHLCV）
            
        Returns:
            np.ndarray: シグナル配列
        """
        # キャッシュキーの生成（データハッシュ）
        data_hash = hash(str(data))
        
        # キャッシュヒットの場合、キャッシュしたシグナルを返す
        if self._cached_data_hash == data_hash and self._cached_signals is not None:
            return self._cached_signals
        
        try:
            # データフレームの場合はnumpy配列に変換
            if isinstance(data, pd.DataFrame):
                ohlcv = data[['open', 'high', 'low', 'close', 'volume']].values
            else:
                ohlcv = data
                
            # 価格は終値を使用
            prices = ohlcv[:, 3]
            
            # インジケーター値の取得（サブクラスで実装）
            indicator_values = self._get_indicator_values(data)
            
            # 隠れダイバージェンスの検出
            signals = self._detect_hidden_divergences(prices, indicator_values)
            
            # キャッシュの更新
            self._cached_data_hash = data_hash
            self._cached_signals = signals
            
            return signals
            
        except Exception as e:
            self.logger.error(f"隠れダイバージェンスシグナル生成中にエラー: {str(e)}")
            # エラー時は0埋めシグナルを返す
            return np.zeros(len(data), dtype=np.int8)
    
    def _get_indicator_values(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        インジケーター値を取得します（サブクラスで実装）
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: インジケーター値
        """
        raise NotImplementedError("サブクラスで実装する必要があります")
    
    def _detect_hidden_divergences(self, prices: np.ndarray, indicator_values: np.ndarray) -> np.ndarray:
        """
        隠れダイバージェンスを検出します
        
        Args:
            prices: 価格データ（通常は終値）
            indicator_values: インジケーター値
            
        Returns:
            np.ndarray: シグナル配列（1: ロングシグナル、-1: ショートシグナル、0: シグナルなし）
        """
        if len(prices) <= self.lookback:
            return np.zeros(len(prices), dtype=np.int8)
        
        # シグナル配列の初期化
        signals = np.zeros(len(prices), dtype=np.int8)
        
        # 前処理として、NaNをnp.nanで埋める
        indicator_values = np.array(indicator_values, dtype=np.float64)
        prices = np.array(prices, dtype=np.float64)
        
        # ローカル極値を見つける
        for i in range(self.lookback, len(prices) - 1):
            # 過去のルックバック期間のデータを取得
            past_prices = prices[i - self.lookback:i]
            past_indicator = indicator_values[i - self.lookback:i]
            
            # 強気の隠れダイバージェンスの検出（価格が高値を切り上げ、インジケーターが高値を切り下げ）
            if self._is_bullish_hidden_divergence(past_prices, past_indicator, prices[i], indicator_values[i]):
                signals[i] = 1
            
            # 弱気の隠れダイバージェンスの検出（価格が安値を切り下げ、インジケーターが安値を切り上げ）
            elif self._is_bearish_hidden_divergence(past_prices, past_indicator, prices[i], indicator_values[i]):
                signals[i] = -1
        
        return signals
    
    def _is_bullish_hidden_divergence(
        self,
        past_prices: np.ndarray,
        past_indicator: np.ndarray,
        current_price: float,
        current_indicator: float
    ) -> bool:
        """
        強気の隠れダイバージェンスを検出します
        
        Args:
            past_prices: 過去の価格データ
            past_indicator: 過去のインジケーター値
            current_price: 現在の価格
            current_indicator: 現在のインジケーター値
            
        Returns:
            bool: 強気の隠れダイバージェンスが検出された場合はTrue
        """
        # 過去の極大値のインデックスを見つける
        max_price_idx = np.nanargmax(past_prices)
        max_indicator_idx = np.nanargmax(past_indicator)
        
        if max_price_idx >= 0 and max_indicator_idx >= 0:
            # 隠れダイバージェンスの条件：
            # 1. 価格が高値を切り上げている（現在の価格が過去の高値より高い）
            # 2. インジケーターが高値を切り下げている（現在のインジケーター値が過去の高値より低い）
            return (current_price > past_prices[max_price_idx] and 
                   current_indicator < past_indicator[max_indicator_idx])
        
        return False
    
    def _is_bearish_hidden_divergence(
        self,
        past_prices: np.ndarray,
        past_indicator: np.ndarray,
        current_price: float,
        current_indicator: float
    ) -> bool:
        """
        弱気の隠れダイバージェンスを検出します
        
        Args:
            past_prices: 過去の価格データ
            past_indicator: 過去のインジケーター値
            current_price: 現在の価格
            current_indicator: 現在のインジケーター値
            
        Returns:
            bool: 弱気の隠れダイバージェンスが検出された場合はTrue
        """
        # 過去の極小値のインデックスを見つける
        min_price_idx = np.nanargmin(past_prices)
        min_indicator_idx = np.nanargmin(past_indicator)
        
        if min_price_idx >= 0 and min_indicator_idx >= 0:
            # 隠れダイバージェンスの条件：
            # 1. 価格が安値を切り下げている（現在の価格が過去の安値より低い）
            # 2. インジケーターが安値を切り上げている（現在のインジケーター値が過去の安値より高い）
            return (current_price < past_prices[min_price_idx] and 
                   current_indicator > past_indicator[min_indicator_idx])
        
        return False 