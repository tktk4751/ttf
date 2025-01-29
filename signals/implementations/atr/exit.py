#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Tuple, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.exit import IExitSignal
from indicators.atr import ATR

class ATRExitSignal(BaseSignal, IExitSignal):
    """
    ATRベースのエグジットシグナル
    
    ロングポジション:
        - 現在価格がエントリー価格 - (ATR * 乗数) を下回ったらエグジット
    ショートポジション:
        - 現在価格がエントリー価格 + (ATR * 乗数) を上回ったらエグジット
    """
    
    def __init__(self, period: int = 14, multiplier: float = 2.0):
        """
        初期化
        
        Args:
            period: ATRの期間
            multiplier: ATRの乗数
        """
        super().__init__("ATRExitSignal", {
            'period': period,
            'multiplier': multiplier
        })
        
        self._atr = ATR(period)
        self.multiplier = multiplier
        
        # キャッシュ用の変数
        self._data_len = 0
        self._atr_values = None
        
        # エントリー情報
        self._entry_price = None
        self._position = None
    
    def set_entry_info(self, entry_price: float, position: int) -> None:
        """
        エントリー情報を設定
        
        Args:
            entry_price: エントリー価格
            position: ポジション方向 (1: ロング, -1: ショート)
        """
        self._entry_price = entry_price
        self._position = position
    
    def calculate_stop_price(self, data: Union[pd.DataFrame, np.ndarray], entry_price: float, position: int) -> float:
        """
        ストップロス価格を計算
        
        Args:
            data: 価格データ
            entry_price: エントリー価格
            position: ポジション方向 (1: ロング, -1: ショート)
            
        Returns:
            float: ストップロス価格
        """
        current_len = len(data)
        
        # データ長が変わった場合のみATRを再計算
        if self._atr_values is None or current_len != self._data_len:
            self._atr_values = self._atr.calculate(data)
            self._data_len = current_len
        
        # 最新のATR値を取得
        current_atr = self._atr_values[-1]
        
        # ポジション方向に応じてストップロス価格を計算
        if position == 1:  # ロング
            return entry_price - (current_atr * self.multiplier)
        else:  # ショート
            return entry_price + (current_atr * self.multiplier)
    
    def get_entry_and_stop_prices(self, data: Union[pd.DataFrame, np.ndarray], position: int) -> Tuple[float, float]:
        """
        エントリー価格とストップロス価格を取得
        
        Args:
            data: 価格データ
            position: ポジション方向 (1: ロング, -1: ショート)
            
        Returns:
            Tuple[float, float]: (エントリー価格, ストップロス価格)
        """
        # 最新の終値をエントリー価格とする
        entry_price = data['close'].iloc[-1] if isinstance(data, pd.DataFrame) else data[-1, 3]
        
        # ストップロス価格を計算
        stop_price = self.calculate_stop_price(data, entry_price, position)
        
        return entry_price, stop_price
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エグジットシグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: エグジットシグナル
        """
        if self._entry_price is None or self._position is None:
            raise ValueError("エントリー情報が設定されていません。set_entry_infoを呼び出してください。")
        
        current_len = len(data)
        
        # データ長が変わった場合のみATRを再計算
        if self._atr_values is None or current_len != self._data_len:
            self._atr_values = self._atr.calculate(data)
            self._data_len = current_len
        
        # 価格データを取得
        if isinstance(data, pd.DataFrame):
            prices = data['close'].values
        else:
            prices = data[:, 3]  # close価格のインデックスは3
        
        # シグナル配列を初期化
        signals = np.zeros(len(prices))
        
        # ATRベースのストップロス価格を計算
        stop_prices = self._entry_price + (self._atr_values * self.multiplier * (-self._position))
        
        # ポジション方向に応じてエグジットシグナルを生成
        if self._position == 1:  # ロング
            signals[prices < stop_prices] = 1
        else:  # ショート
            signals[prices > stop_prices] = -1
        
        return signals 