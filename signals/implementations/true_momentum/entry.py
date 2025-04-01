#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.true_momentum import TrueMomentum


@jit(nopython=True)
def calculate_entry_signals(momentum: np.ndarray, sqz_on: np.ndarray, threshold: float, period: int) -> np.ndarray:
    """
    トゥルーモメンタムのエントリーシグナルを計算（高速化版）
    
    Args:
        momentum: モメンタム値の配列
        sqz_on: スクイーズオン状態の配列（True/False）
        threshold: モメンタムの閾値
        period: 初期期間（シグナルを生成しない期間）
    
    Returns:
        シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
    """
    length = len(momentum)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初の期間はシグナルなし
    for i in range(period, length):
        if sqz_on[i]:  # スクイーズオン状態
            # 買いモメンタム
            if momentum[i] > threshold and momentum[i-1] <= threshold:
                signals[i] = 1
            # 売りモメンタム
            elif momentum[i] < -threshold and momentum[i-1] >= -threshold:
                signals[i] = -1
    
    return signals


class TrueMomentumEntrySignal(BaseSignal, IEntrySignal):
    """
    トゥルーモメンタムによるエントリーシグナル
    
    - スクイーズオン状態でモメンタムが買いのときに1、売りのときに-1を出力します。
    """
    
    def __init__(
        self,
        period: int = 20,
        max_std_mult: float = 2.0,
        min_std_mult: float = 1.0,
        max_kama_slow: int = 55,
        min_kama_slow: int = 30,
        max_kama_fast: int = 13,
        min_kama_fast: int = 2,
        max_atr_period: int = 120,
        min_atr_period: int = 13,
        max_atr_mult: float = 3.0,
        min_atr_mult: float = 1.0,
        max_momentum_period: int = 100,
        min_momentum_period: int = 20,
        momentum_threshold: float = 0.0
    ):
        """
        コンストラクタ
        
        Args:
            period: 計算期間（KAMA、ボリンジャーバンド、トレンドアルファに共通）（デフォルト: 20）
            max_std_mult: 標準偏差乗数の最大値（デフォルト: 2.0）
            min_std_mult: 標準偏差乗数の最小値（デフォルト: 1.0）
            max_kama_slow: KAMAの遅い移動平均の最大期間（デフォルト: 55）
            min_kama_slow: KAMAの遅い移動平均の最小期間（デフォルト: 30）
            max_kama_fast: KAMAの速い移動平均の最大期間（デフォルト: 13）
            min_kama_fast: KAMAの速い移動平均の最小期間（デフォルト: 2）
            max_atr_period: ATR期間の最大値（デフォルト: 120）
            min_atr_period: ATR期間の最小値（デフォルト: 13）
            max_atr_mult: ATR乗数の最大値（デフォルト: 3.0）
            min_atr_mult: ATR乗数の最小値（デフォルト: 1.0）
            max_momentum_period: モメンタム計算の最大期間（デフォルト: 100）
            min_momentum_period: モメンタム計算の最小期間（デフォルト: 20）
            momentum_threshold: モメンタムの閾値（デフォルト: 0.0）
        """
        params = {
            'period': period,
            'max_std_mult': max_std_mult,
            'min_std_mult': min_std_mult,
            'max_kama_slow': max_kama_slow,
            'min_kama_slow': min_kama_slow,
            'max_kama_fast': max_kama_fast,
            'min_kama_fast': min_kama_fast,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_atr_mult': max_atr_mult,
            'min_atr_mult': min_atr_mult,
            'max_momentum_period': max_momentum_period,
            'min_momentum_period': min_momentum_period,
            'momentum_threshold': momentum_threshold
        }
        super().__init__(f"TrueMomentum({period})", params)
        
        self._indicator = TrueMomentum(
            period=period,
            max_std_mult=max_std_mult,
            min_std_mult=min_std_mult,
            max_kama_slow=max_kama_slow,
            min_kama_slow=min_kama_slow,
            max_kama_fast=max_kama_fast,
            min_kama_fast=min_kama_fast,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_atr_mult=max_atr_mult,
            min_atr_mult=min_atr_mult,
            max_momentum_period=max_momentum_period,
            min_momentum_period=min_momentum_period
        )
        
        self._momentum_threshold = momentum_threshold
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # トゥルーモメンタムの計算
            momentum = self._indicator.calculate(data)
            
            # スクイーズ状態を取得
            sqz_on, _, _ = self._indicator.get_squeeze_states()
            
            # シグナルの計算（高速化）
            period = self._params['period']
            threshold = self._momentum_threshold
            
            # Numba最適化関数を呼び出し
            return calculate_entry_signals(momentum, sqz_on, threshold, period)
            
        except Exception as e:
            print(f"エントリーシグナル生成中にエラーが発生しました: {e}")
            # エラーが発生した場合は0埋めのシグナルを返す
            return np.zeros(len(data)) 