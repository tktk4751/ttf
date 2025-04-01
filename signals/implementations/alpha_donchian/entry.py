#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.alpha_donchian import AlphaDonchian


@jit(nopython=True)
def calculate_breakout_signals(
    close: np.ndarray,
    upper_band: np.ndarray,
    lower_band: np.ndarray
) -> np.ndarray:
    """
    ブレイクアウトシグナルを一度に計算（高速化版）
    
    Args:
        close: 終値の配列
        upper_band: 上限バンドの配列
        lower_band: 下限バンドの配列
        
    Returns:
        シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初の部分はシグナルなし（初期値はすでにゼロ）
    # 2点目以降のシグナル計算
    for i in range(1, length):
        # ロングエントリー: 終値がアッパーバンドを上回る
        if not np.isnan(upper_band[i-1]) and close[i] > upper_band[i-1]:
            signals[i] = 1
        # ショートエントリー: 終値がロワーバンドを下回る
        elif not np.isnan(lower_band[i-1]) and close[i] < lower_band[i-1]:
            signals[i] = -1
    
    return signals


class AlphaDonchianBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    アルファドンチャンのブレイクアウトによるエントリーシグナル
    
    - 現在の終値がアルファドンチャンの上限バンドを上回った場合: ロングエントリー (1)
    - 現在の終値がアルファドンチャンの下限バンドを下回った場合: ショートエントリー (-1)
    
    特徴:
    - 効率比（ER）に基づく動的な期間調整で市場環境に適応
    - 25-75パーセンタイルを使った改良されたチャネル計算
    - ハイパフォーマンスなNumba JITコンパイルによる高速化
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_donchian_period: int = 55,
        min_donchian_period: int = 13
    ):
        """
        初期化
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            max_donchian_period: ドンチャン期間の最大値（デフォルト: 55）
            min_donchian_period: ドンチャン期間の最小値（デフォルト: 13）
        """
        super().__init__("AlphaDonchianBreakoutEntrySignal")
        self.alpha_donchian = AlphaDonchian(
            er_period=er_period,
            max_donchian_period=max_donchian_period,
            min_donchian_period=min_donchian_period
        )
        
        # キャッシュ用の変数
        self._signals = None
        self._data_len = 0
        self._last_close_value = None
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # 最適化されたキャッシュ機構 - データ長とデータの最終値で判断
            current_len = len(data)
            
            # 終値の取得
            if isinstance(data, pd.DataFrame):
                close_value = data['close'].iloc[-1] if not data.empty else None
                close = data['close'].values
            else:
                close_value = data[-1, 3] if len(data) > 0 else None
                close = data[:, 3] if data.ndim == 2 else data
            
            # キャッシュチェック - 同じデータなら再計算しない
            if (self._signals is not None and current_len == self._data_len and 
                close_value == self._last_close_value):
                return self._signals
            
            # アルファドンチャンチャネルの計算
            self.alpha_donchian.calculate(data)
            upper, lower, _ = self.alpha_donchian.get_bands()
            
            # ブレイクアウトシグナルの計算（高速化版）
            self._signals = calculate_breakout_signals(close, upper, lower)
            
            # キャッシュ更新
            self._data_len = current_len
            self._last_close_value = close_value
            
            return self._signals
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"AlphaDonchianBreakoutEntrySignal生成中にエラー: {str(e)}")
            # エラー時はゼロシグナルを返す
            if data is not None:
                return np.zeros(len(data), dtype=np.int8)
            return np.array([], dtype=np.int8)
    
    def get_alpha_donchian(self) -> AlphaDonchian:
        """
        使用されているアルファドンチャンインジケーターを取得する
        
        Returns:
            AlphaDonchian: アルファドンチャンインジケーターのインスタンス
        """
        return self.alpha_donchian
    
    def get_dynamic_periods(self) -> np.ndarray:
        """
        動的期間を取得する
        
        Returns:
            np.ndarray: 動的期間の配列
        """
        if self.alpha_donchian is None or not hasattr(self.alpha_donchian, '_result') or self.alpha_donchian._result is None:
            raise RuntimeError("generate()を先に呼び出してください")
        return self.alpha_donchian.get_dynamic_period()
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比を取得する
        
        Returns:
            np.ndarray: 効率比の配列
        """
        if self.alpha_donchian is None or not hasattr(self.alpha_donchian, '_result') or self.alpha_donchian._result is None:
            raise RuntimeError("generate()を先に呼び出してください")
        return self.alpha_donchian.get_efficiency_ratio() 