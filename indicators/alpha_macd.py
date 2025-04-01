#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .alpha_ma import AlphaMA, calculate_alpha_ma, calculate_dynamic_kama_period, calculate_dynamic_kama_constants
from .efficiency_ratio import calculate_efficiency_ratio_for_period


@dataclass
class AlphaMACDResult:
    """AlphaMACDの計算結果"""
    macd: np.ndarray       # MACD線
    signal: np.ndarray     # シグナル線
    histogram: np.ndarray  # ヒストグラム
    er: np.ndarray         # 効率比
    dynamic_kama_period: np.ndarray  # 動的KAMAピリオド
    dynamic_fast_period: np.ndarray  # 動的Fast期間
    dynamic_slow_period: np.ndarray  # 動的Slow期間


@jit(nopython=True)
def calculate_alpha_macd(
    close: np.ndarray,
    er: np.ndarray,
    er_period: int,
    fast_kama_period: np.ndarray,
    slow_kama_period: np.ndarray,
    signal_kama_period: np.ndarray,
    fast_constants: np.ndarray,
    slow_constants: np.ndarray,
    signal_constants: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    AlphaMACDを計算する（高速化版）
    
    Args:
        close: 終値の配列
        er: 効率比の配列
        er_period: 効率比の計算期間
        fast_kama_period: 短期AlphaMAの動的期間配列
        slow_kama_period: 長期AlphaMAの動的期間配列
        signal_kama_period: シグナルAlphaMAの動的期間配列
        fast_constants: 短期AlphaMAの定数配列
        slow_constants: 長期AlphaMAの定数配列
        signal_constants: シグナルAlphaMAの定数配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: MACD線、シグナル線、ヒストグラムの配列
    """
    # 短期と長期のAlphaMAを計算
    fast_alpha_ma = calculate_alpha_ma(
        close, 
        er, 
        er_period, 
        fast_kama_period, 
        fast_constants[:, 0], 
        fast_constants[:, 1]
    )
    
    slow_alpha_ma = calculate_alpha_ma(
        close, 
        er, 
        er_period, 
        slow_kama_period, 
        slow_constants[:, 0], 
        slow_constants[:, 1]
    )
    
    # MACD線を計算
    macd_line = fast_alpha_ma - slow_alpha_ma
    
    # シグナル線を計算（AlphaMAを使用）
    signal_line = calculate_alpha_ma(
        macd_line, 
        er, 
        er_period, 
        signal_kama_period, 
        signal_constants[:, 0], 
        signal_constants[:, 1]
    )
    
    # ヒストグラムを計算
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


class AlphaMACD(Indicator):
    """
    AlphaMACDインジケーター
    
    通常のMACDのEMAをAlphaMAに置き換えたインジケーター。
    効率比（ER）に基づいて以下のパラメータを動的に調整する適応型移動平均線を使用：
    - KAMAピリオド自体
    - KAMAのfast期間
    - KAMAのslow期間
    
    特徴:
    - トレンドが強い時：短いピリオドと速い反応
    - レンジ相場時：長いピリオドとノイズ除去
    """
    
    def __init__(
        self,
        er_period: int = 21,
        fast_max_kama_period: int = 89,
        fast_min_kama_period: int = 8,
        slow_max_kama_period: int = 144,
        slow_min_kama_period: int = 21,
        signal_max_kama_period: int = 55,
        signal_min_kama_period: int = 5,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            fast_max_kama_period: 短期AlphaMAのKAMAピリオドの最大値（デフォルト: 89）
            fast_min_kama_period: 短期AlphaMAのKAMAピリオドの最小値（デフォルト: 8）
            slow_max_kama_period: 長期AlphaMAのKAMAピリオドの最大値（デフォルト: 144）
            slow_min_kama_period: 長期AlphaMAのKAMAピリオドの最小値（デフォルト: 21）
            signal_max_kama_period: シグナルAlphaMAのKAMAピリオドの最大値（デフォルト: 55）
            signal_min_kama_period: シグナルAlphaMAのKAMAピリオドの最小値（デフォルト: 5）
            max_slow_period: 遅い移動平均の最大期間（デフォルト: 89）
            min_slow_period: 遅い移動平均の最小期間（デフォルト: 30）
            max_fast_period: 速い移動平均の最大期間（デフォルト: 13）
            min_fast_period: 速い移動平均の最小期間（デフォルト: 2）
        """
        super().__init__(
            f"AlphaMACD({er_period}, {fast_max_kama_period}, {fast_min_kama_period}, "
            f"{slow_max_kama_period}, {slow_min_kama_period}, {signal_max_kama_period}, {signal_min_kama_period})"
        )
        self.er_period = er_period
        self.fast_max_kama_period = fast_max_kama_period
        self.fast_min_kama_period = fast_min_kama_period
        self.slow_max_kama_period = slow_max_kama_period
        self.slow_min_kama_period = slow_min_kama_period
        self.signal_max_kama_period = signal_max_kama_period
        self.signal_min_kama_period = signal_min_kama_period
        self.max_slow_period = max_slow_period
        self.min_slow_period = min_slow_period
        self.max_fast_period = max_fast_period
        self.min_fast_period = min_fast_period
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> AlphaMACDResult:
        """
        AlphaMACDを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            AlphaMACDの計算結果
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if 'close' not in data.columns:
                    raise ValueError("DataFrameには'close'カラムが必要です")
                close = data['close'].values
            else:
                if data.ndim == 2:
                    close = data[:, 3]  # close
                else:
                    close = data  # 1次元配列として扱う
            
            # データ長の検証
            data_length = len(close)
            self._validate_period(self.er_period, data_length)
            
            # 効率比（ER）の計算
            er = calculate_efficiency_ratio_for_period(close, self.er_period)
            
            # 動的なKAMAピリオドの計算（短期、長期、シグナル）
            fast_kama_period = calculate_dynamic_kama_period(
                er,
                self.fast_max_kama_period,
                self.fast_min_kama_period
            )
            
            slow_kama_period = calculate_dynamic_kama_period(
                er,
                self.slow_max_kama_period,
                self.slow_min_kama_period
            )
            
            signal_kama_period = calculate_dynamic_kama_period(
                er,
                self.signal_max_kama_period,
                self.signal_min_kama_period
            )
            
            # 動的なfast/slow期間の計算（短期、長期、シグナル）
            fast_periods, fast_slow_periods, fast_fast_constants, fast_slow_constants = calculate_dynamic_kama_constants(
                er,
                self.max_slow_period,
                self.min_slow_period,
                self.max_fast_period,
                self.min_fast_period
            )
            
            slow_periods, slow_slow_periods, slow_fast_constants, slow_slow_constants = calculate_dynamic_kama_constants(
                er,
                self.max_slow_period,
                self.min_slow_period,
                self.max_fast_period,
                self.min_fast_period
            )
            
            signal_periods, signal_slow_periods, signal_fast_constants, signal_slow_constants = calculate_dynamic_kama_constants(
                er,
                self.max_slow_period,
                self.min_slow_period,
                self.max_fast_period,
                self.min_fast_period
            )
            
            # 定数を2次元配列に変換
            fast_constants = np.column_stack((fast_fast_constants, fast_slow_constants))
            slow_constants = np.column_stack((slow_fast_constants, slow_slow_constants))
            signal_constants = np.column_stack((signal_fast_constants, signal_slow_constants))
            
            # AlphaMACDの計算
            macd_line, signal_line, histogram = calculate_alpha_macd(
                close,
                er,
                self.er_period,
                fast_kama_period,
                slow_kama_period,
                signal_kama_period,
                fast_constants,
                slow_constants,
                signal_constants
            )
            
            # 結果の保存
            self._result = AlphaMACDResult(
                macd=macd_line,
                signal=signal_line,
                histogram=histogram,
                er=er,
                dynamic_kama_period=fast_kama_period,  # 短期のKAMAピリオドを代表として保存
                dynamic_fast_period=fast_periods,      # 短期のFast期間を代表として保存
                dynamic_slow_period=slow_slow_periods  # 長期のSlow期間を代表として保存
            )
            
            self._values = histogram  # 基底クラスの要件を満たすため
            
            return self._result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"AlphaMACD計算中にエラー: {error_msg}\n{stack_trace}")
            # エラー時に初期化済みの配列を返す
            if 'close' in locals():
                self._values = np.zeros_like(close)
                return AlphaMACDResult(
                    macd=np.zeros_like(close),
                    signal=np.zeros_like(close),
                    histogram=np.zeros_like(close),
                    er=np.zeros_like(close),
                    dynamic_kama_period=np.zeros_like(close),
                    dynamic_fast_period=np.zeros_like(close),
                    dynamic_slow_period=np.zeros_like(close)
                )
            return AlphaMACDResult(
                macd=np.array([]),
                signal=np.array([]),
                histogram=np.array([]),
                er=np.array([]),
                dynamic_kama_period=np.array([]),
                dynamic_fast_period=np.array([]),
                dynamic_slow_period=np.array([])
            )
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        効率比（ER）の値を取得する
        
        Returns:
            np.ndarray: 効率比の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            return np.array([])
        return self._result.er
    
    def get_dynamic_periods(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        動的な期間の値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (KAMAピリオド, Fast期間, Slow期間)の値
        """
        if self._result is None:
            # 結果がない場合は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
        return (
            self._result.dynamic_kama_period,
            self._result.dynamic_fast_period,
            self._result.dynamic_slow_period
        ) 