#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
import logging
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.alpha_ma import AlphaMA


# ロガーの設定
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@jit(nopython=True)
def calculate_crossover_signal(short_ma: np.ndarray, long_ma: np.ndarray) -> np.ndarray:
    """
    クロスオーバーシグナルを計算する（高速化版）
    
    Args:
        short_ma: 短期移動平均の配列
        long_ma: 長期移動平均の配列
    
    Returns:
        シグナルの配列 (1: ロングエントリー, -1: ショートエントリー, 0: シグナルなし)
    """
    length = len(short_ma)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初の要素はクロスの判定ができないのでシグナルなし
    for i in range(1, length):
        # 前日のクロス状態と当日のクロス状態を比較
        prev_short = short_ma[i-1]
        prev_long = long_ma[i-1]
        curr_short = short_ma[i]
        curr_long = long_ma[i]
        
        # ゴールデンクロス（短期が長期を上抜け）
        if prev_short <= prev_long and curr_short > curr_long:
            signals[i] = 1  # ロングエントリー
        
        # デッドクロス（短期が長期を下抜け）
        elif prev_short >= prev_long and curr_short < curr_long:
            signals[i] = -1  # ショートエントリー
        
        # 既存のシグナルを維持
        else:
            # 短期が長期より上にある場合はロングシグナル継続
            if curr_short > curr_long:
                signals[i] = 1
            # 短期が長期より下にある場合はショートシグナル継続
            elif curr_short < curr_long:
                signals[i] = -1
    
    return signals


@jit(nopython=True)
def calculate_triple_crossover_signal(
    short_ma: np.ndarray,
    middle_ma: np.ndarray,
    long_ma: np.ndarray
) -> np.ndarray:
    """
    3本の移動平均を使用したクロスオーバーシグナルを計算する（高速化版）
    
    Args:
        short_ma: 短期移動平均の配列
        middle_ma: 中期移動平均の配列
        long_ma: 長期移動平均の配列
    
    Returns:
        シグナルの配列 (1: ロングエントリー, -1: ショートエントリー, 0: シグナルなし)
    """
    length = len(short_ma)
    signals = np.zeros(length)
    
    # 最初の要素はクロスの判定ができないのでシグナルなし
    for i in range(1, length):
        # 前日の配列状態
        prev_short_above_middle = short_ma[i-1] > middle_ma[i-1]
        prev_middle_above_long = middle_ma[i-1] > long_ma[i-1]
        
        # 当日の配列状態
        curr_short_above_middle = short_ma[i] > middle_ma[i]
        curr_middle_above_long = middle_ma[i] > long_ma[i]
        
        # 完全な上昇配列への変化（ゴールデンクロス）
        if (not prev_short_above_middle or not prev_middle_above_long) and \
           curr_short_above_middle and curr_middle_above_long:
            signals[i] = 1  # ロングエントリー
        
        # 完全な下降配列への変化（デッドクロス）
        elif (prev_short_above_middle or prev_middle_above_long) and \
             not curr_short_above_middle and not curr_middle_above_long:
            signals[i] = -1  # ショートエントリー
    
    return signals


class AlphaMACrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    AlphaMAクロスオーバーを使用したエントリーシグナル
    - 短期AlphaMA > 長期AlphaMA: ロングエントリー (1)
    - 短期AlphaMA < 長期AlphaMA: ショートエントリー (-1)
    
    特徴:
    - 効率比（ER）に基づいて動的に調整されるAlphaMAを使用
    - トレンドが強い時は短いピリオドで速く反応
    - レンジ相場時は長いピリオドでノイズを除去
    """
    
    def __init__(
        self,
        er_period: int = 21,
        short_max_kama_period: int = 89,
        short_min_kama_period: int = 5,
        long_max_kama_period: int = 233,
        long_min_kama_period: int = 21,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2
    ):
        """
        コンストラクタ
        
        Args:
            short_er_period: 短期AlphaMAの効率比の計算期間
            long_er_period: 長期AlphaMAの効率比の計算期間
            short_max_kama_period: 短期AlphaMAのKAMAピリオドの最大値
            short_min_kama_period: 短期AlphaMAのKAMAピリオドの最小値
            long_max_kama_period: 長期AlphaMAのKAMAピリオドの最大値
            long_min_kama_period: 長期AlphaMAのKAMAピリオドの最小値
            max_slow_period: 遅い移動平均の最大期間
            min_slow_period: 遅い移動平均の最小期間
            max_fast_period: 速い移動平均の最大期間
            min_fast_period: 速い移動平均の最小期間
        """
        params = {
            'short_er_period': er_period,
            'long_er_period': er_period,
            'short_max_kama_period': short_max_kama_period,
            'short_min_kama_period': short_min_kama_period,
            'long_max_kama_period': long_max_kama_period,
            'long_min_kama_period': long_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period
        }
        super().__init__(
            f"AlphaMACrossover({short_min_kama_period}, {long_min_kama_period})",
            params
        )
        
        # AlphaMAインジケーターの初期化
        self._short_alpha_ma = AlphaMA(
            er_period=er_period,
            max_kama_period=short_max_kama_period,
            min_kama_period=short_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        
        self._long_alpha_ma = AlphaMA(
            er_period=er_period,
            max_kama_period=long_max_kama_period,
            min_kama_period=long_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # AlphaMAの計算
            short_alpha_ma = self._short_alpha_ma.calculate(data)
            long_alpha_ma = self._long_alpha_ma.calculate(data)
            
            # デバッグ情報
            logger.info(f"短期AlphaMA計算完了: データ長={len(short_alpha_ma)}")
            logger.info(f"長期AlphaMA計算完了: データ長={len(long_alpha_ma)}")
            
            # 最初と最後の値をログ出力
            er_period = self._params['short_er_period']
            logger.info(f"短期AlphaMA: 最初={short_alpha_ma[er_period] if len(short_alpha_ma) > er_period else 'N/A'}, "
                        f"最後={short_alpha_ma[-1] if len(short_alpha_ma) > 0 else 'N/A'}")
            logger.info(f"長期AlphaMA: 最初={long_alpha_ma[er_period] if len(long_alpha_ma) > er_period else 'N/A'}, "
                        f"最後={long_alpha_ma[-1] if len(long_alpha_ma) > 0 else 'N/A'}")
            
            # クロスオーバーシグナルの生成（高速化版）
            signals = calculate_crossover_signal(short_alpha_ma, long_alpha_ma)
            
            # シグナル統計
            logger.info(f"シグナル統計: 合計={np.sum(np.abs(signals))}, "
                        f"ロング={np.sum(signals == 1)}, "
                        f"ショート={np.sum(signals == -1)}")
            
            return signals
        except Exception as e:
            logger.error(f"AlphaMACrossoverEntrySignal生成中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            if 'data' in locals():
                return np.zeros(len(data), dtype=np.int8)
            return np.array([])


class AlphaMATripleCrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    3本のAlphaMAを使用したクロスオーバーエントリーシグナル
    
    シグナル条件:
    - 完全な上昇配列への変化: ロングエントリー (1)
    - 完全な下降配列への変化: ショートエントリー (-1)
    
    特徴:
    - 効率比（ER）に基づいて動的に調整されるAlphaMAを使用
    - トレンドが強い時は短いピリオドで速く反応
    - レンジ相場時は長いピリオドでノイズを除去
    """
    
    def __init__(
        self,
        short_er_period: int = 21,
        middle_er_period: int = 21,
        long_er_period: int = 21,
        short_max_kama_period: int = 89,
        short_min_kama_period: int = 8,
        middle_max_kama_period: int = 144,
        middle_min_kama_period: int = 21,
        long_max_kama_period: int = 233,
        long_min_kama_period: int = 55,
        max_slow_period: int = 89,
        min_slow_period: int = 30,
        max_fast_period: int = 13,
        min_fast_period: int = 2
    ):
        """
        コンストラクタ
        
        Args:
            short_er_period: 短期AlphaMAの効率比の計算期間
            middle_er_period: 中期AlphaMAの効率比の計算期間
            long_er_period: 長期AlphaMAの効率比の計算期間
            short_max_kama_period: 短期AlphaMAのKAMAピリオドの最大値
            short_min_kama_period: 短期AlphaMAのKAMAピリオドの最小値
            middle_max_kama_period: 中期AlphaMAのKAMAピリオドの最大値
            middle_min_kama_period: 中期AlphaMAのKAMAピリオドの最小値
            long_max_kama_period: 長期AlphaMAのKAMAピリオドの最大値
            long_min_kama_period: 長期AlphaMAのKAMAピリオドの最小値
            max_slow_period: 遅い移動平均の最大期間
            min_slow_period: 遅い移動平均の最小期間
            max_fast_period: 速い移動平均の最大期間
            min_fast_period: 速い移動平均の最小期間
        """
        params = {
            'short_er_period': short_er_period,
            'middle_er_period': middle_er_period,
            'long_er_period': long_er_period,
            'short_max_kama_period': short_max_kama_period,
            'short_min_kama_period': short_min_kama_period,
            'middle_max_kama_period': middle_max_kama_period,
            'middle_min_kama_period': middle_min_kama_period,
            'long_max_kama_period': long_max_kama_period,
            'long_min_kama_period': long_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period
        }
        super().__init__(
            f"AlphaMATripleCrossover({short_min_kama_period}, {middle_min_kama_period}, {long_min_kama_period})",
            params
        )
        
        # AlphaMAインジケーターの初期化
        self._short_alpha_ma = AlphaMA(
            er_period=short_er_period,
            max_kama_period=short_max_kama_period,
            min_kama_period=short_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        
        self._middle_alpha_ma = AlphaMA(
            er_period=middle_er_period,
            max_kama_period=middle_max_kama_period,
            min_kama_period=middle_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        
        self._long_alpha_ma = AlphaMA(
            er_period=long_er_period,
            max_kama_period=long_max_kama_period,
            min_kama_period=long_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # AlphaMAの計算
        short_alpha_ma = self._short_alpha_ma.calculate(data)
        middle_alpha_ma = self._middle_alpha_ma.calculate(data)
        long_alpha_ma = self._long_alpha_ma.calculate(data)
        
        # トリプルクロスオーバーシグナルの生成（高速化版）
        signals = calculate_triple_crossover_signal(
            short_alpha_ma, middle_alpha_ma, long_alpha_ma
        )
        
        return signals


class AlphaMACDCrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    AlphaMACDクロスオーバーを使用したエントリーシグナル
    
    シグナル条件:
    - MACD線がシグナル線を上抜け: ロングエントリー (1)
    - MACD線がシグナル線を下抜け: ショートエントリー (-1)
    
    特徴:
    - 効率比（ER）に基づいて動的に調整されるAlphaMAを使用したMACD
    - トレンドが強い時は短いピリオドで速く反応
    - レンジ相場時は長いピリオドでノイズを除去
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
            er_period: 効率比の計算期間
            fast_max_kama_period: 短期AlphaMAのKAMAピリオドの最大値
            fast_min_kama_period: 短期AlphaMAのKAMAピリオドの最小値
            slow_max_kama_period: 長期AlphaMAのKAMAピリオドの最大値
            slow_min_kama_period: 長期AlphaMAのKAMAピリオドの最小値
            signal_max_kama_period: シグナルAlphaMAのKAMAピリオドの最大値
            signal_min_kama_period: シグナルAlphaMAのKAMAピリオドの最小値
            max_slow_period: 遅い移動平均の最大期間
            min_slow_period: 遅い移動平均の最小期間
            max_fast_period: 速い移動平均の最大期間
            min_fast_period: 速い移動平均の最小期間
        """
        params = {
            'er_period': er_period,
            'fast_max_kama_period': fast_max_kama_period,
            'fast_min_kama_period': fast_min_kama_period,
            'slow_max_kama_period': slow_max_kama_period,
            'slow_min_kama_period': slow_min_kama_period,
            'signal_max_kama_period': signal_max_kama_period,
            'signal_min_kama_period': signal_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period
        }
        super().__init__(
            f"AlphaMACDCrossover({fast_min_kama_period}, {slow_min_kama_period}, {signal_min_kama_period})",
            params
        )
        
        # AlphaMACDインジケーターのインポート（ここで動的にインポート）
        from indicators.alpha_macd import AlphaMACD
        
        # AlphaMACDインジケーターの初期化
        self._alpha_macd = AlphaMACD(
            er_period=er_period,
            fast_max_kama_period=fast_max_kama_period,
            fast_min_kama_period=fast_min_kama_period,
            slow_max_kama_period=slow_max_kama_period,
            slow_min_kama_period=slow_min_kama_period,
            signal_max_kama_period=signal_max_kama_period,
            signal_min_kama_period=signal_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # AlphaMACDの計算
        result = self._alpha_macd.calculate(data)
        macd_line = result.macd
        signal_line = result.signal
        
        # クロスオーバーシグナルの生成（高速化版）
        signals = calculate_crossover_signal(macd_line, signal_line)
        
        return signals 