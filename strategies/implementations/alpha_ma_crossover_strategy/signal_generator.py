#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
import logging
from numba import jit

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.alpha_ma.entry import AlphaMACrossoverEntrySignal


# ロガーの設定
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@jit(nopython=True)
def calculate_exit_signal(
    ma_crossover_signal: int,
    position: int
) -> bool:
    """エグジットシグナルを計算（高速化版）
    
    Args:
        ma_crossover_signal: 現在のAlphaMACrossoverEntrySignal
        position: 現在のポジション（1: ロング、-1: ショート）
        
    Returns:
        bool: エグジットすべきかどうか
    """
    # ロングポジションの場合
    if position == 1:
        # AlphaMACrossoverEntrySignalが-1でロング決済
        return ma_crossover_signal == -1
    
    # ショートポジションの場合
    elif position == -1:
        # AlphaMACrossoverEntrySignalが1になったらショート決済
        return ma_crossover_signal == 1
    
    # ポジションがない場合
    return False


class AlphaMACrossoverSignalGenerator(BaseSignalGenerator):
    """
    アルファMAクロスオーバー戦略のシグナル生成クラス
    
    エントリー条件:
    - AlphaMACrossoverEntrySignalが1のときにロング
    - AlphaMACrossoverEntrySignalが-1のときにショート
    
    エグジット条件:
    - AlphaMACrossoverEntrySignalが-1でロング決済
    - AlphaMACrossoverEntrySignalが1になったらショート決済
    """
    
    def __init__(
        self,
        # AlphaMACrossoverEntrySignalのパラメータ
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
        """初期化"""
        super().__init__("AlphaMACrossoverSignalGenerator")
        
        # パラメータの設定
        self._params = {
            # AlphaMACrossoverEntrySignalのパラメータ
            'er_period': er_period,
            'short_max_kama_period': short_max_kama_period,
            'short_min_kama_period': short_min_kama_period,
            'long_max_kama_period': long_max_kama_period,
            'long_min_kama_period': long_min_kama_period,
            'max_slow_period': max_slow_period,
            'min_slow_period': min_slow_period,
            'max_fast_period': max_fast_period,
            'min_fast_period': min_fast_period
        }
        
        # AlphaMACrossoverEntrySignalの初期化
        self.alpha_ma_crossover = AlphaMACrossoverEntrySignal(
            er_period=er_period,
            short_max_kama_period=short_max_kama_period,
            short_min_kama_period=short_min_kama_period,
            long_max_kama_period=long_max_kama_period,
            long_min_kama_period=long_min_kama_period,
            max_slow_period=max_slow_period,
            min_slow_period=min_slow_period,
            max_fast_period=max_fast_period,
            min_fast_period=min_fast_period
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._ma_crossover_signals = None
        self.er_period = er_period
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._ma_crossover_signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data[['open', 'high', 'low', 'close']]
                else:
                    df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                
                # 各シグナルの計算
                try:
                    # MAクロスオーバーシグナルの計算
                    self._ma_crossover_signals = self.alpha_ma_crossover.generate(df)
                    
                    # デバッグ情報
                    logger.info(f"AlphaMACrossoverEntrySignal生成完了: データ長={len(self._ma_crossover_signals)}")
                    logger.info(f"シグナル統計: 合計={np.sum(np.abs(self._ma_crossover_signals))}, "
                                f"ロング={np.sum(self._ma_crossover_signals == 1)}, "
                                f"ショート={np.sum(self._ma_crossover_signals == -1)}")
                    
                    # AlphaMAの値を取得してデバッグ
                    short_alpha_ma = self.alpha_ma_crossover._short_alpha_ma._values
                    long_alpha_ma = self.alpha_ma_crossover._long_alpha_ma._values
                    
                    if short_alpha_ma is not None and long_alpha_ma is not None:
                        # 最初と最後の値をログ出力
                        er_period = self.er_period
                        logger.info(f"短期AlphaMA: 最初={short_alpha_ma[er_period] if len(short_alpha_ma) > er_period else 'N/A'}, "
                                    f"最後={short_alpha_ma[-1] if len(short_alpha_ma) > 0 else 'N/A'}")
                        logger.info(f"長期AlphaMA: 最初={long_alpha_ma[er_period] if len(long_alpha_ma) > er_period else 'N/A'}, "
                                    f"最後={long_alpha_ma[-1] if len(long_alpha_ma) > 0 else 'N/A'}")
                        
                        # クロスオーバーの検出
                        crosses = 0
                        for i in range(1, min(len(short_alpha_ma), len(long_alpha_ma))):
                            if (short_alpha_ma[i-1] <= long_alpha_ma[i-1] and short_alpha_ma[i] > long_alpha_ma[i]) or \
                               (short_alpha_ma[i-1] >= long_alpha_ma[i-1] and short_alpha_ma[i] < long_alpha_ma[i]):
                                crosses += 1
                        
                        logger.info(f"検出されたクロスオーバー数: {crosses}")
                    
                except Exception as e:
                    logger.error(f"シグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._ma_crossover_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._ma_crossover_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._ma_crossover_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._ma_crossover_signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if self._ma_crossover_signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # 現在のMAクロスオーバーシグナルを取得
        current_ma_crossover_signal = self._ma_crossover_signals[index]
        
        # エグジットシグナルの計算（高速化版）
        return calculate_exit_signal(current_ma_crossover_signal, position)
    
    def get_ma_crossover_signals(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        MAクロスオーバーシグナルの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: MAクロスオーバーシグナルの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self._ma_crossover_signals
        except Exception as e:
            logger.error(f"MAクロスオーバーシグナル取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            return np.array([]) 