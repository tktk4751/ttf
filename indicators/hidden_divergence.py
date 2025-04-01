#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Tuple, List
from .indicator import Indicator
from numba import jit, boolean, int32, float64


class HiddenDivergence(Indicator):
    """
    ヒドゥンダイバージェンス検出インディケーター
    
    価格とオシレーター間の隠れた逆行現象（ヒドゥンダイバージェンス）を検出します。
    Numbaによる最適化を実装しています。
    
    - 強気ヒドゥンダイバージェンス：
      価格が安値を切り上げているのに対し、オシレーターが安値を切り下げている状態。
      トレンド継続の可能性を示唆（上昇トレンドの継続）。
    
    - 弱気ヒドゥンダイバージェンス：
      価格が高値を切り下げているのに対し、オシレーターが高値を切り上げている状態。
      トレンド継続の可能性を示唆（下降トレンドの継続）。
    """
    
    def __init__(self, lookback: int = 30):
        """
        コンストラクタ
        
        Args:
            lookback: ヒドゥンダイバージェンス検出のルックバック期間
        """
        super().__init__("HiddenDivergence")
        self.lookback = lookback
        
        # 検出結果を保持する変数
        self._bullish: np.ndarray = None
        self._bearish: np.ndarray = None
    
    def _find_peaks(self, data: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        データから高値と安値のピークを検出
        
        Args:
            data: 検索対象のデータ配列
            
        Returns:
            (peak_indices, trough_indices) のタプル
            - peak_indices: 高値のインデックスリスト
            - trough_indices: 安値のインデックスリスト
        """
        # Numba最適化された関数を呼び出し
        peaks, troughs = _find_peaks_numba(data)
        return peaks.tolist(), troughs.tolist()
    
    def _detect_hidden_divergence(
        self,
        price: np.ndarray,
        oscillator: np.ndarray,
        start_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ヒドゥンダイバージェンスを検出
        
        Args:
            price: 価格データ
            oscillator: オシレーターの値
            start_idx: 検索開始インデックス
            
        Returns:
            (bullish, bearish) のタプル
            各配列は検出されたヒドゥンダイバージェンスの位置を示すブール値の配列
        """
        # ピークを検出
        price_peaks_arr, price_troughs_arr = _find_peaks_numba(price[start_idx:])
        osc_peaks_arr, osc_troughs_arr = _find_peaks_numba(oscillator[start_idx:])
        
        # インデックスを調整
        price_peaks = [i + start_idx for i in price_peaks_arr]
        price_troughs = [i + start_idx for i in price_troughs_arr]
        osc_peaks = [i + start_idx for i in osc_peaks_arr]
        osc_troughs = [i + start_idx for i in osc_troughs_arr]
        
        # Numba最適化された関数を呼び出し
        return _detect_hidden_divergence_numba(
            price, oscillator, 
            np.array(price_peaks), np.array(price_troughs),
            np.array(osc_peaks), np.array(osc_troughs),
            self.lookback
        )
    
    def calculate(self, price: np.ndarray, oscillator: np.ndarray) -> np.ndarray:
        """
        ヒドゥンダイバージェンスを計算
        
        Args:
            price: 価格データ
            oscillator: オシレーターの値
            
        Returns:
            ヒドゥンダイバージェンスの検出結果（1: 強気, -1: 弱気, 0: なし）
        """
        if len(price) != len(oscillator):
            raise ValueError("価格データとオシレーターの長さが一致しません")
        
        # 初期のlookback期間はダイバージェンス計算から除外
        start_idx = self.lookback
        
        # ヒドゥンダイバージェンスを検出
        self._bullish, self._bearish = self._detect_hidden_divergence(
            price, oscillator, start_idx
        )
        
        # 結果を統合
        self._values = np.zeros(len(price))
        self._values[self._bullish] = 1
        self._values[self._bearish] = -1
        
        return self._values
    
    def get_divergence_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        各タイプのヒドゥンダイバージェンス状態を取得
        
        Returns:
            (bullish, bearish) のタプル
            各配列はそれぞれのヒドゥンダイバージェンスが検出された位置を示すブール値の配列
        """
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._bullish, self._bearish


@jit(nopython=True)
def _find_peaks_numba(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba最適化されたピーク検出関数
    
    Args:
        data: 検索対象のデータ配列
        
    Returns:
        (peaks, troughs) のタプル
        - peaks: 高値のインデックス配列
        - troughs: 安値のインデックス配列
    """
    # 一時的な結果を格納するリスト
    peaks_list = []
    troughs_list = []
    
    for i in range(1, len(data)-1):
        if data[i-1] < data[i] > data[i+1]:
            peaks_list.append(i)
        elif data[i-1] > data[i] < data[i+1]:
            troughs_list.append(i)
    
    # NumbaはリストをサポートしていないためNumPy配列に変換
    peaks = np.array(peaks_list, dtype=np.int32)
    troughs = np.array(troughs_list, dtype=np.int32)
    
    return peaks, troughs


@jit(nopython=True)
def _find_nearest_index(indices: np.ndarray, target: int) -> int:
    """
    Numba最適化された最も近いインデックスを見つける関数
    
    Args:
        indices: 検索対象のインデックス配列
        target: 検索する値
        
    Returns:
        最も近いインデックス、または-1（見つからない場合）
    """
    if len(indices) == 0:
        return -1
    
    # 最も近いインデックスを見つける
    min_dist = np.abs(indices[0] - target)
    min_idx = 0
    
    for i in range(1, len(indices)):
        dist = np.abs(indices[i] - target)
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    
    return indices[min_idx]


@jit(nopython=True)
def _detect_hidden_divergence_numba(
    price: np.ndarray,
    oscillator: np.ndarray,
    price_peaks: np.ndarray,
    price_troughs: np.ndarray,
    osc_peaks: np.ndarray,
    osc_troughs: np.ndarray,
    lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba最適化されたヒドゥンダイバージェンス検出関数
    
    Args:
        price: 価格データ
        oscillator: オシレーターの値
        price_peaks: 価格の高値インデックス
        price_troughs: 価格の安値インデックス
        osc_peaks: オシレーターの高値インデックス
        osc_troughs: オシレーターの安値インデックス
        lookback: ルックバック期間
        
    Returns:
        (bullish, bearish) のタプル
        各配列は検出されたヒドゥンダイバージェンスの位置を示すブール値の配列
    """
    # 結果を格納する配列を初期化
    bullish = np.zeros(len(price), dtype=boolean)
    bearish = np.zeros(len(price), dtype=boolean)
    
    # 強気ヒドゥンダイバージェンス
    # 価格が安値を切り上げ（より高い安値）、オシレーターが安値を切り下げ（より低い安値）
    for i in range(len(price_troughs)-1):
        idx1, idx2 = price_troughs[i], price_troughs[i+1]
        if idx2 - idx1 > lookback:
            continue
        
        # 価格が安値を切り上げている
        if price[idx2] > price[idx1]:
            # オシレーターの対応する安値を探す
            osc_idx1 = _find_nearest_index(osc_troughs, idx1)
            osc_idx2 = _find_nearest_index(osc_troughs, idx2)
            
            # オシレーターが安値を切り下げている
            if osc_idx1 != -1 and osc_idx2 != -1 and oscillator[osc_idx2] < oscillator[osc_idx1]:
                bullish[idx2] = True
    
    # 弱気ヒドゥンダイバージェンス
    # 価格が高値を切り下げ（より低い高値）、オシレーターが高値を切り上げ（より高い高値）
    for i in range(len(price_peaks)-1):
        idx1, idx2 = price_peaks[i], price_peaks[i+1]
        if idx2 - idx1 > lookback:
            continue
        
        # 価格が高値を切り下げている
        if price[idx2] < price[idx1]:
            # オシレーターの対応する高値を探す
            osc_idx1 = _find_nearest_index(osc_peaks, idx1)
            osc_idx2 = _find_nearest_index(osc_peaks, idx2)
            
            # オシレーターが高値を切り上げている
            if osc_idx1 != -1 and osc_idx2 != -1 and oscillator[osc_idx2] > oscillator[osc_idx1]:
                bearish[idx2] = True
    
    return bullish, bearish 