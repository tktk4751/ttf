#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Tuple, List
from .indicator import Indicator


class Divergence(Indicator):
    """
    ダイバージェンス検出インディケーター
    
    価格とオシレーター間の逆行現象（ダイバージェンス）を検出します。
    
    - 強気ダイバージェンス：
      価格が安値を切り下げているのに対し、オシレーターが安値を切り上げている状態。
      上昇転換の可能性を示唆。
    
    - 弱気ダイバージェンス：
      価格が高値を切り上げているのに対し、オシレーターが高値を切り下げている状態。
      下落転換の可能性を示唆。
    """
    
    def __init__(self, lookback: int = 30):
        """
        コンストラクタ
        
        Args:
            lookback: ダイバージェンス検出のルックバック期間
        """
        super().__init__("Divergence")
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
        peaks = []
        troughs = []
        
        for i in range(1, len(data)-1):
            if data[i-1] < data[i] > data[i+1]:
                peaks.append(i)
            elif data[i-1] > data[i] < data[i+1]:
                troughs.append(i)
        
        return peaks, troughs
    
    def _detect_divergence(
        self,
        price: np.ndarray,
        oscillator: np.ndarray,
        start_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ダイバージェンスを検出
        
        Args:
            price: 価格データ
            oscillator: オシレーターの値
            start_idx: 検索開始インデックス
            
        Returns:
            (bullish, bearish) のタプル
            各配列は検出されたダイバージェンスの位置を示すブール値の配列
        """
        # 結果を格納する配列を初期化
        bullish = np.zeros(len(price), dtype=bool)
        bearish = np.zeros(len(price), dtype=bool)
        
        # ピークを検出
        price_peaks, price_troughs = self._find_peaks(price[start_idx:])
        osc_peaks, osc_troughs = self._find_peaks(oscillator[start_idx:])
        
        # インデックスを調整
        price_peaks = [i + start_idx for i in price_peaks]
        price_troughs = [i + start_idx for i in price_troughs]
        osc_peaks = [i + start_idx for i in osc_peaks]
        osc_troughs = [i + start_idx for i in osc_troughs]
        
        # 強気ダイバージェンス
        # 価格が安値を切り下げ（より低い安値）、オシレーターが安値を切り上げ（より高い安値）
        for i in range(len(price_troughs)-1):
            idx1, idx2 = price_troughs[i], price_troughs[i+1]
            if idx2 - idx1 > self.lookback:
                continue
            
            # 価格が安値を切り下げている
            if price[idx2] < price[idx1]:
                # オシレーターの対応する安値を探す
                osc_idx1 = min(osc_troughs, key=lambda x: abs(x - idx1))
                osc_idx2 = min(osc_troughs, key=lambda x: abs(x - idx2))
                
                # オシレーターが安値を切り上げている
                if oscillator[osc_idx2] > oscillator[osc_idx1]:
                    bullish[idx2] = True
        
        # 弱気ダイバージェンス
        # 価格が高値を切り上げ（より高い高値）、オシレーターが高値を切り下げ（より低い高値）
        for i in range(len(price_peaks)-1):
            idx1, idx2 = price_peaks[i], price_peaks[i+1]
            if idx2 - idx1 > self.lookback:
                continue
            
            # 価格が高値を切り上げている
            if price[idx2] > price[idx1]:
                # オシレーターの対応する高値を探す
                osc_idx1 = min(osc_peaks, key=lambda x: abs(x - idx1))
                osc_idx2 = min(osc_peaks, key=lambda x: abs(x - idx2))
                
                # オシレーターが高値を切り下げている
                if oscillator[osc_idx2] < oscillator[osc_idx1]:
                    bearish[idx2] = True
        
        return bullish, bearish
    
    def calculate(self, price: np.ndarray, oscillator: np.ndarray) -> np.ndarray:
        """
        ダイバージェンスを計算
        
        Args:
            price: 価格データ
            oscillator: オシレーターの値
            
        Returns:
            ダイバージェンスの検出結果（1: 強気, -1: 弱気, 0: なし）
        """
        if len(price) != len(oscillator):
            raise ValueError("価格データとオシレーターの長さが一致しません")
        
        # 初期のlookback期間はダイバージェンス計算から除外
        start_idx = self.lookback
        
        # ダイバージェンスを検出
        self._bullish, self._bearish = self._detect_divergence(
            price, oscillator, start_idx
        )
        
        # 結果を統合
        self._values = np.zeros(len(price))
        self._values[self._bullish] = 1
        self._values[self._bearish] = -1
        
        return self._values
    
    def get_divergence_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        各タイプのダイバージェンス状態を取得
        
        Returns:
            (bullish, bearish) のタプル
            各配列はそれぞれのダイバージェンスが検出された位置を示すブール値の配列
        """
        if self._values is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._bullish, self._bearish 