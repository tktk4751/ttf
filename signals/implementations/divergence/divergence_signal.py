#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from numba import jit

from signals.base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal


@jit(nopython=True)
def find_peaks(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    データから高値と安値のピークを検出（高速化版）
    
    Args:
        data: 検索対象のデータ配列
        
    Returns:
        (peak_indices, trough_indices) のタプル
        - peak_indices: 高値のインデックス配列
        - trough_indices: 安値のインデックス配列
    """
    peaks = []
    troughs = []
    
    for i in range(1, len(data)-1):
        if data[i-1] < data[i] > data[i+1]:
            peaks.append(i)
        elif data[i-1] > data[i] < data[i+1]:
            troughs.append(i)
    
    return np.array(peaks), np.array(troughs)


@jit(nopython=True)
def detect_divergence(
    price: np.ndarray,
    oscillator: np.ndarray,
    start_idx: int,
    lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ダイバージェンスを検出（高速化版）
    
    Args:
        price: 価格データ
        oscillator: オシレーターの値
        start_idx: 検索開始インデックス
        lookback: ルックバック期間
        
    Returns:
        (bullish, bearish) のタプル
        各配列は検出されたダイバージェンスの位置を示すブール値の配列
    """
    # 結果を格納する配列を初期化
    bullish = np.zeros(len(price), dtype=np.bool_)
    bearish = np.zeros(len(price), dtype=np.bool_)
    
    # データが少なすぎる場合は早期リターン
    if len(price) < start_idx + 2:
        return bullish, bearish
    
    # ピークを検出
    price_peaks, price_troughs = find_peaks(price[start_idx:])
    osc_peaks, osc_troughs = find_peaks(oscillator[start_idx:])
    
    # 空の配列チェック
    if len(price_peaks) == 0 or len(price_troughs) == 0 or len(osc_peaks) == 0 or len(osc_troughs) == 0:
        return bullish, bearish
    
    # インデックスを調整（Numba互換の方法）
    price_peaks = np.array([p + start_idx for p in price_peaks])
    price_troughs = np.array([p + start_idx for p in price_troughs])
    osc_peaks = np.array([p + start_idx for p in osc_peaks])
    osc_troughs = np.array([p + start_idx for p in osc_troughs])
    
    # 強気ダイバージェンス
    if len(price_troughs) >= 2:
        for i in range(len(price_troughs)-1):
            idx1, idx2 = price_troughs[i], price_troughs[i+1]
            if idx2 - idx1 > lookback:
                continue
            
            # 価格が安値を切り下げている
            if price[idx2] < price[idx1]:
                # オシレーターの対応する安値を探す
                min_dist1 = np.inf
                min_dist2 = np.inf
                osc_idx1 = osc_troughs[0]
                osc_idx2 = osc_troughs[0]
                
                # 最も近い安値を探す
                for t in osc_troughs:
                    dist1 = abs(t - idx1)
                    dist2 = abs(t - idx2)
                    if dist1 < min_dist1:
                        min_dist1 = dist1
                        osc_idx1 = t
                    if dist2 < min_dist2:
                        min_dist2 = dist2
                        osc_idx2 = t
                
                # オシレーターが安値を切り上げている
                if oscillator[osc_idx2] > oscillator[osc_idx1]:
                    bullish[idx2] = True
    
    # 弱気ダイバージェンス
    if len(price_peaks) >= 2:
        for i in range(len(price_peaks)-1):
            idx1, idx2 = price_peaks[i], price_peaks[i+1]
            if idx2 - idx1 > lookback:
                continue
            
            # 価格が高値を切り上げている
            if price[idx2] > price[idx1]:
                # オシレーターの対応する高値を探す
                min_dist1 = np.inf
                min_dist2 = np.inf
                osc_idx1 = osc_peaks[0]
                osc_idx2 = osc_peaks[0]
                
                # 最も近い高値を探す
                for p in osc_peaks:
                    dist1 = abs(p - idx1)
                    dist2 = abs(p - idx2)
                    if dist1 < min_dist1:
                        min_dist1 = dist1
                        osc_idx1 = p
                    if dist2 < min_dist2:
                        min_dist2 = dist2
                        osc_idx2 = p
                
                # オシレーターが高値を切り下げている
                if oscillator[osc_idx2] < oscillator[osc_idx1]:
                    bearish[idx2] = True
    
    return bullish, bearish


class DivergenceSignal(BaseSignal, IEntrySignal):
    """
    ダイバージェンスシグナル
    
    価格とオシレーター間の逆行現象（ダイバージェンス）を検出し、
    エントリーシグナルを生成します。
    
    - 強気ダイバージェンス（ロングエントリー）：
      価格が安値を切り下げているのに対し、オシレーターが安値を切り上げている状態。
      上昇転換の可能性を示唆。
    
    - 弱気ダイバージェンス（ショートエントリー）：
      価格が高値を切り上げているのに対し、オシレーターが高値を切り下げている状態。
      下落転換の可能性を示唆。
    """
    
    def __init__(self, lookback: int = 30, params: Dict = None):
        """
        コンストラクタ
        
        Args:
            lookback: ダイバージェンス検出のルックバック期間
            params: その他のパラメータ（オプション）
        """
        super().__init__("Divergence")
        self.lookback = lookback
    
    def generate(self, df: pd.DataFrame, oscillator: np.ndarray) -> np.ndarray:
        """
        ダイバージェンスシグナルを生成
        
        Args:
            df: 価格データを含むDataFrame
            oscillator: オシレーターの値
            
        Returns:
            シグナル配列（1: ロング, -1: ショート, 0: シグナルなし）
        """
        if len(df) != len(oscillator):
            raise ValueError("価格データとオシレーターの長さが一致しません")
        
        # 初期のlookback期間はダイバージェンス計算から除外
        start_idx = self.lookback
        
        # ダイバージェンスを検出（高速化版）
        bullish, bearish = detect_divergence(
            df['close'].values,
            oscillator,
            start_idx,
            self.lookback
        )
        
        # シグナルを生成
        signals = np.zeros(len(df))
        signals[bullish] = 1    # ロングエントリー
        signals[bearish] = -1   # ショートエントリー
        
        return signals 