#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from numba import jit, float64

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult


@jit(nopython=True)
def calculate_dft_dominant_cycle_numba(
    price: np.ndarray,
    window: int = 50,
    cycle_part: float = 0.5,
    max_output: int = 34,
    min_output: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    離散フーリエ変換ドミナントサイクル検出のNumba実装
    
    Args:
        price: 価格データの配列
        window: 分析ウィンドウ長
        cycle_part: サイクル部分の倍率
        max_output: 最大出力値
        min_output: 最小出力値
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ドミナントサイクル, 生の周期, 平滑化周期)
    """
    n = len(price)
    pi = 2 * np.arcsin(1.0)
    
    # 初期化
    hp = np.zeros(n)
    cleaned_data = np.zeros(n)
    dominant_cycle_values = np.zeros(n)
    dom_cycle = np.zeros(n)
    
    for i in range(n):
        # ハイパスフィルターで40期間カットオフでデトレンド
        if i <= 5:
            hp[i] = price[i]
            cleaned_data[i] = price[i]
        else:
            per = 2 * pi / 40
            cos_per = np.cos(per)
            if cos_per != 0:
                alpha1 = (1 - np.sin(per)) / cos_per
            else:
                alpha1 = 0.0
            
            hp[i] = 0.5 * (1 + alpha1) * (price[i] - price[i-1]) + alpha1 * hp[i-1]
            
            # 6タップローパスFIRフィルター
            if i >= 5:
                cleaned_data[i] = (hp[i] + 2 * hp[i-1] + 3 * hp[i-2] + 3 * hp[i-3] + 2 * hp[i-4] + hp[i-5]) / 12
            else:
                cleaned_data[i] = hp[i]
        
        # DFT計算
        if i >= window:
            cosine_part = np.zeros(52)
            sine_part = np.zeros(52)
            pwr = np.zeros(52)
            db = np.zeros(52)
            
            # 各期間に対してDFT計算
            for period in range(8, 51):
                for k in range(window):
                    if i - k >= 0:
                        cyc_per = 2 * pi * k / period
                        cosine_part[period] += cleaned_data[i - k] * np.cos(cyc_per)
                        sine_part[period] += cleaned_data[i - k] * np.sin(cyc_per)
                
                pwr[period] = cosine_part[period] ** 2 + sine_part[period] ** 2
            
            # 正規化のための最大パワーレベルを見つける
            max_pwr = pwr[8]
            for period in range(8, 51):
                if pwr[period] > max_pwr:
                    max_pwr = pwr[period]
            
            # パワーレベルを正規化してデシベルに変換
            for period in range(8, 51):
                if max_pwr > 0 and pwr[period] > 0:
                    ratio = pwr[period] / max_pwr
                    if ratio > 0.01:  # 分母が0になるのを防ぐ
                        db[period] = -10 * np.log10(0.01 / (1 - 0.99 * ratio))
                    else:
                        db[period] = 20
                    
                    if db[period] > 20:
                        db[period] = 20
            
            # 重心アルゴリズムを使用してドミナントサイクルを見つける
            num = 0.0
            denom = 0.0
            for period in range(8, 51):
                if db[period] < 3:
                    three_minus = 3 - db[period]
                    num += period * three_minus
                    denom += three_minus
            
            if denom != 0:
                dominant_cycle_values[i] = num / denom
            else:
                dominant_cycle_values[i] = dominant_cycle_values[i-1] if i > 0 else 15.0
            
            # 最終出力計算
            dc_output = int(np.ceil(cycle_part * dominant_cycle_values[i]))
            if dc_output > max_output:
                dom_cycle[i] = max_output
            elif dc_output < min_output:
                dom_cycle[i] = min_output
            else:
                dom_cycle[i] = dc_output
        else:
            # 初期値
            dominant_cycle_values[i] = 15.0
            dom_cycle[i] = int(np.ceil(cycle_part * 15.0))
    
    # 生の周期値と平滑化周期値
    raw_period = np.copy(dominant_cycle_values)
    smooth_period = np.copy(dom_cycle)
    
    return dom_cycle, raw_period, smooth_period


class EhlersDFTDominantCycle(EhlersDominantCycle):
    """
    エーラーズの離散フーリエ変換ドミナントサイクル検出器
    
    ハイパスフィルター（HP）と6タップローパス有限インパルス応答（FIR）フィルターを入力に実装し、
    離散フーリエ変換計算を行う完全機能式です。分析ウィンドウ長とハイパスフィルターカットオフ
    周波数をリアルタイムで変更できる追加パラメータを追加しました。
    
    特徴:
    - ハイパスフィルターによるデトレンド処理
    - 6タップFIRフィルターによるデータクリーニング
    - 離散フーリエ変換による周波数分析
    - デシベル変換による正規化
    - 重心アルゴリズムによる正確なサイクル検出
    """
    
    # 許可されるソースタイプのリスト
    SRC_TYPES = ['close', 'hlc3', 'hl2', 'ohlc4']
    
    def __init__(
        self,
        window: int = 50,
        cycle_part: float = 0.5,
        max_output: int = 34,
        min_output: int = 1,
        src_type: str = 'close'
    ):
        """
        コンストラクタ
        
        Args:
            window: 分析ウィンドウ長（デフォルト: 50）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 1）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値（デフォルト）
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
        """
        super().__init__(
            f"EhlersDFTDominantCycle({window}, {cycle_part})",
            cycle_part,
            50,  # max_cycle
            8,   # min_cycle
            max_output,
            min_output
        )
        self.window = window
        
        # ソースタイプを保存
        self.src_type = src_type.lower()
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
    
    def calculate_source_values(self, data: Union[pd.DataFrame, np.ndarray], src_type: str) -> np.ndarray:
        """
        指定されたソースタイプに基づいて価格データを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
        
        Returns:
            計算された価格データの配列
        """
        if isinstance(data, pd.DataFrame):
            if src_type == 'close':
                if 'close' in data.columns:
                    return data['close'].values
                elif 'Close' in data.columns:
                    return data['Close'].values
                else:
                    raise ValueError("DataFrameには'close'または'Close'カラムが必要です")
            
            elif src_type == 'hlc3':
                if all(col in data.columns for col in ['high', 'low', 'close']):
                    return (data['high'] + data['low'] + data['close']).values / 3
                elif all(col in data.columns for col in ['High', 'Low', 'Close']):
                    return (data['High'] + data['Low'] + data['Close']).values / 3
                else:
                    raise ValueError("hlc3の計算には'high', 'low', 'close'カラムが必要です")
            
            elif src_type == 'hl2':
                if all(col in data.columns for col in ['high', 'low']):
                    return (data['high'] + data['low']).values / 2
                elif all(col in data.columns for col in ['High', 'Low']):
                    return (data['High'] + data['Low']).values / 2
                else:
                    raise ValueError("hl2の計算には'high', 'low'カラムが必要です")
            
            elif src_type == 'ohlc4':
                if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    return (data['open'] + data['high'] + data['low'] + data['close']).values / 4
                elif all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                    return (data['Open'] + data['High'] + data['Low'] + data['Close']).values / 4
                else:
                    raise ValueError("ohlc4の計算には'open', 'high', 'low', 'close'カラムが必要です")
            
            else:
                raise ValueError(f"無効なソースタイプです: {src_type}")
        
        else:  # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                if src_type == 'close':
                    return data[:, 3]  # close
                elif src_type == 'hlc3':
                    return (data[:, 1] + data[:, 2] + data[:, 3]) / 3  # high, low, close
                elif src_type == 'hl2':
                    return (data[:, 1] + data[:, 2]) / 2  # high, low
                elif src_type == 'ohlc4':
                    return (data[:, 0] + data[:, 1] + data[:, 2] + data[:, 3]) / 4  # open, high, low, close
                else:
                    raise ValueError(f"無効なソースタイプです: {src_type}")
            else:
                return data  # 1次元配列として扱う
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        離散フーリエ変換アルゴリズムを使用してドミナントサイクルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、指定したソースタイプに必要なカラムが必要
        
        Returns:
            ドミナントサイクルの値
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # ソースタイプに基づいて価格データを取得
            price = self.calculate_source_values(data, self.src_type)
            
            # Numba関数を使用してドミナントサイクルを計算
            dom_cycle, raw_period, smooth_period = calculate_dft_dominant_cycle_numba(
                price,
                self.window,
                self.cycle_part,
                self.max_output,
                self.min_output
            )
            
            # 結果を保存
            self._result = DominantCycleResult(
                values=dom_cycle,
                raw_period=raw_period,
                smooth_period=smooth_period
            )
            
            self._values = dom_cycle
            return dom_cycle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"EhlersDFTDominantCycle計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([]) 