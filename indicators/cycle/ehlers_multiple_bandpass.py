#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from numba import jit, float64

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult
from ..kalman.unified_kalman import UnifiedKalman


@jit(nopython=True)
def median_simple(value: float, median_val: float) -> float:
    """簡易メディアン関数（2つの値の場合）"""
    return max(value, median_val)


@jit(nopython=True)
def calculate_multiple_bandpass_numba(
    price: np.ndarray,
    cycle_part: float = 0.5,
    max_output: int = 34,
    min_output: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    複数バンドパスフィルタリングドミナントサイクル検出のNumba実装
    
    Args:
        price: 価格データの配列
        cycle_part: サイクル部分の倍率
        max_output: 最大出力値
        min_output: 最小出力値
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ドミナントサイクル, 生の周期, 平滑化周期)
    """
    n = len(price)
    pi = 2 * np.arcsin(1.0)
    two_pi = 2 * pi
    log10 = np.log(10)
    
    # 初期化
    hp = np.zeros(n)
    smooth_hp = np.zeros(n)
    dc_values = np.zeros(n)
    dom_cyc = np.zeros(n)
    dom_cycle = np.zeros(n)
    
    # 係数計算
    alpha1 = (1 - np.sin(9 / 180 * pi)) / np.cos(9 / 180 * pi)
    alpha1_plus1 = alpha1 + 1
    
    # 配列初期化（各期間用）
    ehlers_i = np.zeros((n, 52))
    old_i = np.zeros((n, 52))
    older_i = np.zeros((n, 52))
    q = np.zeros((n, 52))
    old_q = np.zeros((n, 52))
    older_q = np.zeros((n, 52))
    real = np.zeros((n, 52))
    old_real = np.zeros((n, 52))
    older_real = np.zeros((n, 52))
    imag = np.zeros((n, 52))
    old_imag = np.zeros((n, 52))
    older_imag = np.zeros((n, 52))
    ampl = np.zeros((n, 52))
    old_ampl = np.zeros((n, 52))
    db = np.zeros((n, 52))
    
    for i in range(n):
        # ハイパスフィルター
        if i > 0:
            hp[i] = 0.5 * alpha1_plus1 * (price[i] - price[i-1]) + alpha1 * hp[i-1]
        else:
            hp[i] = 0.0
        
        # スムーズHP
        if i == 0:
            smooth_hp[i] = 0.0
        elif i < 7:
            smooth_hp[i] = price[i] - price[i-1]
        elif i >= 5:
            smooth_hp[i] = (hp[i] + 2 * hp[i-1] + 3 * hp[i-2] + 3 * hp[i-3] + 2 * hp[i-4] + hp[i-5]) / 12
        else:
            smooth_hp[i] = hp[i]
        
        # Delta計算
        ehlers_delta = -0.015 * i + 0.5
        ehlers_delta = max(0.15, ehlers_delta)
        
        if i > 6:
            # 各期間に対してバンドパスフィルタリング
            for n_val in range(8, 51):
                ehlers_beta = np.cos(two_pi / n_val)
                cos720_delta = np.cos(4 * pi * ehlers_delta / n_val)
                
                if cos720_delta != 0:
                    ehlers_gamma = 1 / cos720_delta
                else:
                    ehlers_gamma = 1.0
                
                alpha = ehlers_gamma - np.sqrt(ehlers_gamma ** 2 - 1)
                one_minus_alpha = 1 - alpha
                one_plus_alpha = 1 + alpha
                
                # Q計算
                if i > 0:
                    q[i, n_val] = n_val / two_pi * (smooth_hp[i] - smooth_hp[i-1])
                else:
                    q[i, n_val] = 0.0
                
                # I計算
                ehlers_i[i, n_val] = smooth_hp[i]
                
                # Real計算
                if i >= 2:
                    real[i, n_val] = (0.5 * one_minus_alpha * (ehlers_i[i, n_val] - older_i[i-1, n_val]) +
                                     ehlers_beta * one_plus_alpha * old_real[i-1, n_val] -
                                     alpha * older_real[i-1, n_val])
                else:
                    real[i, n_val] = 0.0
                
                # Imag計算
                if i >= 2:
                    imag[i, n_val] = (0.5 * one_minus_alpha * (q[i, n_val] - older_q[i-1, n_val]) +
                                     ehlers_beta * one_plus_alpha * old_imag[i-1, n_val] -
                                     alpha * older_imag[i-1, n_val])
                else:
                    imag[i, n_val] = 0.0
                
                # 振幅計算
                ampl[i, n_val] = real[i, n_val] ** 2 + imag[i, n_val] ** 2
        
        # 配列更新
        for n_val in range(8, 51):
            if i > 0:
                older_i[i, n_val] = old_i[i-1, n_val]
                older_q[i, n_val] = old_q[i-1, n_val]
                older_real[i, n_val] = old_real[i-1, n_val]
                older_imag[i, n_val] = old_imag[i-1, n_val]
            
            old_i[i, n_val] = ehlers_i[i, n_val]
            old_q[i, n_val] = q[i, n_val]
            old_real[i, n_val] = real[i, n_val]
            old_imag[i, n_val] = imag[i, n_val]
            old_ampl[i, n_val] = ampl[i, n_val]
        
        # 最大振幅計算
        max_ampl = ampl[i, 10] if i > 9 else 0.0
        for n_val in range(8, 51):
            if ampl[i, n_val] > max_ampl:
                max_ampl = ampl[i, n_val]
        
        # dB変換
        for n_val in range(8, 51):
            if max_ampl != 0 and ampl[i, n_val] / max_ampl > 0:
                ratio = ampl[i, n_val] / max_ampl
                if ratio > 0.01:
                    db[i, n_val] = -10 * np.log10(0.01 / (1 - 0.99 * ratio))
                else:
                    db[i, n_val] = 20
                
                if db[i, n_val] > 20:
                    db[i, n_val] = 20
            else:
                db[i, n_val] = 20
        
        # 重心計算
        num = 0.0
        denom = 0.0
        for n_val in range(10, 51):
            if db[i, n_val] <= 3:
                weight = 20 - db[i, n_val]
                num += n_val * weight
                denom += weight
        
        if denom != 0:
            dc_values[i] = num / denom
        else:
            dc_values[i] = dc_values[i-1] if i > 0 else 15.0
        
        # メディアン処理
        dom_cyc[i] = median_simple(dc_values[i], 10.0)
        
        # 最終出力計算
        dc_output = int(np.ceil(cycle_part * dom_cyc[i]))
        if dc_output > max_output:
            dom_cycle[i] = max_output
        elif dc_output < min_output:
            dom_cycle[i] = min_output
        else:
            dom_cycle[i] = dc_output
    
    # 生の周期値と平滑化周期値
    raw_period = np.copy(dc_values)
    smooth_period = np.copy(dom_cycle)
    
    return dom_cycle, raw_period, smooth_period


class EhlersMultipleBandpass(EhlersDominantCycle):
    """
    エーラーズの複数バンドパスフィルタリングドミナントサイクル検出器
    
    複数のバンドパスフィルターを使用してドミナントサイクルを検出します。
    各期間に対して個別のバンドパスフィルターを適用し、実部と虚部の成分を計算して
    振幅ベースの分析を行います。
    
    特徴:
    - 複数の適応バンドパスフィルター
    - 各期間に対する実部・虚部分析
    - 振幅ベースの周期検出
    - デシベル変換による正規化
    - 重心アルゴリズムによる正確なサイクル検出
    """
    
    # 許可されるソースタイプのリスト
    SRC_TYPES = ['close', 'hlc3', 'hl2', 'ohlc4']
    
    def __init__(
        self,
        cycle_part: float = 0.5,
        max_output: int = 34,
        min_output: int = 1,
        src_type: str = 'close'
    ):
        """
        コンストラクタ
        
        Args:
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
            f"EhlersMultipleBandpass({cycle_part})",
            cycle_part,
            50,  # max_cycle
            8,   # min_cycle
            max_output,
            min_output
        )
        
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
        複数バンドパスフィルタリングアルゴリズムを使用してドミナントサイクルを計算する
        
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
            dom_cycle, raw_period, smooth_period = calculate_multiple_bandpass_numba(
                price,
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
            self.logger.error(f"EhlersMultipleBandpass計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([]) 