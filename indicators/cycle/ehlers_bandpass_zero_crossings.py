#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult
try:
    from indicators.kalman.unified_kalman import UnifiedKalman
except ImportError:
    try:
        from ..kalman.unified_kalman import UnifiedKalman
    except ImportError:
        UnifiedKalman = None


@jit(nopython=True)
def calculate_bandpass_zero_crossings_numba(
    price: np.ndarray,
    bandwidth: float = 0.6,
    center_period: float = 15.0,
    max_output: int = 34,
    min_output: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    バンドパスゼロクロッシングドミナントサイクル検出のNumba実装
    
    Args:
        price: 価格データの配列
        bandwidth: バンド幅
        center_period: 中心周期
        max_output: 最大出力値
        min_output: 最小出力値
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ドミナントサイクル, 生の周期, 平滑化周期)
    """
    n = len(price)
    pi = 2 * np.arcsin(1.0)
    
    # 初期化
    hp = np.zeros(n)
    bp = np.zeros(n)
    peak = np.zeros(n)
    real = np.zeros(n)
    dc = np.zeros(n)
    counter = np.zeros(n)
    dom_cycle = np.zeros(n)
    
    # フィルター係数の計算
    alpha2 = (np.cos(0.25 * bandwidth * 2 * pi / center_period) + 
              np.sin(0.25 * bandwidth * 2 * pi / center_period) - 1) / np.cos(0.25 * bandwidth * 2 * pi / center_period)
    
    beta1 = np.cos(2 * pi / center_period)
    gamma1 = 1 / np.cos(2 * pi * bandwidth / center_period)
    alpha1 = gamma1 - np.sqrt(gamma1 * gamma1 - 1)
    
    for i in range(n):
        # ハイパスフィルター
        if i >= 1:
            hp[i] = (1 + alpha2 / 2) * (price[i] - price[i-1]) + (1 - alpha2) * hp[i-1]
        else:
            hp[i] = 0.0
        
        # バンドパスフィルター
        if i == 0 or i == 1:
            bp[i] = 0.0
        elif i >= 2:
            bp[i] = 0.5 * (1 - alpha1) * (hp[i] - hp[i-2]) + beta1 * (1 + alpha1) * bp[i-1] - alpha1 * bp[i-2]
        
        # ピーク計算
        if i > 0:
            peak[i] = 0.991 * peak[i-1]
            if np.abs(bp[i]) > peak[i]:
                peak[i] = np.abs(bp[i])
        else:
            peak[i] = 0.0
        
        # Real計算
        if peak[i] != 0:
            real[i] = bp[i] / peak[i]
        else:
            real[i] = 0.0
        
        # ドミナントサイクル計算
        if i > 0:
            dc[i] = dc[i-1]
            counter[i] = counter[i-1] + 1
        else:
            dc[i] = 6.0
            counter[i] = 1
        
        # DC制限
        if dc[i] < 6:
            dc[i] = 6.0
        
        # ゼロクロッシング検出
        zero_crossing = False
        if i > 0:
            # crossover(Real, 0) or crossunder(Real, 0)
            if (real[i-1] <= 0 and real[i] > 0) or (real[i-1] >= 0 and real[i] < 0):
                zero_crossing = True
        
        if zero_crossing:
            new_dc = 2 * counter[i]
            
            # 変化制限を適用
            if i > 0:
                if new_dc > 1.25 * dc[i-1]:
                    new_dc = 1.25 * dc[i-1]
                elif new_dc < 0.8 * dc[i-1]:
                    new_dc = 0.8 * dc[i-1]
            
            dc[i] = new_dc
            counter[i] = 0
        
        # 最終出力計算
        final_dc = int(np.round(max(min_output, min(dc[i], max_output))))
        dom_cycle[i] = final_dc
    
    # 生の周期値と平滑化周期値
    raw_period = np.copy(dc)
    smooth_period = np.copy(dom_cycle)
    
    return dom_cycle, raw_period, smooth_period


class EhlersBandpassZeroCrossings(EhlersDominantCycle):
    """
    エーラーズのバンドパスゼロクロッシングドミナントサイクル検出器
    
    このアルゴリズムはバンドパスフィルターのゼロクロッシングを使用してドミナントサイクルを定義します。
    バンドパスフィルターを通じて特定の周波数帯域を抽出し、ゼロクロッシングでサイクルを検出します。
    
    特徴:
    - バンドパスフィルターによる周波数帯域の抽出
    - ゼロクロッシング検出による正確なサイクル測定
    - ピーク正規化による振幅の安定化
    - 適応的な変化制限による安定性
    """
    
    # 許可されるソースタイプのリスト
    SRC_TYPES = ['close', 'hlc3', 'hl2', 'ohlc4']
    
    def __init__(
        self,
        bandwidth: float = 0.6,
        center_period: float = 15.0,
        max_output: int = 34,
        min_output: int = 1,
        src_type: str = 'close',
        use_kalman_filter: bool = False,
        kalman_filter_type: str = 'adaptive'
    ):
        """
        コンストラクタ
        
        Args:
            bandwidth: バンド幅（デフォルト: 0.6）
            center_period: 中心周期（デフォルト: 15.0）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 1）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値（デフォルト）
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
            use_kalman_filter: Kalmanフィルターを使用するか（デフォルト: False）
            kalman_filter_type: Kalmanフィルターのタイプ（デフォルト: 'adaptive'）
        """
        super().__init__(
            f"EhlersBandpassZeroCrossings({bandwidth}, {center_period})",
            0.5,  # cycle_part (デフォルト値)
            max_output,  # max_cycle
            min_output,  # min_cycle
            max_output,
            min_output
        )
        self.bandwidth = bandwidth
        self.center_period = center_period
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        
        # ソースタイプを保存
        self.src_type = src_type.lower()
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # Kalmanフィルターの初期化
        self.kalman_filter = None
        if self.use_kalman_filter:
            self.kalman_filter = UnifiedKalman(
                filter_type=kalman_filter_type,
                src_type='close'  # 内部的にはcloseで使用
            )
    
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
        バンドパスゼロクロッシングアルゴリズムを使用してドミナントサイクルを計算する
        
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
            dom_cycle, raw_period, smooth_period = calculate_bandpass_zero_crossings_numba(
                price,
                self.bandwidth,
                self.center_period,
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
            self.logger.error(f"EhlersBandpassZeroCrossings計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([]) 