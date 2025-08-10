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
def median_of_3(a: float, b: float, c: float) -> float:
    """3つの値の中央値を求める"""
    if a <= b <= c or c <= b <= a:
        return b
    elif b <= a <= c or c <= a <= b:
        return a
    else:
        return c


@jit(nopython=True)
def calculate_cycle_period2_numba(
    price: np.ndarray,
    alpha: float = 0.07,
    cycle_part: float = 0.5,
    max_output: int = 34,
    min_output: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    サイクル期間2ドミナントサイクル検出のNumba実装
    
    Args:
        price: 価格データの配列
        alpha: アルファ係数
        cycle_part: サイクル部分の倍率
        max_output: 最大出力値
        min_output: 最小出力値
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ドミナントサイクル, 生の周期, 平滑化周期)
    """
    n = len(price)
    
    # 初期化
    smooth = np.zeros(n)
    cycle = np.zeros(n)
    q1 = np.zeros(n)
    i1 = np.zeros(n)
    delta_phase_ = np.zeros(n)
    delta_phase = np.zeros(n)
    inst_period = np.zeros(n)
    period = np.zeros(n)
    dom_cycle = np.zeros(n)
    
    for i in range(n):
        # スムーズ化
        if i >= 3:
            smooth[i] = (price[i] + 2 * price[i-1] + 2 * price[i-2] + price[i-3]) / 6.0
        elif i >= 1:
            smooth[i] = (price[i] + price[i-1]) / 2.0
        else:
            smooth[i] = price[i]
        
        # サイクル計算
        if i < 7:
            if i >= 2:
                cycle[i] = (price[i] - 2 * price[i-1] + price[i-2]) / 4.0
            else:
                cycle[i] = 0.0
        else:
            cycle_val = (1 - 0.5 * alpha) * (1 - 0.5 * alpha) * (smooth[i] - 2 * smooth[i-1] + smooth[i-2])
            cycle_val += 2 * (1 - alpha) * cycle[i-1] - (1 - alpha) * (1 - alpha) * cycle[i-2]
            cycle[i] = cycle_val
        
        # Q1とI1の計算
        if i >= 6:
            coeff_factor = 0.5 + 0.08 * inst_period[i-1] if i > 0 else 0.58
            q1[i] = (0.0962 * cycle[i] + 0.5769 * cycle[i-2] - 0.5769 * cycle[i-4] - 0.0962 * cycle[i-6]) * coeff_factor
            i1[i] = cycle[i-3]
        
        # DeltaPhase計算
        if i > 0 and q1[i] != 0 and q1[i-1] != 0:
            numerator = i1[i] / q1[i] - i1[i-1] / q1[i-1]
            denominator = 1 + i1[i] * i1[i-1] / (q1[i] * q1[i-1])
            if denominator != 0:
                delta_phase_[i] = numerator / denominator
            else:
                delta_phase_[i] = 0.0
        else:
            delta_phase_[i] = 0.0
        
        # DeltaPhase範囲制限
        if delta_phase_[i] < 0.1:
            delta_phase[i] = 0.1
        elif delta_phase_[i] > 1.1:
            delta_phase[i] = 1.1
        else:
            delta_phase[i] = delta_phase_[i]
        
        # メディアン計算（med(DeltaPhase, DeltaPhase[1], med(DeltaPhase[2], DeltaPhase[3], DeltaPhase[4]))）
        if i >= 4:
            inner_median = median_of_3(delta_phase[i-2], delta_phase[i-3], delta_phase[i-4])
            md = median_of_3(delta_phase[i], delta_phase[i-1], inner_median)
        elif i >= 2:
            md = median_of_3(delta_phase[i], delta_phase[i-1], delta_phase[i-2])
        elif i >= 1:
            md = (delta_phase[i] + delta_phase[i-1]) / 2.0
        else:
            md = delta_phase[i]
        
        # DC計算
        if md == 0:
            dc = 15.0
        else:
            dc = 6.28318 / md + 0.5
        
        # InstPeriod計算
        if i > 0:
            inst_period[i] = 0.33 * dc + 0.67 * inst_period[i-1]
        else:
            inst_period[i] = dc
        
        # Period計算
        if i > 0:
            period[i] = 0.15 * inst_period[i] + 0.85 * period[i-1]
        else:
            period[i] = inst_period[i]
        
        # 最終出力計算
        int_period = int(np.round(period[i] * cycle_part))
        if int_period > max_output:
            dom_cycle[i] = max_output
        elif int_period < min_output:
            dom_cycle[i] = min_output
        else:
            dom_cycle[i] = int_period
    
    return dom_cycle, inst_period, period


class EhlersCyclePeriod2(EhlersDominantCycle):
    """
    エーラーズのサイクル期間2ドミナントサイクル検出器
    
    このアルゴリズムはサイクル期間の別バージョンで、メディアン計算方法が異なります。
    現在のピークまたは谷と次のピークまたは谷の間の概算日数を計算します。
    
    特徴:
    - 価格データのスムーズ化
    - サイクル成分の抽出
    - 位相差分析による周期検出
    - 改良されたメディアンフィルタによるノイズ除去
    """
    
    # 許可されるソースタイプのリスト
    SRC_TYPES = ['close', 'hlc3', 'hl2', 'ohlc4', 'ukf_hlc3']
    
    def __init__(
        self,
        alpha: float = 0.07,
        cycle_part: float = 0.5,
        max_output: int = 34,
        min_output: int = 1,
        src_type: str = 'close',
        use_kalman_filter: bool = False,
        kalman_filter_type: str = 'adaptive'
    ):
        """
        コンストラクタ
        
        Args:
            alpha: アルファ係数（デフォルト: 0.07）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
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
            f"EhlersCyclePeriod2({alpha}, {cycle_part})",
            cycle_part,
            48,  # max_cycle (デフォルト値)
            10,  # min_cycle (デフォルト値)
            max_output,
            min_output
        )
        self.alpha = alpha
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
        サイクル期間2アルゴリズムを使用してドミナントサイクルを計算する
        
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
            dom_cycle, raw_period, smooth_period = calculate_cycle_period2_numba(
                price,
                self.alpha,
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
            self.logger.error(f"EhlersCyclePeriod2計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([]) 