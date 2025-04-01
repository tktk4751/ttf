#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from numba import jit, prange, float64, int32

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult


@jit(nopython=True)
def calculate_phac_dc_numba(
    price: np.ndarray,
    cycle_part: float = 0.5,
    max_cycle: int = 50,
    min_cycle: int = 6,
    max_output: int = 34,
    min_output: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    位相累積（Phase Accumulation）を使ったドミナントサイクル検出のNumba実装
    
    Args:
        price: 価格データの配列
        cycle_part: サイクル部分の倍率
        max_cycle: 最大サイクル期間
        min_cycle: 最小サイクル期間
        max_output: 最大出力値
        min_output: 最小出力値
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ドミナントサイクル, 生の周期, 平滑化周期)
    """
    n = len(price)
    pi = 2 * np.arcsin(1.0)
    
    # 初期化
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    period = np.zeros(n)
    delta_phase = np.zeros(n)
    inst_period = np.zeros(n)
    smooth_period = np.zeros(n)
    dom_cycle = np.zeros(n)
    
    # Phase関連の変数
    phase = np.zeros(n)
    phase_prev = np.zeros(n)
    
    # 最初の数ポイントを初期化
    for i in range(min(7, n)):
        smooth[i] = price[i]
        phase[i] = 2 * pi  # 初期位相
    
    # メインループ
    for i in range(7, n):
        # ヒルベルト変換の実装
        # Smooth computation
        smooth[i] = (4.0 * price[i] + 3.0 * price[i-1] + 2.0 * price[i-2] + price[i-3]) / 10.0
        
        # Detrender computation with adaptive coefficient
        adaptive_coef = 0.075 * period[i-1] + 0.54
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] - 0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * adaptive_coef
        
        # Compute InPhase and Quadrature components
        q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] - 0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * adaptive_coef
        i1[i] = detrender[i-3]
        
        # Smooth the I and Q components
        i1[i] = 0.15 * i1[i] + 0.85 * i1[i-1]
        q1[i] = 0.15 * q1[i] + 0.85 * q1[i-1]
        
        # Use ArcTangent to compute the current phase
        phase_prev[i] = phase[i-1]
        
        if i1[i] != 0:
            # Calculate phase using arctan
            phase[i] = np.arctan(np.abs(q1[i] / i1[i]))
            
            # Resolve the ArcTangent ambiguity for quadrants 2, 3, and 4
            if i1[i] < 0 and q1[i] > 0:
                phase[i] = pi - phase[i]
            elif i1[i] < 0 and q1[i] < 0:
                phase[i] = pi + phase[i]
            elif i1[i] > 0 and q1[i] < 0:
                phase[i] = 2 * pi - phase[i]
        else:
            # If I1 is zero, keep previous phase
            phase[i] = phase[i-1]
        
        # Compute a differential phase
        delta_phase[i] = phase_prev[i] - phase[i]
        
        # Resolve phase wraparound from quadrant 1 to quadrant 4
        if phase_prev[i] < pi / 2 and phase[i] > 3 * pi / 2:
            delta_phase[i] = 2 * pi + phase_prev[i] - phase[i]
        
        # Limit DeltaPhase to be within bounds
        if delta_phase[i] < 7 * pi / 180:  # Minimum 7 degrees
            delta_phase[i] = 7 * pi / 180
        elif delta_phase[i] > pi / 3:  # Maximum 60 degrees
            delta_phase[i] = pi / 3
        
        # Sum Deltaphases to reach 360 degrees.
        # The sum is the instantaneous period.
        phase_sum = 0.0
        inst_cycle = 0
        
        for k in range(41):  # Maximum 40 bars back
            if i-k >= 0:
                phase_sum += delta_phase[i-k]
                if phase_sum > 2 * pi and inst_cycle == 0:
                    inst_cycle = k + 1
                    break
        
        if inst_cycle > 0:
            inst_period[i] = inst_cycle
        else:
            inst_period[i] = inst_period[i-1]
        
        # Smooth the period
        period[i] = 0.25 * inst_period[i] + 0.75 * period[i-1]
        
        # Calculate output cycle
        cycle_value = np.ceil(cycle_part * period[i])
        
        # Limit output
        if cycle_value > max_output:
            dom_cycle[i] = max_output
        elif cycle_value < min_output:
            dom_cycle[i] = min_output
        else:
            dom_cycle[i] = cycle_value
    
    # Store the raw and smoothed periods for return
    raw_period = np.copy(inst_period)
    smooth_period = np.copy(period)
    
    return dom_cycle, raw_period, smooth_period


class EhlersPhAcDC(EhlersDominantCycle):
    """
    エーラーズの位相累積（Phase Accumulation）によるドミナントサイクル検出
    
    このアルゴリズムはヒルベルト変換を使用して価格データから直交成分（I/Q成分）を生成し、
    位相の累積から一周期分の長さを測定することでサイクルを検出します。
    
    特徴:
    - ヒルベルト変換による高精度な位相検出
    - 位相累積による周期検出は高精度だが処理が複雑
    - 適応型フィルターで変化する市場状況に対応
    """
    
    # 許可されるソースタイプのリスト
    SRC_TYPES = ['close', 'hlc3', 'hl2', 'ohlc4']
    
    def __init__(
        self,
        cycle_part: float = 0.5,
        max_cycle: int = 50,
        min_cycle: int = 6,
        max_output: int = 34,
        min_output: int = 1,
        src_type: str = 'close'
    ):
        """
        コンストラクタ
        
        Args:
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 50）
            min_cycle: 最小サイクル期間（デフォルト: 6）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 1）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値（デフォルト）
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
        """
        super().__init__(
            f"EhlersPhAcDC({cycle_part}, {max_cycle}, {min_cycle})",
            cycle_part,
            max_cycle,
            min_cycle,
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
        位相累積を使用してドミナントサイクルを計算する
        
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
            dom_cycle, raw_period, smooth_period = calculate_phac_dc_numba(
                price,
                self.cycle_part,
                self.max_cycle,
                self.min_cycle,
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
            self.logger.error(f"EhlersPhAcDC計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([]) 