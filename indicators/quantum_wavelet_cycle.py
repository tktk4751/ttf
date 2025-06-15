#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange, float64, int32
from scipy import signal
from collections import deque

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult


@njit(nopython=True, fastmath=True, cache=True)
def detrend_series(series: np.ndarray) -> np.ndarray:
    """
    エーラーズ式の低遅延FIRフィルターを使用してトレンドを除去
    (Price + 2*P[1] + 2*P[2] + P[3]) / 6
    """
    n = len(series)
    if n < 4:
        return series
    
    detrended = np.zeros(n - 3)
    
    for i in range(3, n):
        smoothed = (series[i] + 2 * series[i-1] + 2 * series[i-2] + series[i-3]) / 6.0
        detrended[i-3] = series[i] - smoothed
    
    return detrended


@njit(nopython=True, fastmath=True, cache=True)
def complex_morlet_wavelet(n: int, period: float, bandwidth: float = 1.5) -> np.ndarray:
    """
    複素モーレットウェーブレット生成（Numba最適化版）
    """
    t = np.arange(-n//2, n//2, dtype=np.float64)
    
    # スケールパラメータ
    s = period / (2 * np.pi)
    
    # エンベロープ（ガウシアン）
    envelope = np.exp(-0.5 * (t / (s * bandwidth))**2)
    
    # 複素指数項
    oscillation = np.exp(1j * 2 * np.pi * t / period)
    
    # 複素モーレットウェーブレット
    wavelet = envelope * oscillation
    
    # 正規化
    norm = np.sqrt(np.sum(np.abs(wavelet)**2))
    if norm > 0:
        wavelet = wavelet / norm
    
    return wavelet


@njit(nopython=True, fastmath=True, cache=True)
def calculate_cwt_power(
    signal_data: np.ndarray,
    periods: np.ndarray,
    bandwidth: float = 1.5
) -> np.ndarray:
    """
    連続ウェーブレット変換のパワースペクトラム計算（Numba最適化版）
    """
    n_data = len(signal_data)
    n_periods = len(periods)
    power_spectrum = np.zeros(n_periods)
    
    for i in range(n_periods):
        period = periods[i]
        wavelet_length = min(int(period * 3), n_data)  # ウェーブレット長制限
        
        if wavelet_length < 4:
            continue
            
        # モーレットウェーブレット生成
        wavelet = complex_morlet_wavelet(wavelet_length, period, bandwidth)
        
        # 畳み込み計算（最新のデータポイント用）
        start_idx = max(0, n_data - wavelet_length)
        signal_segment = signal_data[start_idx:n_data]
        
        if len(signal_segment) >= len(wavelet):
            conv_result = 0.0 + 0.0j
            for j in range(len(wavelet)):
                if len(signal_segment) - 1 - j >= 0:
                    conv_result += signal_segment[len(signal_segment) - 1 - j] * np.conj(wavelet[j])
            
            # パワー計算
            power_spectrum[i] = np.abs(conv_result)**2
    
    return power_spectrum


@njit(nopython=True, fastmath=True, cache=True)
def kalman_filter_update(
    x_prev: np.ndarray,      # 前の状態 [period, velocity]
    P_prev: np.ndarray,      # 前の共分散行列
    measurement: float,       # 新しい測定値
    q: float,                # プロセスノイズ
    r: float                 # 測定ノイズ
) -> Tuple[np.ndarray, np.ndarray]:
    """
    カルマンフィルター更新（Numba最適化版）
    """
    # 状態遷移行列
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    # 測定関数
    H = np.array([1.0, 0.0])
    # プロセスノイズ共分散
    Q = np.array([[q, 0.0], [0.0, q]])
    # 測定ノイズ共分散
    R = np.array([r])
    
    # 予測ステップ
    x_pred = F @ x_prev
    P_pred = F @ P_prev @ F.T + Q
    
    # 更新ステップ
    y = measurement - H @ x_pred  # 残差
    S = H @ P_pred @ H.T + R      # 残差共分散
    K = (P_pred @ H.T) / S[0]     # カルマンゲイン
    
    x_new = x_pred + K * y
    I_KH = np.eye(2) - np.outer(K, H)
    P_new = I_KH @ P_pred
    
    return x_new, P_new


@njit(nopython=True, fastmath=True, cache=True)
def calculate_quantum_wavelet_cycle_numba(
    price: np.ndarray,
    min_period: int = 8,
    max_period: int = 60,
    history_size: int = 100,
    kalman_q: float = 0.05,
    kalman_r: float = 2.0,
    cycle_part: float = 0.5,
    max_output: int = 34,
    min_output: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantum-Inspired Wavelet Cycle DetectorのNumba実装
    
    Args:
        price: 価格データの配列
        min_period: 最小周期
        max_period: 最大周期
        history_size: 履歴サイズ
        kalman_q: カルマンフィルタープロセスノイズ
        kalman_r: カルマンフィルター測定ノイズ
        cycle_part: サイクル部分の倍率
        max_output: 最大出力値
        min_output: 最小出力値
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ドミナントサイクル, 生の周期, 平滑化周期)
    """
    n = len(price)
    periods = np.arange(min_period, max_period + 1, dtype=np.float64)
    
    # 結果配列初期化
    dom_cycle = np.full(n, float(min_period))
    raw_period = np.full(n, float(min_period))
    smooth_period = np.full(n, float(min_period))
    
    # カルマンフィルター状態初期化
    x = np.array([float(min_period), 0.0])  # [period, velocity]
    P = np.eye(2) * 10.0  # 初期共分散
    
    # 各データポイントで処理
    for i in range(n):
        if i < history_size:
            # 十分なデータがない場合
            dom_cycle[i] = min_period
            raw_period[i] = min_period
            smooth_period[i] = min_period
            continue
        
        # 履歴データ取得
        start_idx = i - history_size + 1
        price_history = price[start_idx:i+1]
        
        # トレンド除去
        detrended = detrend_series(price_history)
        
        if len(detrended) < max_period:
            dom_cycle[i] = dom_cycle[i-1] if i > 0 else min_period
            raw_period[i] = raw_period[i-1] if i > 0 else min_period
            smooth_period[i] = smooth_period[i-1] if i > 0 else min_period
            continue
        
        # CWTパワースペクトラム計算
        power_spectrum = calculate_cwt_power(detrended, periods)
        
        # 総パワーチェック
        total_power = np.sum(power_spectrum)
        if total_power < 1e-9:
            # 信号が平坦の場合、前の値を維持
            dom_cycle[i] = dom_cycle[i-1] if i > 0 else min_period
            raw_period[i] = raw_period[i-1] if i > 0 else min_period
            smooth_period[i] = smooth_period[i-1] if i > 0 else min_period
            continue
        
        # 確率的重心計算（期待値）
        probabilities = power_spectrum / total_power
        raw_dominant_period = np.sum(periods * probabilities)
        raw_period[i] = raw_dominant_period
        
        # カルマンフィルター更新
        x, P = kalman_filter_update(x, P, raw_dominant_period, kalman_q, kalman_r)
        
        # 結果をクランプ
        filtered_period = max(min_period, min(max_period, x[0]))
        smooth_period[i] = filtered_period
        
        # 最終出力計算
        cycle_value = int(np.round(filtered_period * cycle_part))
        dom_cycle[i] = max(min_output, min(max_output, cycle_value))
    
    return dom_cycle, raw_period, smooth_period


class QuantumWaveletCycle(EhlersDominantCycle):
    """
    Quantum-Inspired Wavelet Cycle Detector (QWCD)
    
    このディテクターは以下を組み合わせて市場の支配的サイクルを特定します：
    1. 時間-周波数解析のための連続ウェーブレット変換（CWT）
    2. 生の支配的サイクルを見つけるための確率的重心アプローチ
    3. 極低遅延でサイクル期間を平滑化するためのカルマンフィルター
    
    特徴:
    - 複素モーレットウェーブレットによる高精度時間-周波数解析
    - 確率論的アプローチによる支配的周期検出
    - カルマンフィルターによる極低遅延平滑化
    - 適応型ノイズ処理
    """
    
    # 許可されるソースタイプのリスト
    SRC_TYPES = ['close', 'hlc3', 'hl2', 'ohlc4']
    
    def __init__(
        self,
        min_period: int = 8,
        max_period: int = 60,
        history_size: int = 100,
        kalman_q: float = 0.05,
        kalman_r: float = 2.0,
        cycle_part: float = 0.5,
        max_output: int = 34,
        min_output: int = 1,
        src_type: str = 'close'
    ):
        """
        コンストラクタ
        
        Args:
            min_period: 検索する最小サイクル期間（デフォルト: 8）
            max_period: 検索する最大サイクル期間（デフォルト: 60）
            history_size: 解析用に保持する価格バーの数（デフォルト: 100）
            kalman_q: カルマンフィルタープロセスノイズ（低い値でより滑らかな出力）
            kalman_r: カルマンフィルター測定ノイズ（高い値でより滑らかな出力）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 1）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
        """
        super().__init__(
            f"QuantumWaveletCycle({min_period}, {max_period}, {history_size})",
            cycle_part,
            max_period,  # max_cycle
            min_period,  # min_cycle
            max_output,
            min_output
        )
        
        # パラメータ検証
        if min_period < 2 or max_period <= min_period or history_size < max_period:
            raise ValueError("無効なパラメータ制約です")
        
        self.min_period = min_period
        self.max_period = max_period
        self.history_size = history_size
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        
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
        Quantum Wavelet Cycle Detectorを使用してドミナントサイクルを計算する
        
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
            dom_cycle, raw_period, smooth_period = calculate_quantum_wavelet_cycle_numba(
                price,
                self.min_period,
                self.max_period,
                self.history_size,
                self.kalman_q,
                self.kalman_r,
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
            self.logger.error(f"QuantumWaveletCycle計算中にエラー: {error_msg}\n{stack_trace}")
            return np.array([])
    
    def get_kalman_state(self) -> Optional[dict]:
        """
        カルマンフィルターの内部状態を取得（デバッグ用）
        
        Returns:
            カルマンフィルターの状態辞書またはNone
        """
        if self._result is None:
            return None
        
        return {
            'min_period': self.min_period,
            'max_period': self.max_period,
            'history_size': self.history_size,
            'kalman_q': self.kalman_q,
            'kalman_r': self.kalman_r,
            'cycle_part': self.cycle_part
        } 