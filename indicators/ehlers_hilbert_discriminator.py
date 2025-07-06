#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **エラーズ ヒルベルト判別機 (Ehlers Hilbert Discriminator)** 🎯

ジョンエラーズ氏の理論に基づく正確なヒルベルト変換による市場状態判別器：
- ヒルベルト変換による直交成分(I/Q)の生成
- 瞬間位相とレートの計算による市場状態の分析
- DC/AC成分の分析によるトレンド・サイクル判別
- 適応的閾値による高精度な状態判別

🌟 **判別ロジック (エラーズ理論):**
1. **In-Phase (I) & Quadrature (Q)成分**: ヒルベルト変換による直交信号生成
2. **瞬間位相**: arctan2(Q, I)による位相計算
3. **位相レート**: 位相の時間微分（瞬間周波数）
4. **DC成分優位**: 一方向性 → トレンドモード
5. **AC成分優位**: 振動性 → サイクルモード
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit, types
import traceback
import math

try:
    from .indicator import Indicator
    from .price_source import PriceSource
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource


@dataclass
class HilbertDiscriminatorResult:
    """エラーズ ヒルベルト判別機の計算結果"""
    trend_mode: np.ndarray           # トレンドモード判定 (1=トレンド, 0=サイクル)
    market_state: np.ndarray         # 市場状態 (0=レンジ, 1=弱トレンド, 2=強トレンド)
    in_phase: np.ndarray             # In-Phase成分 (I)
    quadrature: np.ndarray           # Quadrature成分 (Q)
    instantaneous_phase: np.ndarray  # 瞬間位相
    phase_rate: np.ndarray           # 位相レート（瞬間周波数）
    dc_component: np.ndarray         # DC成分（トレンド成分）
    ac_component: np.ndarray         # AC成分（サイクル成分）
    trend_strength: np.ndarray       # トレンド強度 (0-1)
    cycle_strength: np.ndarray       # サイクル強度 (0-1)
    amplitude: np.ndarray            # 瞬間振幅
    frequency: np.ndarray            # 正規化周波数
    confidence: np.ndarray           # 判別信頼度
    raw_values: np.ndarray          # 元の価格データ


@njit(fastmath=True, cache=True)
def calculate_hilbert_transform(
    prices: np.ndarray,
    filter_length: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    エラーズ式ヒルベルト変換フィルター
    
    Args:
        prices: 価格データ
        filter_length: フィルター長（奇数）
    
    Returns:
        (in_phase, quadrature): I/Q成分
    """
    n = len(prices)
    if n < filter_length * 2:
        return np.zeros(n), np.zeros(n)
    
    # ヒルベルト変換係数（エラーズ理論に基づく）
    # 7点FIRフィルターの係数
    hilbert_coeffs = np.array([0.0962, 0.5769, 0.5769, -0.5769, -0.5769, 0.5769, 0.0962])
    
    in_phase = np.zeros(n)
    quadrature = np.zeros(n)
    
    # ヒルベルト変換の実行
    for i in range(filter_length, n):
        # In-Phase成分（元信号の遅延版）
        in_phase[i] = prices[i - 3]  # 3点遅延
        
        # Quadrature成分（ヒルベルト変換）
        q_sum = 0.0
        for j in range(len(hilbert_coeffs)):
            if i - j >= 0:
                q_sum += hilbert_coeffs[j] * prices[i - j]
        quadrature[i] = q_sum
    
    # 初期値の補完
    for i in range(filter_length):
        if n > filter_length:
            in_phase[i] = in_phase[filter_length]
            quadrature[i] = quadrature[filter_length]
    
    return in_phase, quadrature


@njit(fastmath=True, cache=True)
def calculate_instantaneous_parameters(
    in_phase: np.ndarray,
    quadrature: np.ndarray,
    smoothing_factor: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    瞬間パラメータの計算
    
    Args:
        in_phase: In-Phase成分
        quadrature: Quadrature成分
        smoothing_factor: 平滑化係数
    
    Returns:
        (phase, phase_rate, amplitude, frequency): 瞬間パラメータ
    """
    n = len(in_phase)
    instantaneous_phase = np.zeros(n)
    phase_rate = np.zeros(n)
    amplitude = np.zeros(n)
    frequency = np.zeros(n)
    
    # 瞬間位相と振幅の計算
    for i in range(n):
        # 瞬間振幅
        amplitude[i] = math.sqrt(in_phase[i]**2 + quadrature[i]**2)
        
        # 瞬間位相 (-π to π)
        if in_phase[i] != 0.0:
            instantaneous_phase[i] = math.atan2(quadrature[i], in_phase[i])
        else:
            instantaneous_phase[i] = 0.0
    
    # 位相レート（位相微分）の計算
    for i in range(1, n):
        phase_diff = instantaneous_phase[i] - instantaneous_phase[i-1]
        
        # 位相ラッピングの修正
        if phase_diff > math.pi:
            phase_diff -= 2 * math.pi
        elif phase_diff < -math.pi:
            phase_diff += 2 * math.pi
        
        # 平滑化された位相レート
        if i == 1:
            phase_rate[i] = phase_diff
        else:
            phase_rate[i] = smoothing_factor * phase_diff + (1 - smoothing_factor) * phase_rate[i-1]
        
        # 正規化周波数（0-0.5）
        frequency[i] = abs(phase_rate[i]) / (2 * math.pi)
    
    return instantaneous_phase, phase_rate, amplitude, frequency


@njit(fastmath=True, cache=True)
def calculate_dc_ac_components(
    prices: np.ndarray,
    in_phase: np.ndarray,
    quadrature: np.ndarray,
    amplitude: np.ndarray,
    window: int = 14
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DC/AC成分の分析（エラーズ理論正規化版）
    
    Args:
        prices: 価格データ
        in_phase: In-Phase成分
        quadrature: Quadrature成分
        amplitude: 瞬間振幅
        window: 分析ウィンドウ
    
    Returns:
        (dc_component, ac_component): DC/AC成分
    """
    n = len(prices)
    dc_component = np.zeros(n)
    ac_component = np.zeros(n)
    
    for i in range(window, n):
        # DC成分（低周波成分、トレンド）- 過去データのみを使用した移動平均
        dc_sum = 0.0
        for j in range(1, window + 1):  # 現在のポイントを除外
            if i - j >= 0:
                dc_sum += prices[i - j]
        dc_component[i] = dc_sum / window
        
        # AC成分（高周波成分、サイクル）- 現在価格とDC成分の差の絶対値の移動平均
        # エラーズ理論: AC成分は価格のサイクル変動部分
        ac_sum = 0.0
        for j in range(window):
            if i - j >= 0:
                detrended_price = abs(prices[i - j] - dc_component[i])
                ac_sum += detrended_price
        
        ac_component[i] = ac_sum / window
        
        # 最小値を設定（数値安定性）
        if ac_component[i] < 1e-6:
            ac_component[i] = 1e-6
    
    # 初期値の補完
    for i in range(window):
        if n > window:
            dc_component[i] = dc_component[window]
            ac_component[i] = ac_component[window]
    
    return dc_component, ac_component


@njit(fastmath=True, cache=True)
def analyze_market_state_ehlers(
    phase_rate: np.ndarray,
    dc_component: np.ndarray,
    ac_component: np.ndarray,
    amplitude: np.ndarray,
    frequency: np.ndarray,
    window: int = 14,
    phase_rate_threshold: float = 0.05,
    dc_ac_ratio_threshold: float = 1.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    エラーズ理論による市場状態分析
    
    Args:
        phase_rate: 位相レート
        dc_component: DC成分
        ac_component: AC成分
        amplitude: 瞬間振幅
        frequency: 瞬間周波数
        window: 分析ウィンドウ
        phase_rate_threshold: 位相レート閾値
        dc_ac_ratio_threshold: DC/AC比率閾値
    
    Returns:
        (trend_mode, market_state, trend_strength, cycle_strength, confidence)
    """
    n = len(phase_rate)
    trend_mode = np.zeros(n, dtype=np.int8)
    market_state = np.zeros(n, dtype=np.int8)
    trend_strength = np.zeros(n)
    cycle_strength = np.zeros(n)
    confidence = np.zeros(n)
    
    for i in range(window, n):
        # 位相レートの安定性分析
        phase_rate_std = 0.0
        phase_rate_mean = 0.0
        phase_rate_window = phase_rate[i-window:i]
        
        # 平均と標準偏差の計算
        for j in range(window):
            phase_rate_mean += phase_rate_window[j]
        phase_rate_mean /= window
        
        for j in range(window):
            diff = phase_rate_window[j] - phase_rate_mean
            phase_rate_std += diff * diff
        phase_rate_std = math.sqrt(phase_rate_std / window)
        
        # DC/AC比率の計算
        dc_value = dc_component[i]
        ac_value = ac_component[i]
        
        dc_ac_ratio = 1.0
        if ac_value > 1e-10:
            dc_ac_ratio = abs(dc_value) / ac_value
        
        # エラーズ判別ロジック（理論準拠修正版）
        trend_score = 0.0
        cycle_score = 0.0
        
        # 1. DC/AC比率分析（主要判別要因）
        # エラーズ理論: DC/AC比率が判別の最重要要因
        # 過去データベースDC成分での適切な閾値
        adjusted_threshold = dc_ac_ratio_threshold * 5.0  # 1.2 -> 6.0
        if dc_ac_ratio > adjusted_threshold:
            trend_score += 2.0  # DC優位 = トレンド
        else:
            cycle_score += 2.0  # AC優位 = サイクル
        
        # 2. 瞬間振幅の分析（重要要因）
        amp_mean = 0.0
        for j in range(window):
            amp_mean += amplitude[i-window+j]
        amp_mean /= window
        
        # 振幅が大きい = サイクル的、小さい = トレンド的
        relative_amplitude = amp_mean / (abs(dc_value) + 1e-8)
        if relative_amplitude > 0.1:  # 相対振幅が大きい
            cycle_score += 1.5
        else:  # 相対振幅が小さい
            trend_score += 1.5
        
        # 3. 位相レート変動性分析
        abs_phase_rate_mean = 0.0
        for j in range(window):
            abs_phase_rate_mean += abs(phase_rate_window[j])
        abs_phase_rate_mean /= window
        
        # エラーズ理論: 大きな位相レート = サイクル、小さな位相レート = トレンド
        if abs_phase_rate_mean > phase_rate_threshold:
            cycle_score += 1.0  # 大きな位相レート = サイクル
        else:
            trend_score += 1.0  # 小さな位相レート = トレンド
        
        # 4. 周波数安定性分析
        freq_std = 0.0
        freq_mean = 0.0
        freq_window = frequency[i-window:i]
        
        for j in range(window):
            freq_mean += freq_window[j]
        freq_mean /= window
        
        for j in range(window):
            diff = freq_window[j] - freq_mean
            freq_std += diff * diff
        freq_std = math.sqrt(freq_std / window)
        
        # 周波数域判別
        if freq_mean > 0.1:  # 高周波数
            cycle_score += 1.0
        elif freq_mean < 0.05:  # 低周波数
            trend_score += 1.0
        
        # 周波数安定性判別
        if freq_std < 0.05:  # 安定した周波数
            if freq_mean > 0.05:  # 安定した中-高周波数
                cycle_score += 0.5
            else:  # 安定した低周波数
                trend_score += 0.5
        
        # 5. 位相レート方向性の一貫性
        phase_rate_consistency = 0.0
        for j in range(1, window):
            if abs(phase_rate_window[j]) > 1e-8 and abs(phase_rate_window[j-1]) > 1e-8:
                if (phase_rate_window[j] * phase_rate_window[j-1]) > 0:
                    phase_rate_consistency += 1.0
        phase_rate_consistency /= (window - 1)
        
        if phase_rate_consistency > 0.7:  # 位相レートが一貫
            if abs_phase_rate_mean > phase_rate_threshold * 0.5:
                cycle_score += 0.5  # 一貫した大きな位相レート = サイクル
            else:
                trend_score += 0.5  # 一貫した小さな位相レート = トレンド
        
        # 最終判別
        total_score = trend_score + cycle_score
        if total_score > 0:
            trend_strength[i] = trend_score / total_score
            cycle_strength[i] = cycle_score / total_score
        else:
            trend_strength[i] = 0.5
            cycle_strength[i] = 0.5
        
        # モード判定（バランス調整後の閾値）
        if trend_strength[i] > 0.52:  # より中立的な閾値に調整
            trend_mode[i] = 1  # トレンドモード
        else:
            trend_mode[i] = 0  # サイクルモード
        
        # 市場状態判定（バランス調整後の閾値）
        if trend_strength[i] > 0.70:  # 強トレンド閾値
            market_state[i] = 2  # 強トレンド
        elif trend_strength[i] > 0.52:  # 弱トレンド閾値
            market_state[i] = 1  # 弱トレンド
        else:
            market_state[i] = 0  # レンジ
        
        # 信頼度計算
        strength_diff = abs(trend_strength[i] - cycle_strength[i])
        confidence[i] = min(strength_diff * 2, 1.0)
    
    # 初期値の補完
    for i in range(window):
        if n > window:
            trend_mode[i] = trend_mode[window]
            market_state[i] = market_state[window]
            trend_strength[i] = trend_strength[window]
            cycle_strength[i] = cycle_strength[window]
            confidence[i] = confidence[window]
    
    return trend_mode, market_state, trend_strength, cycle_strength, confidence


@njit(fastmath=True, cache=True)
def calculate_ehlers_hilbert_discriminator(
    prices: np.ndarray,
    filter_length: int = 7,
    smoothing_factor: float = 0.2,
    analysis_window: int = 14,
    phase_rate_threshold: float = 0.05,
    dc_ac_ratio_threshold: float = 1.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray]:
    """
    エラーズ ヒルベルト判別機のメイン計算関数
    
    Args:
        prices: 価格データ
        filter_length: ヒルベルトフィルター長
        smoothing_factor: 平滑化係数
        analysis_window: 分析ウィンドウ
        phase_rate_threshold: 位相レート閾値
        dc_ac_ratio_threshold: DC/AC比率閾値
    
    Returns:
        Tuple containing all calculated arrays
    """
    n = len(prices)
    
    if n < filter_length * 3:
        # データ不足の場合、デフォルト値を返す
        empty_result = (
            np.zeros(n, dtype=np.int8),  # trend_mode
            np.zeros(n, dtype=np.int8),  # market_state
            np.zeros(n),  # in_phase
            np.zeros(n),  # quadrature
            np.zeros(n),  # instantaneous_phase
            np.zeros(n),  # phase_rate
            np.zeros(n),  # dc_component
            np.zeros(n),  # ac_component
            np.full(n, 0.5),  # trend_strength
            np.full(n, 0.5),  # cycle_strength
            np.zeros(n),  # amplitude
            np.zeros(n),  # frequency
            np.full(n, 0.5)   # confidence
        )
        return empty_result
    
    # 1. ヒルベルト変換による I/Q成分計算
    in_phase, quadrature = calculate_hilbert_transform(prices, filter_length)
    
    # 2. 瞬間パラメータの計算
    instantaneous_phase, phase_rate, amplitude, frequency = calculate_instantaneous_parameters(
        in_phase, quadrature, smoothing_factor
    )
    
    # 3. DC/AC成分の分析
    dc_component, ac_component = calculate_dc_ac_components(
        prices, in_phase, quadrature, amplitude, analysis_window
    )
    
    # 4. エラーズ市場状態分析
    trend_mode, market_state, trend_strength, cycle_strength, confidence = analyze_market_state_ehlers(
        phase_rate, dc_component, ac_component, amplitude, frequency,
        analysis_window, phase_rate_threshold, dc_ac_ratio_threshold
    )
    
    return (trend_mode, market_state, in_phase, quadrature, instantaneous_phase,
            phase_rate, dc_component, ac_component, trend_strength, cycle_strength,
            amplitude, frequency, confidence)


class EhlersHilbertDiscriminator(Indicator):
    """
    エラーズ ヒルベルト判別機インジケーター
    
    ジョンエラーズ氏のヒルベルト変換理論に基づく市場状態判別：
    - ヒルベルト変換による直交成分の生成
    - 瞬間位相と位相レートの正確な計算
    - DC/AC成分分析によるトレンド・サイクル判別
    - エラーズ理論準拠の高精度判別ロジック
    
    特徴:
    - 位相レート安定性による主要判別
    - DC/AC比率による補助判別
    - 適応的閾値による高精度判定
    - リアルタイム市場状態監視
    """
    
    def __init__(
        self,
        src_type: str = 'close',                    # 価格ソース
        filter_length: int = 7,                     # ヒルベルトフィルター長
        smoothing_factor: float = 0.2,              # 平滑化係数
        analysis_window: int = 14,                  # 分析ウィンドウ
        phase_rate_threshold: float = 0.05,         # 位相レート閾値（0.1→0.05に調整）
        dc_ac_ratio_threshold: float = 1.2          # DC/AC比率閾値（1.5→1.2に調整）
    ):
        """
        コンストラクタ
        
        Args:
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4')
            filter_length: ヒルベルトフィルター長（推奨: 7）
            smoothing_factor: 位相レート平滑化係数（0-1）
            analysis_window: 市場状態分析ウィンドウ
            phase_rate_threshold: 位相レート安定性閾値
            dc_ac_ratio_threshold: DC/AC比率判別閾値
        """
        # 指標名の作成
        indicator_name = f"EhlersHD(src={src_type}, len={filter_length}, α={smoothing_factor})"
        super().__init__(indicator_name)
        
        # パラメータ保存
        self.src_type = src_type.lower()
        self.filter_length = filter_length
        self.smoothing_factor = smoothing_factor
        self.analysis_window = analysis_window
        self.phase_rate_threshold = phase_rate_threshold
        self.dc_ac_ratio_threshold = dc_ac_ratio_threshold
        
        # パラメータ検証
        if self.filter_length < 3 or self.filter_length > 21:
            raise ValueError("フィルター長は3-21の範囲で指定してください")
        
        if self.smoothing_factor <= 0 or self.smoothing_factor >= 1:
            raise ValueError("平滑化係数は0-1の範囲で指定してください")
        
        # ソースタイプの検証
        valid_sources = ['close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open']
        if self.src_type not in valid_sources:
            raise ValueError(f"無効なソースタイプ: {src_type}。有効なオプション: {', '.join(valid_sources)}")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    first_val = float(data.iloc[0].get('close', data.iloc[0, -1]))
                    last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1]))
                else:
                    first_val = last_val = 0.0
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_val = float(data[0, -1])
                        last_val = float(data[-1, -1])
                    else:
                        first_val = float(data[0])
                        last_val = float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            params_sig = f"{self.filter_length}_{self.smoothing_factor}_{self.analysis_window}_{self.src_type}"
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.filter_length}_{self.smoothing_factor}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HilbertDiscriminatorResult:
        """
        エラーズ ヒルベルト判別機を計算
        
        Args:
            data: 価格データ（DataFrameまたは配列）
        
        Returns:
            HilbertDiscriminatorResult: 判別結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return HilbertDiscriminatorResult(
                    trend_mode=cached_result.trend_mode.copy(),
                    market_state=cached_result.market_state.copy(),
                    in_phase=cached_result.in_phase.copy(),
                    quadrature=cached_result.quadrature.copy(),
                    instantaneous_phase=cached_result.instantaneous_phase.copy(),
                    phase_rate=cached_result.phase_rate.copy(),
                    dc_component=cached_result.dc_component.copy(),
                    ac_component=cached_result.ac_component.copy(),
                    trend_strength=cached_result.trend_strength.copy(),
                    cycle_strength=cached_result.cycle_strength.copy(),
                    amplitude=cached_result.amplitude.copy(),
                    frequency=cached_result.frequency.copy(),
                    confidence=cached_result.confidence.copy(),
                    raw_values=cached_result.raw_values.copy()
                )
            
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            
            if len(src_prices) < self.filter_length * 3:
                return self._create_empty_result(len(src_prices), src_prices)
            
            # エラーズ ヒルベルト判別機計算
            (trend_mode, market_state, in_phase, quadrature, instantaneous_phase,
             phase_rate, dc_component, ac_component, trend_strength, cycle_strength,
             amplitude, frequency, confidence) = calculate_ehlers_hilbert_discriminator(
                src_prices, self.filter_length, self.smoothing_factor, self.analysis_window,
                self.phase_rate_threshold, self.dc_ac_ratio_threshold
            )
            
            # 結果作成
            result = HilbertDiscriminatorResult(
                trend_mode=trend_mode.copy(),
                market_state=market_state.copy(),
                in_phase=in_phase.copy(),
                quadrature=quadrature.copy(),
                instantaneous_phase=instantaneous_phase.copy(),
                phase_rate=phase_rate.copy(),
                dc_component=dc_component.copy(),
                ac_component=ac_component.copy(),
                trend_strength=trend_strength.copy(),
                cycle_strength=cycle_strength.copy(),
                amplitude=amplitude.copy(),
                frequency=frequency.copy(),
                confidence=confidence.copy(),
                raw_values=src_prices.copy()
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = trend_mode.astype(float)  # 基底クラス用
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"エラーズHD計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                return self._create_empty_result(len(src_prices), src_prices)
            else:
                return self._create_empty_result(0, np.array([]))
    
    def _create_empty_result(self, length: int, raw_prices: np.ndarray) -> HilbertDiscriminatorResult:
        """空の結果を作成"""
        return HilbertDiscriminatorResult(
            trend_mode=np.full(length, 0, dtype=np.int8),
            market_state=np.full(length, 0, dtype=np.int8),
            in_phase=np.full(length, np.nan),
            quadrature=np.full(length, np.nan),
            instantaneous_phase=np.full(length, np.nan),
            phase_rate=np.full(length, np.nan),
            dc_component=np.full(length, np.nan),
            ac_component=np.full(length, np.nan),
            trend_strength=np.full(length, 0.5),
            cycle_strength=np.full(length, 0.5),
            amplitude=np.full(length, np.nan),
            frequency=np.full(length, np.nan),
            confidence=np.full(length, 0.5),
            raw_values=raw_prices
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """トレンドモード判定結果を取得 (1=トレンド, 0=サイクル)"""
        if not self._result_cache:
            return None
        
        result = self._get_latest_result()
        return result.trend_mode.astype(float).copy() if result else None
    
    def get_trend_mode(self) -> Optional[np.ndarray]:
        """トレンドモード判定を取得"""
        result = self._get_latest_result()
        return result.trend_mode.copy() if result else None
    
    def get_market_state(self) -> Optional[np.ndarray]:
        """市場状態を取得 (0=レンジ, 1=弱トレンド, 2=強トレンド)"""
        result = self._get_latest_result()
        return result.market_state.copy() if result else None
    
    def get_trend_strength(self) -> Optional[np.ndarray]:
        """トレンド強度を取得 (0-1)"""
        result = self._get_latest_result()
        return result.trend_strength.copy() if result else None
    
    def get_cycle_strength(self) -> Optional[np.ndarray]:
        """サイクル強度を取得 (0-1)"""
        result = self._get_latest_result()
        return result.cycle_strength.copy() if result else None
    
    def get_phase_components(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """位相成分を取得 (in_phase, quadrature, instantaneous_phase)"""
        result = self._get_latest_result()
        if result:
            return (result.in_phase.copy(), result.quadrature.copy(), 
                    result.instantaneous_phase.copy())
        return None
    
    def get_phase_rate(self) -> Optional[np.ndarray]:
        """位相レート（瞬間周波数）を取得"""
        result = self._get_latest_result()
        return result.phase_rate.copy() if result else None
    
    def get_dc_ac_components(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """DC/AC成分を取得"""
        result = self._get_latest_result()
        if result:
            return (result.dc_component.copy(), result.ac_component.copy())
        return None
    
    def get_confidence(self) -> Optional[np.ndarray]:
        """判別信頼度を取得"""
        result = self._get_latest_result()
        return result.confidence.copy() if result else None
    
    def is_trend_mode(self, index: int = -1) -> bool:
        """指定インデックスでトレンドモードかどうかを判定"""
        trend_mode = self.get_trend_mode()
        if trend_mode is None or len(trend_mode) == 0:
            return False
        if index == -1:
            index = len(trend_mode) - 1
        return bool(trend_mode[index] == 1)
    
    def is_cycle_mode(self, index: int = -1) -> bool:
        """指定インデックスでサイクルモードかどうかを判定"""
        return not self.is_trend_mode(index)
    
    def get_current_market_state_description(self, index: int = -1) -> str:
        """現在の市場状態の説明を取得"""
        market_state = self.get_market_state()
        trend_strength = self.get_trend_strength()
        cycle_strength = self.get_cycle_strength()
        confidence = self.get_confidence()
        
        if any(x is None for x in [market_state, trend_strength, cycle_strength, confidence]):
            return "データ不足"
        
        if index == -1:
            index = len(market_state) - 1
        
        state = market_state[index]
        t_strength = trend_strength[index]
        c_strength = cycle_strength[index]
        conf = confidence[index]
        
        mode = "トレンドモード" if self.is_trend_mode(index) else "サイクルモード"
        
        if state == 2:
            state_desc = "強トレンド"
        elif state == 1:
            state_desc = "弱トレンド"
        else:
            state_desc = "レンジ"
        
        return f"{mode} - {state_desc} (T:{t_strength:.2f}, C:{c_strength:.2f}, 信頼度:{conf:.2f})"
    
    def _get_latest_result(self) -> Optional[HilbertDiscriminatorResult]:
        """最新の結果を取得"""
        if not self._result_cache:
            return None
        
        if self._cache_keys:
            return self._result_cache[self._cache_keys[-1]]
        else:
            return next(iter(self._result_cache.values()))
    
    def get_discriminator_metadata(self) -> Dict:
        """判別機のメタデータを取得"""
        result = self._get_latest_result()
        if not result:
            return {}
        
        trend_mode_pct = np.mean(result.trend_mode) * 100
        avg_trend_strength = np.nanmean(result.trend_strength)
        avg_cycle_strength = np.nanmean(result.cycle_strength)
        avg_confidence = np.nanmean(result.confidence)
        
        return {
            'discriminator_type': 'Ehlers Hilbert Discriminator',
            'src_type': self.src_type,
            'filter_length': self.filter_length,
            'smoothing_factor': self.smoothing_factor,
            'analysis_window': self.analysis_window,
            'data_points': len(result.trend_mode),
            'trend_mode_percentage': trend_mode_pct,
            'avg_trend_strength': avg_trend_strength,
            'avg_cycle_strength': avg_cycle_strength,
            'avg_confidence': avg_confidence,
            'phase_rate_threshold': self.phase_rate_threshold,
            'dc_ac_ratio_threshold': self.dc_ac_ratio_threshold
        }
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = [] 