#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌀 **Hilbert Transform Unified V1.0 - ヒルベルト変換統合システム** 🌀

複数のインジケーターファイル（ultimate_ma.py, ultimate_breakout_channel.py, 
ultimate_chop_trend.py, ultimate_chop_trend_v2.py, ultimate_volatility.py, 
quantum_supreme_breakout_channel.py等）で実装されているヒルベルト変換アルゴリズムを統合し、
単一のインターフェースで利用可能にします。

🌟 **統合されたヒルベルト変換:**
1. **基本ヒルベルト変換**: ultimate_ma.pyから（基本4点変換）
2. **量子強化ヒルベルト変換**: ultimate_breakout_channel.py、ultimate_volatility.pyから
3. **瞬時解析ヒルベルト変換**: ultimate_chop_trend.pyから（位相・周波数・振幅解析）
4. **瞬時解析V2**: ultimate_chop_trend_v2.pyから（簡易高速版）
5. **Supreme版**: quantum_supreme系から（9点高精度版）
6. **NumPy FFT版**: zen_efficiency_ratio.pyから（FFT近似版）
7. **マルチ解像度版**: cosmic_universal系から（ウェーブレット統合版）

🎨 **設計パターン:**
- EhlersUnifiedDCの設計パターンに従った実装
- 統一されたパラメータインターフェース
- 動的アルゴリズム切り替え機能
- Numba最適化による高速化
- 一貫した結果形式
"""

from typing import Union, Optional, Dict, Tuple, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit
import warnings
import math
warnings.filterwarnings('ignore')

try:
    from .indicator import Indicator
    from .price_source import PriceSource
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource


class HilbertTransformResult(NamedTuple):
    """Hilbert Transform統合結果"""
    amplitude: np.ndarray              # 瞬時振幅
    phase: np.ndarray                  # 瞬時位相
    frequency: np.ndarray              # 瞬時周波数
    algorithm_type: str                # 使用されたアルゴリズムタイプ
    
    # アルゴリズム固有のフィールド（オプショナル）
    trend_component: Optional[np.ndarray] = None     # トレンド成分（chop_trend系）
    trend_strength: Optional[np.ndarray] = None      # トレンド強度（quantum系）
    quantum_entanglement: Optional[np.ndarray] = None # 量子もつれ（quantum系）
    quantum_coherence: Optional[np.ndarray] = None   # 量子コヒーレンス（supreme系）
    wavelet_energy: Optional[np.ndarray] = None      # ウェーブレットエネルギー（multiresolution系）
    cycle_phase: Optional[np.ndarray] = None         # サイクル位相（fft系）
    confidence_score: Optional[np.ndarray] = None    # 信頼度スコア


# === 1. 基本ヒルベルト変換（ultimate_ma.pyから） ===

@njit(fastmath=True, cache=True)
def basic_hilbert_transform_numba(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    🌀 基本ヒルベルト変換フィルター（4点FIR近似）
    ultimate_ma.pyから移植
    """
    n = len(prices)
    amplitude = np.zeros(n)
    phase = np.zeros(n)
    frequency = np.zeros(n)
    
    if n < 8:
        return amplitude, phase, frequency
    
    # 4点ヒルベルト変換（FIRフィルター近似）
    for i in range(4, n-4):
        # 実部（元信号）
        real_part = prices[i]
        
        # 虚部（ヒルベルト変換）- 90度位相シフト
        imag_part = (prices[i-3] - prices[i+3] + 
                    3 * (prices[i+1] - prices[i-1])) / 8.0
        
        # 瞬時振幅
        amplitude[i] = np.sqrt(real_part**2 + imag_part**2)
        
        # 瞬時位相
        if real_part != 0:
            phase[i] = np.arctan2(imag_part, real_part)
        
        # 瞬時周波数（位相微分）
        if i > 4:
            phase_diff = phase[i] - phase[i-1]
            # 位相ラッピング補正
            if phase_diff > np.pi:
                phase_diff -= 2 * np.pi
            elif phase_diff < -np.pi:
                phase_diff += 2 * np.pi
            frequency[i] = abs(phase_diff) / (2 * np.pi)
    
    # 境界値の処理
    for i in range(4):
        amplitude[i] = amplitude[4] if n > 4 else 0.0
        phase[i] = phase[4] if n > 4 else 0.0
        frequency[i] = frequency[4] if n > 4 else 0.0
    for i in range(n-4, n):
        amplitude[i] = amplitude[n-5] if n > 4 else 0.0
        phase[i] = phase[n-5] if n > 4 else 0.0
        frequency[i] = frequency[n-5] if n > 4 else 0.0
    
    return amplitude, phase, frequency


# === 2. 量子強化ヒルベルト変換（ultimate_breakout_channel.py, ultimate_volatility.pyから） ===

@njit(fastmath=True, cache=True)
def quantum_enhanced_hilbert_transform_numba(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🌀 量子強化ヒルベルト変換分析
    ultimate_breakout_channel.py, ultimate_volatility.pyから移植
    """
    n = len(prices)
    amplitude = np.zeros(n)
    phase = np.zeros(n)
    frequency = np.zeros(n)
    trend_strength = np.zeros(n)
    quantum_entanglement = np.zeros(n)
    
    if n < 16:
        return amplitude, phase, frequency, trend_strength, quantum_entanglement
    
    # 量子強化ヒルベルト変換
    for i in range(8, n-8):
        # 実部（元信号の平滑化）
        real_part = (prices[i] + prices[i-1] + prices[i-2] + prices[i-3]) / 4.0
        
        # 虚部（量子強化ヒルベルト変換）
        imag_part = (prices[i-3] - prices[i+3] + 
                    3 * (prices[i+1] - prices[i-1])) / 8.0
        
        # 瞬時振幅（量子もつれ補正）
        amplitude[i] = np.sqrt(real_part**2 + imag_part**2)
        
        # 瞬時位相
        if real_part != 0:
            phase[i] = np.arctan2(imag_part, real_part)
        
        # 瞬時周波数
        if i > 8:
            phase_diff = phase[i] - phase[i-1]
            # 位相ラッピング補正
            if phase_diff > np.pi:
                phase_diff -= 2 * np.pi
            elif phase_diff < -np.pi:
                phase_diff += 2 * np.pi
            frequency[i] = abs(phase_diff) / (2 * np.pi)
        
        # トレンド強度計算（位相ベース）
        if i >= 16:
            phase_momentum = 0.0
            for j in range(8):
                phase_momentum += np.sin(phase[i-j])
            phase_momentum /= 8.0
            trend_strength[i] = min(abs(phase_momentum), 1.0)
        
        # 量子もつれ効果計算
        if i >= 16:
            entanglement = 0.0
            for j in range(1, min(6, i-8)):
                correlation = (prices[i] - prices[i-j]) * (prices[i-j] - prices[i-j-1])
                if correlation != 0:
                    entanglement += np.sin(np.pi * correlation / (abs(correlation) + 1e-10))
            quantum_entanglement[i] = abs(entanglement) / 5.0
    
    # 境界値処理
    for i in range(8):
        amplitude[i] = amplitude[8] if n > 8 else 0.0
        phase[i] = phase[8] if n > 8 else 0.0
        frequency[i] = frequency[8] if n > 8 else 0.0
        trend_strength[i] = trend_strength[8] if n > 8 else 0.0
        quantum_entanglement[i] = quantum_entanglement[8] if n > 8 else 0.0
    for i in range(n-8, n):
        amplitude[i] = amplitude[n-9] if n > 8 else 0.0
        phase[i] = phase[n-9] if n > 8 else 0.0
        frequency[i] = frequency[n-9] if n > 8 else 0.0
        trend_strength[i] = trend_strength[n-9] if n > 8 else 0.0
        quantum_entanglement[i] = quantum_entanglement[n-9] if n > 8 else 0.0
    
    return amplitude, phase, frequency, trend_strength, quantum_entanglement


# === 3. 瞬時解析ヒルベルト変換（ultimate_chop_trend.pyから） ===

@njit(fastmath=True, cache=True)
def instantaneous_analysis_hilbert_transform_numba(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ヒルベルト変換による瞬時解析
    ultimate_chop_trend.pyから移植
    """
    n = len(prices)
    if n < 50:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    phase = np.zeros(n)
    frequency = np.zeros(n)
    amplitude = np.zeros(n)
    trend_component = np.zeros(n)
    
    # 位相差分を使った瞬時周波数の近似
    for i in range(7, n):
        # 4つの位相成分を計算
        real_part = (prices[i] + prices[i-2] + prices[i-4] + prices[i-6]) / 4.0
        
        # 90度位相をずらした虚数部の近似
        imag_part = (prices[i-1] + prices[i-3] + prices[i-5] + prices[i-7]) / 4.0
        
        # 瞬時位相
        if real_part != 0:
            phase[i] = math.atan2(imag_part, real_part)
        
        # 瞬時振幅
        amplitude[i] = math.sqrt(real_part * real_part + imag_part * imag_part)
        
        # 瞬時周波数（位相の差分）
        if i > 7:
            freq_diff = phase[i] - phase[i-1]
            # 位相の巻き戻しを修正
            if freq_diff > math.pi:
                freq_diff -= 2 * math.pi
            elif freq_diff < -math.pi:
                freq_diff += 2 * math.pi
            frequency[i] = abs(freq_diff)
        
        # トレンド成分（位相の方向性）
        if i > 14:
            phase_trend = 0.0
            for j in range(7):
                phase_trend += math.sin(phase[i-j])
            trend_component[i] = phase_trend / 7.0
    
    return phase, frequency, amplitude, trend_component


# === 4. 瞬時解析V2（ultimate_chop_trend_v2.pyから） ===

@njit(fastmath=True, cache=True)
def hilbert_instantaneous_analysis_v2_numba(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    🌀 瞬時ヒルベルト変換解析 V2（高速簡易版）
    ultimate_chop_trend_v2.pyから移植
    """
    n = len(prices)
    if n < 16:
        return np.full(n, 0.5), np.zeros(n), np.full(n, 0.5)
    
    hilbert_signal = np.full(n, 0.5)
    confidence = np.full(n, 0.5)
    phase = np.zeros(n)
    
    for i in range(8, n):
        # 8点ヒルベルト変換
        real_part = (prices[i] + prices[i-2] + prices[i-4] + prices[i-6]) * 0.25
        imag_part = (prices[i-1] + prices[i-3] + prices[i-5] + prices[i-7]) * 0.25
        
        # 瞬時位相
        if real_part != 0:
            phase[i] = math.atan2(imag_part, real_part)
        
        # 瞬時トレンド方向
        if i >= 15:
            phase_momentum = 0.0
            for j in range(7):
                phase_momentum += math.sin(phase[i-j])
            phase_momentum /= 7.0
            
            # 正規化トレンドシグナル（0-1）
            trend_raw = math.tanh(phase_momentum * 2)
            hilbert_signal[i] = trend_raw * 0.5 + 0.5
            
            # 信頼度（振幅と位相安定性ベース）
            amplitude = math.sqrt(real_part**2 + imag_part**2)
            phase_stability = 1.0 / (1.0 + abs(phase_momentum) * 5.0)
            confidence[i] = min(amplitude * phase_stability, 1.0)
    
    return hilbert_signal, confidence, phase


# === 5. Supreme版（quantum_supreme系から） ===

@njit(fastmath=True, cache=True)
def quantum_supreme_hilbert_transform_numba(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    🌀 量子Supreme版ヒルベルト変換（9点高精度版）
    quantum_supreme系から移植
    """
    n = len(prices)
    amplitude = np.zeros(n)
    phase = np.zeros(n)
    frequency = np.zeros(n)
    quantum_coherence = np.zeros(n)
    
    if n < 16:
        return amplitude, phase, frequency, quantum_coherence
    
    # 改良ヒルベルト変換 - より高精度な計算
    for i in range(8, n-8):
        real_part = prices[i]
        
        # 9点ヒルベルト変換（より高精度）
        imag_part = (
            (prices[i-7] - prices[i+7]) +
            3 * (prices[i-5] - prices[i+5]) +
            5 * (prices[i-3] - prices[i+3]) +
            7 * (prices[i-1] - prices[i+1])
        ) / 32.0
        
        # 瞬時振幅
        amplitude[i] = np.sqrt(real_part**2 + imag_part**2)
        
        # 瞬時位相
        if real_part != 0:
            phase[i] = np.arctan2(imag_part, real_part)
        
        # 瞬時周波数（位相の時間微分）
        if i > 8:
            phase_diff = phase[i] - phase[i-1]
            # 位相のラッピング処理
            if phase_diff > np.pi:
                phase_diff -= 2 * np.pi
            elif phase_diff < -np.pi:
                phase_diff += 2 * np.pi
            frequency[i] = phase_diff
        
        # 🔬 量子コヒーレンス計算 - 位相の安定性を測定
        if i >= 16:
            # 過去8点での位相安定性
            phase_variance = 0.0
            for j in range(8):
                phase_variance += (phase[i-j] - phase[i-j-1])**2
            phase_variance /= 8.0
            
            # コヒーレンス = 1 / (1 + phase_variance)
            quantum_coherence[i] = 1.0 / (1.0 + phase_variance)
    
    # 境界値処理
    for i in range(8):
        amplitude[i] = amplitude[8] if n > 8 else 0.0
        phase[i] = phase[8] if n > 8 else 0.0
        frequency[i] = frequency[8] if n > 8 else 0.0
        quantum_coherence[i] = quantum_coherence[8] if n > 8 else 0.0
    for i in range(n-8, n):
        amplitude[i] = amplitude[n-9] if n > 8 else 0.0
        phase[i] = phase[n-9] if n > 8 else 0.0
        frequency[i] = frequency[n-9] if n > 8 else 0.0
        quantum_coherence[i] = quantum_coherence[n-9] if n > 8 else 0.0
    
    return amplitude, phase, frequency, quantum_coherence


# === 6. NumPy FFT版（zen_efficiency_ratio.pyから） ===

@njit(fastmath=True, cache=True)
def fft_hilbert_transform_numba(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NumPy FFTを使用したヒルベルト変換の実装（Numba対応）
    zen_efficiency_ratio.pyから移植
    """
    length = len(prices)
    
    if length < 4:
        return (np.full(length, np.nan), 
                np.full(length, np.nan), 
                np.full(length, np.nan))
    
    amplitude = np.full(length, np.nan)
    phase = np.full(length, np.nan)
    cycle_phase = np.full(length, np.nan)
    
    # 滑らかな包絡線の計算（ヒルベルト振幅の近似）
    window = min(21, length // 4)
    for i in range(window, length):
        # ローカル振幅の計算（標準偏差ベース）
        window_data = prices[i-window:i]
        amplitude[i] = np.std(window_data) * math.sqrt(2)  # 瞬時振幅の近似
        
        # 位相の計算（価格変化の方向とトレンドから推定）
        if i > window + 1:
            # 短期トレンドと長期トレンドの比較
            short_trend = np.mean(window_data[-window//2:]) - np.mean(window_data[:window//2])
            price_velocity = prices[i] - prices[i-1]
            
            # 位相推定（-π to π）
            if abs(short_trend) > 1e-10:
                phase[i] = math.atan2(price_velocity, short_trend)
            else:
                phase[i] = 0.0
            
            # サイクル位相（0 to 2π）
            cycle_phase[i] = (phase[i] + math.pi) % (2 * math.pi)
    
    return amplitude, phase, cycle_phase


# === 7. マルチ解像度版（cosmic_universal系から） ===

@njit(fastmath=True, cache=True)
def hilbert_wavelet_multiresolution_numba(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ヒルベルト・ウェーブレット多重解像度解析
    cosmic_universal系から移植
    """
    n = len(prices)
    amplitude = np.full(n, np.nan)
    phase = np.full(n, np.nan)
    wavelet_energy = np.full(n, np.nan)
    inst_frequency = np.full(n, np.nan)
    
    # ヒルベルト変換の近似実装
    for i in range(2, n-2):
        # 局所的な解析窓
        window_size = min(21, i, n-i-1)
        if window_size < 3:
            continue
            
        local_prices = prices[i-window_size:i+window_size+1]
        
        # 簡易ヒルベルト変換（位相90度シフト）
        if len(local_prices) >= 5:
            # 中心差分による微分近似
            derivative = (local_prices[4] - local_prices[0]) / 4.0
            
            # 解析信号の振幅
            real_part = local_prices[window_size]  # 中心の価格
            imag_part = derivative  # 90度位相シフト成分
            
            amplitude[i] = math.sqrt(real_part**2 + imag_part**2)
            phase[i] = math.atan2(imag_part, real_part)
            
            # ウェーブレットエネルギー（局所エネルギー密度）
            energy = 0.0
            for j in range(len(local_prices)-1):
                energy += (local_prices[j+1] - local_prices[j])**2
            wavelet_energy[i] = energy / len(local_prices)
            
            # 瞬間周波数（位相の時間微分）
            if i > 2 and not np.isnan(phase[i-1]):
                phase_diff = phase[i] - phase[i-1]
                # 位相の連続性を保つ
                while phase_diff > math.pi:
                    phase_diff -= 2 * math.pi
                while phase_diff < -math.pi:
                    phase_diff += 2 * math.pi
                inst_frequency[i] = abs(phase_diff) / (2 * math.pi)
    
    return amplitude, phase, wavelet_energy, inst_frequency


class HilbertTransformUnified(Indicator):
    """
    ヒルベルト変換統合システム - 複数のヒルベルト変換アルゴリズムを統合
    
    EhlersUnifiedDCの設計パターンに従った実装で、以下のアルゴリズムを統合：
    - basic: 基本ヒルベルト変換（4点FIR近似）
    - quantum_enhanced: 量子強化ヒルベルト変換（量子もつれ効果付き）
    - instantaneous: 瞬時解析ヒルベルト変換（トレンド成分解析）
    - instantaneous_v2: 瞬時解析V2（高速簡易版）
    - quantum_supreme: 量子Supreme版（9点高精度版）
    - fft_based: NumPy FFT版（FFT近似）
    - multiresolution: マルチ解像度版（ウェーブレット統合）
    """
    
    # 利用可能なアルゴリズムの定義
    _ALGORITHMS = {
        'basic': basic_hilbert_transform_numba,
        'quantum_enhanced': quantum_enhanced_hilbert_transform_numba,
        'instantaneous': instantaneous_analysis_hilbert_transform_numba,
        'instantaneous_v2': hilbert_instantaneous_analysis_v2_numba,
        'quantum_supreme': quantum_supreme_hilbert_transform_numba,
        'fft_based': fft_hilbert_transform_numba,
        'multiresolution': hilbert_wavelet_multiresolution_numba
    }
    
    # アルゴリズムの説明
    _ALGORITHM_DESCRIPTIONS = {
        'basic': '基本ヒルベルト変換（4点FIR近似・位相遅延ゼロ）',
        'quantum_enhanced': '量子強化ヒルベルト変換（量子もつれ効果・トレンド強度）',
        'instantaneous': '瞬時解析ヒルベルト変換（詳細位相・周波数・トレンド解析）',
        'instantaneous_v2': '瞬時解析V2（高速簡易版・信頼度付き）',
        'quantum_supreme': '量子Supreme版（9点高精度・量子コヒーレンス）',
        'fft_based': 'NumPy FFT版（FFT近似・サイクル位相）',
        'multiresolution': 'マルチ解像度版（ウェーブレット統合・エネルギー解析）'
    }
    
    def __init__(
        self,
        algorithm_type: str = 'basic',
        src_type: str = 'close',
        # 基本パラメータ
        min_periods: int = 16,
        # アルゴリズム固有パラメータ
        window_size: int = 21,  # multiresolution用
        phase_window: int = 8   # 位相解析窓サイズ
    ):
        """
        コンストラクタ
        
        Args:
            algorithm_type: 使用するアルゴリズムタイプ
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4')
            min_periods: 最小計算期間
            window_size: ウィンドウサイズ（multiresolution用）
            phase_window: 位相解析ウィンドウサイズ
        """
        # アルゴリズム名を小文字に変換して正規化
        algorithm_type = algorithm_type.lower()
        
        # アルゴリズムが有効かチェック
        if algorithm_type not in self._ALGORITHMS:
            valid_algorithms = ", ".join(self._ALGORITHMS.keys())
            raise ValueError(f"無効なアルゴリズムタイプです: {algorithm_type}。有効なオプション: {valid_algorithms}")
        
        # 親クラスの初期化
        name = f"HilbertUnified(type={algorithm_type}, src={src_type})"
        super().__init__(name)
        
        # パラメータ保存
        self.algorithm_type = algorithm_type
        self.src_type = src_type
        self.min_periods = min_periods
        self.window_size = window_size
        self.phase_window = phase_window
        
        # 結果のキャッシュ
        self._result: Optional[HilbertTransformResult] = None
        self._cache_hash: Optional[str] = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HilbertTransformResult:
        """
        ヒルベルト変換を計算
        
        Args:
            data: 価格データ
            
        Returns:
            HilbertTransformResult: ヒルベルト変換結果
        """
        # キャッシュチェック
        current_hash = self._get_data_hash(data)
        if self._cache_hash == current_hash and self._result is not None:
            return self._result
        
        try:
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            
            if len(src_prices) < self.min_periods:
                return self._create_empty_result(len(src_prices))
            
            # アルゴリズム実行
            algorithm_func = self._ALGORITHMS[self.algorithm_type]
            
            if self.algorithm_type == 'basic':
                amplitude, phase, frequency = algorithm_func(src_prices)
                result = HilbertTransformResult(
                    amplitude=amplitude,
                    phase=phase,
                    frequency=frequency,
                    algorithm_type=self.algorithm_type
                )
            
            elif self.algorithm_type == 'quantum_enhanced':
                amplitude, phase, frequency, trend_strength, quantum_entanglement = algorithm_func(src_prices)
                result = HilbertTransformResult(
                    amplitude=amplitude,
                    phase=phase,
                    frequency=frequency,
                    trend_strength=trend_strength,
                    quantum_entanglement=quantum_entanglement,
                    algorithm_type=self.algorithm_type
                )
            
            elif self.algorithm_type == 'instantaneous':
                phase, frequency, amplitude, trend_component = algorithm_func(src_prices)
                result = HilbertTransformResult(
                    amplitude=amplitude,
                    phase=phase,
                    frequency=frequency,
                    trend_component=trend_component,
                    algorithm_type=self.algorithm_type
                )
            
            elif self.algorithm_type == 'instantaneous_v2':
                hilbert_signal, confidence, phase = algorithm_func(src_prices)
                # 周波数と振幅を位相から推定
                frequency = np.zeros(len(src_prices))
                amplitude = np.abs(hilbert_signal)
                for i in range(1, len(phase)):
                    phase_diff = phase[i] - phase[i-1]
                    if phase_diff > np.pi:
                        phase_diff -= 2 * np.pi
                    elif phase_diff < -np.pi:
                        phase_diff += 2 * np.pi
                    frequency[i] = abs(phase_diff) / (2 * np.pi)
                
                result = HilbertTransformResult(
                    amplitude=amplitude,
                    phase=phase,
                    frequency=frequency,
                    confidence_score=confidence,
                    algorithm_type=self.algorithm_type
                )
            
            elif self.algorithm_type == 'quantum_supreme':
                amplitude, phase, frequency, quantum_coherence = algorithm_func(src_prices)
                result = HilbertTransformResult(
                    amplitude=amplitude,
                    phase=phase,
                    frequency=frequency,
                    quantum_coherence=quantum_coherence,
                    algorithm_type=self.algorithm_type
                )
            
            elif self.algorithm_type == 'fft_based':
                amplitude, phase, cycle_phase = algorithm_func(src_prices)
                # 周波数を位相から計算
                frequency = np.zeros(len(src_prices))
                for i in range(1, len(phase)):
                    if not np.isnan(phase[i]) and not np.isnan(phase[i-1]):
                        phase_diff = phase[i] - phase[i-1]
                        if phase_diff > np.pi:
                            phase_diff -= 2 * np.pi
                        elif phase_diff < -np.pi:
                            phase_diff += 2 * np.pi
                        frequency[i] = abs(phase_diff) / (2 * np.pi)
                
                result = HilbertTransformResult(
                    amplitude=amplitude,
                    phase=phase,
                    frequency=frequency,
                    cycle_phase=cycle_phase,
                    algorithm_type=self.algorithm_type
                )
            
            elif self.algorithm_type == 'multiresolution':
                amplitude, phase, wavelet_energy, inst_frequency = algorithm_func(src_prices)
                result = HilbertTransformResult(
                    amplitude=amplitude,
                    phase=phase,
                    frequency=inst_frequency,
                    wavelet_energy=wavelet_energy,
                    algorithm_type=self.algorithm_type
                )
            
            else:
                return self._create_empty_result(len(src_prices))
            
            # キャッシュ更新
            self._result = result
            self._cache_hash = current_hash
            
            return result
            
        except Exception as e:
            self.logger.error(f"ヒルベルト変換計算エラー: {e}")
            return self._create_empty_result(len(data) if hasattr(data, '__len__') else 0)
    
    def _create_empty_result(self, length: int) -> HilbertTransformResult:
        """空の結果を作成"""
        return HilbertTransformResult(
            amplitude=np.full(length, np.nan),
            phase=np.full(length, np.nan),
            frequency=np.full(length, np.nan),
            algorithm_type=self.algorithm_type
        )
    
    def _get_data_hash(self, data) -> str:
        """データのハッシュを計算"""
        try:
            if hasattr(data, 'values'):
                return str(hash(data.values.tobytes()))
            else:
                return str(hash(data.tobytes()))
        except:
            return str(len(data))
    
    @classmethod
    def get_available_algorithms(cls) -> Dict[str, str]:
        """利用可能なアルゴリズムのリストを取得"""
        return cls._ALGORITHM_DESCRIPTIONS.copy()
    
    def get_amplitude(self) -> Optional[np.ndarray]:
        """瞬時振幅を取得"""
        if self._result is not None:
            return self._result.amplitude.copy()
        return None
    
    def get_phase(self) -> Optional[np.ndarray]:
        """瞬時位相を取得"""
        if self._result is not None:
            return self._result.phase.copy()
        return None
    
    def get_frequency(self) -> Optional[np.ndarray]:
        """瞬時周波数を取得"""
        if self._result is not None:
            return self._result.frequency.copy()
        return None
    
    def get_trend_components(self) -> Optional[Dict[str, np.ndarray]]:
        """トレンド関連成分を取得"""
        if self._result is None:
            return None
        
        components = {}
        if self._result.trend_component is not None:
            components['trend_component'] = self._result.trend_component.copy()
        if self._result.trend_strength is not None:
            components['trend_strength'] = self._result.trend_strength.copy()
        
        return components if components else None
    
    def get_quantum_components(self) -> Optional[Dict[str, np.ndarray]]:
        """量子関連成分を取得"""
        if self._result is None:
            return None
        
        components = {}
        if self._result.quantum_entanglement is not None:
            components['quantum_entanglement'] = self._result.quantum_entanglement.copy()
        if self._result.quantum_coherence is not None:
            components['quantum_coherence'] = self._result.quantum_coherence.copy()
        
        return components if components else None
    
    def get_wavelet_components(self) -> Optional[Dict[str, np.ndarray]]:
        """ウェーブレット関連成分を取得"""
        if self._result is None:
            return None
        
        components = {}
        if self._result.wavelet_energy is not None:
            components['wavelet_energy'] = self._result.wavelet_energy.copy()
        if self._result.cycle_phase is not None:
            components['cycle_phase'] = self._result.cycle_phase.copy()
        
        return components if components else None
    
    def get_confidence_score(self) -> Optional[np.ndarray]:
        """信頼度スコアを取得"""
        if self._result is not None and self._result.confidence_score is not None:
            return self._result.confidence_score.copy()
        return None
    
    def get_algorithm_metadata(self) -> Dict:
        """アルゴリズムのメタデータを取得"""
        if self._result is None:
            return {}
        
        metadata = {
            'algorithm_type': self.algorithm_type,
            'algorithm_description': self._ALGORITHM_DESCRIPTIONS.get(self.algorithm_type, ''),
            'src_type': self.src_type,
            'data_points': len(self._result.amplitude),
            'avg_amplitude': np.nanmean(self._result.amplitude),
            'avg_frequency': np.nanmean(self._result.frequency),
            'phase_range': [np.nanmin(self._result.phase), np.nanmax(self._result.phase)]
        }
        
        # アルゴリズム固有の情報
        if self._result.trend_strength is not None:
            metadata['avg_trend_strength'] = np.nanmean(self._result.trend_strength)
        if self._result.quantum_entanglement is not None:
            metadata['avg_quantum_entanglement'] = np.nanmean(self._result.quantum_entanglement)
        if self._result.quantum_coherence is not None:
            metadata['avg_quantum_coherence'] = np.nanmean(self._result.quantum_coherence)
        if self._result.wavelet_energy is not None:
            metadata['avg_wavelet_energy'] = np.nanmean(self._result.wavelet_energy)
        if self._result.confidence_score is not None:
            metadata['avg_confidence'] = np.nanmean(self._result.confidence_score)
        
        return metadata
    
    def reset(self) -> None:
        """状態をリセット"""
        self._result = None
        self._cache_hash = None 