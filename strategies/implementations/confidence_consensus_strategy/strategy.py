#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **信頼度ベース・コンセンサス戦略** 🎯

階層的適応型コンセンサス法による高度な信頼度算出戦略。
複数のインジケーターシグナルを重み付きで組み合わせ、
市場状況に応じて動的に調整する洗練されたアプローチ。

🌟 **主要機能:**
1. **フィルター合意度レイヤー** (40%): ハイパーADX、ハイパーER、ハイパートレンドインデックス（全て-1→0変換）
2. **方向性強度レイヤー** (30%): ドンチャンFRAMA 3期間(60, 120, 240)
3. **モメンタム係数レイヤー** (20%): ラゲールRSIトレンドフォロー
4. **ボラティリティ補正レイヤー** (10%): XATR補正要素

📊 **エントリー・エグジット条件:**
- ロングエントリー: 信頼度 >= 0.6
- ショートエントリー: 信頼度 <= -0.6
- ロングエグジット: 信頼度 <= 0.0（ロングポジション時）
- ショートエグジット: 信頼度 >= 0.0（ショートポジション時）
- 方向性シグナルが負の場合、フィルター・ボラティリティ符号逆転
"""

from dataclasses import dataclass
from typing import Union, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from numba import njit
import optuna

# 戦略基底クラス
from ...base.strategy import BaseStrategy
from ...interfaces.strategy import IStrategy

# インジケーター
from indicators.trend_filter.hyper_adx import HyperADX
from indicators.trend_filter.hyper_er import HyperER
from indicators.hyper_trend_index import HyperTrendIndex
from indicators.volatility.x_atr import XATR

# シグナル
from signals.implementations.donchian_frama.entry import DonchianFRAMACrossoverEntrySignal
from signals.implementations.laguerre_rsi.trend_follow_entry import LaguerreRSITrendFollowEntrySignal


@dataclass
class ConfidenceCalculationResult:
    """信頼度計算結果"""
    confidence: np.ndarray                # 最終信頼度（-1〜1）
    filter_consensus: np.ndarray          # フィルター合意度
    directional_strength: np.ndarray      # 方向性強度
    momentum_factor: np.ndarray           # モメンタム係数
    volatility_correction: np.ndarray     # ボラティリティ補正
    entry_signals: np.ndarray             # エントリーシグナル（1=ロング、-1=ショート、0=なし）
    
    # 各レイヤーの詳細
    filter_signals: Dict[str, np.ndarray] # フィルターシグナル詳細
    donchian_signals: Dict[str, np.ndarray] # ドンチャンFRAMAシグナル詳細
    laguerre_signal: np.ndarray           # ラゲールRSIシグナル
    xatr_signal: np.ndarray               # XATRシグナル


@njit(fastmath=True, cache=True)
def calculate_filter_consensus_numba(
    hyper_adx: np.ndarray,
    hyper_er: np.ndarray, 
    hyper_trend: np.ndarray
) -> np.ndarray:
    """
    フィルター合意度を計算する（Numba最適化版）
    
    Args:
        hyper_adx: ハイパーADXのトレンド信号（-1→0として扱う）
        hyper_er: ハイパーERのトレンド信号（-1→0として扱う）
        hyper_trend: ハイパートレンドインデックスのトレンド信号（-1→0として扱う）
        
    Returns:
        フィルター合意度（0〜1の範囲）
    """
    length = len(hyper_adx)
    consensus = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        # NaN値のチェック
        if (np.isnan(hyper_adx[i]) or np.isnan(hyper_er[i]) or np.isnan(hyper_trend[i])):
            continue
            
        # 全てのフィルターシグナルの-1を0として扱う（方向性を示さないため）
        adjusted_adx = 0.0 if hyper_adx[i] < 0 else hyper_adx[i]
        adjusted_er = 0.0 if hyper_er[i] < 0 else hyper_er[i]
        adjusted_trend = 0.0 if hyper_trend[i] < 0 else hyper_trend[i]
        
        # 3つのフィルターの平均（0-1の範囲）
        consensus[i] = (adjusted_adx + adjusted_er + adjusted_trend) / 3.0
    
    return consensus


@njit(fastmath=True, cache=True)
def calculate_directional_strength_numba(
    donchian_60: np.ndarray,
    donchian_120: np.ndarray,
    donchian_240: np.ndarray
) -> np.ndarray:
    """
    方向性強度を計算する（Numba最適化版）
    
    Args:
        donchian_60: 60期間ドンチャンFRAMAシグナル
        donchian_120: 120期間ドンチャンFRAMAシグナル
        donchian_240: 240期間ドンチャンFRAMAシグナル
        
    Returns:
        方向性強度（-1〜1）
    """
    length = len(donchian_60)
    strength = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        # NaN値のチェック
        if (np.isnan(donchian_60[i]) or np.isnan(donchian_120[i]) or np.isnan(donchian_240[i])):
            continue
        
        # 重み付き平均: 短期40%, 中期35%, 長期25%
        strength[i] = (donchian_60[i] * 0.4 + 
                      donchian_120[i] * 0.35 + 
                      donchian_240[i] * 0.25)
    
    return strength


@njit(fastmath=True, cache=True)
def calculate_confidence_score_numba(
    filter_consensus: np.ndarray,
    directional_strength: np.ndarray,
    momentum_factor: np.ndarray,
    volatility_signal: np.ndarray
) -> tuple:
    """
    最終信頼度を計算する（Numba最適化版）
    
    Args:
        filter_consensus: フィルター合意度
        directional_strength: 方向性強度
        momentum_factor: モメンタム係数
        volatility_signal: ボラティリティシグナル
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (信頼度, ボラティリティ補正)
    """
    length = len(filter_consensus)
    confidence = np.full(length, np.nan, dtype=np.float64)
    volatility_correction = np.full(length, 1.0, dtype=np.float64)
    
    for i in range(length):
        # NaN値のチェック
        if (np.isnan(filter_consensus[i]) or np.isnan(directional_strength[i]) or 
            np.isnan(momentum_factor[i]) or np.isnan(volatility_signal[i])):
            continue
        
        # ボラティリティ補正の計算
        volatility_correction[i] = 1.0 + (volatility_signal[i] * 0.1)
        
        # 基本信頼度
        base_confidence = (filter_consensus[i] * 0.4 + 
                          directional_strength[i] * 0.3 + 
                          momentum_factor[i] * 0.2)
        
        # 方向性による符号調整（エレガントなポイント）
        if directional_strength[i] < 0:
            # 方向性が負の場合、フィルターとボラティリティを逆転
            adjusted_confidence = ((-filter_consensus[i]) * 0.4 + 
                                 directional_strength[i] * 0.3 + 
                                 momentum_factor[i] * 0.2)
            confidence[i] = adjusted_confidence * (1.0 - volatility_signal[i] * 0.1)
        else:
            confidence[i] = base_confidence * volatility_correction[i]
        
        # -1から1の範囲にクリップ
        confidence[i] = max(-1.0, min(1.0, confidence[i]))
    
    return confidence, volatility_correction


@njit(fastmath=True, cache=True)
def generate_entry_signals_numba(
    confidence: np.ndarray,
    long_threshold: float = 0.6,
    short_threshold: float = -0.6
) -> np.ndarray:
    """
    信頼度からエントリーシグナルを生成する（Numba最適化版）
    
    Args:
        confidence: 信頼度配列
        long_threshold: ロングエントリー閾値
        short_threshold: ショートエントリー閾値
        
    Returns:
        エントリーシグナル（1=ロング、-1=ショート、0=なし）
    """
    length = len(confidence)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in range(length):
        if np.isnan(confidence[i]):
            continue
            
        if confidence[i] >= long_threshold:
            signals[i] = 1  # ロングエントリー
        elif confidence[i] <= short_threshold:
            signals[i] = -1  # ショートエントリー
        # else: 0 (エントリーなし)
    
    return signals


@njit(fastmath=True, cache=True)
def generate_exit_signals_numba(
    confidence: np.ndarray,
    entry_signals: np.ndarray
) -> np.ndarray:
    """
    信頼度からエグジットシグナルを生成する（Numba最適化版）
    
    ロジック:
    - ロングポジション: 信頼度が0以下になったらエグジット
    - ショートポジション: 信頼度が0以上になったらエグジット
    
    Args:
        confidence: 信頼度配列
        entry_signals: エントリーシグナル配列
        
    Returns:
        エグジットシグナル（1=エグジット、0=ホールド）
    """
    length = len(confidence)
    exit_signals = np.zeros(length, dtype=np.int8)
    current_position = 0  # 0=ポジションなし、1=ロング、-1=ショート
    
    for i in range(length):
        if np.isnan(confidence[i]):
            continue
        
        # エントリーシグナルがある場合、ポジションを更新
        if entry_signals[i] != 0:
            current_position = entry_signals[i]
            continue  # エントリーと同時にエグジットはしない
        
        # 現在のポジションに基づいてエグジット判定
        if current_position == 1:  # ロングポジション
            if confidence[i] <= 0.0:
                exit_signals[i] = 1  # ロングエグジット
                current_position = 0  # ポジションクリア
        elif current_position == -1:  # ショートポジション
            if confidence[i] >= 0.0:
                exit_signals[i] = 1  # ショートエグジット  
                current_position = 0  # ポジションクリア
    
    return exit_signals


class ConfidenceConsensusStrategy(BaseStrategy, IStrategy):
    """
    信頼度ベース・コンセンサス戦略
    
    階層的適応型コンセンサス法により、複数のインジケーターシグナルを
    重み付きで組み合わせて信頼度を算出し、エントリー判定を行う戦略。
    
    特徴:
    - 4層の階層的信頼度計算
    - 方向性に応じた動的符号調整  
    - ボラティリティ環境による補正
    - 高い信頼度閾値による厳選エントリー
    - 信頼度ベースの論理的エグジット（ロング：≤0、ショート：≥0）
    """
    
    def __init__(
        self,
        # エントリー閾値
        long_threshold: float = 0.5,
        short_threshold: float = -0.5,
        
        # フィルターインジケーターパラメータ
        hyper_adx_period: int = 14,
        hyper_er_period: int = 14,
        hyper_trend_period: int = 14,
        
        # ドンチャンFRAMAパラメータ
        donchian_periods: Tuple[int, int, int] = (60, 120, 240),
        
        # ラゲールRSIパラメータ  
        laguerre_gamma: float = 0.5,
        
        # XATRパラメータ
        xatr_period: float = 34.0,
        xatr_tr_method: str = 'atr',
        
        # 重み設定（デフォルト値）
        filter_weight: float = 0.4,
        directional_weight: float = 0.3,
        momentum_weight: float = 0.2,
        volatility_weight: float = 0.1
    ):
        """
        初期化
        
        Args:
            long_threshold: ロングエントリー閾値（デフォルト: 0.6）
            short_threshold: ショートエントリー閾値（デフォルト: -0.6）
            hyper_adx_period: ハイパーADX期間
            hyper_er_period: ハイパーER期間
            hyper_trend_period: ハイパートレンドインデックス期間
            donchian_periods: ドンチャンFRAMA期間(短期, 中期, 長期)
            laguerre_gamma: ラゲールRSIガンマ値
            xatr_period: XATR期間
            xatr_tr_method: XATR TR計算方法
            filter_weight: フィルター重み
            directional_weight: 方向性重み
            momentum_weight: モメンタム重み
            volatility_weight: ボラティリティ重み
        """
        strategy_name = (f"ConfidenceConsensus(L≥{long_threshold:.1f}, S≤{short_threshold:.1f}, "
                        f"γ={laguerre_gamma}, periods={donchian_periods})")
        super().__init__(strategy_name)
        
        # パラメータ保存
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        
        # 重み設定
        self.filter_weight = filter_weight
        self.directional_weight = directional_weight  
        self.momentum_weight = momentum_weight
        self.volatility_weight = volatility_weight
        
        # パラメータ検証
        if long_threshold <= 0 or short_threshold >= 0:
            raise ValueError("閾値設定が無効です: long_threshold > 0, short_threshold < 0")
        
        weight_sum = filter_weight + directional_weight + momentum_weight + volatility_weight
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"重みの合計が1.0ではありません: {weight_sum}")
        
        # インジケーターの初期化
        try:
            # フィルターインジケーター
            self.hyper_adx = HyperADX(
                period=hyper_adx_period,
                use_kalman_filter=True,
                use_roofing_filter=True,
                use_dynamic_period=True
            )
            
            self.hyper_er = HyperER(
                period=hyper_er_period,
                use_kalman_filter=True,
                use_roofing_filter=True,
                use_dynamic_period=True
            )
            
            self.hyper_trend = HyperTrendIndex(
                period=hyper_trend_period,
                use_kalman_filter=True,
                use_roofing_filter=True,
                use_dynamic_period=True
            )
            
            # XATR
            self.xatr = XATR(
                period=xatr_period,
                tr_method=xatr_tr_method,
                smoother_type='ultimate_smoother',
                enable_kalman=False,
                period_mode='dynamic'
            )
            
            self.logger.info("インジケーター初期化完了")
            
        except Exception as e:
            self.logger.error(f"インジケーター初期化失敗: {e}")
            raise
        
        # シグナルの初期化
        try:
            # ドンチャンFRAMAシグナル（3期間）
            self.donchian_signals = {}
            for period in donchian_periods:
                # FRAMA期間は偶数である必要があるため調整
                frama_period =16
                self.donchian_signals[f'period_{period}'] = DonchianFRAMACrossoverEntrySignal(
                    donchian_period=period,
                    frama_period=frama_period,
                    donchian_src_type='hlc3',
                    frama_src_type='hlc3',
                    position_mode=True
                )
            
            # ラゲールRSIシグナル
            self.laguerre_rsi_signal = LaguerreRSITrendFollowEntrySignal(
                gamma=laguerre_gamma,
                buy_band=0.8,
                sell_band=0.2,
                position_mode=True
            )
            
            self.logger.info("シグナル初期化完了")
            
        except Exception as e:
            self.logger.error(f"シグナル初期化失敗: {e}")
            raise
        
        # キャッシュ
        self._result_cache = {}
        self._max_cache_size = 5
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュ値計算"""
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
                    first_val = float(data[0, -1]) if data.ndim > 1 else float(data[0])
                    last_val = float(data[-1, -1]) if data.ndim > 1 else float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            # パラメータシグネチャ
            param_str = f"{self.long_threshold}_{self.short_threshold}_{self.filter_weight}_{self.directional_weight}"
            
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(param_str)}"
            
        except Exception:
            return f"{id(data)}_{self.long_threshold}_{self.short_threshold}"
    
    def calculate_confidence(self, data: Union[pd.DataFrame, np.ndarray]) -> ConfidenceCalculationResult:
        """
        信頼度を計算する
        
        Args:
            data: 価格データ
            
        Returns:
            ConfidenceCalculationResult: 信頼度計算結果
        """
        try:
            # データ長の取得
            data_length = len(data)
            
            # 1. フィルターシグナルの計算
            hyper_adx_result = self.hyper_adx.calculate(data)
            hyper_er_result = self.hyper_er.calculate(data)
            hyper_trend_result = self.hyper_trend.calculate(data)
            
            filter_signals = {
                'hyper_adx': hyper_adx_result.trend_signal,
                'hyper_er': hyper_er_result.trend_signal,
                'hyper_trend': hyper_trend_result.trend_signal
            }
            
            # 2. 方向性シグナルの計算（ドンチャンFRAMA）
            donchian_signals = {}
            donchian_arrays = []
            
            for key, signal in self.donchian_signals.items():
                signal_result = signal.generate(data)
                donchian_signals[key] = signal_result
                donchian_arrays.append(signal_result)
            
            # 3つの期間のシグナルを取得
            donchian_60 = donchian_arrays[0] if len(donchian_arrays) > 0 else np.zeros(data_length)
            donchian_120 = donchian_arrays[1] if len(donchian_arrays) > 1 else np.zeros(data_length)  
            donchian_240 = donchian_arrays[2] if len(donchian_arrays) > 2 else np.zeros(data_length)
            
            # 3. モメンタムシグナルの計算（ラゲールRSI）
            laguerre_signal = self.laguerre_rsi_signal.generate(data)
            
            # 4. ボラティリティシグナルの計算（XATR）
            xatr_result = self.xatr.calculate(data)
            xatr_signal = xatr_result.volatility_signal
            
            # 5. 信頼度計算
            # フィルター合意度
            filter_consensus = calculate_filter_consensus_numba(
                filter_signals['hyper_adx'],
                filter_signals['hyper_er'], 
                filter_signals['hyper_trend']
            )
            
            # 方向性強度
            directional_strength = calculate_directional_strength_numba(
                donchian_60.astype(np.float64),
                donchian_120.astype(np.float64),
                donchian_240.astype(np.float64)
            )
            
            # モメンタム係数（そのまま使用）
            momentum_factor = laguerre_signal.astype(np.float64)
            
            # 最終信頼度計算
            confidence, volatility_correction = calculate_confidence_score_numba(
                filter_consensus,
                directional_strength,
                momentum_factor,
                xatr_signal
            )
            
            # エントリーシグナル生成
            entry_signals = generate_entry_signals_numba(
                confidence, self.long_threshold, self.short_threshold
            )
            
            # 結果作成
            result = ConfidenceCalculationResult(
                confidence=confidence,
                filter_consensus=filter_consensus,
                directional_strength=directional_strength,
                momentum_factor=momentum_factor,
                volatility_correction=volatility_correction,
                entry_signals=entry_signals,
                filter_signals=filter_signals,
                donchian_signals=donchian_signals,
                laguerre_signal=laguerre_signal,
                xatr_signal=xatr_signal
            )
            
            self.logger.debug(f"信頼度計算完了 - データ長: {data_length}")
            return result
            
        except Exception as e:
            self.logger.error(f"信頼度計算中にエラー: {e}")
            # エラー時は空の結果を返す
            empty_array = np.full(len(data), np.nan)
            empty_int_array = np.zeros(len(data), dtype=np.int8)
            
            return ConfidenceCalculationResult(
                confidence=empty_array,
                filter_consensus=empty_array,
                directional_strength=empty_array,
                momentum_factor=empty_array,
                volatility_correction=np.ones(len(data)),
                entry_signals=empty_int_array,
                filter_signals={},
                donchian_signals={},
                laguerre_signal=empty_int_array,
                xatr_signal=empty_array
            )
    
    def generate_entry(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する（BaseStrategy互換）
        
        Args:
            data: 価格データ
            
        Returns:
            エントリーシグナル配列（1=ロング、-1=ショート、0=なし）
        """
        try:
            signals = self.generate_signals(data)
            return signals['entry']
        except Exception as e:
            self.logger.error(f"エントリーシグナル生成中にエラー: {e}")
            return np.zeros(len(data), dtype=np.int8)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナルを生成する（BaseStrategy互換）
        
        Args:
            data: 価格データ
            position: 現在のポジション（1=ロング、-1=ショート）
            index: データのインデックス
            
        Returns:
            エグジットシグナル（True=エグジット、False=ホールド）
        """
        try:
            signals = self.generate_signals(data)
            confidence = signals['confidence']
            
            if index == -1:
                index = len(confidence) - 1
            
            if index < 0 or index >= len(confidence):
                return False
                
            current_confidence = confidence[index]
            if np.isnan(current_confidence):
                return False
            
            # 信頼度ベースのエグジット判定
            if position == 1:  # ロングポジション
                return current_confidence <= -0.3
            elif position == -1:  # ショートポジション
                return current_confidence >= 0.3
            
            return False
            
        except Exception as e:
            self.logger.error(f"エグジットシグナル生成中にエラー: {e}")
            return False

    def generate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        トレーディングシグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            シグナル辞書（entry, exit, position等）
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                return self._result_cache[data_hash]
            
            # 信頼度計算
            confidence_result = self.calculate_confidence(data)
            
            # エントリーシグナル
            entry_signals = confidence_result.entry_signals
            
            # 信頼度ベースのエグジットシグナル
            exit_signals = generate_exit_signals_numba(
                confidence_result.confidence,
                entry_signals
            )
            
            # ポジションシグナル（エントリーと同じ）
            position_signals = entry_signals.copy()
            
            # 結果
            signals = {
                'entry': entry_signals,
                'exit': exit_signals,
                'position': position_signals,
                'confidence': confidence_result.confidence,
                'filter_consensus': confidence_result.filter_consensus,
                'directional_strength': confidence_result.directional_strength,
                'momentum_factor': confidence_result.momentum_factor,
                'volatility_correction': confidence_result.volatility_correction
            }
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = signals
            self._cache_keys.append(data_hash)
            
            self.logger.debug(f"シグナル生成完了 - エントリー数: {np.sum(entry_signals != 0)}")
            return signals
            
        except Exception as e:
            self.logger.error(f"シグナル生成中にエラー: {e}")
            # エラー時は空のシグナルを返す
            empty_signals = np.zeros(len(data), dtype=np.int8)
            empty_values = np.full(len(data), np.nan)
            
            return {
                'entry': empty_signals,
                'exit': empty_signals,
                'position': empty_signals,
                'confidence': empty_values,
                'filter_consensus': empty_values,
                'directional_strength': empty_values,
                'momentum_factor': empty_values,
                'volatility_correction': np.ones(len(data))
            }
    
    @classmethod
    def create_optimization_params(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        最適化パラメータを生成する
        
        Args:
            trial: Optunaトライアル
            
        Returns:
            最適化パラメータ辞書
        """
        return {
            'long_threshold': trial.suggest_float('long_threshold', 0.4, 0.9, step=0.1),
            'short_threshold': trial.suggest_float('short_threshold', -0.9, -0.4, step=0.1),
            'laguerre_gamma': trial.suggest_float('laguerre_gamma', 0.6, 0.9, step=0.1),
            'donchian_periods': trial.suggest_categorical('donchian_periods', [
                (40, 80, 160), (50, 100, 200), (60, 120, 240), (80, 160, 320)
            ]),
            'hyper_adx_period': trial.suggest_int('hyper_adx_period', 10, 20),
            'hyper_er_period': trial.suggest_int('hyper_er_period', 10, 20),
            'hyper_trend_period': trial.suggest_int('hyper_trend_period', 10, 20),
            'xatr_period': trial.suggest_float('xatr_period', 15.0, 30.0, step=5.0)
        }
    
    @classmethod
    def convert_params_to_strategy_format(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        最適化パラメータを戦略パラメータに変換する
        
        Args:
            params: 最適化パラメータ
            
        Returns:
            戦略パラメータ辞書
        """
        return params.copy()

    def get_strategy_info(self) -> Dict[str, Any]:
        """戦略情報を取得"""
        return {
            'name': self.name,
            'type': 'ConfidenceConsensus',
            'long_threshold': self.long_threshold,
            'short_threshold': self.short_threshold,
            'weights': {
                'filter': self.filter_weight,
                'directional': self.directional_weight,
                'momentum': self.momentum_weight,
                'volatility': self.volatility_weight
            },
            'indicators': {
                'hyper_adx': self.hyper_adx.get_indicator_info(),
                'hyper_er': self.hyper_er.get_indicator_info(),
                'hyper_trend': self.hyper_trend.get_indicator_info(),
                'xatr': self.xatr.get_configuration()
            },
            'description': '階層的適応型コンセンサス法による信頼度ベース戦略'
        }
    
    def reset(self) -> None:
        """戦略状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        
        # インジケーターリセット
        for indicator in [self.hyper_adx, self.hyper_er, self.hyper_trend, self.xatr]:
            if hasattr(indicator, 'reset'):
                indicator.reset()
        
        # シグナルリセット
        for signal in self.donchian_signals.values():
            if hasattr(signal, 'reset'):
                signal.reset()
        
        if hasattr(self.laguerre_rsi_signal, 'reset'):
            self.laguerre_rsi_signal.reset()


# 便利関数
def create_confidence_consensus_strategy(
    long_threshold: float = 0.6,
    short_threshold: float = -0.6,
    donchian_periods: Tuple[int, int, int] = (60, 120, 240),
    laguerre_gamma: float = 0.8,
    **kwargs
) -> ConfidenceConsensusStrategy:
    """
    信頼度ベース・コンセンサス戦略を作成する便利関数
    
    Args:
        long_threshold: ロングエントリー閾値
        short_threshold: ショートエントリー閾値
        donchian_periods: ドンチャンFRAMA期間
        laguerre_gamma: ラゲールRSIガンマ値
        **kwargs: その他のパラメータ
        
    Returns:
        設定済みの戦略インスタンス
    """
    return ConfidenceConsensusStrategy(
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        donchian_periods=donchian_periods,
        laguerre_gamma=laguerre_gamma,
        **kwargs
    )


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== 信頼度ベース・コンセンサス戦略のテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 300
    base_price = 100.0
    
    # 複雑な市場データ（トレンド + レンジ + ボラティリティ変化）
    prices = [base_price]
    for i in range(1, length):
        if i < 100:  # 上昇トレンド
            change = 0.003 + np.random.normal(0, 0.008)
        elif i < 200:  # レンジ相場  
            change = np.random.normal(0, 0.012)
        else:  # 下降トレンド
            change = -0.002 + np.random.normal(0, 0.010)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データ生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.01))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.005)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 戦略テスト
    try:
        print("\n戦略インスタンス作成中...")
        strategy = ConfidenceConsensusStrategy(
            long_threshold=0.6,
            short_threshold=-0.6,
            donchian_periods=(60, 120, 240),
            laguerre_gamma=0.8
        )
        
        print("シグナル生成中...")
        signals = strategy.generate_signals(df)
        
        # 結果統計
        entry_signals = signals['entry']
        confidence = signals['confidence']
        
        long_entries = np.sum(entry_signals == 1)
        short_entries = np.sum(entry_signals == -1)
        no_entries = np.sum(entry_signals == 0)
        
        valid_confidence = confidence[~np.isnan(confidence)]
        avg_confidence = np.mean(valid_confidence) if len(valid_confidence) > 0 else 0
        
        print(f"\n=== 結果統計 ===")
        print(f"ロングエントリー: {long_entries}")
        print(f"ショートエントリー: {short_entries}")
        print(f"エントリーなし: {no_entries}")
        print(f"平均信頼度: {avg_confidence:.4f}")
        print(f"信頼度範囲: {np.min(valid_confidence):.4f} - {np.max(valid_confidence):.4f}")
        
        # 各レイヤーの統計
        filter_consensus = signals['filter_consensus']
        directional_strength = signals['directional_strength']
        momentum_factor = signals['momentum_factor']
        
        print(f"\n=== レイヤー統計 ===")
        print(f"フィルター合意度平均: {np.nanmean(filter_consensus):.4f}")
        print(f"方向性強度平均: {np.nanmean(directional_strength):.4f}")
        print(f"モメンタム係数平均: {np.nanmean(momentum_factor):.4f}")
        
        print("\n=== テスト完了 ===")
        
    except Exception as e:
        print(f"テスト中にエラー: {e}")
        import traceback
        traceback.print_exc()