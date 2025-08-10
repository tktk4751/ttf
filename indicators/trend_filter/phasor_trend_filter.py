#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import njit, float64
import math

from ..indicator import Indicator
from ..price_source import PriceSource

# 条件付きインポート（オプション機能）
try:
    from ..kalman.unified_kalman import UnifiedKalman
    UNIFIED_KALMAN_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行（パス調整付き）
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.kalman.unified_kalman import UnifiedKalman
        UNIFIED_KALMAN_AVAILABLE = True
    except ImportError:
        UnifiedKalman = None
        UNIFIED_KALMAN_AVAILABLE = False

# EhlersUnifiedDCとHyperERのインポート（動的期間適応とHyperER適応用）
try:
    from ..cycle.ehlers_unified_dc import EhlersUnifiedDC
    EHLERS_UNIFIED_DC_AVAILABLE = True
except ImportError:
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC
        EHLERS_UNIFIED_DC_AVAILABLE = True
    except ImportError:
        EhlersUnifiedDC = None
        EHLERS_UNIFIED_DC_AVAILABLE = False

try:
    from ..trend_filter.hyper_er import HyperER
    HYPER_ER_AVAILABLE = True
except ImportError:
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.trend_filter.hyper_er import HyperER
        HYPER_ER_AVAILABLE = True
    except ImportError:
        HyperER = None
        HYPER_ER_AVAILABLE = False


@dataclass
class PhasorTrendFilterResult:
    """フェーザー分析トレンドフィルターの計算結果"""
    values: np.ndarray                    # トレンド強度値（0-1の範囲、高い=トレンド）
    phase_angle: np.ndarray              # フェーザー角度（度単位）
    real_component: np.ndarray           # Real component (cosine correlation)
    imag_component: np.ndarray           # Imaginary component (sine correlation)
    instantaneous_period: np.ndarray     # 瞬間周期
    state: np.ndarray                    # トレンド状態（+1: トレンド, 0: サイクリング, -1: レンジ）
    signal: np.ndarray                   # シグナル（+1: 買い, -1: 売り, 0: 中立）
    trend_strength: np.ndarray           # トレンド強度（0-1）
    cycle_confidence: np.ndarray         # サイクル信頼度（0-1）
    filtered_price: np.ndarray           # カルマンフィルター後の価格（使用した場合）
    dynamic_periods: np.ndarray          # 動的期間値（動的期間適応使用時）
    adaptive_trend_thresholds: np.ndarray # 動的適応されたtrend_threshold値（HyperER適応使用時）
    hyper_er_values: np.ndarray          # HyperER値（HyperER適応使用時）


@njit(fastmath=True, cache=True)
def calculate_adaptive_trend_threshold_numba(
    hyper_er_values: np.ndarray,
    base_threshold: float = 6.0,
    min_threshold: float = 3.0,
    max_threshold: float = 12.0
) -> np.ndarray:
    """
    HyperER値に基づいてtrend_thresholdを動的適応させる（Numba最適化版）
    
    Args:
        hyper_er_values: HyperER値の配列（0-1の範囲）
        base_threshold: ベーストレンド閾値（デフォルト: 6.0）
        min_threshold: 最小トレンド閾値（高効率時、デフォルト: 3.0）
        max_threshold: 最大トレンド閾値（低効率時、デフォルト: 12.0）
    
    Returns:
        動的適応されたtrend_threshold値の配列
    """
    length = len(hyper_er_values)
    adaptive_thresholds = np.full(length, base_threshold, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(hyper_er_values[i]):
            # NaN値の場合はベース値を使用
            adaptive_thresholds[i] = base_threshold
            continue
        
        er_value = hyper_er_values[i]
        
        # HyperERを0-1の範囲にクランプ
        er_value = max(0.0, min(1.0, er_value))
        
        # HyperERの値が高いほど低いtrend_threshold（より敏感）
        # HyperERの値が低いほど高いtrend_threshold（より鈍感）
        # 逆線形補間: ER=1.0 → min_threshold, ER=0.0 → max_threshold
        adaptive_thresholds[i] = max_threshold - er_value * (max_threshold - min_threshold)
        
        # 範囲制限（安全性のため）
        adaptive_thresholds[i] = max(min_threshold, min(max_threshold, adaptive_thresholds[i]))
    
    return adaptive_thresholds


@njit(fastmath=True, cache=True)
def calculate_phasor_correlation_numba(signal: np.ndarray, period: int, start_idx: int) -> tuple:
    """
    シグナルとcos/sinの相関を計算する（Numba最適化版）
    
    Args:
        signal: 入力シグナル
        period: 固定周期
        start_idx: 開始インデックス
    
    Returns:
        Tuple[float, float]: (Real, Imaginary) components
    """
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    
    # cosineとの相関（Real component）
    for count in range(period):
        idx = start_idx - count
        if idx >= 0 and idx < len(signal):
            x = signal[idx]
            y = math.cos(2.0 * math.pi * count / period)
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
            syy += y * y
    
    real = 0.0
    if (period * sxx - sx * sx > 0) and (period * syy - sy * sy > 0):
        real = (period * sxy - sx * sy) / math.sqrt((period * sxx - sx * sx) * (period * syy - sy * sy))
    
    # sineとの相関（Imaginary component）
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    
    for count in range(period):
        idx = start_idx - count
        if idx >= 0 and idx < len(signal):
            x = signal[idx]
            y = -math.sin(2.0 * math.pi * count / period)
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
            syy += y * y
    
    imag = 0.0
    if (period * sxx - sx * sx > 0) and (period * syy - sy * sy > 0):
        imag = (period * sxy - sx * sy) / math.sqrt((period * sxx - sx * sx) * (period * syy - sy * sy))
    
    return real, imag


@njit(fastmath=True, cache=True)
def calculate_phasor_trend_filter_numba(
    price: np.ndarray,
    period: int = 28,
    trend_threshold: float = 6.0,
    dynamic_periods: np.ndarray = None,
    adaptive_trend_thresholds: np.ndarray = None
) -> tuple:
    """
    フェーザー分析トレンドフィルターを計算する（Numba最適化版）
    
    Args:
        price: 価格配列
        period: 固定周期（デフォルト: 28）
        trend_threshold: トレンド判定閾値（角度変化率、デフォルト: 6.0度）
        dynamic_periods: 動的期間配列（オプション）
        adaptive_trend_thresholds: 動的適応されたtrend_threshold配列（オプション）
    
    Returns:
        Tuple[np.ndarray, ...]: トレンド強度値, フェーザー角度, Real, Imaginary, 瞬間周期, 状態, シグナル, トレンド強度, サイクル信頼度
    """
    length = len(price)
    
    # 変数の初期化
    real = np.zeros(length, dtype=np.float64)
    imag = np.zeros(length, dtype=np.float64)
    angle = np.zeros(length, dtype=np.float64)
    instantaneous_period = np.full(length, 60.0, dtype=np.float64)  # 初期値は60に設定
    state = np.zeros(length, dtype=np.float64)
    signal = np.zeros(length, dtype=np.float64)
    trend_strength = np.zeros(length, dtype=np.float64)
    cycle_confidence = np.zeros(length, dtype=np.float64)
    trend_values = np.zeros(length, dtype=np.float64)  # 最終的なトレンド強度値
    
    # フェーザー分析の計算
    for i in range(period, length):
        # 動的期間の決定
        current_period = period
        if dynamic_periods is not None and i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
            current_period = max(6, min(int(dynamic_periods[i]), 50))  # 6-50期間に制限
        
        # 動的trend_thresholdの決定
        current_trend_threshold = trend_threshold
        if adaptive_trend_thresholds is not None and i < len(adaptive_trend_thresholds) and not np.isnan(adaptive_trend_thresholds[i]):
            current_trend_threshold = adaptive_trend_thresholds[i]
        
        # 現在のインデックスが期間に対して十分かチェック
        if i < current_period:
            continue
        
        # フェーザー相関の計算
        real_val, imag_val = calculate_phasor_correlation_numba(price, current_period, i)
        real[i] = real_val
        imag[i] = imag_val
        
        # 角度の計算（Ehlers論文のatan関数に基づく）
        if real[i] != 0.0:
            # Ehlers論文：Angle = 90 - Arctangent(Imag / Real)
            angle[i] = 90.0 - math.atan(imag[i] / real[i]) * 180.0 / math.pi
        else:
            angle[i] = angle[i-1] if i > 0 else 0.0
        
        # リアル部が負の場合の補正（Ehlers論文に基づく）
        if real[i] < 0.0:
            angle[i] = angle[i] - 180.0
        
        # 角度のラップアラウンド補正（Ehlers論文の条件）
        if i > 0:
            # 角度が360度のラップアラウンドを起こした場合の補正
            angle_diff_360 = abs(angle[i-1]) - abs(angle[i] - 360.0)
            angle_diff_normal = angle[i] - angle[i-1]
            
            if (angle_diff_360 < angle_diff_normal and 
                angle[i] > 90.0 and angle[i-1] < -90.0):
                angle[i] = angle[i] - 360.0
        
        # 角度は逆向きに進まない（Ehlers論文の条件）
        if i > 0:
            angle_going_backward = (angle[i] < angle[i-1] and 
                                  ((angle[i] > -135.0 and angle[i-1] < 135.0) or 
                                   (angle[i] < -90.0 and angle[i-1] < -90.0)))
            if angle_going_backward:
                angle[i] = angle[i-1]
        
        # 瞬間周期の計算（角度変化率から）
        if i > 0:
            delta_angle = angle[i] - angle[i-1]
            if delta_angle <= 0.0:
                delta_angle = 1.0  # 最小値を設定
            
            if delta_angle != 0.0:
                inst_period = 360.0 / delta_angle
                if inst_period > 60.0:
                    inst_period = 60.0
                elif inst_period < 6.0:
                    inst_period = 6.0
                instantaneous_period[i] = inst_period
            else:
                instantaneous_period[i] = instantaneous_period[i-1]
        
        # サイクル信頼度の計算（Real/Imaginaryの強度に基づく）
        magnitude = math.sqrt(real[i]**2 + imag[i]**2)
        cycle_confidence[i] = min(1.0, magnitude)  # 0-1の範囲に正規化
        
        # トレンド状態の判定（John Ehlersの論文Figure 4に基づく）
        if i > 0:
            delta_angle = angle[i] - angle[i-1]
            abs_delta_angle = abs(delta_angle)
            
            # Ehlers論文：「瞬間周期>60日」または「角度変化率≤trend_threshold」でトレンドモード
            is_trending = (instantaneous_period[i] > 60.0) or (abs_delta_angle <= current_trend_threshold)
            
            if is_trending:
                # トレンド方向の判定（Ehlers論文Figure 4に基づく）
                # Long (+1): 角度が+90度より大きいか-90度より小さい
                # Short (-1): 角度が-90度と+90度の間
                if angle[i] > 90.0 or angle[i] < -90.0:
                    state[i] = 1.0  # Long position
                    trend_strength[i] = 1.0 - (abs_delta_angle / max(current_trend_threshold, 1.0))
                elif angle[i] >= -90.0 and angle[i] <= 90.0:
                    state[i] = -1.0  # Short position (or out)
                    trend_strength[i] = 1.0 - (abs_delta_angle / max(current_trend_threshold, 1.0))
                else:
                    state[i] = 0.0  # エッジケース
                    trend_strength[i] = 0.0
            else:
                # サイクリングモード：瞬間周期≤60日かつ角度変化率>6度
                state[i] = 0.0  # Cycling
                trend_strength[i] = 0.0
        
        # トレンド強度値の計算（0-1の範囲）
        if state[i] != 0.0:  # トレンドモード
            # トレンド強度 = サイクル信頼度 × 角度安定性
            angle_stability = 1.0 - min(1.0, abs(delta_angle) / 45.0)  # 45度を最大とする
            trend_values[i] = cycle_confidence[i] * angle_stability * 0.7 + 0.3
        else:  # サイクリングモード
            trend_values[i] = cycle_confidence[i] * 0.3  # 低いトレンド強度
        
        # シグナル生成
        if i > 1:
            # トレンド状態の変化に基づくシグナル生成
            # 上昇トレンドの開始または強化
            if (state[i] == 1.0 and state[i-1] != 1.0) or (state[i] == 1.0 and trend_strength[i] > trend_strength[i-1]):
                signal[i] = 1.0  # 買いシグナル
            # 下降トレンドの開始または強化
            elif (state[i] == -1.0 and state[i-1] != -1.0) or (state[i] == -1.0 and trend_strength[i] > trend_strength[i-1]):
                signal[i] = -1.0  # 売りシグナル
            # サイクリングまたは弱いトレンド
            else:
                signal[i] = 0.0  # 中立
    
    return trend_values, angle, real, imag, instantaneous_period, state, signal, trend_strength, cycle_confidence


class PhasorTrendFilter(Indicator):
    """
    フェーザー分析トレンドフィルター
    
    John Ehlersの「RECURRING PHASE OF CYCLE ANALYSIS」論文に基づいて実装された
    フェーザー分析を使用したトレンド・レンジ判定インジケーター。
    
    特徴:
    - フェーザー分析による高精度なトレンド・サイクル判定
    - 固定周期でのcos/sin相関によるReal/Imaginaryコンポーネント計算
    - 角度変化率によるトレンド/サイクリング判定
    - トレンド強度とサイクル信頼度の計算
    - カルマンフィルター前処理対応（オプション）
    
    計算手順:
    1. 固定周期でのcos/sin相関を計算してReal/Imaginaryコンポーネントを取得
    2. フェーザー角度を計算（arctangent）
    3. 角度変化率から瞬間周期を計算
    4. 角度変化率に基づいてトレンド/サイクリング状態を判定
    5. トレンド強度とサイクル信頼度を計算
    6. シグナルを生成
    """
    
    def __init__(
        self,
        period: int = 20,                     # フェーザー分析の固定周期
        trend_threshold: float = 6.0,         # トレンド判定閾値（角度変化率）
        src_type: str = 'close',              # ソースタイプ
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = True,      # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,   # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # 動的期間適応パラメータ（EhlersUnifiedDC使用）
        use_dynamic_period: bool = True,      # 動的期間適応を使用するか
        detector_type: str = 'dft_dominant',  # サイクル検出器タイプ
        cycle_part: float = 0.5,              # サイクル部分
        max_cycle: int = 89,                  # 最大サイクル期間
        min_cycle: int = 12,                   # 最小サイクル期間
        max_output: int = 55,                 # 最大出力値
        min_output: int = 5,                  # 最小出力値
        lp_period: int = 13,                  # ローパスフィルター期間
        hp_period: int = 48,                  # ハイパスフィルター期間
        # HyperER動的適応パラメータ
        use_hyper_er_adaptation: bool = True, # HyperERによるtrend_threshold動的適応を使用するか
        hyper_er_period: int = 14,            # HyperER計算期間
        hyper_er_midline_period: int = 100,   # HyperERミッドライン計算期間
        min_trend_threshold: float = 3.0,     # 最小トレンド閾値（高効率時）
        max_trend_threshold: float = 12.0,    # 最大トレンド閾値（低効率時）
        # HyperER詳細パラメータ
        hyper_er_src_type: str = 'oc2',       # HyperERソースタイプ
        hyper_er_use_roofing_filter: bool = True, # HyperERルーフィングフィルターを使用するか
        hyper_er_use_smoothing: bool = True,  # HyperER平滑化を使用するか
        hyper_er_smoother_type: str = 'zlema' # HyperERスムーサータイプ
    ):
        """
        コンストラクタ
        
        Args:
            period: フェーザー分析の固定周期（デフォルト: 28）
            trend_threshold: トレンド判定閾値（角度変化率、デフォルト: 6.0度）
            src_type: ソースタイプ（デフォルト: 'close'）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
        """
        # インジケーター名の作成
        indicator_name = f"PhasorTrendFilter(period={period}, threshold={trend_threshold:.1f}, {src_type}"
        if use_kalman_filter:
            indicator_name += f", kalman={kalman_filter_type}"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.period = period
        self.trend_threshold = trend_threshold
        self.src_type = src_type.lower()
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_observation_noise = kalman_observation_noise
        
        # 動的期間適応パラメータ
        self.use_dynamic_period = use_dynamic_period
        self.detector_type = detector_type
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.lp_period = lp_period
        self.hp_period = hp_period
        
        # HyperER動的適応パラメータ
        self.use_hyper_er_adaptation = use_hyper_er_adaptation
        self.hyper_er_period = hyper_er_period
        self.hyper_er_midline_period = hyper_er_midline_period
        self.min_trend_threshold = min_trend_threshold
        self.max_trend_threshold = max_trend_threshold
        self.hyper_er_src_type = hyper_er_src_type
        self.hyper_er_use_roofing_filter = hyper_er_use_roofing_filter
        self.hyper_er_use_smoothing = hyper_er_use_smoothing
        self.hyper_er_smoother_type = hyper_er_smoother_type
        
        # ソースタイプの検証
        try:
            available_sources = PriceSource.get_available_sources()
            if self.src_type not in available_sources:
                raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(available_sources.keys())}")
        except AttributeError:
            # get_available_sources()がない場合は基本的なソースタイプのみチェック
            basic_sources = ['close', 'high', 'low', 'open', 'hl2', 'hlc3', 'ohlc4','oc2']
            if self.src_type not in basic_sources:
                raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(basic_sources)}")
        
        # パラメータ検証
        if period <= 0:
            raise ValueError("periodは正の整数である必要があります")
        if trend_threshold <= 0:
            raise ValueError("trend_thresholdは0より大きい必要があります")
        if use_kalman_filter and kalman_process_noise <= 0:
            raise ValueError("kalman_process_noiseは0より大きい必要があります")
        
        # カルマンフィルターの初期化（オプション）
        self.kalman_filter = None
        if self.use_kalman_filter:
            if not UNIFIED_KALMAN_AVAILABLE:
                self.logger.error("統合カルマンフィルターが利用できません。indicators.kalman.unified_kalmanをインポートできません。")
                self.use_kalman_filter = False
                self.logger.warning("カルマンフィルター機能を無効にしました")
            else:
                try:
                    self.kalman_filter = UnifiedKalman(
                        filter_type=self.kalman_filter_type,
                        src_type=self.src_type,
                        process_noise_scale=self.kalman_process_noise,
                        observation_noise_scale=self.kalman_observation_noise
                    )
                    self.logger.info(f"統合カルマンフィルターを初期化しました: {self.kalman_filter_type}")
                except Exception as e:
                    self.logger.error(f"統合カルマンフィルターの初期化に失敗: {e}")
                    self.use_kalman_filter = False
                    self.logger.warning("カルマンフィルター機能を無効にしました")
        
        # EhlersUnifiedDCの初期化（動的期間適応用）
        self.cycle_detector = None
        if self.use_dynamic_period:
            if EHLERS_UNIFIED_DC_AVAILABLE:
                try:
                    self.cycle_detector = EhlersUnifiedDC(
                        detector_type=self.detector_type,
                        cycle_part=self.cycle_part,
                        max_cycle=self.max_cycle,
                        min_cycle=self.min_cycle,
                        max_output=self.max_output,
                        min_output=self.min_output,
                        src_type=self.src_type,  # oc2をそのまま使用
                        lp_period=self.lp_period,
                        hp_period=self.hp_period,
                        use_kalman_filter=False  # 事前フィルタリングは無効化
                    )
                    self.logger.info(f"EhlersUnifiedDCを初期化しました: detector_type={self.detector_type}")
                except Exception as e:
                    self.logger.error(f"EhlersUnifiedDCの初期化に失敗: {e}")
                    self.use_dynamic_period = False
                    self.logger.warning("動的期間適応機能を無効にしました")
            else:
                self.logger.error("EhlersUnifiedDCが利用できません")
                self.use_dynamic_period = False
                self.logger.warning("動的期間適応機能を無効にしました")
        
        # HyperERの初期化（動的閾値適応用）
        self.hyper_er = None
        if self.use_hyper_er_adaptation:
            if HYPER_ER_AVAILABLE:
                try:
                    self.hyper_er = HyperER(
                        period=self.hyper_er_period,
                        midline_period=self.hyper_er_midline_period,
                        er_src_type=self.hyper_er_src_type,  # 正しいパラメータ名
                        use_roofing_filter=self.hyper_er_use_roofing_filter,
                        use_smoothing=self.hyper_er_use_smoothing,
                        smoother_type=self.hyper_er_smoother_type
                    )
                    self.logger.info(f"HyperERを初期化しました: period={self.hyper_er_period}")
                except Exception as e:
                    self.logger.error(f"HyperERの初期化に失敗: {e}")
                    self.use_hyper_er_adaptation = False
                    self.logger.warning("HyperER動的適応機能を無効にしました")
            else:
                self.logger.error("HyperERが利用できません")
                self.use_hyper_er_adaptation = False
                self.logger.warning("HyperER動的適応機能を無効にしました")
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 20
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        try:
            # データ情報の取得
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0].get('close', data.iloc[0, -1])) if length > 0 else 0.0
                last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1])) if length > 0 else 0.0
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
            
            # パラメータ情報（動的適応パラメータも含む）
            kalman_sig = f"{self.kalman_filter_type}_{self.kalman_process_noise}" if self.use_kalman_filter else "None"
            dynamic_sig = f"{self.use_dynamic_period}_{self.detector_type}" if self.use_dynamic_period else "False"
            hyper_er_sig = f"{self.use_hyper_er_adaptation}_{self.hyper_er_period}_{self.min_trend_threshold}_{self.max_trend_threshold}" if self.use_hyper_er_adaptation else "False"
            params_sig = f"{self.period}_{self.trend_threshold}_{self.src_type}_{kalman_sig}_{dynamic_sig}_{hyper_er_sig}"
            
            # 高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.period}_{self.trend_threshold}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> PhasorTrendFilterResult:
        """
        フェーザー分析トレンドフィルターを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、OHLC + 選択したソースタイプに必要なカラムが必要
        
        Returns:
            PhasorTrendFilterResult: フェーザー分析トレンドフィルターの計算結果
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ（高速化）
            data_hash = self._get_data_hash(data)
            
            # キャッシュにある場合は取得して返す
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新（最も新しく使われたキーを最後に）
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return PhasorTrendFilterResult(
                    values=cached_result.values.copy(),
                    phase_angle=cached_result.phase_angle.copy(),
                    real_component=cached_result.real_component.copy(),
                    imag_component=cached_result.imag_component.copy(),
                    instantaneous_period=cached_result.instantaneous_period.copy(),
                    state=cached_result.state.copy(),
                    signal=cached_result.signal.copy(),
                    trend_strength=cached_result.trend_strength.copy(),
                    cycle_confidence=cached_result.cycle_confidence.copy(),
                    filtered_price=cached_result.filtered_price.copy(),
                    # 動的適応結果も復元
                    dynamic_periods=getattr(cached_result, 'dynamic_periods', np.array([])).copy(),
                    adaptive_trend_thresholds=getattr(cached_result, 'adaptive_trend_thresholds', np.array([])).copy(),
                    hyper_er_values=getattr(cached_result, 'hyper_er_values', np.array([])).copy()
                )
            
            # 価格ソースの計算
            price_source = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換（float64型で統一）
            price_source = np.asarray(price_source, dtype=np.float64)
            
            # 動的期間適応（EhlersUnifiedDC使用）
            dynamic_periods = None
            if self.use_dynamic_period and self.cycle_detector is not None:
                try:
                    cycle_result = self.cycle_detector.calculate(data)
                    # サイクル検出結果を期間値として使用
                    dynamic_periods = np.asarray(cycle_result, dtype=np.float64)
                    # 期間の範囲制限
                    dynamic_periods = np.clip(dynamic_periods, self.min_cycle, self.max_cycle)
                    self.logger.debug(f"動的期間適応を適用: 平均期間={np.mean(dynamic_periods):.2f}")
                except Exception as e:
                    self.logger.warning(f"動的期間適応の計算エラー: {e}。固定期間を使用します。")
                    dynamic_periods = None
            
            # HyperER動的適応（trend_threshold調整）
            adaptive_trend_thresholds = None
            hyper_er_values = None
            if self.use_hyper_er_adaptation and self.hyper_er is not None:
                try:
                    hyper_er_result = self.hyper_er.calculate(data)
                    
                    # HyperER結果からER値を取得
                    if hasattr(hyper_er_result, 'normalized_er'):
                        hyper_er_values = np.asarray(hyper_er_result.normalized_er, dtype=np.float64)
                    elif hasattr(hyper_er_result, 'efficiency_ratio'):
                        hyper_er_values = np.asarray(hyper_er_result.efficiency_ratio, dtype=np.float64)
                    elif hasattr(hyper_er_result, 'values'):
                        hyper_er_values = np.asarray(hyper_er_result.values, dtype=np.float64)
                    elif isinstance(hyper_er_result, (np.ndarray, list)):
                        hyper_er_values = np.asarray(hyper_er_result, dtype=np.float64)
                    
                    if hyper_er_values is not None:
                        # HyperERに基づく動的閾値計算
                        adaptive_trend_thresholds = calculate_adaptive_trend_threshold_numba(
                            hyper_er_values,
                            base_threshold=self.trend_threshold,
                            min_threshold=self.min_trend_threshold,
                            max_threshold=self.max_trend_threshold
                        )
                        self.logger.debug(f"HyperER動的適応を適用: 平均閾値={np.mean(adaptive_trend_thresholds):.2f}")
                    else:
                        self.logger.warning("HyperER結果の取得に失敗しました。固定閾値を使用します。")
                        
                except Exception as e:
                    self.logger.warning(f"HyperER動的適応の計算エラー: {e}。固定閾値を使用します。")
                    hyper_er_values = None
                    adaptive_trend_thresholds = None
            
            # カルマンフィルターによる前処理（オプション）
            filtered_price = price_source.copy()
            if self.use_kalman_filter and self.kalman_filter is not None:
                try:
                    kalman_result = self.kalman_filter.calculate(data)
                    
                    # カルマンフィルター結果の処理
                    kalman_values = None
                    
                    if hasattr(kalman_result, 'values'):
                        kalman_values = kalman_result.values
                    elif hasattr(kalman_result, 'filtered_values'):
                        kalman_values = kalman_result.filtered_values
                    elif isinstance(kalman_result, (np.ndarray, list)):
                        kalman_values = kalman_result
                    else:
                        kalman_values = kalman_result
                    
                    if kalman_values is not None:
                        try:
                            kalman_values = np.asarray(kalman_values, dtype=np.float64)
                            
                            if kalman_values.ndim == 0:
                                raise ValueError("カルマンフィルター結果がスカラー値です")
                            elif kalman_values.ndim > 1:
                                kalman_values = kalman_values.flatten()
                            
                            if len(kalman_values) != len(price_source):
                                self.logger.warning(f"カルマンフィルター結果のサイズ不一致: {len(kalman_values)} != {len(price_source)}。元の価格を使用します。")
                                filtered_price = price_source.copy()
                            else:
                                # NaN値の処理
                                nan_mask = np.isnan(kalman_values)
                                if np.any(nan_mask):
                                    kalman_values[nan_mask] = price_source[nan_mask]
                                
                                filtered_price = kalman_values
                                self.logger.debug("カルマンフィルターによる価格前処理を適用しました")
                        except Exception as array_error:
                            self.logger.warning(f"カルマンフィルター結果の配列変換エラー: {array_error}。元の価格を使用します。")
                            filtered_price = price_source.copy()
                    else:
                        self.logger.warning("カルマンフィルター結果がNoneです。元の価格を使用します。")
                        filtered_price = price_source.copy()
                        
                except Exception as e:
                    self.logger.warning(f"カルマンフィルター適用中にエラー: {e}。元の価格を使用します。")
                    filtered_price = price_source.copy()
            
            # データ長の検証
            data_length = len(filtered_price)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < self.period + 10:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低{self.period + 10}点以上を推奨します。")
            
            # フェーザー分析トレンドフィルターの計算（Numba最適化関数を使用）
            trend_values, phase_angle, real_component, imag_component, instantaneous_period, state, signal, trend_strength, cycle_confidence = calculate_phasor_trend_filter_numba(
                filtered_price, self.period, self.trend_threshold, dynamic_periods, adaptive_trend_thresholds
            )
            
            # 結果の保存
            result = PhasorTrendFilterResult(
                values=trend_values.copy(),
                phase_angle=phase_angle.copy(),
                real_component=real_component.copy(),
                imag_component=imag_component.copy(),
                instantaneous_period=instantaneous_period.copy(),
                state=state.copy(),
                signal=signal.copy(),
                trend_strength=trend_strength.copy(),
                cycle_confidence=cycle_confidence.copy(),
                filtered_price=filtered_price.copy(),
                # 動的適応結果を追加
                dynamic_periods=dynamic_periods.copy() if dynamic_periods is not None else np.array([]),
                adaptive_trend_thresholds=adaptive_trend_thresholds.copy() if adaptive_trend_thresholds is not None else np.array([]),
                hyper_er_values=hyper_er_values.copy() if hyper_er_values is not None else np.array([])
            )
            
            # キャッシュを更新
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = trend_values  # 基底クラスの要件を満たすため
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"フェーザー分析トレンドフィルター計算中にエラー: {error_msg}\\n{stack_trace}")
            
            # エラー時は空の結果を返す
            empty_array = np.array([])
            return PhasorTrendFilterResult(
                values=empty_array,
                phase_angle=empty_array,
                real_component=empty_array,
                imag_component=empty_array,
                instantaneous_period=empty_array,
                state=empty_array,
                signal=empty_array,
                trend_strength=empty_array,
                cycle_confidence=empty_array,
                filtered_price=empty_array,
                # 動的適応結果も空で初期化
                dynamic_periods=empty_array,
                adaptive_trend_thresholds=empty_array,
                hyper_er_values=empty_array
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """トレンド強度値を取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.values.copy()
    
    def get_phase_angle(self) -> Optional[np.ndarray]:
        """フェーザー角度を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.phase_angle.copy()
    
    def get_real_component(self) -> Optional[np.ndarray]:
        """Real componentを取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.real_component.copy()
    
    def get_imag_component(self) -> Optional[np.ndarray]:
        """Imaginary componentを取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.imag_component.copy()
    
    def get_instantaneous_period(self) -> Optional[np.ndarray]:
        """瞬間周期を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.instantaneous_period.copy()
    
    def get_state(self) -> Optional[np.ndarray]:
        """トレンド状態を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.state.copy()
    
    def get_signal(self) -> Optional[np.ndarray]:
        """シグナルを取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.signal.copy()
    
    def get_trend_strength(self) -> Optional[np.ndarray]:
        """トレンド強度を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.trend_strength.copy()
    
    def get_cycle_confidence(self) -> Optional[np.ndarray]:
        """サイクル信頼度を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.cycle_confidence.copy()
    
    def get_filtered_price(self) -> Optional[np.ndarray]:
        """カルマンフィルター後の価格を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.filtered_price.copy()
    
    def get_phasor_components(self) -> Optional[tuple]:
        """フェーザーのRealとImaginaryコンポーネントを取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.real_component.copy(), result.imag_component.copy()
    
    def get_dynamic_periods(self) -> Optional[np.ndarray]:
        """動的期間値を取得する（EhlersUnifiedDC使用時）"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        dynamic_periods = getattr(result, 'dynamic_periods', np.array([]))
        return dynamic_periods.copy() if len(dynamic_periods) > 0 else None
    
    def get_adaptive_trend_thresholds(self) -> Optional[np.ndarray]:
        """動的適応されたtrend_threshold値を取得する（HyperER適応使用時）"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        adaptive_thresholds = getattr(result, 'adaptive_trend_thresholds', np.array([]))
        return adaptive_thresholds.copy() if len(adaptive_thresholds) > 0 else None
    
    def get_hyper_er_values(self) -> Optional[np.ndarray]:
        """HyperER値を取得する（HyperER適応使用時）"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        hyper_er_values = getattr(result, 'hyper_er_values', np.array([]))
        return hyper_er_values.copy() if len(hyper_er_values) > 0 else None
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'period': self.period,
            'trend_threshold': self.trend_threshold,
            'src_type': self.src_type,
            'use_kalman_filter': self.use_kalman_filter,
            'kalman_filter_type': self.kalman_filter_type if self.use_kalman_filter else None,
            'kalman_process_noise': self.kalman_process_noise if self.use_kalman_filter else None,
            'kalman_observation_noise': self.kalman_observation_noise if self.use_kalman_filter else None,
            # 動的適応機能の情報
            'use_dynamic_period': self.use_dynamic_period,
            'detector_type': self.detector_type if self.use_dynamic_period else None,
            'use_hyper_er_adaptation': self.use_hyper_er_adaptation,
            'hyper_er_period': self.hyper_er_period if self.use_hyper_er_adaptation else None,
            'min_trend_threshold': self.min_trend_threshold if self.use_hyper_er_adaptation else None,
            'max_trend_threshold': self.max_trend_threshold if self.use_hyper_er_adaptation else None,
            'description': 'フェーザー分析ベーストレンド・レンジ判定フィルター（John Ehlersの論文に基づく）、EhlersUnifiedDC動的期間適応・HyperER動的閾値適応対応'
        }
    
    def reset(self) -> None:
        """インディケーターの状態をリセットする"""
        super().reset()
        if self.kalman_filter:
            self.kalman_filter.reset()
        if self.cycle_detector:
            self.cycle_detector.reset()
        if self.hyper_er:
            self.hyper_er.reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_phasor_trend_filter(
    data: Union[pd.DataFrame, np.ndarray],
    period: int = 28,
    trend_threshold: float = 6.0,
    src_type: str = 'close',
    use_kalman_filter: bool = False,
    kalman_filter_type: str = 'unscented',
    kalman_process_noise: float = 0.01,
    kalman_observation_noise: float = 0.001,
    **kwargs
) -> np.ndarray:
    """
    フェーザー分析トレンドフィルターの計算（便利関数）
    
    Args:
        data: 価格データ
        period: フェーザー分析の固定周期
        trend_threshold: トレンド判定閾値
        src_type: ソースタイプ
        use_kalman_filter: カルマンフィルターを使用するか
        kalman_filter_type: カルマンフィルタータイプ
        kalman_process_noise: プロセスノイズ
        kalman_observation_noise: 観測ノイズ
        **kwargs: その他のパラメータ
        
    Returns:
        トレンド強度値
    """
    indicator = PhasorTrendFilter(
        period=period,
        trend_threshold=trend_threshold,
        src_type=src_type,
        use_kalman_filter=use_kalman_filter,
        kalman_filter_type=kalman_filter_type,
        kalman_process_noise=kalman_process_noise,
        kalman_observation_noise=kalman_observation_noise,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== フェーザー分析トレンドフィルター インジケーターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # トレンドとレンジが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # トレンド相場
            change = 0.002 + np.random.normal(0, 0.01)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.008)
        elif i < 150:  # 強いトレンド相場
            change = 0.004 + np.random.normal(0, 0.015)
        else:  # レンジ相場
            change = np.random.normal(0, 0.006)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
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
    
    # 基本版フェーザー分析トレンドフィルターをテスト
    print("\\n基本版フェーザー分析トレンドフィルターをテスト中...")
    ptf_basic = PhasorTrendFilter(
        period=28,
        trend_threshold=6.0,
        src_type='close',
        use_kalman_filter=False
    )
    try:
        result_basic = ptf_basic.calculate(df)
        print(f"  結果の型: {type(result_basic)}")
        print(f"  トレンド強度配列の形状: {result_basic.values.shape}")
        print(f"  フェーザー角度配列の形状: {result_basic.phase_angle.shape}")
        print(f"  状態配列の形状: {result_basic.state.shape}")
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()
        result_basic = None
    
    if result_basic is not None:
        valid_count = np.sum(~np.isnan(result_basic.values))
        mean_trend = np.nanmean(result_basic.values)
        mean_phase = np.nanmean(result_basic.phase_angle)
        uptrend_count = np.sum(result_basic.state == 1)
        downtrend_count = np.sum(result_basic.state == -1)
        cycling_count = np.sum(result_basic.state == 0)
        buy_signals = np.sum(result_basic.signal == 1)
        sell_signals = np.sum(result_basic.signal == -1)
        
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  平均トレンド強度: {mean_trend:.4f}")
        print(f"  平均フェーザー角度: {mean_phase:.2f}°")
        print(f"  上昇トレンド: {uptrend_count}期間")
        print(f"  下降トレンド: {downtrend_count}期間")
        print(f"  サイクリング: {cycling_count}期間")
        print(f"  買いシグナル: {buy_signals}回")
        print(f"  売りシグナル: {sell_signals}回")
    else:
        print("  基本版フェーザー分析トレンドフィルターの計算に失敗しました")
    
    # カルマンフィルター版をテスト
    print("\\nカルマンフィルター版フェーザー分析トレンドフィルターをテスト中...")
    ptf_kalman = PhasorTrendFilter(
        period=28,
        trend_threshold=6.0,
        src_type='close',
        use_kalman_filter=True,
        kalman_filter_type='unscented'
    )
    try:
        result_kalman = ptf_kalman.calculate(df)
        
        valid_count_kalman = np.sum(~np.isnan(result_kalman.values))
        mean_trend_kalman = np.nanmean(result_kalman.values)
        
        print(f"  有効値数: {valid_count_kalman}/{len(df)}")
        print(f"  平均トレンド強度（カルマン版）: {mean_trend_kalman:.4f}")
        
        # 比較統計
        if result_basic is not None and valid_count > 0 and valid_count_kalman > 0:
            min_length = min(valid_count, valid_count_kalman)
            correlation = np.corrcoef(
                result_basic.values[~np.isnan(result_basic.values)][-min_length:],
                result_kalman.values[~np.isnan(result_kalman.values)][-min_length:]
            )[0, 1]
            print(f"  基本版とカルマン版の相関: {correlation:.4f}")
    except Exception as e:
        print(f"  カルマンフィルター版でエラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n=== テスト完了 ===")