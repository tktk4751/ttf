#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
import pandas as pd
from numba import njit
import traceback

from .indicator import Indicator
from .price_source import PriceSource

# 統合カルマンフィルターインポート
try:
    from .kalman.unified_kalman import UnifiedKalman
    UNIFIED_KALMAN_AVAILABLE = True
except ImportError:
    try:
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

# Ultimate Smootherインポート
try:
    from .smoother.ultimate_smoother import UltimateSmoother
    ULTIMATE_SMOOTHER_AVAILABLE = True
except ImportError:
    try:
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.smoother.ultimate_smoother import UltimateSmoother
        ULTIMATE_SMOOTHER_AVAILABLE = True
    except ImportError:
        UltimateSmoother = None
        ULTIMATE_SMOOTHER_AVAILABLE = False

# HyperER動的適応用インポート
try:
    from .trend_filter.hyper_er import HyperER
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
class EhlersInstantaneousTrendlineResult:
    """Ehlers Instantaneous Trendlineの計算結果"""
    itrend_values: np.ndarray      # ITrend値
    trigger_values: np.ndarray     # Trigger値
    signal_values: np.ndarray      # シグナル値 (1: bullish, -1: bearish, 0: neutral)
    alpha_values: np.ndarray       # 使用されたアルファ値
    filtered_prices: np.ndarray    # 最終的な平滑化後の価格
    smoothing_applied: str         # 適用された平滑化方法


@njit(fastmath=True, cache=True)
def calculate_dynamic_alpha_from_hyper_er(hyper_er_values: np.ndarray, alpha_min: float, alpha_max: float) -> np.ndarray:
    """
    HyperER値に基づいて動的にアルファ値を計算する
    
    Args:
        hyper_er_values: HyperER値の配列（0-1の範囲）
        alpha_min: アルファ最小値（HyperER値低い時に使用）
        alpha_max: アルファ最大値（HyperER値高い時に使用）
    
    Returns:
        動的アルファ値配列
        
    Note:
        - HyperER高い値（効率的）→ アルファ大きく（高速レスポンス）
        - HyperER低い値（非効率的）→ アルファ小さく（低速レスポンス）
    """
    length = len(hyper_er_values)
    dynamic_alpha = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        hyper_er_value = hyper_er_values[i] if not np.isnan(hyper_er_values[i]) else 0.0
        
        # HyperER値高い（効率的）→ アルファ大きく（高速）
        # HyperER値低い（非効率的）→ アルファ小さく（低速）
        dynamic_alpha[i] = alpha_min + hyper_er_value * (alpha_max - alpha_min)
    
    return dynamic_alpha


@njit(fastmath=True, cache=True)
def calculate_ehlers_instantaneous_trendline_core(
    price: np.ndarray, 
    alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ehlers Instantaneous Trendlineのコア計算
    
    Args:
        price: 価格データ
        alpha: アルファ値（0.01-1.0の範囲）
    
    Returns:
        ITrend値、Trigger値、シグナル値のタプル
    """
    length = len(price)
    
    # 結果配列を初期化
    itrend = np.zeros(length, dtype=np.float64)
    trigger = np.zeros(length, dtype=np.float64)
    signal = np.zeros(length, dtype=np.int8)
    
    # 初期値設定（NaN）
    for i in range(length):
        itrend[i] = np.nan
        trigger[i] = np.nan
        signal[i] = 0
    
    # 初期の7バーまでは簡単な移動平均を使用
    for i in range(min(7, length)):
        if i == 0:
            if not np.isnan(price[i]):
                itrend[i] = price[i]
        elif i == 1:
            if not np.isnan(price[i]) and not np.isnan(price[i-1]):
                itrend[i] = (price[i] + 2 * price[i-1]) / 3
        elif i == 2:
            if not np.isnan(price[i]) and not np.isnan(price[i-1]) and not np.isnan(price[i-2]):
                itrend[i] = (price[i] + 2 * price[i-1] + price[i-2]) / 4
        else:
            # i >= 3の場合の初期化処理
            if (not np.isnan(price[i]) and not np.isnan(price[i-1]) and 
                not np.isnan(price[i-2])):
                itrend[i] = (price[i] + 2 * price[i-1] + price[i-2]) / 4
    
    # メイン計算（7バー目以降）
    for i in range(7, length):
        if (np.isnan(price[i]) or np.isnan(price[i-1]) or np.isnan(price[i-2]) or
            np.isnan(itrend[i-1]) or np.isnan(itrend[i-2])):
            # NaNの場合は前の値を維持
            if i > 0:
                itrend[i] = itrend[i-1]
            continue
        
        # itrend計算
        # itrend := ((alpha - (pow(alpha, 2) / 4)) * src) + (0.5 * pow(alpha, 2) * nz(src[1])) - 
        #          ((alpha - (0.75 * pow(alpha, 2))) * nz(src[2])) + (2 * (1 - alpha) * nz(itrend[1])) - 
        #          (pow(1 - alpha, 2) * nz(itrend[2]))
        
        alpha_sq = alpha * alpha
        alpha_sq_quarter = alpha_sq / 4.0
        alpha_sq_three_quarter = 0.75 * alpha_sq
        one_minus_alpha = 1.0 - alpha
        one_minus_alpha_sq = one_minus_alpha * one_minus_alpha
        
        term1 = (alpha - alpha_sq_quarter) * price[i]
        term2 = 0.5 * alpha_sq * price[i-1]
        term3 = -(alpha - alpha_sq_three_quarter) * price[i-2]
        term4 = 2.0 * one_minus_alpha * itrend[i-1]
        term5 = -one_minus_alpha_sq * itrend[i-2]
        
        itrend[i] = term1 + term2 + term3 + term4 + term5
    
    # Trigger計算とシグナル生成
    for i in range(2, length):
        if not np.isnan(itrend[i]) and not np.isnan(itrend[i-2]):
            # trigger = (2 * itrend) - nz(itrend[2])
            trigger[i] = 2.0 * itrend[i] - itrend[i-2]
            
            # シグナル生成
            if not np.isnan(trigger[i]):
                if trigger[i] > itrend[i]:
                    signal[i] = 1  # Bullish
                elif trigger[i] < itrend[i]:
                    signal[i] = -1  # Bearish
                else:
                    signal[i] = 0  # Neutral
    
    return itrend, trigger, signal


@njit(fastmath=True, cache=True)
def calculate_ehlers_instantaneous_trendline_dynamic_alpha_core(
    price: np.ndarray, 
    dynamic_alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    動的アルファ対応Ehlers Instantaneous Trendline計算
    
    Args:
        price: 価格データ
        dynamic_alpha: 動的アルファ値配列
    
    Returns:
        ITrend値、Trigger値、シグナル値のタプル
    """
    length = len(price)
    
    # 結果配列を初期化
    itrend = np.zeros(length, dtype=np.float64)
    trigger = np.zeros(length, dtype=np.float64)
    signal = np.zeros(length, dtype=np.int8)
    
    # 初期値設定（NaN）
    for i in range(length):
        itrend[i] = np.nan
        trigger[i] = np.nan
        signal[i] = 0
    
    # 初期の7バーまでは簡単な移動平均を使用
    for i in range(min(7, length)):
        if i == 0:
            if not np.isnan(price[i]):
                itrend[i] = price[i]
        elif i == 1:
            if not np.isnan(price[i]) and not np.isnan(price[i-1]):
                itrend[i] = (price[i] + 2 * price[i-1]) / 3
        elif i == 2:
            if not np.isnan(price[i]) and not np.isnan(price[i-1]) and not np.isnan(price[i-2]):
                itrend[i] = (price[i] + 2 * price[i-1] + price[i-2]) / 4
        else:
            if (not np.isnan(price[i]) and not np.isnan(price[i-1]) and 
                not np.isnan(price[i-2])):
                itrend[i] = (price[i] + 2 * price[i-1] + price[i-2]) / 4
    
    # メイン計算（7バー目以降）
    for i in range(7, length):
        if (np.isnan(price[i]) or np.isnan(price[i-1]) or np.isnan(price[i-2]) or
            np.isnan(itrend[i-1]) or np.isnan(itrend[i-2])):
            # NaNの場合は前の値を維持
            if i > 0:
                itrend[i] = itrend[i-1]
            continue
        
        # 動的アルファ値を取得
        alpha = dynamic_alpha[i] if i < len(dynamic_alpha) and not np.isnan(dynamic_alpha[i]) else 0.07
        
        # itrend計算（動的アルファを使用）
        alpha_sq = alpha * alpha
        alpha_sq_quarter = alpha_sq / 4.0
        alpha_sq_three_quarter = 0.75 * alpha_sq
        one_minus_alpha = 1.0 - alpha
        one_minus_alpha_sq = one_minus_alpha * one_minus_alpha
        
        term1 = (alpha - alpha_sq_quarter) * price[i]
        term2 = 0.5 * alpha_sq * price[i-1]
        term3 = -(alpha - alpha_sq_three_quarter) * price[i-2]
        term4 = 2.0 * one_minus_alpha * itrend[i-1]
        term5 = -one_minus_alpha_sq * itrend[i-2]
        
        itrend[i] = term1 + term2 + term3 + term4 + term5
    
    # Trigger計算とシグナル生成
    for i in range(2, length):
        if not np.isnan(itrend[i]) and not np.isnan(itrend[i-2]):
            trigger[i] = 2.0 * itrend[i] - itrend[i-2]
            
            # シグナル生成
            if not np.isnan(trigger[i]):
                if trigger[i] > itrend[i]:
                    signal[i] = 1  # Bullish
                elif trigger[i] < itrend[i]:
                    signal[i] = -1  # Bearish
                else:
                    signal[i] = 0  # Neutral
    
    return itrend, trigger, signal


class EhlersInstantaneousTrendline(Indicator):
    """
    Ehlers Instantaneous Trendline V2
    Copyright (c) 2019-present, Franklin Moormann (cheatcountry)
    
    John Ehlersによる瞬時トレンドライン指標：
    - 価格の瞬時的なトレンドを検出
    - アルファ値による応答性の調整
    - Trigger線とITrend線の関係でシグナル生成
    - HyperERによる動的アルファ適応機能
    
    特徴:
    - ITrend: 瞬時トレンドライン
    - Trigger: (2 * ITrend) - ITrend[2]
    - シグナル: Trigger > ITrend (Bullish), Trigger < ITrend (Bearish)
    - プライスソースとカルマン統合フィルター対応
    - HyperERによるアルファ値動的適応
    """
    
    def __init__(
        self,
        alpha: float = 0.07,                    # アルファ値（デフォルト: 0.07）
        src_type: str = 'hl2',                  # ソースタイプ
        # HyperER動的適応パラメータ
        enable_hyper_er_adaptation: bool = True,  # HyperER動的適応を有効にするか
        hyper_er_period: int = 14,               # HyperER計算期間
        hyper_er_midline_period: int = 100,      # HyperERミッドライン期間
        alpha_min: float = 0.04,                 # アルファ最小値（HyperER低い時）
        alpha_max: float = 0.15,                 # アルファ最大値（HyperER高い時）
        # 平滑化モード設定
        smoothing_mode: str = 'none',            # 'none', 'kalman', 'ultimate', 'kalman_ultimate'
        # 統合カルマンフィルターパラメータ
        kalman_filter_type: str = 'simple',      # カルマンフィルタータイプ
        kalman_process_noise: float = 1e-5,      # プロセスノイズ
        kalman_min_observation_noise: float = 1e-6, # 最小観測ノイズ
        kalman_adaptation_window: int = 5,       # 適応ウィンドウ
        # Ultimate Smootherパラメータ
        ultimate_smoother_period: int = 10       # Ultimate Smoother期間
    ):
        """
        コンストラクタ
        
        Args:
            alpha: アルファ値（0.01-1.0の範囲、デフォルト: 0.07）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4', 'oc2')
            enable_hyper_er_adaptation: HyperER動的適応を有効にするか（デフォルト: True）
            hyper_er_period: HyperER計算期間（デフォルト: 14）
            hyper_er_midline_period: HyperERミッドライン期間（デフォルト: 100）
            alpha_min: アルファ最小値（HyperER低い時）（デフォルト: 0.04）
            alpha_max: アルファ最大値（HyperER高い時）（デフォルト: 0.15）
            smoothing_mode: 平滑化モード（デフォルト: 'none'）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'simple'）
            kalman_process_noise: プロセスノイズ（デフォルト: 1e-5）
            kalman_min_observation_noise: 最小観測ノイズ（デフォルト: 1e-6）
            kalman_adaptation_window: 適応ウィンドウ（デフォルト: 5）
            ultimate_smoother_period: Ultimate Smoother期間（デフォルト: 10）
        """
        # 動的適応文字列の作成
        adaptation_str = ""
        if enable_hyper_er_adaptation:
            adaptation_str += f"_hyper_er({hyper_er_period},{hyper_er_midline_period},alpha_{alpha_min}-{alpha_max})"
        if smoothing_mode != 'none':
            adaptation_str += f"_smooth({smoothing_mode})"
        
        # 指標名の作成
        alpha_str = f"{alpha_min}-{alpha_max}" if enable_hyper_er_adaptation else str(alpha)
        indicator_name = f"EhlersInstantaneousTrendline(alpha={alpha_str}, src={src_type}{adaptation_str})"
        super().__init__(indicator_name)
        
        # パラメータの検証
        if alpha <= 0 or alpha > 1:
            raise ValueError("alphaは0より大きく1以下である必要があります")
        if alpha_min <= 0 or alpha_min > 1:
            raise ValueError("alpha_minは0より大きく1以下である必要があります")
        if alpha_max <= 0 or alpha_max > 1:
            raise ValueError("alpha_maxは0より大きく1以下である必要があります")
        if alpha_min >= alpha_max:
            raise ValueError("alpha_maxはalpha_minより大きい必要があります")
        
        # 平滑化モード設定の検証
        valid_smoothing_modes = ['none', 'kalman', 'ultimate', 'kalman_ultimate']
        if smoothing_mode not in valid_smoothing_modes:
            raise ValueError(f"無効な平滑化モード: {smoothing_mode}。有効なオプション: {', '.join(valid_smoothing_modes)}")
        
        # パラメータを保存
        self.alpha = alpha
        self.src_type = src_type.lower()
        
        # HyperER動的適応パラメータ
        self.enable_hyper_er_adaptation = enable_hyper_er_adaptation
        self.hyper_er_period = hyper_er_period
        self.hyper_er_midline_period = hyper_er_midline_period
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # 平滑化モード設定
        self.smoothing_mode = smoothing_mode
        
        # 統合カルマンフィルターパラメータ
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_min_observation_noise = kalman_min_observation_noise
        self.kalman_adaptation_window = kalman_adaptation_window
        
        # Ultimate Smootherパラメータ
        self.ultimate_smoother_period = ultimate_smoother_period
        
        # HyperER動的適応インジケーターの初期化（遅延インポート）
        self.hyper_er = None
        self._last_hyper_er_values = None
        self._hyper_er_initialized = False
        
        # 平滑化フィルターの初期化
        self.kalman_filter = None
        self.ultimate_smoother = None
        
        # カルマンフィルターの初期化（必要な場合）
        if self.smoothing_mode in ['kalman', 'kalman_ultimate']:
            if not UNIFIED_KALMAN_AVAILABLE:
                self.logger.error("統合カルマンフィルターが利用できません。indicators.kalman.unified_kalmanをインポートできません。")
                self.smoothing_mode = 'none'
                self.logger.warning("平滑化機能を無効にしました")
            else:
                try:
                    self.kalman_filter = UnifiedKalman(
                        filter_type=self.kalman_filter_type,
                        src_type=self.src_type,
                        process_noise=self.kalman_process_noise,
                        min_observation_noise=self.kalman_min_observation_noise,
                        adaptation_window=self.kalman_adaptation_window
                    )
                    self.logger.info(f"統合カルマンフィルターを初期化しました: {self.kalman_filter_type}")
                except Exception as e:
                    self.logger.error(f"統合カルマンフィルターの初期化に失敗: {e}")
                    self.smoothing_mode = 'none'
                    self.logger.warning("平滑化機能を無効にしました")
        
        # Ultimate Smootherの初期化（必要な場合）
        if self.smoothing_mode in ['ultimate', 'kalman_ultimate']:
            if not ULTIMATE_SMOOTHER_AVAILABLE:
                self.logger.error("Ultimate Smootherが利用できません。indicators.smoother.ultimate_smootherをインポートできません。")
                if self.smoothing_mode == 'ultimate':
                    self.smoothing_mode = 'none'
                elif self.smoothing_mode == 'kalman_ultimate':
                    self.smoothing_mode = 'kalman'
                self.logger.warning("Ultimate Smoother機能を無効にしました")
            else:
                try:
                    self.ultimate_smoother = UltimateSmoother(
                        period=self.ultimate_smoother_period,
                        src_type=self.src_type,
                        period_mode='fixed'
                    )
                    self.logger.info(f"Ultimate Smootherを初期化しました: 期間={self.ultimate_smoother_period}")
                except Exception as e:
                    self.logger.error(f"Ultimate Smootherの初期化に失敗: {e}")
                    if self.smoothing_mode == 'ultimate':
                        self.smoothing_mode = 'none'
                    elif self.smoothing_mode == 'kalman_ultimate':
                        self.smoothing_mode = 'kalman'
                    self.logger.warning("Ultimate Smoother機能を無効にしました")
        
        # ソースタイプの検証（PriceSourceで処理されるため削除可能だが、互換性のため残す）
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
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
            
            # パラメータ情報
            params_sig = f"{self.alpha}_{self.src_type}_{self.enable_hyper_er_adaptation}"
            
            # ハッシュ計算
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.alpha}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> EhlersInstantaneousTrendlineResult:
        """
        Ehlers Instantaneous Trendlineを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            EhlersInstantaneousTrendlineResult: ITrend値、Trigger値、シグナル値、アルファ値を含む結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return EhlersInstantaneousTrendlineResult(
                    itrend_values=cached_result.itrend_values.copy(),
                    trigger_values=cached_result.trigger_values.copy(),
                    signal_values=cached_result.signal_values.copy(),
                    alpha_values=cached_result.alpha_values.copy(),
                    filtered_prices=cached_result.filtered_prices.copy(),
                    smoothing_applied=cached_result.smoothing_applied
                )
            
            # 1. ソース価格データを取得
            source_price = PriceSource.calculate_source(data, self.src_type)
            
            # 2. 平滑化処理（モードに応じて選択）
            smoothed_price, smoothing_applied = self._apply_smoothing(source_price, data)
            
            # 3. 計算用の価格データを設定
            price = smoothed_price
            
            # NumPy配列に変換（float64型で統一）
            price = np.asarray(price, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < 7:
                self.logger.warning(f"データ長({data_length})が最小必要期間(7)より短いです")
            
            # HyperER動的適応の計算（オプション）
            dynamic_alpha = None
            alpha_values = None
            
            if self.enable_hyper_er_adaptation:
                # 遅延インポートでHyperERを初期化
                if not self._hyper_er_initialized:
                    try:
                        if not HYPER_ER_AVAILABLE:
                            raise ImportError("HyperERが利用できません")
                        
                        self.hyper_er = HyperER(
                            period=self.hyper_er_period,
                            midline_period=self.hyper_er_midline_period,
                            er_src_type=self.src_type
                        )
                        self._hyper_er_initialized = True
                        self.logger.info(f"HyperER動的適応を初期化しました: 期間={self.hyper_er_period}")
                    except Exception as e:
                        self.logger.warning(f"HyperERインジケーターの初期化に失敗しました: {e}")
                        self.enable_hyper_er_adaptation = False
                
                if self.hyper_er is not None:
                    try:
                        hyper_er_result = self.hyper_er.calculate(data)
                        if hyper_er_result is not None and hasattr(hyper_er_result, 'values'):
                            hyper_er_values = np.asarray(hyper_er_result.values, dtype=np.float64)
                            dynamic_alpha = calculate_dynamic_alpha_from_hyper_er(
                                hyper_er_values, self.alpha_min, self.alpha_max
                            )
                            alpha_values = dynamic_alpha.copy()
                            self._last_hyper_er_values = hyper_er_values.copy()
                    except Exception as e:
                        self.logger.warning(f"HyperER動的適応計算に失敗しました: {e}")
                        # フォールバック: 前回の値を使用または固定値
                        if self._last_hyper_er_values is not None:
                            dynamic_alpha = calculate_dynamic_alpha_from_hyper_er(
                                self._last_hyper_er_values, self.alpha_min, self.alpha_max
                            )
                            alpha_values = dynamic_alpha.copy()
            
            # アルファ値の配列を作成（動的適応なしの場合）
            if alpha_values is None:
                alpha_values = np.full(data_length, self.alpha, dtype=np.float64)
            
            # Ehlers Instantaneous Trendlineの計算
            if self.enable_hyper_er_adaptation and dynamic_alpha is not None:
                # 動的アルファ版を使用
                itrend_values, trigger_values, signal_values = calculate_ehlers_instantaneous_trendline_dynamic_alpha_core(
                    price, dynamic_alpha
                )
            else:
                # 固定アルファ版を使用
                itrend_values, trigger_values, signal_values = calculate_ehlers_instantaneous_trendline_core(
                    price, self.alpha
                )
            
            # 結果の保存
            result = EhlersInstantaneousTrendlineResult(
                itrend_values=itrend_values.copy(),
                trigger_values=trigger_values.copy(),
                signal_values=signal_values.copy(),
                alpha_values=alpha_values.copy(),
                filtered_prices=smoothed_price.copy() if isinstance(smoothed_price, np.ndarray) else np.array(smoothed_price),
                smoothing_applied=smoothing_applied
            )
            
            # キャッシュを更新
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = itrend_values  # 基底クラスの要件を満たすため（ITrend値をメインとする）
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Ehlers Instantaneous Trendline計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = EhlersInstantaneousTrendlineResult(
                itrend_values=np.array([]),
                trigger_values=np.array([]),
                signal_values=np.array([]),
                alpha_values=np.array([]),
                filtered_prices=np.array([]),
                smoothing_applied='error'
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """ITrend値のみを取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.itrend_values.copy()
    
    def get_itrend_values(self) -> Optional[np.ndarray]:
        """
        ITrend値を取得する
        
        Returns:
            np.ndarray: ITrend値
        """
        return self.get_values()
    
    def get_trigger_values(self) -> Optional[np.ndarray]:
        """
        Trigger値を取得する
        
        Returns:
            np.ndarray: Trigger値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.trigger_values.copy()
    
    def get_signal_values(self) -> Optional[np.ndarray]:
        """
        シグナル値を取得する
        
        Returns:
            np.ndarray: シグナル値 (1: bullish, -1: bearish, 0: neutral)
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.signal_values.copy()
    
    def get_alpha_values(self) -> Optional[np.ndarray]:
        """
        使用されたアルファ値を取得する
        
        Returns:
            np.ndarray: アルファ値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.alpha_values.copy()
    
    def get_smoothed_prices(self) -> Optional[np.ndarray]:
        """
        平滑化後の価格を取得する
        
        Returns:
            np.ndarray: 平滑化後の価格
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.filtered_prices.copy()
    
    def get_smoothing_applied(self) -> Optional[str]:
        """
        適用された平滑化方法を取得する
        
        Returns:
            str: 適用された平滑化方法
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.smoothing_applied
    
    def _apply_smoothing(self, source_price: np.ndarray, data: Union[pd.DataFrame, np.ndarray]) -> tuple[np.ndarray, str]:
        """
        指定された平滑化モードに応じて価格を平滑化する
        
        Args:
            source_price: 元の価格データ
            data: 元の価格データ（DataFrameまたはNumPy配列）
            
        Returns:
            tuple[np.ndarray, str]: 平滑化後の価格、適用された平滑化方法
        """
        smoothed_price = source_price.copy()
        applied_method = 'none'
        
        try:
            if self.smoothing_mode == 'none':
                # 平滑化なし
                applied_method = 'none'
                
            elif self.smoothing_mode == 'kalman':
                # カルマンフィルターのみ
                if self.kalman_filter is not None:
                    kalman_result = self.kalman_filter.calculate(data)
                    if kalman_result is not None and hasattr(kalman_result, 'values'):
                        smoothed_price = kalman_result.values
                        applied_method = 'kalman'
                        self.logger.debug("カルマンフィルターを適用しました")
                    else:
                        self.logger.warning("カルマンフィルター結果が無効。元の価格を使用します。")
                        applied_method = 'kalman_failed'
                else:
                    self.logger.warning("カルマンフィルターが初期化されていません。")
                    applied_method = 'kalman_unavailable'
                    
            elif self.smoothing_mode == 'ultimate':
                # Ultimate Smootherのみ
                if self.ultimate_smoother is not None:
                    # 価格データをDataFrame形式に変換してUltimate Smootherに渡す
                    temp_df = self._create_temp_dataframe(smoothed_price, data)
                    ultimate_result = self.ultimate_smoother.calculate(temp_df)
                    if ultimate_result is not None and hasattr(ultimate_result, 'values'):
                        smoothed_price = ultimate_result.values
                        applied_method = 'ultimate'
                        self.logger.debug("Ultimate Smootherを適用しました")
                    else:
                        self.logger.warning("Ultimate Smoother結果が無効。元の価格を使用します。")
                        applied_method = 'ultimate_failed'
                else:
                    self.logger.warning("Ultimate Smootherが初期化されていません。")
                    applied_method = 'ultimate_unavailable'
                    
            elif self.smoothing_mode == 'kalman_ultimate':
                # カルマンフィルター → Ultimate Smootherの順で二次平滑化
                kalman_applied = False
                ultimate_applied = False
                
                # 第1段階: カルマンフィルター
                if self.kalman_filter is not None:
                    kalman_result = self.kalman_filter.calculate(data)
                    if kalman_result is not None and hasattr(kalman_result, 'values'):
                        smoothed_price = kalman_result.values
                        kalman_applied = True
                        self.logger.debug("第1段階: カルマンフィルターを適用しました")
                    else:
                        self.logger.warning("カルマンフィルター結果が無効。元の価格を使用します。")
                else:
                    self.logger.warning("カルマンフィルターが初期化されていません。")
                
                # 第2段階: Ultimate Smoother（カルマンフィルター後の価格に対して）
                if self.ultimate_smoother is not None:
                    # カルマンフィルター後の価格をDataFrame形式に変換
                    temp_df = self._create_temp_dataframe(smoothed_price, data)
                    ultimate_result = self.ultimate_smoother.calculate(temp_df)
                    if ultimate_result is not None and hasattr(ultimate_result, 'values'):
                        smoothed_price = ultimate_result.values
                        ultimate_applied = True
                        self.logger.debug("第2段階: Ultimate Smootherを適用しました")
                    else:
                        self.logger.warning("Ultimate Smoother結果が無効。")
                else:
                    self.logger.warning("Ultimate Smootherが初期化されていません。")
                
                # 適用結果の確認
                if kalman_applied and ultimate_applied:
                    applied_method = 'kalman_ultimate'
                elif kalman_applied:
                    applied_method = 'kalman_only'
                elif ultimate_applied:
                    applied_method = 'ultimate_only'
                else:
                    applied_method = 'kalman_ultimate_failed'
                    
        except Exception as e:
            self.logger.error(f"平滑化処理中にエラー: {e}")
            applied_method = f'{self.smoothing_mode}_error'
        
        return smoothed_price, applied_method
    
    def _create_temp_dataframe(self, price_array: np.ndarray, original_data: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        価格配列を使用してUltimate Smoother用の一時的なDataFrameを作成
        
        Args:
            price_array: 価格データ配列
            original_data: 元のデータ（インデックス取得用）
            
        Returns:
            pd.DataFrame: Ultimate Smoother用DataFrame
        """
        try:
            # インデックスの取得
            if isinstance(original_data, pd.DataFrame):
                index = original_data.index
            else:
                # NumPy配列の場合は連番インデックスを作成
                index = pd.RangeIndex(len(price_array))
            
            # 基本的なOHLCVを作成（価格は全て同じ値を使用）
            temp_df = pd.DataFrame({
                'open': price_array,
                'high': price_array,
                'low': price_array,
                'close': price_array,
                'volume': np.ones_like(price_array)  # ダミーボリューム
            }, index=index[:len(price_array)])
            
            return temp_df
            
        except Exception as e:
            self.logger.error(f"一時的なDataFrame作成中にエラー: {e}")
            # フォールバック: 最小限のDataFrameを作成
            return pd.DataFrame({
                'open': price_array,
                'high': price_array,
                'low': price_array,
                'close': price_array,
                'volume': np.ones_like(price_array)
            })
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        if self.kalman_filter:
            self.kalman_filter.reset()
        if self.ultimate_smoother:
            self.ultimate_smoother.reset()
        self._result_cache = {}
        self._cache_keys = []