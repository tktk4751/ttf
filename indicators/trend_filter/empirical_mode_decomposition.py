#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import jit, njit, float64

from ..indicator import Indicator
from ..price_source import PriceSource
from ..smoother.unified_smoother import UnifiedSmoother
from ..utils.percentile_analysis import (
    calculate_percentile,
    calculate_trend_classification,
    PercentileAnalysisMixin
)

# 条件付きインポート（オプション機能）
try:
    from ..cycle.ehlers_unified_dc import EhlersUnifiedDC
    EHLERS_UNIFIED_DC_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行（パス調整付き）
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


@dataclass
class EMDResult:
    """Empirical Mode Decompositionの計算結果"""
    bandpass: np.ndarray              # バンドパスフィルター出力（サイクル成分）
    trend: np.ndarray                 # トレンド成分
    peaks: np.ndarray                 # ピーク値の配列
    valleys: np.ndarray               # バレー値の配列
    avg_peak: np.ndarray              # 平均ピーク値
    avg_valley: np.ndarray            # 平均バレー値
    upper_threshold: np.ndarray       # 上部閾値
    lower_threshold: np.ndarray       # 下部閾値
    mode_signal: np.ndarray           # モード信号（1=上昇トレンド、-1=下降トレンド、0=サイクル）
    filtered_price: np.ndarray        # カルマンフィルター適用後の価格（オプション）
    smoothed_bandpass: np.ndarray     # 平滑化されたバンドパス出力（オプション）
    # パーセンタイルベースのトレンド分析
    percentiles: Optional[np.ndarray]     # パーセンタイル値
    trend_state: Optional[np.ndarray]     # トレンド状態（-1=下降、0=レンジ、1=上昇）
    trend_intensity: Optional[np.ndarray] # トレンド強度（0-1）


@njit(fastmath=True, cache=True)
def calculate_bandpass_filter_numba(
    prices: np.ndarray,
    period: int,
    delta: float
) -> np.ndarray:
    """
    バンドパスフィルターを計算する（Numba最適化版）
    
    Args:
        prices: 価格配列
        period: フィルター周期
        delta: バンド幅パラメータ（小さいほど狭帯域）
        
    Returns:
        バンドパスフィルター出力
    """
    length = len(prices)
    bp = np.full(length, np.nan, dtype=np.float64)
    
    if length < 3:
        return bp
    
    # フィルターパラメータの計算
    beta = np.cos(2.0 * np.pi / period)
    gamma = 1.0 / np.cos(720.0 * delta / period * np.pi / 180.0)
    alpha = gamma - np.sqrt(gamma * gamma - 1.0)
    
    # 初期値設定
    bp[0] = 0.0
    bp[1] = 0.0
    
    # バンドパスフィルターの計算
    for i in range(2, length):
        bp[i] = (0.5 * (1.0 - alpha) * (prices[i] - prices[i - 2]) + 
                beta * (1.0 + alpha) * bp[i - 1] - 
                alpha * bp[i - 2])
    
    return bp


@njit(fastmath=True, cache=True)
def calculate_trend_component_numba(
    bandpass: np.ndarray,
    period: int
) -> np.ndarray:
    """
    トレンド成分を計算する（Numba最適化版）
    
    Args:
        bandpass: バンドパスフィルター出力
        period: 移動平均期間（通常は2*period）
        
    Returns:
        トレンド成分
    """
    length = len(bandpass)
    trend = np.full(length, np.nan, dtype=np.float64)
    
    avg_period = 2 * period  # 2周期の平均
    
    for i in range(avg_period - 1, length):
        # 有効なデータの平均を計算
        sum_val = 0.0
        count = 0
        
        for j in range(i - avg_period + 1, i + 1):
            if not np.isnan(bandpass[j]):
                sum_val += bandpass[j]
                count += 1
        
        if count > 0:
            trend[i] = sum_val / count
    
    return trend


@njit(fastmath=True, cache=True)
def detect_peaks_valleys_numba(
    bandpass: np.ndarray
) -> tuple:
    """
    ピークとバレーを検出する（Numba最適化版）
    
    Args:
        bandpass: バンドパスフィルター出力
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ピーク配列, バレー配列)
    """
    length = len(bandpass)
    peaks = np.full(length, np.nan, dtype=np.float64)
    valleys = np.full(length, np.nan, dtype=np.float64)
    
    if length < 3:
        return peaks, valleys
    
    for i in range(1, length - 1):
        if (not np.isnan(bandpass[i]) and 
            not np.isnan(bandpass[i - 1]) and 
            not np.isnan(bandpass[i + 1])):
            
            # ピークの検出
            if bandpass[i] > bandpass[i - 1] and bandpass[i] > bandpass[i + 1]:
                peaks[i] = bandpass[i]
            
            # バレーの検出
            if bandpass[i] < bandpass[i - 1] and bandpass[i] < bandpass[i + 1]:
                valleys[i] = bandpass[i]
    
    return peaks, valleys


@njit(fastmath=True, cache=True)
def calculate_mode_signal_numba(
    trend: np.ndarray,
    upper_threshold: np.ndarray,
    lower_threshold: np.ndarray
) -> np.ndarray:
    """
    モード信号を計算する（Numba最適化版）
    
    Args:
        trend: トレンド成分
        upper_threshold: 上部閾値
        lower_threshold: 下部閾値
        
    Returns:
        モード信号（1=上昇トレンド、-1=下降トレンド、0=サイクル）
    """
    length = len(trend)
    mode_signal = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if (not np.isnan(trend[i]) and 
            not np.isnan(upper_threshold[i]) and 
            not np.isnan(lower_threshold[i])):
            
            if trend[i] > upper_threshold[i]:
                mode_signal[i] = 1.0  # 上昇トレンド
            elif trend[i] < lower_threshold[i]:
                mode_signal[i] = -1.0  # 下降トレンド
            else:
                mode_signal[i] = 0.0  # サイクルモード
    
    return mode_signal


# @njit(fastmath=True, cache=False) # 一時的にNumba無効化
def calculate_averaged_peaks_valleys_improved_numba(
    peaks: np.ndarray,
    valleys: np.ndarray,
    avg_period: int,
    fraction: float
) -> tuple:
    """
    平均化されたピーク・バレーと閾値を計算する（Numba最適化版）
    論文に忠実でありながら実用的な改善を含む
    
    Args:
        peaks: ピーク配列
        valleys: バレー配列
        avg_period: 平均化期間
        fraction: 閾値計算用の係数
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        (平均ピーク, 平均バレー, 上部閾値, 下部閾値)
    """
    length = len(peaks)
    avg_peak = np.full(length, np.nan, dtype=np.float64)
    avg_valley = np.full(length, np.nan, dtype=np.float64)
    upper_threshold = np.full(length, np.nan, dtype=np.float64)
    lower_threshold = np.full(length, np.nan, dtype=np.float64)
    
    min_required_points = 1  # 最低必要なピーク・バレー数（実用性を重視）
    
    for i in range(avg_period - 1, length):
        # ピークの平均（論文に忠実な実装 + 実用的改善）
        peak_sum = 0.0
        peak_count = 0
        
        # 基本期間での検索
        for j in range(i - avg_period + 1, i + 1):
            if not np.isnan(peaks[j]):
                peak_sum += peaks[j]
                peak_count += 1
        
        # 十分なピークが見つからない場合、拡張検索（実用的改善）
        if peak_count < min_required_points:
            # 拡張検索：現在位置より前の全有効ピークを順方向で探索
            search_sum = 0.0
            search_count = 0
            
            # 順方向で全有効ピークを収集
            for j in range(i + 1):
                if not np.isnan(peaks[j]):
                    search_sum += peaks[j]
                    search_count += 1
            
            if search_count >= min_required_points:
                peak_sum = search_sum
                peak_count = search_count
        
        if peak_count >= min_required_points:
            avg_peak[i] = peak_sum / peak_count
        
        # バレーの平均（同様の改善）
        valley_sum = 0.0
        valley_count = 0
        
        # 基本期間での検索
        for j in range(i - avg_period + 1, i + 1):
            if not np.isnan(valleys[j]):
                valley_sum += valleys[j]
                valley_count += 1
        
        # 十分なバレーが見つからない場合、拡張検索
        if valley_count < min_required_points:
            # 拡張検索：現在位置より前の全有効バレーを順方向で探索
            search_sum = 0.0
            search_count = 0
            
            # 順方向で全有効バレーを収集
            for j in range(i + 1):
                if not np.isnan(valleys[j]):
                    search_sum += valleys[j]
                    search_count += 1
            
            if search_count >= min_required_points:
                valley_sum = search_sum
                valley_count = search_count
        
        if valley_count >= min_required_points:
            avg_valley[i] = valley_sum / valley_count
        
        # 閾値の計算（論文通り）
        if not np.isnan(avg_peak[i]):
            upper_threshold[i] = fraction * avg_peak[i]
        
        if not np.isnan(avg_valley[i]):
            lower_threshold[i] = fraction * avg_valley[i]
    
    return avg_peak, avg_valley, upper_threshold, lower_threshold


class EMD(Indicator, PercentileAnalysisMixin):
    """
    Empirical Mode Decomposition（経験的モード分解）インジケーター
    
    John F. EhlersとRic Wayによる手法：
    市場データをサイクル成分とトレンド成分に分解し、
    市場モード（サイクルモード vs トレンドモード）を判定
    
    特徴:
    - バンドパスフィルターでサイクル成分を抽出
    - バンドパス出力の移動平均でトレンド成分を抽出
    - ピーク・バレー分析によるモード判定
    - カルマンフィルター・スムーサー統合対応
    - パーセンタイル分析機能付き
    
    計算手順:
    1. ソース価格データを取得
    2. カルマンフィルターを適用（オプション）
    3. バンドパスフィルターでサイクル成分を抽出
    4. バンドパス出力の移動平均でトレンド成分を計算
    5. ピーク・バレーを検出して平均化
    6. 閾値によるモード判定
    """
    
    def __init__(
        self,
        period: int = 20,
        delta: float = 0.1,
        fraction: float = 0.25,
        avg_period: int = 50,
        src_type: str = 'hl2',
        # 平滑化オプション
        use_smoothing: bool = False,
        smoother_type: str = 'super_smoother',
        smoother_period: int = 8,
        smoother_src_type: str = 'close',
        # エラーズ統合サイクル検出器パラメータ
        use_dynamic_period: bool = False,
        detector_type: str = 'phac_e',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 55,
        min_output: int = 5,
        # 統合カルマンフィルターパラメータ
        use_kalman_filter: bool = False,
        kalman_filter_type: str = 'unscented',
        kalman_process_noise: float = 0.01,
        kalman_observation_noise: float = 0.001,
        # パーセンタイルベーストレンド分析パラメータ
        enable_percentile_analysis: bool = True,
        percentile_lookback_period: int = 50,
        percentile_low_threshold: float = 0.25,
        percentile_high_threshold: float = 0.75
    ):
        """
        コンストラクタ
        
        Args:
            period: バンドパスフィルター周期（デフォルト: 20）
            delta: バンド幅パラメータ（デフォルト: 0.1、小さいほど狭帯域）
            fraction: 閾値計算用係数（デフォルト: 0.25）
            avg_period: ピーク・バレー平均化期間（デフォルト: 50）
            src_type: ソースタイプ（デフォルト: 'hl2'）
            use_smoothing: 平滑化を使用するか（デフォルト: False）
            smoother_type: 統合スムーサータイプ（デフォルト: 'super_smoother'）
            smoother_period: スムーサー期間（デフォルト: 8）
            smoother_src_type: スムーサーソースタイプ（デフォルト: 'close'）
            use_dynamic_period: 動的期間適応を使用するか（デフォルト: False）
            detector_type: サイクル検出器タイプ（デフォルト: 'phac_e'）
            lp_period: ローパスフィルター期間（デフォルト: 5）
            hp_period: ハイパスフィルター期間（デフォルト: 144）
            cycle_part: サイクル部分（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 144）
            min_cycle: 最小サイクル期間（デフォルト: 5）
            max_output: 最大出力値（デフォルト: 55）
            min_output: 最小出力値（デフォルト: 5）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: カルマンフィルタープロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: カルマンフィルター観測ノイズ（デフォルト: 0.001）
            enable_percentile_analysis: パーセンタイル分析を有効にするか（デフォルト: True）
            percentile_lookback_period: パーセンタイル分析のルックバック期間（デフォルト: 50）
            percentile_low_threshold: パーセンタイル分析の低閾値（デフォルト: 0.25）
            percentile_high_threshold: パーセンタイル分析の高閾値（デフォルト: 0.75）
        """
        indicator_name = f"EMD({period}, delta={delta:.2f}, fraction={fraction:.2f}, src={src_type}"
        if use_dynamic_period:
            indicator_name += f", dynamic={detector_type}"
        if use_kalman_filter:
            indicator_name += f", kalman={kalman_filter_type}"
        if use_smoothing:
            indicator_name += f", smooth={smoother_type}({smoother_period})"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        
        # パラメータ保存
        self.period = period
        self.delta = delta
        self.fraction = fraction
        self.avg_period = avg_period
        self.src_type = src_type
        self.use_smoothing = use_smoothing
        self.smoother_type = smoother_type
        self.smoother_period = smoother_period
        self.smoother_src_type = smoother_src_type
        
        # エラーズ統合サイクル検出器パラメータ
        self.use_dynamic_period = use_dynamic_period
        self.detector_type = detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        
        # 統合カルマンフィルターパラメータ
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_observation_noise = kalman_observation_noise
        
        # パーセンタイルベーストレンド分析パラメータの初期化
        self._add_percentile_analysis_params(
            enable_percentile_analysis=enable_percentile_analysis,
            percentile_lookback_period=percentile_lookback_period,
            percentile_low_threshold=percentile_low_threshold,
            percentile_high_threshold=percentile_high_threshold
        )
        
        # パラメータ検証
        if self.period <= 0:
            raise ValueError("periodは0より大きい必要があります")
        if self.delta <= 0:
            raise ValueError("deltaは0より大きい必要があります")
        if self.fraction <= 0:
            raise ValueError("fractionは0より大きい必要があります")
        if self.avg_period <= 0:
            raise ValueError("avg_periodは0より大きい必要があります")
        if self.use_dynamic_period and self.max_cycle <= self.min_cycle:
            raise ValueError("max_cycleはmin_cycleより大きい必要があります")
        if self.use_kalman_filter and self.kalman_process_noise <= 0:
            raise ValueError("kalman_process_noiseは0より大きい必要があります")
        
        # エラーズ統合サイクル検出器の初期化（動的期間適応が有効な場合）
        self.cycle_detector = None
        
        if self.use_dynamic_period:
            if not EHLERS_UNIFIED_DC_AVAILABLE:
                self.logger.error("エラーズ統合サイクル検出器が利用できません。indicators.cycle.ehlers_unified_dcをインポートできません。")
                self.use_dynamic_period = False
                self.logger.warning("動的期間適応機能を無効にしました")
            else:
                try:
                    self.cycle_detector = EhlersUnifiedDC(
                        detector_type=self.detector_type,
                        cycle_part=self.cycle_part,
                        max_cycle=self.max_cycle,
                        min_cycle=self.min_cycle,
                        max_output=self.max_output,
                        min_output=self.min_output,
                        src_type='hlc3',
                        use_kalman_filter=False,
                        lp_period=self.lp_period,
                        hp_period=self.hp_period
                    )
                    self.logger.info(f"エラーズ統合サイクル検出器を初期化しました: {self.detector_type}")
                except Exception as e:
                    self.logger.error(f"エラーズ統合サイクル検出器の初期化に失敗: {e}")
                    self.use_dynamic_period = False
                    self.logger.warning("動的期間適応機能を無効にしました")
        
        # 統合カルマンフィルターの初期化（カルマンフィルターが有効な場合）
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
        
        # 統合スムーサーの初期化（オプション）
        self.smoother = None
        if self.use_smoothing:
            try:
                self.smoother = UnifiedSmoother(
                    smoother_type=self.smoother_type,
                    src_type=self.smoother_src_type,
                    period=self.smoother_period
                )
            except Exception as e:
                self.logger.error(f"統合スムーサーの初期化に失敗: {e}")
                self.use_smoothing = False
                self.logger.warning("平滑化機能を無効にしました")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
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
            
            # パラメータ情報
            param_str = (f"{self.period}_{self.delta}_{self.fraction}_{self.avg_period}_"
                        f"{self.src_type}_{self.use_smoothing}_{self.smoother_type}_"
                        f"{self.smoother_period}_{self.smoother_src_type}")
            
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(param_str)}"
            
        except Exception:
            return f"{id(data)}_{self.period}_{self.delta}"

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> EMDResult:
        """
        Empirical Mode Decompositionを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                必要なカラム: high, low, close, open（カルマンフィルター用）
        
        Returns:
            EMDResult: EMDの計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # キャッシュヒット
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return EMDResult(
                    bandpass=cached_result.bandpass.copy(),
                    trend=cached_result.trend.copy(),
                    peaks=cached_result.peaks.copy(),
                    valleys=cached_result.valleys.copy(),
                    avg_peak=cached_result.avg_peak.copy(),
                    avg_valley=cached_result.avg_valley.copy(),
                    upper_threshold=cached_result.upper_threshold.copy(),
                    lower_threshold=cached_result.lower_threshold.copy(),
                    mode_signal=cached_result.mode_signal.copy(),
                    filtered_price=cached_result.filtered_price.copy(),
                    smoothed_bandpass=cached_result.smoothed_bandpass.copy(),
                    percentiles=cached_result.percentiles.copy() if cached_result.percentiles is not None else None,
                    trend_state=cached_result.trend_state.copy() if cached_result.trend_state is not None else None,
                    trend_intensity=cached_result.trend_intensity.copy() if cached_result.trend_intensity is not None else None
                )
            
            # データの準備と検証
            if isinstance(data, pd.DataFrame):
                required_cols = ['high', 'low', 'close']
                if self.use_kalman_filter:
                    required_cols.extend(['open'])  # カルマンフィルター用
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"必要なカラムが不足しています: {missing_cols}")
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # データ長の検証
            data_length = len(data)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            min_required_length = max(self.period * 2, self.avg_period)
            if data_length < min_required_length:
                self.logger.warning(f"データ長（{data_length}）が必要な期間（{min_required_length}）より短いです")
            
            # 1. ソース価格データを取得
            source_prices = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換
            if not isinstance(source_prices, np.ndarray):
                source_prices = np.array(source_prices)
            if source_prices.dtype != np.float64:
                source_prices = source_prices.astype(np.float64)
            
            # 2. カルマンフィルターによる価格データのフィルタリング（オプション）
            filtered_prices = source_prices
            if self.use_kalman_filter and self.kalman_filter is not None:
                try:
                    # ソース価格をDataFrame形式に変換してカルマンフィルターに入力
                    if isinstance(data, pd.DataFrame):
                        kalman_input = data.copy()
                        # カルマンフィルター用のソースタイプが既に存在する場合はそのまま使用
                        if self.src_type not in kalman_input.columns:
                            kalman_input[self.src_type] = source_prices
                    else:
                        # NumPy配列の場合はDataFrameに変換
                        if data.shape[1] >= 4:  # OHLC
                            kalman_input = pd.DataFrame({
                                'open': data[:, 0],
                                'high': data[:, 1],
                                'low': data[:, 2],
                                'close': data[:, 3]
                            })
                            # ソースタイプを追加
                            kalman_input[self.src_type] = source_prices
                        else:
                            # 最小限のDataFrame
                            kalman_input = pd.DataFrame({self.src_type: source_prices})
                    
                    kalman_result = self.kalman_filter.calculate(kalman_input)
                    # カルマンフィルターの結果が辞書形式の場合は適切に処理
                    if hasattr(kalman_result, 'values'):
                        filtered_prices = kalman_result.values
                    elif hasattr(kalman_result, 'filtered_values'):
                        filtered_prices = kalman_result.filtered_values
                    else:
                        # 結果がarray形式の場合
                        filtered_prices = np.array(kalman_result)
                    
                    # NumPy配列として確保
                    if not isinstance(filtered_prices, np.ndarray):
                        filtered_prices = np.array(filtered_prices)
                    if filtered_prices.dtype != np.float64:
                        filtered_prices = filtered_prices.astype(np.float64)
                    
                    self.logger.debug("カルマンフィルターによる価格データのフィルタリングを適用しました")
                        
                except Exception as e:
                    self.logger.warning(f"カルマンフィルター適用中にエラー: {e}。元の値を使用します。")
                    filtered_prices = source_prices
            
            # 動的期間の計算（オプション）
            current_period = self.period
            if self.use_dynamic_period and self.cycle_detector is not None:
                try:
                    cycle_values = self.cycle_detector.calculate(data)
                    
                    # サイクル値から期間を計算
                    valid_cycles = cycle_values[~np.isnan(cycle_values) & (cycle_values > 0)]
                    if len(valid_cycles) > 0:
                        # 最新のサイクル値を期間として使用
                        latest_cycle = valid_cycles[-1] if len(valid_cycles) > 0 else self.period
                        current_period = max(self.min_cycle, min(int(latest_cycle), self.max_cycle))
                    
                    self.logger.debug(f"動的期間を計算しました: {current_period}")
                        
                except Exception as e:
                    self.logger.warning(f"動的期間計算中にエラー: {e}。固定期間を使用します。")
                    current_period = self.period
            
            # 3. バンドパスフィルターでサイクル成分を抽出
            bandpass = calculate_bandpass_filter_numba(
                filtered_prices, current_period, self.delta
            )
            
            # 4. 平滑化（オプション）
            smoothed_bandpass = np.full_like(bandpass, np.nan)
            if self.use_smoothing and self.smoother is not None:
                try:
                    # バンドパス出力をDataFrame形式に変換
                    # FRAMAなどのスムーサーがhigh/lowを必要とする場合に対応
                    bp_df = pd.DataFrame({
                        'close': bandpass,
                        'high': bandpass,  # バンドパス出力をhighとしても使用
                        'low': bandpass,   # バンドパス出力をlowとしても使用
                        'open': bandpass   # バンドパス出力をopenとしても使用
                    })
                    
                    # 平滑化を適用
                    smoother_result = self.smoother.calculate(bp_df)
                    if hasattr(smoother_result, 'values'):
                        smoothed_bandpass = smoother_result.values
                    else:
                        smoothed_bandpass = np.array(smoother_result)
                except Exception as e:
                    self.logger.warning(f"平滑化処理中にエラー: {e}。生の値を使用します。")
                    smoothed_bandpass = bandpass.copy()
            else:
                smoothed_bandpass = bandpass.copy()
            
            # 最終的なバンドパス値（平滑化が有効な場合は平滑化値、そうでなければ生の値）
            final_bandpass = smoothed_bandpass if self.use_smoothing else bandpass
            
            # 5. トレンド成分を計算
            trend = calculate_trend_component_numba(
                final_bandpass, current_period
            )
            
            # 6. ピークとバレーを検出
            peaks, valleys = detect_peaks_valleys_numba(final_bandpass)
            
            # 7. 平均化されたピーク・バレーと閾値を計算
            avg_peak, avg_valley, upper_threshold, lower_threshold = calculate_averaged_peaks_valleys_improved_numba(
                peaks, valleys, self.avg_period, self.fraction
            )
            
            # 8. モード信号を計算
            mode_signal = calculate_mode_signal_numba(
                trend, upper_threshold, lower_threshold
            )
            
            # 9. パーセンタイルベーストレンド分析（オプション）
            percentiles, trend_state, trend_intensity = self._calculate_percentile_analysis(
                trend, 'trend'
            )
            
            # 結果の作成
            result = EMDResult(
                bandpass=final_bandpass.copy(),
                trend=trend.copy(),
                peaks=peaks.copy(),
                valleys=valleys.copy(),
                avg_peak=avg_peak.copy(),
                avg_valley=avg_valley.copy(),
                upper_threshold=upper_threshold.copy(),
                lower_threshold=lower_threshold.copy(),
                mode_signal=mode_signal.copy(),
                filtered_price=filtered_prices.copy(),
                smoothed_bandpass=smoothed_bandpass.copy(),
                percentiles=percentiles.copy() if percentiles is not None else None,
                trend_state=trend_state.copy() if trend_state is not None else None,
                trend_intensity=trend_intensity.copy() if trend_intensity is not None else None
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            # 基底クラス用の値設定（トレンド成分をメイン値として使用）
            self._values = trend
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"EMD計算中にエラー: {error_msg}\\n{stack_trace}")
            
            # エラー時は空の結果を返す
            empty_array = np.array([])
            return EMDResult(
                bandpass=empty_array,
                trend=empty_array,
                peaks=empty_array,
                valleys=empty_array,
                avg_peak=empty_array,
                avg_valley=empty_array,
                upper_threshold=empty_array,
                lower_threshold=empty_array,
                mode_signal=empty_array,
                filtered_price=empty_array,
                smoothed_bandpass=empty_array,
                percentiles=None,
                trend_state=None,
                trend_intensity=None
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """トレンド値を取得（後方互換性のため）"""
        if not self._result_cache:
            return None
        
        result = self._get_latest_result()
        return result.trend.copy() if result else None
    
    def get_bandpass(self) -> Optional[np.ndarray]:
        """バンドパス出力を取得"""
        result = self._get_latest_result()
        return result.bandpass.copy() if result else None
    
    def get_trend(self) -> Optional[np.ndarray]:
        """トレンド成分を取得"""
        result = self._get_latest_result()
        return result.trend.copy() if result else None
    
    def get_peaks(self) -> Optional[np.ndarray]:
        """ピーク値を取得"""
        result = self._get_latest_result()
        return result.peaks.copy() if result else None
    
    def get_valleys(self) -> Optional[np.ndarray]:
        """バレー値を取得"""
        result = self._get_latest_result()
        return result.valleys.copy() if result else None
    
    def get_mode_signal(self) -> Optional[np.ndarray]:
        """モード信号を取得"""
        result = self._get_latest_result()
        return result.mode_signal.copy() if result else None
    
    def get_upper_threshold(self) -> Optional[np.ndarray]:
        """上部閾値を取得"""
        result = self._get_latest_result()
        return result.upper_threshold.copy() if result else None
    
    def get_lower_threshold(self) -> Optional[np.ndarray]:
        """下部閾値を取得"""
        result = self._get_latest_result()
        return result.lower_threshold.copy() if result else None
    
    def get_percentiles(self) -> Optional[np.ndarray]:
        """パーセンタイル値を取得"""
        result = self._get_latest_result()
        return result.percentiles.copy() if result and result.percentiles is not None else None
    
    def get_trend_state(self) -> Optional[np.ndarray]:
        """トレンド状態を取得"""
        result = self._get_latest_result()
        return result.trend_state.copy() if result and result.trend_state is not None else None
    
    def get_trend_intensity(self) -> Optional[np.ndarray]:
        """トレンド強度を取得"""
        result = self._get_latest_result()
        return result.trend_intensity.copy() if result and result.trend_intensity is not None else None
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'period': self.period,
            'delta': self.delta,
            'fraction': self.fraction,
            'avg_period': self.avg_period,
            'src_type': self.src_type,
            'use_smoothing': self.use_smoothing,
            'smoother_type': self.smoother_type if self.use_smoothing else None,
            'smoother_period': self.smoother_period if self.use_smoothing else None,
            'use_dynamic_period': self.use_dynamic_period,
            'detector_type': self.detector_type if self.use_dynamic_period else None,
            'use_kalman_filter': self.use_kalman_filter,
            'kalman_filter_type': self.kalman_filter_type if self.use_kalman_filter else None,
            'enable_percentile_analysis': self.enable_percentile_analysis,
            'percentile_lookback_period': self.percentile_lookback_period if self.enable_percentile_analysis else None,
            'percentile_low_threshold': self.percentile_low_threshold if self.enable_percentile_analysis else None,
            'percentile_high_threshold': self.percentile_high_threshold if self.enable_percentile_analysis else None,
            'description': 'Ehlers Empirical Mode Decomposition - 市場データをサイクル成分とトレンド成分に分解し、市場モードを判定'
        }
    
    def _get_latest_result(self) -> Optional[EMDResult]:
        """最新の結果を取得"""
        if not self._result_cache:
            return None
        
        if self._cache_keys:
            return self._result_cache[self._cache_keys[-1]]
        else:
            return next(iter(self._result_cache.values()))
    
    def reset(self) -> None:
        """インディケーターの状態をリセット"""
        super().reset()
        if self.smoother:
            self.smoother.reset()
        if self.cycle_detector:
            self.cycle_detector.reset()
        if self.kalman_filter:
            self.kalman_filter.reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_emd(
    data: Union[pd.DataFrame, np.ndarray],
    period: int = 20,
    delta: float = 0.1,
    fraction: float = 0.25,
    avg_period: int = 50,
    src_type: str = 'hl2',
    use_smoothing: bool = False,
    smoother_type: str = 'super_smoother',
    use_dynamic_period: bool = False,
    use_kalman_filter: bool = False,
    enable_percentile_analysis: bool = True,
    percentile_lookback_period: int = 50,
    percentile_low_threshold: float = 0.25,
    percentile_high_threshold: float = 0.75,
    **kwargs
) -> np.ndarray:
    """
    EMDの計算（便利関数）
    
    Args:
        data: 価格データ
        period: バンドパスフィルター周期
        delta: バンド幅パラメータ
        fraction: 閾値計算用係数
        avg_period: ピーク・バレー平均化期間
        src_type: ソースタイプ
        use_smoothing: 平滑化を使用するか
        smoother_type: スムーサータイプ
        use_dynamic_period: 動的期間適応を使用するか
        use_kalman_filter: カルマンフィルターを使用するか
        enable_percentile_analysis: パーセンタイル分析を有効にするか
        percentile_lookback_period: パーセンタイル分析のルックバック期間
        percentile_low_threshold: パーセンタイル分析の低閾値
        percentile_high_threshold: パーセンタイル分析の高閾値
        **kwargs: その他のパラメータ
        
    Returns:
        トレンド成分値
    """
    indicator = EMD(
        period=period,
        delta=delta,
        fraction=fraction,
        avg_period=avg_period,
        src_type=src_type,
        use_smoothing=use_smoothing,
        smoother_type=smoother_type,
        use_dynamic_period=use_dynamic_period,
        use_kalman_filter=use_kalman_filter,
        enable_percentile_analysis=enable_percentile_analysis,
        percentile_lookback_period=percentile_lookback_period,
        percentile_low_threshold=percentile_low_threshold,
        percentile_high_threshold=percentile_high_threshold,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.trend


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== EMD インジケーターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # 論文で例示されたようなデータを生成（トレンドとサイクルが混在）
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 強いトレンド相場
            trend_component = 0.005  # 上昇トレンド
            cycle_component = 0.02 * np.sin(2 * np.pi * i / 20)  # 20期間サイクル
            noise = np.random.normal(0, 0.008)
        elif i < 100:  # サイクル相場
            trend_component = 0.0  # トレンドなし
            cycle_component = 0.03 * np.sin(2 * np.pi * i / 25)  # 25期間サイクル
            noise = np.random.normal(0, 0.010)
        elif i < 150:  # 弱いトレンド相場
            trend_component = -0.002  # 下降トレンド
            cycle_component = 0.015 * np.sin(2 * np.pi * i / 18)  # 18期間サイクル
            noise = np.random.normal(0, 0.006)
        else:  # 複合相場
            trend_component = 0.003  # 上昇トレンド
            cycle_component = 0.025 * np.sin(2 * np.pi * i / 22)  # 22期間サイクル
            noise = np.random.normal(0, 0.009)
        
        change = trend_component + cycle_component + noise
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
    
    # EMDを計算（基本版）
    print("\\n基本版EMDをテスト中...")
    emd = EMD(period=20, delta=0.1, fraction=0.25, use_kalman_filter=False)
    result = emd.calculate(df)
    
    valid_bandpass_count = np.sum(~np.isnan(result.bandpass))
    valid_trend_count = np.sum(~np.isnan(result.trend))
    mean_trend = np.nanmean(result.trend)
    
    # モード信号の分析
    mode_counts = np.bincount((result.mode_signal[~np.isnan(result.mode_signal)] + 1).astype(int))
    total_valid_modes = np.sum(~np.isnan(result.mode_signal))
    
    print(f"  有効バンドパス値数: {valid_bandpass_count}/{len(df)}")
    print(f"  有効トレンド値数: {valid_trend_count}/{len(df)}")
    print(f"  平均トレンド値: {mean_trend:.6f}")
    
    if total_valid_modes > 0:
        downtrend_ratio = mode_counts[0] / total_valid_modes if len(mode_counts) > 0 else 0
        cycle_ratio = mode_counts[1] / total_valid_modes if len(mode_counts) > 1 else 0
        uptrend_ratio = mode_counts[2] / total_valid_modes if len(mode_counts) > 2 else 0
        
        print(f"  下降トレンド比率: {downtrend_ratio:.2%}")
        print(f"  サイクルモード比率: {cycle_ratio:.2%}")
        print(f"  上昇トレンド比率: {uptrend_ratio:.2%}")
    
    # カルマンフィルター + 平滑化版をテスト
    print("\\nカルマンフィルター + 平滑化版EMDをテスト中...")
    emd_advanced = EMD(
        period=20,
        delta=0.5,  # 帯域を広げてより敏感に
        fraction=0.25,
        use_kalman_filter=True,
        kalman_filter_type='unscented',
        use_smoothing=True,
        smoother_type='frama'
    )
    result_advanced = emd_advanced.calculate(df)
    
    valid_trend_advanced = np.sum(~np.isnan(result_advanced.trend))
    mean_trend_advanced = np.nanmean(result_advanced.trend)
    
    print(f"  有効トレンド値数: {valid_trend_advanced}/{len(df)}")
    print(f"  平均トレンド値（高機能版）: {mean_trend_advanced:.6f}")
    
    # 比較統計
    if valid_trend_count > 0 and valid_trend_advanced > 0:
        # 両方の結果が有効な範囲で相関を計算
        min_valid = min(valid_trend_count, valid_trend_advanced)
        basic_trend = result.trend[~np.isnan(result.trend)][-min_valid:]
        advanced_trend = result_advanced.trend[~np.isnan(result_advanced.trend)][-min_valid:]
        
        if len(basic_trend) > 0 and len(advanced_trend) > 0:
            correlation = np.corrcoef(basic_trend, advanced_trend)[0, 1]
            print(f"  基本版と高機能版の相関: {correlation:.4f}")
    
    print("\\n=== テスト完了 ===")