#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import jit, njit, float64

from .indicator import Indicator
from .price_source import PriceSource
from .smoother.unified_smoother import UnifiedSmoother
from .smoother.roofing_filter import RoofingFilter
from .utils.percentile_analysis import (
    calculate_percentile,
    calculate_trend_classification,
    PercentileAnalysisMixin
)

# 条件付きインポート（オプション機能）
try:
    from .cycle.ehlers_unified_dc import EhlersUnifiedDC
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


@dataclass
class HyperTrendIndexResult:
    """ハイパートレンドインデックスの計算結果"""
    values: np.ndarray                    # ハイパートレンドインデックス値（0-1の範囲、1に近いほど強いトレンド）
    raw_trend_index: np.ndarray          # フィルタリング前の生のトレンドインデックス値
    filtered_prices: np.ndarray          # カルマンフィルタリング後の価格（オプション）
    roofing_values: np.ndarray           # ルーフィングフィルター値（オプション）
    smoothed_values: np.ndarray          # 平滑化されたトレンドインデックス値（オプション）
    cycle_periods: np.ndarray            # サイクル期間値（動的期間使用時）
    # X Trend Index由来のデータ
    dominant_cycle: np.ndarray           # ドミナントサイクル値（チョピネス期間として使用）
    choppiness_index: np.ndarray         # チョピネス指数
    range_index: np.ndarray              # レンジ指数
    stddev_factor: np.ndarray            # 標準偏差係数
    tr: np.ndarray                       # True Range
    atr: np.ndarray                      # Average True Range
    # ミッドラインとトレンド判定
    midline: np.ndarray                  # ミッドライン値
    trend_signal: np.ndarray             # トレンド判定信号（1=トレンド、-1=レンジ）
    # パーセンタイルベースのトレンド分析
    percentiles: Optional[np.ndarray]     # パーセンタイル値
    trend_state: Optional[np.ndarray]     # トレンド状態（-1=レンジ、0=中、1=トレンド）
    trend_intensity: Optional[np.ndarray] # トレンド強度（0-1）


@njit(fastmath=True, cache=True)
def calculate_tr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    True Rangeを計算する（Numba最適化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        True Range の配列
    """
    length = len(high)
    tr = np.zeros(length, dtype=np.float64)
    
    # 最初の要素は単純なレンジ
    tr[0] = high[0] - low[0]
    
    # 2番目以降の要素はTRを計算
    for i in range(1, length):
        tr1 = high[i] - low[i]  # 当日の高値 - 当日の安値
        tr2 = abs(high[i] - close[i-1])  # |当日の高値 - 前日の終値|
        tr3 = abs(low[i] - close[i-1])  # |当日の安値 - 前日の終値|
        tr[i] = max(tr1, tr2, tr3)
    
    return tr


@njit(fastmath=True, cache=True)
def calculate_choppiness_index_numba(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray, 
    period: np.ndarray, 
    tr: np.ndarray
) -> np.ndarray:
    """
    動的期間によるチョピネス指数を計算する（Numba最適化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period: 動的な期間の配列
        tr: True Rangeの配列

    Returns:
        チョピネス指数の配列（0-100の範囲）
    """
    length = len(high)
    chop = np.zeros(length, dtype=np.float64)
    
    for i in range(1, length):
        curr_period = int(period[i])
        if curr_period < 2:
            curr_period = 2
        
        if i < curr_period:
            continue
        
        # True Range の合計
        tr_sum = 0.0
        for j in range(i - curr_period + 1, i + 1):
            tr_sum += tr[j]
        
        # 期間内の最高値と最安値を取得
        idx_start = i - curr_period + 1
        period_high = high[idx_start]
        period_low = low[idx_start]
        
        for j in range(idx_start + 1, i + 1):
            if high[j] > period_high:
                period_high = high[j]
            if low[j] < period_low:
                period_low = low[j]
        
        # 価格レンジの計算
        price_range = period_high - period_low
        
        # チョピネス指数の計算
        if price_range > 0 and curr_period > 1 and tr_sum > 0:
            log_period = np.log10(float(curr_period))
            chop_val = 100.0 * np.log10(tr_sum / price_range) / log_period
            # 値を0-100の範囲に制限
            chop[i] = max(0.0, min(100.0, chop_val))
        else:
            chop[i] = 0.0
    
    return chop


@njit(fastmath=True, cache=True)
def calculate_stddev_factor_numba(atr: np.ndarray, period: int = 14, lookback: int = 14) -> np.ndarray:
    """
    ATRの標準偏差係数を計算する（Numba最適化版）

    Args:
        atr: ATR配列
        period: 標準偏差計算期間
        lookback: 最小標準偏差のルックバック期間
    Returns:
        標準偏差係数
    """
    n = len(atr)
    stddev = np.zeros(n, dtype=np.float64)
    lowest_stddev = np.full(n, np.inf, dtype=np.float64)
    stddev_factor = np.ones(n, dtype=np.float64)

    for i in range(n):
        if i >= period - 1:
            start_idx = i - period + 1
            
            # 標準偏差計算
            mean_val = 0.0
            for j in range(start_idx, i + 1):
                mean_val += atr[j]
            mean_val /= period
            
            variance = 0.0
            for j in range(start_idx, i + 1):
                diff = atr[j] - mean_val
                variance += diff * diff
            variance /= period
            
            curr_stddev = np.sqrt(max(0.0, variance))
            stddev[i] = curr_stddev

            # 最小標準偏差の更新
            lowest_lookback_start = max(0, i - lookback + 1)
            min_std = np.inf
            for j in range(lowest_lookback_start, i + 1):
                if j < len(stddev) and np.isfinite(stddev[j]) and stddev[j] < min_std:
                    min_std = stddev[j]
            
            if min_std != np.inf:
                lowest_stddev[i] = min_std
            else:
                lowest_stddev[i] = curr_stddev if np.isfinite(curr_stddev) else 1.0

            # 標準偏差係数の計算
            if stddev[i] > 0 and np.isfinite(lowest_stddev[i]):
                stddev_factor[i] = lowest_stddev[i] / stddev[i]
            elif i > 0:
                stddev_factor[i] = stddev_factor[i-1]  # 前の値を使用
            else:
                stddev_factor[i] = 1.0  # 初期値

        elif i > 0:
            # データ不足の場合は前の値を使用
            stddev[i] = stddev[i-1]
            lowest_stddev[i] = lowest_stddev[i-1]
            stddev_factor[i] = stddev_factor[i-1]
        else:
            # 最初の要素はデフォルト値
            stddev[i] = np.nan
            lowest_stddev[i] = np.inf
            stddev_factor[i] = 1.0

    return stddev_factor


@njit(fastmath=True, cache=True)
def calculate_trend_index_numba(
    chop: np.ndarray,
    stddev_factor: np.ndarray
) -> np.ndarray:
    """
    トレンドインデックスを計算する（Numba最適化版）
    
    Args:
        chop: チョピネス指数の配列（0-100の範囲）
        stddev_factor: 標準偏差ファクターの配列
    
    Returns:
        トレンドインデックスの配列（0-1の範囲、1に近いほど強いトレンド）
    """
    # チョピネス指数と標準偏差係数を組み合わせたレンジインデックスを計算
    range_index = chop * stddev_factor
    
    # トレンド指数として常に反転し、0-1に正規化
    trend_index = 1.0 - (range_index / 100.0)
    
    # 値を0-1の範囲にクリップ
    trend_index = np.maximum(0.0, np.minimum(1.0, trend_index))
    
    return trend_index


@njit(fastmath=True, cache=True)
def calculate_midline_and_signal_trend(
    trend_values: np.ndarray,
    midline_period: int = 100
) -> tuple:
    """
    ミッドラインとトレンド信号を計算する（Numba最適化版）
    
    Args:
        trend_values: トレンド値の配列
        midline_period: ミッドライン計算期間
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ミッドライン, トレンド信号)
    """
    length = len(trend_values)
    midline = np.full(length, np.nan, dtype=np.float64)
    trend_signal = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(midline_period - 1, length):
        # 期間内の最高値と最安値を計算
        period_max = -np.inf
        period_min = np.inf
        valid_count = 0
        
        for j in range(i - midline_period + 1, i + 1):
            if not np.isnan(trend_values[j]):
                if trend_values[j] > period_max:
                    period_max = trend_values[j]
                if trend_values[j] < period_min:
                    period_min = trend_values[j]
                valid_count += 1
        
        if valid_count >= midline_period // 2:
            # ミッドライン = (最高値 + 最安値) / 2
            midline[i] = (period_max + period_min) / 2.0
            
            # トレンド信号の判定
            if not np.isnan(trend_values[i]):
                if trend_values[i] > midline[i]:
                    trend_signal[i] = 1.0   # 強いトレンド相場
                else:
                    trend_signal[i] = -1.0  # レンジ相場
    
    return midline, trend_signal


class HyperTrendIndex(Indicator, PercentileAnalysisMixin):
    """
    ハイパートレンドインデックス（Hyper Trend Index）インジケーター
    
    X Trend Indexをベースに、以下のフィルタリング処理を追加：
    
    計算フロー:
    1. ソース価格→統合カルマンフィルターによるフィルター処理(オプション)
    2. サイクル検出器による期間検出
    3. ルーフィングフィルターによるフィルター処理(オプション)
    4. フィルター処理済価格によるトレンドインデックス計算
    5. 平滑化フィルターによる処理(オプション)
    
    特徴:
    - 統合カルマンフィルターによる価格ノイズ除去
    - エラーズ統合サイクル検出器による動的期間適応
    - ルーフィングフィルターによる追加フィルタリング
    - 統合スムーサーによる最終結果の平滑化
    - 0-1の値範囲でトレンド強度を表現
    - パーセンタイルベースのトレンド分析機能
    """
    
    def __init__(
        self,
        # 基本パラメータ
        period: int = 55,
        midline_period: int = 100,
        src_type: str = 'hlc3',
        # 統合カルマンフィルターパラメータ
        use_kalman_filter: bool = False,
        kalman_filter_type: str = 'simple',
        kalman_process_noise: float = 1e-5,
        kalman_min_observation_noise: float = 1e-6,
        kalman_adaptation_window: int = 5,
        # エラーズ統合サイクル検出器パラメータ
        use_dynamic_period: bool = True,
        detector_type: str = 'dft_dominant',
        lp_period: int = 12,
        hp_period: int = 124,
        cycle_part: float = 0.4,
        max_cycle: int = 124,
        min_cycle: int = 12,
        max_output: int = 89,
        min_output: int = 5,
        # ルーフィングフィルターパラメータ
        use_roofing_filter: bool = True,
        roofing_hp_cutoff: float = 55.0,
        roofing_ss_band_edge: float = 10.0,
        # 平滑化オプション
        use_smoothing: bool = True,
        smoother_type: str = 'laguerre',
        smoother_period: int = 12,
        smoother_src_type: str = 'close',
        # 標準偏差係数パラメータ
        stddev_period: int = 14,
        stddev_lookback: int = 14,
        # 固定しきい値パラメータ
        fixed_threshold: float = 0.65,
        # パーセンタイルベーストレンド分析パラメータ
        enable_percentile_analysis: bool = False,
        percentile_lookback_period: int = 50,
        percentile_low_threshold: float = 0.25,
        percentile_high_threshold: float = 0.75
    ):
        """
        コンストラクタ
        
        Args:
            period: トレンドインデックス計算期間（デフォルト: 14）
            midline_period: ミッドライン計算期間（デフォルト: 100）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: True）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'adaptive'）
            kalman_process_noise: カルマンフィルターのプロセスノイズ（デフォルト: 1e-5）
            kalman_min_observation_noise: 最小観測ノイズ（デフォルト: 1e-6）
            kalman_adaptation_window: 適応ウィンドウ（デフォルト: 5）
            use_dynamic_period: 動的期間適応を使用するか（デフォルト: True）
            detector_type: サイクル検出器タイプ（デフォルト: 'hody_e'）
            lp_period: ローパスフィルター期間（デフォルト: 13）
            hp_period: ハイパスフィルター期間（デフォルト: 124）
            cycle_part: サイクル部分（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 124）
            min_cycle: 最小サイクル期間（デフォルト: 13）
            max_output: 最大出力値（デフォルト: 89）
            min_output: 最小出力値（デフォルト: 5）
            use_roofing_filter: ルーフィングフィルターを使用するか（デフォルト: True）
            roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ（デフォルト: 48.0）
            roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ（デフォルト: 10.0）
            use_smoothing: 平滑化を使用するか（デフォルト: True）
            smoother_type: 統合スムーサータイプ（デフォルト: 'frama'）
            smoother_period: スムーサー期間（デフォルト: 12）
            smoother_src_type: スムーサーソースタイプ（デフォルト: 'close'）
            stddev_period: 標準偏差計算期間（デフォルト: 14）
            stddev_lookback: 標準偏差ルックバック期間（デフォルト: 14）
            fixed_threshold: 固定しきい値（デフォルト: 0.65）
            enable_percentile_analysis: パーセンタイル分析を有効にするか（デフォルト: True）
            percentile_lookback_period: パーセンタイル分析のルックバック期間（デフォルト: 50）
            percentile_low_threshold: パーセンタイル分析の低閾値（デフォルト: 0.25）
            percentile_high_threshold: パーセンタイル分析の高閾値（デフォルト: 0.75）
        """
        indicator_name = f"HyperTrendIndex({period}, midline={midline_period}, src={src_type}"
        if use_kalman_filter:
            indicator_name += f", kalman={kalman_filter_type}"
        if use_dynamic_period:
            indicator_name += f", dynamic={detector_type}"
        if use_roofing_filter:
            indicator_name += f", roofing(hp={roofing_hp_cutoff}, ss={roofing_ss_band_edge})"
        if use_smoothing:
            indicator_name += f", smooth={smoother_type}({smoother_period})"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        
        # パラメータ保存
        self.period = period
        self.midline_period = midline_period
        self.src_type = src_type
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_min_observation_noise = kalman_min_observation_noise
        self.kalman_adaptation_window = kalman_adaptation_window
        self.use_roofing_filter = use_roofing_filter
        self.roofing_hp_cutoff = roofing_hp_cutoff
        self.roofing_ss_band_edge = roofing_ss_band_edge
        self.use_smoothing = use_smoothing
        self.smoother_type = smoother_type
        self.smoother_period = smoother_period
        self.smoother_src_type = smoother_src_type
        self.stddev_period = stddev_period
        self.stddev_lookback = stddev_lookback
        self.fixed_threshold = fixed_threshold
        
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
        if self.midline_period <= 0:
            raise ValueError("midline_periodは0より大きい必要があります")
        if self.use_dynamic_period and self.max_cycle <= self.min_cycle:
            raise ValueError("max_cycleはmin_cycleより大きい必要があります")
        if self.use_roofing_filter and self.roofing_ss_band_edge >= self.roofing_hp_cutoff:
            raise ValueError("roofing_ss_band_edgeはroofing_hp_cutoffより小さい必要があります")
        
        # 統合カルマンフィルターの初期化（価格フィルタリングが有効な場合）
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
                        process_noise=self.kalman_process_noise,
                        min_observation_noise=self.kalman_min_observation_noise,
                        adaptation_window=self.kalman_adaptation_window
                    )
                    self.logger.info(f"統合カルマンフィルターを初期化しました: {self.kalman_filter_type}")
                except Exception as e:
                    self.logger.error(f"統合カルマンフィルターの初期化に失敗: {e}")
                    self.use_kalman_filter = False
                    self.logger.warning("カルマンフィルター機能を無効にしました")
        
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
                        src_type=self.src_type,
                        use_kalman_filter=False,
                        lp_period=self.lp_period,
                        hp_period=self.hp_period
                    )
                    self.logger.info(f"エラーズ統合サイクル検出器を初期化しました: {self.detector_type}")
                except Exception as e:
                    self.logger.error(f"エラーズ統合サイクル検出器の初期化に失敗: {e}")
                    self.use_dynamic_period = False
                    self.logger.warning("動的期間適応機能を無効にしました")
        
        # ルーフィングフィルターの初期化（ルーフィングフィルターが有効な場合）
        self.roofing_filter = None
        if self.use_roofing_filter:
            try:
                self.roofing_filter = RoofingFilter(
                    src_type=self.src_type,
                    hp_cutoff=self.roofing_hp_cutoff,
                    ss_band_edge=self.roofing_ss_band_edge
                )
                self.logger.info(f"ルーフィングフィルターを初期化しました: hp={self.roofing_hp_cutoff}, ss={self.roofing_ss_band_edge}")
            except Exception as e:
                self.logger.error(f"ルーフィングフィルターの初期化に失敗: {e}")
                self.use_roofing_filter = False
                self.logger.warning("ルーフィングフィルター機能を無効にしました")
        
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
            param_str = (f"{self.period}_{self.midline_period}_{self.src_type}_"
                        f"{self.use_kalman_filter}_{self.kalman_filter_type}_"
                        f"{self.use_dynamic_period}_{self.detector_type}_"
                        f"{self.use_roofing_filter}_{self.roofing_hp_cutoff}_"
                        f"{self.roofing_ss_band_edge}_{self.use_smoothing}_{self.smoother_type}_"
                        f"{self.smoother_period}_{self.smoother_src_type}")
            
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(param_str)}"
            
        except Exception:
            return f"{id(data)}_{self.period}_{self.midline_period}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperTrendIndexResult:
        """
        ハイパートレンドインデックスを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                必要なカラム: high, low, close, open（ルーフィングフィルター用）
        
        Returns:
            HyperTrendIndexResult: ハイパートレンドインデックスの計算結果
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
                return HyperTrendIndexResult(
                    values=cached_result.values.copy(),
                    raw_trend_index=cached_result.raw_trend_index.copy(),
                    filtered_prices=cached_result.filtered_prices.copy(),
                    roofing_values=cached_result.roofing_values.copy(),
                    smoothed_values=cached_result.smoothed_values.copy(),
                    cycle_periods=cached_result.cycle_periods.copy(),
                    dominant_cycle=cached_result.dominant_cycle.copy(),
                    choppiness_index=cached_result.choppiness_index.copy(),
                    range_index=cached_result.range_index.copy(),
                    stddev_factor=cached_result.stddev_factor.copy(),
                    tr=cached_result.tr.copy(),
                    atr=cached_result.atr.copy(),
                    midline=cached_result.midline.copy(),
                    trend_signal=cached_result.trend_signal.copy(),
                    percentiles=cached_result.percentiles.copy() if cached_result.percentiles is not None else None,
                    trend_state=cached_result.trend_state.copy() if cached_result.trend_state is not None else None,
                    trend_intensity=cached_result.trend_intensity.copy() if cached_result.trend_intensity is not None else None
                )
            
            # データの準備と検証
            if isinstance(data, pd.DataFrame):
                required_cols = ['high', 'low', 'close']
                if self.use_roofing_filter:
                    required_cols.extend(['open'])  # ルーフィングフィルター用
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"必要なカラムが不足しています: {missing_cols}")
                
                h = np.asarray(data['high'].values, dtype=np.float64)
                l = np.asarray(data['low'].values, dtype=np.float64)
                c = np.asarray(data['close'].values, dtype=np.float64)
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                h = np.asarray(data[:, 1], dtype=np.float64)
                l = np.asarray(data[:, 2], dtype=np.float64)
                c = np.asarray(data[:, 3], dtype=np.float64)
            
            # データ長の検証
            data_length = len(data)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < max(self.period, self.midline_period):
                self.logger.warning(f"データ長（{data_length}）が必要な期間（{max(self.period, self.midline_period)}）より短いです")
            
            # 1. ソース価格データを取得
            source_prices = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換
            if not isinstance(source_prices, np.ndarray):
                source_prices = np.array(source_prices)
            if source_prices.dtype != np.float64:
                source_prices = source_prices.astype(np.float64)
            
            # 2. 統合カルマンフィルターによる価格フィルタリング（オプション）
            filtered_prices = source_prices.copy()
            if self.use_kalman_filter and self.kalman_filter is not None:
                try:
                    kalman_result = self.kalman_filter.calculate(data)
                    kalman_values = kalman_result.values
                    
                    # 有効な値が十分にある場合のみ使用
                    valid_kalman = np.sum(~np.isnan(kalman_values))
                    if valid_kalman > len(kalman_values) * 0.5:  # 有効値が50%以上の場合
                        filtered_prices = kalman_values.copy()
                        self.logger.debug("統合カルマンフィルターを適用しました")
                    else:
                        self.logger.debug("統合カルマンフィルターの有効値が不十分。元の価格を使用します")
                        
                except Exception as e:
                    self.logger.warning(f"統合カルマンフィルター適用中にエラー: {e}。元の値を使用します。")
                    filtered_prices = source_prices.copy()
            
            # 3. 動的期間の計算（オプション）
            dynamic_periods = None
            cycle_periods = np.full(data_length, self.period, dtype=np.float64)
            
            if self.use_dynamic_period and self.cycle_detector is not None:
                try:
                    cycle_values = self.cycle_detector.calculate(data)
                    
                    # サイクル値から期間を計算
                    valid_cycles = cycle_values[~np.isnan(cycle_values) & (cycle_values > 0)]
                    if len(valid_cycles) > 0:
                        # サイクル値を期間に変換（例：サイクル値の半分を期間とする）
                        dynamic_periods = np.where(
                            ~np.isnan(cycle_values) & (cycle_values > 0),
                            np.clip(cycle_values * 0.5, 5, 50),
                            self.period
                        )
                        cycle_periods = cycle_values.copy()
                    else:
                        dynamic_periods = np.full(len(data), self.period)
                        
                    self.logger.debug(f"動的期間を計算しました。範囲: {np.min(dynamic_periods):.1f} - {np.max(dynamic_periods):.1f}")
                        
                except Exception as e:
                    self.logger.warning(f"動的期間計算中にエラー: {e}。固定期間を使用します。")
                    dynamic_periods = None
            
            # ドミナントサイクル値（チョピネス期間として使用）  
            dominant_cycle = cycle_periods if dynamic_periods is not None else np.full(data_length, self.period)
            
            # 4. ルーフィングフィルターによる追加フィルタリング（オプション）
            roofing_values = np.full_like(filtered_prices, np.nan)
            if self.use_roofing_filter and self.roofing_filter is not None:
                try:
                    roofing_result = self.roofing_filter.calculate(data)
                    roofing_values = roofing_result.values
                    
                    # ルーフィングフィルターの結果を使用
                    # ルーフィングフィルターは価格の振動成分を抽出するので、直接フィルタリング済み価格として使用
                    # NaN値が多い場合は元の価格を使用
                    valid_roofing = np.sum(~np.isnan(roofing_values))
                    if valid_roofing > len(roofing_values) * 0.5:  # 有効値が50%以上の場合
                        # ルーフィングフィルター値をソース価格のスケールに調整
                        roofing_range = np.nanmax(roofing_values) - np.nanmin(roofing_values)
                        price_range = np.nanmax(filtered_prices) - np.nanmin(filtered_prices)
                        if roofing_range > 0 and price_range > 0:
                            scale_factor = price_range / roofing_range * 0.1  # 10%の影響度
                            filtered_prices = filtered_prices + roofing_values * scale_factor
                        else:
                            filtered_prices = filtered_prices
                        
                        self.logger.debug("ルーフィングフィルターによる追加フィルタリングを適用しました")
                    else:
                        self.logger.debug("ルーフィングフィルターの有効値が不十分。元の価格を使用します")
                        
                except Exception as e:
                    self.logger.warning(f"ルーフィングフィルター適用中にエラー: {e}。元の値を使用します。")
                    roofing_values = np.full_like(filtered_prices, np.nan)
            
            # NumPy配列として確保
            if not isinstance(filtered_prices, np.ndarray):
                filtered_prices = np.array(filtered_prices)
            if filtered_prices.dtype != np.float64:
                filtered_prices = filtered_prices.astype(np.float64)
            
            # 5. True Rangeの計算
            tr = calculate_tr_numba(h, l, c)
            
            # 6. チョピネス指数の計算
            chop_period = np.maximum(2, dominant_cycle).astype(np.int32)
            choppiness_index = calculate_choppiness_index_numba(h, l, c, chop_period, tr)
            
            # 7. 適応型ATRの計算（フィルタリング済み価格を使用）
            # 簡易版ATRをフィルタリング済み価格から計算
            atr = np.zeros_like(filtered_prices)
            for i in range(1, len(atr)):
                period = int(dominant_cycle[i]) if dynamic_periods is not None else self.period
                period = max(2, min(period, i + 1))
                
                if i >= period:
                    start_idx = i - period + 1
                    price_changes = np.abs(np.diff(filtered_prices[start_idx:i+1]))
                    atr[i] = np.mean(price_changes) if len(price_changes) > 0 else 0.0
                else:
                    atr[i] = atr[i-1] if i > 0 else 0.0
            
            # 8. 標準偏差係数の計算
            stddev_factor = calculate_stddev_factor_numba(atr, self.stddev_period, self.stddev_lookback)
            
            # 9. トレンドインデックスの計算
            raw_trend_index = calculate_trend_index_numba(choppiness_index, stddev_factor)
            
            # 10. 平滑化（オプション）
            smoothed_values = raw_trend_index.copy()  # デフォルトで生の値を使用
            if self.use_smoothing and self.smoother is not None:
                try:
                    # NaN値の処理 - スムーサーが正常に動作するようにNaN値を前方補完
                    clean_trend_index = raw_trend_index.copy()
                    nan_mask = np.isnan(clean_trend_index)
                    
                    if np.any(nan_mask):
                        # 最初の有効値を見つけて前方補完
                        first_valid_idx = np.where(~nan_mask)[0]
                        if len(first_valid_idx) > 0:
                            first_valid = first_valid_idx[0]
                            first_value = clean_trend_index[first_valid]
                            # 最初の有効値より前をその値で補完
                            clean_trend_index[:first_valid] = first_value
                            
                            # 残りのNaN値は前方補完
                            for i in range(len(clean_trend_index)):
                                if np.isnan(clean_trend_index[i]) and i > 0:
                                    clean_trend_index[i] = clean_trend_index[i-1]
                    
                    # トレンドインデックス値をDataFrame形式に変換
                    if isinstance(data, pd.DataFrame):
                        # 元のデータがDataFrameの場合、必要なカラムを保持しつつトレンドインデックス値を使用
                        trend_df = data.copy()
                        trend_df['close'] = clean_trend_index
                        # インデックスの長さを調整
                        if len(trend_df) != len(clean_trend_index):
                            trend_df = trend_df.iloc[:len(clean_trend_index)].copy()
                            trend_df['close'] = clean_trend_index
                    else:
                        # NumPy配列の場合は基本的なDataFrameを作成
                        trend_df = pd.DataFrame({'close': clean_trend_index})
                        # high, lowカラムもcloseと同じ値で作成（FRAMAなど用）
                        trend_df['high'] = clean_trend_index
                        trend_df['low'] = clean_trend_index
                        trend_df['open'] = clean_trend_index
                    
                    # 平滑化を適用
                    smoother_result = self.smoother.calculate(trend_df)
                    if smoother_result is not None and hasattr(smoother_result, 'values'):
                        smooth_values = smoother_result.values
                        # 有効な平滑化結果がある場合のみ使用
                        if np.sum(~np.isnan(smooth_values)) > 0:
                            # 元のNaN位置を復元
                            if np.any(nan_mask):
                                smooth_values[nan_mask] = np.nan
                            smoothed_values = smooth_values
                            self.logger.debug(f"平滑化処理完了: 有効値数 {np.sum(~np.isnan(smooth_values))}")
                        else:
                            self.logger.warning("平滑化結果がすべてNaN。生の値を使用します。")
                    else:
                        self.logger.warning("平滑化結果が無効。生の値を使用します。")
                except Exception as e:
                    self.logger.warning(f"平滑化処理中にエラー: {e}。生の値を使用します。")
            
            # 最終的なトレンドインデックス値
            final_trend_values = smoothed_values if self.use_smoothing else raw_trend_index
            
            # 11. ミッドラインとトレンド信号を計算
            midline, trend_signal = calculate_midline_and_signal_trend(
                final_trend_values, self.midline_period
            )
            
            # 12. パーセンタイルベーストレンド分析（オプション）
            percentiles, trend_state, trend_intensity = self._calculate_percentile_analysis(
                final_trend_values, 'trend'
            )
            
            # レンジインデックスの計算
            range_index = 100.0 - choppiness_index
            
            # 結果の作成
            result = HyperTrendIndexResult(
                values=final_trend_values.copy(),
                raw_trend_index=raw_trend_index.copy(),
                filtered_prices=filtered_prices.copy(),
                roofing_values=roofing_values.copy(),
                smoothed_values=smoothed_values.copy(),
                cycle_periods=cycle_periods.copy(),
                dominant_cycle=dominant_cycle.copy(),
                choppiness_index=choppiness_index.copy(),
                range_index=range_index.copy(),
                stddev_factor=stddev_factor.copy(),
                tr=tr.copy(),
                atr=atr.copy(),
                midline=midline.copy(),
                trend_signal=trend_signal.copy(),
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
            
            # 基底クラス用の値設定
            self._values = final_trend_values
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ハイパートレンドインデックス計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            empty_array = np.array([])
            return HyperTrendIndexResult(
                values=empty_array,
                raw_trend_index=empty_array,
                filtered_prices=empty_array,
                roofing_values=empty_array,
                smoothed_values=empty_array,
                cycle_periods=empty_array,
                dominant_cycle=empty_array,
                choppiness_index=empty_array,
                range_index=empty_array,
                stddev_factor=empty_array,
                tr=empty_array,
                atr=empty_array,
                midline=empty_array,
                trend_signal=empty_array,
                percentiles=None,
                trend_state=None,
                trend_intensity=None
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """ハイパートレンドインデックス値を取得（後方互換性のため）"""
        if not self._result_cache:
            return None
        
        result = self._get_latest_result()
        return result.values.copy() if result else None
    
    def get_raw_trend_index(self) -> Optional[np.ndarray]:
        """生のトレンドインデックス値を取得"""
        result = self._get_latest_result()
        return result.raw_trend_index.copy() if result else None
    
    def get_filtered_prices(self) -> Optional[np.ndarray]:
        """フィルタリングされた価格を取得"""
        result = self._get_latest_result()
        return result.filtered_prices.copy() if result else None
    
    def get_roofing_values(self) -> Optional[np.ndarray]:
        """ルーフィングフィルター値を取得"""
        result = self._get_latest_result()
        return result.roofing_values.copy() if result else None
    
    def get_smoothed_values(self) -> Optional[np.ndarray]:
        """平滑化された値を取得"""
        result = self._get_latest_result()
        return result.smoothed_values.copy() if result else None
    
    def get_cycle_periods(self) -> Optional[np.ndarray]:
        """サイクル期間値を取得"""
        result = self._get_latest_result()
        return result.cycle_periods.copy() if result else None
    
    def get_dominant_cycle(self) -> Optional[np.ndarray]:
        """ドミナントサイクル値を取得"""
        result = self._get_latest_result()
        return result.dominant_cycle.copy() if result else None
    
    def get_choppiness_index(self) -> Optional[np.ndarray]:
        """チョピネス指数を取得"""
        result = self._get_latest_result()
        return result.choppiness_index.copy() if result else None
    
    def get_range_index(self) -> Optional[np.ndarray]:
        """レンジ指数を取得"""
        result = self._get_latest_result()
        return result.range_index.copy() if result else None
    
    def get_stddev_factor(self) -> Optional[np.ndarray]:
        """標準偏差係数を取得"""
        result = self._get_latest_result()
        return result.stddev_factor.copy() if result else None
    
    def get_tr(self) -> Optional[np.ndarray]:
        """True Range値を取得"""
        result = self._get_latest_result()
        return result.tr.copy() if result else None
    
    def get_atr(self) -> Optional[np.ndarray]:
        """ATR値を取得"""
        result = self._get_latest_result()
        return result.atr.copy() if result else None
    
    def get_midline(self) -> Optional[np.ndarray]:
        """ミッドライン値を取得"""
        result = self._get_latest_result()
        return result.midline.copy() if result else None
    
    def get_trend_signal(self) -> Optional[np.ndarray]:
        """トレンド信号を取得"""
        result = self._get_latest_result()
        return result.trend_signal.copy() if result else None
    
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
            'midline_period': self.midline_period,
            'src_type': self.src_type,
            'use_kalman_filter': self.use_kalman_filter,
            'kalman_filter_type': self.kalman_filter_type if self.use_kalman_filter else None,
            'use_dynamic_period': self.use_dynamic_period,
            'detector_type': self.detector_type if self.use_dynamic_period else None,
            'use_roofing_filter': self.use_roofing_filter,
            'roofing_hp_cutoff': self.roofing_hp_cutoff if self.use_roofing_filter else None,
            'roofing_ss_band_edge': self.roofing_ss_band_edge if self.use_roofing_filter else None,
            'use_smoothing': self.use_smoothing,
            'smoother_type': self.smoother_type if self.use_smoothing else None,
            'smoother_period': self.smoother_period if self.use_smoothing else None,
            'stddev_period': self.stddev_period,
            'stddev_lookback': self.stddev_lookback,
            'fixed_threshold': self.fixed_threshold,
            'enable_percentile_analysis': self.enable_percentile_analysis,
            'description': 'X Trend Indexベースの高度なトレンド指標（カルマンフィルター・動的期間・ルーフィング・平滑化・パーセンタイル分析対応）'
        }
    
    def _get_latest_result(self) -> Optional[HyperTrendIndexResult]:
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
        if self.roofing_filter:
            self.roofing_filter.reset()
        if self.kalman_filter:
            self.kalman_filter.reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_hyper_trend_index(
    data: Union[pd.DataFrame, np.ndarray],
    period: int = 14,
    midline_period: int = 100,
    src_type: str = 'hlc3',
    use_kalman_filter: bool = True,
    kalman_filter_type: str = 'adaptive',
    use_dynamic_period: bool = True,
    detector_type: str = 'hody_e',
    use_roofing_filter: bool = True,
    roofing_hp_cutoff: float = 48.0,
    roofing_ss_band_edge: float = 10.0,
    use_smoothing: bool = True,
    smoother_type: str = 'frama',
    smoother_period: int = 12,
    enable_percentile_analysis: bool = True,
    percentile_lookback_period: int = 50,
    percentile_low_threshold: float = 0.25,
    percentile_high_threshold: float = 0.75,
    **kwargs
) -> np.ndarray:
    """
    ハイパートレンドインデックスの計算（便利関数）
    
    Args:
        data: 価格データ
        period: トレンドインデックス計算期間
        midline_period: ミッドライン計算期間
        src_type: ソースタイプ
        use_kalman_filter: カルマンフィルターを使用するか
        kalman_filter_type: カルマンフィルタータイプ
        use_dynamic_period: 動的期間適応を使用するか
        detector_type: サイクル検出器タイプ
        use_roofing_filter: ルーフィングフィルターを使用するか
        roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ
        roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ
        use_smoothing: 平滑化を使用するか
        smoother_type: スムーサータイプ
        smoother_period: スムーサー期間
        enable_percentile_analysis: パーセンタイル分析を有効にするか
        percentile_lookback_period: パーセンタイル分析のルックバック期間
        percentile_low_threshold: パーセンタイル分析の低閾値
        percentile_high_threshold: パーセンタイル分析の高閾値
        **kwargs: その他のパラメータ
        
    Returns:
        ハイパートレンドインデックス値
    """
    indicator = HyperTrendIndex(
        period=period,
        midline_period=midline_period,
        src_type=src_type,
        use_kalman_filter=use_kalman_filter,
        kalman_filter_type=kalman_filter_type,
        use_dynamic_period=use_dynamic_period,
        detector_type=detector_type,
        use_roofing_filter=use_roofing_filter,
        roofing_hp_cutoff=roofing_hp_cutoff,
        roofing_ss_band_edge=roofing_ss_band_edge,
        use_smoothing=use_smoothing,
        smoother_type=smoother_type,
        smoother_period=smoother_period,
        enable_percentile_analysis=enable_percentile_analysis,
        percentile_lookback_period=percentile_lookback_period,
        percentile_low_threshold=percentile_low_threshold,
        percentile_high_threshold=percentile_high_threshold,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== ハイパートレンドインデックス インジケーターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # トレンドとレンジが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 強いトレンド相場
            change = 0.005 + np.random.normal(0, 0.008)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.015)
        elif i < 150:  # 非常に強いトレンド相場
            change = 0.008 + np.random.normal(0, 0.005)
        else:  # 弱いトレンド相場
            change = 0.002 + np.random.normal(0, 0.012)
        
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
    
    # ハイパートレンドインデックスを計算（基本版）
    print("\n基本版ハイパートレンドインデックスをテスト中...")
    hyper_trend = HyperTrendIndex(
        period=14, 
        midline_period=50, 
        use_kalman_filter=False,
        use_dynamic_period=False,
        use_roofing_filter=False,
        use_smoothing=False
    )
    result = hyper_trend.calculate(df)
    
    valid_count = np.sum(~np.isnan(result.values))
    mean_trend = np.nanmean(result.values)
    trend_ratio = np.sum(result.trend_signal == 1) / np.sum(~np.isnan(result.trend_signal))
    
    print(f"  有効値数: {valid_count}/{len(df)}")
    print(f"  平均トレンドインデックス: {mean_trend:.4f}")
    print(f"  強いトレンド信号比率: {trend_ratio:.2%}")
    
    # フル機能版をテスト
    print("\nフル機能版ハイパートレンドインデックスをテスト中...")
    hyper_trend_full = HyperTrendIndex(
        period=14,
        midline_period=50,
        use_kalman_filter=True,
        kalman_filter_type='adaptive',
        use_dynamic_period=True,
        detector_type='hody_e',
        use_roofing_filter=True,
        roofing_hp_cutoff=48.0,
        roofing_ss_band_edge=10.0,
        use_smoothing=True,
        smoother_type='frama',
        smoother_period=12,
        enable_percentile_analysis=True
    )
    
    try:
        result_full = hyper_trend_full.calculate(df)
        
        valid_count_full = np.sum(~np.isnan(result_full.values))
        mean_trend_full = np.nanmean(result_full.values)
        
        print(f"  有効値数: {valid_count_full}/{len(df)}")
        print(f"  平均トレンドインデックス（フル機能）: {mean_trend_full:.4f}")
        
        # 各コンポーネントの統計
        if valid_count_full > 0:
            print(f"  チョピネス指数平均: {np.nanmean(result_full.choppiness_index):.2f}")
            print(f"  標準偏差係数平均: {np.nanmean(result_full.stddev_factor):.4f}")
            print(f"  フィルタリング済み価格レンジ: {np.nanmin(result_full.filtered_prices):.2f} - {np.nanmax(result_full.filtered_prices):.2f}")
            
            if result_full.percentiles is not None:
                print(f"  パーセンタイル値レンジ: {np.nanmin(result_full.percentiles):.4f} - {np.nanmax(result_full.percentiles):.4f}")
        
        # 比較統計
        if valid_count > 0 and valid_count_full > 0:
            common_length = min(valid_count, valid_count_full)
            basic_values = result.values[~np.isnan(result.values)][-common_length:]
            full_values = result_full.values[~np.isnan(result_full.values)][-common_length:]
            
            if len(basic_values) > 0 and len(full_values) > 0:
                correlation = np.corrcoef(basic_values, full_values)[0, 1]
                print(f"  基本版とフル機能版の相関: {correlation:.4f}")
        
    except Exception as e:
        print(f"  フル機能版でエラー: {e}")
        print("  一部の依存関係が不足している可能性があります")
    
    # 便利関数をテスト
    print("\n便利関数をテスト中...")
    try:
        simple_result = calculate_hyper_trend_index(
            df, 
            period=14, 
            use_kalman_filter=False,
            use_dynamic_period=False,
            use_roofing_filter=False,
            use_smoothing=False
        )
        
        valid_simple = np.sum(~np.isnan(simple_result))
        mean_simple = np.nanmean(simple_result)
        
        print(f"  便利関数結果: 有効値数 {valid_simple}/{len(df)}, 平均値 {mean_simple:.4f}")
        
    except Exception as e:
        print(f"  便利関数でエラー: {e}")
    
    print("\n=== テスト完了 ===")