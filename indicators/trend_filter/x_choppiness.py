#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import jit, njit, float64

from ..indicator import Indicator
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
class XChoppinessResult:
    """Xチョピネスの計算結果"""
    values: np.ndarray               # Xチョピネス値（0-1の範囲、高い=トレンド）
    raw_choppiness: np.ndarray       # 生のチョピネス値（反転前）
    smoothed_choppiness: np.ndarray  # 平滑化されたチョピネス値（オプション）
    midline: np.ndarray              # ミッドライン値
    trend_signal: np.ndarray         # トレンド判定信号（1=トレンド、-1=レンジ）
    str_values: np.ndarray           # STR値
    # パーセンタイルベースのトレンド分析
    percentiles: Optional[np.ndarray]     # パーセンタイル値
    trend_state: Optional[np.ndarray]     # トレンド状態（-1=レンジ、0=中、1=トレンド）
    trend_intensity: Optional[np.ndarray] # トレンド強度（0-1）


@njit(fastmath=True, cache=True)
def calculate_true_range_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> tuple:
    """
    True Range値を計算する（Numba最適化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        tuple: (True High, True Low, True Range)
    """
    length = len(close)
    true_high = np.zeros(length, dtype=np.float64)
    true_low = np.zeros(length, dtype=np.float64)
    true_range = np.zeros(length, dtype=np.float64)
    
    # 最初の値は現在の高値/安値を使用
    true_high[0] = high[0]
    true_low[0] = low[0]
    true_range[0] = high[0] - low[0]
    
    for i in range(1, length):
        # True High = Close[1] > High ? Close[1] : High
        if close[i-1] > high[i]:
            true_high[i] = close[i-1]
        else:
            true_high[i] = high[i]
        
        # True Low = Close[1] < Low ? Close[1] : Low
        if close[i-1] < low[i]:
            true_low[i] = close[i-1]
        else:
            true_low[i] = low[i]
        
        # True Range = True High - True Low
        true_range[i] = true_high[i] - true_low[i]
    
    return true_high, true_low, true_range


@njit(fastmath=True, cache=True)
def calculate_str_numba(
    true_range: np.ndarray,
    period: int,
    dynamic_periods: np.ndarray = None
) -> np.ndarray:
    """
    STR（Smooth True Range）を計算する（Numba最適化版）
    
    Args:
        true_range: True Range値の配列
        period: STR期間
        dynamic_periods: 動的期間配列（オプション）
    
    Returns:
        STR値の配列
    """
    length = len(true_range)
    str_values = np.full(length, np.nan, dtype=np.float64)
    
    # 初期値（最初のTrue Range値）
    if length > 0 and not np.isnan(true_range[0]):
        str_values[0] = true_range[0]
    
    for i in range(1, length):
        if np.isnan(true_range[i]):
            continue
            
        # 動的期間または固定期間を使用
        current_period = period
        if dynamic_periods is not None and i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
            current_period = max(2, min(int(dynamic_periods[i]), 100))  # 2-100期間に制限
        
        # Ultimate Smootherのアルゴリズムに基づくSTR計算
        # α = 2 / (period + 1)
        alpha = 2.0 / (current_period + 1.0)
        
        if not np.isnan(str_values[i-1]):
            # EMA風の平滑化
            str_values[i] = alpha * true_range[i] + (1.0 - alpha) * str_values[i-1]
        else:
            str_values[i] = true_range[i]
    
    return str_values


@njit(fastmath=True, cache=True)
def calculate_x_choppiness_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    dynamic_periods: np.ndarray = None
) -> tuple:
    """
    Xチョピネスを計算する（Numba最適化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period: 基本計算期間
        dynamic_periods: 動的期間配列（オプション）
        
    Returns:
        tuple: (Xチョピネス値の配列（反転済み、0-1の範囲）, STR値の配列)
    """
    length = len(high)
    
    # True Range値とSTR値を計算
    true_high, true_low, true_range = calculate_true_range_numba(high, low, close)
    str_values = calculate_str_numba(true_range, period, dynamic_periods)
    
    x_choppiness = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(period - 1, length):
        # 動的期間または固定期間を使用
        current_period = period
        if dynamic_periods is not None and i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
            current_period = max(5, min(int(dynamic_periods[i]), 50))  # 5-50期間に制限
        
        # 現在のインデックスが期間に対して十分かチェック
        if i < current_period - 1:
            continue
            
        # 期間内のSTRの合計を計算
        str_sum = 0.0
        valid_count = 0
        
        for j in range(i - current_period + 1, i + 1):
            if not np.isnan(str_values[j]):
                str_sum += str_values[j]
                valid_count += 1
        
        if valid_count < current_period // 2:  # 有効なデータが半分以下の場合はスキップ
            continue
            
        # 期間内の最高値と最安値を取得
        period_high = np.nanmax(high[i - current_period + 1:i + 1])
        period_low = np.nanmin(low[i - current_period + 1:i + 1])
        price_range = period_high - period_low
        
        if price_range > 0 and str_sum > 0:
            # チョピネス計算（通常のチョピネス式）
            choppiness = 100 * np.log10(str_sum / price_range) / np.log10(current_period)
            
            # 0-100の範囲にクリップ
            choppiness = max(0.0, min(100.0, choppiness))
            
            # 0-1の範囲に正規化
            normalized_choppiness = choppiness / 100.0
            
            # 値を反転（高い値=トレンド、低い値=レンジ）
            x_choppiness[i] = 1.0 - normalized_choppiness
    
    return x_choppiness, str_values


@njit(fastmath=True, cache=True)
def calculate_midline_and_signal(
    x_choppiness: np.ndarray,
    midline_period: int = 100
) -> tuple:
    """
    ミッドラインとトレンド信号を計算する（Numba最適化版）
    
    Args:
        x_choppiness: Xチョピネス値の配列
        midline_period: ミッドライン計算期間
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ミッドライン, トレンド信号)
    """
    length = len(x_choppiness)
    midline = np.full(length, np.nan, dtype=np.float64)
    trend_signal = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(midline_period - 1, length):
        # 期間内の最高値と最安値を計算
        period_data = x_choppiness[i - midline_period + 1:i + 1]
        
        # NaN値を除外
        valid_data = period_data[~np.isnan(period_data)]
        
        if len(valid_data) >= midline_period // 2:
            period_max = np.max(valid_data)
            period_min = np.min(valid_data)
            
            # ミッドライン = (最高値 + 最安値) / 2
            midline[i] = (period_max + period_min) / 2.0
            
            # トレンド信号の判定
            if not np.isnan(x_choppiness[i]):
                if x_choppiness[i] > midline[i]:
                    trend_signal[i] = 1.0   # トレンド相場
                else:
                    trend_signal[i] = -1.0  # レンジ相場
    
    return midline, trend_signal


class XChoppiness(Indicator, PercentileAnalysisMixin):
    """
    Xチョピネス（X-Choppiness）インジケーター
    
    通常のチョピネスインデックスをSTRベースで改良したバージョン：
    
    特徴:
    - ATRの代わりにSTR（Smooth True Range）を使用
    - 0-1の値範囲（通常のチョピネスは0-100）
    - 値を反転：高い値=トレンド相場、低い値=レンジ相場
    - 100期間ミッドラインによるトレンド判定機能
    - オプションで統合スムーサーによる平滑化機能
    
    計算式:
    1. STRを計算
    2. X-Choppiness = 1.0 - (100 * log10(STR_sum / price_range) / log10(period)) / 100
    3. ミッドライン = (過去100期間のX-Choppiness最高値 + 最安値) / 2
    4. トレンド信号 = X-Choppiness > ミッドライン ? 1 : -1
    """
    
    def __init__(
        self,
        period: int = 14,
        midline_period: int = 100,
        # 平滑化オプション
        use_smoothing: bool = True,
        smoother_type: str = 'frama',
        smoother_period: int = 8,
        smoother_src_type: str = 'close',
        # エラーズ統合サイクル検出器パラメータ
        use_dynamic_period: bool = False,
        detector_type: str = 'hody_e',
        lp_period: int = 12,
        hp_period: int = 124,
        cycle_part: float = 0.5,
        max_cycle: int = 124,
        min_cycle: int = 12,
        max_output: int = 89,
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
            period: Xチョピネス計算期間（デフォルト: 14、STR期間も兼ねる）
            midline_period: ミッドライン計算期間（デフォルト: 100）
            use_smoothing: 平滑化を使用するか（デフォルト: True）
            smoother_type: 統合スムーサータイプ（デフォルト: 'frama'）
            smoother_period: スムーサー期間（デフォルト: 8）
            smoother_src_type: スムーサーソースタイプ（デフォルト: 'close'）
            use_dynamic_period: 動的期間適応を使用するか（デフォルト: False）
            detector_type: サイクル検出器タイプ（デフォルト: 'hody_e'）
            lp_period: ローパスフィルター期間（デフォルト: 12）
            hp_period: ハイパスフィルター期間（デフォルト: 124）
            cycle_part: サイクル部分（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 124）
            min_cycle: 最小サイクル期間（デフォルト: 12）
            max_output: 最大出力値（デフォルト: 89）
            min_output: 最小出力値（デフォルト: 5）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: カルマンフィルタープロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: カルマンフィルター観測ノイズ（デフォルト: 0.001）
        """
        indicator_name = f"XChoppiness({period}, midline={midline_period}"
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
        self.midline_period = midline_period
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
        
        # パーセンタイルベーストレンド分析パラメータ
        self._add_percentile_analysis_params(
            enable_percentile_analysis=enable_percentile_analysis,
            percentile_lookback_period=percentile_lookback_period,
            percentile_low_threshold=percentile_low_threshold,
            percentile_high_threshold=percentile_high_threshold
        )
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_observation_noise = kalman_observation_noise
        
        # パラメータ検証
        if self.period <= 0:
            raise ValueError("periodは0より大きい必要があります")
        if self.period < 2:
            raise ValueError("periodは2以上である必要があります（STR安定性のため）")
        if self.midline_period <= 0:
            raise ValueError("midline_periodは0より大きい必要があります")
        if self.use_dynamic_period and self.max_cycle <= self.min_cycle:
            raise ValueError("max_cycleはmin_cycleより大きい必要があります")
        if self.use_kalman_filter and self.kalman_process_noise <= 0:
            raise ValueError("kalman_process_noiseは0より大きい必要があります")
        
        # STR関連パラメータ（内部実装で使用）
        # STRは内部でNumba関数として実装
        
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
                        src_type='hl2',  # 高値と安値をフィルタリング
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
            param_str = (f"{self.period}_{self.midline_period}_{self.str_period}_"
                        f"{self.str_src_type}_{self.use_smoothing}_{self.smoother_type}_"
                        f"{self.smoother_period}_{self.smoother_src_type}")
            
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(param_str)}"
            
        except Exception:
            return f"{id(data)}_{self.period}_{self.midline_period}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> XChoppinessResult:
        """
        Xチョピネスを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                必要なカラム: high, low, close
        
        Returns:
            XChoppinessResult: Xチョピネスの計算結果
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
                return XChoppinessResult(
                    values=cached_result.values.copy(),
                    raw_choppiness=cached_result.raw_choppiness.copy(),
                    smoothed_choppiness=cached_result.smoothed_choppiness.copy(),
                    midline=cached_result.midline.copy(),
                    trend_signal=cached_result.trend_signal.copy(),
                    str_values=cached_result.str_values.copy(),
                    percentiles=cached_result.percentiles.copy() if cached_result.percentiles is not None else None,
                    trend_state=cached_result.trend_state.copy() if cached_result.trend_state is not None else None,
                    trend_intensity=cached_result.trend_intensity.copy() if cached_result.trend_intensity is not None else None
                )
            
            # データの準備
            if isinstance(data, pd.DataFrame):
                required_cols = ['high', 'low', 'close']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"必要なカラムが不足しています: {missing_cols}")
                
                high = data['high'].to_numpy()
                low = data['low'].to_numpy()
                close = data['close'].to_numpy()
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                high = data[:, 1]   # high
                low = data[:, 2]    # low
                close = data[:, 3]  # close
            
            # カルマンフィルターによる高値・安値のフィルタリング（オプション）
            if self.use_kalman_filter and self.kalman_filter is not None:
                try:
                    # 高値・安値のデータフレームを作成
                    hl_data = pd.DataFrame({
                        'high': high,
                        'low': low,
                        'close': (high + low) / 2  # HL2を計算
                    })
                    
                    # カルマンフィルターを適用
                    kalman_result = self.kalman_filter.calculate(hl_data)
                    filtered_hl2 = kalman_result.values
                    
                    # フィルタリングされた値に基づいて高値・安値を調整
                    if len(filtered_hl2) == len(high):
                        original_hl2 = (high + low) / 2
                        adjustment_ratio = np.where(
                            original_hl2 != 0,
                            filtered_hl2 / original_hl2,
                            1.0
                        )
                        
                        # NaNや無効値の処理
                        adjustment_ratio = np.where(
                            np.isfinite(adjustment_ratio) & (adjustment_ratio > 0),
                            adjustment_ratio,
                            1.0
                        )
                        
                        high = high * adjustment_ratio
                        low = low * adjustment_ratio
                        
                        self.logger.debug("カルマンフィルターによる高値・安値の調整を適用しました")
                    else:
                        self.logger.warning("カルマンフィルター結果のサイズ不一致。元の値を使用します。")
                        
                except Exception as e:
                    self.logger.warning(f"カルマンフィルター適用中にエラー: {e}。元の値を使用します。")
            
            # NumPy配列に変換
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            close = np.asarray(close, dtype=np.float64)
            
            # データ長の検証
            data_length = len(high)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < max(self.period, self.midline_period):
                self.logger.warning(f"データ長（{data_length}）が必要な期間（{max(self.period, self.midline_period)}）より短いです")
            
            # 動的期間の計算（オプション）
            dynamic_periods = None
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
                    else:
                        dynamic_periods = np.full(len(data), self.period)
                        
                    self.logger.debug(f"動的期間を計算しました。範囲: {np.min(dynamic_periods):.1f} - {np.max(dynamic_periods):.1f}")
                        
                except Exception as e:
                    self.logger.warning(f"動的期間計算中にエラー: {e}。固定期間を使用します。")
                    dynamic_periods = None
            
            # Xチョピネス（とSTR）を計算（内部実装）
            raw_choppiness, str_values = calculate_x_choppiness_numba(
                high, low, close, self.period, dynamic_periods
            )
            
            # 平滑化（オプション）
            smoothed_choppiness = np.full_like(raw_choppiness, np.nan)
            if self.use_smoothing and self.smoother is not None:
                try:
                    # チョピネス値をDataFrame形式に変換
                    chop_df = pd.DataFrame({'close': raw_choppiness})
                    
                    # 平滑化を適用
                    smoother_result = self.smoother.calculate(chop_df)
                    smoothed_choppiness = smoother_result.values
                except Exception as e:
                    self.logger.warning(f"平滑化処理中にエラー: {e}。生の値を使用します。")
                    smoothed_choppiness = raw_choppiness.copy()
            else:
                smoothed_choppiness = raw_choppiness.copy()
            
            # 最終的なチョピネス値（平滑化が有効な場合は平滑化値、そうでなければ生の値）
            final_choppiness = smoothed_choppiness if self.use_smoothing else raw_choppiness
            
            # ミッドラインとトレンド信号を計算
            midline, trend_signal = calculate_midline_and_signal(
                final_choppiness, self.midline_period
            )
            
            # パーセンタイルベーストレンド分析（オプション）
            percentiles, trend_state, trend_intensity = self._calculate_percentile_analysis(
                final_choppiness, 'trend'
            )
            
            # 結果の作成
            result = XChoppinessResult(
                values=final_choppiness.copy(),
                raw_choppiness=raw_choppiness.copy(),
                smoothed_choppiness=smoothed_choppiness.copy(),
                midline=midline.copy(),
                trend_signal=trend_signal.copy(),
                str_values=str_values.copy(),
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
            self._values = final_choppiness
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Xチョピネス計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            empty_array = np.array([])
            return XChoppinessResult(
                values=empty_array,
                raw_choppiness=empty_array,
                smoothed_choppiness=empty_array,
                midline=empty_array,
                trend_signal=empty_array,
                str_values=empty_array,
                percentiles=None,
                trend_state=None,
                trend_intensity=None
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """Xチョピネス値を取得（後方互換性のため）"""
        if not self._result_cache:
            return None
        
        result = self._get_latest_result()
        return result.values.copy() if result else None
    
    def get_raw_choppiness(self) -> Optional[np.ndarray]:
        """生のチョピネス値を取得"""
        result = self._get_latest_result()
        return result.raw_choppiness.copy() if result else None
    
    def get_smoothed_choppiness(self) -> Optional[np.ndarray]:
        """平滑化されたチョピネス値を取得"""
        result = self._get_latest_result()
        return result.smoothed_choppiness.copy() if result else None
    
    def get_midline(self) -> Optional[np.ndarray]:
        """ミッドライン値を取得"""
        result = self._get_latest_result()
        return result.midline.copy() if result else None
    
    def get_trend_signal(self) -> Optional[np.ndarray]:
        """トレンド信号を取得"""
        result = self._get_latest_result()
        return result.trend_signal.copy() if result else None
    
    def get_str_values(self) -> Optional[np.ndarray]:
        """STR値を取得"""
        result = self._get_latest_result()
        return result.str_values.copy() if result else None
    
    def get_percentiles(self) -> Optional[np.ndarray]:
        """パーセンタイル値を取得"""
        result = self._get_latest_result()
        return result.percentiles.copy() if result and result.percentiles is not None else None
    
    def get_trend_state(self) -> Optional[np.ndarray]:
        """トレンド状態を取得（-1=レンジ、0=中、1=トレンド）"""
        result = self._get_latest_result()
        return result.trend_state.copy() if result and result.trend_state is not None else None
    
    def get_trend_intensity(self) -> Optional[np.ndarray]:
        """トレンド強度を取得（0-1の範囲）"""
        result = self._get_latest_result()
        return result.trend_intensity.copy() if result and result.trend_intensity is not None else None
    
    def get_percentile_analysis_summary(self) -> Dict[str, Any]:
        """パーセンタイル分析の要約情報を取得"""
        result = self._get_latest_result()
        if not result:
            return {}
        
        return self._get_percentile_analysis_summary(
            result.percentiles, result.trend_state
        )
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'period': self.period,
            'midline_period': self.midline_period,
            'use_smoothing': self.use_smoothing,
            'smoother_type': self.smoother_type if self.use_smoothing else None,
            'smoother_period': self.smoother_period if self.use_smoothing else None,
            'use_dynamic_period': self.use_dynamic_period,
            'detector_type': self.detector_type if self.use_dynamic_period else None,
            'use_kalman_filter': self.use_kalman_filter,
            'kalman_filter_type': self.kalman_filter_type if self.use_kalman_filter else None,
            'description': 'STRベースの改良チョピネスインデックス（内部STR実装、0-1範囲、高値=トレンド、動的期間・カルマンフィルター対応）'
        }
    
    def _get_latest_result(self) -> Optional[XChoppinessResult]:
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
def calculate_x_choppiness(
    data: Union[pd.DataFrame, np.ndarray],
    period: int = 14,
    midline_period: int = 100,
    use_smoothing: bool = False,
    smoother_type: str = 'frama',
    use_dynamic_period: bool = False,
    use_kalman_filter: bool = False,
    enable_percentile_analysis: bool = True,
    percentile_lookback_period: int = 50,
    percentile_low_threshold: float = 0.25,
    percentile_high_threshold: float = 0.75,
    **kwargs
) -> np.ndarray:
    """
    Xチョピネスの計算（便利関数）
    
    Args:
        data: 価格データ
        period: チョピネス計算期間（STR期間も兼ねる）
        midline_period: ミッドライン計算期間
        use_smoothing: 平滑化を使用するか
        smoother_type: スムーサータイプ
        use_dynamic_period: 動的期間適応を使用するか
        use_kalman_filter: カルマンフィルターを使用するか
        **kwargs: その他のパラメータ
        
    Returns:
        Xチョピネス値
    """
    indicator = XChoppiness(
        period=period,
        midline_period=midline_period,
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
    return result.values


if __name__ == "__main__":
    """直接実行時のテスト"""
    from datetime import datetime, timedelta
    
    print("=== Xチョピネス インジケーターのテスト ===")
    
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
    
    # Xチョピネスを計算（基本版）
    print("\n基本版Xチョピネスをテスト中...")
    x_chop = XChoppiness(period=14, midline_period=50)
    result = x_chop.calculate(df)
    
    valid_count = np.sum(~np.isnan(result.values))
    mean_x_chop = np.nanmean(result.values)
    trend_ratio = np.sum(result.trend_signal == 1) / np.sum(~np.isnan(result.trend_signal))
    
    print(f"  有効値数: {valid_count}/{len(df)}")
    print(f"  平均Xチョピネス: {mean_x_chop:.4f}")
    print(f"  トレンド信号比率: {trend_ratio:.2%}")
    
    # 平滑化版をテスト
    print("\n平滑化版Xチョピネスをテスト中...")
    x_chop_smooth = XChoppiness(
        period=14,
        midline_period=50,
        use_smoothing=True,
        smoother_type='frama',
        smoother_period=8
    )
    result_smooth = x_chop_smooth.calculate(df)
    
    valid_count_smooth = np.sum(~np.isnan(result_smooth.values))
    mean_x_chop_smooth = np.nanmean(result_smooth.values)
    
    print(f"  有効値数: {valid_count_smooth}/{len(df)}")
    print(f"  平均Xチョピネス（平滑化）: {mean_x_chop_smooth:.4f}")
    
    # 比較統計
    if valid_count > 0 and valid_count_smooth > 0:
        correlation = np.corrcoef(
            result.values[~np.isnan(result.values)][-min(valid_count, valid_count_smooth):],
            result_smooth.values[~np.isnan(result_smooth.values)][-min(valid_count, valid_count_smooth):]
        )[0, 1]
        print(f"  基本版と平滑化版の相関: {correlation:.4f}")
    
    print("\n=== テスト完了 ===")