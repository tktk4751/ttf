#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **X_ADX - 拡張的Average Directional Index** 🎯

NormalizedADXを強化し、X_Choppinessと同様の動的期間対応とカルマン統合、
スムーサー統合を実装した次世代のトレンドインジケーター。

🌟 **主要機能:**
1. **True Range計算方法の選択**: ATRベース vs STRベース
2. **統合スムージング**: unified_smootherによる多様なスムージング手法
3. **カルマンフィルター統合**: 価格データをフィルタリングしてノイズ除去
4. **処理フロー**: ソース価格→カルマンフィルター→ADX計算→スムージング
5. **100期間ミッドライン**: X_Choppinessと同様のトレンド判定ロジック

📊 **処理順序:**
1. プライスソース取得（高値、安値、終値）
2. カルマンフィルター適用（オプション）
3. True Range計算（ATRまたはSTR方式）
4. ADX計算とスムージング
5. ミッドライン計算によるトレンド判定

🔧 **パラメータ:**
- tr_method: 'atr' または 'str' - True Range計算方法
- smoother_type: スムージング手法（FRAMA, Super Smoother, Ultimate Smoother, ZLEMA）
- enable_kalman: カルマンフィルター使用フラグ
- kalman_type: カルマンフィルター種別
- period_mode: 'fixed' または 'dynamic' - 期間モード
- cycle_detector_type: サイクル検出器タイプ（動的期間モード用）
- midline_period: ミッドライン計算期間（デフォルト: 100）
"""

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
import pandas as pd
from numba import njit
import traceback

from ..indicator import Indicator
from ..price_source import PriceSource
from ..utils.percentile_analysis import (
    calculate_percentile,
    calculate_trend_classification,
    PercentileAnalysisMixin
)

# 条件付きインポート（オプション機能）
try:
    from ..smoother.unified_smoother import UnifiedSmoother
    UNIFIED_SMOOTHER_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行（パス調整付き）
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.smoother.unified_smoother import UnifiedSmoother
        UNIFIED_SMOOTHER_AVAILABLE = True
    except ImportError:
        UnifiedSmoother = None
        UNIFIED_SMOOTHER_AVAILABLE = False

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
class XADXResult:
    """X_ADXの計算結果"""
    values: np.ndarray                    # X_ADX値（正規化されたADX）
    raw_adx: np.ndarray                   # 生のADX値（平滑化前）
    smoothed_adx: np.ndarray              # 平滑化されたADX値（オプション）
    midline: np.ndarray                   # ミッドライン値
    trend_signal: np.ndarray              # トレンド判定信号（1=トレンド、-1=レンジ）
    tr_values: np.ndarray                 # True Range値
    plus_di: np.ndarray                   # +DI値
    minus_di: np.ndarray                  # -DI値
    # パーセンタイルベースのトレンド分析
    percentiles: Optional[np.ndarray]     # パーセンタイル値
    trend_state: Optional[np.ndarray]     # トレンド状態（-1=レンジ、0=中、1=トレンド）
    trend_intensity: Optional[np.ndarray] # トレンド強度（0-1）


@njit(fastmath=True, cache=True)
def calculate_true_range_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> np.ndarray:
    """
    ATR方式でTrue Rangeを計算する（Numba最適化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        
    Returns:
        True Range値の配列
    """
    length = len(high)
    tr = np.zeros(length, dtype=np.float64)
    
    # 最初の値
    tr[0] = high[0] - low[0]
    
    # 2番目以降
    for i in range(1, length):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    return tr


@njit(fastmath=True, cache=True)
def calculate_true_range_str(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: float
) -> np.ndarray:
    """
    STR方式でTrue Rangeを計算する（Numba最適化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        period: STR計算期間
        
    Returns:
        STR値の配列
    """
    length = len(high)
    str_values = np.zeros(length, dtype=np.float64)
    
    # 期間の整数部分を取得
    int_period = int(period)
    
    for i in range(int_period, length):
        # STRの計算（期間内の True Range の移動平均）
        tr_sum = 0.0
        count = 0
        
        for j in range(max(1, i - int_period + 1), i + 1):
            if j < length:
                tr1 = high[j] - low[j]
                tr2 = abs(high[j] - close[j-1]) if j > 0 else 0.0
                tr3 = abs(low[j] - close[j-1]) if j > 0 else 0.0
                tr_val = max(tr1, tr2, tr3)
                tr_sum += tr_val
                count += 1
        
        if count > 0:
            str_values[i] = tr_sum / count
    
    return str_values


@njit(fastmath=True, cache=True)
def calculate_directional_movement(
    high: np.ndarray,
    low: np.ndarray
) -> tuple:
    """
    Directional Movement（+DM, -DM）を計算する（Numba最適化版）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (+DM, -DM)の配列
    """
    length = len(high)
    plus_dm = np.zeros(length, dtype=np.float64)
    minus_dm = np.zeros(length, dtype=np.float64)
    
    for i in range(1, length):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        
        if up > down and up > 0:
            plus_dm[i] = up
        else:
            plus_dm[i] = 0.0
            
        if down > up and down > 0:
            minus_dm[i] = down
        else:
            minus_dm[i] = 0.0
    
    return plus_dm, minus_dm


@njit(fastmath=True, cache=True)
def calculate_raw_adx_components_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tr_values: np.ndarray,
    period: int,
    dynamic_periods: np.ndarray = None
) -> tuple:
    """
    ADXの基本コンポーネント（DX値、+DI、-DI）を計算する（Numba最適化版）
    スムージングは後で統合スムーサーで行う
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        tr_values: True Range値の配列
        period: 基本計算期間
        dynamic_periods: 動的期間配列（オプション）
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (DX値, +DI値, -DI値)
    """
    length = len(high)
    dx_values = np.full(length, np.nan, dtype=np.float64)
    plus_di = np.full(length, np.nan, dtype=np.float64)
    minus_di = np.full(length, np.nan, dtype=np.float64)
    
    # +DM, -DMの計算
    plus_dm, minus_dm = calculate_directional_movement(high, low)
    
    # 指数移動平均の計算用の係数（基本的なスムージング用）
    alpha = 2.0 / (period + 1.0)
    
    # 平滑化用の変数
    smoothed_tr = np.zeros(length, dtype=np.float64)
    smoothed_plus_dm = np.zeros(length, dtype=np.float64)
    smoothed_minus_dm = np.zeros(length, dtype=np.float64)
    
    for i in range(period - 1, length):
        # 動的期間または固定期間を使用
        current_period = period
        if dynamic_periods is not None and i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
            current_period = max(5, min(int(dynamic_periods[i]), 50))  # 5-50期間に制限
        
        current_alpha = 2.0 / (current_period + 1.0)
        
        # 最初の値の初期化
        if i == period - 1:
            smoothed_tr[i] = tr_values[i]
            smoothed_plus_dm[i] = plus_dm[i]
            smoothed_minus_dm[i] = minus_dm[i]
        else:
            # 指数移動平均による平滑化（基本的なスムージングのみ）
            smoothed_tr[i] = (tr_values[i] * current_alpha) + (smoothed_tr[i-1] * (1 - current_alpha))
            smoothed_plus_dm[i] = (plus_dm[i] * current_alpha) + (smoothed_plus_dm[i-1] * (1 - current_alpha))
            smoothed_minus_dm[i] = (minus_dm[i] * current_alpha) + (smoothed_minus_dm[i-1] * (1 - current_alpha))
        
        # +DI, -DIの計算
        if smoothed_tr[i] > 0:
            plus_di[i] = smoothed_plus_dm[i] / smoothed_tr[i]
            minus_di[i] = smoothed_minus_dm[i] / smoothed_tr[i]
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
        
        # DXの計算（ADXの前段階、スムージングなし）
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx_values[i] = abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx_values[i] = 0.0
    
    return dx_values, plus_di, minus_di


@njit(fastmath=True, cache=True)
def calculate_midline_and_trend_signal(
    x_adx: np.ndarray,
    midline_period: int = 100
) -> tuple:
    """
    ミッドラインとトレンド信号を計算する（Numba最適化版）
    
    Args:
        x_adx: X_ADX値の配列
        midline_period: ミッドライン計算期間
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ミッドライン, トレンド信号)
    """
    length = len(x_adx)
    midline = np.full(length, np.nan, dtype=np.float64)
    trend_signal = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(midline_period - 1, length):
        # 期間内の最高値と最安値を計算
        period_data = x_adx[i - midline_period + 1:i + 1]
        
        # NaN値を除外
        valid_data = period_data[~np.isnan(period_data)]
        
        if len(valid_data) >= midline_period // 2:
            period_max = np.max(valid_data)
            period_min = np.min(valid_data)
            
            # ミッドライン = (最高値 + 最安値) / 2
            midline[i] = (period_max + period_min) / 2.0
            
            # トレンド信号の判定
            if not np.isnan(x_adx[i]):
                if x_adx[i] > midline[i]:
                    trend_signal[i] = 1.0  # トレンド
                else:
                    trend_signal[i] = -1.0  # レンジ
    
    return midline, trend_signal


class XADX(Indicator, PercentileAnalysisMixin):
    """
    X_ADX（拡張的Average Directional Index）インジケーター
    
    NormalizedADXをベースに、X_Choppinessと同様の機能を追加：
    - カルマンフィルター統合
    - 統合スムーサー対応
    - 動的期間適応
    - True Range計算方法の選択（ATR/STR）
    - 100期間ミッドラインによるトレンド判定
    """
    
    def __init__(self,
                 period: int = 13,
                 midline_period: int = 100,
                 tr_method: str = 'atr',
                 str_period: float = 20.0,
                 src_type: str = 'hlc3',
                 # 平滑化オプション
                 use_smoothing: bool = True,
                 smoother_type: str = 'super_smoother',
                 smoother_period: int = 10,
                 smoother_src_type: str = 'close',
                 # 動的期間オプション
                 use_dynamic_period: bool = False,
                 detector_type: str = 'hody_e',
                 lp_period: int = 13,
                 hp_period: int = 124,
                 cycle_part: float = 0.5,
                 max_cycle: int = 124,
                 min_cycle: int = 13,
                 max_output: int = 124,
                 min_output: int = 13,
                 # カルマンフィルターオプション
                 use_kalman_filter: bool = False,
                 kalman_filter_type: str = 'unscented',
                 kalman_process_noise: float = 0.01,
                 kalman_observation_noise: float = 0.001,
                 # パーセンタイル分析オプション
                 enable_percentile_analysis: bool = True,
                 percentile_lookback_period: int = 50,
                 percentile_low_threshold: float = 0.25,
                 percentile_high_threshold: float = 0.75,
                 **kwargs):
        """
        コンストラクタ
        
        Args:
            period: ADX計算期間
            midline_period: ミッドライン計算期間
            tr_method: True Range計算方法（'atr' または 'str'）
            str_period: STR期間（str_method='str'の場合）
            src_type: プライスソースタイプ
            use_smoothing: 平滑化を使用するか
            smoother_type: 統合スムーサータイプ
            smoother_period: スムーサー期間
            smoother_src_type: スムーサーソースタイプ
            use_dynamic_period: 動的期間適応を使用するか
            detector_type: サイクル検出器タイプ
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            use_kalman_filter: カルマンフィルターを使用するか
            kalman_filter_type: カルマンフィルタータイプ
            kalman_process_noise: カルマンフィルタープロセスノイズ
            kalman_observation_noise: カルマンフィルター観測ノイズ
        """
        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_period else ""
        kalman_str = f"_kalman({kalman_filter_type})" if use_kalman_filter else ""
        smoother_str = f"_smooth({smoother_type})" if use_smoothing else ""
        
        super().__init__(f"X_ADX(p={period},mid={midline_period},tr={tr_method}{dynamic_str}{kalman_str}{smoother_str})")
        
        self.period = period
        self.midline_period = midline_period
        self.tr_method = tr_method
        self.str_period = str_period
        self.src_type = src_type
        
        # 平滑化オプション
        self.use_smoothing = use_smoothing
        self.smoother_type = smoother_type
        self.smoother_period = smoother_period
        self.smoother_src_type = smoother_src_type
        
        # 動的期間オプション
        self.use_dynamic_period = use_dynamic_period
        self.detector_type = detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        
        # カルマンフィルターオプション
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_observation_noise = kalman_observation_noise
        
        # パーセンタイル分析パラメータの初期化
        self._add_percentile_analysis_params(
            enable_percentile_analysis=enable_percentile_analysis,
            percentile_lookback_period=percentile_lookback_period,
            percentile_low_threshold=percentile_low_threshold,
            percentile_high_threshold=percentile_high_threshold,
            **kwargs
        )
        
        # 依存インジケーターの初期化
        self._init_dependencies()
        
        self._cache = {}
        self._result: Optional[XADXResult] = None
    
    def _init_dependencies(self):
        """依存するインジケーターを初期化"""
        # スムーサーの初期化
        self.smoother = None
        if self.use_smoothing and UNIFIED_SMOOTHER_AVAILABLE:
            try:
                # 動的期間対応スムーサーのチェック
                dynamic_supported_smoothers = ['ultimate_smoother', 'frama', 'super_smoother', 'zero_lag_ema', 'zlema']
                smoother_period_mode = 'fixed'  # デフォルトは固定期間
                
                if self.use_dynamic_period:
                    if self.smoother_type in dynamic_supported_smoothers:
                        smoother_period_mode = 'dynamic'
                        self.logger.info(f"{self.smoother_type}は動的期間に対応しています。動的モードで初期化します。")
                    else:
                        self.logger.warning(
                            f"{self.smoother_type}は動的期間に対応していません。"
                            f"固定期間モードでスムージングします。"
                            f"動的期間対応スムーサー: {', '.join(dynamic_supported_smoothers)}"
                        )
                
                # スムーサーパラメータの設定
                smoother_params = {
                    'smoother_type': self.smoother_type,
                    'period': self.smoother_period,
                    'src_type': self.smoother_src_type,
                    'period_mode': smoother_period_mode
                }
                
                # 動的期間パラメータの追加（ultimate_smootherのみ）
                if smoother_period_mode == 'dynamic':
                    smoother_params.update({
                        'cycle_detector_type': self.detector_type,
                        'cycle_part': self.cycle_part,
                        'max_cycle': self.max_cycle,
                        'min_cycle': self.min_cycle,
                        'max_output': self.max_output,
                        'min_output': self.min_output,
                        'lp_period': self.lp_period,
                        'hp_period': self.hp_period
                    })
                
                self.smoother = UnifiedSmoother(**smoother_params)
                
            except Exception as e:
                self.logger.warning(f"統合スムーサーの初期化に失敗しました: {e}")
                # フォールバック: 簡素なスムーサーを作成
                try:
                    self.smoother = UnifiedSmoother(
                        smoother_type=self.smoother_type,
                        period=self.smoother_period,
                        src_type=self.smoother_src_type,
                        period_mode='fixed'  # フォールバックは固定期間
                    )
                except Exception as e2:
                    self.logger.warning(f"簡素なスムーサーの初期化も失敗しました: {e2}")
                    self.use_smoothing = False
        elif self.use_smoothing and not UNIFIED_SMOOTHER_AVAILABLE:
            self.logger.warning("UnifiedSmootherが利用できません。平滑化は無効化されます。")
            self.use_smoothing = False
        
        # ドミナントサイクル検出器の初期化
        self.dc_detector = None
        if self.use_dynamic_period and EHLERS_UNIFIED_DC_AVAILABLE:
            try:
                self.dc_detector = EhlersUnifiedDC(
                    detector_type=self.detector_type,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period,
                    cycle_part=self.cycle_part,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    src_type=self.src_type
                )
            except Exception as e:
                self.logger.warning(f"ドミナントサイクル検出器の初期化に失敗しました: {e}")
                self.use_dynamic_period = False
        elif self.use_dynamic_period and not EHLERS_UNIFIED_DC_AVAILABLE:
            self.logger.warning("EhlersUnifiedDCが利用できません。動的期間は無効化されます。")
            self.use_dynamic_period = False
        
        # カルマンフィルターの初期化
        self.kalman_filter = None
        if self.use_kalman_filter and UNIFIED_KALMAN_AVAILABLE:
            try:
                self.kalman_filter = UnifiedKalman(
                    kalman_type=self.kalman_filter_type,
                    process_noise=self.kalman_process_noise,
                    observation_noise=self.kalman_observation_noise
                )
            except Exception as e:
                self.logger.warning(f"カルマンフィルターの初期化に失敗しました: {e}")
                self.use_kalman_filter = False
        elif self.use_kalman_filter and not UNIFIED_KALMAN_AVAILABLE:
            self.logger.warning("UnifiedKalmanが利用できません。カルマンフィルターは無効化されます。")
            self.use_kalman_filter = False
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> XADXResult:
        """
        X_ADXを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high', 'low', 'close'カラムが必要
        
        Returns:
            XADXResult: X_ADX値とトレンド情報を含む結果
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
                
                high = data['high'].values.astype(np.float64)
                low = data['low'].values.astype(np.float64)
                close = data['close'].values.astype(np.float64)
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                
                high = data[:, 1].astype(np.float64)
                low = data[:, 2].astype(np.float64)
                close = data[:, 3].astype(np.float64)
            
            length = len(high)
            
            # カルマンフィルターの適用（オプション）
            if self.use_kalman_filter and self.kalman_filter is not None:
                try:
                    # 各価格系列にカルマンフィルターを適用
                    filtered_high = self.kalman_filter.filter(high)
                    filtered_low = self.kalman_filter.filter(low)
                    filtered_close = self.kalman_filter.filter(close)
                    
                    if filtered_high is not None and len(filtered_high) == length:
                        high = filtered_high.astype(np.float64)
                    if filtered_low is not None and len(filtered_low) == length:
                        low = filtered_low.astype(np.float64)
                    if filtered_close is not None and len(filtered_close) == length:
                        close = filtered_close.astype(np.float64)
                        
                except Exception as e:
                    self.logger.warning(f"カルマンフィルターの適用に失敗しました: {e}")
            
            # True Range の計算
            if self.tr_method == 'str':
                tr_values = calculate_true_range_str(high, low, close, self.str_period)
            else:  # 'atr'
                tr_values = calculate_true_range_atr(high, low, close)
            
            # 動的期間の計算（オプション）
            dynamic_periods = None
            if self.use_dynamic_period and self.dc_detector is not None:
                try:
                    if isinstance(data, pd.DataFrame):
                        dc_result = self.dc_detector.calculate(data)
                    else:
                        df = pd.DataFrame({
                            'open': data[:, 0],
                            'high': data[:, 1], 
                            'low': data[:, 2],
                            'close': data[:, 3]
                        })
                        dc_result = self.dc_detector.calculate(df)
                    
                    if dc_result is not None:
                        dynamic_periods = np.asarray(dc_result, dtype=np.float64)
                        
                except Exception as e:
                    self.logger.warning(f"ドミナントサイクル検出に失敗しました: {e}")
            
            # ADXの基本コンポーネント（DX、+DI、-DI）を計算
            dx_values, plus_di, minus_di = calculate_raw_adx_components_numba(
                high, low, close, tr_values, self.period, dynamic_periods
            )
            
            # DX値から統合スムーサーでADXを計算（動的期間対応）
            final_adx = np.full(length, np.nan, dtype=np.float64)
            raw_adx = np.full(length, np.nan, dtype=np.float64)
            
            if self.use_smoothing and self.smoother is not None:
                try:
                    if isinstance(data, pd.DataFrame):
                        # DX値をDataFrameに変換してスムージング（統合スムーサーでADXを作成）
                        dx_df = pd.DataFrame({
                            'open': data['open'],
                            'high': data['high'],
                            'low': data['low'],
                            'close': dx_values,  # DX値をcloseとして使用
                            'volume': data.get('volume', pd.Series([1] * len(data)))
                        }, index=data.index)
                        
                        smoother_result = self.smoother.calculate(dx_df)
                        if smoother_result is not None and len(smoother_result) == length:
                            final_adx = smoother_result.astype(np.float64)
                            
                        # raw_adxは動的期間対応のEMAでのスムージング
                        for i in range(self.period - 1, length):
                            if not np.isnan(dx_values[i]):
                                # 動的期間または固定期間を使用
                                current_period = self.period
                                if dynamic_periods is not None and i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
                                    current_period = max(5, min(int(dynamic_periods[i]), 50))  # 5-50期間に制限
                                
                                adx_alpha = 2.0 / (current_period + 1.0)
                                if i == self.period - 1:
                                    raw_adx[i] = dx_values[i]
                                else:
                                    raw_adx[i] = (dx_values[i] * adx_alpha) + (raw_adx[i-1] * (1 - adx_alpha))
                    
                except Exception as e:
                    self.logger.warning(f"統合スムーサーの適用に失敗しました: {e}")
                    # フォールバック: 動的期間対応のEMAでADXを計算
                    for i in range(self.period - 1, length):
                        if not np.isnan(dx_values[i]):
                            # 動的期間または固定期間を使用
                            current_period = self.period
                            if dynamic_periods is not None and i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
                                current_period = max(5, min(int(dynamic_periods[i]), 50))  # 5-50期間に制限
                            
                            adx_alpha = 2.0 / (current_period + 1.0)
                            if i == self.period - 1:
                                final_adx[i] = dx_values[i]
                            else:
                                final_adx[i] = (dx_values[i] * adx_alpha) + (final_adx[i-1] * (1 - adx_alpha))
                    raw_adx = final_adx.copy()
            else:
                # スムージングなしの場合は動的期間対応のEMAでADXを計算
                for i in range(self.period - 1, length):
                    if not np.isnan(dx_values[i]):
                        # 動的期間または固定期間を使用
                        current_period = self.period
                        if dynamic_periods is not None and i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
                            current_period = max(5, min(int(dynamic_periods[i]), 50))  # 5-50期間に制限
                        
                        adx_alpha = 2.0 / (current_period + 1.0)
                        if i == self.period - 1:
                            final_adx[i] = dx_values[i]
                        else:
                            final_adx[i] = (dx_values[i] * adx_alpha) + (final_adx[i-1] * (1 - adx_alpha))
                raw_adx = final_adx.copy()
            
            # スムージング後のADXを作成（統合スムーサー使用時とそうでない時の区別）
            smoothed_adx = final_adx if self.use_smoothing and self.smoother is not None else np.full(length, np.nan)
            
            # ミッドラインとトレンド信号の計算
            midline, trend_signal = calculate_midline_and_trend_signal(final_adx, self.midline_period)
            
            # パーセンタイル分析の実行
            percentiles, trend_state, trend_intensity = self._calculate_percentile_analysis(
                final_adx, analysis_type='trend'
            )
            
            # 結果の作成
            result = XADXResult(
                values=final_adx,
                raw_adx=raw_adx,
                smoothed_adx=smoothed_adx,
                midline=midline,
                trend_signal=trend_signal,
                tr_values=tr_values,
                plus_di=plus_di,
                minus_di=minus_di,
                percentiles=percentiles,
                trend_state=trend_state,
                trend_intensity=trend_intensity
            )
            
            self._result = result
            self._values = final_adx  # Indicatorクラスの標準出力
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"X_ADX '{self.name}' 計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は NaN で埋めた結果を返す
            length = len(data) if hasattr(data, '__len__') else 100
            error_result = XADXResult(
                values=np.full(length, np.nan),
                raw_adx=np.full(length, np.nan),
                smoothed_adx=np.full(length, np.nan),
                midline=np.full(length, np.nan),
                trend_signal=np.full(length, np.nan),
                tr_values=np.full(length, np.nan),
                plus_di=np.full(length, np.nan),
                minus_di=np.full(length, np.nan),
                percentiles=None,
                trend_state=None,
                trend_intensity=None
            )
            return error_result
    
    def reset(self) -> None:
        """インジケータの状態をリセット"""
        super().reset()
        self._result = None
        self._cache = {}
        
        # 依存インジケーターのリセット
        if self.smoother and hasattr(self.smoother, 'reset'):
            self.smoother.reset()
        if self.dc_detector and hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset()
        if self.kalman_filter and hasattr(self.kalman_filter, 'reset'):
            self.kalman_filter.reset()
    
    # パーセンタイル分析関連のgetter メソッド
    def get_percentiles(self) -> Optional[np.ndarray]:
        """パーセンタイル値を取得"""
        return self._result.percentiles if self._result else None
    
    def get_trend_state(self) -> Optional[np.ndarray]:
        """トレンド状態を取得"""
        return self._result.trend_state if self._result else None
    
    def get_trend_intensity(self) -> Optional[np.ndarray]:
        """トレンド強度を取得"""
        return self._result.trend_intensity if self._result else None
    
    def get_indicator_info(self) -> dict:
        """インジケーター情報を取得"""
        percentile_str = f", percentile_analysis=True(lookback={self.percentile_lookback_period}, thresholds=[{self.percentile_low_threshold}-{self.percentile_high_threshold}])" if self.enable_percentile_analysis else ""
        
        info = {
            'name': f"X_ADX({self.period}, midline={self.midline_period}, tr={self.tr_method}({self.str_period if self.tr_method == 'str' else 'atr'}), dynamic={self.detector_type if self.use_dynamic_period else 'none'}, kalman={self.kalman_filter_type if self.use_kalman_filter else 'none'}, smooth={self.smoother_type}({self.smoother_period})){percentile_str}",
            'period': self.period,
            'midline_period': self.midline_period,
            'tr_method': self.tr_method,
            'str_period': self.str_period if self.tr_method == 'str' else None,
            'src_type': self.src_type,
            'use_smoothing': self.use_smoothing,
            'smoother_type': self.smoother_type if self.use_smoothing else None,
            'smoother_period': self.smoother_period if self.use_smoothing else None,
            'smoother_src_type': self.smoother_src_type if self.use_smoothing else None,
            'use_dynamic_period': self.use_dynamic_period,
            'detector_type': self.detector_type if self.use_dynamic_period else None,
            'use_kalman_filter': self.use_kalman_filter,
            'kalman_filter_type': self.kalman_filter_type if self.use_kalman_filter else None,
            'enable_percentile_analysis': self.enable_percentile_analysis,
            'percentile_lookback_period': self.percentile_lookback_period if self.enable_percentile_analysis else None,
            'percentile_low_threshold': self.percentile_low_threshold if self.enable_percentile_analysis else None,
            'percentile_high_threshold': self.percentile_high_threshold if self.enable_percentile_analysis else None,
            'description': 'ADXベースの拡張トレンドインジケーター（0-1範囲、高値=強いトレンド、カルマンフィルター・動的期間・パーセンタイル分析対応）'
        }
        return info


# 便利関数
def calculate_x_adx(
    data: Union[pd.DataFrame, np.ndarray],
    period: int = 13,
    midline_period: int = 100,
    tr_method: str = 'atr',
    str_period: float = 20.0,
    src_type: str = 'hlc3',
    use_smoothing: bool = True,
    smoother_type: str = 'super_smoother',
    smoother_period: int = 10,
    use_dynamic_period: bool = False,
    detector_type: str = 'hody_e',
    use_kalman_filter: bool = False,
    kalman_filter_type: str = 'unscented',
    enable_percentile_analysis: bool = False,
    percentile_lookback_period: int = 50,
    percentile_low_threshold: float = 0.25,
    percentile_high_threshold: float = 0.75,
    **kwargs
) -> np.ndarray:
    """
    X_ADX（拡張的Average Directional Index）を計算する便利関数
    
    Args:
        data: 価格データ（DataFrameまたはNumPy配列）
        period: ADX計算期間
        midline_period: ミッドライン計算期間
        tr_method: True Range計算方法（'atr' または 'str'）
        str_period: STR期間（tr_method='str'の場合）
        src_type: プライスソースタイプ
        use_smoothing: 平滑化を使用するか
        smoother_type: 統合スムーサータイプ
        smoother_period: スムーサー期間
        use_dynamic_period: 動的期間適応を使用するか
        detector_type: サイクル検出器タイプ
        use_kalman_filter: カルマンフィルターを使用するか
        kalman_filter_type: カルマンフィルタータイプ
        enable_percentile_analysis: パーセンタイル分析を有効にするか
        percentile_lookback_period: パーセンタイル計算のルックバック期間
        percentile_low_threshold: パーセンタイル低閾値
        percentile_high_threshold: パーセンタイル高閾値
        **kwargs: その他のパラメータ
        
    Returns:
        X_ADX値の配列
    """
    indicator = XADX(
        period=period,
        midline_period=midline_period,
        tr_method=tr_method,
        str_period=str_period,
        src_type=src_type,
        use_smoothing=use_smoothing,
        smoother_type=smoother_type,
        smoother_period=smoother_period,
        use_dynamic_period=use_dynamic_period,
        detector_type=detector_type,
        use_kalman_filter=use_kalman_filter,
        kalman_filter_type=kalman_filter_type,
        enable_percentile_analysis=enable_percentile_analysis,
        percentile_lookback_period=percentile_lookback_period,
        percentile_low_threshold=percentile_low_threshold,
        percentile_high_threshold=percentile_high_threshold,
        **kwargs
    )
    
    result = indicator.calculate(data)
    return result.values