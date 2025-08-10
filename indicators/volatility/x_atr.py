#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **X_ATR - 拡張的Average True Range** 🎯

STRとATRを統合し、True Range計算方法とスムージング手法を選択可能にした
次世代のボラティリティインジケーター。

🌟 **主要機能:**
1. **TR計算方法の選択**: ATRベース vs STRベース
2. **統合スムージング**: unified_smootherによる多様なスムージング手法
3. **カルマンフィルター統合**: 高値・安値をフィルタリングしてノイズ除去
4. **処理フロー**: ソース価格→カルマンフィルター→TR計算→スムージング

📊 **処理順序:**
1. プライスソース取得
2. カルマンフィルター適用（オプション）
3. True Range計算（ATRまたはSTR方式）
4. スムージング適用

🔧 **パラメータ:**
- tr_method: 'atr' または 'str' - True Range計算方法
- smoother_type: スムージング手法（FRAMA, Super Smoother, Ultimate Smoother, ZLEMA）
- enable_kalman: カルマンフィルター使用フラグ
- kalman_type: カルマンフィルター種別
- period_mode: 'fixed' または 'dynamic' - 期間モード
- cycle_detector_type: サイクル検出器タイプ（動的期間モード用）
"""

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import njit
import traceback


@njit(fastmath=True, cache=True)
def calculate_percentage_values(
    values: np.ndarray,
    close_prices: np.ndarray
) -> np.ndarray:
    """
    金額ベースの値を%ベースに変換する（Numba最適化版）
    
    Args:
        values: 金額ベースの値の配列
        close_prices: 終値の配列
        
    Returns:
        %ベースの値の配列
    """
    length = len(values)
    percentage_values = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if not np.isnan(values[i]) and not np.isnan(close_prices[i]) and close_prices[i] > 0:
            percentage_values[i] = (values[i] / close_prices[i]) * 100.0
    
    return percentage_values


@njit(fastmath=True, cache=True)
def calculate_str_percentile(str_values: np.ndarray, lookback_period: int) -> np.ndarray:
    """
    STRパーセンタイル計算 - 高精度版
    
    Args:
        str_values: STR値の配列
        lookback_period: ルックバック期間
        
    Returns:
        パーセンタイル値の配列（0-1の範囲）
    """
    length = len(str_values)
    percentiles = np.zeros(length, dtype=np.float64)
    
    for i in range(lookback_period, length):
        # 過去のSTR値を取得
        historical_values = str_values[i-lookback_period:i]
        
        # 現在値との比較
        current_value = str_values[i]
        
        # パーセンタイル計算（高精度）
        count_below = 0
        count_equal = 0
        
        for val in historical_values:
            if val < current_value:
                count_below += 1
            elif val == current_value:
                count_equal += 1
        
        # より正確なパーセンタイル計算
        if len(historical_values) > 0:
            percentiles[i] = (count_below + count_equal * 0.5) / len(historical_values)
        else:
            percentiles[i] = 0.5
    
    return percentiles


@njit(fastmath=True, cache=True)
def calculate_volatility_classification(
    str_percentiles: np.ndarray,
    x_atr_values: np.ndarray,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7
) -> tuple:
    """
    パーセンタイルに基づくボラティリティ分類
    
    Args:
        str_percentiles: STRパーセンタイル値
        x_atr_values: X_ATR値
        low_threshold: 低ボラティリティ閾値
        high_threshold: 高ボラティリティ閾値
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ボラティリティ状態, ボラティリティ強度)
    """
    length = len(str_percentiles)
    volatility_state = np.full(length, np.nan, dtype=np.float64)
    volatility_intensity = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if not np.isnan(str_percentiles[i]):
            percentile = str_percentiles[i]
            
            # ボラティリティ状態の分類
            if percentile <= low_threshold:
                volatility_state[i] = 1.0  # 低ボラティリティ
            elif percentile >= high_threshold:
                volatility_state[i] = -1.0  # 高ボラティリティ
            else:
                volatility_state[i] = 0.0  # 中ボラティリティ
            
            # ボラティリティ強度（0-1の範囲で正規化）
            if percentile <= 0.5:
                # 低ボラティリティ側の強度
                volatility_intensity[i] = (0.5 - percentile) / 0.5
            else:
                # 高ボラティリティ側の強度
                volatility_intensity[i] = (percentile - 0.5) / 0.5
    
    return volatility_state, volatility_intensity


@njit(fastmath=True, cache=True)
def calculate_midline_and_volatility_signal(
    x_atr: np.ndarray,
    midline_period: int = 100
) -> tuple:
    """
    ミッドラインとボラティリティ信号を計算する（Numba最適化版）
    
    Args:
        x_atr: X_ATR値の配列
        midline_period: ミッドライン計算期間
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (ミッドライン, ボラティリティ信号)
    """
    length = len(x_atr)
    midline = np.full(length, np.nan, dtype=np.float64)
    volatility_signal = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(midline_period - 1, length):
        # 期間内の最高値と最安値を計算
        period_data = x_atr[i - midline_period + 1:i + 1]
        
        # NaN値を除外
        valid_data = period_data[~np.isnan(period_data)]
        
        if len(valid_data) >= midline_period // 2:
            period_max = np.max(valid_data)
            period_min = np.min(valid_data)
            
            # ミッドライン = (最高値 + 最安値) / 2
            midline[i] = (period_max + period_min) / 2.0
            
            # ボラティリティ信号の判定
            if not np.isnan(x_atr[i]):
                if x_atr[i] > midline[i]:
                    volatility_signal[i] = -1.0  # 高ボラティリティ
                else:
                    volatility_signal[i] = 1.0   # 低ボラティリティ
    
    return midline, volatility_signal

try:
    from ..indicator import Indicator
    from ..price_source import PriceSource
    from ..smoother.unified_smoother import UnifiedSmoother
    from ..kalman.unified_kalman import UnifiedKalman
    # STRとATRのコア関数をインポート
    from ..str import calculate_true_range_values as str_calculate_true_range_values
    from ..atr import calculate_true_range as atr_calculate_true_range
except ImportError:
    # Fallback for standalone execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator import Indicator
    from price_source import PriceSource
    from smoother.unified_smoother import UnifiedSmoother
    from kalman.unified_kalman import UnifiedKalman
    from str import calculate_true_range_values as str_calculate_true_range_values
    from atr import calculate_true_range as atr_calculate_true_range


@dataclass
class XATRResult:
    """X_ATRの計算結果"""
    values: np.ndarray                    # メインのX_ATR値（金額ベース）
    values_percentage: np.ndarray         # X_ATR値（%ベース）
    true_range: np.ndarray               # True Range値（金額ベース）
    true_range_percentage: np.ndarray    # True Range値（%ベース）
    raw_high: np.ndarray                 # 元の高値
    raw_low: np.ndarray                  # 元の安値
    raw_close: np.ndarray                # 元の終値（%計算用）
    filtered_high: Optional[np.ndarray]  # カルマンフィルター適用後の高値（使用時のみ）
    filtered_low: Optional[np.ndarray]   # カルマンフィルター適用後の安値（使用時のみ）
    tr_method: str                       # 使用されたTR計算方法
    smoother_type: str                   # 使用されたスムーサータイプ
    parameters: Dict[str, Any]           # 使用されたパラメータ
    dynamic_periods: Optional[np.ndarray] # 動的期間配列（動的モード時のみ）
    midline: np.ndarray                  # ミッドライン値（金額ベース）
    midline_percentage: np.ndarray       # ミッドライン値（%ベース）
    volatility_signal: np.ndarray        # ボラティリティ信号（1=低ボラ、-1=高ボラ）
    # パーセンタイルベースのボラティリティ分析
    str_percentiles: Optional[np.ndarray]    # STRパーセンタイル値
    volatility_state: Optional[np.ndarray]   # ボラティリティ状態（-1=高、0=中、1=低）
    volatility_intensity: Optional[np.ndarray] # ボラティリティ強度（0-1）


class XATR(Indicator):
    """
    X_ATR（拡張的Average True Range）インジケーター
    
    STRとATRを統合し、TR計算方法とスムージング手法を選択可能にした
    次世代のボラティリティインジケーター。
    
    特徴:
    - TR計算方法の選択（ATRベース vs STRベース）
    - 統合スムージング（unified_smoother）
    - カルマンフィルター統合（オプション）
    - 動的適応期間対応（エーラーズ統合サイクル検出器）
    - 統一されたインターフェース
    """
    
    def __init__(
        self,
        period: float = 12.0,
        tr_method: str = 'str',          # 'atr' または 'str'
        smoother_type: str = 'frama',    # unified_smootherの種別
        src_type: str = 'close',         # プライスソース
        enable_kalman: bool = False,     # カルマンフィルター使用フラグ
        kalman_type: str = 'unscented',  # カルマンフィルター種別
        # 動的適応パラメータ
        period_mode: str = 'fixed',      # 'fixed' または 'dynamic'
        cycle_detector_type: str = 'absolute_ultimate',
        cycle_detector_cycle_part: float = 0.5,
        cycle_detector_max_cycle: int = 55,
        cycle_detector_min_cycle: int = 5,
        cycle_period_multiplier: float = 1.0,
        cycle_detector_period_range: Tuple[int, int] = (5, 120),
        # ミッドラインパラメータ
        midline_period: int = 100,       # ミッドライン計算期間
        # パーセンタイルベースボラティリティ分析パラメータ
        enable_percentile_analysis: bool = True,  # パーセンタイル分析を有効にするか
        percentile_lookback_period: int = 50,     # パーセンタイル計算のルックバック期間
        percentile_low_threshold: float = 0.25,   # 低ボラティリティ閾値
        percentile_high_threshold: float = 0.75,  # 高ボラティリティ閾値
        # スムーサーパラメータ
        smoother_params: Optional[Dict[str, Any]] = None,
        # カルマンフィルターパラメータ
        kalman_params: Optional[Dict[str, Any]] = None
    ):
        """
        コンストラクタ
        
        Args:
            period: スムージング期間
            tr_method: True Range計算方法 ('atr' または 'str')
            smoother_type: スムージング手法
            src_type: プライスソース
            enable_kalman: カルマンフィルター使用フラグ
            kalman_type: カルマンフィルター種別
            
            # 動的適応パラメータ
            period_mode: 期間モード ('fixed' または 'dynamic')
            cycle_detector_type: サイクル検出器タイプ ('hody', 'phac', 'dudi', etc.)
            cycle_detector_cycle_part: サイクル検出器のサイクル部分倍率
            cycle_detector_max_cycle: サイクル検出器の最大サイクル期間
            cycle_detector_min_cycle: サイクル検出器の最小サイクル期間
            cycle_period_multiplier: サイクル期間の乗数
            cycle_detector_period_range: サイクル検出器の周期範囲
            
            smoother_params: スムーサー固有パラメータ
            kalman_params: カルマンフィルター固有パラメータ
        """
        # インディケーター名の設定
        kalman_str = f"_K({kalman_type})" if enable_kalman else ""
        dynamic_str = f"({period_mode})" if period_mode == 'dynamic' else ""
        indicator_name = f"X_ATR({tr_method.upper()}, {smoother_type}, p={period}{dynamic_str}{kalman_str})"
        super().__init__(indicator_name)
        
        # パラメータ検証
        if tr_method.lower() not in ['atr', 'str']:
            raise ValueError(f"無効なtr_method: {tr_method}。'atr' または 'str' を指定してください。")
        
        if period_mode.lower() not in ['fixed', 'dynamic']:
            raise ValueError(f"無効なperiod_mode: {period_mode}。'fixed' または 'dynamic' を指定してください。")
        
        # パラメータ保存
        self.period = period
        self.tr_method = tr_method.lower()
        self.smoother_type = smoother_type.lower()
        self.src_type = src_type.lower()
        self.enable_kalman = enable_kalman
        self.kalman_type = kalman_type.lower()
        
        # 動的適応パラメータ
        self.period_mode = period_mode.lower()
        self.cycle_detector_type = cycle_detector_type
        self.cycle_detector_cycle_part = cycle_detector_cycle_part
        self.cycle_detector_max_cycle = cycle_detector_max_cycle
        self.cycle_detector_min_cycle = cycle_detector_min_cycle
        self.cycle_period_multiplier = cycle_period_multiplier
        self.cycle_detector_period_range = cycle_detector_period_range
        
        # ミッドラインパラメータ
        self.midline_period = midline_period
        
        # パーセンタイルベースボラティリティ分析パラメータ
        self.enable_percentile_analysis = enable_percentile_analysis
        self.percentile_lookback_period = percentile_lookback_period
        self.percentile_low_threshold = percentile_low_threshold
        self.percentile_high_threshold = percentile_high_threshold
        
        # パラメータの初期化
        self.smoother_params = smoother_params or {}
        self.kalman_params = kalman_params or {}
        
        # パラメータ検証
        if self.period <= 0:
            raise ValueError("periodは0より大きい必要があります")
        
        # サイクル検出器の初期化（動的モード時のみ）
        self.cycle_detector = None
        
        if self.period_mode == 'dynamic':
            try:
                # EhlersUnifiedDCのインポート
                from ..cycle.ehlers_unified_dc import EhlersUnifiedDC
                
                self.cycle_detector = EhlersUnifiedDC(
                    detector_type=self.cycle_detector_type,
                    cycle_part=self.cycle_detector_cycle_part,
                    max_cycle=self.cycle_detector_max_cycle,
                    min_cycle=self.cycle_detector_min_cycle,
                    src_type=self.src_type,
                    period_range=self.cycle_detector_period_range
                )
                self.logger.info(f"X_ATR: 動的適応サイクル検出器を初期化: {self.cycle_detector_type}")
                
            except ImportError as e:
                self.logger.error(f"X_ATR: EhlersUnifiedDCのインポートに失敗: {e}")
                self.period_mode = 'fixed'
                self.logger.warning("X_ATR: 動的適応モードの初期化に失敗したため、固定モードにフォールバックしました。")
            except Exception as e:
                self.logger.error(f"X_ATR: サイクル検出器の初期化に失敗: {e}")
                self.period_mode = 'fixed'
                self.logger.warning("X_ATR: 動的適応モードの初期化に失敗したため、固定モードにフォールバックしました。")
        
        # スムーサーの初期化
        smoother_config = {'period': self.period}
        smoother_config.update(self.smoother_params)
        
        # 動的期間対応のパラメータを追加
        if self.period_mode == 'dynamic':
            smoother_config.update({
                'period_mode': self.period_mode,
                'cycle_detector_type': self.cycle_detector_type,
                'cycle_detector_cycle_part': self.cycle_detector_cycle_part,
                'cycle_detector_max_cycle': self.cycle_detector_max_cycle,
                'cycle_detector_min_cycle': self.cycle_detector_min_cycle,
                'cycle_period_multiplier': self.cycle_period_multiplier,
                'cycle_detector_period_range': self.cycle_detector_period_range
            })
        
        self.smoother = UnifiedSmoother(
            smoother_type=self.smoother_type,
            src_type='close',  # TRデータに対して適用
            **smoother_config
        )
        
        # カルマンフィルターの初期化（必要時のみ）
        self.kalman_filter_high = None
        self.kalman_filter_low = None
        
        if self.enable_kalman:
            try:
                # 高値用カルマンフィルター
                self.kalman_filter_high = UnifiedKalman(
                    filter_type=self.kalman_type,
                    src_type='high',
                    **self.kalman_params
                )
                
                # 安値用カルマンフィルター
                self.kalman_filter_low = UnifiedKalman(
                    filter_type=self.kalman_type,
                    src_type='low',
                    **self.kalman_params
                )
                
                self.logger.info(f"カルマンフィルター初期化完了: {self.kalman_type}")
                
            except Exception as e:
                self.logger.error(f"カルマンフィルターの初期化に失敗: {e}")
                self.enable_kalman = False
                self.logger.warning("カルマンフィルターを無効化しました")
        
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
            
            # パラメータシグネチャ
            params_sig = (
                f"{self.tr_method}_{self.smoother_type}_{self.period}_"
                f"{self.period_mode}_{self.cycle_detector_type}_"
                f"{self.enable_kalman}_{self.kalman_type}_"
                f"{hash(str(sorted(self.smoother_params.items())))}_"
                f"{hash(str(sorted(self.kalman_params.items())))}"
            )
            
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.tr_method}_{self.smoother_type}_{self.period}"
    
    def _get_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        動的適応期間を計算する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 動的期間配列
        """
        data_length = len(data) if hasattr(data, '__len__') else 0
        
        # デフォルト値で初期化
        periods = np.full(data_length, self.period, dtype=np.float64)
        
        # 動的適応期間の計算
        if self.period_mode == 'dynamic' and self.cycle_detector is not None:
            try:
                # ドミナントサイクルを計算
                dominant_cycles = self.cycle_detector.calculate(data)
                
                if dominant_cycles is not None and len(dominant_cycles) == data_length:
                    # サイクル期間に乗数を適用
                    adjusted_cycles = dominant_cycles * self.cycle_period_multiplier
                    
                    # サイクル期間を適切な範囲にクリップ
                    periods = np.clip(adjusted_cycles, 
                                     self.cycle_detector_min_cycle, 
                                     self.cycle_detector_max_cycle)
                    
                    self.logger.debug(f"X_ATR動的期間計算完了 - 期間範囲: [{np.min(periods):.1f}-{np.max(periods):.1f}]")
                else:
                    self.logger.warning("X_ATR: ドミナントサイクルの計算結果が無効です。固定期間を使用します。")
                    
            except Exception as e:
                self.logger.error(f"X_ATR: 動的期間計算中にエラー: {e}")
                # エラー時は固定期間を使用
        
        return periods
    
    def _apply_kalman_filtering(
        self, 
        data: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        高値と安値にカルマンフィルターを適用
        
        Args:
            data: 価格データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: フィルタリングされた高値と安値
        """
        try:
            # 高値フィルタリング
            high_result = self.kalman_filter_high.calculate(data)
            filtered_high = high_result.values
            
            # 安値フィルタリング
            low_result = self.kalman_filter_low.calculate(data)
            filtered_low = low_result.values
            
            self.logger.debug("カルマンフィルター適用完了")
            return filtered_high, filtered_low
            
        except Exception as e:
            self.logger.error(f"カルマンフィルター適用中にエラー: {e}")
            # フォールバック: 元の高値・安値を返す
            if isinstance(data, pd.DataFrame):
                return data['high'].values, data['low'].values
            else:
                return data[:, 1], data[:, 2]  # high, low
    
    def _calculate_true_range(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> np.ndarray:
        """
        指定された方法でTrue Rangeを計算
        
        Args:
            high: 高値配列
            low: 安値配列
            close: 終値配列
            
        Returns:
            True Range配列
        """
        if self.tr_method == 'str':
            # STR方式のTrue Range計算
            true_high, true_low, true_range = str_calculate_true_range_values(high, low, close)
            return true_range
        else:
            # ATR方式のTrue Range計算
            return atr_calculate_true_range(high, low, close)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> XATRResult:
        """
        X_ATRを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                必要なカラム: high, low, close
        
        Returns:
            XATRResult: X_ATRの計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return XATRResult(
                    values=cached_result.values.copy(),
                    values_percentage=cached_result.values_percentage.copy(),
                    true_range=cached_result.true_range.copy(),
                    true_range_percentage=cached_result.true_range_percentage.copy(),
                    raw_high=cached_result.raw_high.copy(),
                    raw_low=cached_result.raw_low.copy(),
                    raw_close=cached_result.raw_close.copy(),
                    filtered_high=cached_result.filtered_high.copy() if cached_result.filtered_high is not None else None,
                    filtered_low=cached_result.filtered_low.copy() if cached_result.filtered_low is not None else None,
                    tr_method=cached_result.tr_method,
                    smoother_type=cached_result.smoother_type,
                    parameters=cached_result.parameters.copy(),
                    dynamic_periods=cached_result.dynamic_periods.copy() if cached_result.dynamic_periods is not None else None,
                    midline=cached_result.midline.copy(),
                    midline_percentage=cached_result.midline_percentage.copy(),
                    volatility_signal=cached_result.volatility_signal.copy(),
                    str_percentiles=cached_result.str_percentiles.copy() if cached_result.str_percentiles is not None else None,
                    volatility_state=cached_result.volatility_state.copy() if cached_result.volatility_state is not None else None,
                    volatility_intensity=cached_result.volatility_intensity.copy() if cached_result.volatility_intensity is not None else None
                )
            
            # データの準備
            if isinstance(data, pd.DataFrame):
                required_cols = ['high', 'low', 'close']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"必要なカラムが不足しています: {missing_cols}")
                
                raw_high = data['high'].to_numpy()
                raw_low = data['low'].to_numpy()
                close = data['close'].to_numpy()
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                raw_high = data[:, 1]    # high
                raw_low = data[:, 2]     # low
                close = data[:, 3]       # close
            
            # NumPy配列に変換
            raw_high = np.asarray(raw_high, dtype=np.float64)
            raw_low = np.asarray(raw_low, dtype=np.float64)
            close = np.asarray(close, dtype=np.float64)
            
            # データ長の検証
            data_length = len(close)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            # カルマンフィルター適用（オプション）
            if self.enable_kalman and self.kalman_filter_high and self.kalman_filter_low:
                filtered_high, filtered_low = self._apply_kalman_filtering(data)
                # フィルタリング後の値を使用してTR計算
                working_high = filtered_high
                working_low = filtered_low
            else:
                # 元の値を使用
                working_high = raw_high
                working_low = raw_low
                filtered_high = None
                filtered_low = None
            
            # True Range計算
            true_range = self._calculate_true_range(working_high, working_low, close)
            
            # 動的期間の計算
            dynamic_periods = self._get_dynamic_periods(data)
            
            # True RangeをDataFrame形式に変換してスムーサーに渡す
            # スムーサーがFRAMAの場合、高値・安値も含める必要がある
            tr_df = pd.DataFrame({
                'open': true_range,
                'high': true_range,
                'low': true_range,
                'close': true_range,
                'volume': np.ones_like(true_range)
            })
            
            # スムージング適用
            smoother_result = self.smoother.calculate(tr_df)
            x_atr_values = smoother_result.values
            
            # ミッドラインとボラティリティ信号の計算
            midline, volatility_signal = calculate_midline_and_volatility_signal(
                x_atr_values, self.midline_period
            )
            
            # パーセンタイルベースのボラティリティ分析（オプション）
            str_percentiles = None
            volatility_state = None
            volatility_intensity = None
            
            if self.enable_percentile_analysis:
                try:
                    # STRパーセンタイルを計算（X_ATR値を使用）
                    str_percentiles = calculate_str_percentile(
                        x_atr_values, self.percentile_lookback_period
                    )
                    
                    # ボラティリティ分類を計算
                    volatility_state, volatility_intensity = calculate_volatility_classification(
                        str_percentiles, x_atr_values,
                        self.percentile_low_threshold, self.percentile_high_threshold
                    )
                    
                    self.logger.debug("パーセンタイルベースボラティリティ分析完了")
                    
                except Exception as e:
                    self.logger.warning(f"パーセンタイル分析中にエラー: {e}")
                    # エラー時はNoneのまま
            else:
                self.logger.debug("パーセンタイル分析はスキップされました")
            
            # %ベース値の計算
            x_atr_percentage = calculate_percentage_values(x_atr_values, close)
            true_range_percentage = calculate_percentage_values(true_range, close)
            midline_percentage = calculate_percentage_values(midline, close)
            
            # 結果の保存
            result = XATRResult(
                values=x_atr_values.copy(),
                values_percentage=x_atr_percentage.copy(),
                true_range=true_range.copy(),
                true_range_percentage=true_range_percentage.copy(),
                raw_high=raw_high.copy(),
                raw_low=raw_low.copy(),
                raw_close=close.copy(),
                filtered_high=filtered_high.copy() if filtered_high is not None else None,
                filtered_low=filtered_low.copy() if filtered_low is not None else None,
                tr_method=self.tr_method,
                smoother_type=self.smoother_type,
                parameters={
                    'period': self.period,
                    'tr_method': self.tr_method,
                    'smoother_type': self.smoother_type,
                    'enable_kalman': self.enable_kalman,
                    'kalman_type': self.kalman_type,
                    'period_mode': self.period_mode,
                    'cycle_detector_type': self.cycle_detector_type,
                    'cycle_detector_cycle_part': self.cycle_detector_cycle_part,
                    'cycle_detector_max_cycle': self.cycle_detector_max_cycle,
                    'cycle_detector_min_cycle': self.cycle_detector_min_cycle,
                    'cycle_period_multiplier': self.cycle_period_multiplier,
                    'cycle_detector_period_range': self.cycle_detector_period_range,
                    'midline_period': self.midline_period,
                    'enable_percentile_analysis': self.enable_percentile_analysis,
                    'percentile_lookback_period': self.percentile_lookback_period,
                    'percentile_low_threshold': self.percentile_low_threshold,
                    'percentile_high_threshold': self.percentile_high_threshold,
                    'smoother_params': self.smoother_params.copy(),
                    'kalman_params': self.kalman_params.copy()
                },
                dynamic_periods=dynamic_periods.copy() if self.period_mode == 'dynamic' else None,
                midline=midline.copy(),
                midline_percentage=midline_percentage.copy(),
                volatility_signal=volatility_signal.copy(),
                str_percentiles=str_percentiles.copy() if str_percentiles is not None else None,
                volatility_state=volatility_state.copy() if volatility_state is not None else None,
                volatility_intensity=volatility_intensity.copy() if volatility_intensity is not None else None
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            # 基底クラス用の値設定
            self._values = x_atr_values
            
            self.logger.debug(f"X_ATR計算完了 - TR方法: {self.tr_method}, スムーサー: {self.smoother_type}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"X_ATR計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            data_length = len(data) if hasattr(data, '__len__') else 0
            return XATRResult(
                values=np.full(data_length, np.nan),
                values_percentage=np.full(data_length, np.nan),
                true_range=np.full(data_length, np.nan),
                true_range_percentage=np.full(data_length, np.nan),
                raw_high=np.full(data_length, np.nan),
                raw_low=np.full(data_length, np.nan),
                raw_close=np.full(data_length, np.nan),
                filtered_high=None,
                filtered_low=None,
                tr_method=self.tr_method,
                smoother_type=self.smoother_type,
                parameters={},
                dynamic_periods=None,
                midline=np.full(data_length, np.nan),
                midline_percentage=np.full(data_length, np.nan),
                volatility_signal=np.full(data_length, np.nan),
                str_percentiles=None,
                volatility_state=None,
                volatility_intensity=None
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """X_ATR値のみを取得する（後方互換性のため）"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.values.copy()
    
    def get_true_range(self) -> Optional[np.ndarray]:
        """True Range値を取得する"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.true_range.copy()
    
    def get_filtered_prices(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """カルマンフィルター適用後の高値・安値を取得する"""
        if not self._result_cache or not self._cache_keys:
            return None, None
        
        result = self._result_cache[self._cache_keys[-1]]
        filtered_high = result.filtered_high.copy() if result.filtered_high is not None else None
        filtered_low = result.filtered_low.copy() if result.filtered_low is not None else None
        return filtered_high, filtered_low
    
    def get_dynamic_periods(self) -> Optional[np.ndarray]:
        """動的期間配列を取得する"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.dynamic_periods.copy() if result.dynamic_periods is not None else None
    
    def get_dynamic_periods_info(self) -> Dict[str, Any]:
        """動的適応期間の情報を取得する"""
        info = {
            'period_mode': self.period_mode,
            'cycle_detector_available': self.cycle_detector is not None
        }
        
        # サイクル検出器の情報
        if self.cycle_detector is not None:
            info.update({
                'cycle_detector_type': self.cycle_detector_type,
                'cycle_detector_cycle_part': self.cycle_detector_cycle_part,
                'cycle_detector_max_cycle': self.cycle_detector_max_cycle,
                'cycle_detector_min_cycle': self.cycle_detector_min_cycle,
                'cycle_period_multiplier': self.cycle_period_multiplier,
                'cycle_detector_period_range': self.cycle_detector_period_range
            })
        
        return info
    
    def get_midline(self) -> Optional[np.ndarray]:
        """ミッドライン値を取得する"""
        result = self._get_latest_result()
        return result.midline.copy() if result else None
    
    def get_volatility_signal(self) -> Optional[np.ndarray]:
        """ボラティリティ信号を取得する"""
        result = self._get_latest_result()
        return result.volatility_signal.copy() if result else None
    
    def get_values_percentage(self) -> Optional[np.ndarray]:
        """X_ATR値（%ベース）を取得する"""
        result = self._get_latest_result()
        return result.values_percentage.copy() if result else None
    
    def get_true_range_percentage(self) -> Optional[np.ndarray]:
        """True Range値（%ベース）を取得する"""
        result = self._get_latest_result()
        return result.true_range_percentage.copy() if result else None
    
    def get_midline_percentage(self) -> Optional[np.ndarray]:
        """ミッドライン値（%ベース）を取得する"""
        result = self._get_latest_result()
        return result.midline_percentage.copy() if result else None
    
    def get_str_percentiles(self) -> Optional[np.ndarray]:
        """STRパーセンタイル値を取得する"""
        result = self._get_latest_result()
        return result.str_percentiles.copy() if result and result.str_percentiles is not None else None
    
    def get_volatility_state(self) -> Optional[np.ndarray]:
        """ボラティリティ状態を取得する（-1=高、0=中、1=低）"""
        result = self._get_latest_result()
        return result.volatility_state.copy() if result and result.volatility_state is not None else None
    
    def get_volatility_intensity(self) -> Optional[np.ndarray]:
        """ボラティリティ強度を取得する（0-1の範囲）"""
        result = self._get_latest_result()
        return result.volatility_intensity.copy() if result and result.volatility_intensity is not None else None
    
    def get_percentile_analysis_summary(self) -> Dict[str, Any]:
        """パーセンタイル分析の要約情報を取得する"""
        result = self._get_latest_result()
        if not result:
            return {}
        
        summary = {
            'percentile_analysis_enabled': self.enable_percentile_analysis,
            'lookback_period': self.percentile_lookback_period,
            'low_threshold': self.percentile_low_threshold,
            'high_threshold': self.percentile_high_threshold
        }
        
        if result.str_percentiles is not None:
            percentiles = result.str_percentiles
            valid_percentiles = percentiles[~np.isnan(percentiles)]
            
            if len(valid_percentiles) > 0:
                summary.update({
                    'percentile_mean': np.mean(valid_percentiles),
                    'percentile_std': np.std(valid_percentiles),
                    'percentile_min': np.min(valid_percentiles),
                    'percentile_max': np.max(valid_percentiles),
                    'current_percentile': percentiles[-1] if not np.isnan(percentiles[-1]) else None
                })
        
        if result.volatility_state is not None:
            state = result.volatility_state
            valid_state = state[~np.isnan(state)]
            
            if len(valid_state) > 0:
                # 各状態の分布を計算
                low_vol_count = np.sum(valid_state == 1.0)
                mid_vol_count = np.sum(valid_state == 0.0)
                high_vol_count = np.sum(valid_state == -1.0)
                total_count = len(valid_state)
                
                summary.update({
                    'volatility_distribution': {
                        'low': low_vol_count / total_count,
                        'medium': mid_vol_count / total_count,
                        'high': high_vol_count / total_count
                    },
                    'current_volatility_state': state[-1] if not np.isnan(state[-1]) else None
                })
        
        return summary
    
    def _get_latest_result(self) -> Optional[XATRResult]:
        """最新の結果を取得"""
        if not self._result_cache:
            return None
        
        if self._cache_keys:
            return self._result_cache[self._cache_keys[-1]]
        else:
            return next(iter(self._result_cache.values()))
    
    def get_configuration(self) -> Dict[str, Any]:
        """現在の設定情報を取得する"""
        return {
            'period': self.period,
            'tr_method': self.tr_method,
            'smoother_type': self.smoother_type,
            'src_type': self.src_type,
            'enable_kalman': self.enable_kalman,
            'kalman_type': self.kalman_type,
            'period_mode': self.period_mode,
            'cycle_detector_type': self.cycle_detector_type,
            'cycle_detector_cycle_part': self.cycle_detector_cycle_part,
            'cycle_detector_max_cycle': self.cycle_detector_max_cycle,
            'cycle_detector_min_cycle': self.cycle_detector_min_cycle,
            'cycle_period_multiplier': self.cycle_period_multiplier,
            'cycle_detector_period_range': self.cycle_detector_period_range,
            'midline_period': self.midline_period,
            'smoother_params': self.smoother_params.copy(),
            'kalman_params': self.kalman_params.copy()
        }
    
    def reset(self) -> None:
        """インディケーターの状態をリセットする"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        
        # サブコンポーネントのリセット
        if hasattr(self.smoother, 'reset'):
            self.smoother.reset()
        
        if self.kalman_filter_high and hasattr(self.kalman_filter_high, 'reset'):
            self.kalman_filter_high.reset()
        
        if self.kalman_filter_low and hasattr(self.kalman_filter_low, 'reset'):
            self.kalman_filter_low.reset()
        
        if self.cycle_detector and hasattr(self.cycle_detector, 'reset'):
            self.cycle_detector.reset()


# 便利関数
def calculate_x_atr(
    data: Union[pd.DataFrame, np.ndarray],
    period: float = 20.0,
    tr_method: str = 'atr',
    smoother_type: str = 'frama',
    enable_kalman: bool = False,
    kalman_type: str = 'unscented',
    period_mode: str = 'fixed',
    cycle_detector_type: str = 'absolute_ultimate',
    midline_period: int = 100,
    enable_percentile_analysis: bool = True,
    percentile_lookback_period: int = 50,
    percentile_low_threshold: float = 0.25,
    percentile_high_threshold: float = 0.75,
    **kwargs
) -> np.ndarray:
    """
    X_ATRの計算（便利関数）
    
    Args:
        data: 価格データ
        period: スムージング期間
        tr_method: True Range計算方法
        smoother_type: スムージング手法
        enable_kalman: カルマンフィルター使用フラグ
        kalman_type: カルマンフィルター種別
        period_mode: 期間モード ('fixed' または 'dynamic')
        cycle_detector_type: サイクル検出器タイプ
        midline_period: ミッドライン計算期間
        enable_percentile_analysis: パーセンタイル分析を有効にするか
        percentile_lookback_period: パーセンタイル計算のルックバック期間
        percentile_low_threshold: 低ボラティリティ閾値
        percentile_high_threshold: 高ボラティリティ閾値
        **kwargs: 追加パラメータ
        
    Returns:
        X_ATR値の配列
    """
    x_atr = XATR(
        period=period,
        tr_method=tr_method,
        smoother_type=smoother_type,
        enable_kalman=enable_kalman,
        kalman_type=kalman_type,
        period_mode=period_mode,
        cycle_detector_type=cycle_detector_type,
        midline_period=midline_period,
        enable_percentile_analysis=enable_percentile_analysis,
        percentile_lookback_period=percentile_lookback_period,
        percentile_low_threshold=percentile_low_threshold,
        percentile_high_threshold=percentile_high_threshold,
        **kwargs
    )
    result = x_atr.calculate(data)
    return result.values


if __name__ == "__main__":
    """直接実行時のテスト"""
    try:
        import matplotlib.pyplot as plt
        from datetime import datetime, timedelta
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False
    
    print("=== X_ATRのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    trend = 0.001
    volatility = 0.02
    
    prices = [base_price]
    for i in range(1, length):
        change = trend + np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, volatility * close * 0.5))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, volatility * close * 0.2)
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
    
    # 各設定でのテスト
    test_configs = [
        {'tr_method': 'atr', 'smoother_type': 'frama', 'enable_kalman': False, 'period_mode': 'fixed'},
        {'tr_method': 'str', 'smoother_type': 'frama', 'enable_kalman': False, 'period_mode': 'fixed'},
        {'tr_method': 'atr', 'smoother_type': 'zero_lag_ema', 'enable_kalman': False, 'period_mode': 'fixed'},
        {'tr_method': 'atr', 'smoother_type': 'frama', 'enable_kalman': True, 'kalman_type': 'unscented', 'period_mode': 'fixed'},
        {'tr_method': 'atr', 'smoother_type': 'ultimate_smoother', 'enable_kalman': False, 'period_mode': 'dynamic', 'cycle_detector_type': 'absolute_ultimate'},
    ]
    
    results = {}
    
    for i, config in enumerate(test_configs):
        try:
            print(f"\n設定 {i+1}: {config}")
            x_atr = XATR(period=20.0, **config)
            result = x_atr.calculate(df)
            
            mean_value = np.nanmean(result.values)
            valid_count = np.sum(~np.isnan(result.values))
            
            results[f"config_{i+1}"] = result
            
            print(f"  平均X_ATR: {mean_value:.4f}")
            print(f"  有効値数: {valid_count}/{len(df)}")
            print(f"  カルマンフィルター使用: {config.get('enable_kalman', False)}")
            print(f"  期間モード: {config.get('period_mode', 'fixed')}")
            
            # 動的期間モードの場合は期間情報も表示
            if config.get('period_mode') == 'dynamic':
                dynamic_periods = result.dynamic_periods
                if dynamic_periods is not None:
                    print(f"  動的期間範囲: {np.min(dynamic_periods):.1f} - {np.max(dynamic_periods):.1f}")
                else:
                    print(f"  動的期間: 計算されていません")
            
        except Exception as e:
            print(f"  エラー: {e}")
    
    print("\n=== テスト完了 ===")