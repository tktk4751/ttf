#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Zero Lag EMA (ZLEMA) - ゼロラグ指数移動平均** 🎯

ゼロラグ指数移動平均線の実装：
- 従来のEMAの遅延を除去する技術
- より高速な価格変動への反応
- トレンドフォロー戦略に最適化
- プライスソース対応とパフォーマンス最適化

🌟 **ZLEMAの特徴:**
1. **ゼロラグ設計**: 価格変動への即座の反応
2. **EMAベース**: 指数移動平均の高い精度
3. **ノイズ除去**: スムージング効果を維持
4. **多様なプライスソース**: close, hlc3, hl2, ohlc4など対応

📊 **計算方法:**
1. LagReducedData = 2 * Price - EMA(Price, period)
2. ZLEMA = EMA(LagReducedData, period)

🔬 **使用例:**
- 短期トレンド検出
- エントリー・エグジットシグナル
- 他の指標との組み合わせ
"""

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
import pandas as pd
from numba import njit
import traceback

try:
    from ..indicator import Indicator
    from ..price_source import PriceSource
except ImportError:
    # Fallback for potential execution context issues
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicator import Indicator
    from price_source import PriceSource

# 条件付きインポート（動的期間用）
try:
    from ..cycle.ehlers_unified_dc import EhlersUnifiedDC
    EHLERS_UNIFIED_DC_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行
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


@dataclass
class ZLEMAResult:
    """ゼロラグEMAの計算結果"""
    values: np.ndarray           # ZLEMA値
    ema_values: np.ndarray       # 基本EMA値
    lag_reduced_data: np.ndarray # ラグ除去データ
    raw_values: np.ndarray       # 元の価格データ


@njit(fastmath=True, cache=True)
def calculate_zlema_core(prices: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ゼロラグEMAのコア計算関数
    
    Args:
        prices: 価格データ
        period: EMA期間
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ZLEMA値、EMA値、ラグ除去データ
    """
    length = len(prices)
    
    if length < period:
        return (np.full(length, np.nan), 
                np.full(length, np.nan), 
                np.full(length, np.nan))
    
    # EMAの平滑化定数
    alpha = 2.0 / (period + 1.0)
    
    # 結果配列の初期化
    ema_values = np.zeros(length, dtype=np.float64)
    lag_reduced_data = np.zeros(length, dtype=np.float64)
    zlema_values = np.zeros(length, dtype=np.float64)
    
    # NaNで初期化
    for i in range(length):
        ema_values[i] = np.nan
        lag_reduced_data[i] = np.nan
        zlema_values[i] = np.nan
    
    # EMAの初期値設定 (SMAで初期化)
    if length >= period:
        # 最初のperiod分のSMAを計算
        sma_sum = 0.0
        for i in range(period):
            sma_sum += prices[i]
        
        initial_ema = sma_sum / period
        ema_values[period - 1] = initial_ema
        
        # EMAの計算（period以降）
        for i in range(period, length):
            ema_values[i] = alpha * prices[i] + (1.0 - alpha) * ema_values[i - 1]
        
        # ラグ除去データの計算
        for i in range(period - 1, length):
            lag_reduced_data[i] = 2.0 * prices[i] - ema_values[i]
        
        # ZLEMAの計算
        # 最初の値はラグ除去データと同じ
        zlema_values[period - 1] = lag_reduced_data[period - 1]
        
        # 以降はラグ除去データのEMAを計算
        for i in range(period, length):
            zlema_values[i] = alpha * lag_reduced_data[i] + (1.0 - alpha) * zlema_values[i - 1]
    
    return zlema_values, ema_values, lag_reduced_data


@njit(fastmath=True, cache=True)
def calculate_fast_zlema(prices: np.ndarray, period: int, 
                        fast_alpha: Optional[float] = None) -> np.ndarray:
    """
    高速ゼロラグEMA計算（最適化版）
    
    Args:
        prices: 価格データ
        period: EMA期間
        fast_alpha: カスタムアルファ値（指定時は高速化）
        
    Returns:
        ZLEMA値
    """
    length = len(prices)
    
    if length < 2:
        return np.full(length, np.nan)
    
    # アルファ値の設定
    if fast_alpha is None:
        alpha = 2.0 / (period + 1.0)
    else:
        alpha = fast_alpha
    
    zlema = np.zeros(length, dtype=np.float64)
    ema = np.zeros(length, dtype=np.float64)
    
    # 初期値設定
    zlema[0] = prices[0]
    ema[0] = prices[0]
    
    for i in range(1, length):
        # EMAの更新
        ema[i] = alpha * prices[i] + (1.0 - alpha) * ema[i - 1]
        
        # ラグ除去とZLEMAの計算
        lag_reduced = 2.0 * prices[i] - ema[i]
        zlema[i] = alpha * lag_reduced + (1.0 - alpha) * zlema[i - 1]
    
    return zlema


@njit(fastmath=True, cache=True)
def calculate_zlema_dynamic_core(
    prices: np.ndarray, 
    period: int, 
    dynamic_periods: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    動的期間対応ゼロラグEMAのコア計算関数
    
    Args:
        prices: 価格データ
        period: 基本EMA期間
        dynamic_periods: 動的期間配列（オプション）
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: ZLEMA値、EMA値、ラグ除去データ
    """
    length = len(prices)
    
    if length < 2:
        return (np.full(length, np.nan), 
                np.full(length, np.nan), 
                np.full(length, np.nan))
    
    # 結果配列の初期化
    ema_values = np.zeros(length, dtype=np.float64)
    lag_reduced_data = np.zeros(length, dtype=np.float64)
    zlema_values = np.zeros(length, dtype=np.float64)
    
    # 初期値設定
    ema_values[0] = prices[0]
    lag_reduced_data[0] = prices[0]
    zlema_values[0] = prices[0]
    
    for i in range(1, length):
        # 動的期間または固定期間を使用
        current_period = period
        if dynamic_periods is not None and i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
            current_period = max(2, min(int(dynamic_periods[i]), 50))  # 2-50期間に制限
        
        # アルファ値の計算
        alpha = 2.0 / (current_period + 1.0)
        
        # EMAの更新
        ema_values[i] = alpha * prices[i] + (1.0 - alpha) * ema_values[i - 1]
        
        # ラグ除去データの計算
        lag_reduced_data[i] = 2.0 * prices[i] - ema_values[i]
        
        # ZLEMAの計算
        zlema_values[i] = alpha * lag_reduced_data[i] + (1.0 - alpha) * zlema_values[i - 1]
    
    return zlema_values, ema_values, lag_reduced_data


class ZeroLagEMA(Indicator):
    """
    ゼロラグ指数移動平均（ZLEMA）
    
    従来のEMAの遅延を除去した指数移動平均：
    - より高速な価格変動への反応
    - スムージング効果を維持
    - プライスソース対応
    - Numba最適化によるパフォーマンス向上
    """
    
    def __init__(
        self,
        period: int = 21,
        src_type: str = 'close',
        fast_mode: bool = False,
        custom_alpha: Optional[float] = None,
        # 動的期間パラメータ
        period_mode: str = 'fixed',
        cycle_detector_type: str = 'hody_e',
        lp_period: int = 13,
        hp_period: int = 124,
        cycle_part: float = 0.5,
        max_cycle: int = 124,
        min_cycle: int = 13,
        max_output: int = 124,
        min_output: int = 13
    ):
        """
        コンストラクタ
        
        Args:
            period: EMA期間（デフォルト: 21）
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open')
            fast_mode: 高速モード（True: 簡略化計算、False: 詳細計算）
            custom_alpha: カスタムアルファ値（fast_mode時に使用）
            period_mode: 期間モード ('fixed' または 'dynamic')
            cycle_detector_type: サイクル検出器タイプ
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
        """
        # 動的期間文字列の作成
        dynamic_str = f"_dynamic({cycle_detector_type})" if period_mode == 'dynamic' else ""
        
        indicator_name = f"ZLEMA(period={period}, src={src_type}{dynamic_str})"
        super().__init__(indicator_name)
        
        # パラメータ検証
        if period < 1:
            raise ValueError("periodは1以上である必要があります")
        
        self.period = period
        self.src_type = src_type.lower()
        self.fast_mode = fast_mode
        self.custom_alpha = custom_alpha
        
        # 動的期間パラメータ
        self.period_mode = period_mode.lower()
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        
        # 動的期間検証
        if self.period_mode not in ['fixed', 'dynamic']:
            raise ValueError(f"無効な期間モード: {period_mode}")
        
        # ドミナントサイクル検出器の初期化
        self.dc_detector = None
        self._last_dc_values = None
        if self.period_mode == 'dynamic' and EHLERS_UNIFIED_DC_AVAILABLE:
            try:
                self.dc_detector = EhlersUnifiedDC(
                    detector_type=self.cycle_detector_type,
                    cycle_part=self.cycle_part,
                    max_cycle=self.max_cycle,
                    min_cycle=self.min_cycle,
                    max_output=self.max_output,
                    min_output=self.min_output,
                    src_type=self.src_type,
                    lp_period=self.lp_period,
                    hp_period=self.hp_period
                )
            except Exception as e:
                self.logger.warning(f"ドミナントサイクル検出器の初期化に失敗しました: {e}")
                self.period_mode = 'fixed'
        elif self.period_mode == 'dynamic' and not EHLERS_UNIFIED_DC_AVAILABLE:
            self.logger.warning("EhlersUnifiedDCが利用できません。固定期間モードに変更します。")
            self.period_mode = 'fixed'
        
        # ソースタイプの検証
        valid_sources = ['close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open']
        if self.src_type not in valid_sources:
            raise ValueError(f"無効なソースタイプ: {src_type}")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 5
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
            
            params_sig = f"{self.period}_{self.src_type}_{self.fast_mode}_{self.custom_alpha}_{self.period_mode}_{self.cycle_detector_type}"
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.period}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ZLEMAResult:
        """
        ゼロラグEMAを計算
        
        Args:
            data: 価格データ
            
        Returns:
            ZLEMAResult: 計算結果
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
                return ZLEMAResult(
                    values=cached_result.values.copy(),
                    ema_values=cached_result.ema_values.copy(),
                    lag_reduced_data=cached_result.lag_reduced_data.copy(),
                    raw_values=cached_result.raw_values.copy()
                )
            
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            data_length = len(src_prices)
            
            if data_length < 2:
                return self._create_empty_result(data_length, src_prices)
            
            # 動的期間の計算（オプション）
            dynamic_periods = None
            if self.period_mode == 'dynamic' and self.dc_detector is not None:
                try:
                    dc_result = self.dc_detector.calculate(data)
                    if dc_result is not None:
                        dynamic_periods = np.asarray(dc_result, dtype=np.float64)
                        self._last_dc_values = dynamic_periods.copy()
                except Exception as e:
                    self.logger.warning(f"ドミナントサイクル検出に失敗しました: {e}")
                    # フォールバック: 前回の値を使用
                    if self._last_dc_values is not None:
                        dynamic_periods = self._last_dc_values
            
            # ZLEMA計算の実行
            if self.fast_mode:
                # 高速モード（動的期間非対応）
                zlema_values = calculate_fast_zlema(
                    src_prices, 
                    self.period, 
                    self.custom_alpha
                )
                # 簡略化結果
                ema_values = np.full(data_length, np.nan)
                lag_reduced_data = np.full(data_length, np.nan)
            else:
                # 詳細モード
                if self.period_mode == 'dynamic' and dynamic_periods is not None:
                    # 動的期間対応版を使用
                    zlema_values, ema_values, lag_reduced_data = calculate_zlema_dynamic_core(
                        src_prices, self.period, dynamic_periods
                    )
                else:
                    # 固定期間版を使用
                    zlema_values, ema_values, lag_reduced_data = calculate_zlema_core(
                        src_prices, 
                        self.period
                    )
            
            # 結果の作成
            result = ZLEMAResult(
                values=zlema_values.copy(),
                ema_values=ema_values.copy(),
                lag_reduced_data=lag_reduced_data.copy(),
                raw_values=src_prices.copy()
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            # 基底クラス用の値設定
            self._values = zlema_values
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ZLEMA計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                return self._create_empty_result(len(src_prices), src_prices)
            else:
                return self._create_empty_result(0, np.array([]))
    
    def _create_empty_result(self, length: int, raw_prices: np.ndarray) -> ZLEMAResult:
        """空の結果を作成"""
        return ZLEMAResult(
            values=np.full(length, np.nan),
            ema_values=np.full(length, np.nan),
            lag_reduced_data=np.full(length, np.nan),
            raw_values=raw_prices
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """ZLEMA値を取得"""
        if not self._result_cache:
            return None
        
        result = self._get_latest_result()
        return result.values.copy() if result else None
    
    def get_ema_values(self) -> Optional[np.ndarray]:
        """基本EMA値を取得"""
        result = self._get_latest_result()
        return result.ema_values.copy() if result else None
    
    def get_lag_reduced_data(self) -> Optional[np.ndarray]:
        """ラグ除去データを取得"""
        result = self._get_latest_result()
        return result.lag_reduced_data.copy() if result else None
    
    def get_raw_values(self) -> Optional[np.ndarray]:
        """元の価格データを取得"""
        result = self._get_latest_result()
        return result.raw_values.copy() if result else None
    
    def _get_latest_result(self) -> Optional[ZLEMAResult]:
        """最新の結果を取得"""
        if not self._result_cache:
            return None
        
        if self._cache_keys:
            return self._result_cache[self._cache_keys[-1]]
        else:
            return next(iter(self._result_cache.values()))
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        self._last_dc_values = None
        if self.dc_detector and hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset()


# 便利な関数
def zlema(data: Union[pd.DataFrame, np.ndarray], period: int = 21, src_type: str = 'close') -> np.ndarray:
    """
    ゼロラグEMAの計算（便利関数）
    
    Args:
        data: 価格データ
        period: EMA期間
        src_type: 価格ソース
        
    Returns:
        ZLEMA値
    """
    indicator = ZeroLagEMA(period=period, src_type=src_type)
    result = indicator.calculate(data)
    return result.values


def fast_zlema(data: Union[pd.DataFrame, np.ndarray], period: int = 21, 
               src_type: str = 'close', alpha: Optional[float] = None) -> np.ndarray:
    """
    高速ゼロラグEMAの計算（便利関数）
    
    Args:
        data: 価格データ
        period: EMA期間
        src_type: 価格ソース
        alpha: カスタムアルファ値
        
    Returns:
        ZLEMA値
    """
    indicator = ZeroLagEMA(period=period, src_type=src_type, fast_mode=True, custom_alpha=alpha)
    result = indicator.calculate(data)
    return result.values