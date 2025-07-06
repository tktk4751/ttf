#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Union, Optional, NamedTuple
from numba import jit
import traceback

try:
    from .indicator import Indicator
    from .atr import ATR
except ImportError:
    # Fallback for potential execution context issues
    print("Warning: Could not import from relative path. Assuming base classes are available.")
    class Indicator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    from atr import ATR


class VolatilityRegimeResult(NamedTuple):
    """ボラティリティレジーム判定結果"""
    values: np.ndarray          # ボラティリティレジーム値 (1=高ボラ, -1=低ボラ, 0=中性)
    short_atr: np.ndarray       # 短期ATR値
    long_atr: np.ndarray        # 長期ATR値
    volatility_ratio: np.ndarray # ボラティリティ比率 (短期ATR / 長期ATR)
    trend_signals: np.ndarray   # トレンド信号 (1=up, -1=down, 0=range)
    current_regime: str         # 現在のボラティリティレジーム ('high', 'low', 'neutral')
    current_regime_value: int   # 現在のレジーム値 (1, -1, 0)


@jit(nopython=True, cache=True)
def calculate_volatility_regime_signals(
    short_atr: np.ndarray,
    long_atr: np.ndarray,
    threshold_high: float = 1.2,
    threshold_low: float = 0.8
) -> tuple:
    """
    ボラティリティレジーム信号を計算する（Numba JIT）
    
    Args:
        short_atr: 短期ATR値の配列
        long_atr: 長期ATR値の配列  
        threshold_high: 高ボラティリティ判定の閾値（短期/長期の比率）
        threshold_low: 低ボラティリティ判定の閾値（短期/長期の比率）
    
    Returns:
        tuple: (regime_signals, volatility_ratio)
               regime_signals: 1=高ボラ, -1=低ボラ, 0=中性
               volatility_ratio: 短期ATR / 長期ATR
    """
    length = len(short_atr)
    regime_signals = np.zeros(length, dtype=np.int8)
    volatility_ratio = np.full(length, np.nan)
    
    for i in range(length):
        if not np.isnan(short_atr[i]) and not np.isnan(long_atr[i]) and long_atr[i] > 0:
            ratio = short_atr[i] / long_atr[i]
            volatility_ratio[i] = ratio
            
            if ratio >= threshold_high:
                regime_signals[i] = 1   # 高ボラティリティ
            elif ratio <= threshold_low:
                regime_signals[i] = -1  # 低ボラティリティ
            else:
                regime_signals[i] = 0   # 中性
    
    return regime_signals, volatility_ratio


@jit(nopython=True, cache=True)
def calculate_current_volatility_regime(regime_signals: np.ndarray) -> tuple:
    """
    現在のボラティリティレジーム状態を計算する（Numba JIT）
    
    Args:
        regime_signals: ボラティリティレジーム信号配列 (1=高ボラ, -1=低ボラ, 0=中性)
    
    Returns:
        tuple: (regime_index, regime_value)
               regime_index: 0=中性, 1=高ボラ, 2=低ボラ (regime_names用のインデックス)
               regime_value: 0=中性, 1=高ボラ, -1=低ボラ (実際のレジーム値)
    """
    length = len(regime_signals)
    if length == 0:
        return 0, 0  # 中性
    
    # 最新の値を取得
    latest_regime = regime_signals[-1]
    
    if latest_regime == 1:  # 高ボラティリティ
        return 1, 1   # 高ボラ
    elif latest_regime == -1:  # 低ボラティリティ
        return 2, -1  # 低ボラ
    else:  # 中性
        return 0, 0   # 中性


class VolatilityRegime(Indicator):
    """
    ボラティリティレジーム判別インジケーター
    短期と長期のATRを比較してボラティリティの状態を判定する
    
    特徴:
    - 短期ATR（デフォルト13期間）と長期ATR（デフォルト89期間）を使用
    - 短期ATR > 長期ATR * threshold_high → 高ボラティリティ
    - 短期ATR < 長期ATR * threshold_low → 低ボラティリティ  
    - それ以外は中性
    - ATRと同じスムージング方法、動的期間対応
    - ボラティリティ比率（短期ATR/長期ATR）も提供
    """
    
    def __init__(self,
                 short_period: int = 13,
                 long_period: int = 89,
                 threshold_high: float = 1.2,
                 threshold_low: float = 0.8,
                 smoothing_method: str = 'alma',
                 use_dynamic_period: bool = False,
                 cycle_part: float = 0.5,
                 detector_type: str = 'absolute_ultimate',
                 max_cycle: int = 120,
                 min_cycle: int = 5,
                 max_output: int = 120,
                 min_output: int = 5,
                 slope_index: int = 1,
                 range_threshold: float = 0.005,
                 lp_period: int = 10,
                 hp_period: int = 48):
        """
        コンストラクタ
        
        Args:
            short_period: 短期ATR期間（デフォルト: 13）
            long_period: 長期ATR期間（デフォルト: 89）
            threshold_high: 高ボラティリティ判定閾値（デフォルト: 1.2）
            threshold_low: 低ボラティリティ判定閾値（デフォルト: 0.8）
            smoothing_method: スムージング方法 ('wilder', 'hma', 'alma', 'zlema')
            use_dynamic_period: 動的期間を使用するかどうか
            cycle_part: サイクル部分の倍率（動的期間モード用）
            detector_type: 検出器タイプ（動的期間モード用）
            max_cycle: 最大サイクル期間（動的期間モード用）
            min_cycle: 最小サイクル期間（動的期間モード用）
            max_output: 最大出力値（動的期間モード用）
            min_output: 最小出力値（動的期間モード用）
            slope_index: トレンド判定期間（デフォルト: 1）
            range_threshold: range判定の基本閾値（デフォルト: 0.005）
            lp_period: ローパスフィルター期間（動的期間モード用）
            hp_period: ハイパスフィルター期間（動的期間モード用）
        """
        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_period else ""
        super().__init__(f"VolatilityRegime(short={short_period},long={long_period},th_h={threshold_high},th_l={threshold_low},smooth={smoothing_method}{dynamic_str})")
        
        self.short_period = short_period
        self.long_period = long_period
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.smoothing_method = smoothing_method
        self.use_dynamic_period = use_dynamic_period
        self.slope_index = slope_index
        self.range_threshold = range_threshold
        
        # 動的期間モード用パラメータ
        self.cycle_part = cycle_part
        self.detector_type = detector_type
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.lp_period = lp_period
        self.hp_period = hp_period
        
        # ATRインジケーターの初期化
        self.short_atr = ATR(
            period=self.short_period,
            smoothing_method=self.smoothing_method,
            use_dynamic_period=self.use_dynamic_period,
            cycle_part=self.cycle_part,
            detector_type=self.detector_type,
            max_cycle=self.max_cycle,
            min_cycle=self.min_cycle,
            max_output=self.max_output,
            min_output=self.min_output,
            slope_index=self.slope_index,
            range_threshold=self.range_threshold,
            lp_period=self.lp_period,
            hp_period=self.hp_period
        )
        
        self.long_atr = ATR(
            period=self.long_period,
            smoothing_method=self.smoothing_method,
            use_dynamic_period=self.use_dynamic_period,
            cycle_part=self.cycle_part,
            detector_type=self.detector_type,
            max_cycle=self.max_cycle,
            min_cycle=self.min_cycle,
            max_output=self.max_output,
            min_output=self.min_output,
            slope_index=self.slope_index,
            range_threshold=self.range_threshold,
            lp_period=self.lp_period,
            hp_period=self.hp_period
        )
        
        self._cache = {}
        self._result: Optional[VolatilityRegimeResult] = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        try:
            if isinstance(data, pd.DataFrame):
                # DataFrameの場合は形状と端点でハッシュを計算
                shape_tuple = data.shape
                first_row_tuple = tuple(data.iloc[0]) if len(data) > 0 else ()
                last_row_tuple = tuple(data.iloc[-1]) if len(data) > 0 else ()
                data_repr_tuple = (shape_tuple, first_row_tuple, last_row_tuple)
                data_hash_val = hash(data_repr_tuple)
            elif isinstance(data, np.ndarray):
                # NumPy配列の場合はバイト表現でハッシュ
                data_hash_val = hash(data.tobytes())
            else:
                # その他のデータ型は文字列表現でハッシュ化
                data_hash_val = hash(str(data))

        except Exception as e:
            self.logger.warning(f"データハッシュ計算中にエラー: {e}. データ全体の文字列表現を使用します。", exc_info=True)
            data_hash_val = hash(str(data))

        # パラメータ文字列の作成
        if self.use_dynamic_period:
            param_str = (f"short={self.short_period}_long={self.long_period}_th_h={self.threshold_high}_th_l={self.threshold_low}_"
                        f"smooth={self.smoothing_method}_dynamic={self.detector_type}_{self.max_output}_{self.min_output}")
        else:
            param_str = f"short={self.short_period}_long={self.long_period}_th_h={self.threshold_high}_th_l={self.threshold_low}_smooth={self.smoothing_method}"

        return f"{data_hash_val}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> VolatilityRegimeResult:
        """
        ボラティリティレジームを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high'と'low'と'close'カラムが必要
        
        Returns:
            VolatilityRegimeResult: ボラティリティレジーム判定結果
        """
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
            self.logger.warning("入力データが空です。空の結果を返します。")
            empty_result = VolatilityRegimeResult(
                values=np.array([], dtype=np.int8),
                short_atr=np.array([]),
                long_atr=np.array([]),
                volatility_ratio=np.array([]),
                trend_signals=np.array([], dtype=np.int8),
                current_regime='neutral',
                current_regime_value=0
            )
            return empty_result
            
        try:
            data_hash = self._get_data_hash(data)

            # キャッシュチェック
            if data_hash in self._cache and self._result is not None:
                # データ長が一致するか確認
                if len(self._result.values) == current_data_len:
                    return VolatilityRegimeResult(
                        values=self._result.values.copy(),
                        short_atr=self._result.short_atr.copy(),
                        long_atr=self._result.long_atr.copy(),
                        volatility_ratio=self._result.volatility_ratio.copy(),
                        trend_signals=self._result.trend_signals.copy(),
                        current_regime=self._result.current_regime,
                        current_regime_value=self._result.current_regime_value
                    )
                else:
                    self.logger.debug(f"キャッシュのデータ長が異なるため再計算します。")
                    # キャッシュを無効化
                    del self._cache[data_hash]
                    self._result = None

            # 短期と長期のATRを計算
            short_atr_result = self.short_atr.calculate(data)
            long_atr_result = self.long_atr.calculate(data)
            
            short_atr_values = short_atr_result.values
            long_atr_values = long_atr_result.values
            
            # ボラティリティレジーム信号を計算
            regime_signals, volatility_ratio = calculate_volatility_regime_signals(
                short_atr_values, long_atr_values, self.threshold_high, self.threshold_low
            )
            
            # 現在のボラティリティレジームを計算
            regime_index, regime_value = calculate_current_volatility_regime(regime_signals)
            regime_names = ['neutral', 'high', 'low']
            current_regime = regime_names[regime_index]
            
            # トレンド信号は短期ATRのものを使用
            trend_signals = short_atr_result.trend_signals

            result = VolatilityRegimeResult(
                values=regime_signals,
                short_atr=short_atr_values,
                long_atr=long_atr_values,
                volatility_ratio=volatility_ratio,
                trend_signals=trend_signals,
                current_regime=current_regime,
                current_regime_value=regime_value
            )

            # 計算結果を保存
            self._result = result
            self._cache[data_hash] = self._result
            return VolatilityRegimeResult(
                values=result.values.copy(),
                short_atr=result.short_atr.copy(),
                long_atr=result.long_atr.copy(),
                volatility_ratio=result.volatility_ratio.copy(),
                trend_signals=result.trend_signals.copy(),
                current_regime=result.current_regime,
                current_regime_value=result.current_regime_value
            )
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"VolatilityRegime '{self.name}' 計算中に予期せぬエラー: {error_msg}\n{stack_trace}")
            # Return NaNs matching the input data length
            self._result = None
            error_result = VolatilityRegimeResult(
                values=np.zeros(current_data_len, dtype=np.int8),
                short_atr=np.full(current_data_len, np.nan),
                long_atr=np.full(current_data_len, np.nan),
                volatility_ratio=np.full(current_data_len, np.nan),
                trend_signals=np.zeros(current_data_len, dtype=np.int8),
                current_regime='neutral',
                current_regime_value=0
            )
            return error_result

    def get_values(self) -> Optional[np.ndarray]:
        """ボラティリティレジーム値のみを取得する（後方互換性のため）"""
        if self._result is not None:
            return self._result.values.copy()
        return None

    def get_short_atr(self) -> Optional[np.ndarray]:
        """短期ATR値を取得する"""
        if self._result is not None:
            return self._result.short_atr.copy()
        return None

    def get_long_atr(self) -> Optional[np.ndarray]:
        """長期ATR値を取得する"""
        if self._result is not None:
            return self._result.long_atr.copy()
        return None

    def get_volatility_ratio(self) -> Optional[np.ndarray]:
        """ボラティリティ比率（短期ATR/長期ATR）を取得する"""
        if self._result is not None:
            return self._result.volatility_ratio.copy()
        return None

    def get_trend_signals(self) -> Optional[np.ndarray]:
        """トレンド信号を取得する"""
        if self._result is not None:
            return self._result.trend_signals.copy()
        return None

    def get_current_regime(self) -> str:
        """現在のボラティリティレジーム状態を取得する"""
        if self._result is not None:
            return self._result.current_regime
        return 'neutral'

    def get_current_regime_value(self) -> int:
        """現在のボラティリティレジーム値を取得する"""
        if self._result is not None:
            return self._result.current_regime_value
        return 0

    def get_dynamic_periods(self) -> tuple:
        """
        動的期間の値を取得する（動的期間モードのみ）
        
        Returns:
            tuple: (短期動的期間, 長期動的期間)
        """
        if not self.use_dynamic_period:
            return np.array([]), np.array([])
        
        short_periods = self.short_atr.get_dynamic_periods()
        long_periods = self.long_atr.get_dynamic_periods()
        
        return short_periods, long_periods

    def reset(self) -> None:
        """インジケータの状態（キャッシュ、結果）をリセットする"""
        super().reset()
        self._result = None
        self._cache = {}
        self.short_atr.reset()
        self.long_atr.reset()
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。") 