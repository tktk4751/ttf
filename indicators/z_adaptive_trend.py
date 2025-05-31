#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, Literal
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, int64, boolean

from .indicator import Indicator
from .price_source import PriceSource
from .c_atr import CATR
from .z_adaptive_ma import ZAdaptiveMA
from .cycle_efficiency_ratio import CycleEfficiencyRatio
from .x_trend_index import XTrendIndex
from .z_adaptive_trend_index import ZAdaptiveTrendIndex


@dataclass
class ZAdaptiveTrendResult:
    """Zアダプティブトレンドの計算結果"""
    middle: np.ndarray        # 中心線（選択されたソースタイプまたはZAdaptiveMA）
    upper_band: np.ndarray    # 上限バンド（下降トレンド時のみ表示）
    lower_band: np.ndarray    # 下限バンド（上昇トレンド時のみ表示）
    trend: np.ndarray         # トレンド方向（1:上昇、-1:下降）
    er: np.ndarray            # Efficiency Ratio (CER)
    dynamic_multiplier: np.ndarray  # 動的ATR乗数
    c_atr: np.ndarray         # CATR値
    trigger_values: np.ndarray # トリガー値（simple_adjustment計算用）


@njit(float64[:](float64[:], float64, float64), fastmath=True, parallel=True, cache=True)
def calculate_simple_adjustment_multiplier_optimized(trigger: np.ndarray, max_multiplier: float, min_multiplier: float) -> np.ndarray:
    """
    シンプルアジャストメント動的乗数を計算する（最適化&並列版）
    
    Args:
        trigger: トリガー値の配列（0〜1.0の範囲）
        max_multiplier: 最大乗数
        min_multiplier: 最小乗数
    
    Returns:
        動的乗数値の配列 = max_multiplier - trigger * (max_multiplier - min_multiplier)
        トリガー値が0の時はmax_multiplier、トリガー値が1の時はmin_multiplier
    """
    # 差分を計算
    diff = max_multiplier - min_multiplier
    
    result = np.empty_like(trigger)
    
    for i in prange(len(trigger)):
        # トリガー値をクランプ（0-1の範囲に制限）
        safe_trigger = min(max(trigger[i] if not np.isnan(trigger[i]) else 0.0, 0.0), 1.0)
        
        # 線形補間で動的乗数を計算
        # トリガー値が高いほど乗数が小さくなる（ボラティリティが低下）
        result[i] = max_multiplier - safe_trigger * diff
    
    return result


@njit(fastmath=True, parallel=True, cache=True)
def calculate_z_adaptive_trend_optimized(
    middle: np.ndarray,
    c_atr: np.ndarray,
    dynamic_multiplier: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Zアダプティブトレンドを計算する（最適化&並列版）
    SuperTrendと同様のロジック：中心線に対してATR*乗数でバンドを計算
    
    Args:
        middle: 中心線の配列（選択されたソースタイプまたはZAdaptiveMA）
        c_atr: CATRの配列（ボラティリティ測定・金額ベース）
        dynamic_multiplier: 動的乗数の配列
    
    Returns:
        中心線、上限バンド、下限バンド、トレンド方向のタプル
    """
    length = len(middle)
    
    # 結果用の配列を初期化
    upper_band = np.empty(length, dtype=np.float64)
    lower_band = np.empty(length, dtype=np.float64)
    trend = np.zeros(length, dtype=np.int8)
    
    # まず基本的なバンドを計算
    final_upper_band = np.empty(length, dtype=np.float64)
    final_lower_band = np.empty(length, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(middle[i]) or np.isnan(c_atr[i]) or np.isnan(dynamic_multiplier[i]):
            final_upper_band[i] = np.nan
            final_lower_band[i] = np.nan
            continue
            
        # 基本的なバンド幅（SuperTrendと同様に中心線を基準点として使用）
        band_width = c_atr[i] * dynamic_multiplier[i]
        final_upper_band[i] = middle[i] + band_width
        final_lower_band[i] = middle[i] - band_width
    
    # 最初の値を設定
    if length > 0 and not np.isnan(middle[0]) and not np.isnan(final_upper_band[0]):
        trend[0] = 1 if middle[0] > final_upper_band[0] else -1
    
        # 最初のバンド値
        if trend[0] == 1:
            # 上昇トレンドの場合、下限バンド（サポートライン）のみを表示
            upper_band[0] = np.nan
            lower_band[0] = final_lower_band[0]
        else:
            # 下降トレンドの場合、上限バンド（レジスタンスライン）のみを表示
            upper_band[0] = final_upper_band[0]
            lower_band[0] = np.nan
    else:
        upper_band[0] = np.nan
        lower_band[0] = np.nan
    
    # バンドとトレンドの計算（SuperTrendロジック）
    for i in range(1, length):
        if np.isnan(middle[i]) or np.isnan(final_upper_band[i-1]) or np.isnan(final_lower_band[i-1]):
            trend[i] = 0  # データがない場合はトレンドなし
            upper_band[i] = np.nan
            lower_band[i] = np.nan
            continue
            
        # トレンド判定（SuperTrendのロジック）
        if middle[i] > final_upper_band[i-1]:
            trend[i] = 1
        elif middle[i] < final_lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
            # バンドの調整（SuperTrendのロジック）
            if trend[i] == 1 and final_lower_band[i] < final_lower_band[i-1]:
                final_lower_band[i] = final_lower_band[i-1]
            elif trend[i] == -1 and final_upper_band[i] > final_upper_band[i-1]:
                final_upper_band[i] = final_upper_band[i-1]
        
        # トレンドに基づいてバンドを設定
        if trend[i] == 1:
            # 上昇トレンドの場合、下限バンド（サポートライン）のみを表示
            upper_band[i] = np.nan
            lower_band[i] = final_lower_band[i]
        else:
            # 下降トレンドの場合、上限バンド（レジスタンスライン）のみを表示
            upper_band[i] = final_upper_band[i]
            lower_band[i] = np.nan
    
    return middle, upper_band, lower_band, trend


class ZAdaptiveTrend(Indicator):
    """
    ZAdaptiveTrend（Zアダプティブトレンド）インディケーター
    
    特徴:
    - SuperTrendとZAdaptiveChannelの組み合わせ
    - 選択可能なトリガーソース（CER、X-Trend、Z-Trend）に基づくシンプルアジャストメント動的乗数適応
    - 選択可能な中心線ソース（価格ソースまたはZAdaptiveMA）
    - CATRを使用したボラティリティベースのバンド
    - トレンド判定機能：上昇/下降トレンドを自動判定
    - トレンドに基づいたバンド表示（上昇トレンドは下限バンドのみ、下降トレンドは上限バンドのみ）
    
    使用方法:
    - トレンドの方向性判定
    - 動的なサポート/レジスタンスレベルの特定
    - トレンドの方向性とボラティリティに基づくエントリー/エグジット
    - 効率比を使用したトレンド分析
    """
    
    def __init__(
        self,
        # シンプルアジャストメント乗数パラメータ
        max_multiplier: float = 6.0,     # 最大乗数（トリガー値0の時）
        min_multiplier: float = 1.0,     # 最小乗数（トリガー値1の時）
        
        # トレンド用ソースタイプ選択（中心線とトレンド判定に使用）
        trend_src_type: str = 'hlc3',    # 'open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4', 'z_adaptive_ma'
        
        # トリガーソース選択
        trigger_source: str = 'x_trend',     # 'cer', 'x_trend', 'z_trend'
        
        # CATR用ソースタイプ
        catr_src_type: str = 'hlc3',     # 'open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4'
        
        # CER用ソースタイプ
        cer_src_type: str = 'hlc3',      # 'open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4'
        
        # CERパラメータ
        detector_type: str = 'dudi_e',    # CER用ドミナントサイクル検出器タイプ
        cycle_part: float = 0.4,          # CER用サイクル部分
        lp_period: int = 5,              # CER用ローパスフィルター期間
        hp_period: int = 100,             # CER用ハイパスフィルター期間
        max_cycle: int = 120,             # CER用最大サイクル期間
        min_cycle: int = 10,              # CER用最小サイクル期間
        max_output: int = 75,            # CER用最大出力値
        min_output: int = 5,             # CER用最小出力値
        use_kalman_filter: bool = False,  # CER用カルマンフィルター使用有無
        
        # Xトレンドインデックスパラメータ（trigger_source='x_trend'の場合に使用）
        x_detector_type: str = 'dudi_e',
        x_cycle_part: float = 0.7,
        x_max_cycle: int = 120,
        x_min_cycle: int = 5,
        x_max_output: int = 55,
        x_min_output: int = 8,
        x_smoother_type: str = 'alma',
        fixed_threshold: float = 0.65,
        
        # ZAdaptiveMA用パラメータ（trend_src_type='z_adaptive_ma'の場合に使用）
        fast_period: int = 2,             # 速い移動平均の期間（固定値）
        slow_period: int = 30             # 遅い移動平均の期間（固定値）
    ):
        """
        コンストラクタ
        
        Args:
            max_multiplier: 最大乗数（トリガー値0の時に使用）
            min_multiplier: 最小乗数（トリガー値1の時に使用）
            
            trend_src_type: トレンド用ソースタイプ（中心線とトレンド判定に使用）
                'open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4': 価格ソース
                'z_adaptive_ma': ZAdaptiveMAを使用
            
            trigger_source: トリガー値のソース
                'cer': サイクル効率比（デフォルト）
                'x_trend': Xトレンドインデックス
                'z_trend': Zアダプティブトレンドインデックス
            
            catr_src_type: CATR計算用のソースタイプ
            cer_src_type: CER計算用のソースタイプ
            
            detector_type: CER用ドミナントサイクル検出器タイプ
            cycle_part: CER用サイクル部分
            lp_period: CER用ローパスフィルター期間
            hp_period: CER用ハイパスフィルター期間
            max_cycle: CER用最大サイクル期間
            min_cycle: CER用最小サイクル期間
            max_output: CER用最大出力値
            min_output: CER用最小出力値
            use_kalman_filter: CER用カルマンフィルター使用有無
            
            x_detector_type: Xトレンド用検出器タイプ
            x_cycle_part: Xトレンド用サイクル部分
            x_max_cycle: Xトレンド用最大サイクル期間
            x_min_cycle: Xトレンド用最小サイクル期間
            x_max_output: Xトレンド用最大出力値
            x_min_output: Xトレンド用最小出力値
            x_smoother_type: Xトレンド用平滑化タイプ
            fixed_threshold: 固定しきい値（XTrendIndex用）
            
            fast_period: 速い移動平均の期間（固定値）
            slow_period: 遅い移動平均の期間（固定値）
        """
        # 有効なtrigger_sourceをチェック
        if trigger_source not in ['cer', 'x_trend', 'z_trend']:
            trigger_source = 'cer'
        
        # 有効なtrend_src_typeをチェック
        valid_trend_sources = ['open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4', 'z_adaptive_ma']
        if trend_src_type not in valid_trend_sources:
            trend_src_type = 'hlc3'
        
        super().__init__(f"ZAdaptiveTrend({trigger_source},{trend_src_type},{max_multiplier}-{min_multiplier},{cycle_part})")
        
        # パラメータの保存
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self.trend_src_type = trend_src_type
        self.trigger_source = trigger_source
        self.catr_src_type = catr_src_type
        self.cer_src_type = cer_src_type
        
        # 依存オブジェクトの初期化 - 最適化: 必要な場合のみ初期化
        # 1. CycleEfficiencyRatio (trigger_source='cer'の場合のみ)
        self.cycle_er = None
        if trigger_source == 'cer':
            self.cycle_er = CycleEfficiencyRatio(
                detector_type=detector_type,
                cycle_part=cycle_part,
                lp_period=lp_period,
                hp_period=hp_period,
                max_cycle=max_cycle,
                min_cycle=min_cycle,
                max_output=max_output,
                min_output=min_output,
                use_kalman_filter=use_kalman_filter,
                src_type=cer_src_type  # CER専用のソースタイプを使用
            )
        
        # 2. Xトレンドインデックス (trigger_source='x_trend'の場合のみ)
        self.x_trend_index = None
        if trigger_source == 'x_trend':
            self.x_trend_index = XTrendIndex(
                detector_type=x_detector_type,
                cycle_part=x_cycle_part,
                max_cycle=x_max_cycle,
                min_cycle=x_min_cycle,
                max_output=x_max_output,
                min_output=x_min_output,
                src_type=cer_src_type,  # X-Trendはcer_src_typeを使用
                lp_period=lp_period,
                hp_period=hp_period,
                smoother_type=x_smoother_type,
                fixed_threshold=fixed_threshold
            )
        
        # 3. Zアダプティブトレンドインデックス (trigger_source='z_trend'の場合のみ)
        self.z_trend_index = None
        if trigger_source == 'z_trend':
            self.z_trend_index = ZAdaptiveTrendIndex(
                detector_type=x_detector_type,
                cycle_part=x_cycle_part,
                max_cycle=x_max_cycle,
                min_cycle=x_min_cycle,
                max_output=x_max_output,
                min_output=x_min_output,
                src_type=cer_src_type,  # Z-Trendもcer_src_typeを使用
                lp_period=lp_period,
                hp_period=hp_period,
                smoother_type=x_smoother_type,
                # CER パラメータ
                cer_detector_type=detector_type,
                cer_lp_period=lp_period,
                cer_hp_period=hp_period,
                cer_cycle_part=cycle_part,
                cer_max_cycle=max_cycle,
                cer_min_cycle=min_cycle,
                cer_max_output=max_output,
                cer_min_output=min_output,
                cer_src_type=cer_src_type,
                use_kalman_filter=use_kalman_filter,
            )
        
        # 4. ZAdaptiveMA（trend_src_type='z_adaptive_ma'の場合のみ初期化）
        self._z_adaptive_ma = None
        if trend_src_type == 'z_adaptive_ma':
            self._z_adaptive_ma = ZAdaptiveMA(fast_period=fast_period, slow_period=slow_period)
        
        # 5. CATR（CATR専用のソースタイプを使用）
        self._c_atr = CATR(src_type=catr_src_type)
        
        # 6. PriceSource
        self._price_source = PriceSource()
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 10  # キャッシュの最大サイズ
        self._cache_keys = []  # キャッシュキーの順序管理用
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を生成（高速化版）
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        # DataFrameの場合はサイズと最初と最後の値のみを使用
        if isinstance(data, pd.DataFrame):
            shape = data.shape
            # 最初と最後の10行のみ使用（大きなデータセットの場合も高速）
            if len(data) > 20:
                first_last = (
                    tuple(data.iloc[0].values) + 
                    tuple(data.iloc[-1].values) +
                    (data.shape[0],)  # データの長さも含める
                )
            else:
                # 小さなデータセットはすべて使用
                first_last = tuple(data.values.flatten()[-20:])
        else:
            shape = data.shape
            # NumPy配列も同様
            if len(data) > 20:
                if data.ndim > 1:
                    first_last = tuple(data[0]) + tuple(data[-1]) + (data.shape[0],)
                else:
                    first_last = (data[0], data[-1], data.shape[0])
            else:
                first_last = tuple(data.flatten()[-20:])
            
        # パラメータとサイズ、データのサンプルを組み合わせたハッシュを返す
        params_str = f"{self.trigger_source}_{self.trend_src_type}_{self.catr_src_type}_{self.cer_src_type}_{self.max_multiplier}_{self.min_multiplier}_ZAT"
        
        return f"{params_str}_{hash(first_last + (shape,))}"
    
    def _calculate_trigger_values(self, data) -> np.ndarray:
        """
        選択されたソースに基づいてトリガー値を計算
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 0-1の範囲のトリガー値
        """
        if self.trigger_source == 'cer':
            # CERの場合は絶対値を取って0-1に正規化
            if self.cycle_er is None:
                # フォールバック: 固定値を返す
                return np.zeros(len(data) if hasattr(data, '__len__') else 100)
            raw_er = self.cycle_er.calculate(data)
            trigger_values = np.abs(raw_er)
            return trigger_values
            
        elif self.trigger_source == 'x_trend':
            # Xトレンドインデックスは既に0-1の範囲なのでそのまま使用
            if self.x_trend_index is None:
                # フォールバック: 固定値を返す
                return np.zeros(len(data) if hasattr(data, '__len__') else 100)
            result = self.x_trend_index.calculate(data)
            return result.values
            
        elif self.trigger_source == 'z_trend':
            # Zアダプティブトレンドインデックスも既に0-1の範囲なのでそのまま使用
            if self.z_trend_index is None:
                # フォールバック: 固定値を返す
                return np.zeros(len(data) if hasattr(data, '__len__') else 100)
            result = self.z_trend_index.calculate(data)
            return result.values
            
        else:
            # デフォルトはCER
            if self.cycle_er is None:
                # フォールバック: 固定値を返す
                return np.zeros(len(data) if hasattr(data, '__len__') else 100)
            raw_er = self.cycle_er.calculate(data)
            trigger_values = np.abs(raw_er)
            return trigger_values
    
    def _get_middle_line(self, data, trigger_values: np.ndarray) -> np.ndarray:
        """
        選択されたソースタイプに基づいて中心線を取得
        
        Args:
            data: 価格データ
            trigger_values: トリガー値（ZAdaptiveMAで使用）
            
        Returns:
            np.ndarray: 中心線の値
        """
        if self.trend_src_type == 'z_adaptive_ma':
            # ZAdaptiveMAを使用
            if self._z_adaptive_ma is None:
                # 初期化されていない場合のフォールバック
                return self._price_source.get_source(data, 'hlc3')
            
            # trigger_sourceに応じて適切なソース値を取得
            if self.trigger_source == 'cer':
                if self.cycle_er is not None:
                    ma_source_values = self.cycle_er.calculate(data)
                else:
                    ma_source_values = trigger_values  # フォールバック
            else:
                ma_source_values = trigger_values  # X-TrendやZ-Trendの場合
                
            return self._z_adaptive_ma.calculate(data, ma_source_values)
        else:
            # 価格ソースを使用
            return self._price_source.get_source(data, self.trend_src_type)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Zアダプティブトレンドを計算（高速化版）
        
        Args:
            data: DataFrame または numpy 配列
        
        Returns:
            np.ndarray: 中心線（選択されたソースタイプまたはZAdaptiveMA）の値
        """
        try:
            # データハッシュを計算して、キャッシュが有効かどうかを確認
            data_hash = self._get_data_hash(data)
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新（最も新しく使われたキーを最後に）
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                return self._result_cache[data_hash].middle
            
            # 1. トリガー値の計算
            trigger_values = self._calculate_trigger_values(data)
            
            # 2. シンプルアジャストメント動的乗数の計算
            dynamic_multiplier = calculate_simple_adjustment_multiplier_optimized(trigger_values, self.max_multiplier, self.min_multiplier)
            
            # 3. 中心線の計算（選択されたソースタイプまたはZAdaptiveMA）
            middle = self._get_middle_line(data, trigger_values)
            
            # 4. CATRの計算（CATR専用のソースタイプを使用）
            self._c_atr.calculate(data)
            
            # 金額ベースのCATRを取得 - 重要: バンド計算には金額ベースのATRを使用する
            c_atr = self._c_atr.get_absolute_atr()
            
            # 5. Zアダプティブトレンドの計算（中心線、上限バンド、下限バンド、トレンド） - 最適化版
            middle_result, upper_band, lower_band, trend = calculate_z_adaptive_trend_optimized(
                middle,
                c_atr,
                dynamic_multiplier
            )
            
            # 効率比の取得（結果に含めるため）
            if self.cycle_er is not None:
                er = self.cycle_er.calculate(data)
            else:
                er = np.zeros_like(trigger_values)
            
            # 結果をキャッシュ
            result = ZAdaptiveTrendResult(
                middle=middle_result,
                upper_band=upper_band,
                lower_band=lower_band,
                trend=trend,
                er=er,
                dynamic_multiplier=dynamic_multiplier,
                c_atr=c_atr,
                trigger_values=trigger_values
            )
            
            # キャッシュサイズ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            return middle_result  # 元のインターフェイスと互換性を保つため中心線を返す
            
        except Exception as e:
            self.logger.error(f"Zアダプティブトレンド計算中にエラー: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return np.array([])
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        バンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([]), np.array([]), np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.middle, result.upper_band, result.lower_band
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            return np.array([]), np.array([]), np.array([])
    
    def get_trend(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンド値（1:上昇トレンド、-1:下降トレンド、0:トレンドなし）
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.trend
        except Exception as e:
            self.logger.error(f"トレンド値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        効率比（CER）の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 効率比の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.er
        except Exception as e:
            self.logger.error(f"効率比取得中にエラー: {str(e)}")
            return np.array([])
    
    # 後方互換性のため
    def get_cycle_er(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        効率比（CER）の値を取得（後方互換性のため）
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 効率比の値
        """
        return self.get_efficiency_ratio(data)
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.dynamic_multiplier
        except Exception as e:
            self.logger.error(f"動的乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_trigger_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トリガー値を取得（simple_adjustment計算で使用された値）
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トリガー値（選択されたトリガーソースの値）
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.trigger_values
        except Exception as e:
            self.logger.error(f"トリガー値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_c_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CATR値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: CATR値
        """
        try:
            if data is not None:
                self.calculate(data)
            
            # 最新の結果を使用
            if not self._result_cache:
                return np.array([])
                
            # 最新のキャッシュを使用
            if self._cache_keys:
                result = self._result_cache[self._cache_keys[-1]]
            else:
                # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
                result = next(iter(self._result_cache.values()))
                
            return result.c_atr
        except Exception as e:
            self.logger.error(f"CATR取得中にエラー: {str(e)}")
            return np.array([])
    
    def reset(self) -> None:
        """
        状態をリセット
        """
        # キャッシュをクリア
        self._result_cache = {}
        self._cache_keys = []
        
        # 依存オブジェクトもリセット（存在する場合のみ）
        if self.cycle_er is not None:
            self.cycle_er.reset()
        if self.x_trend_index is not None:
            self.x_trend_index.reset()
        if self.z_trend_index is not None:
            self.z_trend_index.reset()
        if self._z_adaptive_ma is not None:
            self._z_adaptive_ma.reset()
        self._c_atr.reset() 