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
class ZAdaptiveChannelResult:
    """Zアダプティブチャネルの計算結果"""
    middle: np.ndarray        # 中心線（ZAdaptiveMA）
    upper: np.ndarray         # 上限バンド
    lower: np.ndarray         # 下限バンド
    er: np.ndarray            # Efficiency Ratio (CER)
    dynamic_multiplier: np.ndarray  # 動的ATR乗数
    z_atr: np.ndarray         # CATR値
    max_mult_values: np.ndarray  # 動的に計算されたmax_multiplier値
    min_mult_values: np.ndarray  # 動的に計算されたmin_multiplier値
    multiplier_trigger: np.ndarray  # 乗数計算に使用されたトリガー値


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True, target='parallel')
def calculate_dynamic_multiplier_vec(trigger: float, max_mult: float, min_mult: float) -> float:
    """
    トリガー値に基づいて動的乗数を計算する（ベクトル化&並列版）
    
    Args:
        trigger: トリガー値（0〜1.0の範囲、トレンドインデックスやCERの絶対値）
        max_mult: 最大乗数（トレンド時に使用）
        min_mult: 最小乗数（非トレンド時に使用）
    
    Returns:
        動的乗数値
    """
    # トリガー値はCERの場合は絶対値を使用、XトレンドやZアダプティブトレンドの場合はそのまま使用
    # どちらにしても0〜1の範囲で扱う
    
    # ZChannelと同様の計算式を使用：トリガー値が高いほどバンド幅が小さくなる
    return max_mult - trigger * (max_mult - min_mult)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True, target='parallel')
def calculate_dynamic_max_multiplier(trigger: float, max_max_mult: float, min_max_mult: float) -> float:
    """
    トリガー値に基づいて動的最大乗数を計算する（ベクトル化&並列版）
    
    Args:
        trigger: トリガー値（0〜1.0の範囲、トレンドインデックスやCERの絶対値）
        max_max_mult: 最大乗数の最大値（強いトレンド時に使用）
        min_max_mult: 最大乗数の最小値（中程度のトレンド時に使用）
    
    Returns:
        動的最大乗数値
    """
    # トリガー値は0〜1の範囲で扱う
    
    # ZChannelと同様の計算式：トリガー値が高いほど最大乗数が小さくなる
    return max_max_mult - trigger * (max_max_mult - min_max_mult)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True, target='parallel')
def calculate_dynamic_min_multiplier(trigger: float, max_min_mult: float, min_min_mult: float) -> float:
    """
    トリガー値に基づいて動的最小乗数を計算する（ベクトル化&並列版）
    
    Args:
        trigger: トリガー値（0〜1.0の範囲、トレンドインデックスやCERの絶対値）
        max_min_mult: 最小乗数の最大値（中程度のトレンド時に使用）
        min_min_mult: 最小乗数の最小値（非トレンド時に使用）
    
    Returns:
        動的最小乗数値
    """
    # トリガー値は0〜1の範囲で扱う
    
    # ZChannelと同様の計算式：トリガー値が高いほど最小乗数が小さくなる
    return max_min_mult - trigger * (max_min_mult - min_min_mult)


@njit(float64[:](float64[:], float64, float64), fastmath=True, parallel=True, cache=True)
def calculate_dynamic_multiplier_optimized(trigger: np.ndarray, max_mult: float, min_mult: float) -> np.ndarray:
    """
    トリガー値に基づいて動的乗数を計算する（最適化&並列版）
    
    Args:
        trigger: トリガー値の配列（0〜1.0の範囲）
        max_mult: 最大乗数（トレンド時に使用）
        min_mult: 最小乗数（非トレンド時に使用）
    
    Returns:
        動的乗数値の配列
    """
    result = np.empty_like(trigger)
    
    for i in prange(len(trigger)):
        # トリガー値は0〜1の範囲で扱う
        
        # ZChannelと同様の計算式を使用：トリガー値が高いほどバンド幅が小さくなる
        result[i] = max_mult - trigger[i] * (max_mult - min_mult)
    
    return result


@njit(fastmath=True, parallel=True, cache=True)
def calculate_z_adaptive_channel_optimized(
    z_ma: np.ndarray,
    z_atr: np.ndarray,
    dynamic_multiplier: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Zアダプティブチャネルを計算する（最適化&並列版）
    
    Args:
        z_ma: ZAdaptiveMAの配列（中心線）
        z_atr: CATRの配列（ボラティリティ測定・金額ベース）
        dynamic_multiplier: 動的乗数の配列
    
    Returns:
        中心線、上限バンド、下限バンドのタプル
    """
    length = len(z_ma)
    
    # 結果用の配列を初期化（中心線は既存配列を再利用）
    middle = z_ma  # 直接参照で最適化（コピー不要）
    upper = np.empty(length, dtype=np.float64)
    lower = np.empty(length, dtype=np.float64)
    
    # 並列処理で各時点でのバンドを計算
    for i in prange(length):
        # 動的乗数をATRに適用
        if np.isnan(z_ma[i]) or np.isnan(z_atr[i]) or np.isnan(dynamic_multiplier[i]):
            upper[i] = np.nan
            lower[i] = np.nan
            continue
            
        band_width = z_atr[i] * dynamic_multiplier[i]
        
        # 金額ベースのATRを使用（絶対値）
        upper[i] = z_ma[i] + band_width
        lower[i] = z_ma[i] - band_width
    
    return middle, upper, lower


class ZAdaptiveChannel(Indicator):
    """
    ZAdaptiveChannel（Zアダプティブチャネル）インディケーター
    
    特徴:
    - 効率比(CER)、Xトレンドインデックス、Zアダプティブトレンドインデックスのいずれかに基づいて
      動的に調整されるチャネル幅
    - ZAdaptiveMAを中心線として使用
    - CATRを使用したボラティリティベースのバンド
    - トレンド強度に応じて自動調整されるATR乗数
    
    使用方法:
    - 動的なサポート/レジスタンスレベルの特定
    - トレンドの方向性とボラティリティに基づくエントリー/エグジット
    - トレンド分析
    """
    
    def __init__(
        self,
        # 動的乗数の範囲パラメータ
        max_max_multiplier: float = 8.0,    # 最大乗数の最大値
        min_max_multiplier: float = 3.0,    # 最大乗数の最小値
        max_min_multiplier: float = 1.5,    # 最小乗数の最大値
        min_min_multiplier: float = 0.5,    # 最小乗数の最小値
        src_type: str = 'hlc3',       # 'open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4'
        
        # トリガーソース選択
        multiplier_source: str = 'cer',  # 'cer', 'x_trend', 'z_trend'
        
        # CERパラメータ
        detector_type: str = 'dudi_e',     # CER用ドミナントサイクル検出器タイプ
        cycle_part: float = 0.4,           # CER用サイクル部分
        lp_period: int = 5,               # CER用ローパスフィルター期間
        hp_period: int = 100,              # CER用ハイパスフィルター期間
        max_cycle: int = 120,              # CER用最大サイクル期間
        min_cycle: int = 10,               # CER用最小サイクル期間
        max_output: int = 75,             # CER用最大出力値
        min_output: int = 5,              # CER用最小出力値
        use_kalman_filter: bool = False,   # CER用カルマンフィルター使用有無
        
        # ZAdaptiveMA用パラメータ
        fast_period: int = 2,             # 速い移動平均の期間（固定値）
        slow_period: int = 30,            # 遅い移動平均の期間（固定値）
        
        # Xトレンドインデックスパラメータ（multiplier_source='x_trend'の場合に使用）
        x_detector_type: str = 'phac_e',
        x_cycle_part: float = 0.5,
        x_max_cycle: int = 55,
        x_min_cycle: int = 5,
        x_max_output: int = 34,
        x_min_output: int = 5,
        x_smoother_type: str = 'alma',
        
        # 動的しきい値のパラメータ
        max_threshold: float = 0.75,
        min_threshold: float = 0.55
    ):
        """
        コンストラクタ
        
        Args:
            max_max_multiplier: 最大乗数の最大値（動的乗数使用時）
            min_max_multiplier: 最大乗数の最小値（動的乗数使用時）
            max_min_multiplier: 最小乗数の最大値（動的乗数使用時）
            min_min_multiplier: 最小乗数の最小値（動的乗数使用時）
            src_type: ソースタイプ
            
            multiplier_source: 乗数計算に使用するトリガーのソース
                'cer': サイクル効率比（デフォルト）
                'x_trend': Xトレンドインデックス
                'z_trend': Zアダプティブトレンドインデックス
            
            detector_type: CER用ドミナントサイクル検出器タイプ
            cycle_part: CER用サイクル部分
            lp_period: CER用ローパスフィルター期間
            hp_period: CER用ハイパスフィルター期間
            max_cycle: CER用最大サイクル期間
            min_cycle: CER用最小サイクル期間
            max_output: CER用最大出力値
            min_output: CER用最小出力値
            use_kalman_filter: CER用カルマンフィルター使用有無
            
            fast_period: 速い移動平均の期間（固定値）
            slow_period: 遅い移動平均の期間（固定値）
            
            x_detector_type: Xトレンド用検出器タイプ (multiplier_source='x_trend'の場合)
            x_cycle_part: Xトレンド用サイクル部分
            x_max_cycle: Xトレンド用最大サイクル期間
            x_min_cycle: Xトレンド用最小サイクル期間
            x_max_output: Xトレンド用最大出力値
            x_min_output: Xトレンド用最小出力値
            x_smoother_type: Xトレンド用平滑化タイプ
            
            max_threshold: 動的しきい値の最大値（Xトレンド使用時）
            min_threshold: 動的しきい値の最小値（Xトレンド使用時）
        """
        # 有効なmultiplier_sourceをチェック
        if multiplier_source not in ['cer', 'x_trend', 'z_trend']:
            self.logger.warning(f"無効なmultiplier_source: {multiplier_source}。'cer'を使用します。")
            multiplier_source = 'cer'
        
        super().__init__(f"ZAdaptiveChannel({multiplier_source},{max_max_multiplier},{cycle_part},{src_type})")
        
        # パラメータの保存
        self.max_max_multiplier = max_max_multiplier
        self.min_max_multiplier = min_max_multiplier
        self.max_min_multiplier = max_min_multiplier
        self.min_min_multiplier = min_min_multiplier
        self.src_type = src_type
        self.multiplier_source = multiplier_source
        
        # 依存オブジェクトの初期化
        # 1. CycleEfficiencyRatio (常に初期化)
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
            src_type=src_type
        )
        
        # 2. Xトレンドインデックス (multiplier_source='x_trend'または'z_trend'の場合)
        self.x_trend_index = None
        if multiplier_source in ['x_trend', 'z_trend']:
            self.x_trend_index = XTrendIndex(
                detector_type=x_detector_type,
                cycle_part=x_cycle_part,
                max_cycle=x_max_cycle,
                min_cycle=x_min_cycle,
                max_output=x_max_output,
                min_output=x_min_output,
                src_type=src_type,
                lp_period=lp_period,
                hp_period=hp_period,
                smoother_type=x_smoother_type,
                # CER パラメータ (XTrendIndex 内部で使用)
                cer_detector_type=detector_type,
                cer_lp_period=lp_period,
                cer_hp_period=hp_period,
                cer_cycle_part=cycle_part,
                max_threshold=max_threshold,
                min_threshold=min_threshold
            )
        
        # 3. Zアダプティブトレンドインデックス (multiplier_source='z_trend'の場合)
        self.z_trend_index = None
        if multiplier_source == 'z_trend':
            self.z_trend_index = ZAdaptiveTrendIndex(
                detector_type=x_detector_type,
                cycle_part=x_cycle_part,
                max_cycle=x_max_cycle,
                min_cycle=x_min_cycle,
                max_output=x_max_output,
                min_output=x_min_output,
                src_type=src_type,
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
                cer_src_type=src_type,
                use_kalman_filter=use_kalman_filter,
                # しきい値
                max_threshold=max_threshold,
                min_threshold=min_threshold
            )
        
        # 4. ZAdaptiveMA
        self._z_adaptive_ma = ZAdaptiveMA(fast_period=fast_period, slow_period=slow_period)
        
        # 5. CATR
        self._c_atr = CATR()
        
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
        params_str = (
            f"{self.src_type}_{self.max_max_multiplier}_{self.min_max_multiplier}_"
            f"{self.max_min_multiplier}_{self.min_min_multiplier}_{self.multiplier_source}"
        )
        
        return f"{params_str}_{hash(first_last + (shape,))}"
    
    def _calculate_trigger_values(self, data) -> np.ndarray:
        """
        選択されたソースに基づいてトリガー値を計算
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 0-1の範囲のトリガー値
        """
        if self.multiplier_source == 'cer':
            # CERの場合は絶対値を取って0-1に正規化
            raw_er = self.cycle_er.calculate(data)
            trigger_values = np.abs(raw_er)
            return trigger_values
            
        elif self.multiplier_source == 'x_trend':
            # Xトレンドインデックスは既に0-1の範囲なのでそのまま使用
            result = self.x_trend_index.calculate(data)
            return result.values
            
        elif self.multiplier_source == 'z_trend':
            # Zアダプティブトレンドインデックスも既に0-1の範囲なのでそのまま使用
            result = self.z_trend_index.calculate(data)
            return result.values
            
        else:
            # デフォルトはCER
            raw_er = self.cycle_er.calculate(data)
            trigger_values = np.abs(raw_er)
            return trigger_values
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Zアダプティブチャネルを計算（高速化版）
        
        Args:
            data: DataFrame または numpy 配列
        
        Returns:
            np.ndarray: 中心線（ZAdaptiveMA）の値
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
            
            # 1. 効率比の計算（CER）- ZAdaptiveMAと_c_atrで使用するため常に計算
            er = self.cycle_er.calculate(data)
            
            # 2. トリガー値の計算
            trigger_values = self._calculate_trigger_values(data)
            
            # 3. 動的な乗数値の計算（並列化・ベクトル化版）
            # 3.1. 動的な最大乗数の計算
            max_mult_values = calculate_dynamic_max_multiplier(
                trigger_values, 
                self.max_max_multiplier, 
                self.min_max_multiplier
            )
            
            # 3.2. 動的な最小乗数の計算
            min_mult_values = calculate_dynamic_min_multiplier(
                trigger_values, 
                self.max_min_multiplier, 
                self.min_min_multiplier
            )
            
            # 3.3. 線形計算を使用して最終的な動的乗数を計算
            dynamic_multiplier = calculate_dynamic_multiplier_vec(trigger_values, max_mult_values, min_mult_values)
            
            # 4. ZAdaptiveMAの計算（中心線）
            z_ma = self._z_adaptive_ma.calculate(data, er)
            
            # 5. CATRの計算
            self._c_atr.calculate(data, er)
            
            # 金額ベースのCATRを取得 - 重要: バンド計算には金額ベースのATRを使用する
            z_atr = self._c_atr.get_absolute_atr()
            
            # 6. Zアダプティブチャネルの計算（中心線、上限バンド、下限バンド）- 最適化版
            middle, upper, lower = calculate_z_adaptive_channel_optimized(
                z_ma,
                z_atr,
                dynamic_multiplier
            )
            
            # 結果をキャッシュ
            result = ZAdaptiveChannelResult(
                middle=middle,
                upper=upper,
                lower=lower,
                er=er,
                dynamic_multiplier=dynamic_multiplier,
                z_atr=z_atr,
                max_mult_values=max_mult_values,
                min_mult_values=min_mult_values,
                multiplier_trigger=trigger_values
            )
            
            # キャッシュサイズ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            return middle  # 元のインターフェイスと互換性を保つため中心線を返す
            
        except Exception as e:
            self.logger.error(f"Zアダプティブチャネル計算中にエラー: {str(e)}")
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
                
            return result.middle, result.upper, result.lower
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            return np.array([]), np.array([]), np.array([])
    
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
    
    def get_multiplier_trigger(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        乗数計算に使用されたトリガー値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トリガー値
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
                
            return result.multiplier_trigger
        except Exception as e:
            self.logger.error(f"トリガー値取得中にエラー: {str(e)}")
            return np.array([])
    
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
    
    def get_z_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
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
                
            return result.z_atr
        except Exception as e:
            self.logger.error(f"ZATR取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_max_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的最大乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的最大乗数の値
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
                
            return result.max_mult_values
        except Exception as e:
            self.logger.error(f"動的最大乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_dynamic_min_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的最小乗数の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的最小乗数の値
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
                
            return result.min_mult_values
        except Exception as e:
            self.logger.error(f"動的最小乗数取得中にエラー: {str(e)}")
            return np.array([])
    
    def reset(self) -> None:
        """
        状態をリセット
        """
        # キャッシュをクリア
        self._result_cache = {}
        self._cache_keys = []
        
        # 依存オブジェクトもリセット
        self.cycle_er.reset()
        if self.x_trend_index is not None:
            self.x_trend_index.reset()
        if self.z_trend_index is not None:
            self.z_trend_index.reset()
        self._z_adaptive_ma.reset()
        self._c_atr.reset() 