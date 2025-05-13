#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import jit, vectorize, njit, prange

from .indicator import Indicator
from .z_trend_index import XTrendIndex
from .cycle_efficiency_ratio import CycleEfficiencyRatio
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .ehlers_hody_dc import EhlersHoDyDC
from .z_adx import ZADX


@dataclass
class ZTrendFilterResult:
    """Zトレンドフィルターの計算結果"""
    values: np.ndarray          # Zトレンドフィルターの値（0-1の範囲で正規化）
    trend_index: np.ndarray     # Zトレンドインデックスの値
    zadx: np.ndarray            # ZADX値（0-1の範囲）
    er: np.ndarray              # サイクル効率比（CER）
    combined_rms: np.ndarray    # トレンドインデックス、ZADX、ERの二乗平均平方根（RMS）
    rms_window: np.ndarray      # RMS計算のウィンドウサイズ
    dynamic_threshold: np.ndarray  # 動的しきい値


@njit(fastmath=True)
def calculate_simple_average_combination(trend_index: np.ndarray, zadx: np.ndarray, er: np.ndarray, ti_weight: float = 0.4, zadx_weight: float = 0.4) -> np.ndarray:
    """
    トレンドインデックス、ZADX、ERの単純加重平均（高速化版）
    
    Args:
        trend_index: Zトレンドインデックスの配列
        zadx: ZADX値の配列（0-1の範囲）
        er: 効率比の配列
        ti_weight: トレンドインデックスの重み（デフォルト: 0.4）
        zadx_weight: ZADXの重み（デフォルト: 0.4）
    
    Returns:
        組み合わせた値の配列
    """
    length = len(trend_index)
    result = np.zeros(length, dtype=np.float64)
    
    # 残りの重みはERに割り当て
    er_weight = 1.0 - ti_weight - zadx_weight
    
    # 重みの正規化（合計が1になるようにする）
    total_weight = ti_weight + zadx_weight + er_weight
    if total_weight != 1.0:
        ti_weight = ti_weight / total_weight
        zadx_weight = zadx_weight / total_weight
        er_weight = er_weight / total_weight
    
    for i in range(length):
        if np.isnan(trend_index[i]) or np.isnan(zadx[i]) or np.isnan(er[i]):
            result[i] = np.nan
            continue
        
        # ERの絶対値を使用
        er_abs = abs(er[i])
        
        # シンプルな加重平均
        result[i] = ti_weight * trend_index[i] + zadx_weight * zadx[i] + er_weight * er_abs
    
    return result


@njit(fastmath=True)
def calculate_sigmoid_enhanced_combination(trend_index: np.ndarray, zadx: np.ndarray, er: np.ndarray, ti_weight: float = 0.4, zadx_weight: float = 0.4) -> np.ndarray:
    """
    トレンドインデックス、ZADX、ERのシグモイド強調加重平均（高速化版）
    
    Args:
        trend_index: Zトレンドインデックスの配列
        zadx: ZADX値の配列（0-1の範囲）
        er: 効率比の配列
        ti_weight: トレンドインデックスの重み（デフォルト: 0.4）
        zadx_weight: ZADXの重み（デフォルト: 0.4）
    
    Returns:
        組み合わせた値の配列
    """
    length = len(trend_index)
    result = np.zeros(length, dtype=np.float64)
    
    # 残りの重みはERに割り当て
    er_weight = 1.0 - ti_weight - zadx_weight
    
    # 重みの正規化（合計が1になるようにする）
    total_weight = ti_weight + zadx_weight + er_weight
    if total_weight != 1.0:
        ti_weight = ti_weight / total_weight
        zadx_weight = zadx_weight / total_weight
        er_weight = er_weight / total_weight
    
    for i in range(length):
        if np.isnan(trend_index[i]) or np.isnan(zadx[i]) or np.isnan(er[i]):
            result[i] = np.nan
            continue
        
        # ERの絶対値を使用
        er_abs = abs(er[i])
        
        # 基本の加重平均を計算
        base_combination = ti_weight * trend_index[i] + zadx_weight * zadx[i] + er_weight * er_abs
        
        # シグモイド関数による非線形強調
        # シグモイド関数: 1 / (1 + exp(-k * (x - 0.5)))
        # kは勾配を制御するパラメータ（大きいほど急勾配）
        k = 12.0  # 勾配パラメータ
        sigmoid_enhanced = 1.0 / (1.0 + np.exp(-k * (base_combination - 0.5)))
        
        result[i] = sigmoid_enhanced
    
    return result


@njit(fastmath=True, parallel=True)
def calculate_rms_combination(
    trend_index: np.ndarray, 
    zadx: np.ndarray,
    er: np.ndarray, 
    ti_weight: float = 0.4, 
    zadx_weight: float = 0.4,
    window: np.ndarray = None
) -> np.ndarray:
    """
    トレンドインデックス、ZADX、効率比（ER）の二乗平均平方根（RMS）による組み合わせ
    
    Args:
        trend_index: Zトレンドインデックスの配列
        zadx: ZADX値の配列（0-1の範囲）
        er: 効率比の配列
        ti_weight: トレンドインデックスの重み（デフォルト: 0.4）
        zadx_weight: ZADXの重み（デフォルト: 0.4）
        window: RMS計算のウィンドウサイズの配列（オプション）
    
    Returns:
        ZADXを含む組み合わせ値の配列
    """
    size = len(trend_index)
    result = np.zeros(size)
    
    # 窓幅が指定されていない場合はデフォルト値を使用
    if window is None:
        window = np.ones(size) * 5  # デフォルトの窓幅
    
    # 残りの重みはERに割り当て
    er_weight = 1.0 - ti_weight - zadx_weight
    
    # 重みの正規化（合計が1になるようにする）
    total_weight = ti_weight + zadx_weight + er_weight
    if total_weight != 1.0:
        ti_weight = ti_weight / total_weight
        zadx_weight = zadx_weight / total_weight
        er_weight = er_weight / total_weight
    
    # 各時点でのRMS組み合わせを計算
    for i in range(size):
        if np.isnan(trend_index[i]) or np.isnan(er[i]) or np.isnan(zadx[i]):
            result[i] = np.nan
            continue
        
        # 窓幅を整数に変換
        w = max(int(window[i]), 1)
        
        # 窓の範囲を取得（過去w個のポイント）
        start_idx = max(0, i - w + 1)
        
        # 窓内のデータを取得
        ti_window = trend_index[start_idx:i+1]
        zadx_window = zadx[start_idx:i+1]
        er_window = er[start_idx:i+1]
        
        # 各要素の二乗を計算
        ti_squared = np.power(ti_window, 2)
        zadx_squared = np.power(zadx_window, 2)
        er_squared = np.power(er_window, 2)
        
        # 加重平均を計算
        weighted_sum = (
            ti_weight * np.mean(ti_squared) + 
            zadx_weight * np.mean(zadx_squared) + 
            er_weight * np.mean(er_squared)
        )
        
        # 二乗平均平方根（RMS）を計算
        result[i] = np.sqrt(weighted_sum)
    
    return result


@njit(fastmath=True)
def calculate_dynamic_threshold(
    er: np.ndarray,
    max_threshold: float,
    min_threshold: float
) -> np.ndarray:
    """
    効率比に基づいて動的なしきい値を計算する（高速化版）
    
    Args:
        er: 効率比の配列
        max_threshold: しきい値の最大値
        min_threshold: しきい値の最小値
    
    Returns:
        動的なしきい値の配列
    """
    length = len(er)
    threshold = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        if np.isnan(er[i]):
            threshold[i] = np.nan
            continue
        
        # ERの絶対値を使用
        er_abs = abs(er[i])
        
        # ERが高いほど（トレンドが強いほど）しきい値は高く
        # ERが低いほど（レンジ相場ほど）しきい値は低く
        threshold[i] = min_threshold + er_abs * (max_threshold - min_threshold)
    
    return threshold


@njit(fastmath=True)
def calculate_z_trend_filter(
    trend_index: np.ndarray, 
    zadx: np.ndarray,  # ZADXの値（0-1の範囲）
    er: np.ndarray, 
    combination_weight: float, 
    zadx_weight: float,  # ZADX用の重み
    combination_method: int,
    rms_window: np.ndarray,
    max_threshold: float,
    min_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zトレンドフィルターを計算する
    
    Args:
        trend_index: Zトレンドインデックスの配列
        zadx: ZADX値の配列（0-1の範囲）
        er: 効率比の配列
        combination_weight: トレンドインデックスの重み
        zadx_weight: ZADX用の重み
        combination_method: 組み合わせ方法
            0: Sigmoid強調
            1: RMS（二乗平均平方根）
            2: Simple（シンプルな加重平均）
        rms_window: RMS計算に使用するウィンドウサイズの配列
        max_threshold: 最大しきい値
        min_threshold: 最小しきい値
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (組み合わせ値, 動的しきい値)
    """
    # 選択された組み合わせ方法に基づいて値を計算
    if combination_method == 1:  # RMS
        combined_values = calculate_rms_combination(
            trend_index, zadx, er, combination_weight, zadx_weight, rms_window
        )
    elif combination_method == 2:  # Simple
        combined_values = calculate_simple_average_combination(
            trend_index, zadx, er, combination_weight, zadx_weight
        )
    else:  # Sigmoid (default)
        combined_values = calculate_sigmoid_enhanced_combination(
            trend_index, zadx, er, combination_weight, zadx_weight
        )
    
    # 動的しきい値の計算
    dynamic_threshold = calculate_dynamic_threshold(
        er, max_threshold, min_threshold
    )
    
    return combined_values, dynamic_threshold


class ZTrendFilter(Indicator):
    """
    Zトレンドフィルター（ZTrendFilter）インジケーター
    
    Zトレンドインデックスとサイクル効率比（CER）を様々な方法で組み合わせた
    高度なトレンド/レンジ検出フィルターです。
    
    特徴:
    - ZTrendIndexを使用したトレンド/レンジ検出
    - サイクル効率比（CER）との複数の組み合わせ方法
    - 動的なしきい値による適応的フィルタリング
    - 3種類の組み合わせ方法:
      - Sigmoid強調: 非線形強調による分離
      - RMS: 二乗平均平方根による組み合わせ
      - Simple: シンプルな加重平均
    - すべてNumba最適化による高速計算
    
    CERはドミナントサイクル検出を使用して効率比を計算するため、
    より正確な市場状態の把握が可能になります。
    """
    
    def __init__(
        self,
        # ZTrendIndexのパラメータ
        max_stddev_period: int = 13,
        min_stddev_period: int = 5,
        max_lookback_period: int = 13,
        min_lookback_period: int = 5,
        # RMSウィンドウのパラメータ
        max_rms_window: int = 13,
        min_rms_window: int = 5,
        # しきい値のパラメータ
        max_threshold: float = 0.75,
        min_threshold: float = 0.55,
        # サイクル効率比(CER)のパラメーター
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 62,
        cycle_part: float = 0.5,
        # 組み合わせパラメータ
        combination_weight: float = 0.4,
        zadx_weight: float = 0.4,
        combination_method: str = "sigmoid",  # "sigmoid", "rms", "simple"
        # Zトレンドインデックスの追加パラメータ
        max_chop_dc_cycle_part: float = 0.5,
        max_chop_dc_max_cycle: int = 144,
        max_chop_dc_min_cycle: int = 10,
        max_chop_dc_max_output: int = 34,
        max_chop_dc_min_output: int = 13,
        min_chop_dc_cycle_part: float = 0.25,
        min_chop_dc_max_cycle: int = 55,
        min_chop_dc_min_cycle: int = 5,
        min_chop_dc_max_output: int = 13,
        min_chop_dc_min_output: int = 5,
        smoother_type: str = 'alma'  # 'alma'または'hyper'
    ):
        """
        コンストラクタ
        
        Args:
            max_chop_period: チョピネス期間の最大値
            min_chop_period: チョピネス期間の最小値
            max_atr_period: ATR期間の最大値
            min_atr_period: ATR期間の最小値
            max_stddev_period: 標準偏差期間の最大値
            min_stddev_period: 標準偏差期間の最小値
            max_lookback_period: ルックバック期間の最大値
            min_lookback_period: ルックバック期間の最小値
            max_rms_window: RMSウィンドウの最大サイズ
            min_rms_window: RMSウィンドウの最小サイズ
            max_threshold: しきい値の最大値
            min_threshold: しきい値の最小値
            cycle_detector_type: サイクル検出器の種類
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分
            combination_weight: 組み合わせの重み
            combination_method: 組み合わせ方法（"sigmoid", "rms", "simple"）
            max_chop_dc_cycle_part: 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_max_cycle: 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_min_cycle: 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_max_output: 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_min_output: 最大チョピネス期間用ドミナントサイクル設定
            min_chop_dc_cycle_part: 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_max_cycle: 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_min_cycle: 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_max_output: 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_min_output: 最小チョピネス期間用ドミナントサイクル設定
            smoother_type: 平滑化アルゴリズム（'alma'または'hyper'）
        """
        indicator_name = f"ZTrendFilter({combination_method}, {combination_weight}, {cycle_detector_type})"
        super().__init__(indicator_name)
        
        # ZTrendIndexのパラメータ
        self.max_stddev_period = max_stddev_period
        self.min_stddev_period = min_stddev_period
        self.max_lookback_period = max_lookback_period
        self.min_lookback_period = min_lookback_period
        
        # RMSウィンドウのパラメータ
        self.max_rms_window = max_rms_window
        self.min_rms_window = min_rms_window
        
        # しきい値のパラメータ
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        
        # CERのパラメータ
        self.cycle_detector_type = cycle_detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        
        # 組み合わせパラメータ
        self.combination_weight = combination_weight
        self.combination_method_str = combination_method.lower()
        
        # 組み合わせメソッドを設定
        if combination_method.lower() == "rms":
            self.combination_method = 1
        elif combination_method.lower() == "simple":
            self.combination_method = 2
        else:  # "sigmoid"
            self.combination_method = 0
        
        # ZADXの統合用パラメータ
        self.use_zadx = True
        self.zadx_weight = zadx_weight
        
        # ZADX（オプション）
        self.zadx = ZADX(
            max_dc_cycle_part=max_chop_dc_cycle_part,
            max_dc_max_cycle=max_chop_dc_max_cycle,
            max_dc_min_cycle=max_chop_dc_min_cycle,
            max_dc_max_output=max_chop_dc_max_output,
            max_dc_min_output=max_chop_dc_min_output,
            min_dc_cycle_part=min_chop_dc_cycle_part,
            min_dc_max_cycle=min_chop_dc_max_cycle,
            min_dc_min_cycle=min_chop_dc_min_cycle,
            min_dc_max_output=min_chop_dc_max_output,
            min_dc_min_output=min_chop_dc_min_output,
            er_period=21,
            smoother_type=smoother_type
        )
        
        # Zトレンドインデックスの追加パラメータ
        self.max_chop_dc_cycle_part = max_chop_dc_cycle_part
        self.max_chop_dc_max_cycle = max_chop_dc_max_cycle
        self.max_chop_dc_min_cycle = max_chop_dc_min_cycle
        self.max_chop_dc_max_output = max_chop_dc_max_output
        self.max_chop_dc_min_output = max_chop_dc_min_output
        self.min_chop_dc_cycle_part = min_chop_dc_cycle_part
        self.min_chop_dc_max_cycle = min_chop_dc_max_cycle
        self.min_chop_dc_min_cycle = min_chop_dc_min_cycle
        self.min_chop_dc_max_output = min_chop_dc_max_output
        self.min_chop_dc_min_output = min_chop_dc_min_output
        self.smoother_type = smoother_type
        
        # ZTrendIndexのインスタンス化
        self.z_trend_index = XTrendIndex(
            # 最大チョピネス期間用ドミナントサイクル設定
            max_chop_dc_cycle_part=self.max_chop_dc_cycle_part,
            max_chop_dc_max_cycle=self.max_chop_dc_max_cycle,
            max_chop_dc_min_cycle=self.max_chop_dc_min_cycle,
            max_chop_dc_max_output=self.max_chop_dc_max_output,
            max_chop_dc_min_output=self.max_chop_dc_min_output,
            
            # 最小チョピネス期間用ドミナントサイクル設定
            min_chop_dc_cycle_part=self.min_chop_dc_cycle_part,
            min_chop_dc_max_cycle=self.min_chop_dc_max_cycle,
            min_chop_dc_min_cycle=self.min_chop_dc_min_cycle,
            min_chop_dc_max_output=self.min_chop_dc_max_output,
            min_chop_dc_min_output=self.min_chop_dc_min_output,
            
            # 標準偏差と標準偏差ルックバック期間の設定
            max_stddev_period=self.max_stddev_period,
            min_stddev_period=self.min_stddev_period,
            max_lookback_period=self.max_lookback_period,
            min_lookback_period=self.min_lookback_period,
            
            # サイクル効率比(CER)のパラメーター
            cycle_detector_type=self.cycle_detector_type,
            lp_period=self.lp_period,
            hp_period=self.hp_period,
            cycle_part=self.cycle_part,
            
            # ZATR用パラメータ
            smoother_type=self.smoother_type
        )
        
        # サイクル効率比を計算するためのインスタンス
        self.cycle_er = CycleEfficiencyRatio(
            cycle_detector_type=self.cycle_detector_type,
            lp_period=self.lp_period,
            hp_period=self.hp_period,
            cycle_part=self.cycle_part
        )
        
        # 結果のキャッシュ
        self._result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            cols = ['open', 'high', 'low', 'close']
            data_hash = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            # NumPy配列の場合は全体をハッシュする
            data_hash = hash(tuple(map(tuple, data)))
        
        # パラメータ値を含めることで、同じデータでもパラメータが異なる場合に再計算する
        param_str = f"{self.combination_method}_{self.combination_weight}_{self.max_threshold}_{self.min_threshold}"
        param_str += f"_{self.cycle_detector_type}_{self.lp_period}_{self.hp_period}_{self.cycle_part}"
        
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ZTrendFilterResult:
        """
        Zトレンドフィルターを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
                NumPy配列の場合、1次元の価格データか、複数列の場合は3列目が終値
        
        Returns:
            ZTrendFilterResult: 計算結果を含むオブジェクト
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # Zトレンドインデックスを計算
            trend_index_result = self.z_trend_index.calculate(data)
            # Zトレンドインデックスの値（np.ndarray）を取得
            trend_index = trend_index_result.values
            
            # ZADX計算
            zadx_values = self.zadx.calculate(data)
            
            # サイクル効率比（ER）を計算
            er = self.cycle_er.calculate(data)
            
            # RMSウィンドウを計算
            if hasattr(self.cycle_er, 'get_dominant_cycle_values'):
                # 効率比から周期データが取得できる場合
                dominant_cycle = self.cycle_er.get_dominant_cycle_values()
                if len(dominant_cycle) > 0:
                    rms_window = np.clip(
                        dominant_cycle * 0.5, 
                        self.min_rms_window, 
                        self.max_rms_window
                    )
                else:
                    rms_window = np.ones(len(er)) * self.min_rms_window
            else:
                # 単純な固定値を使用
                rms_window = np.ones(len(er)) * self.min_rms_window
            
            # Zトレンドフィルターを計算
            combined_values, dynamic_threshold = calculate_z_trend_filter(
                trend_index,
                zadx_values,  # ZADXの値（0-1の範囲）
                er,
                self.combination_weight,
                self.zadx_weight,  # ZADX用の重み
                self.combination_method,
                rms_window,
                self.max_threshold,
                self.min_threshold
            )
            
            # 結果を保存
            self._result = ZTrendFilterResult(
                values=combined_values,
                trend_index=trend_index,
                zadx=zadx_values,
                er=er,
                combined_rms=combined_values,
                rms_window=rms_window,
                dynamic_threshold=dynamic_threshold
            )
            
            self._values = combined_values  # 基底クラスの要件を満たすため
            
            return self._result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ZTrendFilter計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時に空の結果を返す
            return ZTrendFilterResult(
                values=np.array([]),
                trend_index=np.array([]),
                zadx=np.array([]),
                er=np.array([]),
                combined_rms=np.array([]),
                rms_window=np.array([]),
                dynamic_threshold=np.array([])
            )
    
    def get_zadx(self) -> np.ndarray:
        """
        ZADX値を取得する
        
        Returns:
            np.ndarray: ZADX値
        """
        if self._result is None:
            return np.array([])
        return self._result.zadx
    
    def get_trend_index(self) -> np.ndarray:
        """
        Zトレンドインデックスの値を取得する
        
        Returns:
            np.ndarray: Zトレンドインデックスの値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.trend_index
    
    def get_efficiency_ratio(self) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.er
    
    def get_combined_rms(self) -> np.ndarray:
        """
        組み合わせ値を取得する
        
        Returns:
            np.ndarray: 組み合わせ値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.combined_rms
    
    def get_rms_window(self) -> np.ndarray:
        """
        RMS計算ウィンドウサイズを取得する
        
        Returns:
            np.ndarray: RMS計算ウィンドウサイズ
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.rms_window
    
    def get_dynamic_threshold(self) -> np.ndarray:
        """
        動的しきい値を取得する
        
        Returns:
            np.ndarray: 動的しきい値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.dynamic_threshold
    
    def get_state(self) -> np.ndarray:
        """
        フィルター状態を取得する（値がしきい値を上回るとトレンド、下回るとレンジ）
        
        Returns:
            np.ndarray: フィルター状態（1=トレンド、0=レンジ、NaN=不明）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        length = len(self._result.values)
        state = np.full(length, np.nan)
        
        for i in range(length):
            if np.isnan(self._result.values[i]) or np.isnan(self._result.dynamic_threshold[i]):
                continue
            
            if self._result.values[i] >= self._result.dynamic_threshold[i]:
                state[i] = 1.0  # トレンド
            else:
                state[i] = 0.0  # レンジ
        
        return state
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self.z_trend_index.reset()
        self.cycle_er.reset()
        self.zadx.reset() 