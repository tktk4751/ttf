#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import jit, vectorize, njit, prange

from .indicator import Indicator
from .z_adx import ZADX
from .z_rsx import ZRSX
from .cycle_efficiency_ratio import CycleEfficiencyRatio
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .cycle.ehlers_hody_dc import EhlersHoDyDC


@dataclass
class ZLongReversalIndexResult:
    """Zロングリバーサルインデックスの計算結果"""
    values: np.ndarray          # Zロングリバーサルインデックスの値（0-1の範囲で正規化）
    zadx: np.ndarray            # ZADX値（0-1の範囲）
    zrsx: np.ndarray            # ZRSX値（0-100の範囲）
    er: np.ndarray              # サイクル効率比（CER）
    combined_rms: np.ndarray    # ZADX、ZRSX、ERの二乗平均平方根（RMS）
    rms_window: np.ndarray      # RMS計算のウィンドウサイズ
    dynamic_threshold: np.ndarray  # 動的しきい値


@njit(fastmath=True)
def calculate_simple_average_combination(zadx: np.ndarray, zrsx: np.ndarray, er: np.ndarray, 
                                         zadx_weight: float = 0.4, zrsx_weight: float = 0.4) -> np.ndarray:
    """
    ZADX、ZRSX、ERの単純加重平均（高速化版）
    
    Args:
        zadx: ZADX値の配列（0-1の範囲）
        zrsx: ZRSX値の配列（0-1の範囲に正規化済み）
        er: 効率比の配列
        zadx_weight: ZADXの重み（デフォルト: 0.4）
        zrsx_weight: ZRSXの重み（デフォルト: 0.4）
    
    Returns:
        組み合わせた値の配列
    """
    length = len(zadx)
    result = np.zeros(length, dtype=np.float64)
    
    # 残りの重みはERに割り当て
    er_weight = 1.0 - zadx_weight - zrsx_weight
    
    # 重みの正規化（合計が1になるようにする）
    total_weight = zadx_weight + zrsx_weight + er_weight
    if total_weight != 1.0:
        zadx_weight = zadx_weight / total_weight
        zrsx_weight = zrsx_weight / total_weight
        er_weight = er_weight / total_weight
    
    for i in range(length):
        if np.isnan(zadx[i]) or np.isnan(zrsx[i]) or np.isnan(er[i]):
            result[i] = np.nan
            continue
        
        # ERの絶対値を使用
        er_abs = abs(er[i])
        
        # シンプルな加重平均
        result[i] = zadx_weight * zadx[i] + zrsx_weight * zrsx[i] + er_weight * er_abs
    
    return result


@njit(fastmath=True)
def calculate_sigmoid_enhanced_combination(zadx: np.ndarray, zrsx: np.ndarray, er: np.ndarray, 
                                          zadx_weight: float = 0.4, zrsx_weight: float = 0.4) -> np.ndarray:
    """
    ZADX、ZRSX、ERのシグモイド強調加重平均（高速化版）
    
    Args:
        zadx: ZADX値の配列（0-1の範囲）
        zrsx: ZRSX値の配列（0-1の範囲に正規化済み）
        er: 効率比の配列
        zadx_weight: ZADXの重み（デフォルト: 0.4）
        zrsx_weight: ZRSXの重み（デフォルト: 0.4）
    
    Returns:
        組み合わせた値の配列
    """
    length = len(zadx)
    result = np.zeros(length, dtype=np.float64)
    
    # 残りの重みはERに割り当て
    er_weight = 1.0 - zadx_weight - zrsx_weight
    
    # 重みの正規化（合計が1になるようにする）
    total_weight = zadx_weight + zrsx_weight + er_weight
    if total_weight != 1.0:
        zadx_weight = zadx_weight / total_weight
        zrsx_weight = zrsx_weight / total_weight
        er_weight = er_weight / total_weight
    
    for i in range(length):
        if np.isnan(zadx[i]) or np.isnan(zrsx[i]) or np.isnan(er[i]):
            result[i] = np.nan
            continue
        
        # ERの絶対値を使用
        er_abs = abs(er[i])
        
        # 基本の加重平均を計算
        base_combination = zadx_weight * zadx[i] + zrsx_weight * zrsx[i] + er_weight * er_abs
        
        # シグモイド関数による非線形強調
        # シグモイド関数: 1 / (1 + exp(-k * (x - 0.5)))
        # kは勾配を制御するパラメータ（大きいほど急勾配）
        k = 12.0  # 勾配パラメータ
        sigmoid_enhanced = 1.0 / (1.0 + np.exp(-k * (base_combination - 0.5)))
        
        result[i] = sigmoid_enhanced
    
    return result


@njit(fastmath=True, parallel=True)
def calculate_rms_combination(
    zadx: np.ndarray, 
    zrsx: np.ndarray,
    er: np.ndarray, 
    zadx_weight: float = 0.4, 
    zrsx_weight: float = 0.4,
    window: np.ndarray = None
) -> np.ndarray:
    """
    ZADX、ZRSX、効率比（ER）の二乗平均平方根（RMS）による組み合わせ
    
    Args:
        zadx: ZADX値の配列（0-1の範囲）
        zrsx: ZRSX値の配列（0-1の範囲に正規化済み）
        er: 効率比の配列
        zadx_weight: ZADXの重み（デフォルト: 0.4）
        zrsx_weight: ZRSXの重み（デフォルト: 0.4）
        window: RMS計算のウィンドウサイズの配列（オプション）
    
    Returns:
        ZADX、ZRSX、ERを含む組み合わせ値の配列
    """
    size = len(zadx)
    result = np.zeros(size)
    
    # 窓幅が指定されていない場合はデフォルト値を使用
    if window is None:
        window = np.ones(size) * 5  # デフォルトの窓幅
    
    # 残りの重みはERに割り当て
    er_weight = 1.0 - zadx_weight - zrsx_weight
    
    # 重みの正規化（合計が1になるようにする）
    total_weight = zadx_weight + zrsx_weight + er_weight
    if total_weight != 1.0:
        zadx_weight = zadx_weight / total_weight
        zrsx_weight = zrsx_weight / total_weight
        er_weight = er_weight / total_weight
    
    # 各時点でのRMS組み合わせを計算
    for i in range(size):
        if np.isnan(zadx[i]) or np.isnan(er[i]) or np.isnan(zrsx[i]):
            result[i] = np.nan
            continue
        
        # 窓幅を整数に変換
        w = max(int(window[i]), 1)
        
        # 窓の範囲を取得（過去w個のポイント）
        start_idx = max(0, i - w + 1)
        
        # 窓内のデータを取得
        zadx_window = zadx[start_idx:i+1]
        zrsx_window = zrsx[start_idx:i+1]
        er_window = er[start_idx:i+1]
        
        # 各要素の二乗を計算
        zadx_squared = np.power(zadx_window, 2)
        zrsx_squared = np.power(zrsx_window, 2)
        er_squared = np.power(np.abs(er_window), 2)
        
        # 加重平均を計算
        weighted_sum = (
            zadx_weight * np.mean(zadx_squared) + 
            zrsx_weight * np.mean(zrsx_squared) + 
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
def calculate_z_reversal_index(
    zadx: np.ndarray,  # ZADXの値（0-1の範囲）
    zrsx: np.ndarray,  # ZRSXの値（0-100の範囲）
    er: np.ndarray, 
    zadx_weight: float,  # ZADX用の重み
    zrsx_weight: float,  # ZRSX用の重み
    combination_method: int,
    rms_window: np.ndarray,
    max_threshold: float,
    min_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zロングリバーサルインデックスを計算する
    
    Args:
        zadx: ZADX値の配列（0-1の範囲）
        zrsx: ZRSX値の配列（0-100の範囲）
        er: 効率比の配列
        zadx_weight: ZADX用の重み
        zrsx_weight: ZRSX用の重み
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
            zadx, zrsx, er, zadx_weight, zrsx_weight, rms_window
        )
    elif combination_method == 2:  # Simple
        combined_values = calculate_simple_average_combination(
            zadx, zrsx, er, zadx_weight, zrsx_weight
        )
    else:  # Sigmoid (default)
        combined_values = calculate_sigmoid_enhanced_combination(
            zadx, zrsx, er, zadx_weight, zrsx_weight
        )
    
    # 動的しきい値の計算
    dynamic_threshold = calculate_dynamic_threshold(
        er, max_threshold, min_threshold
    )
    
    return combined_values, dynamic_threshold


class ZLongReversalIndex(Indicator):
    """
    Zロングリバーサルインデックス（ZLongReversalIndex）インジケーター
    
    ZADXとZRSXを様々な方法で組み合わせた高度なロングリバーサル検出インデックスです。
    
    特徴:
    - ZADXを使用したトレンド強度の検出
    - ZRSXを使用した過買い/過売りの検出
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
        # RMSウィンドウのパラメータ
        max_rms_window: int = 13,
        min_rms_window: int = 5,
        # しきい値のパラメータ
        max_threshold: float = 0.9,
        min_threshold: float = 0.75,
        # サイクル効率比(CER)のパラメーター
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 62,
        cycle_part: float = 0.5,
        # 組み合わせパラメータ
        zadx_weight: float = 0.4,
        zrsx_weight: float = 0.4,
        combination_method: str = "sigmoid",  # "sigmoid", "rms", "simple"
        
        # ZADX用パラメータ
        zadx_max_dc_cycle_part: float = 0.5,
        zadx_max_dc_max_cycle: int = 34,
        zadx_max_dc_min_cycle: int = 5,
        zadx_max_dc_max_output: int = 21,
        zadx_max_dc_min_output: int = 8,
        zadx_min_dc_cycle_part: float = 0.25,
        zadx_min_dc_max_cycle: int = 21,
        zadx_min_dc_min_cycle: int = 3,
        zadx_min_dc_max_output: int = 13,
        zadx_min_dc_min_output: int = 3,
        zadx_er_period: int = 21,
        
        # ZRSX用パラメータ
        zrsx_max_dc_cycle_part: float = 0.5,
        zrsx_max_dc_max_cycle: int = 55,
        zrsx_max_dc_min_cycle: int = 5,
        zrsx_max_dc_max_output: int = 21,
        zrsx_max_dc_min_output: int = 10,
        zrsx_min_dc_cycle_part: float = 0.25,
        zrsx_min_dc_max_cycle: int = 34,
        zrsx_min_dc_min_cycle: int = 3,
        zrsx_min_dc_max_output: int = 10,
        zrsx_min_dc_min_output: int = 5,
        zrsx_er_period: int = 10,
        
        smoother_type: str = 'alma'  # 'alma'または'hyper'
    ):
        """
        コンストラクタ
        
        Args:
            max_rms_window: RMSウィンドウの最大サイズ
            min_rms_window: RMSウィンドウの最小サイズ
            max_threshold: しきい値の最大値
            min_threshold: しきい値の最小値
            cycle_detector_type: サイクル検出器の種類
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            cycle_part: サイクル部分
            zadx_weight: ZADX用の重み
            zrsx_weight: ZRSX用の重み
            combination_method: 組み合わせ方法（"sigmoid", "rms", "simple"）
            
            zadx_max_dc_cycle_part: ZADX用ドミナントサイクル設定
            zadx_max_dc_max_cycle: ZADX用ドミナントサイクル設定
            zadx_max_dc_min_cycle: ZADX用ドミナントサイクル設定
            zadx_max_dc_max_output: ZADX用ドミナントサイクル設定
            zadx_max_dc_min_output: ZADX用ドミナントサイクル設定
            zadx_min_dc_cycle_part: ZADX用ドミナントサイクル設定
            zadx_min_dc_max_cycle: ZADX用ドミナントサイクル設定
            zadx_min_dc_min_cycle: ZADX用ドミナントサイクル設定
            zadx_min_dc_max_output: ZADX用ドミナントサイクル設定
            zadx_min_dc_min_output: ZADX用ドミナントサイクル設定
            zadx_er_period: ZADX用効率比期間
            
            zrsx_max_dc_cycle_part: ZRSX用ドミナントサイクル設定
            zrsx_max_dc_max_cycle: ZRSX用ドミナントサイクル設定
            zrsx_max_dc_min_cycle: ZRSX用ドミナントサイクル設定
            zrsx_max_dc_max_output: ZRSX用ドミナントサイクル設定
            zrsx_max_dc_min_output: ZRSX用ドミナントサイクル設定
            zrsx_min_dc_cycle_part: ZRSX用ドミナントサイクル設定
            zrsx_min_dc_max_cycle: ZRSX用ドミナントサイクル設定
            zrsx_min_dc_min_cycle: ZRSX用ドミナントサイクル設定
            zrsx_min_dc_max_output: ZRSX用ドミナントサイクル設定
            zrsx_min_dc_min_output: ZRSX用ドミナントサイクル設定
            zrsx_er_period: ZRSX用効率比期間
            
            smoother_type: 平滑化アルゴリズム（'alma'または'hyper'）
        """
        indicator_name = f"ZLongReversalIndex({combination_method}, {zadx_weight}, {zrsx_weight}, {cycle_detector_type})"
        super().__init__(indicator_name)
        
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
        self.zadx_weight = zadx_weight
        self.zrsx_weight = zrsx_weight
        self.combination_method_str = combination_method.lower()
        
        # 組み合わせメソッドを設定
        if combination_method.lower() == "rms":
            self.combination_method = 1
        elif combination_method.lower() == "simple":
            self.combination_method = 2
        else:  # "sigmoid"
            self.combination_method = 0
        
        # 共通パラメータ
        self.smoother_type = smoother_type
        
        # ZADX
        self.zadx = ZADX(
            max_dc_cycle_part=zadx_max_dc_cycle_part,
            max_dc_max_cycle=zadx_max_dc_max_cycle,
            max_dc_min_cycle=zadx_max_dc_min_cycle,
            max_dc_max_output=zadx_max_dc_max_output,
            max_dc_min_output=zadx_max_dc_min_output,
            min_dc_cycle_part=zadx_min_dc_cycle_part,
            min_dc_max_cycle=zadx_min_dc_max_cycle,
            min_dc_min_cycle=zadx_min_dc_min_cycle,
            min_dc_max_output=zadx_min_dc_max_output,
            min_dc_min_output=zadx_min_dc_min_output,
            er_period=zadx_er_period,
            smoother_type=smoother_type
        )
        
        # ZRSX
        self.zrsx = ZRSX(
            max_dc_cycle_part=zrsx_max_dc_cycle_part,
            max_dc_max_cycle=zrsx_max_dc_max_cycle,
            max_dc_min_cycle=zrsx_max_dc_min_cycle,
            max_dc_max_output=zrsx_max_dc_max_output,
            max_dc_min_output=zrsx_max_dc_min_output,
            min_dc_cycle_part=zrsx_min_dc_cycle_part,
            min_dc_max_cycle=zrsx_min_dc_max_cycle,
            min_dc_min_cycle=zrsx_min_dc_min_cycle,
            min_dc_max_output=zrsx_min_dc_max_output,
            min_dc_min_output=zrsx_min_dc_min_output,
            er_period=zrsx_er_period
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
        param_str = f"{self.combination_method}_{self.zadx_weight}_{self.zrsx_weight}_{self.max_threshold}_{self.min_threshold}"
        param_str += f"_{self.cycle_detector_type}_{self.lp_period}_{self.hp_period}_{self.cycle_part}"
        
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ZLongReversalIndexResult:
        """
        Zロングリバーサルインデックスを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
                NumPy配列の場合、1次元の価格データか、複数列の場合は3列目が終値
        
        Returns:
            ZLongReversalIndexResult: 計算結果を含むオブジェクト
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # ZADX計算
            zadx_values = self.zadx.calculate(data)
            
            # ZRSX計算
            zrsx_values = self.zrsx.calculate(data)
            
            # ZRSXの値を0-1の範囲に正規化（元々は0-100の範囲）
            normalized_zrsx_values = zrsx_values / 100.0
            
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
            
            # Zロングリバーサルインデックスを計算
            combined_values, dynamic_threshold = calculate_z_reversal_index(
                zadx_values,
                normalized_zrsx_values,  # 正規化したZRSX値を使用
                er,
                self.zadx_weight,
                self.zrsx_weight,
                self.combination_method,
                rms_window,
                self.max_threshold,
                self.min_threshold
            )
            
            # 結果を保存
            self._result = ZLongReversalIndexResult(
                values=combined_values,
                zadx=zadx_values,
                zrsx=zrsx_values,  # 元の0-100の範囲のZRSX値を保存
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
            self.logger.error(f"ZLongReversalIndex計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時に空の結果を返す
            return ZLongReversalIndexResult(
                values=np.array([]),
                zadx=np.array([]),
                zrsx=np.array([]),
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
    
    def get_zrsx(self) -> np.ndarray:
        """
        ZRSX値を取得する（元の0-100の範囲）
        
        Returns:
            np.ndarray: ZRSX値（0-100の範囲）
        """
        if self._result is None:
            return np.array([])
        return self._result.zrsx
    
    def get_normalized_zrsx(self) -> np.ndarray:
        """
        正規化されたZRSX値を取得する（0-1の範囲）
        
        Returns:
            np.ndarray: 正規化されたZRSX値（0-1の範囲）
        """
        if self._result is None:
            return np.array([])
        return self._result.zrsx / 100.0
    
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
        インデックス状態を取得する（値がしきい値を上回ると反転シグナル、下回ると非反転）
        
        Returns:
            np.ndarray: インデックス状態（1=反転シグナル、0=非反転、NaN=不明）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        
        length = len(self._result.values)
        state = np.full(length, np.nan)
        
        for i in range(length):
            if np.isnan(self._result.values[i]) or np.isnan(self._result.dynamic_threshold[i]):
                continue
            
            if self._result.values[i] >= self._result.dynamic_threshold[i]:
                state[i] = 1.0  # 反転シグナル
            else:
                state[i] = 0.0  # 非反転
        
        return state
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self.zadx.reset()
        self.zrsx.reset()
        self.cycle_er.reset() 


# 後方互換性のために、ZReversalIndexをZLongReversalIndexのエイリアスとして定義
ZReversalIndex = ZLongReversalIndex 