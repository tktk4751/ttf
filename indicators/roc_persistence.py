#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Optional
import numpy as np
import pandas as pd
from numba import njit, prange
from dataclasses import dataclass

from .indicator import Indicator
from .cycle_roc import CycleROC


@dataclass
class ROCPersistenceResult:
    """ROC継続性の計算結果"""
    values: np.ndarray                  # ROC継続性の値（-1から1）
    persistence_periods: np.ndarray     # 現在の継続期間
    roc_directions: np.ndarray         # ROCの方向（1=正、-1=負、0=ゼロ）
    roc_values: np.ndarray             # 元のROC値
    cycle_periods: np.ndarray          # サイクル期間


@njit
def calculate_roc_persistence_numba(roc_values: np.ndarray, max_periods: int = 144) -> tuple:
    """
    ROC継続性を計算するNumba最適化関数
    
    Args:
        roc_values: ROC値の配列
        max_periods: 最大継続期間（デフォルト144）
    
    Returns:
        tuple: (persistence_values, persistence_periods, directions)
    """
    n = len(roc_values)
    persistence_values = np.full(n, np.nan, dtype=np.float64)
    persistence_periods = np.full(n, 0, dtype=np.int32)
    directions = np.full(n, 0, dtype=np.int32)
    
    if n == 0:
        return persistence_values, persistence_periods, directions
    
    current_direction = 0  # 0=初期状態、1=正、-1=負
    current_periods = 0    # 現在の継続期間
    
    for i in range(n):
        if np.isnan(roc_values[i]):
            persistence_values[i] = np.nan
            persistence_periods[i] = current_periods
            directions[i] = current_direction
            continue
        
        # ROCの方向を判定
        if roc_values[i] > 0.0:
            new_direction = 1  # 正
        elif roc_values[i] < 0.0:
            new_direction = -1  # 負
        else:
            new_direction = 0  # ゼロ
        
        # 方向が変わったかチェック
        if new_direction != current_direction:
            # 方向が変わったので期間をリセット
            current_direction = new_direction
            current_periods = 1
        else:
            # 同じ方向なので期間を増加
            current_periods += 1
        
        # 最大期間の制限
        if current_periods > max_periods:
            current_periods = max_periods
        
        # 継続性の値を計算（-1から1の範囲）
        if current_direction == 1:
            # 正の方向：0から1の範囲
            persistence_value = min(current_periods / max_periods, 1.0)
        elif current_direction == -1:
            # 負の方向：0から-1の範囲
            persistence_value = -min(current_periods / max_periods, 1.0)
        else:
            # ゼロの場合
            persistence_value = 0.0
        
        persistence_values[i] = persistence_value
        persistence_periods[i] = current_periods
        directions[i] = current_direction
    
    return persistence_values, persistence_periods, directions


@njit
def calculate_smoothed_persistence(persistence_values: np.ndarray, smooth_period: int = 3) -> np.ndarray:
    """
    継続性の値を平滑化する
    
    Args:
        persistence_values: 継続性の値
        smooth_period: 平滑化期間
    
    Returns:
        平滑化された継続性の値
    """
    n = len(persistence_values)
    smoothed = np.full(n, np.nan, dtype=np.float64)
    
    if n == 0 or smooth_period <= 0:
        return persistence_values.copy()
    
    for i in range(n):
        if np.isnan(persistence_values[i]):
            smoothed[i] = np.nan
            continue
            
        # 平滑化期間の開始インデックス
        start_idx = max(0, i - smooth_period + 1)
        
        # 有効な値の平均を計算
        valid_count = 0
        sum_values = 0.0
        
        for j in range(start_idx, i + 1):
            if not np.isnan(persistence_values[j]):
                sum_values += persistence_values[j]
                valid_count += 1
        
        if valid_count > 0:
            smoothed[i] = sum_values / valid_count
        else:
            smoothed[i] = np.nan
    
    return smoothed


class ROCPersistence(Indicator):
    """
    ROC継続性インジケーター
    
    ROCが正または負の領域にどれだけ長く滞在しているかを測定し、
    -1から1の範囲で継続性を表現します。
    
    特徴:
    - ROCが正の領域に長くいるほど1に近づく
    - ROCが負の領域に長くいるほど-1に近づく  
    - 最大144期間で飽和（自動的に1または-1になる）
    - サイクルROCをベースに動的期間を使用
    - Numba最適化で高速計算
    """
    
    def __init__(
        self,
        # CycleROCのパラメータ
        detector_type: str = 'hody_e',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 89,
        min_output: int = 13,
        src_type: str = 'hlc3',
        smooth_roc: bool = True,
        roc_alma_period: int = 5,
        roc_alma_offset: float = 0.85,
        roc_alma_sigma: float = 6,
        signal_threshold: float = 0.0,
        # ROC継続性のパラメータ
        max_persistence_periods: int = 144,
        smooth_persistence: bool = False,
        persistence_smooth_period: int = 3
    ):
        """
        コンストラクタ
        
        Args:
            # CycleROCのパラメータは全て継承
            max_persistence_periods: 最大継続期間（デフォルト144）
            smooth_persistence: 継続性値の平滑化を行うかどうか
            persistence_smooth_period: 継続性値の平滑化期間
        """
        smooth_str = f"_smooth={'Y' if smooth_persistence else 'N'}" if smooth_persistence else ""
        indicator_name = f"ROCPersist(det={detector_type},max={max_persistence_periods},src={src_type}{smooth_str})"
        super().__init__(indicator_name)
        
        # ROC継続性のパラメータ
        self.max_persistence_periods = max_persistence_periods
        self.smooth_persistence = smooth_persistence
        self.persistence_smooth_period = persistence_smooth_period
        
        # CycleROCインジケーターを作成
        self.cycle_roc = CycleROC(
            detector_type=detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            smooth_roc=smooth_roc,
            roc_alma_period=roc_alma_period,
            roc_alma_offset=roc_alma_offset,
            roc_alma_sigma=roc_alma_sigma,
            signal_threshold=signal_threshold
        )
        
        # 結果キャッシュ
        self._values = None
        self._data_hash = None
        self._result = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する"""
        # CycleROCのハッシュを基本として、独自パラメータを追加
        base_hash = str(hash(str(data)))  # 簡略化されたデータハッシュ
        
        param_str = (
            f"max_persist={self.max_persistence_periods}_"
            f"smooth_persist={self.smooth_persistence}_{self.persistence_smooth_period}_"
            f"cycle_roc_params={self.cycle_roc}"
        )
        return f"{base_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ROC継続性を計算
        
        Args:
            data: OHLC価格データ（DataFrameまたはNumpy配列）
        
        Returns:
            np.ndarray: ROC継続性の値（-1から1の範囲）
        """
        try:
            # ハッシュチェックでキャッシュ利用
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._values is not None:
                return self._values
            
            # CycleROCを計算
            roc_values = self.cycle_roc.calculate(data)
            roc_result = self.cycle_roc.get_result()
            
            if roc_result is None:
                data_len = len(data) if hasattr(data, '__len__') else 0
                self._values = np.full(data_len, np.nan)
                self._data_hash = None
                return self._values
            
            # Numba最適化関数でROC継続性を計算
            persistence_values, persistence_periods, directions = calculate_roc_persistence_numba(
                roc_values, self.max_persistence_periods
            )
            
            # 平滑化（有効な場合）
            final_values = persistence_values
            if self.smooth_persistence:
                final_values = calculate_smoothed_persistence(
                    persistence_values, self.persistence_smooth_period
                )
            
            # 結果を保存してキャッシュ
            self._values = final_values
            self._data_hash = data_hash
            
            # 結果オブジェクトを作成
            self._result = ROCPersistenceResult(
                values=final_values,
                persistence_periods=persistence_periods,
                roc_directions=directions,
                roc_values=roc_values,
                cycle_periods=roc_result.cycle_periods
            )
            
            return final_values
            
        except Exception as e:
            import traceback
            error_msg = f"ROC継続性計算中にエラー: {str(e)}"
            stack_trace = traceback.format_exc()
            self.logger.error(f"{error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            
            # エラー時はNaN配列
            self._values = np.full(data_len, np.nan)
            self._data_hash = None
            
            # エラー時の結果オブジェクト
            self._result = ROCPersistenceResult(
                values=np.full(data_len, np.nan),
                persistence_periods=np.full(data_len, 0, dtype=np.int32),
                roc_directions=np.full(data_len, 0, dtype=np.int32),
                roc_values=np.full(data_len, np.nan),
                cycle_periods=np.full(data_len, np.nan)
            )
            
            return self._values
    
    def get_result(self) -> Optional[ROCPersistenceResult]:
        """
        計算結果全体を取得する
        
        Returns:
            ROCPersistenceResult: 計算結果オブジェクト
        """
        return self._result
    
    def get_persistence_periods(self) -> np.ndarray:
        """
        現在の継続期間を取得する
        
        Returns:
            np.ndarray: 継続期間の配列
        """
        if self._result is not None:
            return self._result.persistence_periods
        return np.array([])
    
    def get_roc_directions(self) -> np.ndarray:
        """
        ROCの方向を取得する (1=正、-1=負、0=ゼロ)
        
        Returns:
            np.ndarray: ROC方向の配列
        """
        if self._result is not None:
            return self._result.roc_directions
        return np.array([])
    
    def get_roc_values(self) -> np.ndarray:
        """
        元のROC値を取得する
        
        Returns:
            np.ndarray: ROC値の配列
        """
        if self._result is not None:
            return self._result.roc_values
        return np.array([])
    
    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._values = None
        self._result = None
        self._data_hash = None
        if hasattr(self.cycle_roc, 'reset'):
            self.cycle_roc.reset()
    
    def __str__(self) -> str:
        """文字列表現"""
        smooth_str = f", smooth={self.smooth_persistence}" if self.smooth_persistence else ""
        return f"ROCPersistence(max={self.max_persistence_periods}{smooth_str})" 