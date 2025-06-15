#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Any
import numpy as np
import pandas as pd
from numba import jit, prange

try:
    from .indicator import Indicator
except ImportError:
    # スタンドアロン実行時の対応
    from indicator import Indicator

@dataclass
class DominantCycleResult:
    """ドミナントサイクル検出の計算結果"""
    values: np.ndarray        # ドミナントサイクルの値
    raw_period: np.ndarray    # 生の周期値（制限前）
    smooth_period: np.ndarray # 平滑化された周期値


class EhlersDominantCycle(Indicator):
    """
    エーラーズのドミナントサイクル検出の基底クラス
    
    このクラスはエーラーズによるドミナントサイクル検出アルゴリズムの共通実装を提供します。
    具体的なアルゴリズムは派生クラスで実装されます。
    """
    
    def __init__(
        self,
        name: str,
        cycle_part: float = 0.5,
        max_cycle: int = 50,
        min_cycle: int = 6,
        max_output: int = 34,
        min_output: int = 1
    ):
        """
        コンストラクタ
        
        Args:
            name: インディケーターの名前
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 50）
            min_cycle: 最小サイクル期間（デフォルト: 6）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 1）
        """
        super().__init__(name)
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self._result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合は必要なカラムのみハッシュする
            if 'close' in data.columns:
                data_hash = hash(tuple(data['close'].values))
            else:
                # closeカラムがない場合は全カラムのハッシュ
                data_hash = hash(tuple(map(tuple, data.values)))
        else:
            # NumPy配列の場合
            if data.ndim == 2 and data.shape[1] >= 4:
                # OHLCデータの場合はcloseだけハッシュ
                data_hash = hash(tuple(data[:, 3]))
            else:
                # それ以外は全体をハッシュ
                data_hash = hash(tuple(map(tuple, data)) if data.ndim == 2 else tuple(data))
        
        # パラメータ値を含める
        param_str = f"{self.cycle_part}_{self.max_cycle}_{self.min_cycle}_{self.max_output}_{self.min_output}"
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        ドミナントサイクルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'close'カラムが必要
        
        Returns:
            ドミナントサイクルの値
        """
        raise NotImplementedError("派生クラスで実装する必要があります")
    
    def limit_and_smooth_period(self, period: np.ndarray, previous_period: Optional[np.ndarray] = None) -> np.ndarray:
        """
        周期値に制限を適用し、平滑化する
        
        Args:
            period: 周期値配列
            previous_period: 前回の周期値（Noneの場合は制限なし）
            
        Returns:
            制限・平滑化された周期値
        """
        # NaNや無効値を処理
        result = np.copy(period)
        mask = np.isnan(result) | np.isinf(result)
        if mask.any():
            if previous_period is not None:
                result[mask] = previous_period[mask]
            else:
                result[mask] = self.min_cycle
        
        # 前回の値との急激な変化を制限（オプション）
        if previous_period is not None:
            # 1.5倍以上の増加を制限
            increase_mask = result > 1.5 * previous_period
            result[increase_mask] = 1.5 * previous_period[increase_mask]
            
            # 0.67倍以下の減少を制限
            decrease_mask = result < 0.67 * previous_period
            result[decrease_mask] = 0.67 * previous_period[decrease_mask]
        
        # 最小・最大周期の範囲に制限
        result = np.clip(result, self.min_cycle, self.max_cycle)
        
        return result
    
    def calculate_output_cycle(self, period: np.ndarray) -> np.ndarray:
        """
        出力サイクル値を計算する
        
        Args:
            period: 周期値配列
            
        Returns:
            出力用にスケールされたサイクル値
        """
        # CycPartを適用して出力値を計算
        output = np.ceil(period * self.cycle_part)
        
        # 出力範囲に制限
        output = np.clip(output, self.min_output, self.max_output)
        
        return output
    
    def get_raw_period(self) -> np.ndarray:
        """
        生の周期値を取得する
        
        Returns:
            np.ndarray: 生の周期値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.raw_period
    
    def get_smooth_period(self) -> np.ndarray:
        """
        平滑化された周期値を取得する
        
        Returns:
            np.ndarray: 平滑化された周期値
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.smooth_period
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None 