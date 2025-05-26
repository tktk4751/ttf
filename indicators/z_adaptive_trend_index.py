#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit

from .indicator import Indicator
from .x_trend_index import XTrendIndex
from .cycle_efficiency_ratio import CycleEfficiencyRatio


@dataclass
class ZAdaptiveTrendIndexResult:
    """Zアダプティブトレンドインデックスの計算結果"""
    values: np.ndarray            # Zアダプティブトレンドインデックスの値
    x_trend_values: np.ndarray    # Xトレンドインデックスの値
    er_values: np.ndarray         # サイクル効率比（CER）の値


@njit(fastmath=True)
def calculate_z_adaptive_trend_index(
    x_trend_values: np.ndarray,
    er_values: np.ndarray
) -> np.ndarray:
    """
    Zアダプティブトレンドインデックスを計算する

    Args:
        x_trend_values: Xトレンドインデックスの値
        er_values: サイクル効率比の値

    Returns:
        Zアダプティブトレンドインデックスの値
    """
    length = len(x_trend_values)
    result = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        x_val = x_trend_values[i]
        er_val = er_values[i]
        
        # どちらかがNaNの場合は結果もNaN
        if np.isnan(x_val) or np.isnan(er_val):
            result[i] = np.nan
            continue
        
        # XトレンドインデックスとERの平均値を計算
        # ERは-1から1の範囲なので、0～1に正規化
        normalized_er = (er_val + 1.0) / 2.0
        result[i] = (x_val + normalized_er) / 2.0
        
        # 値は0～1の範囲に制限
        result[i] = max(0.0, min(1.0, result[i]))
    
    return result


class ZAdaptiveTrendIndex(Indicator):
    """
    Zアダプティブトレンドインデックス

    Xトレンドインデックスとサイクル効率比（CER）を組み合わせて
    トレンドとレンジの状態をより正確に把握するためのインジケーター。
    
    特徴:
    - XトレンドインデックスとサイクルERの平均値を使用
    - ERの値は-1～1から0～1に正規化してから平均を取る
    - 最終的な値は0～1の範囲（1に近いほど強いトレンド、0に近いほど強いレンジ）
    """
    
    def __init__(
        self,
        # Xトレンドインデックスのパラメータをそのまま受け取り
        detector_type: str = 'phac_e',
        cycle_part: float = 0.5,
        max_cycle: int = 55,
        min_cycle: int = 5,
        max_output: int = 34,
        min_output: int = 5,
        src_type: str = 'hlc3',
        lp_period: int = 5,
        hp_period: int = 55,
        smoother_type: str = 'alma',
        
        # CycleEfficiencyRatioのパラメータ
        cer_detector_type: str = 'phac_e',
        cer_lp_period: int = 5,
        cer_hp_period: int = 144,
        cer_cycle_part: float = 0.5,
        cer_max_cycle: int = 144,
        cer_min_cycle: int = 5,
        cer_max_output: int = 55,
        cer_min_output: int = 5,
        cer_src_type: str = 'hlc3',
        use_kalman_filter: bool = False,
        
        # 動的しきい値のパラメータ
        max_threshold: float = 0.75,
        min_threshold: float = 0.55
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: EhlersUnifiedDCで使用する検出器タイプ
            cycle_part: DCのサイクル部分の倍率
            max_cycle: DCの最大サイクル期間
            min_cycle: DCの最小サイクル期間
            max_output: DCの最大出力値
            min_output: DCの最小出力値
            src_type: DC計算に使用する価格ソース
            lp_period: 拡張DC用のローパスフィルター期間
            hp_period: 拡張DC用のハイパスフィルター期間
            smoother_type: CATRで使用する平滑化アルゴリズム
            
            cer_detector_type: CycleERで使用する検出器タイプ
            cer_lp_period: CER用のローパスフィルター期間
            cer_hp_period: CER用のハイパスフィルター期間
            cer_cycle_part: CER用のサイクル部分の倍率
            cer_max_cycle: CER用のDCの最大サイクル期間
            cer_min_cycle: CER用のDCの最小サイクル期間
            cer_max_output: CER用のDCの最大出力値
            cer_min_output: CER用のDCの最小出力値
            cer_src_type: CER計算に使用する価格ソース
            use_kalman_filter: CERでカルマンフィルターを使用するか
            
            max_threshold: 動的しきい値の最大値
            min_threshold: 動的しきい値の最小値
        """
        super().__init__(
            f"ZAdaptiveTrendIndex({detector_type}, {cer_detector_type})"
        )
        
        # Xトレンドインデックスのインスタンス化
        self.x_trend_index = XTrendIndex(
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            lp_period=lp_period,
            hp_period=hp_period,
            smoother_type=smoother_type,

        )
        
        # サイクル効率比のインスタンス化
        self.cycle_er = CycleEfficiencyRatio(
            detector_type=cer_detector_type,
            lp_period=cer_lp_period,
            hp_period=cer_hp_period,
            cycle_part=cer_cycle_part,
            max_cycle=cer_max_cycle,
            min_cycle=cer_min_cycle,
            max_output=cer_max_output,
            min_output=cer_min_output,
            src_type=cer_src_type,
            use_kalman_filter=use_kalman_filter
        )
        
        # 結果キャッシュ
        self._result = None
        self._data_hash = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        if isinstance(data, pd.DataFrame):
            cols = ['open', 'high', 'low', 'close']
            data_hash_part = hash(tuple(map(tuple, (data[col].values for col in cols if col in data.columns))))
        else:
            data_hash_part = hash(tuple(map(tuple, data)))
        
        # XトレンドインデックスとCERのハッシュを組み合わせる
        x_trend_hash = self.x_trend_index._get_data_hash(data)
        cer_hash = self.cycle_er._get_data_hash(data)
        
        return f"{data_hash_part}_{x_trend_hash}_{cer_hash}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ZAdaptiveTrendIndexResult:
        """
        Zアダプティブトレンドインデックスを計算する

        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合は'open', 'high', 'low', 'close'カラムが必要

        Returns:
            ZAdaptiveTrendIndexResult: 計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result
            
            self._data_hash = data_hash
            
            # XトレンドインデックスとCERを計算
            x_trend_result = self.x_trend_index.calculate(data)
            x_trend_values = x_trend_result.values
            
            # CERの計算
            er_values = self.cycle_er.calculate(data)
            
            # Zアダプティブトレンドインデックスを計算
            z_adaptive_values = calculate_z_adaptive_trend_index(
                x_trend_values, er_values
            )
            
            # 結果オブジェクトを作成
            result = ZAdaptiveTrendIndexResult(
                values=z_adaptive_values,
                x_trend_values=x_trend_values,
                er_values=er_values
            )
            
            self._result = result
            self._values = z_adaptive_values  # Indicatorクラスの標準出力
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Zアダプティブトレンドインデックス計算中にエラー: {error_msg}\n{stack_trace}")
            
            n = len(data) if hasattr(data, '__len__') else 0
            empty_result = ZAdaptiveTrendIndexResult(
                values=np.full(n, np.nan),
                x_trend_values=np.full(n, np.nan),
                er_values=np.full(n, np.nan)
            )
            
            self._result = None
            self._values = np.full(n, np.nan)
            self._data_hash = None
            
            return empty_result
    
    # --- Getter Methods ---
    def get_result(self) -> Optional[ZAdaptiveTrendIndexResult]:
        """計算結果全体を取得する"""
        return self._result
    
    def get_x_trend_values(self) -> np.ndarray:
        """XトレンドインデックスのZ値を取得する"""
        if self._result is None:
            return np.array([])
        return self._result.x_trend_values
    
    def get_er_values(self) -> np.ndarray:
        """サイクル効率比（CER）を取得する"""
        if self._result is None:
            return np.array([])
        return self._result.er_values
    
    def reset(self) -> None:
        """インジケーターの状態をリセットする"""
        super().reset()
        self.x_trend_index.reset()
        self.cycle_er.reset()
        self._result = None
        self._data_hash = None 