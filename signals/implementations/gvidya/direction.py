#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.gvidya import GVIDYA


@jit(nopython=True)
def calculate_stages(short_gvidya: np.ndarray, middle_gvidya: np.ndarray, long_gvidya: np.ndarray) -> np.ndarray:
    """ステージを計算する（高速化版）"""
    length = len(short_gvidya)
    stages = np.zeros(length, dtype=np.int8)
    
    for i in range(length):
        # ステージ1: 短期 > 中期 > 長期
        if short_gvidya[i] > middle_gvidya[i] and middle_gvidya[i] > long_gvidya[i]:
            stages[i] = 1
        # ステージ2: 中期 > 短期 > 長期
        elif middle_gvidya[i] > short_gvidya[i] and short_gvidya[i] > long_gvidya[i]:
            stages[i] = 2
        # ステージ3: 中期 > 長期 > 短期
        elif middle_gvidya[i] > long_gvidya[i] and long_gvidya[i] > short_gvidya[i]:
            stages[i] = 3
        # ステージ4: 長期 > 中期 > 短期
        elif long_gvidya[i] > middle_gvidya[i] and middle_gvidya[i] > short_gvidya[i]:
            stages[i] = 4
        # ステージ5: 長期 > 短期 > 中期
        elif long_gvidya[i] > short_gvidya[i] and short_gvidya[i] > middle_gvidya[i]:
            stages[i] = 5
        # ステージ6: 短期 > 長期 > 中期
        elif short_gvidya[i] > long_gvidya[i] and long_gvidya[i] > middle_gvidya[i]:
            stages[i] = 6
    
    return stages


@jit(nopython=True)
def calculate_direction_signals(stages: np.ndarray) -> np.ndarray:
    """方向性シグナルを計算する（高速化版）"""
    signals = np.zeros_like(stages, dtype=np.int8)
    
    # ステージ5、6、1でロング、ステージ2、3、4でショート
    for i in range(len(stages)):
        if stages[i] in [5, 6, 1]:
            signals[i] = 1
        elif stages[i] in [2, 3, 4]:
            signals[i] = -1
    
    return signals


class GVIDYACirculationSignal(BaseSignal, IDirectionSignal):
    """
    G-VIDYAを使用した方向性シグナル（移動平均線大循環）
    
    ステージの定義:
    1: 短期 > 中期 > 長期  （安定上昇相場）
    2: 中期 > 短期 > 長期  （上昇相場の終焉）
    3: 中期 > 長期 > 短期   (下降相場の入口)
    4: 長期 > 中期 > 短期   (安定下降相場)
    5: 長期 > 短期 > 中期   (下降相場の終焉)
    6: 短期 > 長期 > 中期   (上昇相場の入口)
    """
    
    def __init__(
        self,
        params: Dict[str, Any] = None,
        sd_period: int = 28,
        gaussian_length: int = 4,
        gaussian_sigma: float = 2.0
    ):
        """
        コンストラクタ
        
        Args:
            params: パラメータ辞書
                - short_period: 短期G-VIDYAの期間
                - middle_period: 中期G-VIDYAの期間
                - long_period: 長期G-VIDYAの期間
            sd_period: 標準偏差の計算期間
            gaussian_length: ガウシアンフィルターの長さ
            gaussian_sigma: ガウシアンフィルターのシグマ
        """
        # デフォルトのパラメータ設定
        default_params = {
            'short_period': 9,
            'middle_period': 21,
            'long_period': 55
        }
        
        # パラメータの設定
        if params is not None:
            default_params.update(params)
        
        super().__init__(
            f"GVIDYACirculation({default_params['short_period']}, {default_params['middle_period']}, {default_params['long_period']})",
            default_params
        )
        
        # G-VIDYAインジケーターの初期化
        self._short_gvidya = GVIDYA(
            vidya_period=default_params['short_period'],
            sd_period=sd_period,
            gaussian_length=gaussian_length,
            gaussian_sigma=gaussian_sigma
        )
        self._middle_gvidya = GVIDYA(
            vidya_period=default_params['middle_period'],
            sd_period=sd_period,
            gaussian_length=gaussian_length,
            gaussian_sigma=gaussian_sigma
        )
        self._long_gvidya = GVIDYA(
            vidya_period=default_params['long_period'],
            sd_period=sd_period,
            gaussian_length=gaussian_length,
            gaussian_sigma=gaussian_sigma
        )
        
        # 現在のステージを保持
        self._current_stage = 0
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # G-VIDYAの計算
        short_gvidya = self._short_gvidya.calculate(data)
        middle_gvidya = self._middle_gvidya.calculate(data)
        long_gvidya = self._long_gvidya.calculate(data)
        
        if short_gvidya is None or middle_gvidya is None or long_gvidya is None:
            return np.zeros(len(data))
        
        # ステージの計算（高速化版）
        stages = calculate_stages(short_gvidya, middle_gvidya, long_gvidya)
        
        # 最後のステージを保存
        self._current_stage = stages[-1]
        
        # 方向性シグナルの計算（高速化版）
        return calculate_direction_signals(stages)
    
    def get_stage(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        現在のステージを返す
        
        Args:
            data: 価格データ
        
        Returns:
            ステージ番号の配列 (1-6)
            1: 短期 > 中期 > 長期  （安定上昇相場）
            2: 中期 > 短期 > 長期  （上昇相場の終焉）
            3: 中期 > 長期 > 短期   (下降相場の入口)
            4: 長期 > 中期 > 短期   (安定下降相場)
            5: 長期 > 短期 > 中期   (下降相場の終焉)
            6: 短期 > 長期 > 中期   (上昇相場の入口)
        """
        # G-VIDYAの計算
        short_gvidya = self._short_gvidya.calculate(data)
        middle_gvidya = self._middle_gvidya.calculate(data)
        long_gvidya = self._long_gvidya.calculate(data)
        
        if short_gvidya is None or middle_gvidya is None or long_gvidya is None:
            return np.zeros(len(data))
        
        # ステージの計算（高速化版）
        return calculate_stages(short_gvidya, middle_gvidya, long_gvidya)
    
    def get_current_stage(self) -> int:
        """
        現在のステージを取得
        
        Returns:
            int: 現在のステージ
        """
        return self._current_stage 