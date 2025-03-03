#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.kama import KaufmanAdaptiveMA

class KAMATrendFollowingStrategy(BaseSignal, IEntrySignal):
    """
    KAMAを使用したトレンドフォロー戦略
    - 短期KAMA > 長期KAMA: ロング方向 (1)
    - 短期KAMA < 長期KAMA: ショート方向 (-1)
    """
    
    def __init__(
        self,
        short_period: int = 9,
        long_period: int = 21
    ):
        """
        コンストラクタ
        
        Args:
            short_period: 短期KAMAの期間
            long_period: 長期KAMAの期間
        """
        params = {
            'short_period': short_period,
            'long_period': long_period
        }
        super().__init__(f"KAMATrendFollowing({short_period}, {long_period})", params)
        
        # KAMAインジケーターの初期化
        self._short_kama = KaufmanAdaptiveMA(short_period)
        self._long_kama = KaufmanAdaptiveMA(long_period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング方向, -1: ショート方向)
        """
        # KAMAの計算
        short_kama = self._short_kama.calculate(data)
        long_kama = self._long_kama.calculate(data)
        
        # シグナルの生成
        # 短期KAMAが長期KAMAより上にある場合はロング方向(1)
        # 短期KAMAが長期KAMAより下にある場合はショート方向(-1)
        signals = np.where(short_kama > long_kama, 1, -1)
        
        return signals

class KAMACirculationSignal(BaseSignal, IEntrySignal):
    """
    KAMAの大循環を使用したエントリーシグナル
    
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
        params: Dict[str, int] = None
    ):
        """
        コンストラクタ
        
        Args:
            params: 各期間のパラメータ
                - short_period: 短期KAMAの期間
                - middle_period: 中期KAMAの期間
                - long_period: 長期KAMAの期間
        """
        if params is None:
            params = {
                'short_period': 21,
                'middle_period': 89,
                'long_period': 233
            }
        
        super().__init__("KAMACirculation", params)
        
        # KAMAインジケーターの初期化
        # KAMAのデフォルトパラメータ: fast_period=2, slow_period=30を使用
        self._short_kama = KaufmanAdaptiveMA(params['short_period'])
        self._middle_kama = KaufmanAdaptiveMA(params['middle_period'])
        self._long_kama = KaufmanAdaptiveMA(params['long_period'])
        
        # 現在のステージを保持
        self._current_stage = 0
    
    def _determine_stage(self, short_kama: float, middle_kama: float, long_kama: float) -> int:
        """
        KAMAの位置関係からステージを判断
        
        Args:
            short_kama: 短期KAMA
            middle_kama: 中期KAMA
            long_kama: 長期KAMA
            
        Returns:
            int: ステージ番号 (1-6)
            1: 短期 > 中期 > 長期  （安定上昇相場）
            2: 中期 > 短期 > 長期  （上昇相場の終焉）
            3: 中期 > 長期 > 短期   (下降相場の入口)
            4: 長期 > 中期 > 短期   (安定下降相場)
            5: 長期 > 短期 > 中期   (下降相場の終焉)
            6: 短期 > 長期 > 中期   (上昇相場の入口)
        """
        if short_kama > middle_kama > long_kama:
            return 1  # 安定上昇相場
        elif middle_kama > short_kama > long_kama:
            return 2  # 上昇相場の終焉
        elif middle_kama > long_kama > short_kama:
            return 3  # 下降相場の入口
        elif long_kama > middle_kama > short_kama:
            return 4  # 安定下降相場
        elif long_kama > short_kama > middle_kama:
            return 5  # 下降相場の終焉
        elif short_kama > long_kama > middle_kama:
            return 6  # 上昇相場の入口
        return 0  # 不明
    
    def get_current_stage(self) -> int:
        """
        現在のステージを取得
        
        Returns:
            int: 現在のステージ
        """
        return self._current_stage
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
            
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        # KAMAの計算
        short_kama = self._short_kama.calculate(data)
        middle_kama = self._middle_kama.calculate(data)
        long_kama = self._long_kama.calculate(data)
        
        # シグナルの初期化
        signals = np.zeros(len(data))
        stages = np.zeros(len(data))
        
        # 各時点でのステージを判断
        for i in range(len(data)):
            stage = self._determine_stage(short_kama[i], middle_kama[i], long_kama[i])
            stages[i] = stage
            
            # ステージ2または3でショートエントリー
            if stage in [2, 3, 4]:
                signals[i] = -1
            # ステージ5、6、または1でロングエントリー
            elif stage in [5, 6, 1]:
                signals[i] = 1
        
        # 最後のステージを保存
        self._current_stage = stages[-1]
        
        return signals 