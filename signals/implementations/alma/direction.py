#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.alma import ALMA

class ALMATrendFollowingStrategy(BaseSignal, IDirectionSignal):
    """
    ALMAを使用した方向性シグナル
    - 短期ALMA > 長期ALMA: ロング方向 (1)
    - 短期ALMA < 長期ALMA: ショート方向 (-1)
    """
    
    def __init__(
        self,
        short_period: int = 9,
        long_period: int = 21,
        sigma: float = 6.0,
        offset: float = 0.85
    ):
        """
        コンストラクタ
        
        Args:
            short_period: 短期ALMAの期間
            long_period: 長期ALMAの期間
            sigma: ガウス分布の標準偏差
            offset: 重みの中心位置（0-1）
        """
        params = {
            'short_period': short_period,
            'long_period': long_period,
            'sigma': sigma,
            'offset': offset
        }
        super().__init__(f"ALMADirection({short_period}, {long_period})", params)
        
        # ALMAインジケーターの初期化
        self._short_alma = ALMA(short_period, sigma, offset)
        self._long_alma = ALMA(long_period, sigma, offset)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング方向, -1: ショート方向)
        """
        # ALMAの計算
        short_alma = self._short_alma.calculate(data)
        long_alma = self._long_alma.calculate(data)
        
        # 方向性シグナルの生成
        # 短期ALMAが長期ALMAより上にある場合はロング方向(1)
        # 短期ALMAが長期ALMAより下にある場合はショート方向(-1)
        signals = np.where(short_alma > long_alma, 1, -1)
        
        return signals 
    


class ALMACirculationSignal(BaseSignal, IDirectionSignal):
    """
    ALMAを使用した方向性シグナル（移動平均線大循環）
    
    ステージの定義:
    1: 短期 > 中期 > 長期  （安定上昇相場）
    2: 中期 > 短期 > 長期  （上昇相場の終焉）
    3: 中期 > 長期 > 短期   (下降相場の入口)
    4: 長期 > 中期 > 短期   (安定下降相場)
    5: 長期 > 短期 > 中期   (下降相場の終焉)
    6: 短期 > 長期 > 中期   (上昇相場の入口)
    """
    
    def __init__(self, sigma: float = 6.0, offset: float = 0.85, params: Dict[str, Any] = None):
        """
        コンストラクタ
        
        Args:
            sigma: ガウス分布の標準偏差
            offset: 重みの中心位置（0-1）
            params: パラメータ辞書
                - short_period: 短期ALMAの期間
                - middle_period: 中期ALMAの期間
                - long_period: 長期ALMAの期間
        """
        # デフォルトのパラメータ設定
        default_params = {
            'short_period': 9,
            'middle_period': 21,
            'long_period': 55
        }
        
        # パラメータのマージ
        self._params = params or default_params
        
        super().__init__(
            f"ALMACirculation({self._params['short_period']}, {self._params['middle_period']}, {self._params['long_period']})",
            self._params
        )
        
        # ALMAインジケーターの初期化
        self._short_alma = ALMA(self._params['short_period'], sigma, offset)
        self._middle_alma = ALMA(self._params['middle_period'], sigma, offset)
        self._long_alma = ALMA(self._params['long_period'], sigma, offset)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: エントリー, -1: エグジット, 0: その他)
        """
        # ALMAの計算
        short_alma = self._short_alma.calculate(data)
        middle_alma = self._middle_alma.calculate(data)
        long_alma = self._long_alma.calculate(data)
        
        # シグナルの初期化
        signals = np.zeros(len(data))
        
        # ステージの判定
        # ステージ1: 短期 > 中期 > 長期
        stage1 = (short_alma > middle_alma) & (middle_alma > long_alma)
        
        # ステージ2: 中期 > 短期 > 長期
        stage2 = (middle_alma > short_alma) & (short_alma > long_alma)
        
        # ステージ3: 中期 > 長期 > 短期
        stage3 = (middle_alma > long_alma) & (long_alma > short_alma)
        
        # ステージ4: 長期 > 中期 > 短期
        stage4 = (long_alma > middle_alma) & (middle_alma > short_alma)
        
        # ステージ5: 長期 > 短期 > 中期
        stage5 = (long_alma > short_alma) & (short_alma > middle_alma)
        
        # ステージ6: 短期 > 長期 > 中期
        stage6 = (short_alma > long_alma) & (long_alma > middle_alma)
        
        # エントリーシグナル（ステージ2または3）
        entry_signal = stage2 | stage3
        
        # エグジットシグナル（ステージ5、6、または1）
        exit_signal = stage5 | stage6 | stage1
        
        # シグナルの設定
        signals = np.where(entry_signal, -1, signals)  # 売りシグナル
        signals = np.where(exit_signal, 1, signals)    # 買いシグナル（エグジット）
        
        return signals 
    
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
        # ALMAの計算
        short_alma = self._short_alma.calculate(data)
        middle_alma = self._middle_alma.calculate(data)
        long_alma = self._long_alma.calculate(data)
        
        # シグナルの初期化
        stages = np.zeros(len(data))
        
        # ステージの判定
        # ステージ1: 短期 > 中期 > 長期
        stage1 = (short_alma > middle_alma) & (middle_alma > long_alma)
        
        # ステージ2: 中期 > 短期 > 長期
        stage2 = (middle_alma > short_alma) & (short_alma > long_alma)
        
        # ステージ3: 中期 > 長期 > 短期
        stage3 = (middle_alma > long_alma) & (long_alma > short_alma)
        
        # ステージ4: 長期 > 中期 > 短期
        stage4 = (long_alma > middle_alma) & (middle_alma > short_alma)
        
        # ステージ5: 長期 > 短期 > 中期
        stage5 = (long_alma > short_alma) & (short_alma > middle_alma)
        
        # ステージ6: 短期 > 長期 > 中期
        stage6 = (short_alma > long_alma) & (long_alma > middle_alma)
        
        # ステージの設定
        stages = np.where(stage1, 1, stages)
        stages = np.where(stage2, 2, stages)
        stages = np.where(stage3, 3, stages)
        stages = np.where(stage4, 4, stages)
        stages = np.where(stage5, 5, stages)
        stages = np.where(stage6, 6, stages)
        
        return stages


class ALMADirectionSignal2(BaseSignal, IDirectionSignal):
    """ALMAディレクションシグナル2
    
    n期間ALMAが終値より下にあるときは1（上昇トレンド）、
    上にあるときは-1（下降トレンド）を出力する
    """
    
    def __init__(self, period: int = 200):
        """
        初期化
        
        Args:
            period: ALMA期間（デフォルト: 200）
        """
        super().__init__("ALMADirectionSignal", {"period": period})
        self.alma = ALMA(period=period)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成
        
        Args:
            data: 価格データ
            
        Returns:
            1（上昇トレンド）または-1（下降トレンド）の配列
        """
        df = pd.DataFrame(data)
        close = df['close'].values
        
        # ALMAの計算
        alma_values = self.alma.calculate(data)
        
        # シグナルの生成
        signals = np.where(close > alma_values, 1, -1)
        
        return signals 


class ALMATripleDirectionSignal(BaseSignal, IDirectionSignal):
    """
    3本のALMAを使用したディレクションシグナル
    
    シグナル条件:
    1: 短期 > 中期 > 長期 （完全な上昇配列）
    -1: 短期 < 中期 < 長期 （完全な下降配列）
    0: その他の配列パターン
    """
    
    def __init__(
        self,
        short_period: int = 9,
        middle_period: int = 21,
        long_period: int = 55,
        sigma: float = 6.0,
        offset: float = 0.85
    ):
        """
        コンストラクタ
        
        Args:
            short_period: 短期ALMAの期間
            middle_period: 中期ALMAの期間
            long_period: 長期ALMAの期間
            sigma: ガウス分布の標準偏差
            offset: 重みの中心位置（0-1）
        """
        params = {
            'short_period': short_period,
            'middle_period': middle_period,
            'long_period': long_period,
            'sigma': sigma,
            'offset': offset
        }
        super().__init__(f"ALMATripleDirection({short_period}, {middle_period}, {long_period})", params)
        
        # ALMAインジケーターの初期化
        self._short_alma = ALMA(short_period, sigma, offset)
        self._middle_alma = ALMA(middle_period, sigma, offset)
        self._long_alma = ALMA(long_period, sigma, offset)
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: 上昇配列, -1: 下降配列, 0: その他)
        """
        # ALMAの計算
        short_alma = self._short_alma.calculate(data)
        middle_alma = self._middle_alma.calculate(data)
        long_alma = self._long_alma.calculate(data)
        
        # シグナルの生成
        # 上昇配列: 短期 > 中期 > 長期
        uptrend = (short_alma > middle_alma) & (middle_alma > long_alma)
        
        # 下降配列: 短期 < 中期 < 長期
        downtrend = (short_alma < middle_alma) & (middle_alma < long_alma)
        
        # シグナルの設定
        signals = np.zeros(len(data))
        signals = np.where(uptrend, 1, signals)
        signals = np.where(downtrend, -1, signals)
        
        return signals