#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.alpha_squeeze import AlphaSqueeze


class AlphaSqueezeDirectionSignal(BaseSignal, IDirectionSignal):
    """
    アルファスクイーズによるディレクションシグナル
    
    特徴:
    - アルファボリンジャーバンドとアルファケルトナーチャネルを組み合わせた高度なスクイーズ検出
    - 効率比（ER）に基づくすべてのパラメータの動的最適化
    
    シグナル:
    - スクイーズオン（1）: ボリンジャーバンドがケルトナーチャネルの外側に出た状態
      - 価格のボラティリティが拡大し、トレンドが発生している可能性がある
    - スクイーズオフ（-1）: ボリンジャーバンドがケルトナーチャネルの内側にある状態
      - 価格のボラティリティが低下し、大きな値動きの前触れとなる可能性がある
    """
    
    def __init__(
        self,
        er_period: int = 21,
        max_band_period: int = 55,
        min_band_period: int = 13,
        bb_max_mult: float = 2,
        bb_min_mult: float = 1.0,
        kc_max_mult: float = 3,
        kc_min_mult: float = 1.0,
        alma_offset: float = 0.85,
        alma_sigma: float = 6
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            
            max_band_period: バンド計算の最大期間（デフォルト: 55）
            min_band_period: バンド計算の最小期間（デフォルト: 13）
            
            bb_max_mult: ボリンジャーバンドの標準偏差乗数の最大値（デフォルト: 2）
            bb_min_mult: ボリンジャーバンドの標準偏差乗数の最小値（デフォルト: 1.0）
            
            kc_max_mult: ケルトナーチャネルのATR乗数の最大値（デフォルト: 3）
            kc_min_mult: ケルトナーチャネルのATR乗数の最小値（デフォルト: 1.0）
            
            alma_offset: ALMAのオフセット（デフォルト: 0.85）
            alma_sigma: ALMAのシグマ（デフォルト: 6）
        """
        params = {
            'er_period': er_period,
            'max_band_period': max_band_period,
            'min_band_period': min_band_period,
            'bb_max_mult': bb_max_mult,
            'bb_min_mult': bb_min_mult,
            'kc_max_mult': kc_max_mult,
            'kc_min_mult': kc_min_mult,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma
        }
        super().__init__(f"AlphaSqueezeDirection({er_period})", params)
        
        # アルファスクイーズインジケーターの初期化
        self._alpha_squeeze = AlphaSqueeze(
            er_period=er_period,
            max_band_period=max_band_period,
            min_band_period=min_band_period,
            bb_max_mult=bb_max_mult,
            bb_min_mult=bb_min_mult,
            kc_max_mult=kc_max_mult,
            kc_min_mult=kc_min_mult,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma
        )
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
                DataFrameの場合、'high', 'low', 'close'カラムが必要
                NumPy配列の場合、2次元で[時間, [open, high, low, close, ...]]の形式
        
        Returns:
            シグナルの配列 (1: スクイーズオン、-1: スクイーズオフ、0: 非スクイーズ)
        """
        try:
            # アルファスクイーズの計算
            self._alpha_squeeze.calculate(data)
            
            # スクイーズシグナルの取得
            squeeze_signal = self._alpha_squeeze.get_squeeze_signal()
            
            # 最初の期間はシグナルなし（ウォームアップ期間）
            warmup_period = self._params['max_band_period']
            if len(squeeze_signal) > warmup_period:
                squeeze_signal[:warmup_period] = 0
            
            return squeeze_signal
            
        except Exception as e:
            import logging
            logging.getLogger(self.__class__.__name__).error(f"シグナル生成中にエラー: {str(e)}")
            
            # エラー時は空のシグナルを返す
            if isinstance(data, pd.DataFrame):
                return np.zeros(len(data))
            else:
                return np.zeros(len(data) if data.ndim == 2 else 0) 