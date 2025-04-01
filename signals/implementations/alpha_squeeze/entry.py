#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.alpha_squeeze import AlphaSqueeze


class AlphaSqueezeEntrySignal(BaseSignal, IEntrySignal):
    """
    アルファスクイーズによるエントリーシグナル
    
    特徴:
    - アルファボリンジャーバンドとアルファケルトナーチャネルを組み合わせた高度なスクイーズ検出
    - 効率比（ER）に基づくすべてのパラメータの動的最適化
    
    シグナル:
    - スクイーズオン（1）: ボリンジャーバンドがケルトナーチャネルの外側に出た状態
      - 価格のボラティリティが拡大し、トレンドが発生している可能性がある
      - ロング/ショートエントリーの可能性（方向性は別途判断）
    - スクイーズオフ（-1）: ボリンジャーバンドがケルトナーチャネルの内側にある状態
      - 価格のボラティリティが低下し、大きな値動きの前触れとなる可能性がある
      - エントリーを控える、または利益確定の可能性
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
        alma_sigma: float = 6,
        lookback: int = 1
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
            
            lookback: 状態変化を検出するための遡り期間（デフォルト: 1）
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
            'alma_sigma': alma_sigma,
            'lookback': lookback
        }
        super().__init__(f"AlphaSqueezeEntry({er_period},{lookback})", params)
        
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
        
        self._lookback = lookback
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
                DataFrameの場合、'high', 'low', 'close'カラムが必要
                NumPy配列の場合、2次元で[時間, [open, high, low, close, ...]]の形式
        
        Returns:
            シグナルの配列 (1: スクイーズオン開始、-1: スクイーズオフ開始、0: 変化なし)
        """
        try:
            # アルファスクイーズの計算
            self._alpha_squeeze.calculate(data)
            
            # スクイーズシグナルの取得
            squeeze_signal = self._alpha_squeeze.get_squeeze_signal()
            
            # 状態変化の検出
            data_length = len(squeeze_signal)
            entry_signals = np.zeros(data_length)
            
            # ウォームアップ期間
            warmup_period = self._params['max_band_period']
            
            # 状態変化の検出（lookback期間前と比較）
            for i in range(warmup_period + self._lookback, data_length):
                # 現在のスクイーズ状態
                current_state = squeeze_signal[i]
                
                # lookback期間前のスクイーズ状態
                previous_state = squeeze_signal[i - self._lookback]
                
                # 状態変化の検出
                if current_state != previous_state:
                    entry_signals[i] = current_state
            
            return entry_signals
            
        except Exception as e:
            import logging
            logging.getLogger(self.__class__.__name__).error(f"シグナル生成中にエラー: {str(e)}")
            
            # エラー時は空のシグナルを返す
            if isinstance(data, pd.DataFrame):
                return np.zeros(len(data))
            else:
                return np.zeros(len(data) if data.ndim == 2 else 0) 