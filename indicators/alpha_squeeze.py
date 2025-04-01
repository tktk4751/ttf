#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .alpha_bollinger_bands import AlphaBollingerBands
from .alpha_keltner_channel import AlphaKeltnerChannel


@dataclass
class AlphaSqueezeResult:
    """AlphaSqueezeの計算結果"""
    squeeze_signal: np.ndarray  # スクイーズシグナル（1=スクイーズオン、-1=スクイーズオフ、0=非スクイーズ）
    sqz_on: np.ndarray          # スクイーズ状態（1=オン）
    sqz_off: np.ndarray         # スクイーズ解放状態（1=オフ）
    no_sqz: np.ndarray          # 非スクイーズ状態（1=非スクイーズ）
    bb_upper: np.ndarray        # アルファボリンジャー上限
    bb_lower: np.ndarray        # アルファボリンジャー下限
    kc_upper: np.ndarray        # アルファケルトナー上限
    kc_lower: np.ndarray        # アルファケルトナー下限


@jit(nopython=True)
def calculate_squeeze_states(
    bb_lower: np.ndarray,
    bb_upper: np.ndarray,
    kc_lower: np.ndarray,
    kc_upper: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    スクイーズ状態を計算（高速化版）
    
    Args:
        bb_lower: BBの下限
        bb_upper: BBの上限
        kc_lower: KCの下限
        kc_upper: KCの上限
        
    Returns:
        (sqz_on, sqz_off, no_sqz) のタプル
    """
    length = len(bb_lower)
    # 整数型を使用（Numba対応）
    sqz_on = np.zeros(length, dtype=np.int32)
    sqz_off = np.zeros(length, dtype=np.int32)
    no_sqz = np.zeros(length, dtype=np.int32)
    
    for i in range(length):
        # NaNをスキップ
        if (np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(kc_upper[i])):
            continue
            
        # スクイーズ状態の判定
        if bb_lower[i] < kc_lower[i] and bb_upper[i] > kc_upper[i]:
            sqz_on[i] = 1
        elif bb_lower[i] > kc_lower[i] and bb_upper[i] < kc_upper[i]:
            sqz_off[i] = 1
        else:
            no_sqz[i] = 1
    
    return sqz_on, sqz_off, no_sqz


@jit(nopython=True)
def calculate_squeeze_signal(
    sqz_on: np.ndarray,
    sqz_off: np.ndarray
) -> np.ndarray:
    """
    スクイーズシグナルを計算（高速化版）
    
    Args:
        sqz_on: スクイーズオン状態
        sqz_off: スクイーズオフ状態
        
    Returns:
        スクイーズシグナル（1=スクイーズオン、-1=スクイーズオフ、0=非スクイーズ）
    """
    length = len(sqz_on)
    signal = np.zeros(length, dtype=np.int32)
    
    for i in range(length):
        if sqz_on[i] == 1:
            signal[i] = 1
        elif sqz_off[i] == 1:
            signal[i] = -1
    
    return signal


class AlphaSqueeze(Indicator):
    """
    アルファスクイーズ（Alpha Squeeze）インジケーター
    
    特徴:
    - アルファボリンジャーバンドとアルファケルトナーチャネルを組み合わせた高度なスクイーズ検出
    - 効率比（ER）に基づくすべてのパラメータの動的最適化
    
    市場状態に応じた最適な挙動:
    - トレンド強い（ER高い）:
      - より狭いバンド幅でトレンドをより早く検出
    - トレンド弱い（ER低い）:
      - より広いバンド幅でノイズによる誤シグナルを減少
    
    スクイーズ状態:
    - スクイーズオン（1）: ボリンジャーバンドがケルトナーチャネルの外側に出た状態
      - 価格のボラティリティが拡大し、トレンドが発生している可能性がある
    - スクイーズオフ（-1）: ボリンジャーバンドがケルトナーチャネルの内側にある状態
      - 価格のボラティリティが低下し、大きな値動きの前触れとなる可能性がある
    - 非スクイーズ（0）: その他の状態
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
        super().__init__(f"AlphaSqueeze({er_period})")
        
        self.er_period = er_period
        
        # アルファボリンジャーバンドの初期化
        self.alpha_bb = AlphaBollingerBands(
            er_period=er_period,
            max_kama_period=max_band_period,
            min_kama_period=min_band_period,
            std_dev_period=max_band_period,  # 標準偏差期間は最大期間を使用
            max_multiplier=bb_max_mult,
            min_multiplier=bb_min_mult
        )
        
        # アルファケルトナーチャネルの初期化
        self.alpha_kc = AlphaKeltnerChannel(
            er_period=er_period,
            max_kama_period=max_band_period,
            min_kama_period=min_band_period,
            max_atr_period=max_band_period,
            min_atr_period=min_band_period,
            max_multiplier=kc_max_mult,
            min_multiplier=kc_min_mult,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma
        )
        
        self._result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        アルファスクイーズを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high', 'low', 'close'カラムが必要
        
        Returns:
            スクイーズシグナルの配列（1=スクイーズオン、-1=スクイーズオフ、0=非スクイーズ）
        """
        try:
            # データの検証と変換
            if isinstance(data, pd.DataFrame):
                if not all(col in data.columns for col in ['high', 'low', 'close']):
                    raise ValueError("DataFrameには'high', 'low', 'close'カラムが必要です")
                high = data['high'].values
                low = data['low'].values
                close = data['close'].values
            else:
                if data.ndim == 2 and data.shape[1] >= 4:
                    high = data[:, 1]  # high
                    low = data[:, 2]   # low
                    close = data[:, 3] # close
                else:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # データ長の検証
            data_length = len(high)
            self._validate_period(self.er_period, data_length)
            
            # アルファボリンジャーバンドの計算
            bb_middle = self.alpha_bb.calculate(data)
            bb_middle, bb_upper, bb_lower = self.alpha_bb.get_bands()
            
            # アルファケルトナーチャネルの計算
            kc_middle = self.alpha_kc.calculate(data)
            kc_middle, kc_upper, kc_lower = self.alpha_kc.get_bands()
            
            # スクイーズ状態の判定
            sqz_on, sqz_off, no_sqz = calculate_squeeze_states(
                bb_lower, bb_upper,
                kc_lower, kc_upper
            )
            
            # スクイーズシグナルの計算
            squeeze_signal = calculate_squeeze_signal(sqz_on, sqz_off)
            
            # 結果の保存
            self._result = AlphaSqueezeResult(
                squeeze_signal=squeeze_signal,
                sqz_on=sqz_on,
                sqz_off=sqz_off,
                no_sqz=no_sqz,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                kc_upper=kc_upper,
                kc_lower=kc_lower
            )
            
            # 基底クラスの要件を満たすため
            self._values = squeeze_signal
            return squeeze_signal
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"AlphaSqueeze計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時でも最低限の結果を初期化
            if 'close' in locals():
                empty_array = np.zeros_like(close)
                self._values = empty_array
                return empty_array
            return np.array([])
    
    def get_squeeze_states(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        スクイーズ状態を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (sqz_on, sqz_off, no_sqz)のタプル
                - sqz_on: スクイーズオン状態（1=スクイーズ中）
                - sqz_off: スクイーズオフ状態（1=スクイーズ解放直後）
                - no_sqz: 非スクイーズ状態（1=通常状態）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.sqz_on, self._result.sqz_off, self._result.no_sqz
    
    def get_squeeze_signal(self) -> np.ndarray:
        """
        スクイーズシグナルを取得する
        
        Returns:
            np.ndarray: スクイーズシグナル（1=スクイーズオン、-1=スクイーズオフ、0=非スクイーズ）
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return self._result.squeeze_signal
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        バンド値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                (bb_upper, bb_lower, kc_upper, kc_lower)のタプル
        """
        if self._result is None:
            raise RuntimeError("calculate()を先に呼び出してください")
        return (
            self._result.bb_upper,
            self._result.bb_lower,
            self._result.kc_upper,
            self._result.kc_lower
        )
    
    def get_bollinger(self) -> AlphaBollingerBands:
        """
        アルファボリンジャーバンドのインスタンスを取得する
        
        Returns:
            AlphaBollingerBands: アルファボリンジャーバンドのインスタンス
        """
        return self.alpha_bb
    
    def get_keltner(self) -> AlphaKeltnerChannel:
        """
        アルファケルトナーチャネルのインスタンスを取得する
        
        Returns:
            AlphaKeltnerChannel: アルファケルトナーチャネルのインスタンス
        """
        return self.alpha_kc 