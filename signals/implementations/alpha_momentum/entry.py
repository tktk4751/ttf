#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any
import numpy as np
import pandas as pd
from numba import jit

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.alpha_momentum import AlphaMomentum


@jit(nopython=True)
def calculate_entry_signals(
    momentum: np.ndarray,
    sqz_on: np.ndarray,
    sqz_off: np.ndarray,
    momentum_threshold: float
) -> np.ndarray:
    """
    エントリーシグナルを計算する（Numba最適化版）
    
    Args:
        momentum: モメンタム値の配列
        sqz_on: スクイーズオン状態の配列（True=スクイーズ中）
        sqz_off: スクイーズオフ状態の配列（True=スクイーズ解放直後）
        momentum_threshold: モメンタムの閾値
        
    Returns:
        np.ndarray: エントリーシグナルの配列（1=ロング、-1=ショート、0=シグナルなし）
    """
    length = len(momentum)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最低でも2つのデータポイントが必要
    if length < 2:
        return signals
    
    # シグナルの生成
    for i in range(1, length):
        # スクイーズ状態確認と、その後のスクイーズ解放でのシグナル生成
        if sqz_on[i-1] and sqz_off[i]:
            # 買いモメンタム（モメンタムが閾値を超えている）
            if momentum[i] > momentum_threshold:
                signals[i] = 1
            # 売りモメンタム（モメンタムが閾値を下回っている）
            elif momentum[i] < -momentum_threshold:
                signals[i] = -1
    
    return signals


class AlphaMomentumEntrySignal(BaseSignal, IEntrySignal):
    """
    アルファモメンタムによるエントリーシグナル
    
    特徴:
    - 効率比（ER）に基づく動的パラメータ最適化
    - アルファボリンジャーバンドとアルファケルトナーチャネルを組み合わせた高度なスクイーズ検出
    - Numba最適化によるシグナル計算の高速化
    
    エントリー条件:
    - ロング: スクイーズ状態から解放され、モメンタムが正（上昇トレンド）
    - ショート: スクイーズ状態から解放され、モメンタムが負（下降トレンド）
    """
    
    def __init__(
        self,
        er_period: int = 21,
        bb_max_period: int = 55,
        bb_min_period: int = 13,
        kc_max_period: int = 55,
        kc_min_period: int = 13,
        bb_max_mult: float = 2.0,
        bb_min_mult: float = 1.0,
        kc_max_mult: float = 3.0,
        kc_min_mult: float = 1.0,
        max_length: int = 34,
        min_length: int = 8,
        alma_offset: float = 0.85,
        alma_sigma: float = 6,
        momentum_threshold: float = 0.0
    ):
        """
        コンストラクタ
        
        Args:
            er_period: 効率比の計算期間（デフォルト: 21）
            bb_max_period: ボリンジャーバンドの標準偏差計算の最大期間（デフォルト: 55）
            bb_min_period: ボリンジャーバンドの標準偏差計算の最小期間（デフォルト: 13）
            kc_max_period: ケルトナーチャネルのAlphaATR最大期間（デフォルト: 55）
            kc_min_period: ケルトナーチャネルのAlphaATR最小期間（デフォルト: 13）
            bb_max_mult: ボリンジャーバンドの標準偏差乗数の最大値（デフォルト: 2.0）
            bb_min_mult: ボリンジャーバンドの標準偏差乗数の最小値（デフォルト: 1.0）
            kc_max_mult: ケルトナーチャネルのATR乗数の最大値（デフォルト: 3.0）
            kc_min_mult: ケルトナーチャネルのATR乗数の最小値（デフォルト: 1.0）
            max_length: モメンタム計算の最大期間（デフォルト: 34）
            min_length: モメンタム計算の最小期間（デフォルト: 8）
            alma_offset: ALMAのオフセット（デフォルト: 0.85）
            alma_sigma: ALMAのシグマ（デフォルト: 6）
            momentum_threshold: モメンタムの閾値（デフォルト: 0.0）
        """
        params = {
            'er_period': er_period,
            'bb_max_period': bb_max_period,
            'bb_min_period': bb_min_period,
            'kc_max_period': kc_max_period,
            'kc_min_period': kc_min_period,
            'bb_max_mult': bb_max_mult,
            'bb_min_mult': bb_min_mult,
            'kc_max_mult': kc_max_mult,
            'kc_min_mult': kc_min_mult,
            'max_length': max_length,
            'min_length': min_length,
            'alma_offset': alma_offset,
            'alma_sigma': alma_sigma,
            'momentum_threshold': momentum_threshold
        }
        
        super().__init__(
            f"AlphaMomentum({er_period},{bb_max_period},{kc_max_period})",
            params
        )
        
        # アルファモメンタムインジケーターの初期化
        self._alpha_momentum = AlphaMomentum(
            er_period=er_period,
            bb_max_period=bb_max_period,
            bb_min_period=bb_min_period,
            kc_max_period=kc_max_period,
            kc_min_period=kc_min_period,
            bb_max_mult=bb_max_mult,
            bb_min_mult=bb_min_mult,
            kc_max_mult=kc_max_mult,
            kc_min_mult=kc_min_mult,
            max_length=max_length,
            min_length=min_length,
            alma_offset=alma_offset,
            alma_sigma=alma_sigma
        )
        
        # モメンタムの閾値
        self._momentum_threshold = momentum_threshold
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、'high', 'low', 'close'カラムが必要
                
        Returns:
            np.ndarray: エントリーシグナルの配列（1=ロング、-1=ショート、0=シグナルなし）
        """
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # アルファモメンタムの計算
                momentum = self._alpha_momentum.calculate(data)
                
                # スクイーズ状態の取得
                sqz_on, sqz_off, _ = self._alpha_momentum.get_squeeze_states()
                
                # シグナルの生成（Numba最適化）
                self._signals = calculate_entry_signals(
                    momentum,
                    sqz_on,
                    sqz_off,
                    self._momentum_threshold
                )
                
                self._data_len = current_len
            
            return self._signals
            
        except Exception as e:
            self.logger.error(f"AlphaMomentumEntrySignal生成中にエラー: {str(e)}")
            # エラー時はゼロ配列を返す
            return np.zeros(len(data), dtype=np.int8)
    
    def get_momentum(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        アルファモメンタム値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合はシグナルを再計算します。
            
        Returns:
            np.ndarray: モメンタム値の配列
        """
        try:
            if data is not None:
                self._alpha_momentum.calculate(data)
                
            return self._alpha_momentum.get_momentum()
        except Exception as e:
            self.logger.error(f"モメンタム値取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_squeeze_states(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        スクイーズ状態を取得する
        
        Args:
            data: オプションの価格データ。指定された場合はシグナルを再計算します。
            
        Returns:
            tuple: (sqz_on, sqz_off, no_sqz)のタプル
        """
        try:
            if data is not None:
                self._alpha_momentum.calculate(data)
                
            return self._alpha_momentum.get_squeeze_states()
        except Exception as e:
            self.logger.error(f"スクイーズ状態取得中にエラー: {str(e)}")
            empty = np.array([], dtype=np.bool_)
            return empty, empty, empty
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        バンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合はシグナルを再計算します。
            
        Returns:
            tuple: (bb_upper, bb_lower, kc_upper, kc_lower)のタプル
        """
        try:
            if data is not None:
                self._alpha_momentum.calculate(data)
                
            return self._alpha_momentum.get_bands()
        except Exception as e:
            self.logger.error(f"バンド値取得中にエラー: {str(e)}")
            empty = np.array([])
            return empty, empty, empty, empty 