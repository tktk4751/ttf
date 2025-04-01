#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.alpha_band import AlphaBand


@njit(fastmath=True, parallel=True)
def calculate_breakout_signals(close: np.ndarray, upper: np.ndarray, lower: np.ndarray, lookback: int) -> np.ndarray:
    """
    ブレイクアウトシグナルを計算する（高速化版）
    
    Args:
        close: 終値の配列
        upper: アッパーバンドの配列
        lower: ロワーバンドの配列
        lookback: 過去のバンドを参照する期間
    
    Returns:
        シグナルの配列
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # ブレイクアウトの判定（並列処理化）
    for i in prange(lookback, length):
        # 終値とバンドの値が有効かチェック
        if np.isnan(close[i]) or np.isnan(upper[i-lookback]) or np.isnan(lower[i-lookback]):
            signals[i] = 0
            continue
            
        # ロングエントリー: 終値がアッパーバンドを上回る
        if close[i] > upper[i-lookback]:
            signals[i] = 1
        # ショートエントリー: 終値がロワーバンドを下回る
        elif close[i] < lower[i-lookback]:
            signals[i] = -1
    
    return signals


class AlphaBandBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    アルファバンドのブレイクアウトによるエントリーシグナル
    
    特徴:
    - サイクル効率比（CER）に基づく動的な適応性
    - 平滑化アルゴリズム（ALMAまたはハイパースムーサー）を使用したAlphaATR
    - ボラティリティに応じた最適なバンド幅
    - トレンドの強さに合わせた自動調整
    
    シグナル条件:
    - 現在の終値が指定期間前のアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値が指定期間前のロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        cycle_detector_type: str = 'hody_dc',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_kama_period: int = 55,
        min_kama_period: int = 8,
        max_atr_period: int = 55,
        min_atr_period: int = 8,
        max_multiplier: float = 3.0,
        min_multiplier: float = 1.5,
        smoother_type: str = 'alma',
        lookback: int = 1
    ):
        """
        コンストラクタ
        
        Args:
            cycle_detector_type: サイクル検出器の種類（デフォルト: 'hody_dc'）
                'dudi_dc' - 二重微分
                'hody_dc' - ホモダイン判別機
                'phac_dc' - 位相累積
                'dudi_dce' - 拡張二重微分
                'hody_dce' - 拡張ホモダイン判別機
                'phac_dce' - 拡張位相累積
            lp_period: ローパスフィルターの期間（デフォルト: 5）
            hp_period: ハイパスフィルターの期間（デフォルト: 144）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_kama_period: AlphaMAのKAMA最大期間（デフォルト: 55）
            min_kama_period: AlphaMAのKAMA最小期間（デフォルト: 8）
            max_atr_period: AlphaATRの最大期間（デフォルト: 55）
            min_atr_period: AlphaATRの最小期間（デフォルト: 8）
            max_multiplier: ATR乗数の最大値（デフォルト: 3.0）
            min_multiplier: ATR乗数の最小値（デフォルト: 1.5）
            smoother_type: 平滑化アルゴリズム（デフォルト: 'alma'）
                'alma' - ALMA（Arnaud Legoux Moving Average）
                'hyper' - ハイパースムーサー（3段階平滑化）
            lookback: 過去のバンドを参照する期間（デフォルト: 1）
        """
        params = {
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_kama_period': max_kama_period,
            'min_kama_period': min_kama_period,
            'max_atr_period': max_atr_period,
            'min_atr_period': min_atr_period,
            'max_multiplier': max_multiplier,
            'min_multiplier': min_multiplier,
            'smoother_type': smoother_type,
            'lookback': lookback
        }
        super().__init__(
            f"AlphaBandBreakout({cycle_detector_type}, {max_multiplier}, {min_multiplier}, {smoother_type}, {lookback})",
            params
        )
        
        # アルファバンドのインスタンス化
        self._alpha_band = AlphaBand(
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_kama_period=max_kama_period,
            min_kama_period=min_kama_period,
            max_atr_period=max_atr_period,
            min_atr_period=min_atr_period,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
            smoother_type=smoother_type
        )
        
        # 結果キャッシュ
        self._signals = None
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
        
        return f"{data_hash}_{hash(frozenset(self._params.items()))}"
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._signals is not None:
                return self._signals
                
            self._data_hash = data_hash
            
            # アルファバンドの計算
            result = self._alpha_band.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if result is None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                return self._signals
            
            # 終値の取得
            close = data['close'].values if isinstance(data, pd.DataFrame) else data[:, 3]
            
            # バンドの取得
            _, upper, lower = self._alpha_band.get_bands()
            
            # ブレイクアウトシグナルの計算（高速化版）
            lookback = self._params['lookback']
            signals = calculate_breakout_signals(
                close,
                upper,
                lower,
                lookback
            )
            
            # 結果をキャッシュ
            self._signals = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"AlphaBandBreakoutEntrySignal計算中にエラー: {str(e)}")
            self._signals = np.zeros(len(data), dtype=np.int8)
            return self._signals
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        アルファバンドのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self._alpha_band.get_bands()
    
    def get_cycle_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        if data is not None:
            self.generate(data)
            
        return self._alpha_band.get_cycle_er()
    
    def get_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的乗数の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        if data is not None:
            self.generate(data)
            
        return self._alpha_band.get_dynamic_multiplier()
    
    def get_alpha_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        AlphaATRの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: AlphaATRの値
        """
        if data is not None:
            self.generate(data)
            
        return self._alpha_band.get_alpha_atr()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self._alpha_band.reset() if hasattr(self._alpha_band, 'reset') else None
        self._signals = None
        self._data_hash = None 