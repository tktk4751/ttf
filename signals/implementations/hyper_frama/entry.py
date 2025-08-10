#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.hyper_frama import HyperFRAMA


@njit(fastmath=True, parallel=True)
def calculate_position_signals(
    frama_values: np.ndarray, 
    adjusted_frama_values: np.ndarray
) -> np.ndarray:
    """
    FRAMAとAdjusted FRAMAの位置関係シグナルを計算する（高速化版）
    
    Args:
        frama_values: 通常FRAMA値の配列
        adjusted_frama_values: アルファ調整FRAMA値の配列
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(frama_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 位置関係の判定（並列処理化）
    for i in prange(length):
        # FRAMA値とAdjusted FRAMA値が有効かチェック
        if np.isnan(frama_values[i]) or np.isnan(adjusted_frama_values[i]):
            signals[i] = 0
            continue
            
        # FRAMA > Adjusted FRAMA: ロングシグナル
        if frama_values[i] > adjusted_frama_values[i]:
            signals[i] = 1
        # FRAMA < Adjusted FRAMA: ショートシグナル
        elif frama_values[i] < adjusted_frama_values[i]:
            signals[i] = -1
    
    return signals


@njit(fastmath=True, parallel=False)
def calculate_crossover_signals(
    frama_values: np.ndarray, 
    adjusted_frama_values: np.ndarray
) -> np.ndarray:
    """
    FRAMAとAdjusted FRAMAのクロスオーバーシグナルを計算する（改良版）
    
    クロス検出アルゴリズム:
    1. 短期MA（FRAMA）と長期MA（Adjusted FRAMA）の位置関係を示す配列を作成
    2. 前日のPositionと比較するために、Positionを1つずらした配列を作成
    3. Positionが-1から1に変わった点をゴールデンクロス（ロングシグナル）とする
    4. Positionが1から-1に変わった点をデッドクロス（ショートシグナル）とする
    
    Args:
        frama_values: 通常FRAMA値の配列（短期MA）
        adjusted_frama_values: アルファ調整FRAMA値の配列（長期MA）
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(frama_values)
    signals = np.zeros(length, dtype=np.int8)
    
    if length < 2:
        return signals
    
    # 1. 短期MAと長期MAの位置関係を示す配列を作成（1: 短期 > 長期, -1: 短期 < 長期, 0: 無効）
    position = np.zeros(length, dtype=np.int8)
    for i in range(length):
        if np.isnan(frama_values[i]) or np.isnan(adjusted_frama_values[i]):
            position[i] = 0  # 無効データ
        elif frama_values[i] > adjusted_frama_values[i]:
            position[i] = 1   # 短期 > 長期
        else:
            position[i] = -1  # 短期 <= 長期
    
    # 2. 前日のPositionと比較してクロスを検出
    for i in range(1, length):
        # 現在と前回のPositionが両方とも有効な場合のみクロス判定
        if position[i] != 0 and position[i-1] != 0:
            # 3. Positionが-1から1に変わった点をゴールデンクロス（ロングシグナル）
            if position[i-1] == -1 and position[i] == 1:
                signals[i] = 1
            # 4. Positionが1から-1に変わった点をデッドクロス（ショートシグナル）
            elif position[i-1] == 1 and position[i] == -1:
                signals[i] = -1
    
    return signals


class HyperFRAMAPositionEntrySignal(BaseSignal, IEntrySignal):
    """
    ハイパーFRAMA位置関係によるエントリーシグナル
    
    特徴:
    - 通常のFRAMAとアルファ調整FRAMAの位置関係でシグナル生成
    - フラクタル次元に基づく適応型移動平均線
    - アルファ調整係数により柔軟な設定が可能
    
    シグナル条件:
    - FRAMA > Adjusted FRAMA: ロングシグナル (1)
    - FRAMA < Adjusted FRAMA: ショートシグナル (-1)
    - FRAMA = Adjusted FRAMA: シグナルなし (0)
    """
    
    def __init__(
        self,
        # HyperFRAMAパラメータ
        period: int = 16,
        src_type: str = 'hl2',
        fc: int = 1,
        sc: int = 198,
        alpha_multiplier: float = 0.5,
        # 動的期間パラメータ
        period_mode: str = 'fixed',
        cycle_detector_type: str = 'hody_e',
        lp_period: int = 13,
        hp_period: int = 124,
        cycle_part: float = 0.5,
        max_cycle: int = 89,
        min_cycle: int = 8,
        max_output: int = 124,
        min_output: int = 8
    ):
        """
        初期化
        
        Args:
            period: 期間（偶数である必要がある、デフォルト: 16）
            src_type: ソースタイプ（デフォルト: 'hl2'）
            fc: Fast Constant（デフォルト: 1）
            sc: Slow Constant（デフォルト: 198）
            alpha_multiplier: アルファ調整係数（デフォルト: 0.5）
            period_mode: 期間モード ('fixed' または 'dynamic')
            cycle_detector_type: サイクル検出器タイプ
            ... その他の動的期間パラメータ
        """
        super().__init__(
            f"HyperFRAMAPositionEntrySignal(period={period}, alpha_mult={alpha_multiplier}, {src_type})"
        )
        
        # パラメータの保存
        self._params = {
            'period': period,
            'src_type': src_type,
            'fc': fc,
            'sc': sc,
            'alpha_multiplier': alpha_multiplier,
            'period_mode': period_mode,
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output
        }
        
        # HyperFRAMAインジケーターの初期化
        self.hyper_frama = HyperFRAMA(
            period=period,
            src_type=src_type,
            fc=fc,
            sc=sc,
            alpha_multiplier=alpha_multiplier,
            period_mode=period_mode,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output
        )
        
        # キャッシュの初期化
        self._signals_cache = {}
        
    def _get_data_hash(self, ohlcv_data):
        """
        データハッシュを取得する
        
        Args:
            ohlcv_data: OHLCVデータ
            
        Returns:
            データのハッシュ値
        """
        # DataFrameの場合はNumpy配列に変換
        if isinstance(ohlcv_data, pd.DataFrame):
            # 必要なカラムがあれば抽出、なければそのまま変換
            if all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                ohlcv_array = ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values
            else:
                ohlcv_array = ohlcv_data.values
        else:
            ohlcv_array = ohlcv_data
            
        # Numpy配列でない場合はエラー
        if not isinstance(ohlcv_array, np.ndarray):
            raise TypeError("ohlcv_data must be a numpy array or pandas DataFrame")
        
        # 配列のハッシュと設定パラメータのハッシュを組み合わせる
        return hash((ohlcv_array.tobytes(), *sorted(self._params.items())))
    
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
            if data_hash in self._signals_cache:
                return self._signals_cache[data_hash]
                
            # HyperFRAMAの計算
            hyper_frama_result = self.hyper_frama.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if hyper_frama_result is None or len(hyper_frama_result.frama_values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # FRAMA値とAdjusted FRAMA値の取得
            frama_values = hyper_frama_result.frama_values
            adjusted_frama_values = hyper_frama_result.half_frama_values
            
            # 位置関係シグナルの計算（高速化版）
            signals = calculate_position_signals(
                frama_values,
                adjusted_frama_values
            )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"HyperFRAMAPositionEntrySignal計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        FRAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: FRAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_frama.get_frama_values()
    
    def get_adjusted_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Adjusted FRAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Adjusted FRAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_frama.get_half_frama_values()
    
    def get_fractal_dimension(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        フラクタル次元を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フラクタル次元
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_frama.get_fractal_dimension()
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Alpha値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Alpha値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_frama.get_alpha()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.hyper_frama.reset() if hasattr(self.hyper_frama, 'reset') else None
        self._signals_cache = {}


class HyperFRAMACrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    ハイパーFRAMAクロスオーバーによるエントリーシグナル
    
    特徴:
    - 通常のFRAMAとアルファ調整FRAMAのクロスオーバーでシグナル生成
    - フラクタル次元に基づく適応型移動平均線
    - アルファ調整係数により柔軟な設定が可能
    
    シグナル条件:
    - ロング: 前回 FRAMA <= Adjusted FRAMA かつ 現在 FRAMA > Adjusted FRAMA (1)
    - ショート: 前回 FRAMA >= Adjusted FRAMA かつ 現在 FRAMA < Adjusted FRAMA (-1)
    - その他: シグナルなし (0)
    """
    
    def __init__(
        self,
        # HyperFRAMAパラメータ
        period: int = 16,
        src_type: str = 'hl2',
        fc: int = 1,
        sc: int = 198,
        alpha_multiplier: float = 0.5,
        # 動的期間パラメータ
        period_mode: str = 'fixed',
        cycle_detector_type: str = 'hody_e',
        lp_period: int = 13,
        hp_period: int = 124,
        cycle_part: float = 0.5,
        max_cycle: int = 89,
        min_cycle: int = 8,
        max_output: int = 124,
        min_output: int = 8
    ):
        """
        初期化
        
        Args:
            period: 期間（偶数である必要がある、デフォルト: 16）
            src_type: ソースタイプ（デフォルト: 'hl2'）
            fc: Fast Constant（デフォルト: 1）
            sc: Slow Constant（デフォルト: 198）
            alpha_multiplier: アルファ調整係数（デフォルト: 0.5）
            period_mode: 期間モード ('fixed' または 'dynamic')
            cycle_detector_type: サイクル検出器タイプ
            ... その他の動的期間パラメータ
        """
        super().__init__(
            f"HyperFRAMACrossoverEntrySignal(period={period}, alpha_mult={alpha_multiplier}, {src_type})"
        )
        
        # パラメータの保存
        self._params = {
            'period': period,
            'src_type': src_type,
            'fc': fc,
            'sc': sc,
            'alpha_multiplier': alpha_multiplier,
            'period_mode': period_mode,
            'cycle_detector_type': cycle_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_cycle': max_cycle,
            'min_cycle': min_cycle,
            'max_output': max_output,
            'min_output': min_output
        }
        
        # HyperFRAMAインジケーターの初期化
        self.hyper_frama = HyperFRAMA(
            period=period,
            src_type=src_type,
            fc=fc,
            sc=sc,
            alpha_multiplier=alpha_multiplier,
            period_mode=period_mode,
            cycle_detector_type=cycle_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output
        )
        
        # キャッシュの初期化
        self._signals_cache = {}
        
    def _get_data_hash(self, ohlcv_data):
        """
        データハッシュを取得する
        
        Args:
            ohlcv_data: OHLCVデータ
            
        Returns:
            データのハッシュ値
        """
        # DataFrameの場合はNumpy配列に変換
        if isinstance(ohlcv_data, pd.DataFrame):
            # 必要なカラムがあれば抽出、なければそのまま変換
            if all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                ohlcv_array = ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values
            else:
                ohlcv_array = ohlcv_data.values
        else:
            ohlcv_array = ohlcv_data
            
        # Numpy配列でない場合はエラー
        if not isinstance(ohlcv_array, np.ndarray):
            raise TypeError("ohlcv_data must be a numpy array or pandas DataFrame")
        
        # 配列のハッシュと設定パラメータのハッシュを組み合わせる
        return hash((ohlcv_array.tobytes(), *sorted(self._params.items())))
    
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
            if data_hash in self._signals_cache:
                return self._signals_cache[data_hash]
                
            # HyperFRAMAの計算
            hyper_frama_result = self.hyper_frama.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if hyper_frama_result is None or len(hyper_frama_result.frama_values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # FRAMA値とAdjusted FRAMA値の取得
            frama_values = hyper_frama_result.frama_values
            adjusted_frama_values = hyper_frama_result.half_frama_values
            
            # クロスオーバーシグナルの計算（高速化版）
            signals = calculate_crossover_signals(
                frama_values,
                adjusted_frama_values
            )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"HyperFRAMACrossoverEntrySignal計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        FRAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: FRAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_frama.get_frama_values()
    
    def get_adjusted_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Adjusted FRAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Adjusted FRAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_frama.get_half_frama_values()
    
    def get_fractal_dimension(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        フラクタル次元を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フラクタル次元
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_frama.get_fractal_dimension()
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Alpha値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Alpha値
        """
        if data is not None:
            self.generate(data)
            
        return self.hyper_frama.get_alpha()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.hyper_frama.reset() if hasattr(self.hyper_frama, 'reset') else None
        self._signals_cache = {}