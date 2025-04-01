#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional, List
import numpy as np
import pandas as pd
from numba import njit, prange, vectorize
import hashlib
from dataclasses import dataclass

from ...base_signal import BaseSignal
from ...interfaces.entry import  IEntrySignal
from ...interfaces.exit import IExitSignal
from indicators.c_channel import CChannel, CChannelResult


@dataclass
class SignalCache:
    """シグナル計算結果のキャッシュ"""
    entry_signals: np.ndarray = None
    exit_signals: np.ndarray = None
    entry_bands: Tuple[np.ndarray, np.ndarray, np.ndarray] = None
    exit_bands: Tuple[np.ndarray, np.ndarray, np.ndarray] = None
    cer: np.ndarray = None
    entry_multiplier: np.ndarray = None
    exit_multiplier: np.ndarray = None
    c_atr: np.ndarray = None
    data_hash: str = None
    data_len: int = 0


@vectorize(['boolean(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def is_above_upper_band(close: float, upper: float, epsilon: float=1e-10) -> bool:
    """価格が上限バンドを上回っているかを判定（ベクトル化版）"""
    return close > upper + epsilon


@vectorize(['boolean(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def is_below_lower_band(close: float, lower: float, epsilon: float=1e-10) -> bool:
    """価格が下限バンドを下回っているかを判定（ベクトル化版）"""
    return close < lower - epsilon


@njit(fastmath=True, cache=True)
def calculate_breakout_signals(close: np.ndarray, upper_band: np.ndarray, lower_band: np.ndarray, lookback: int = 1) -> np.ndarray:
    """
    ブレイクアウトシグナルを計算する高速実装
    
    Args:
        close: 終値の配列
        upper_band: 上限バンドの配列
        lower_band: 下限バンドの配列
        lookback: ルックバック期間（デフォルト1）
        
    Returns:
        np.ndarray: 1（ロングエントリー）、-1（ショートエントリー）、0（シグナルなし）の配列
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    epsilon = 1e-12  # より小さな誤差許容値で感度を向上
    
    # 最初のルックバック期間はシグナルなし
    for i in range(lookback, n):
        # 価格が上限バンドを上回った場合（ロングエントリー）
        if close[i] > upper_band[i-lookback] + epsilon:
            signals[i] = 1
        # 価格が下限バンドを下回った場合（ショートエントリー）
        elif close[i] < lower_band[i-lookback] - epsilon:
            signals[i] = -1
    
    return signals


@njit('int8[:](float64[:], float64[:], float64[:], int8[:], int32)', fastmath=True, cache=True)
def calculate_exit_signals(close: np.ndarray, upper_band: np.ndarray, lower_band: np.ndarray, 
                          position: np.ndarray, lookback: int = 1) -> np.ndarray:
    """
    エグジットシグナルを計算する関数
    
    Parameters:
    -----------
    close : np.ndarray
        終値の配列
    upper_band : np.ndarray
        上限バンドの配列
    lower_band : np.ndarray
        下限バンドの配列
    position : np.ndarray
        現在のポジション (1: ロング, -1: ショート, 0: ノーポジション)
    lookback : int
        ルックバック期間
        
    Returns:
    --------
    np.ndarray
        エグジットシグナルの配列 (1: ロングエグジット, -1: ショートエグジット, 0: ノーシグナル)
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    epsilon = 1e-6  # より大きなイプシロン値で感度を高める
    
    # 最初のルックバック期間はシグナルなし
    for i in range(lookback, n):
        # ロングポジションの場合 - 価格が下限バンドを下回るか近づいたらエグジット
        if position[i] == 1:
            if close[i] <= lower_band[i-lookback] + epsilon:
                signals[i] = 1  # ロングエグジットシグナル
        
        # ショートポジションの場合 - 価格が上限バンドを上回るか近づいたらエグジット
        elif position[i] == -1:
            if close[i] >= upper_band[i-lookback] - epsilon:
                signals[i] = -1  # ショートエグジットシグナル
    
    return signals


@njit(fastmath=True, cache=True)
def generate_position(entry_signals: np.ndarray, exit_signals: np.ndarray) -> np.ndarray:
    """
    エントリーとエグジットシグナルからポジションを生成
    
    Args:
        entry_signals: エントリーシグナル配列
        exit_signals: エグジットシグナル配列
        
    Returns:
        np.ndarray: ポジション配列（1:ロング, -1:ショート, 0:なし）
    """
    n = len(entry_signals)
    positions = np.zeros(n, dtype=np.int8)
    current_position = 0
    
    for i in range(n):
        # エグジットシグナルの処理
        if current_position == 1 and exit_signals[i] == 1:
            current_position = 0
        elif current_position == -1 and exit_signals[i] == -1:
            current_position = 0
        
        # エントリーシグナルの処理（ポジションがない場合のみ）
        if current_position == 0:
            if entry_signals[i] == 1:
                current_position = 1
            elif entry_signals[i] == -1:
                current_position = -1
        
        positions[i] = current_position
    
    return positions


class DoubleCCBreakoutSignal(BaseSignal, IEntrySignal, IExitSignal):
    """
    ダブルCチャネルブレイクアウトシグナル
    
    2つのCチャネルを使用したブレイクアウトシグナルを生成します：
    1. エントリー用Cチャネル（広めのバンド）
    2. エグジット用Cチャネル（狭めのバンド）
    
    エントリー条件:
    - ロング: 価格がエントリー用Cチャネルの上限バンドを上回る
    - ショート: 価格がエントリー用Cチャネルの下限バンドを下回る
    
    エグジット条件:
    - ロング: 価格がエグジット用Cチャネルの下限バンドを下回る
    - ショート: 価格がエグジット用Cチャネルの上限バンドを上回る
    """
    
    def __init__(
        self,
        # Cチャネル共通パラメータ
        detector_type: str = 'phac_e',
        cer_detector_type: str = None,  # CER用の検出器タイプ
        lp_period: int = 5,
        hp_period: int = 55,
        cycle_part: float = 0.7,
        smoother_type: str = 'alma',
        src_type: str = 'hlc3',
        band_lookback: int = 1,
        
        # エントリー用動的乗数の範囲パラメータ（より感度を高く）
        entry_max_max_multiplier: float = 5.0,    # 最大乗数の最大値（エントリー用）
        entry_min_max_multiplier: float = 3.0,    # 最大乗数の最小値（エントリー用）
        entry_max_min_multiplier: float = 2.0,    # 最小乗数の最大値（エントリー用）
        entry_min_min_multiplier: float = 1.2,    # 最小乗数の最小値（エントリー用）
        
        # エグジット用動的乗数の範囲パラメータ（より早いエグジットのため）
        exit_max_max_multiplier: float = 2.0,     # 最大乗数の最大値（エグジット用）
        exit_min_max_multiplier: float = 1.5,     # 最大乗数の最小値（エグジット用）
        exit_max_min_multiplier: float = 0.8,     # 最小乗数の最大値（エグジット用）
        exit_min_min_multiplier: float = 0.3,     # 最小乗数の最小値（エグジット用）
        
        # CMA用パラメータ
        cma_detector_type: str = 'hody_e',
        cma_cycle_part: float = 0.5,
        cma_lp_period: int = 5,
        cma_hp_period: int = 55,
        cma_max_cycle: int = 144,
        cma_min_cycle: int = 5,
        cma_max_output: int = 62,
        cma_min_output: int = 13,
        cma_fast_period: int = 2,
        cma_slow_period: int = 30,
        cma_src_type: str = 'hlc3',
        
        # CATR用パラメータ
        catr_detector_type: str = 'hody',
        catr_cycle_part: float = 0.5,
        catr_lp_period: int = 5,
        catr_hp_period: int = 55,
        catr_max_cycle: int = 55,
        catr_min_cycle: int = 5,
        catr_max_output: int = 34,
        catr_min_output: int = 5,
        catr_smoother_type: str = 'alma'
    ):
        """初期化"""
        super().__init__("DoubleCCBreakoutSignal")
        
        # CER検出器タイプの初期化（None の場合は detector_type を使用）
        if cer_detector_type is None:
            cer_detector_type = detector_type
        
        # エントリー用Cチャネルの初期化（広めのバンド）
        self.entry_c_channel = CChannel(
            # 基本パラメータ
            detector_type=detector_type,
            cer_detector_type=cer_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            smoother_type=smoother_type,
            src_type=src_type,
            
            # 動的乗数の範囲パラメータ（広め）
            max_max_multiplier=entry_max_max_multiplier,
            min_max_multiplier=entry_min_max_multiplier,
            max_min_multiplier=entry_max_min_multiplier,
            min_min_multiplier=entry_min_min_multiplier,
            
            # CMA用パラメータ
            cma_detector_type=cma_detector_type,
            cma_cycle_part=cma_cycle_part,
            cma_lp_period=cma_lp_period,
            cma_hp_period=cma_hp_period,
            cma_max_cycle=cma_max_cycle,
            cma_min_cycle=cma_min_cycle,
            cma_max_output=cma_max_output,
            cma_min_output=cma_min_output,
            cma_fast_period=cma_fast_period,
            cma_slow_period=cma_slow_period,
            cma_src_type=cma_src_type,
            
            # CATR用パラメータ
            catr_detector_type=catr_detector_type,
            catr_cycle_part=catr_cycle_part,
            catr_lp_period=catr_lp_period,
            catr_hp_period=catr_hp_period,
            catr_max_cycle=catr_max_cycle,
            catr_min_cycle=catr_min_cycle,
            catr_max_output=catr_max_output,
            catr_min_output=catr_min_output,
            catr_smoother_type=catr_smoother_type
        )
        
        # エグジット用Cチャネルの初期化（狭めのバンド）
        self.exit_c_channel = CChannel(
            # 基本パラメータ
            detector_type=detector_type,
            cer_detector_type=cer_detector_type,
            lp_period=lp_period,
            hp_period=hp_period,
            cycle_part=cycle_part,
            smoother_type=smoother_type,
            src_type=src_type,
            
            # 動的乗数の範囲パラメータ（狭め）
            max_max_multiplier=exit_max_max_multiplier,
            min_max_multiplier=exit_min_max_multiplier,
            max_min_multiplier=exit_max_min_multiplier,
            min_min_multiplier=exit_min_min_multiplier,
            
            # CMA用パラメータ
            cma_detector_type=cma_detector_type,
            cma_cycle_part=cma_cycle_part,
            cma_lp_period=cma_lp_period,
            cma_hp_period=cma_hp_period,
            cma_max_cycle=cma_max_cycle,
            cma_min_cycle=cma_min_cycle,
            cma_max_output=cma_max_output,
            cma_min_output=cma_min_output,
            cma_fast_period=cma_fast_period,
            cma_slow_period=cma_slow_period,
            cma_src_type=cma_src_type,
            
            # CATR用パラメータ
            catr_detector_type=catr_detector_type,
            catr_cycle_part=catr_cycle_part,
            catr_lp_period=catr_lp_period,
            catr_hp_period=catr_hp_period,
            catr_max_cycle=catr_max_cycle,
            catr_min_cycle=catr_min_cycle,
            catr_max_output=catr_max_output,
            catr_min_output=catr_min_output,
            catr_smoother_type=catr_smoother_type
        )
        
        # パラメータの保存
        self._params = {
            'detector_type': detector_type,
            'cer_detector_type': cer_detector_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'smoother_type': smoother_type,
            'src_type': src_type,
            'band_lookback': band_lookback,
            'entry_max_max_multiplier': entry_max_max_multiplier,
            'entry_min_max_multiplier': entry_min_max_multiplier,
            'entry_max_min_multiplier': entry_max_min_multiplier,
            'entry_min_min_multiplier': entry_min_min_multiplier,
            'exit_max_max_multiplier': exit_max_max_multiplier,
            'exit_min_max_multiplier': exit_min_max_multiplier,
            'exit_max_min_multiplier': exit_max_min_multiplier,
            'exit_min_min_multiplier': exit_min_min_multiplier
        }
        
        # ルックバック期間
        self.band_lookback = band_lookback
        
        # キャッシュの初期化
        self._cache = SignalCache()
    
    def _get_data_hash(self, ohlcv_data) -> str:
        """データのハッシュ値を計算して高速なキャッシュ検証を可能にする"""
        try:
            if isinstance(ohlcv_data, pd.DataFrame):
                # 高速なハッシュ計算のために重要なカラムだけを使用
                h = hashlib.md5()
                h.update(ohlcv_data['open'].values.tobytes())
                h.update(ohlcv_data['high'].values.tobytes())
                h.update(ohlcv_data['low'].values.tobytes())
                h.update(ohlcv_data['close'].values.tobytes())
                return h.hexdigest()
            else:
                return hashlib.md5(ohlcv_data.tobytes()).hexdigest()
        except Exception:
            # ハッシュ計算に失敗した場合はデータのIDを使用
            return str(id(ohlcv_data))
    
    def _convert_to_numpy(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """データをNumpy配列に変換（高速計算用）"""
        if isinstance(data, pd.DataFrame):
            return data[['open', 'high', 'low', 'close']].values
        return data

    def _check_and_update_cache(self, data: Union[pd.DataFrame, np.ndarray]) -> bool:
        """キャッシュ状態をチェックして必要に応じて更新"""
        data_hash = self._get_data_hash(data)
        
        # データが変更されていない場合はキャッシュを使用
        if (self._cache.entry_signals is not None and
            self._cache.data_hash == data_hash and 
            self._cache.data_len == len(data)):
            return True
        
        # データが変更された場合はハッシュを更新
        self._cache.data_hash = data_hash
        self._cache.data_len = len(data)
        return False
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを生成
        
        Args:
            data: 価格データ（DataFrameまたはnumpy配列）
            
        Returns:
            np.ndarray: エントリーシグナル配列 (1: ロング、-1: ショート、0: シグナルなし)
        """
        try:
            # キャッシュチェック - データが変わっていなければ再計算せずに返す
            if self._check_and_update_cache(data):
                return self._cache.entry_signals
            
            # データがPandasのDataFrameの場合、必要な列のみを抽出
            if isinstance(data, pd.DataFrame):
                df = data[['open', 'high', 'low', 'close']]
            else:
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
            
            # エントリー用Cチャネルの計算
            self.entry_c_channel.calculate(df)
            
            # バンド値の取得
            middle, upper, lower = self.entry_c_channel.get_bands()
            
            # NaNチェック - 安全な値に置き換え
            if np.isnan(upper).any() or np.isnan(lower).any():
                upper = np.nan_to_num(upper, nan=np.inf)
                lower = np.nan_to_num(lower, nan=-np.inf)
            
            # エントリーシグナル計算（Numba使用）
            self._cache.entry_signals = calculate_breakout_signals(
                df['close'].values, 
                upper, 
                lower, 
                self.band_lookback
            )
            
            # 追加のデバッグ情報
            # トレード信号の数をログに表示（開発/デバッグ用）
            long_signals = np.sum(self._cache.entry_signals == 1)
            short_signals = np.sum(self._cache.entry_signals == -1)
            if long_signals > 0 or short_signals > 0:
                print(f"生成されたエントリーシグナル - ロング: {long_signals}、ショート: {short_signals}")
            
            # キャッシュをクリア（バンド値とCER/CATR情報は必要なときに計算）
            self._cache.entry_bands = None
            self._cache.exit_bands = None
            
            return self._cache.entry_signals
            
        except Exception as e:
            import traceback
            print(f"エントリーシグナル生成中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.zeros(len(data), dtype=np.int8)
    
    def generate_exit(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エグジットシグナルを生成する
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, np.ndarray]
            価格データ
            
        Returns:
        --------
        np.ndarray
            エグジットシグナルの配列
        """
        # データフレームに変換（必要な場合）
        df = self._to_df(data)
        
        # バンド値計算
        if self._calculate_bands_for_exit:
            # エグジット用の独自バンドを計算
            upper, lower = self._calculate_bands(
                df, 
                self._entry_exit_split_ratio_upper * self._exit_multiplier_upper, 
                self._entry_exit_split_ratio_lower * self._exit_multiplier_lower
            )
        else:
            # 同じバンドを使用
            upper, lower = self._upper_band, self._lower_band
            
        # NaN値を安全に置き換え
        upper = np.nan_to_num(upper, nan=np.nanmean(upper))
        lower = np.nan_to_num(lower, nan=np.nanmean(lower))
        
        # 仮想的なポジションを計算
        positions = generate_position(self._cache.entry_signals, np.zeros_like(self._cache.entry_signals))
        
        # ポジションの分布を確認（デバッグ用）
        long_pos_count = np.sum(positions == 1)
        short_pos_count = np.sum(positions == -1)
        print(f"[ポジション分布] ロング: {long_pos_count}, ショート: {short_pos_count}, 合計: {len(positions)}")
        
        # エグジットシグナル計算
        self._cache.exit_signals = calculate_exit_signals(
            df['close'].values, 
            upper, 
            lower, 
            positions,
            self.band_lookback
        )
        
        # エグジットシグナルの詳細分析（デバッグ用）
        long_exits = np.sum(self._cache.exit_signals == 1)
        short_exits = np.sum(self._cache.exit_signals == -1)
        print(f"[エグジットシグナル] ロングエグジット: {long_exits}, ショートエグジット: {short_exits}")
        
        # ロングポジションがあるのにロングエグジットがない場合の詳細分析
        if long_pos_count > 0 and long_exits == 0:
            print(f"[警告] ロングポジションが{long_pos_count}件あるのに、ロングエグジットが0件です")
            
            # 価格と下限バンドの関係を分析
            long_pos_indices = np.where(positions == 1)[0]
            if len(long_pos_indices) > 0:
                close_prices = df['close'].values
                
                # 下限バンドの統計情報
                min_lower = np.min(lower[long_pos_indices])
                max_lower = np.max(lower[long_pos_indices])
                avg_lower = np.mean(lower[long_pos_indices])
                print(f"[下限バンド統計] 最小: {min_lower:.4f}, 最大: {max_lower:.4f}, 平均: {avg_lower:.4f}")
                
                # ロングポジションでの価格と下限バンドの比較
                for idx in long_pos_indices[:min(10, len(long_pos_indices))]:  # 最大10件まで表示
                    if idx < len(close_prices) and idx < len(lower):
                        price = close_prices[idx]
                        band = lower[idx]
                        diff = price - band
                        print(f"[ロングポジション詳細] インデックス: {idx}, 価格: {price:.4f}, 下限バンド: {band:.4f}, 差: {diff:.4f}")
        
        return self._cache.exit_signals
    
    def get_entry_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        エントリー用Cチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            # データが指定されていてエントリーシグナルが計算されていない場合は計算
            if data is not None and not self._check_and_update_cache(data):
                self.generate(data)
            
            # バンド値がキャッシュされていない場合は計算
            if self._cache.entry_bands is None:
                self._cache.entry_bands = self.entry_c_channel.get_bands()
                
            return self._cache.entry_bands
        except Exception as e:
            import traceback
            print(f"エントリーバンド値取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            empty = np.array([])
            return empty, empty, empty
    
    def get_exit_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        エグジット用Cチャネルのバンド値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        try:
            # データが指定されていてエントリーシグナルが計算されていない場合は計算
            if data is not None and not self._check_and_update_cache(data):
                self.generate(data)
                self.generate_exit(data)
            
            # バンド値がキャッシュされていない場合は計算
            if self._cache.exit_bands is None:
                self._cache.exit_bands = self.exit_c_channel.get_bands()
                
            return self._cache.exit_bands
        except Exception as e:
            import traceback
            print(f"エグジットバンド値取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            empty = np.array([])
            return empty, empty, empty
    
    def get_cycle_efficiency_ratio(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        サイクル効率比（CER）の値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: サイクル効率比の値
        """
        try:
            # データが指定されていてエントリーシグナルが計算されていない場合は計算
            if data is not None and not self._check_and_update_cache(data):
                self.generate(data)
            
            # CERがキャッシュされていない場合は計算
            if self._cache.cer is None:
                self._cache.cer = self.entry_c_channel.get_cycle_er()
                
            return self._cache.cer
        except Exception as e:
            import traceback
            print(f"CER取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def get_entry_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        エントリー用動的乗数の値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        try:
            # データが指定されていてエントリーシグナルが計算されていない場合は計算
            if data is not None and not self._check_and_update_cache(data):
                self.generate(data)
            
            # 動的乗数がキャッシュされていない場合は計算
            if self._cache.entry_multiplier is None:
                self._cache.entry_multiplier = self.entry_c_channel.get_dynamic_multiplier()
                
            return self._cache.entry_multiplier
        except Exception as e:
            import traceback
            print(f"エントリー用動的乗数取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def get_exit_dynamic_multiplier(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        エグジット用動的乗数の値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: 動的乗数の値
        """
        try:
            # データが指定されていてエントリーシグナルが計算されていない場合は計算
            if data is not None and not self._check_and_update_cache(data):
                self.generate(data)
                self.generate_exit(data)
            
            # 動的乗数がキャッシュされていない場合は計算
            if self._cache.exit_multiplier is None:
                self._cache.exit_multiplier = self.exit_c_channel.get_dynamic_multiplier()
                
            return self._cache.exit_multiplier
        except Exception as e:
            import traceback
            print(f"エグジット用動的乗数取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def get_c_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        CATR値を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            np.ndarray: CATR値
        """
        try:
            # データが指定されていてエントリーシグナルが計算されていない場合は計算
            if data is not None and not self._check_and_update_cache(data):
                self.generate(data)
            
            # CATRがキャッシュされていない場合は計算
            if self._cache.c_atr is None:
                self._cache.c_atr = self.entry_c_channel.get_c_atr()
                
            return self._cache.c_atr
        except Exception as e:
            import traceback
            print(f"CATR取得中にエラー: {str(e)}\n{traceback.format_exc()}")
            return np.array([])
    
    def reset(self) -> None:
        """キャッシュとシグナルをリセット"""
        self._cache = SignalCache()
        self.entry_c_channel.reset()
        self.exit_c_channel.reset() 