#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from numba import jit, njit, prange, float64, int8, boolean, int64, optional

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.hyper_channel import HyperChannel


@njit(int8[:](float64[:], float64[:], float64[:], int64), fastmath=True, parallel=True, cache=True)
def calculate_hyper_breakout_signals(close: np.ndarray, upper: np.ndarray, lower: np.ndarray, lookback: int) -> np.ndarray:
    """
    ハイパーチャネルブレイクアウトシグナルを計算する（高性能版）
    
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
    for i in prange(lookback + 1, length):
        # 終値とバンドの値が有効かチェック
        if (np.isnan(close[i]) or np.isnan(close[i-1]) or 
            np.isnan(upper[i]) or np.isnan(upper[i-1]) or 
            np.isnan(lower[i]) or np.isnan(lower[i-1])):
            signals[i] = 0
            continue
            
        # ロングエントリー: 前回の終値が前回のアッパーバンドを上回っていないかつ現在の終値が現在のアッパーバンドを上回る
        if close[i-1] <= upper[i-1] and close[i] > upper[i]:
            signals[i] = 1
        # ショートエントリー: 前回の終値が前回のロワーバンドを下回っていないかつ現在の終値が現在のロワーバンドを下回る
        elif close[i-1] >= lower[i-1] and close[i] < lower[i]:
            signals[i] = -1
        # 前回のチェックを追加（より多くのシグナルを生成）- 近似クロスオーバー
        elif lookback > 0 and i > lookback:
            # 直近の近似クロスオーバーもチェック
            if close[i] > close[i-1] and close[i-1] <= upper[i-1] and close[i] >= upper[i] * 0.995 and close[i-1] < upper[i-1] * 0.995:
                signals[i] = 1  # ほぼアッパーバンドでクロスオーバー
            elif close[i] < close[i-1] and close[i-1] >= lower[i-1] and close[i] <= lower[i] * 1.005 and close[i-1] > lower[i-1] * 1.005:
                signals[i] = -1  # ほぼロワーバンドでクロスオーバー
    
    return signals


@njit(fastmath=True, cache=True)
def extract_ohlc_from_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    NumPy配列からOHLC価格データを抽出する（高速化版）
    
    Args:
        data: 価格データの配列（OHLCフォーマット）
        
    Returns:
        open, high, low, closeの値をそれぞれ含むタプル
    """
    if data.ndim == 1:
        # 1次元配列の場合はすべて同じ値とみなす
        return data, data, data, data
    else:
        # 2次元配列の場合はOHLCとして抽出
        if data.shape[1] >= 4:
            return data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        elif data.shape[1] == 1:
            # 1列のみの場合はすべて同じ値とみなす
            return data[:, 0], data[:, 0], data[:, 0], data[:, 0]
        else:
            # 列数が不足している場合は終値のみ使用
            raise ValueError(f"データの列数が不足しています: 必要=4, 実際={data.shape[1]}")


class HyperChannelBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    ハイパーチャネルのブレイクアウトによるエントリーシグナル
    
    特徴:
    - Unified Smootherによる高精度ミッドライン
    - X_ATRを使用したボラティリティベースのバンド
    - HyperER/HyperADXによる動的適応乗数
    - カルマンフィルターとルーフィングフィルター統合
    
    シグナル条件:
    - 現在の終値が指定期間前のアッパーバンドを上回った場合: ロングエントリー (1)
    - 現在の終値が指定期間前のロワーバンドを下回った場合: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        # 基本パラメータ
        band_lookback: int = 1,
        
        # ハイパーチャネルのパラメータ（オプション）
        hyper_channel_params: Dict[str, Any] = None
    ):
        """
        初期化
        
        Args:
            band_lookback: 過去バンド参照期間（デフォルト: 1）
            hyper_channel_params: HyperChannelに渡すパラメータ辞書（オプション）
        """
        # パラメータ設定
        hyper_params = hyper_channel_params or {}
        
        # チャネルパラメータ値を取得（シグナル名用）
        smoother_type = hyper_params.get('smoother_type', 'frama')
        adaptation_type = hyper_params.get('adaptation_type', 'hyper_er')
        max_multiplier = hyper_params.get('max_multiplier', 6.0)
        min_multiplier = hyper_params.get('min_multiplier', 0.5)
        
        super().__init__(
            f"HyperChannelBreakoutEntrySignal({smoother_type}, {adaptation_type}, {max_multiplier}, {min_multiplier}, {band_lookback})"
        )
        
        # 基本パラメータの保存
        self._params = {
            'band_lookback': band_lookback,
            **hyper_params  # その他のハイパーチャネルパラメータ
        }
        
        # パラメータのハッシュ値を事前計算（_get_data_hash処理の高速化）
        self._params_hash = hash(tuple(sorted(self._params.items())))
            
        # ハイパーチャネルの初期化（すべてのパラメータを渡す）
        self.hyper_channel = HyperChannel(**hyper_params)
        
        # 参照期間の設定
        self.band_lookback = band_lookback
        
        # キャッシュの初期化（サイズ制限付き）
        self._signals_cache = {}
        self._max_cache_size = 5  # キャッシュの最大サイズ
        self._cache_keys = []  # キャッシュキーの順序管理用
        
        # 最後に計算したバンド値のキャッシュ
        self._last_result = None
        self._last_data_hash = None
        
    def _get_data_hash(self, ohlcv_data):
        """
        データハッシュを取得する（超高速化版）
        
        Args:
            ohlcv_data: OHLCVデータ
            
        Returns:
            データのハッシュ値
        """
        # 超高速化: 最小限のデータサンプリング
        try:
            if isinstance(ohlcv_data, pd.DataFrame):
                length = len(ohlcv_data)
                if length > 0:
                    first_close = float(ohlcv_data.iloc[0].get('close', ohlcv_data.iloc[0, -1]))
                    last_close = float(ohlcv_data.iloc[-1].get('close', ohlcv_data.iloc[-1, -1]))
                    data_signature = (length, first_close, last_close)
                else:
                    data_signature = (0, 0.0, 0.0)
            else:
                # NumPy配列の場合
                length = len(ohlcv_data)
                if length > 0:
                    if ohlcv_data.ndim > 1:
                        first_val = float(ohlcv_data[0, -1])  # 最後の列（通常close）
                        last_val = float(ohlcv_data[-1, -1])
                    else:
                        first_val = float(ohlcv_data[0])
                        last_val = float(ohlcv_data[-1])
                    data_signature = (length, first_val, last_val)
                else:
                    data_signature = (0, 0.0, 0.0)
            
            # データハッシュの計算（事前計算済みのパラメータハッシュを使用）
            return hash((self._params_hash, hash(data_signature)))
            
        except Exception:
            # フォールバック: 最小限のハッシュ
            return hash((self._params_hash, id(ohlcv_data)))
    
    def _extract_close(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        データから終値を効率的に抽出する（高速化版）
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 終値の配列
        """
        if isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                return data['close'].values
            else:
                raise ValueError("データには'close'カラムが必要です")
        else:
            # NumPy配列
            if data.ndim == 1:
                return data  # 1次元配列はそのまま終値として扱う
            elif data.shape[1] >= 4:
                return data[:, 3]  # 4列以上ある場合は4列目を終値として扱う
            elif data.shape[1] == 1:
                return data[:, 0]  # 1列のみの場合はその列を終値として扱う
            else:
                raise ValueError(f"データの列数が不足しています: 必要=4, 実際={data.shape[1]}")
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナルを生成する（高速化版）
        
        Args:
            data: 価格データ
        
        Returns:
            シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
        """
        try:
            # データの長さをチェック
            if isinstance(data, pd.DataFrame):
                data_len = len(data)
            else:
                data_len = data.shape[0]
            
            if data_len <= self.band_lookback + 1:
                # データが少なすぎる場合はゼロシグナルを返す
                return np.zeros(data_len, dtype=np.int8)
            
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            data_hash = self._get_data_hash(data)
            if data_hash in self._signals_cache:
                # キャッシュヒット - キャッシュキーの順序を更新
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                return self._signals_cache[data_hash]
            
            # 終値を取得
            close = self._extract_close(data)
            
            # バンド値がキャッシュされている場合はスキップ
            if data_hash == self._last_data_hash and self._last_result is not None:
                result = self._last_result
            else:
                # ハイパーチャネルの計算
                result = self.hyper_channel.calculate(data)
                
                # 計算が失敗した場合はゼロシグナルを返す
                if result is None or len(result.midline) == 0:
                    signals = np.zeros(data_len, dtype=np.int8)
                    
                    # キャッシュサイズ管理
                    if len(self._signals_cache) >= self._max_cache_size and self._cache_keys:
                        oldest_key = self._cache_keys.pop(0)
                        if oldest_key in self._signals_cache:
                            del self._signals_cache[oldest_key]
                    
                    # 結果をキャッシュ
                    self._signals_cache[data_hash] = signals
                    self._cache_keys.append(data_hash)
                    
                    return signals
                
                # 結果をキャッシュ
                self._last_result = result
                self._last_data_hash = data_hash
            
            # ブレイクアウトシグナルの計算（高速化版）
            signals = calculate_hyper_breakout_signals(
                close,
                result.upper_band,
                result.lower_band,
                self.band_lookback
            )
            
            # キャッシュサイズ管理
            if len(self._signals_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._signals_cache:
                    del self._signals_cache[oldest_key]
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            self._cache_keys.append(data_hash)
            
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"HyperChannelBreakoutEntrySignal計算中にエラー: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # エラー時に新しいハッシュキーを生成せず、一時的なゼロシグナルを返す
            if isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
                return np.zeros(len(data), dtype=np.int8)
            else:
                return np.array([], dtype=np.int8)
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ハイパーチャネルのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            data_hash = self._get_data_hash(data)
            # データハッシュが最後に計算したものと同じかチェック
            if data_hash != self._last_data_hash or self._last_result is None:
                # 異なる場合は再計算が必要
                self.generate(data)
            # バンド値はgenerate内でキャッシュされる
        
        # 最後に計算したバンド値を返す（なければハイパーチャネルから取得）
        if self._last_result is not None:
            return self._last_result.midline, self._last_result.upper_band, self._last_result.lower_band
        
        return self.hyper_channel.get_bands()
    
    def get_x_atr(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        X_ATRの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: X_ATRの値
        """
        if data is not None:
            self.generate(data)
            
        if self._last_result is not None:
            return self._last_result.x_atr
        
        return self.hyper_channel.get_x_atr()
    
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
            
        if self._last_result is not None:
            return self._last_result.dynamic_multiplier
        
        return self.hyper_channel.get_dynamic_multiplier()
    
    def get_adaptation_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        動的適応値の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 動的適応値の値
        """
        if data is not None:
            self.generate(data)
            
        if self._last_result is not None:
            return self._last_result.adaptation_values
        
        return self.hyper_channel.get_adaptation_values()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self.hyper_channel, 'reset'):
            self.hyper_channel.reset()
        
        # キャッシュをクリア
        self._signals_cache = {}
        self._cache_keys = []
        
        # バンド値のキャッシュもクリア
        self._last_result = None
        self._last_data_hash = None