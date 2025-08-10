#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange, float64, int8

from ...base_signal import BaseSignal
from ...interfaces.direction import IDirectionSignal
from indicators.hyper_channel import HyperChannel


@njit(int8[:](float64[:], float64[:]), fastmath=True, parallel=True, cache=True)
def calculate_hyper_direction_signals(close: np.ndarray, midline: np.ndarray) -> np.ndarray:
    """
    ハイパーチャネルの方向シグナルを計算する（高性能版）
    
    Args:
        close: 終値の配列
        midline: ミッドラインの配列
    
    Returns:
        方向シグナルの配列 (1: ロング方向, -1: ショート方向, 0: 中立)
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # 方向の判定（並列処理化）
    for i in prange(1, length):
        # 終値とミッドラインの値が有効かチェック
        if np.isnan(close[i]) or np.isnan(midline[i]):
            signals[i] = 0
            continue
            
        # ロング方向: 終値がミッドラインより上
        if close[i] > midline[i]:
            signals[i] = 1
        # ショート方向: 終値がミッドラインより下
        elif close[i] < midline[i]:
            signals[i] = -1
        # 中立: 終値がミッドラインと同じ
        else:
            signals[i] = 0
    
    return signals


@njit(int8[:](float64[:], float64[:], float64[:]), fastmath=True, cache=True)
def calculate_enhanced_direction_signals(
    close: np.ndarray, 
    midline: np.ndarray, 
    dynamic_multiplier: np.ndarray
) -> np.ndarray:
    """
    動的乗数を考慮した拡張方向シグナルを計算する（高性能版）
    
    Args:
        close: 終値の配列
        midline: ミッドラインの配列
        dynamic_multiplier: 動的乗数の配列
    
    Returns:
        拡張方向シグナルの配列 (1: 強いロング方向, -1: 強いショート方向, 0: 中立)
    """
    length = min(len(close), len(midline), len(dynamic_multiplier))
    signals = np.zeros(length, dtype=np.int8)
    
    for i in range(1, length):
        # 値が有効かチェック
        if (np.isnan(close[i]) or np.isnan(midline[i]) or 
            np.isnan(dynamic_multiplier[i])):
            signals[i] = 0
            continue
        
        # 動的乗数に基づいたバイアス計算
        # 低い乗数（トレンド強い）: より敏感
        # 高い乗数（レンジング）: より保守的
        multiplier_bias = 1.0 / max(dynamic_multiplier[i], 0.1)
        
        price_difference = close[i] - midline[i]
        
        # バイアス調整された閾値
        threshold = midline[i] * 0.001 * multiplier_bias  # 0.1%をベースライン
        
        if price_difference > threshold:
            signals[i] = 1  # 強いロング方向
        elif price_difference < -threshold:
            signals[i] = -1  # 強いショート方向
        else:
            signals[i] = 0  # 中立
    
    return signals


class HyperChannelDirectionSignal(BaseSignal, IDirectionSignal):
    """
    ハイパーチャネルの方向シグナル
    
    特徴:
    - Unified Smootherによる高精度ミッドライン
    - 終値とミッドラインの位置関係による方向判定
    - 動的乗数を考慮した拡張方向シグナル（オプション）
    - X_ATRによるボラティリティベースの調整
    
    シグナル条件:
    - 終値 > ミッドライン: ロング方向 (1)
    - 終値 < ミッドライン: ショート方向 (-1)
    - 終値 = ミッドライン: 中立 (0)
    
    拡張モード（enhanced=True）:
    - 動的乗数に基づいた感度調整
    - トレンド時（低い乗数）: より敏感
    - レンジング時（高い乗数）: より保守的
    """
    
    def __init__(
        self,
        # 基本パラメータ
        enhanced: bool = False,               # 拡張方向シグナルを使用
        
        # ハイパーチャネルのパラメータ（オプション）
        hyper_channel_params: Dict[str, Any] = None
    ):
        """
        初期化
        
        Args:
            enhanced: 拡張方向シグナルを使用（デフォルト: False）
            hyper_channel_params: HyperChannelに渡すパラメータ辞書（オプション）
        """
        # パラメータ設定
        hyper_params = hyper_channel_params or {}
        
        # チャネルパラメータ値を取得（シグナル名用）
        smoother_type = hyper_params.get('smoother_type', 'frama')
        adaptation_type = hyper_params.get('adaptation_type', 'hyper_er')
        
        mode_str = "Enhanced" if enhanced else "Basic"
        
        super().__init__(
            f"HyperChannelDirectionSignal({mode_str}, {smoother_type}, {adaptation_type})"
        )
        
        # 基本パラメータの保存
        self._params = {
            'enhanced': enhanced,
            **hyper_params  # その他のハイパーチャネルパラメータ
        }
        
        # パラメータのハッシュ値を事前計算（_get_data_hash処理の高速化）
        self._params_hash = hash(tuple(sorted(self._params.items())))
            
        # ハイパーチャネルの初期化（すべてのパラメータを渡す）
        self.hyper_channel = HyperChannel(**hyper_params)
        
        # 設定の保存
        self.enhanced = enhanced
        
        # キャッシュの初期化（サイズ制限付き）
        self._signals_cache = {}
        self._max_cache_size = 5  # キャッシュの最大サイズ
        self._cache_keys = []  # キャッシュキーの順序管理用
        
        # 最後に計算した結果のキャッシュ
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
        方向シグナルを生成する（高速化版）
        
        Args:
            data: 価格データ
        
        Returns:
            方向シグナルの配列 (1: ロング方向, -1: ショート方向, 0: 中立)
        """
        try:
            # データの長さをチェック
            if isinstance(data, pd.DataFrame):
                data_len = len(data)
            else:
                data_len = data.shape[0]
            
            if data_len <= 1:
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
            
            # 結果がキャッシュされている場合はスキップ
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
            
            # 方向シグナルの計算（高速化版）
            if self.enhanced:
                # 拡張方向シグナル（動的乗数考慮）
                signals = calculate_enhanced_direction_signals(
                    close,
                    result.midline,
                    result.dynamic_multiplier
                )
            else:
                # 基本方向シグナル
                signals = calculate_hyper_direction_signals(
                    close,
                    result.midline
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
            print(f"HyperChannelDirectionSignal計算中にエラー: {str(e)}")
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
        
        # 結果のキャッシュもクリア
        self._last_result = None
        self._last_data_hash = None