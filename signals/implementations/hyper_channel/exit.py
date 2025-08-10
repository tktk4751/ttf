#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange, float64, int8, boolean

from ...base_signal import BaseSignal
from ...interfaces.exit import IExitSignal
from indicators.hyper_channel import HyperChannel


@njit(boolean[:](float64[:], float64[:], float64[:], float64[:], int8), fastmath=True, parallel=True, cache=True)
def calculate_hyper_channel_exit_signals(
    close: np.ndarray, 
    midline: np.ndarray, 
    upper: np.ndarray, 
    lower: np.ndarray, 
    position: int
) -> np.ndarray:
    """
    ハイパーチャネルエグジットシグナルを計算する（高性能版）
    
    Args:
        close: 終値の配列
        midline: ミッドラインの配列
        upper: アッパーバンドの配列
        lower: ロワーバンドの配列
        position: ポジション（1=ロング、-1=ショート）
    
    Returns:
        エグジットシグナルの配列（True=エグジット）
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.bool_)
    
    # エグジット条件の判定（並列処理化）
    for i in prange(1, length):
        # 値が有効かチェック
        if (np.isnan(close[i]) or np.isnan(close[i-1]) or 
            np.isnan(midline[i]) or np.isnan(upper[i]) or np.isnan(lower[i])):
            signals[i] = False
            continue
        
        if position == 1:  # ロングポジション
            # 1. ミッドライン割れ
            midline_break = close[i-1] > midline[i-1] and close[i] <= midline[i]
            
            # 2. 下限バンドブレイクアウト
            lower_break = close[i-1] >= lower[i-1] and close[i] < lower[i]
            
            # 3. 価格が大幅に下落（緊急エグジット）
            emergency_exit = close[i] < close[i-1] * 0.98  # 2%下落
            
            signals[i] = midline_break or lower_break or emergency_exit
            
        elif position == -1:  # ショートポジション
            # 1. ミッドライン突破
            midline_break = close[i-1] < midline[i-1] and close[i] >= midline[i]
            
            # 2. 上限バンドブレイクアウト
            upper_break = close[i-1] <= upper[i-1] and close[i] > upper[i]
            
            # 3. 価格が大幅に上昇（緊急エグジット）
            emergency_exit = close[i] > close[i-1] * 1.02  # 2%上昇
            
            signals[i] = midline_break or upper_break or emergency_exit
    
    return signals


@njit(boolean[:](float64[:], float64[:], float64[:], int8), fastmath=True, cache=True)
def calculate_enhanced_exit_signals(
    close: np.ndarray,
    midline: np.ndarray,
    dynamic_multiplier: np.ndarray,
    position: int
) -> np.ndarray:
    """
    動的乗数を考慮した拡張エグジットシグナルを計算する（高性能版）
    
    Args:
        close: 終値の配列
        midline: ミッドラインの配列
        dynamic_multiplier: 動的乗数の配列
        position: ポジション（1=ロング、-1=ショート）
    
    Returns:
        拡張エグジットシグナルの配列（True=エグジット）
    """
    length = min(len(close), len(midline), len(dynamic_multiplier))
    signals = np.zeros(length, dtype=np.bool_)
    
    for i in range(1, length):
        # 値が有効かチェック
        if (np.isnan(close[i]) or np.isnan(close[i-1]) or 
            np.isnan(midline[i]) or np.isnan(dynamic_multiplier[i])):
            signals[i] = False
            continue
        
        # 動的閾値の計算
        # 高い乗数（レンジング）: より早いエグジット
        # 低い乗数（トレンド）: より遅いエグジット
        multiplier_factor = max(dynamic_multiplier[i], 0.5)
        adaptive_threshold = midline[i] * 0.002 * multiplier_factor  # 0.2%をベースライン
        
        if position == 1:  # ロングポジション
            # 適応的ミッドライン割れ
            price_distance = close[i] - midline[i]
            
            # 前回は閾値を上回っていたが、今回は下回った
            prev_distance = close[i-1] - midline[i-1]
            
            adaptive_exit = (prev_distance > adaptive_threshold and 
                           price_distance <= -adaptive_threshold)
            
            signals[i] = adaptive_exit
            
        elif position == -1:  # ショートポジション
            # 適応的ミッドライン突破
            price_distance = close[i] - midline[i]
            
            # 前回は閾値を下回っていたが、今回は上回った
            prev_distance = close[i-1] - midline[i-1]
            
            adaptive_exit = (prev_distance < -adaptive_threshold and 
                           price_distance >= adaptive_threshold)
            
            signals[i] = adaptive_exit
    
    return signals


class HyperChannelExitSignal(BaseSignal, IExitSignal):
    """
    ハイパーチャネルのエグジットシグナル
    
    特徴:
    - Unified Smootherによる高精度ミッドライン
    - ミッドライン回帰エグジット
    - バンドブレイクアウト（逆方向）エグジット
    - 動的乗数を考慮した拡張エグジット（オプション）
    - 緊急エグジット（急激な価格変動）
    
    エグジット条件:
    ロングポジション:
    - ミッドライン割れ（上から下へクロス）
    - 下限バンドブレイクアウト
    - 緊急エグジット（2%下落）
    
    ショートポジション:
    - ミッドライン突破（下から上へクロス）
    - 上限バンドブレイクアウト
    - 緊急エグジット（2%上昇）
    
    拡張モード（enhanced=True）:
    - 動的乗数に基づいた適応的閾値
    - レンジング時（高い乗数）: より早いエグジット
    - トレンド時（低い乗数）: より遅いエグジット
    """
    
    def __init__(
        self,
        # 基本パラメータ
        enhanced: bool = False,               # 拡張エグジットシグナルを使用
        enable_emergency_exit: bool = True,   # 緊急エグジットを有効化
        emergency_threshold: float = 0.02,    # 緊急エグジット閾値（2%）
        
        # ハイパーチャネルのパラメータ（オプション）
        hyper_channel_params: Dict[str, Any] = None
    ):
        """
        初期化
        
        Args:
            enhanced: 拡張エグジットシグナルを使用（デフォルト: False）
            enable_emergency_exit: 緊急エグジットを有効化（デフォルト: True）
            emergency_threshold: 緊急エグジット閾値（デフォルト: 0.02 = 2%）
            hyper_channel_params: HyperChannelに渡すパラメータ辞書（オプション）
        """
        # パラメータ設定
        hyper_params = hyper_channel_params or {}
        
        # チャネルパラメータ値を取得（シグナル名用）
        smoother_type = hyper_params.get('smoother_type', 'frama')
        adaptation_type = hyper_params.get('adaptation_type', 'hyper_er')
        
        mode_str = "Enhanced" if enhanced else "Standard"
        emergency_str = "Emergency" if enable_emergency_exit else "Normal"
        
        super().__init__(
            f"HyperChannelExitSignal({mode_str}, {emergency_str}, {smoother_type}, {adaptation_type})"
        )
        
        # 基本パラメータの保存
        self._params = {
            'enhanced': enhanced,
            'enable_emergency_exit': enable_emergency_exit,
            'emergency_threshold': emergency_threshold,
            **hyper_params  # その他のハイパーチャネルパラメータ
        }
        
        # パラメータのハッシュ値を事前計算（_get_data_hash処理の高速化）
        self._params_hash = hash(tuple(sorted(self._params.items())))
            
        # ハイパーチャネルの初期化（すべてのパラメータを渡す）
        self.hyper_channel = HyperChannel(**hyper_params)
        
        # 設定の保存
        self.enhanced = enhanced
        self.enable_emergency_exit = enable_emergency_exit
        self.emergency_threshold = emergency_threshold
        
        # キャッシュの初期化（サイズ制限付き）
        self._signals_cache = {}
        self._max_cache_size = 5  # キャッシュの最大サイズ
        self._cache_keys = []  # キャッシュキーの順序管理用
        
        # 最後に計算した結果のキャッシュ
        self._last_result = None
        self._last_data_hash = None
    
    def _get_data_hash(self, ohlcv_data, position):
        """
        データハッシュを取得する（超高速化版）
        
        Args:
            ohlcv_data: OHLCVデータ
            position: ポジション（1=ロング、-1=ショート）
            
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
            
            # データハッシュの計算（事前計算済みのパラメータハッシュとポジションを使用）
            return hash((self._params_hash, hash(data_signature), position))
            
        except Exception:
            # フォールバック: 最小限のハッシュ
            return hash((self._params_hash, id(ohlcv_data), position))
    
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
    
    def should_exit(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットすべきかどうかを判定する
        
        Args:
            data: 価格データ
            position: 現在のポジション（1: ロング、-1: ショート）
            index: チェックするインデックス（デフォルト: -1=最新）
        
        Returns:
            bool: エグジットすべきかどうか
        """
        try:
            # データの長さをチェック
            if isinstance(data, pd.DataFrame):
                data_len = len(data)
            else:
                data_len = data.shape[0]
            
            if data_len <= 1:
                return False
            
            # インデックスの調整
            if index == -1:
                index = data_len - 1
            
            if index < 1 or index >= data_len:
                return False
            
            # エグジットシグナルを生成
            exit_signals = self.generate(data, position)
            
            return bool(exit_signals[index])
            
        except Exception as e:
            print(f"HyperChannelExitSignal判定中にエラー: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def generate(self, data: Union[pd.DataFrame, np.ndarray], position: int) -> np.ndarray:
        """
        エグジットシグナルを生成する（高速化版）
        
        Args:
            data: 価格データ
            position: ポジション（1: ロング、-1: ショート）
        
        Returns:
            エグジットシグナルの配列（True=エグジット）
        """
        try:
            # データの長さをチェック
            if isinstance(data, pd.DataFrame):
                data_len = len(data)
            else:
                data_len = data.shape[0]
            
            if data_len <= 1:
                # データが少なすぎる場合はFalseシグナルを返す
                return np.zeros(data_len, dtype=np.bool_)
            
            # キャッシュチェック - 同じデータとポジションの場合は計算をスキップ
            data_hash = self._get_data_hash(data, position)
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
                
                # 計算が失敗した場合はFalseシグナルを返す
                if result is None or len(result.midline) == 0:
                    signals = np.zeros(data_len, dtype=np.bool_)
                    
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
            
            # エグジットシグナルの計算（高速化版）
            if self.enhanced:
                # 拡張エグジットシグナル（動的乗数考慮）
                signals = calculate_enhanced_exit_signals(
                    close,
                    result.midline,
                    result.dynamic_multiplier,
                    position
                )
            else:
                # 標準エグジットシグナル
                signals = calculate_hyper_channel_exit_signals(
                    close,
                    result.midline,
                    result.upper_band,
                    result.lower_band,
                    position
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
            # エラーが発生した場合は警告を出力し、Falseシグナルを返す
            print(f"HyperChannelExitSignal計算中にエラー: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # エラー時に新しいハッシュキーを生成せず、一時的なFalseシグナルを返す
            if isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
                return np.zeros(len(data), dtype=np.bool_)
            else:
                return np.array([], dtype=np.bool_)
    
    def get_band_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ハイパーチャネルのバンド値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上限バンド, 下限バンド)のタプル
        """
        if data is not None:
            # エグジットシグナルではポジションが不明なので、ダミーポジションで計算
            data_hash = self._get_data_hash(data, 1)  # ダミーポジション
            # データハッシュが最後に計算したものと同じかチェック
            if data_hash != self._last_data_hash or self._last_result is None:
                # 異なる場合は再計算が必要（ダミーポジションで）
                self.generate(data, 1)
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
            self.generate(data, 1)  # ダミーポジション
            
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
            self.generate(data, 1)  # ダミーポジション
            
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
            self.generate(data, 1)  # ダミーポジション
            
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