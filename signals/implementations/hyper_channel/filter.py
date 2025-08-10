#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange, float64, int8
from enum import Enum

from ...base_signal import BaseSignal
from ...interfaces.filter import IFilterSignal
from indicators.hyper_channel import HyperChannel
from indicators.trend_filter.hyper_er import HyperER
from indicators.hyper_trend_index import HyperTrendIndex
from indicators.trend_filter.hyper_adx import HyperADX


class HyperChannelFilterType(Enum):
    """ハイパーチャネルフィルター用のフィルタータイプ"""
    CHANNEL_POSITION = "channel_position"           # チャネル位置フィルター
    CHANNEL_WIDTH = "channel_width"                 # チャネル幅フィルター
    DYNAMIC_MULTIPLIER = "dynamic_multiplier"       # 動的乗数フィルター
    ADAPTATION_VALUES = "adaptation_values"         # 適応値フィルター
    COMBINED = "combined"                           # 統合フィルター


@njit(int8[:](float64[:], float64[:], float64[:]), fastmath=True, parallel=True, cache=True)
def calculate_channel_position_filter(
    close: np.ndarray, 
    upper: np.ndarray, 
    lower: np.ndarray
) -> np.ndarray:
    """
    チャネル位置フィルター: 終値のチャネル内位置に基づく
    
    Args:
        close: 終値の配列
        upper: 上限バンドの配列
        lower: 下限バンドの配列
    
    Returns:
        フィルターシグナルの配列 (1: 許可, -1: 禁止, 0: 中立)
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in prange(length):
        # 値が有効かチェック
        if np.isnan(close[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            signals[i] = 0
            continue
        
        # チャネル幅を計算
        channel_width = upper[i] - lower[i]
        if channel_width <= 0:
            signals[i] = 0
            continue
        
        # チャネル内での相対位置を計算 (0-1)
        relative_position = (close[i] - lower[i]) / channel_width
        
        # 位置に基づくフィルター
        if relative_position > 0.7:  # 上部70%以上
            signals[i] = 1  # ロング許可
        elif relative_position < 0.3:  # 下部30%以下
            signals[i] = 1  # ショート許可
        else:
            signals[i] = -1  # 中央付近は禁止
    
    return signals


@njit(int8[:](float64[:], float64[:]), fastmath=True, cache=True)
def calculate_channel_width_filter(
    upper: np.ndarray, 
    lower: np.ndarray
) -> np.ndarray:
    """
    チャネル幅フィルター: チャネル幅に基づく
    
    Args:
        upper: 上限バンドの配列
        lower: 下限バンドの配列
    
    Returns:
        フィルターシグナルの配列 (1: 許可, -1: 禁止, 0: 中立)
    """
    length = len(upper)
    signals = np.zeros(length, dtype=np.int8)
    
    # 移動平均でチャネル幅の基準を計算
    lookback = min(20, length // 2)
    
    for i in range(lookback, length):
        # 値が有効かチェック
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            signals[i] = 0
            continue
        
        # 現在のチャネル幅
        current_width = upper[i] - lower[i]
        if current_width <= 0:
            signals[i] = 0
            continue
        
        # 過去の平均幅を計算
        avg_width = 0.0
        valid_count = 0
        for j in range(max(0, i - lookback), i):
            if not (np.isnan(upper[j]) or np.isnan(lower[j])):
                width = upper[j] - lower[j]
                if width > 0:
                    avg_width += width
                    valid_count += 1
        
        if valid_count == 0:
            signals[i] = 0
            continue
        
        avg_width /= valid_count
        
        # 相対的な幅に基づくフィルター
        width_ratio = current_width / avg_width
        
        if width_ratio > 1.2:  # 平均より20%以上広い
            signals[i] = 1  # ブレイクアウト許可
        elif width_ratio < 0.8:  # 平均より20%以上狭い
            signals[i] = -1  # レンジング状態、トレード禁止
        else:
            signals[i] = 0  # 中立
    
    return signals


@njit(int8[:](float64[:]), fastmath=True, cache=True)
def calculate_dynamic_multiplier_filter(dynamic_multiplier: np.ndarray) -> np.ndarray:
    """
    動的乗数フィルター: 動的乗数の値に基づく
    
    Args:
        dynamic_multiplier: 動的乗数の配列
    
    Returns:
        フィルターシグナルの配列 (1: 許可, -1: 禁止, 0: 中立)
    """
    length = len(dynamic_multiplier)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in range(length):
        # 値が有効かチェック  
        if np.isnan(dynamic_multiplier[i]):
            signals[i] = 0
            continue
        
        multiplier = dynamic_multiplier[i]
        
        # 動的乗数に基づくフィルター
        if multiplier < 1.5:  # 低い乗数（強いトレンド）
            signals[i] = 1  # トレード許可
        elif multiplier > 4.0:  # 高い乗数（レンジング）
            signals[i] = -1  # トレード禁止
        else:
            signals[i] = 0  # 中立
    
    return signals


@njit(int8[:](float64[:]), fastmath=True, cache=True)
def calculate_adaptation_values_filter(adaptation_values: np.ndarray) -> np.ndarray:
    """
    適応値フィルター: HyperER/HyperADXの適応値に基づく
    
    Args:
        adaptation_values: 適応値の配列
    
    Returns:
        フィルターシグナルの配列 (1: 許可, -1: 禁止, 0: 中立)
    """
    length = len(adaptation_values)
    signals = np.zeros(length, dtype=np.int8)
    
    for i in range(length):
        # 値が有効かチェック
        if np.isnan(adaptation_values[i]):
            signals[i] = 0
            continue
        
        value = abs(adaptation_values[i])  # 絶対値で評価
        
        # 適応値に基づくフィルター
        if value > 0.7:  # 高い効率性/強いトレンド
            signals[i] = 1  # トレード許可
        elif value < 0.3:  # 低い効率性/弱いトレンド
            signals[i] = -1  # トレード禁止
        else:
            signals[i] = 0  # 中立
    
    return signals


@njit(int8[:](int8[:], int8[:], int8[:], int8[:]), fastmath=True, cache=True)
def combine_filter_signals(
    position_signals: np.ndarray,
    width_signals: np.ndarray,
    multiplier_signals: np.ndarray,
    adaptation_signals: np.ndarray
) -> np.ndarray:
    """
    複数のフィルターシグナルを統合する
    
    Args:
        position_signals: チャネル位置フィルターシグナル
        width_signals: チャネル幅フィルターシグナル  
        multiplier_signals: 動的乗数フィルターシグナル
        adaptation_signals: 適応値フィルターシグナル
    
    Returns:
        統合フィルターシグナルの配列 (1: 許可, -1: 禁止, 0: 中立)
    """
    length = min(len(position_signals), len(width_signals), 
                len(multiplier_signals), len(adaptation_signals))
    signals = np.zeros(length, dtype=np.int8)
    
    for i in range(length):
        # 各フィルターの投票を集計
        positive_votes = 0
        negative_votes = 0
        
        filters = [position_signals[i], width_signals[i], 
                  multiplier_signals[i], adaptation_signals[i]]
        
        for signal in filters:
            if signal == 1:
                positive_votes += 1
            elif signal == -1:
                negative_votes += 1
        
        # 多数決で決定
        if positive_votes > negative_votes and positive_votes >= 2:
            signals[i] = 1  # 許可
        elif negative_votes > positive_votes and negative_votes >= 2:
            signals[i] = -1  # 禁止
        else:
            signals[i] = 0  # 中立
    
    return signals


class HyperChannelFilterSignal(BaseSignal, IFilterSignal):
    """
    ハイパーチャネルのフィルターシグナル
    
    特徴:
    - チャネル位置フィルター: 終値のチャネル内位置に基づく
    - チャネル幅フィルター: チャネル幅の変化に基づく
    - 動的乗数フィルター: 動的乗数の値に基づく
    - 適応値フィルター: HyperER/HyperADXの適応値に基づく
    - 統合フィルター: 複数フィルターの多数決
    
    フィルター条件:
    チャネル位置フィルター:
    - 終値が上部70%以上または下部30%以下: 許可 (1)
    - 終値が中央付近: 禁止 (-1)
    
    チャネル幅フィルター:
    - 平均より20%以上広い: 許可 (1, ブレイクアウト環境)
    - 平均より20%以上狭い: 禁止 (-1, レンジング環境)
    
    動的乗数フィルター:
    - 乗数 < 1.5 (強いトレンド): 許可 (1)
    - 乗数 > 4.0 (レンジング): 禁止 (-1)
    
    適応値フィルター:
    - 適応値 > 0.7 (高効率/強トレンド): 許可 (1)
    - 適応値 < 0.3 (低効率/弱トレンド): 禁止 (-1)
    """
    
    def __init__(
        self,
        # フィルタータイプ選択
        filter_type: HyperChannelFilterType = HyperChannelFilterType.COMBINED,
        
        # チャネル位置フィルターパラメータ
        position_upper_threshold: float = 0.7,        # 上部閾値
        position_lower_threshold: float = 0.3,        # 下部閾値
        
        # チャネル幅フィルターパラメータ
        width_lookback: int = 20,                     # 平均幅計算期間
        width_expansion_threshold: float = 1.2,       # 拡張閾値
        width_contraction_threshold: float = 0.8,     # 収縮閾値
        
        # 動的乗数フィルターパラメータ
        multiplier_trend_threshold: float = 1.5,      # トレンド閾値
        multiplier_range_threshold: float = 4.0,      # レンジング閾値
        
        # 適応値フィルターパラメータ
        adaptation_high_threshold: float = 0.7,       # 高効率閾値
        adaptation_low_threshold: float = 0.3,        # 低効率閾値
        
        # ハイパーチャネルのパラメータ（オプション）
        hyper_channel_params: Dict[str, Any] = None
    ):
        """
        初期化
        
        Args:
            filter_type: フィルタータイプ
            position_upper_threshold: チャネル位置フィルターの上部閾値
            position_lower_threshold: チャネル位置フィルターの下部閾値
            width_lookback: チャネル幅フィルターの平均幅計算期間
            width_expansion_threshold: チャネル幅拡張閾値
            width_contraction_threshold: チャネル幅収縮閾値
            multiplier_trend_threshold: 動的乗数トレンド閾値
            multiplier_range_threshold: 動的乗数レンジング閾値  
            adaptation_high_threshold: 適応値高効率閾値
            adaptation_low_threshold: 適応値低効率閾値
            hyper_channel_params: HyperChannelに渡すパラメータ辞書（オプション）
        """
        # パラメータ設定
        hyper_params = hyper_channel_params or {}
        
        # チャネルパラメータ値を取得（シグナル名用）
        smoother_type = hyper_params.get('smoother_type', 'frama')
        adaptation_type = hyper_params.get('adaptation_type', 'hyper_er')
        
        filter_name = filter_type.value if isinstance(filter_type, HyperChannelFilterType) else str(filter_type)
        
        super().__init__(
            f"HyperChannelFilterSignal({filter_name}, {smoother_type}, {adaptation_type})"
        )
        
        # 基本パラメータの保存
        self._params = {
            'filter_type': filter_type,
            'position_upper_threshold': position_upper_threshold,
            'position_lower_threshold': position_lower_threshold,
            'width_lookback': width_lookback,
            'width_expansion_threshold': width_expansion_threshold,
            'width_contraction_threshold': width_contraction_threshold,
            'multiplier_trend_threshold': multiplier_trend_threshold,
            'multiplier_range_threshold': multiplier_range_threshold,
            'adaptation_high_threshold': adaptation_high_threshold,
            'adaptation_low_threshold': adaptation_low_threshold,
            **hyper_params  # その他のハイパーチャネルパラメータ
        }
        
        # パラメータのハッシュ値を事前計算（_get_data_hash処理の高速化）
        self._params_hash = hash(tuple(sorted(self._params.items())))
            
        # ハイパーチャネルの初期化（すべてのパラメータを渡す）
        self.hyper_channel = HyperChannel(**hyper_params)
        
        # 設定の保存
        self.filter_type = filter_type if isinstance(filter_type, HyperChannelFilterType) else HyperChannelFilterType(filter_type)
        self.position_upper_threshold = position_upper_threshold
        self.position_lower_threshold = position_lower_threshold
        self.width_lookback = width_lookback 
        self.width_expansion_threshold = width_expansion_threshold
        self.width_contraction_threshold = width_contraction_threshold
        self.multiplier_trend_threshold = multiplier_trend_threshold
        self.multiplier_range_threshold = multiplier_range_threshold
        self.adaptation_high_threshold = adaptation_high_threshold
        self.adaptation_low_threshold = adaptation_low_threshold
        
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
        フィルターシグナルを生成する（高速化版）
        
        Args:
            data: 価格データ
        
        Returns:
            フィルターシグナルの配列 (1: 許可, -1: 禁止, 0: 中立)
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
            
            # フィルターシグナルの計算（高速化版）
            if self.filter_type == HyperChannelFilterType.CHANNEL_POSITION:
                signals = calculate_channel_position_filter(
                    close, result.upper_band, result.lower_band
                )
            elif self.filter_type == HyperChannelFilterType.CHANNEL_WIDTH:
                signals = calculate_channel_width_filter(
                    result.upper_band, result.lower_band
                )
            elif self.filter_type == HyperChannelFilterType.DYNAMIC_MULTIPLIER:
                signals = calculate_dynamic_multiplier_filter(
                    result.dynamic_multiplier
                )
            elif self.filter_type == HyperChannelFilterType.ADAPTATION_VALUES:
                signals = calculate_adaptation_values_filter(
                    result.adaptation_values
                )
            elif self.filter_type == HyperChannelFilterType.COMBINED:
                # 統合フィルター: 全ての個別フィルターを計算して統合
                position_signals = calculate_channel_position_filter(
                    close, result.upper_band, result.lower_band
                )
                width_signals = calculate_channel_width_filter(
                    result.upper_band, result.lower_band
                )
                multiplier_signals = calculate_dynamic_multiplier_filter(
                    result.dynamic_multiplier
                )
                adaptation_signals = calculate_adaptation_values_filter(
                    result.adaptation_values
                )
                
                signals = combine_filter_signals(
                    position_signals, width_signals, 
                    multiplier_signals, adaptation_signals
                )
            else:
                # デフォルト: 中立シグナル
                signals = np.zeros(data_len, dtype=np.int8)
            
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
            print(f"HyperChannelFilterSignal計算中にエラー: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # エラー時に新しいハッシュキーを生成せず、一時的なゼロシグナルを返す
            if isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
                return np.zeros(len(data), dtype=np.int8)
            else:
                return np.array([], dtype=np.int8)
    
    def should_allow_trade(self, data: Union[pd.DataFrame, np.ndarray], index: int = -1) -> bool:
        """
        トレードを許可すべきかどうかを判定する
        
        Args:
            data: 価格データ
            index: チェックするインデックス（デフォルト: -1=最新）
        
        Returns:
            bool: トレードを許可すべきかどうか
        """
        try:
            filter_signals = self.generate(data)
            
            if index == -1:
                index = len(filter_signals) - 1
            
            if index < 0 or index >= len(filter_signals):
                return False
            
            return filter_signals[index] == 1
            
        except Exception as e:
            print(f"HyperChannelFilterSignal判定中にエラー: {str(e)}")
            return False
    
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