#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from numba import jit, njit, prange, float64, int8, boolean, int64, optional

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.z_adaptive_flow import ZAdaptiveFlow


@njit(int8[:](float64[:], float64[:]), fastmath=True, cache=True)
def calculate_level_cross_signals(close_prices: np.ndarray, level_line: np.ndarray) -> np.ndarray:
    """
    終値とLevelラインのクロスオーバーに基づいてシグナルを計算する（高性能版）
    
    Args:
        close_prices: 終値の配列
        level_line: Levelラインの配列
    
    Returns:
        シグナルの配列
        1: ロングシグナル（前回終値 < Level、現在終値 > Level）
        -1: ショートシグナル（前回終値 > Level、現在終値 < Level）
        0: シグナルなし
    """
    length = len(close_prices)
    signals = np.zeros(length, dtype=np.int8)
    
    # Levelクロスの判定
    for i in range(1, length):
        # 前回と今回の終値とLevelラインをチェック
        prev_close = close_prices[i-1]
        curr_close = close_prices[i]
        prev_level = level_line[i-1]
        curr_level = level_line[i]
        
        # 有効性チェック（NaNや無限大でない）
        if (not np.isfinite(prev_close) or not np.isfinite(curr_close) or 
            not np.isfinite(prev_level) or not np.isfinite(curr_level)):
            continue
        
        # 前回終値 < Level かつ 現在終値 > Level: ロングシグナル
        if prev_close < prev_level and curr_close > curr_level:
            signals[i] = 1
        # 前回終値 > Level かつ 現在終値 < Level: ショートシグナル
        elif prev_close > prev_level and curr_close < curr_level:
            signals[i] = -1
    
    return signals


class ZAdaptiveFlowTrendChangeSignal(BaseSignal, IEntrySignal):
    """
    Z Adaptive FlowのLevelラインと終値のクロスオーバーによるシグナル
    
    特徴:
    - 終値とLevelラインのクロスオーバーを検出
    - 前回終値 < Level、現在終値 > Levelでロングシグナル
    - 前回終値 > Level、現在終値 < Levelでショートシグナル
    - クロスがない場合はシグナルなし
    
    シグナル条件:
    - 前回終値 < Level かつ 現在終値 > Level: ロングシグナル (1)
    - 前回終値 > Level かつ 現在終値 < Level: ショートシグナル (-1)
    - クロスなし: シグナルなし (0)
    """
    
    def __init__(
        self,
        # Z Adaptive Flowのパラメータ（オプション）
        z_adaptive_flow_params: Dict[str, Any] = None
    ):
        """
        初期化
        
        Args:
            z_adaptive_flow_params: ZAdaptiveFlowに渡すパラメータ辞書（オプション）
        """
        # パラメータ設定
        z_params = z_adaptive_flow_params or {}
        
        # MAタイプとボラティリティタイプを取得（シグナル名用）
        ma_type = z_params.get('ma_type', 'zlema')
        volatility_type = z_params.get('volatility_type', 'volatility')
        length = z_params.get('length', 10)
        
        super().__init__(
            f"ZAdaptiveFlowLevelCrossSignal({ma_type}, {volatility_type}, {length})"
        )
        
        # 基本パラメータの保存
        self._params = {
            **z_params  # Z Adaptive Flowパラメータ
        }
        
        # パラメータのハッシュ値を事前計算（_get_data_hash処理の高速化）
        self._params_hash = hash(tuple(sorted(self._params.items())))
            
        # Z Adaptive Flowの初期化（すべてのパラメータを渡す）
        self.z_adaptive_flow = ZAdaptiveFlow(**z_params)
        
        # キャッシュの初期化（サイズ制限付き）
        self._signals_cache = {}
        self._max_cache_size = 5  # キャッシュの最大サイズ
        self._cache_keys = []  # キャッシュキーの順序管理用
        
        # 最後に計算した結果のキャッシュ
        self._last_level_line = None
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
    
    def _extract_close_prices(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        データから終値を抽出する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 終値の配列
        """
        try:
            if isinstance(data, pd.DataFrame):
                if 'close' in data.columns:
                    return data['close'].values.astype(np.float64)
                else:
                    # 最後の列を終値として使用
                    return data.iloc[:, -1].values.astype(np.float64)
            else:
                # NumPy配列の場合
                if data.ndim > 1:
                    # 最後の列を終値として使用
                    return data[:, -1].astype(np.float64)
                else:
                    # 1次元配列の場合はそのまま使用
                    return data.astype(np.float64)
        except Exception:
            # エラー時は空の配列を返す
            return np.array([], dtype=np.float64)
    
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
            
            # Levelラインがキャッシュされている場合はスキップ
            if data_hash == self._last_data_hash and self._last_level_line is not None:
                level_line = self._last_level_line
            else:
                # Z Adaptive Flowの計算
                result = self.z_adaptive_flow.calculate(data)
                
                # 計算が失敗した場合はゼロシグナルを返す
                if result is None:
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
                
                # Levelラインの取得
                level_line = result.level
                
                # Levelラインをキャッシュ
                self._last_level_line = level_line
                self._last_data_hash = data_hash
            
            # 終値を抽出
            close_prices = self._extract_close_prices(data)
            
            # 終値とLevelラインの長さが一致しない場合はゼロシグナルを返す
            if len(close_prices) != len(level_line):
                signals = np.zeros(data_len, dtype=np.int8)
            else:
                # Levelクロスシグナルの計算（高速化版）
                signals = calculate_level_cross_signals(close_prices, level_line)
            
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
            print(f"ZAdaptiveFlowLevelCrossSignal計算中にエラー: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # エラー時に新しいハッシュキーを生成せず、一時的なゼロシグナルを返す
            if isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
                return np.zeros(len(data), dtype=np.int8)
            else:
                return np.array([], dtype=np.int8)
    
    def get_trend_state(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        トレンド状態を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: トレンド状態の値 (1: bullish, -1: bearish)
        """
        if data is not None:
            data_hash = self._get_data_hash(data)
            # データハッシュが最後に計算したものと同じかチェック
            if data_hash != self._last_data_hash or self._last_level_line is None:
                # 異なる場合は再計算が必要
                self.generate(data)
            # Levelラインはgenerate内でキャッシュされる
        
        # Z Adaptive FlowからトレンドStateを取得
        return self.z_adaptive_flow.get_trend_state()
    
    def get_level_line(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Levelラインを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Levelラインの値
        """
        if data is not None:
            data_hash = self._get_data_hash(data)
            # データハッシュが最後に計算したものと同じかチェック
            if data_hash != self._last_data_hash or self._last_level_line is None:
                # 異なる場合は再計算が必要
                self.generate(data)
            # Levelラインはgenerate内でキャッシュされる
        
        # 最後に計算したLevelラインを返す（なければZ Adaptive Flowから取得）
        if self._last_level_line is not None:
            return self._last_level_line
        
        # Z Adaptive Flowから直接取得
        result = self.z_adaptive_flow.get_detailed_result()
        if result is not None:
            return result.level
        
        return np.array([], dtype=np.float64)
    
    def get_detailed_result(self, data: Union[pd.DataFrame, np.ndarray] = None):
        """
        Z Adaptive Flowの詳細な計算結果を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            ZAdaptiveFlowResult: 詳細な計算結果
        """
        if data is not None:
            self.generate(data)
            
        return self.z_adaptive_flow.get_detailed_result()
    
    def get_trend_lines(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        メインのトレンドライン（basis, level）を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            basis, level のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self.z_adaptive_flow.get_trend_lines()
    
    def get_bands(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        バンド（upper, lower）を取得
        
        Args:
            data: オプションの価格データ
            
        Returns:
            upper, lower のタプル
        """
        if data is not None:
            self.generate(data)
            
        return self.z_adaptive_flow.get_bands()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self.z_adaptive_flow, 'reset'):
            self.z_adaptive_flow.reset()
        
        # キャッシュをクリア
        self._signals_cache = {}
        self._cache_keys = []
        
        # Levelラインのキャッシュもクリア
        self._last_level_line = None
        self._last_data_hash = None 