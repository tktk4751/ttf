#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from numba import jit, njit, prange, float64, int8, boolean, int64, optional

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.cosmic_universal_adaptive_channel import CosmicUniversalAdaptiveChannel


@njit(int8[:](float64[:], float64[:], float64[:], int64), fastmath=True, parallel=True, cache=True)
def calculate_cosmic_breakout_signals(
    close: np.ndarray, 
    upper: np.ndarray, 
    lower: np.ndarray, 
    lookback: int
) -> np.ndarray:
    """
    宇宙チャネルブレイクアウトシグナルを計算する（高性能版）
    
    Args:
        close: 終値の配列
        upper: 上部宇宙チャネルの配列
        lower: 下部宇宙チャネルの配列
        lookback: 過去のチャネルを参照する期間
    
    Returns:
        シグナルの配列 (1: ロング, -1: ショート, 0: シグナルなし)
    """
    length = len(close)
    signals = np.zeros(length, dtype=np.int8)
    
    # ブレイクアウトの判定（並列処理化）
    for i in prange(lookback + 1, length):
        # 終値とチャネルの値が有効かチェック
        if (np.isnan(close[i]) or np.isnan(close[i-1]) or 
            np.isnan(upper[i]) or np.isnan(upper[i-1]) or 
            np.isnan(lower[i]) or np.isnan(lower[i-1])):
            signals[i] = 0
            continue
            
        # ロングエントリー: 前回の終値が前回の上部チャネル以下かつ現在の終値が現在の上部チャネル以上
        if close[i-1] <= upper[i-1] and close[i] >= upper[i]:
            signals[i] = 1
        # ショートエントリー: 前回の終値が前回の下部チャネル以上かつ現在の終値が現在の下部チャネル以下
        elif close[i-1] >= lower[i-1] and close[i] <= lower[i]:
            signals[i] = -1
        # 追加の近似ブレイクアウトも検出（より多くのシグナルを生成）
        elif lookback > 0 and i > lookback:
            # 上方向の近似ブレイクアウト
            if (close[i] > close[i-1] and 
                close[i-1] <= upper[i-1] and 
                close[i] >= upper[i] * 0.9995 and 
                close[i-1] < upper[i-1] * 0.9995):
                signals[i] = 1
            # 下方向の近似ブレイクアウト
            elif (close[i] < close[i-1] and 
                  close[i-1] >= lower[i-1] and 
                  close[i] <= lower[i] * 1.0005 and 
                  close[i-1] > lower[i-1] * 1.0005):
                signals[i] = -1
    
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


class CosmicUniversalAdaptiveChannelBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    宇宙統一適応チャネルのブレイクアウトによるエントリーシグナル
    
    特徴:
    - 量子統計熱力学エンジンによる動的適応性
    - フラクタル液体力学システムによる市場フロー解析
    - ヒルベルト・ウェーブレット多重解像度解析
    - 適応カオス理論センターライン
    - 宇宙統計エントロピーフィルター
    - 多次元ベイズ適応システム
    
    シグナル条件:
    - 前回終値が前回の上部チャネル以下かつ現在終値が現在の上部チャネル以上: ロングエントリー (1)
    - 前回終値が前回の下部チャネル以上かつ現在終値が現在の下部チャネル以下: ショートエントリー (-1)
    """
    
    def __init__(
        self,
        # 基本パラメータ
        channel_lookback: int = 1,
        
        # 宇宙チャネルのパラメータ（オプション）
        cosmic_channel_params: Dict[str, Any] = None
    ):
        """
        初期化
        
        Args:
            channel_lookback: 過去チャネル参照期間（デフォルト: 1）
            cosmic_channel_params: CosmicUniversalAdaptiveChannelに渡すパラメータ辞書（オプション）
        """
        # パラメータ設定
        cosmic_params = cosmic_channel_params or {}
        
        # チャネルパラメータを取得（シグナル名用）
        quantum_window = cosmic_params.get('quantum_window', 34)
        fractal_window = cosmic_params.get('fractal_window', 21)
        chaos_window = cosmic_params.get('chaos_window', 55)
        base_multiplier = cosmic_params.get('base_multiplier', 2.0)
        
        super().__init__(
            f"CosmicUniversalAdaptiveChannelBreakoutEntrySignal(q={quantum_window}, f={fractal_window}, c={chaos_window}, m={base_multiplier}, lb={channel_lookback})"
        )
        
        # 基本パラメータの保存
        self._params = {
            'channel_lookback': channel_lookback,
            **cosmic_params  # その他の宇宙チャネルパラメータ
        }
        
        # パラメータのハッシュ値を事前計算（_get_data_hash処理の高速化）
        self._params_hash = hash(tuple(sorted(self._params.items())))
            
        # 宇宙統一適応チャネルの初期化（すべてのパラメータを渡す）
        self.cosmic_channel = CosmicUniversalAdaptiveChannel(**cosmic_params)
        
        # 参照期間の設定
        self.channel_lookback = channel_lookback
        
        # キャッシュの初期化（サイズ制限付き）
        self._signals_cache = {}
        self._max_cache_size = 5  # キャッシュの最大サイズ
        self._cache_keys = []  # キャッシュキーの順序管理用
        
        # 最後に計算したチャネル値のキャッシュ
        self._last_centerline = None
        self._last_upper = None
        self._last_lower = None
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
            
            if data_len <= self.channel_lookback + 1:
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
            
            # チャネル値がキャッシュされている場合はスキップ
            if (data_hash == self._last_data_hash and 
                self._last_upper is not None and 
                self._last_lower is not None):
                centerline, upper, lower = self._last_centerline, self._last_upper, self._last_lower
                result = self._last_result
            else:
                # 宇宙統一適応チャネルの計算
                result = self.cosmic_channel.calculate(data)
                
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
                
                # チャネルの取得
                centerline = result.cosmic_centerline
                upper = result.upper_channel
                lower = result.lower_channel
                
                # チャネル値をキャッシュ
                self._last_centerline = centerline
                self._last_upper = upper
                self._last_lower = lower
                self._last_result = result
                self._last_data_hash = data_hash
            
            # ブレイクアウトシグナルの計算（高速化版）
            signals = calculate_cosmic_breakout_signals(
                close,
                upper,
                lower,
                self.channel_lookback
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
            print(f"CosmicUniversalAdaptiveChannelBreakoutEntrySignal計算中にエラー: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # エラー時に新しいハッシュキーを生成せず、一時的なゼロシグナルを返す
            if isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
                return np.zeros(len(data), dtype=np.int8)
            else:
                return np.array([], dtype=np.int8)
    
    def get_channel_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        宇宙統一適応チャネルのチャネル値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上部チャネル, 下部チャネル)のタプル
        """
        if data is not None:
            data_hash = self._get_data_hash(data)
            # データハッシュが最後に計算したものと同じかチェック
            if data_hash != self._last_data_hash or self._last_upper is None:
                # 異なる場合は再計算が必要
                self.generate(data)
        
        # 最後に計算したチャネル値を返す
        if (self._last_centerline is not None and 
            self._last_upper is not None and 
            self._last_lower is not None):
            return self._last_centerline, self._last_upper, self._last_lower
        
        # フォールバック: 空の配列を返す
        empty_array = np.array([])
        return empty_array, empty_array, empty_array
    
    def get_cosmic_intelligence_report(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict:
        """
        宇宙知能レポートを取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict: 宇宙知能レポート
        """
        if data is not None:
            self.generate(data)
            
        if self._last_result is not None:
            return self.cosmic_channel.get_cosmic_intelligence_report()
        else:
            return {"status": "no_data"}
    
    def get_quantum_entanglement(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        量子もつれ強度の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 量子もつれ強度の値
        """
        if data is not None:
            self.generate(data)
            
        if self._last_result is not None:
            return self._last_result.quantum_entanglement
        else:
            return np.array([])
    
    def get_fractal_dimension(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        フラクタル次元の値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フラクタル次元の値
        """
        if data is not None:
            self.generate(data)
            
        if self._last_result is not None:
            return self._last_result.fractal_dimension
        else:
            return np.array([])
    
    def get_cosmic_phase(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        宇宙フェーズの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 宇宙フェーズの値
        """
        if data is not None:
            self.generate(data)
            
        if self._last_result is not None:
            return self._last_result.cosmic_phase
        else:
            return np.array([])
    
    def get_omniscient_confidence(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        全知信頼度スコアの値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 全知信頼度スコアの値
        """
        if data is not None:
            self.generate(data)
            
        if self._last_result is not None:
            return self._last_result.omniscient_confidence
        else:
            return np.array([])
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        if hasattr(self.cosmic_channel, 'reset'):
            self.cosmic_channel.reset()
        
        # キャッシュをクリア
        self._signals_cache = {}
        self._cache_keys = []
        
        # チャネル値のキャッシュもクリア
        self._last_centerline = None
        self._last_upper = None
        self._last_lower = None
        self._last_result = None
        self._last_data_hash = None 