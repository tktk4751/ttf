#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange, vectorize
import traceback
# from ...base_signal import BaseSignal
# from ...interfaces.entry import IEntrySignal

# --- 依存関係のインポート ---
try:
    # from signals.base.entry_signal import BaseEntrySignal, SignalResult
    from ...base_signal import BaseSignal # Correct import
    from ...interfaces.entry import IEntrySignal # Correct import
    from indicators.x_channel import XChannel # XChannelをインポート
    from indicators.price_source import PriceSource
except ImportError:
    # フォールバック (テストや静的解析用)
    print("Warning: Could not import from relative path. Assuming base classes/functions are available.")
    # ダミークラス定義（ここでは省略）
    class BaseSignal: # Dummy BaseSignal
        def __init__(self, *args, **kwargs): self.logger = self._get_logger()
        def generate(self, data): return np.zeros(len(data))
        def reset(self): pass
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class IEntrySignal: # Dummy Interface
        pass
    class XChannel: # Dummy XChannel
        def __init__(self, **kwargs): pass
        def calculate(self, data): return np.random.rand(len(data))
        def get_bands(self): return np.array([]), np.array([]), np.array([])
        def reset(self): pass

    class PriceSource: # Dummy
        @staticmethod
        def calculate_source(data, src_type): return data['close'].values if isinstance(data, pd.DataFrame) else data[:,3]


@vectorize(['int8(float64, float64, float64)'], nopython=True, fastmath=True, target='parallel', cache=True)
def check_breakout_vectorized(close: float, upper_band: float, lower_band: float) -> np.int8:
    """
    価格とバンドを比較してブレイクアウトシグナルを生成するベクトル化関数
    
    Args:
        close: 現在の価格
        upper_band: 上限バンド
        lower_band: 下限バンド
        
    Returns:
        np.int8: シグナル値（1=買い、-1=売り、0=なし）
    """
    # NaNチェック
    if np.isnan(close) or np.isnan(upper_band) or np.isnan(lower_band):
        return np.int8(0)
    
    # 上抜けブレイクアウト
    if close > upper_band:
        return np.int8(1)
    # 下抜けブレイクアウト
    elif close < lower_band:
        return np.int8(-1)
    # シグナルなし
    return np.int8(0)


@njit(fastmath=True, parallel=True, cache=True)
def check_breakout_signals_numba(
    close: np.ndarray,
    upper_band: np.ndarray,
    lower_band: np.ndarray
) -> np.ndarray:
    """
    Numbaを使用してブレイクアウトシグナルを高速に計算する（並列最適化版）

    Args:
        close (np.ndarray): 終値の配列
        upper_band (np.ndarray): Xチャネルの上限バンド
        lower_band (np.ndarray): Xチャネルの下限バンド

    Returns:
        np.ndarray: シグナル配列 (1: 買い, -1: 売り, 0: なし)
    """
    n = len(close)
    signals = np.zeros(n, dtype=np.int8)
    
    # 処理するデータサイズに基づいて最適化
    if n < 1000:
        # 小さなデータセットはシンプルに処理
        for i in range(1, n):
            # NaNチェック
            if np.isnan(close[i]) or np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
                continue
                
            # 買いシグナル (上抜けブレイクアウト)
            if close[i] > upper_band[i-1]:
                signals[i] = 1
            # 売りシグナル (下抜けブレイクアウト)
            elif close[i] < lower_band[i-1]:
                signals[i] = -1
    else:
        # 大きなデータセットはチャンク分割して並列処理
        chunk_size = max(1000, n // 16)  # 16はコア数の想定値
        num_chunks = (n + chunk_size - 1) // chunk_size  # 切り上げ除算
        
        for chunk_idx in prange(num_chunks):
            start_idx = max(1, chunk_idx * chunk_size)  # インデックス0はスキップ（前日比較のため）
            end_idx = min(start_idx + chunk_size, n)
            
            for i in range(start_idx, end_idx):
                # NaNチェック
                if np.isnan(close[i]) or np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
                    continue
                    
                # 買いシグナル (上抜けブレイクアウト)
                if close[i] > upper_band[i-1]:
                    signals[i] = 1
                # 売りシグナル (下抜けブレイクアウト)
                elif close[i] < lower_band[i-1]:
                    signals[i] = -1
    
    return signals


@njit(fastmath=True, cache=True)
def prepare_lookback_bands(upper_band: np.ndarray, lower_band: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    バンド値を1日シフトしてルックバック比較用に準備する（最適化版）
    
    Args:
        upper_band: 元の上限バンド配列
        lower_band: 元の下限バンド配列
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 1日シフトした上限・下限バンドのタプル
    """
    n = len(upper_band)
    
    # 新しい配列をプリアロケーション
    shifted_upper = np.empty_like(upper_band)
    shifted_lower = np.empty_like(lower_band)
    
    # インデックス0は無視（NaNまたは0で埋める）
    shifted_upper[0] = np.nan
    shifted_lower[0] = np.nan
    
    # 1日シフト（インデックスi-1の値をiに移動）
    for i in range(1, n):
        shifted_upper[i] = upper_band[i-1]
        shifted_lower[i] = lower_band[i-1]
    
    return shifted_upper, shifted_lower


class XChannelBreakoutEntrySignal(BaseSignal, IEntrySignal):
    """
    XChannelのバンドブレイクアウトに基づいてエントリーシグナルを生成するクラス（最適化版）。

    シグナルロジック:
    - 買い: 現在の終値が前日のXChannel上限バンドを上回った場合。
    - 売り: 現在の終値が前日のXChannel下限バンドを下回った場合。
    
    特徴:
    - NumbaとNumPyによる高速計算
    - 大規模データセットでの並列処理
    - 効率的なキャッシュ管理
    """

    def __init__(
        self,
        x_channel_params: Dict[str, Any], # XChannelインジケータのパラメータ
        use_close_confirmation: bool = True, # 後方互換性のため残す (無視される)
        band_lookback: int = 1, # 後方互換性のため残す (無視される)
        **kwargs # その他の余分な引数を受け入れる
    ):
        """
        コンストラクタ。

        Args:
            x_channel_params (Dict[str, Any]): XChannelインジケータに渡すパラメータの辞書。
            use_close_confirmation (bool, optional): 後方互換性のため残すパラメータ (使用されない)
            band_lookback (int, optional): 後方互換性のため残すパラメータ (使用されない)
        """
        # シンプルなシグナル名
        default_name = "XChBreakoutEntry"

        super().__init__(default_name)

        # パラメータの保存
        self._internal_params = {
            'x_channel_params': x_channel_params
        }
        
        # XChannelパラメータからソースタイプを抽出（デフォルトはclose）
        self._src_type = x_channel_params.get('xma_src_type', 'close')
        
        # XChannelインジケータのインスタンス化
        self.x_channel = XChannel(**x_channel_params)

        # インジケータ参照のリスト
        self._indicator_references = [self.x_channel]

        # 最適化用のベクトル化処理フラグ
        self._use_vectorized = True  # ベクトル化処理を有効にする（データサイズに応じて自動調整）
        self._large_data_threshold = 10000  # この値以上のデータサイズでチャンク処理を使用

        # 結果キャッシュ
        self._signals_cache: Dict[str, np.ndarray] = {}
        self._data_hash: Optional[str] = None
        
        # ロガーの設定
        self.logger = self._get_logger()

    def _get_logger(self):
        import logging
        # Use the class name for the logger
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _get_indicator_references(self) -> list:
        """使用するインジケータの参照リストを返す。"""
        return self._indicator_references

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する（最適化版）"""
        data_hash_val = None
        try:
            if isinstance(data, pd.DataFrame):
                # 必要なカラム (close) の値でハッシュ（高速化）
                if 'close' in data.columns:
                    # 最初と最後の10行だけを使ってハッシュ計算（メモリ効率）
                    n_rows = len(data)
                    if n_rows <= 20:
                        sample = data['close'].values
                    else:
                        # 最初と最後の10行を連結
                        sample = np.concatenate([
                            data['close'].values[:10],
                            data['close'].values[-10:]
                        ])
                    data_hash_val = hash(sample.tobytes())
                else:
                    # close列がない場合は全データをハッシュ
                    shape_str = f"{data.shape[0]}x{data.shape[1]}"
                    first_last = f"{data.iloc[0].sum()}_{data.iloc[-1].sum()}" if len(data) > 0 else "empty"
                    data_hash_val = hash(f"{shape_str}_{first_last}")
            elif isinstance(data, np.ndarray):
                # NumPy配列の場合も最初と最後の部分だけをハッシュ
                n_rows = len(data)
                if n_rows <= 20:
                    sample = data
                else:
                    # 最初と最後の10行を連結（メモリ効率）
                    sample = np.concatenate([data[:10], data[-10:]])
                data_hash_val = hash(sample.tobytes())
            else:
                data_hash_val = hash(str(data))
        except Exception as e:
            self.logger.warning(f"データハッシュ計算エラー: {e}. フォールバック使用.", exc_info=False)
            data_hash_val = hash(str(data))

        # シグナル固有パラメータとインジケータパラメータをハッシュに含める
        try:
            # パラメータのハッシュは前計算しておき再利用（パフォーマンス向上）
            if not hasattr(self, '_param_hash'):
                sorted_params = tuple(sorted((k, str(v)) for k, v in self._internal_params['x_channel_params'].items()))
                self._param_hash = hash(sorted_params)
            
            return f"{data_hash_val}_{self._param_hash}"
        except Exception as e:
            self.logger.warning(f"パラメータハッシュ計算エラー: {e}. 単純ハッシュ使用.", exc_info=False)
            return f"{data_hash_val}_params"

    def generate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        シグナル計算ロジックを実装する（最適化版）。

        Args:
            data (Union[pd.DataFrame, np.ndarray]): OHLCデータ。

        Returns:
            np.ndarray: エントリーシグナル (1: 買い, -1: 売り, 0: なし)
        """
        # データ長を取得（早期リターン用）
        current_data_len = len(data) if hasattr(data, '__len__') else 0
        if current_data_len == 0:
            self.logger.warning("入力データが空です。")
            return np.zeros(0, dtype=np.int8)

        try:
            # キャッシュチェック - 高速パス
            data_hash = self._get_data_hash(data)
            if data_hash in self._signals_cache:
                # 結果の長さを確認
                cached_signals = self._signals_cache[data_hash]
                if len(cached_signals) == current_data_len:
                    return cached_signals
                else:
                    self.logger.debug("Cache length mismatch, recalculating signal.")
                    # キャッシュ長不一致ならリセット（サイズ変更があった場合）
                    self.reset()

            # --- 1. データ準備 (最適化) --- 
            # NumPy配列に効率的に変換
            close = PriceSource.calculate_source(data, self._src_type)
            
            # close配列のデータ型を確認・修正（numba用）
            if not isinstance(close, np.ndarray):
                close = np.asarray(close, dtype=np.float64)
            elif close.dtype != np.float64:
                close = close.astype(np.float64)

            # --- 2. XChannel計算 --- 
            # calculateを呼び出すだけで内部で計算・キャッシュされる
            self.x_channel.calculate(data)
            # 計算されたバンドを取得
            _, upper_band, lower_band = self.x_channel.get_bands()
            
            # 配列をNumPy float64に統一（Numba最適化用）
            upper_band = np.asarray(upper_band, dtype=np.float64)
            lower_band = np.asarray(lower_band, dtype=np.float64)

            # 配列長の最終確認
            if not (len(close) == len(upper_band) == len(lower_band)):
                self.logger.error("Input arrays have inconsistent lengths after indicator calculation.")
                # Return zeros, but don't cache
                return np.zeros(current_data_len, dtype=np.int8)

            # --- 3. シグナル計算（データサイズに応じて最適な方法を選択）---
            signals = None
            
            if current_data_len <= 1:
                # 1要素以下の場合は単純に0を返す
                signals = np.zeros(current_data_len, dtype=np.int8)
            elif self._use_vectorized and current_data_len < self._large_data_threshold:
                try:
                    # ベクトル化処理用にバンドを1日シフト
                    shifted_upper, shifted_lower = prepare_lookback_bands(upper_band, lower_band)
                    
                    # ベクトル化関数を使用（小〜中規模データに最適）
                    signals = check_breakout_vectorized(close, shifted_upper, shifted_lower)
                    
                    # 最初の要素は常に0（前日比較のため）
                    if len(signals) > 0:
                        signals[0] = 0
                except Exception as e:
                    self.logger.warning(f"ベクトル化処理中にエラー: {e}, フォールバック使用")
                    # フォールバックとしてNumba関数を使用
                    signals = check_breakout_signals_numba(close, upper_band, lower_band)
            else:
                # 大規模データセット用のNumba最適化関数を使用
                signals = check_breakout_signals_numba(close, upper_band, lower_band)

            # --- 4. 結果をキャッシュして返す ---
            self._signals_cache[data_hash] = signals
            self._data_hash = data_hash
            return signals

        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Error calculating XChannel breakout signal: {error_msg}\n{stack_trace}")
            # エラー時はキャッシュ削除
            self._signals_cache.pop(data_hash, None)
            self._data_hash = None
            return np.zeros(current_data_len, dtype=np.int8)

    def reset(self):
        """シグナルの状態とキャッシュをリセットする。"""
        super().reset()
        self._signals_cache = {}
        self._data_hash = None
        # Reset the underlying indicator
        self.x_channel.reset()
        self.logger.debug(f"Signal '{self.name}' reset.") 