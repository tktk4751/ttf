#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.mama import MAMA


@njit(fastmath=True, parallel=True)
def calculate_crossover_signals(
    mama_values: np.ndarray, 
    fama_values: np.ndarray
) -> np.ndarray:
    """
    MAMAとFAMAのクロスオーバーシグナルを計算する（高速化版）
    
    Args:
        mama_values: MAMA値の配列
        fama_values: FAMA値の配列
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(mama_values)
    signals = np.zeros(length, dtype=np.int8)
    
    # 最初のポイントは前の値がないためシグナルなし
    if length < 2:
        return signals
    
    # クロスオーバーの判定（並列処理化）
    for i in prange(1, length):
        # 現在値と前の値が有効かチェック
        if (np.isnan(mama_values[i]) or np.isnan(fama_values[i]) or 
            np.isnan(mama_values[i-1]) or np.isnan(fama_values[i-1])):
            signals[i] = 0
            continue
            
        # 前の時点での位置関係
        prev_mama_above = mama_values[i-1] > fama_values[i-1]
        prev_mama_below = mama_values[i-1] < fama_values[i-1]
        
        # 現在の時点での位置関係
        curr_mama_above = mama_values[i] > fama_values[i]
        curr_mama_below = mama_values[i] < fama_values[i]
        
        # ゴールデンクロス: MAMA が FAMA を下から上に抜ける（ロングシグナル）
        if prev_mama_below and curr_mama_above:
            signals[i] = 1
        # デッドクロス: MAMA が FAMA を上から下に抜ける（ショートシグナル）
        elif prev_mama_above and curr_mama_below:
            signals[i] = -1
    
    return signals


class MAMACrossoverSignal(BaseSignal, IEntrySignal):
    """
    MAMA/FAMAクロスオーバーによるエントリーシグナル
    
    特徴:
    - MAMA (Mother of Adaptive Moving Average) / FAMA (Following Adaptive Moving Average)
    - 市場のサイクルに応じて自動的に期間を調整する適応型移動平均線
    - Ehlers's MESA (Maximum Entropy Spectrum Analysis) アルゴリズムベース
    - トレンド強度に応じて応答速度を調整
    
    シグナル条件:
    - ゴールデンクロス (MAMA が FAMA を下から上に抜ける): ロングシグナル (1)
    - デッドクロス (MAMA が FAMA を上から下に抜ける): ショートシグナル (-1)
    - その他: シグナルなし (0)
    """
    
    def __init__(
        self,
        # MAMAパラメータ
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.05,              # 低速制限値
        src_type: str = 'hlc3',                # ソースタイプ
        ukf_params: Optional[Dict] = None      # UKFパラメータ
    ):
        """
        初期化
        
        Args:
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
                UKFソース: 'ukf', 'ukf_close', 'ukf_hlc3', 'ukf_hl2', 'ukf_ohlc4'
            ukf_params: UKFパラメータ（UKFソース使用時のオプション）
        """
        super().__init__(
            f"MAMACrossoverSignal(fast={fast_limit}, slow={slow_limit}, {src_type})"
        )
        
        # パラメータの保存
        self._params = {
            'fast_limit': fast_limit,
            'slow_limit': slow_limit,
            'src_type': src_type,
            'ukf_params': ukf_params
        }
        
        # MAMAインジケーターの初期化
        self.mama = MAMA(
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            ukf_params=ukf_params
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
                
            # MAMAの計算
            mama_result = self.mama.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if mama_result is None or len(mama_result.mama_values) == 0:
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # MAMA値とFAMA値の取得
            mama_values = mama_result.mama_values
            fama_values = mama_result.fama_values
            
            # クロスオーバーシグナルの計算（高速化版）
            signals = calculate_crossover_signals(
                mama_values,
                fama_values
            )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"MAMACrossoverSignal計算中にエラー: {str(e)}")
            # エラー時に新しいハッシュキーを生成せず、一時的なゼロシグナルを返す
            # キャッシュすると別のエラーの可能性があるため、ここではキャッシュしない
            return np.zeros(len(data), dtype=np.int8)
    
    def get_mama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        MAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: MAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.mama.get_mama_values()
    
    def get_fama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        FAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: FAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.mama.get_fama_values()
    
    def get_period_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Period値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: Period値
        """
        if data is not None:
            self.generate(data)
            
        return self.mama.get_period_values()
    
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
            
        return self.mama.get_alpha_values()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.mama.reset() if hasattr(self.mama, 'reset') else None
        self._signals_cache = {}