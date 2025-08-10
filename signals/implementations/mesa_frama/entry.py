#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Any, Optional
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base_signal import BaseSignal
from ...interfaces.entry import IEntrySignal
from indicators.mesa_frama import MESA_FRAMA


@njit(fastmath=True, parallel=True)
def calculate_dual_mesa_frama_crossover_signals(
    fast_mesa_frama: np.ndarray, 
    slow_mesa_frama: np.ndarray,
    threshold: float = 0.0
) -> np.ndarray:
    """
    2本のMESA_FRAMAのクロスオーバーシグナルを計算する（高速化版）
    
    Args:
        fast_mesa_frama: 短期MESA_FRAMA値の配列
        slow_mesa_frama: 長期MESA_FRAMA値の配列
        threshold: 閾値（デフォルト: 0.0）
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(fast_mesa_frama)
    signals = np.zeros(length, dtype=np.int8)
    
    # クロスオーバーの判定（並列処理化）
    for i in prange(1, length):
        # MESA_FRAMA値が有効かチェック
        if (np.isnan(fast_mesa_frama[i]) or np.isnan(slow_mesa_frama[i]) or
            np.isnan(fast_mesa_frama[i-1]) or np.isnan(slow_mesa_frama[i-1])):
            signals[i] = 0
            continue
            
        # 短期線が長期線を上抜け: ロングシグナル
        if (fast_mesa_frama[i-1] <= slow_mesa_frama[i-1] + threshold and 
            fast_mesa_frama[i] > slow_mesa_frama[i] + threshold):
            signals[i] = 1
        # 短期線が長期線を下抜け: ショートシグナル
        elif (fast_mesa_frama[i-1] >= slow_mesa_frama[i-1] - threshold and 
              fast_mesa_frama[i] < slow_mesa_frama[i] - threshold):
            signals[i] = -1
    
    return signals


@njit(fastmath=True, parallel=True) 
def calculate_dual_mesa_frama_position_signals(
    fast_mesa_frama: np.ndarray,
    slow_mesa_frama: np.ndarray
) -> np.ndarray:
    """
    2本のMESA_FRAMAの位置関係シグナルを計算する（高速化版）
    
    Args:
        fast_mesa_frama: 短期MESA_FRAMA値の配列
        slow_mesa_frama: 長期MESA_FRAMA値の配列
    
    Returns:
        シグナルの配列（1: ロング, -1: ショート, 0: シグナルなし）
    """
    length = len(fast_mesa_frama)
    signals = np.zeros(length, dtype=np.int8)
    
    # 位置関係の判定（並列処理化）
    for i in prange(length):
        # MESA_FRAMA値が有効かチェック
        if np.isnan(fast_mesa_frama[i]) or np.isnan(slow_mesa_frama[i]):
            signals[i] = 0
            continue
            
        # 短期線 > 長期線: ロングシグナル
        if fast_mesa_frama[i] > slow_mesa_frama[i]:
            signals[i] = 1
        # 短期線 < 長期線: ショートシグナル
        elif fast_mesa_frama[i] < slow_mesa_frama[i]:
            signals[i] = -1
    
    return signals


class MESAFRAMACrossoverEntrySignal(BaseSignal, IEntrySignal):
    """
    MESA_FRAMA デュアルクロスオーバーによるエントリーシグナル
    
    特徴:
    - 2本のMESA_FRAMA (短期線・長期線のクロスオーバー戦略)
    - MESA適応期間を使用したFractal Adaptive Moving Average
    - MAMAの期間決定アルゴリズムとFRAMAのフラクタル適応性を組み合わせ
    - 市場サイクルに応じた動的期間調整
    - フラクタル次元ベースの応答性制御
    - カルマンフィルターとゼロラグ処理の統合（オプション）
    
    シグナル条件:
    1. クロスオーバーモード:
       - 短期MESA_FRAMAが長期MESA_FRAMAを上抜け: ロングシグナル (1)
       - 短期MESA_FRAMAが長期MESA_FRAMAを下抜け: ショートシグナル (-1)
    2. 位置関係モード:
       - 短期MESA_FRAMA > 長期MESA_FRAMA: ロングシグナル (1)
       - 短期MESA_FRAMA < 長期MESA_FRAMA: ショートシグナル (-1)
    """
    
    def __init__(
        self,
        # 短期MESA_FRAMAパラメータ
        fast_base_period: int = 8,               # 短期基本期間
        fast_src_type: str = 'hl2',              # 短期ソースタイプ
        fast_fc: int = 1,                        # 短期Fast Constant
        fast_sc: int = 198,                      # 短期Slow Constant
        fast_mesa_fast_limit: float = 0.5,      # 短期MESA高速制限値
        fast_mesa_slow_limit: float = 0.05,      # 短期MESA低速制限値
        # 長期MESA_FRAMAパラメータ
        slow_base_period: int = 32,              # 長期基本期間
        slow_src_type: str = 'hl2',              # 長期ソースタイプ
        slow_fc: int = 1,                        # 長期Fast Constant
        slow_sc: int = 198,                      # 長期Slow Constant
        slow_mesa_fast_limit: float = 0.3,      # 長期MESA高速制限値
        slow_mesa_slow_limit: float = 0.02,     # 長期MESA低速制限値
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,         # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented',   # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,      # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True,               # ゼロラグ処理を使用するか
        # シグナルパラメータ
        signal_mode: str = 'crossover',          # シグナルモード ('crossover' または 'position')
        crossover_threshold: float = 0.0        # クロスオーバー閾値
    ):
        """
        初期化
        
        Args:
            fast_base_period: 短期基本期間（偶数である必要がある、デフォルト: 8）
            fast_src_type: 短期ソースタイプ（デフォルト: 'hl2'）
            fast_fc: 短期Fast Constant（デフォルト: 1）
            fast_sc: 短期Slow Constant（デフォルト: 198）
            fast_mesa_fast_limit: 短期MESA高速制限値（デフォルト: 0.7）
            fast_mesa_slow_limit: 短期MESA低速制限値（デフォルト: 0.1）
            slow_base_period: 長期基本期間（偶数である必要がある、デフォルト: 32）
            slow_src_type: 長期ソースタイプ（デフォルト: 'hl2'）
            slow_fc: 長期Fast Constant（デフォルト: 1）
            slow_sc: 長期Slow Constant（デフォルト: 198）
            slow_mesa_fast_limit: 長期MESA高速制限値（デフォルト: 0.3）
            slow_mesa_slow_limit: 長期MESA低速制限値（デフォルト: 0.02）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
            signal_mode: シグナルモード（デフォルト: 'crossover'）
            crossover_threshold: クロスオーバー閾値（デフォルト: 0.0）
        """
        super().__init__(
            f"MESAFRAMADualCrossoverEntrySignal(fast={fast_base_period}, slow={slow_base_period}, mode={signal_mode})"
        )
        
        # パラメータの保存
        self._params = {
            'fast_base_period': fast_base_period,
            'fast_src_type': fast_src_type,
            'fast_fc': fast_fc,
            'fast_sc': fast_sc,
            'fast_mesa_fast_limit': fast_mesa_fast_limit,
            'fast_mesa_slow_limit': fast_mesa_slow_limit,
            'slow_base_period': slow_base_period,
            'slow_src_type': slow_src_type,
            'slow_fc': slow_fc,
            'slow_sc': slow_sc,
            'slow_mesa_fast_limit': slow_mesa_fast_limit,
            'slow_mesa_slow_limit': slow_mesa_slow_limit,
            'use_kalman_filter': use_kalman_filter,
            'kalman_filter_type': kalman_filter_type,
            'kalman_process_noise': kalman_process_noise,
            'kalman_observation_noise': kalman_observation_noise,
            'use_zero_lag': use_zero_lag,
            'signal_mode': signal_mode,
            'crossover_threshold': crossover_threshold
        }
        
        # パラメータ検証
        if signal_mode not in ['crossover', 'position']:
            raise ValueError(f"無効なシグナルモード: {signal_mode}。有効な値: 'crossover', 'position'")
        if fast_base_period % 2 != 0 or slow_base_period % 2 != 0:
            raise ValueError("基本期間は偶数である必要があります")
        if fast_base_period >= slow_base_period:
            raise ValueError("短期期間は長期期間より小さくなければなりません")
        
        # 短期MESA_FRAMAインジケーターの初期化
        self.fast_mesa_frama = MESA_FRAMA(
            base_period=fast_base_period,
            src_type=fast_src_type,
            fc=fast_fc,
            sc=fast_sc,
            mesa_fast_limit=fast_mesa_fast_limit,
            mesa_slow_limit=fast_mesa_slow_limit,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag
        )
        
        # 長期MESA_FRAMAインジケーターの初期化
        self.slow_mesa_frama = MESA_FRAMA(
            base_period=slow_base_period,
            src_type=slow_src_type,
            fc=slow_fc,
            sc=slow_sc,
            mesa_fast_limit=slow_mesa_fast_limit,
            mesa_slow_limit=slow_mesa_slow_limit,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag
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
            if all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close']):
                ohlcv_array = ohlcv_data[['open', 'high', 'low', 'close']].values
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
        デュアルMESA_FRAMAクロスオーバーシグナルを生成する
        
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
                
            # 短期MESA_FRAMAの計算
            fast_mesa_frama_result = self.fast_mesa_frama.calculate(data)
            
            # 長期MESA_FRAMAの計算
            slow_mesa_frama_result = self.slow_mesa_frama.calculate(data)
            
            # 計算が失敗した場合はゼロシグナルを返す
            if (fast_mesa_frama_result is None or len(fast_mesa_frama_result.values) == 0 or
                slow_mesa_frama_result is None or len(slow_mesa_frama_result.values) == 0):
                self._signals_cache[data_hash] = np.zeros(len(data), dtype=np.int8)
                return self._signals_cache[data_hash]
            
            # MESA_FRAMA値の取得
            fast_mesa_frama_values = fast_mesa_frama_result.values
            slow_mesa_frama_values = slow_mesa_frama_result.values
            
            # シグナルの計算（高速化版）
            if self._params['signal_mode'] == 'crossover':
                # デュアルMESA_FRAMAクロスオーバーシグナル
                signals = calculate_dual_mesa_frama_crossover_signals(
                    fast_mesa_frama_values,
                    slow_mesa_frama_values,
                    self._params['crossover_threshold']
                )
            else:
                # デュアルMESA_FRAMA位置関係シグナル
                signals = calculate_dual_mesa_frama_position_signals(
                    fast_mesa_frama_values,
                    slow_mesa_frama_values
                )
            
            # 結果をキャッシュ
            self._signals_cache[data_hash] = signals
            return signals
            
        except Exception as e:
            # エラーが発生した場合は警告を出力し、ゼロシグナルを返す
            print(f"MESAFRAMADualCrossoverEntrySignal計算中にエラー: {str(e)}")
            return np.zeros(len(data), dtype=np.int8)
    
    def get_fast_mesa_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        短期MESA_FRAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 短期MESA_FRAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.fast_mesa_frama.get_values()
        
    def get_slow_mesa_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        長期MESA_FRAMA値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 長期MESA_FRAMA値
        """
        if data is not None:
            self.generate(data)
            
        return self.slow_mesa_frama.get_values()
        
    def get_mesa_frama_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        両方のMESA_FRAMA値を取得する（後方互換性のため）
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (短期MESA_FRAMA値, 長期MESA_FRAMA値)
        """
        if data is not None:
            self.generate(data)
            
        return self.fast_mesa_frama.get_values(), self.slow_mesa_frama.get_values()
    
    def get_fractal_dimension(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        フラクタル次元を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (短期フラクタル次元, 長期フラクタル次元)
        """
        if data is not None:
            self.generate(data)
            
        return self.fast_mesa_frama.get_fractal_dimension(), self.slow_mesa_frama.get_fractal_dimension()
    
    def get_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        MESA動的期間を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (短期MESA動的期間, 長期MESA動的期間)
        """
        if data is not None:
            self.generate(data)
            
        return self.fast_mesa_frama.get_dynamic_periods(), self.slow_mesa_frama.get_dynamic_periods()
    
    def get_mesa_phase(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        MESA位相を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (短期MESA位相, 長期MESA位相)
        """
        if data is not None:
            self.generate(data)
            
        return self.fast_mesa_frama.get_mesa_phase(), self.slow_mesa_frama.get_mesa_phase()
    
    def get_alpha_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        フラクタルアルファ値を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (短期フラクタルアルファ値, 長期フラクタルアルファ値)
        """
        if data is not None:
            self.generate(data)
            
        return self.fast_mesa_frama.get_alpha(), self.slow_mesa_frama.get_alpha()
    
    def get_filtered_price(self, data: Union[pd.DataFrame, np.ndarray] = None) -> tuple:
        """
        カルマンフィルター後の価格を取得する
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            tuple: (短期フィルタリングされた価格, 長期フィルタリングされた価格)
        """
        if data is not None:
            self.generate(data)
            
        return self.fast_mesa_frama.get_filtered_price(), self.slow_mesa_frama.get_filtered_price()
        
    def reset(self) -> None:
        """
        シグナルの状態をリセットする
        """
        super().reset()
        self.fast_mesa_frama.reset() if hasattr(self.fast_mesa_frama, 'reset') else None
        self.slow_mesa_frama.reset() if hasattr(self.slow_mesa_frama, 'reset') else None
        self._signals_cache = {}