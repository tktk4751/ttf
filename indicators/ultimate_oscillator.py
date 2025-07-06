#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ehlers' Ultimate Oscillator Implementation
Based on John Ehlers' "Ultimate Oscillator" (Traders Tips 4/2025)
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from numba import njit
import traceback
import math

from .indicator import Indicator
from .price_source import PriceSource


@dataclass
class UltimateOscillatorResult:
    """アルティメットオシレーターの計算結果"""
    values: np.ndarray              # アルティメットオシレーター値
    signals: np.ndarray             # 信号値（差分）
    rms_values: np.ndarray          # RMS値
    highpass_short: np.ndarray      # 短期ハイパスフィルター値
    highpass_long: np.ndarray       # 長期ハイパスフィルター値


@njit(fastmath=True, cache=True)
def calculate_highpass3(
    data: np.ndarray,
    period: float
) -> np.ndarray:
    """
    3次ハイパスフィルターを計算する（Numba最適化版）
    
    Based on Ehlers' HighPass3 implementation
    
    Args:
        data: 価格配列
        period: フィルター期間
    
    Returns:
        np.ndarray: ハイパスフィルター値
    """
    length = len(data)
    hp = np.zeros(length, dtype=np.float64)
    
    if length < 3 or period <= 0:
        return hp
    
    # Ehlers' coefficients
    a1 = math.exp(-1.414 * math.pi / period)
    c2 = 2.0 * a1 * math.cos(1.414 * math.pi / 2.0 / period)
    c3 = -a1 * a1
    c1 = (1.0 + c2 - c3) / 4.0
    
    # 最初の3つの値は0に設定
    for i in range(min(3, length)):
        hp[i] = 0.0
    
    # ハイパスフィルターの計算
    for i in range(2, length):
        if i >= 2:
            hp[i] = (c1 * (data[i] - 2.0 * data[i-1] + data[i-2]) + 
                    c2 * hp[i-1] + 
                    c3 * hp[i-2])
        else:
            hp[i] = 0.0
    
    return hp


@njit(fastmath=True, cache=True)
def calculate_rms(
    data: np.ndarray,
    period: int
) -> np.ndarray:
    """
    RMS（Root Mean Square）を計算する（Numba最適化版）
    
    Args:
        data: 信号配列
        period: RMS計算期間
    
    Returns:
        np.ndarray: RMS値
    """
    length = len(data)
    rms = np.zeros(length, dtype=np.float64)
    
    if length == 0 or period <= 0:
        return rms
    
    for i in range(length):
        if i < period - 1:
            # 期間未満の場合は最初から現在までの平均
            if i == 0:
                rms[i] = abs(data[i]) if abs(data[i]) > 1e-10 else 1e-10
            else:
                sum_sq = 0.0
                for j in range(i + 1):
                    sum_sq += data[j] * data[j]
                rms[i] = math.sqrt(sum_sq / (i + 1))
                if rms[i] < 1e-10:
                    rms[i] = 1e-10
        else:
            # 期間分の平方和を計算
            sum_sq = 0.0
            for j in range(period):
                val = data[i - j]
                sum_sq += val * val
            rms[i] = math.sqrt(sum_sq / period)
            if rms[i] < 1e-10:
                rms[i] = 1e-10
    
    return rms


@njit(fastmath=True, cache=True)
def calculate_ultimate_oscillator(
    price: np.ndarray,
    edge: int = 30,
    width: int = 2,
    rms_period: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    アルティメットオシレーターを計算する（Numba最適化版）
    
    Based on Ehlers' Ultimate Oscillator implementation
    
    Args:
        price: 価格配列
        edge: エッジ期間（デフォルト: 30）
        width: 幅倍数（デフォルト: 2）
        rms_period: RMS計算期間（デフォルト: 100）
    
    Returns:
        Tuple[np.ndarray, ...]: Ultimate Oscillator値, 信号値, RMS値, 短期HP値, 長期HP値
    """
    length = len(price)
    
    if length < 10:
        # データが少ない場合は空の配列を返す
        empty_array = np.zeros(length, dtype=np.float64)
        return empty_array, empty_array, empty_array, empty_array, empty_array
    
    # 2つの異なる期間のハイパスフィルターを計算
    short_period = float(edge)
    long_period = float(width * edge)
    
    # ハイパスフィルターの計算
    hp_short = calculate_highpass3(price, short_period)
    hp_long = calculate_highpass3(price, long_period)
    
    # 差分信号の計算
    signals = hp_long - hp_short
    
    # RMS正規化
    rms_values = calculate_rms(signals, rms_period)
    
    # Ultimate Oscillatorの計算
    ultimate_osc = np.zeros(length, dtype=np.float64)
    for i in range(length):
        if rms_values[i] > 1e-10:
            ultimate_osc[i] = signals[i] / rms_values[i]
        else:
            ultimate_osc[i] = 0.0
    
    return ultimate_osc, signals, rms_values, hp_short, hp_long


class UltimateOscillator(Indicator):
    """
    Ehlers' Ultimate Oscillator インジケーター
    
    2つの異なる期間のハイパスフィルターの差分を取り、RMSで正規化したオシレーター：
    - 短期と長期のハイパスフィルターの差分による信号検出
    - RMS正規化による標準化
    - 第3次ハイパスフィルターによる高精度ノイズ除去
    
    特徴:
    - 市場サイクルの変化を高精度で検出
    - ノイズ除去と信号抽出の両立
    - 正規化による安定した振動範囲
    """
    
    def __init__(
        self,
        edge: int = 30,                        # エッジ期間
        width: int = 2,                        # 幅倍数
        rms_period: int = 100,                 # RMS計算期間
        src_type: str = 'close',               # ソースタイプ
        ukf_params: Optional[Dict] = None      # UKFパラメータ（UKFソース使用時）
    ):
        """
        コンストラクタ
        
        Args:
            edge: エッジ期間（デフォルト: 30）
                短期ハイパスフィルターの期間
            width: 幅倍数（デフォルト: 2）
                長期ハイパスフィルターの期間 = edge * width
            rms_period: RMS計算期間（デフォルト: 100）
                正規化に使用するRMSの計算期間
            src_type: ソースタイプ
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
                UKFソース: 'ukf', 'ukf_close', 'ukf_hlc3', 'ukf_hl2', 'ukf_ohlc4'
                - 'close': 終値（デフォルト）
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
            ukf_params: UKFパラメータ（UKFソース使用時のオプション）
                alpha: UKFのalpha値（デフォルト: 0.001）
                beta: UKFのbeta値（デフォルト: 2.0）
                kappa: UKFのkappa値（デフォルト: 0.0）
                process_noise_scale: プロセスノイズスケール（デフォルト: 0.001）
                volatility_window: ボラティリティ計算ウィンドウ（デフォルト: 10）
                adaptive_noise: 適応ノイズの使用（デフォルト: True）
        """
        # インジケーター名の作成
        indicator_name = f"UltimateOscillator(edge={edge}, width={width}, rms={rms_period}, {src_type})"
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.edge = edge
        self.width = width
        self.rms_period = rms_period
        self.src_type = src_type.lower()
        self.ukf_params = ukf_params
        
        # パラメータ検証
        if self.edge <= 0:
            raise ValueError("edgeは0より大きい必要があります")
        if self.width <= 0:
            raise ValueError("widthは0より大きい必要があります")
        if self.rms_period <= 0:
            raise ValueError("rms_periodは0より大きい必要があります")
        
        # ソースタイプの検証（PriceSourceから利用可能なタイプを取得）
        available_sources = PriceSource.get_available_sources()
        if self.src_type not in available_sources:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(available_sources.keys())}")
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 20
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する（超高速版）
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        # 超高速化のため最小限のサンプリング
        try:
            # データ情報の取得
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0].get('close', data.iloc[0, -1])) if length > 0 else 0.0
                last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1])) if length > 0 else 0.0
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_val = float(data[0, -1])
                        last_val = float(data[-1, -1])
                    else:
                        first_val = float(data[0])
                        last_val = float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            # 最小限のパラメータ情報
            ukf_sig = str(self.ukf_params) if self.ukf_params else "None"
            params_sig = f"{self.edge}_{self.width}_{self.rms_period}_{self.src_type}_{ukf_sig}"
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.edge}_{self.width}_{self.rms_period}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateOscillatorResult:
        """
        アルティメットオシレーターを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、選択したソースタイプに必要なカラムが必要
        
        Returns:
            UltimateOscillatorResult: アルティメットオシレーターの値と関連情報を含む結果
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ（高速化）
            data_hash = self._get_data_hash(data)
            
            # キャッシュにある場合は取得して返す
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新（最も新しく使われたキーを最後に）
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return UltimateOscillatorResult(
                    values=cached_result.values.copy(),
                    signals=cached_result.signals.copy(),
                    rms_values=cached_result.rms_values.copy(),
                    highpass_short=cached_result.highpass_short.copy(),
                    highpass_long=cached_result.highpass_long.copy()
                )
            
            # 価格ソースの計算
            price_source = PriceSource.calculate_source(data, self.src_type, self.ukf_params)
            
            # NumPy配列に変換（float64型で統一）
            price_source = np.asarray(price_source, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price_source)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            min_required = max(self.edge * self.width, self.rms_period) + 10
            if data_length < min_required:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低{min_required}点以上を推奨します。")
            
            # アルティメットオシレーターの計算（Numba最適化関数を使用）
            ultimate_values, signals, rms_values, hp_short, hp_long = calculate_ultimate_oscillator(
                price_source, self.edge, self.width, self.rms_period
            )
            
            # 結果の保存（参照問題を避けるため必要な部分だけコピー）
            result = UltimateOscillatorResult(
                values=ultimate_values.copy(),
                signals=signals.copy(),
                rms_values=rms_values.copy(),
                highpass_short=hp_short.copy(),
                highpass_long=hp_long.copy()
            )
            
            # キャッシュを更新
            # キャッシュサイズ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = ultimate_values  # 基底クラスの要件を満たすため
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"UltimateOscillator計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = UltimateOscillatorResult(
                values=np.array([]),
                signals=np.array([]),
                rms_values=np.array([]),
                highpass_short=np.array([]),
                highpass_long=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """アルティメットオシレーター値のみを取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.values.copy()
    
    def get_signals(self) -> Optional[np.ndarray]:
        """
        信号値を取得する
        
        Returns:
            np.ndarray: 信号値（ハイパスフィルターの差分）
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.signals.copy()
    
    def get_rms_values(self) -> Optional[np.ndarray]:
        """
        RMS値を取得する
        
        Returns:
            np.ndarray: RMS値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.rms_values.copy()
    
    def get_highpass_components(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        ハイパスフィルター成分を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (短期ハイパスフィルター, 長期ハイパスフィルター)
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.highpass_short.copy(), result.highpass_long.copy()
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result_cache = {}
        self._cache_keys = [] 