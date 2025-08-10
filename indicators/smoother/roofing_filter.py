#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
import pandas as pd
from numba import njit
import traceback

from ..indicator import Indicator
from ..price_source import PriceSource


@dataclass
class RoofingFilterResult:
    """Roofingフィルターの計算結果"""
    values: np.ndarray          # フィルタ済み値
    highpass: np.ndarray        # HighPassフィルター値
    supersmoother: np.ndarray   # SuperSmootherフィルター値


@njit(fastmath=True, cache=True)
def calculate_supersmoother_core(data: np.ndarray, band_edge: float) -> np.ndarray:
    """
    SuperSmootherフィルターを計算する
    
    Args:
        data: 入力データ
        band_edge: バンドエッジ（周期）
    
    Returns:
        np.ndarray: SuperSmootherフィルター値
    """
    length = len(data)
    filt = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        filt[i] = np.nan
    
    # フィルター係数の計算
    a1 = np.exp(-1.414 * np.pi / band_edge)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / band_edge)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    # 初期値設定
    for i in range(min(2, length)):
        if not np.isnan(data[i]):
            filt[i] = data[i]
    
    # SuperSmootherフィルター計算
    for i in range(2, length):
        if np.isnan(data[i]):
            filt[i] = filt[i-1] if i > 0 else np.nan
            continue
            
        filt[i] = c1 * (data[i] + data[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]
    
    return filt


@njit(fastmath=True, cache=True)
def calculate_highpass_core(data: np.ndarray, cutoff_period: float) -> np.ndarray:
    """
    2極HighPassフィルターを計算する
    
    Args:
        data: 入力データ
        cutoff_period: カットオフ周期
    
    Returns:
        np.ndarray: HighPassフィルター値
    """
    length = len(data)
    hp = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        hp[i] = np.nan
    
    # フィルター係数の計算
    alpha1 = (np.cos(0.707 * 2 * np.pi / cutoff_period) + np.sin(0.707 * 2 * np.pi / cutoff_period) - 1) / np.cos(0.707 * 2 * np.pi / cutoff_period)
    
    # 初期値設定
    for i in range(min(2, length)):
        if not np.isnan(data[i]):
            hp[i] = 0.0
    
    # HighPassフィルター計算
    for i in range(2, length):
        if np.isnan(data[i]):
            hp[i] = hp[i-1] if i > 0 else 0.0
            continue
            
        hp[i] = ((1 - alpha1 / 2) * (1 - alpha1 / 2) * (data[i] - 2 * data[i-1] + data[i-2]) + 
                 2 * (1 - alpha1) * hp[i-1] - 
                 (1 - alpha1) * (1 - alpha1) * hp[i-2])
    
    return hp


@njit(fastmath=True, cache=True)
def calculate_roofing_filter_core(data: np.ndarray, hp_cutoff: float, ss_band_edge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Roofingフィルターを計算する（HighPass + SuperSmoother）
    
    Args:
        data: 入力データ
        hp_cutoff: HighPassフィルターのカットオフ周期
        ss_band_edge: SuperSmootherフィルターのバンドエッジ
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Roofingフィルター値、HighPass値、SuperSmoother値
    """
    length = len(data)
    
    # HighPassフィルター
    hp = calculate_highpass_core(data, hp_cutoff)
    
    # SuperSmootherフィルター（HighPassの結果に適用）
    filt = calculate_supersmoother_core(hp, ss_band_edge)
    
    return filt, hp, filt


class RoofingFilter(Indicator):
    """
    Roofing Filter - John Ehlers
    
    "Predictive Indicators for Effective Trading Strategies"より
    
    Roofingフィルターは以下の組み合わせ：
    1. 2極HighPassフィルター：長期トレンドを除去（Spectral Dilationの除去）
    2. SuperSmootherフィルター：高周波ノイズを除去（Aliasing Noiseの除去）
    
    特徴：
    - 10-48バーの周期成分のみを通す帯域フィルター
    - より正確な振動の検出が可能
    - 従来のオシレーターのSpectral Dilation歪みを除去
    """
    
    def __init__(
        self,
        src_type: str = 'close',         # ソースタイプ
        hp_cutoff: float = 48.0,         # HighPassカットオフ周期
        ss_band_edge: float = 10.0       # SuperSmootherバンドエッジ周期
    ):
        """
        コンストラクタ
        
        Args:
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            hp_cutoff: HighPassフィルターのカットオフ周期（デフォルト: 48）
            ss_band_edge: SuperSmootherフィルターのバンドエッジ周期（デフォルト: 10）
        """
        indicator_name = f"RoofingFilter(src={src_type}, hp={hp_cutoff}, ss={ss_band_edge})"
        super().__init__(indicator_name)
        
        # パラメータの検証
        if hp_cutoff <= 0:
            raise ValueError("HighPassカットオフ周期は正の値である必要があります")
        if ss_band_edge <= 0:
            raise ValueError("SuperSmootherバンドエッジ周期は正の値である必要があります")
        if ss_band_edge >= hp_cutoff:
            raise ValueError("SuperSmootherバンドエッジはHighPassカットオフより小さい必要があります")
        
        # パラメータを保存
        self.src_type = src_type.lower()
        self.hp_cutoff = hp_cutoff
        self.ss_band_edge = ss_band_edge
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
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
            
            # パラメータ情報
            params_sig = f"{self.src_type}_{self.hp_cutoff}_{self.ss_band_edge}"
            
            # ハッシュ計算
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.src_type}_{self.hp_cutoff}_{self.ss_band_edge}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> RoofingFilterResult:
        """
        Roofingフィルターを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            RoofingFilterResult: フィルター済み値、HighPass値、SuperSmoother値を含む結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return RoofingFilterResult(
                    values=cached_result.values.copy(),
                    highpass=cached_result.highpass.copy(),
                    supersmoother=cached_result.supersmoother.copy()
                )
            
            # PriceSourceを使って価格データを取得
            price = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換（float64型で統一）
            price = np.asarray(price, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            min_required = max(int(self.hp_cutoff), int(self.ss_band_edge)) + 5
            if data_length < min_required:
                self.logger.warning(f"データ長({data_length})が推奨最小長({min_required})より短いです")
            
            # Roofingフィルターの計算
            roofing_values, hp_values, ss_values = calculate_roofing_filter_core(
                price, self.hp_cutoff, self.ss_band_edge
            )
            
            # 結果の保存
            result = RoofingFilterResult(
                values=roofing_values.copy(),
                highpass=hp_values.copy(),
                supersmoother=ss_values.copy()
            )
            
            # キャッシュを更新
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = roofing_values  # 基底クラスの要件を満たすため
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Roofingフィルター計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = RoofingFilterResult(
                values=np.array([]),
                highpass=np.array([]),
                supersmoother=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """Roofingフィルター値のみを取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.values.copy()
    
    def get_highpass(self) -> Optional[np.ndarray]:
        """
        HighPassフィルター値を取得する
        
        Returns:
            np.ndarray: HighPassフィルター値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.highpass.copy()
    
    def get_supersmoother(self) -> Optional[np.ndarray]:
        """
        SuperSmootherフィルター値を取得する
        
        Returns:
            np.ndarray: SuperSmootherフィルター値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.supersmoother.copy()
    
    def filter(self, data: Union[pd.DataFrame, np.ndarray]) -> Optional[np.ndarray]:
        """
        Roofingフィルターを適用する（HyperADX等からの互換性用メソッド）
        
        Args:
            data: 価格データ（NumPy配列またはDataFrame）
            
        Returns:
            フィルタリング済みの値
        """
        try:
            if isinstance(data, np.ndarray):
                # NumPy配列の場合、DataFrameに変換
                if data.ndim == 1:
                    # 1次元配列の場合、closeとして使用
                    df = pd.DataFrame({'close': data})
                else:
                    # 2次元配列の場合、OHLC形式と仮定
                    df = pd.DataFrame({
                        'open': data[:, 0] if data.shape[1] > 0 else data[:, -1],
                        'high': data[:, 1] if data.shape[1] > 1 else data[:, -1],
                        'low': data[:, 2] if data.shape[1] > 2 else data[:, -1],
                        'close': data[:, 3] if data.shape[1] > 3 else data[:, -1]
                    })
                result = self.calculate(df)
            else:
                result = self.calculate(data)
            
            return result.values if result else None
            
        except Exception as e:
            self.logger.warning(f"Roofingフィルターの適用に失敗しました: {e}")
            return None
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result_cache = {}
        self._cache_keys = []


class SuperSmoother(Indicator):
    """
    SuperSmoother Filter - John Ehlers
    
    エイリアシングノイズを除去するための高性能フィルター
    従来のEMAよりも優れたノイズ除去性能とより少ないラグを持つ
    """
    
    def __init__(
        self,
        band_edge: float = 10.0,         # バンドエッジ周期
        src_type: str = 'close'          # ソースタイプ
    ):
        """
        コンストラクタ
        
        Args:
            band_edge: バンドエッジ周期（デフォルト: 10）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
        """
        indicator_name = f"SuperSmoother(band_edge={band_edge}, src={src_type})"
        super().__init__(indicator_name)
        
        # パラメータの検証
        if band_edge <= 0:
            raise ValueError("バンドエッジ周期は正の値である必要があります")
        
        # パラメータを保存
        self.band_edge = band_edge
        self.src_type = src_type.lower()
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        SuperSmootherフィルターを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            np.ndarray: SuperSmootherフィルター値
        """
        try:
            # PriceSourceを使って価格データを取得
            price = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換（float64型で統一）
            price = np.asarray(price, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < int(self.band_edge):
                self.logger.warning(f"データ長({data_length})がバンドエッジ周期({self.band_edge})より短いです")
            
            # SuperSmootherフィルターの計算
            result = calculate_supersmoother_core(price, self.band_edge)
            
            self._values = result  # 基底クラスの要件を満たすため
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"SuperSmoother計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            return np.array([])


class HighPassFilter(Indicator):
    """
    2極HighPass Filter - John Ehlers
    
    Spectral Dilationを除去するための2極HighPassフィルター
    従来の1極フィルターより効果的に長期成分を除去
    """
    
    def __init__(
        self,
        cutoff_period: float = 48.0,     # カットオフ周期
        src_type: str = 'close'          # ソースタイプ
    ):
        """
        コンストラクタ
        
        Args:
            cutoff_period: カットオフ周期（デフォルト: 48）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
        """
        indicator_name = f"HighPassFilter(cutoff={cutoff_period}, src={src_type})"
        super().__init__(indicator_name)
        
        # パラメータの検証
        if cutoff_period <= 0:
            raise ValueError("カットオフ周期は正の値である必要があります")
        
        # パラメータを保存
        self.cutoff_period = cutoff_period
        self.src_type = src_type.lower()
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        HighPassフィルターを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            np.ndarray: HighPassフィルター値
        """
        try:
            # PriceSourceを使って価格データを取得
            price = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換（float64型で統一）
            price = np.asarray(price, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < int(self.cutoff_period):
                self.logger.warning(f"データ長({data_length})がカットオフ周期({self.cutoff_period})より短いです")
            
            # HighPassフィルターの計算
            result = calculate_highpass_core(price, self.cutoff_period)
            
            self._values = result  # 基底クラスの要件を満たすため
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"HighPass計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            return np.array([])