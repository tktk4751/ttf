#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit

from .indicator import Indicator
from .price_source import PriceSource
from .smoother.ultimate_smoother import UltimateSmoother


class UltimateERResult(NamedTuple):
    """アルティメットER計算結果"""
    values: np.ndarray              # 最終的なアルティメットER値
    raw_er: np.ndarray              # 元のER値
    smoothed_price: np.ndarray      # 平滑化された価格


@jit(nopython=True, cache=True)
def calculate_traditional_er(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    従来の効率比（Efficiency Ratio）の計算
    
    Args:
        prices: 価格配列
        period: 計算期間
        
    Returns:
        効率比配列（0-1の範囲）
    """
    n = len(prices)
    er = np.zeros(n)
    
    if n < period + 1:
        return er
    
    for i in range(period, n):
        # 価格変化の合計
        price_change = abs(prices[i] - prices[i - period])
        
        # 各期間の価格変化の絶対値の合計
        path_length = 0.0
        for j in range(1, period + 1):
            path_length += abs(prices[i - j + 1] - prices[i - j])
        
        # 効率比の計算
        if path_length > 0:
            er[i] = price_change / path_length
        else:
            er[i] = 0.0
    
    return er


class UltimateEfficiencyRatio(Indicator):
    """
    アルティメット効率比 (Ultimate Efficiency Ratio)
    
    従来の効率比（ER）をアルティメットスムーサーで平滑化したシンプルな実装
    
    🌟 **設計原理:**
    - 従来のER計算をベースに
    - アルティメットスムーサーで価格を事前平滑化
    - ノイズを除去してより安定したER値を生成
    
    ⚡ **特徴:**
    - シンプルで理解しやすい実装
    - アルティメットスムーサーによるノイズ除去
    - 高速なNumba最適化
    - 従来のERとの互換性
    """
    
    def __init__(
        self,
        period: int = 34,                          # ER計算期間
        smoother_period: float = 8.0,             # Ultimate Smoother期間
        src_type: str = 'ukf_hlc3',                    # ソースタイプ
        ukf_params: Optional[dict] = None          # UKFパラメータ
    ):
        """
        コンストラクタ
        
        Args:
            period: ER計算期間（デフォルト: 14）
            smoother_period: Ultimate Smoother期間（デフォルト: 20.0）
            src_type: ソースタイプ
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4'
                UKFソース: 'ukf', 'ukf_close', 'ukf_hlc3', 'ukf_hl2', 'ukf_ohlc4'
            ukf_params: UKFパラメータ（UKFソース使用時）
        """
        # 指標名の作成
        indicator_name = f"UltimateER(period={period}, smoother={smoother_period}, {src_type})"
        super().__init__(indicator_name)
        
        # パラメータの保存
        self.period = period
        self.smoother_period = smoother_period
        self.src_type = src_type.lower()
        self.ukf_params = ukf_params
        
        # パラメータ検証
        if self.period <= 0:
            raise ValueError("periodは0より大きい必要があります")
        if self.smoother_period <= 0:
            raise ValueError("smoother_periodは0より大きい必要があります")
        
        # ソースタイプの検証
        available_sources = PriceSource.get_available_sources()
        if self.src_type not in available_sources:
            raise ValueError(f"無効なソースタイプ: {self.src_type}。有効なオプション: {', '.join(available_sources.keys())}")
        
        # Ultimate Smootherの初期化
        self.ultimate_smoother = UltimateSmoother(
            period=self.smoother_period,
            src_type=self.src_type,
            ukf_params=self.ukf_params
        )
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
        
        # 追加の内部状態
        self._last_result = None
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュの計算"""
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
            
            # パラメータシグネチャ
            params_sig = f"{self.period}_{self.smoother_period}_{self.src_type}"
            
            # 高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.period}_{self.smoother_period}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> UltimateERResult:
        """
        アルティメット効率比を計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
            
        Returns:
            UltimateERResult: アルティメットER値と関連情報を含む結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # キャッシュヒット
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                self._last_result = cached_result
                return cached_result
            
            # 価格ソースの計算
            price_source = PriceSource.calculate_source(data, self.src_type, self.ukf_params)
            price_source = np.asarray(price_source, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price_source)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            min_required_length = max(self.period + 1, 20)
            if data_length < min_required_length:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低{min_required_length}点以上を推奨します。")
            
            # 1. Ultimate Smootherによる価格平滑化
            smoother_result = self.ultimate_smoother.calculate(data)
            smoothed_price = smoother_result.values
            
            # 2. 従来のER計算（平滑化された価格を使用）
            raw_er = calculate_traditional_er(smoothed_price, self.period)
            
            # 3. 最終的なアルティメットER値（そのまま使用）
            ultimate_er_values = raw_er.copy()
            
            # 結果の作成
            result = UltimateERResult(
                values=ultimate_er_values,
                raw_er=raw_er,
                smoothed_price=smoothed_price
            )
            
            # キャッシュ更新
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = ultimate_er_values  # 基底クラスの要件
            self._last_result = result
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"UltimateER計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            empty_result = UltimateERResult(
                values=np.array([]),
                raw_er=np.array([]),
                smoothed_price=np.array([])
            )
            return empty_result
    
    def get_current_er_value(self) -> float:
        """現在のアルティメットER値を取得"""
        if self._last_result is None or len(self._last_result.values) == 0:
            return 0.0
        return float(self._last_result.values[-1])
    
    def get_raw_er_value(self) -> float:
        """現在の元のER値を取得"""
        if self._last_result is None or len(self._last_result.raw_er) == 0:
            return 0.0
        return float(self._last_result.raw_er[-1])
    
    def get_smoothing_effect(self) -> dict:
        """平滑化効果の統計を取得"""
        if self._last_result is None or len(self._last_result.values) == 0:
            return {}
        
        # 元の価格と平滑化価格の比較
        original_price = PriceSource.calculate_source(self._last_data, self.src_type, self.ukf_params)
        smoothed_price = self._last_result.smoothed_price
        
        if len(original_price) != len(smoothed_price):
            return {}
        
        # ボラティリティの比較
        original_vol = np.nanstd(original_price)
        smoothed_vol = np.nanstd(smoothed_price)
        
        noise_reduction = (original_vol - smoothed_vol) / original_vol if original_vol > 0 else 0.0
        
        return {
            'original_volatility': float(original_vol),
            'smoothed_volatility': float(smoothed_vol),
            'noise_reduction_ratio': float(noise_reduction),
            'noise_reduction_percentage': float(noise_reduction * 100)
        }
    
    def reset(self) -> None:
        """インディケーターの状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        self._last_result = None
        if hasattr(self, 'ultimate_smoother'):
            self.ultimate_smoother.reset() 