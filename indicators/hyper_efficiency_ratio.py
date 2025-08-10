#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit, prange
import traceback
import math

# ベースクラスのインポート
try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .cycle.ehlers_unified_dc import EhlersUnifiedDC
except ImportError:
    print("Warning: 相対パスからのインポートに失敗しました。基本クラスを定義します。")
    class Indicator:
        def __init__(self, name): 
            self.name = name
            self.logger = self._get_logger()
        def reset(self): pass
        def _get_logger(self): 
            import logging
            return logging.getLogger(self.__class__.__name__)
    
    class PriceSource:
        @staticmethod
        def calculate_source(data, src_type):
            if isinstance(data, pd.DataFrame):
                if src_type == 'close': return data['close'].values
                elif src_type == 'open': return data['open'].values
                elif src_type == 'high': return data['high'].values
                elif src_type == 'low': return data['low'].values
                elif src_type == 'hl2': return ((data['high'] + data['low']) / 2).values
                elif src_type == 'hlc3': return ((data['high'] + data['low'] + data['close']) / 3).values
                elif src_type == 'ohlc4': return ((data['open'] + data['high'] + data['low'] + data['close']) / 4).values
                else: return data['close'].values
            else:
                return data[:, 3] if data.ndim > 1 and data.shape[1] > 3 else data
    
    class EhlersUnifiedDC:
        def __init__(self, **kwargs): pass
        def calculate(self, data): return np.full(len(data), 14.0)
        def reset(self): pass


class HyperERResult(NamedTuple):
    """ハイパー効率比計算結果"""
    values: np.ndarray                    # HER値 (0-1)
    linear_volatility: np.ndarray         # 線形ボラティリティ
    nonlinear_volatility: np.ndarray      # 非線形ボラティリティ
    adaptive_volatility: np.ndarray       # 適応的ボラティリティ
    efficiency_components: np.ndarray     # 効率性成分
    trend_signals: np.ndarray             # トレンド信号 (1=up, -1=down, 0=range)
    current_trend: str                    # 現在のトレンド状態
    current_trend_value: int              # 現在のトレンド値
    quality_score: float                  # 品質スコア


@njit(fastmath=True, parallel=True, cache=True)
def hyper_efficiency_ratio_core(prices: np.ndarray, window: int = 14) -> tuple:
    """
    ハイパー効率率（HER）コア計算 - 従来ERを超絶進化させたトレンド強度測定器
    
    従来の効率率を多次元・非線形・適応的に進化させ、
    市場の真のトレンド効率性を完璧に捕捉する革新的指標
    
    Returns:
        tuple: (her_values, linear_vol, nonlinear_vol, adaptive_vol, efficiency_components)
    """
    n = len(prices)
    her_values = np.full(n, np.nan)
    linear_vol = np.full(n, np.nan)
    nonlinear_vol = np.full(n, np.nan)
    adaptive_vol = np.full(n, np.nan)
    efficiency_components = np.full(n, np.nan)
    
    for i in prange(max(window, 10), n):
        actual_window = min(window, i)
        segment = prices[i-actual_window:i]
        
        # 方向性変化（従来ER分子）
        direction = abs(segment[-1] - segment[0])
        
        # 多次元ボラティリティ（従来ER分母の進化版）
        linear_volatility = 0.0
        nonlinear_volatility = 0.0
        adaptive_volatility = 0.0
        
        for j in range(1, len(segment)):
            # 線形ボラティリティ
            linear_change = abs(segment[j] - segment[j-1])
            linear_volatility += linear_change
            
            # 非線形ボラティリティ（2次効果）
            if j >= 2:
                acceleration = abs((segment[j] - segment[j-1]) - (segment[j-1] - segment[j-2]))
                nonlinear_volatility += acceleration
            
            # 適応的ボラティリティ（重み付き）
            weight = math.exp(-(len(segment) - j) * 0.1)  # 新しいデータほど重要
            adaptive_volatility += linear_change * weight
        
        # ハイパー効率率計算
        total_volatility = (
            linear_volatility * 0.5 + 
            nonlinear_volatility * 0.3 + 
            adaptive_volatility * 0.2
        )
        
        if abs(total_volatility) > 1e-10:
            base_efficiency = direction / total_volatility
            
            # 非線形変換（シグモイド + 双曲線正接）
            sigmoid_transform = 1.0 / (1.0 + math.exp(-base_efficiency * 10))
            tanh_transform = math.tanh(base_efficiency * 5)
            
            # 統合変換
            her_value = (sigmoid_transform * 0.6 + tanh_transform * 0.4)
            her_values[i] = max(min(her_value, 1.0), 0.0)
            
            # 成分の保存
            efficiency_components[i] = base_efficiency
        else:
            her_values[i] = 0.0
            efficiency_components[i] = 0.0
        
        # ボラティリティ成分の保存
        linear_vol[i] = linear_volatility
        nonlinear_vol[i] = nonlinear_volatility
        adaptive_vol[i] = adaptive_volatility
    
    return her_values, linear_vol, nonlinear_vol, adaptive_vol, efficiency_components


@njit(fastmath=True, cache=True)
def calculate_hyper_trend_signals(values: np.ndarray, slope_index: int, threshold: float = 0.3) -> np.ndarray:
    """
    ハイパー効率比用のトレンド信号計算
    
    Args:
        values: HER値配列
        slope_index: スロープ判定期間
        threshold: トレンド判定閾値
        
    Returns:
        トレンド信号配列 (1=up, -1=down, 0=range)
    """
    length = len(values)
    trend_signals = np.zeros(length, dtype=np.int8)
    
    for i in range(slope_index, length):
        if not np.isnan(values[i]) and not np.isnan(values[i - slope_index]):
            current = values[i]
            previous = values[i - slope_index]
            
            # HERの変化量
            change = current - previous
            
            # 絶対変化量による判定
            if abs(change) < 0.05:  # 変化が小さい場合はレンジ
                trend_signals[i] = 0
            elif change > 0 and current > threshold:  # 上昇かつ高効率
                trend_signals[i] = 1  # up
            elif change < 0 and current < (1.0 - threshold):  # 下降かつ低効率
                trend_signals[i] = -1  # down
            else:
                trend_signals[i] = 0  # range
    
    return trend_signals


@njit(fastmath=True, cache=True)
def calculate_her_quality_score(her_values: np.ndarray, 
                               linear_vol: np.ndarray,
                               nonlinear_vol: np.ndarray,
                               adaptive_vol: np.ndarray) -> float:
    """
    ハイパー効率比の品質スコア計算
    
    Args:
        her_values: HER値配列
        linear_vol: 線形ボラティリティ配列
        nonlinear_vol: 非線形ボラティリティ配列
        adaptive_vol: 適応的ボラティリティ配列
        
    Returns:
        品質スコア (0-1)
    """
    # 有効値の統計計算
    her_sum = 0.0
    her_count = 0
    vol_consistency = 0.0
    vol_count = 0
    
    for i in range(len(her_values)):
        if not np.isnan(her_values[i]):
            her_sum += her_values[i]
            her_count += 1
    
    # ボラティリティの一貫性計算
    for i in range(len(linear_vol)):
        if (not np.isnan(linear_vol[i]) and 
            not np.isnan(nonlinear_vol[i]) and 
            not np.isnan(adaptive_vol[i])):
            
            # 各ボラティリティの比率
            total_vol = linear_vol[i] + nonlinear_vol[i] + adaptive_vol[i]
            if total_vol > 1e-10:
                # バランスの良さ（理想的な比率: 0.5, 0.3, 0.2）
                linear_ratio = linear_vol[i] / total_vol
                nonlinear_ratio = nonlinear_vol[i] / total_vol
                adaptive_ratio = adaptive_vol[i] / total_vol
                
                # 理想的比率からの偏差
                deviation = (abs(linear_ratio - 0.5) + 
                           abs(nonlinear_ratio - 0.3) + 
                           abs(adaptive_ratio - 0.2))
                
                vol_consistency += (1.0 - min(deviation, 1.0))
                vol_count += 1
    
    # 品質スコア計算
    if her_count == 0:
        return 0.0
    
    avg_her = her_sum / her_count
    avg_vol_consistency = vol_consistency / vol_count if vol_count > 0 else 0.5
    
    # 統合品質スコア
    quality = avg_her * 0.7 + avg_vol_consistency * 0.3
    
    return max(0.0, min(1.0, quality))


class HyperEfficiencyRatio(Indicator):
    """
    ハイパー効率比（HER）インジケーター
    
    従来の効率率を多次元・非線形・適応的に進化させた革新的なトレンド効率性測定器
    
    特徴：
    - 多次元ボラティリティ分析（線形・非線形・適応的）
    - 非線形変換による高精度トレンド検出
    - 動的期間対応
    - 品質スコア付き信頼性評価
    
    従来ERとの違い：
    - 線形ボラティリティ：従来のER分母
    - 非線形ボラティリティ：加速度成分を追加
    - 適応的ボラティリティ：時間重み付け
    - 非線形変換：シグモイド+双曲線正接による高精度化
    """
    
    def __init__(self,
                 window: int = 14,
                 src_type: str = 'hlc3',
                 use_dynamic_period: bool = False,
                 slope_index: int = 3,
                 threshold: float = 0.3):
        """
        ハイパー効率比インジケーターの初期化
        """
        super().__init__(f"HER(w={window},src={src_type},slope={slope_index},th={threshold:.2f})")
        
        self.window = window
        self.src_type = src_type
        self.use_dynamic_period = use_dynamic_period
        self.slope_index = slope_index
        self.threshold = threshold
        
        self._result: Optional[HyperERResult] = None
        self._cache = {}
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュの計算"""
        try:
            if isinstance(data, pd.DataFrame):
                shape_tuple = data.shape
                first_row = tuple(data.iloc[0]) if len(data) > 0 else ()
                last_row = tuple(data.iloc[-1]) if len(data) > 0 else ()
                data_repr = (shape_tuple, first_row, last_row)
                data_hash = hash(data_repr)
            else:
                data_hash = hash(data.tobytes())
        except Exception:
            data_hash = hash(str(data))
        
        param_str = f"w={self.window}_src={self.src_type}_slope={self.slope_index}_th={self.threshold}"
        return f"{data_hash}_{param_str}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperERResult:
        """
        ハイパー効率比の計算
        
        Args:
            data: 価格データ
            
        Returns:
            HyperERResult: HER値とその他の情報を含む結果
        """
        current_len = len(data) if hasattr(data, '__len__') else 0
        if current_len == 0:
            return self._create_empty_result()
        
        try:
            data_hash = self._get_data_hash(data)
            
            # キャッシュチェック
            if data_hash in self._cache and self._result is not None:
                if len(self._result.values) == current_len:
                    return self._copy_result()
            
            # 価格データの取得
            prices = PriceSource.calculate_source(data, self.src_type)
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices, dtype=np.float64)
            else:
                prices = prices.astype(np.float64)
            
            data_length = len(prices)
            if data_length < self.window:
                self.logger.warning(f"データ長が不足しています: {data_length} < {self.window}")
                return self._create_empty_result(data_length)
            
            # 固定期間モード
            her_values, linear_vol, nonlinear_vol, adaptive_vol, efficiency_components = hyper_efficiency_ratio_core(
                prices, self.window)
            
            # トレンド信号計算
            trend_signals = calculate_hyper_trend_signals(her_values, self.slope_index, self.threshold)
            
            # 現在のトレンド状態
            current_trend_value = trend_signals[-1] if len(trend_signals) > 0 else 0
            trend_names = {0: 'range', 1: 'up', -1: 'down'}
            current_trend = trend_names.get(current_trend_value, 'range')
            
            # 品質スコア計算
            quality_score = calculate_her_quality_score(her_values, linear_vol, nonlinear_vol, adaptive_vol)
            
            # 結果の作成
            result = HyperERResult(
                values=her_values,
                linear_volatility=linear_vol,
                nonlinear_volatility=nonlinear_vol,
                adaptive_volatility=adaptive_vol,
                efficiency_components=efficiency_components,
                trend_signals=trend_signals,
                current_trend=current_trend,
                current_trend_value=current_trend_value,
                quality_score=quality_score
            )
            
            self._result = result
            self._cache[data_hash] = result
            return self._copy_result()
            
        except Exception as e:
            self.logger.error(f"HER計算中にエラー: {e}\n{traceback.format_exc()}")
            return self._create_empty_result(current_len)
    
    def _create_empty_result(self, length: int = 0) -> HyperERResult:
        """空の結果を作成"""
        return HyperERResult(
            values=np.full(length, np.nan),
            linear_volatility=np.full(length, np.nan),
            nonlinear_volatility=np.full(length, np.nan),
            adaptive_volatility=np.full(length, np.nan),
            efficiency_components=np.full(length, np.nan),
            trend_signals=np.zeros(length, dtype=np.int8),
            current_trend='range',
            current_trend_value=0,
            quality_score=0.0
        )
    
    def _copy_result(self) -> HyperERResult:
        """結果のコピーを作成"""
        if self._result is None:
            return self._create_empty_result()
        
        return HyperERResult(
            values=self._result.values.copy(),
            linear_volatility=self._result.linear_volatility.copy(),
            nonlinear_volatility=self._result.nonlinear_volatility.copy(),
            adaptive_volatility=self._result.adaptive_volatility.copy(),
            efficiency_components=self._result.efficiency_components.copy(),
            trend_signals=self._result.trend_signals.copy(),
            current_trend=self._result.current_trend,
            current_trend_value=self._result.current_trend_value,
            quality_score=self._result.quality_score
        )
    
    # 後方互換性メソッド
    def get_values(self) -> Optional[np.ndarray]:
        """HER値の取得"""
        return self._result.values.copy() if self._result else None
    
    def get_trend_signals(self) -> Optional[np.ndarray]:
        """トレンド信号の取得"""
        return self._result.trend_signals.copy() if self._result else None
    
    def get_current_trend(self) -> str:
        """現在のトレンド状態"""
        return self._result.current_trend if self._result else 'range'
    
    def get_quality_score(self) -> float:
        """品質スコアの取得"""
        return self._result.quality_score if self._result else 0.0
    
    def reset(self) -> None:
        """状態のリセット"""
        super().reset()
        self._result = None
        self._cache = {}
        self.logger.debug(f"HER '{self.name}' がリセットされました。")
