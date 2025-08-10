#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Hyper Channel - 高度なケルトナーチャネル** 🎯

従来のケルトナーチャネルを大幅に強化したハイブリッドチャネルインジケーター：
- ミッドライン: Unified Smoother（複数スムーサーから選択可能）
- バンド幅: Ultimate ATR（高精度ボラティリティ測定）
- 動的適応: HyperER/HyperADXによる乗数調整
- 統合フィルタリング: カルマンフィルターとルーフィングフィルター統合

🌟 **主要機能:**
1. **Unified Smoother ミッドライン**: FRAMA, Super Smoother, Ultimate Smoother等から選択
2. **X_ATR**: 拡張的ATRによる高精度ボラティリティ測定でバンド幅を算出
3. **動的乗数調整**: HyperER/HyperADXに基づく適応的バンド幅
4. **統合フィルタリング**: カルマンフィルターとルーフィングフィルター
5. **高速計算**: Numba JIT最適化による高速処理

📊 **チャネル構造:**
- Upper Band = Midline + (X_ATR × Dynamic Multiplier)
- Midline = Unified Smoother値
- Lower Band = Midline - (X_ATR × Dynamic Multiplier)

🔧 **パラメータ:**
- Unified Smoother: スムーサータイプと関連パラメータ
- X_ATR: 拡張的ATRボラティリティ計算パラメータ
- Dynamic Adaptation: HyperER/HyperADXによる乗数制御
- Filtering: カルマンフィルターとルーフィングフィルター設定
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from numba import njit, vectorize, prange
import traceback

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .smoother.unified_smoother import UnifiedSmoother
    from .volatility.x_atr import XATR
    from .trend_filter.hyper_er import HyperER
    from .trend_filter.hyper_adx import HyperADX
except ImportError:
    # フォールバック (テストや静的解析用)
    import sys
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from indicators.indicator import Indicator
    from indicators.price_source import PriceSource
    from indicators.smoother.unified_smoother import UnifiedSmoother
    from indicators.volatility.x_atr import XATR
    from indicators.trend_filter.hyper_er import HyperER
    from indicators.trend_filter.hyper_adx import HyperADX


@dataclass
class HyperChannelResult:
    """ハイパーチャネルの計算結果"""
    midline: np.ndarray                    # ミッドライン (Unified Smoother値)
    upper_band: np.ndarray                 # 上限バンド
    lower_band: np.ndarray                 # 下限バンド
    x_atr: np.ndarray                      # X_ATR値
    dynamic_multiplier: np.ndarray         # 動的乗数
    adaptation_values: np.ndarray          # 動的適応に使用した値 (HyperER or HyperADX)
    smoother_type: str                     # 使用されたスムーサータイプ
    adaptation_type: str                   # 使用された適応タイプ
    # オプション追加情報
    filtered_midline: Optional[np.ndarray] = None  # フィルタリング後のミッドライン（使用時のみ）
    raw_midline: Optional[np.ndarray] = None       # 生のミッドライン（比較用）


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_multiplier_vec(adaptation_value: float, max_mult: float, min_mult: float) -> float:
    """
    適応値（HyperER/HyperADX）に基づいて動的なATR乗数を計算する（ベクトル化版）
    
    Args:
        adaptation_value: 適応値（0-1の範囲）
        max_mult: 最大乗数（レンジ相場時）
        min_mult: 最小乗数（トレンド時）
    
    Returns:
        動的な乗数の値
    """
    if np.isnan(adaptation_value):
        return max_mult  # デフォルト値
    
    # 0-1の範囲にクリップ
    clamped_value = max(0.0, min(1.0, abs(adaptation_value)))
    
    # HyperER: 高い値（効率的）→ 小さい乗数（タイトなバンド）
    # HyperADX: 高い値（強いトレンド）→ 小さい乗数（タイトなバンド）
    # 両方とも同じロジック: 値が高い→小さい乗数、値が低い→大きい乗数
    multiplier = max_mult - clamped_value * (max_mult - min_mult)
    
    return multiplier


@njit(fastmath=True, cache=True)
def calculate_hyper_channel_bands(
    midline: np.ndarray,
    x_atr: np.ndarray,
    dynamic_multiplier: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ハイパーチャネルのバンドを計算する（Numba最適化版）
    
    Args:
        midline: ミッドライン値の配列
        x_atr: X_ATR値の配列
        dynamic_multiplier: 動的乗数の配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (ミッドライン, 上限バンド, 下限バンド)
    """
    length = min(len(midline), len(x_atr), len(dynamic_multiplier))
    
    if length == 0:
        empty_array = np.zeros(0, dtype=np.float64)
        return empty_array, empty_array, empty_array
    
    # 配列長を調整
    midline_adj = midline[-length:] if len(midline) > length else midline
    atr_adj = x_atr[-length:] if len(x_atr) > length else x_atr
    mult_adj = dynamic_multiplier[-length:] if len(dynamic_multiplier) > length else dynamic_multiplier
    
    # 結果配列の初期化
    upper_band = np.empty(length, dtype=np.float64)
    lower_band = np.empty(length, dtype=np.float64)
    
    # バンド計算
    for i in range(length):
        if not (np.isnan(midline_adj[i]) or np.isnan(atr_adj[i]) or np.isnan(mult_adj[i])):
            band_width = atr_adj[i] * mult_adj[i]
            upper_band[i] = midline_adj[i] + band_width
            lower_band[i] = midline_adj[i] - band_width
        else:
            upper_band[i] = np.nan
            lower_band[i] = np.nan
    
    return midline_adj, upper_band, lower_band


class HyperChannel(Indicator):
    """
    ハイパーチャネル（Hyper Channel）インジケーター
    
    高度なケルトナーチャネル実装：
    - ミッドライン: Unified Smoother（多様なスムーサーから選択）
    - バンド幅: X_ATR（拡張的ATRによる高精度ボラティリティ）
    - 動的適応: HyperER/HyperADXによる乗数調整
    - 統合フィルタリング: カルマンフィルターとルーフィングフィルター
    
    特徴:
    - 市場状態に応じたバンド幅の自動調整
    - X_ATRによる高精度ボラティリティ測定による優れたサポート・レジスタンス検出
    - 複数のスムージング手法による柔軟なミッドライン設定
    - Numba最適化による高速処理
    """
    
    def __init__(
        self,
        # Unified Smoother パラメータ
        smoother_type: str = 'frama',           # スムーサータイプ
        smoother_period: int = 14,              # スムーサー期間
        smoother_src_type: str = 'oc2',         # スムーサー用価格ソース
        
        # X_ATR パラメータ
        atr_period: float = 14.0,               # X_ATR期間
        atr_tr_method: str = 'str',             # X_ATR TR計算方法
        atr_smoother_type: str = 'laguerre',       # X_ATR スムーサータイプ
        atr_src_type: str = 'close',            # X_ATR用価格ソース
        
        # 動的適応パラメータ
        adaptation_type: str = 'hyper_er',      # 'hyper_er' または 'hyper_adx'
        max_multiplier: float = 6.0,            # 最大乗数（レンジ相場時）
        min_multiplier: float = 0.5,            # 最小乗数（トレンド時）
        
        # HyperER パラメータ
        hyper_er_period: int = 14,
        hyper_er_midline_period: int = 100,
        hyper_er_src_type: str = 'close',
        
        # HyperADX パラメータ
        hyper_adx_period: int = 14,
        hyper_adx_midline_period: int = 100,
        
        # フィルタリングパラメータ
        use_kalman_filter: bool = True,         # カルマンフィルター使用
        kalman_filter_type: str = 'simple',  # カルマンフィルタータイプ
        kalman_process_noise: float = 1e-5,     # プロセスノイズ
        kalman_observation_noise: float = 1e-6, # 観測ノイズ
        
        use_roofing_filter: bool = True,        # ルーフィングフィルター使用
        roofing_hp_cutoff: float = 55.0,        # ハイパスカットオフ
        roofing_ss_band_edge: float = 10.0,     # スーパースムーサーバンドエッジ
        
        # 追加パラメータ（スムーサー固有）
        **smoother_kwargs
    ):
        """
        ハイパーチャネルの初期化
        
        Args:
            smoother_type: 使用するスムーサータイプ
            smoother_period: スムーサー期間
            smoother_src_type: スムーサー用価格ソース
            atr_period: X_ATR期間
            atr_tr_method: X_ATR TR計算方法
            atr_smoother_type: X_ATR スムーサータイプ
            atr_src_type: X_ATR用価格ソース
            adaptation_type: 動的適応タイプ
            max_multiplier: 最大乗数
            min_multiplier: 最小乗数
            hyper_er_period: HyperER期間
            hyper_er_midline_period: HyperERミッドライン期間
            hyper_er_src_type: HyperER価格ソース
            hyper_adx_period: HyperADX期間
            hyper_adx_midline_period: HyperADXミッドライン期間
            use_kalman_filter: カルマンフィルター使用
            kalman_filter_type: カルマンフィルタータイプ
            kalman_process_noise: プロセスノイズ
            kalman_observation_noise: 観測ノイズ
            use_roofing_filter: ルーフィングフィルター使用
            roofing_hp_cutoff: ハイパスカットオフ
            roofing_ss_band_edge: スーパースムーサーバンドエッジ
            **smoother_kwargs: スムーサー固有の追加パラメータ
        """
        # インジケーター名の作成
        filter_str = ""
        if use_kalman_filter:
            filter_str += f"_kalman({kalman_filter_type})"
        if use_roofing_filter:
            filter_str += f"_roofing({roofing_hp_cutoff},{roofing_ss_band_edge})"
        
        indicator_name = f"HyperChannel({smoother_type}_{smoother_period},{adaptation_type}_mult{min_multiplier}-{max_multiplier}{filter_str})"
        super().__init__(indicator_name)
        
        # パラメータ検証
        if adaptation_type not in ['hyper_er', 'hyper_adx']:
            raise ValueError("adaptation_type must be 'hyper_er' or 'hyper_adx'")      
        if max_multiplier <= min_multiplier:
            raise ValueError("max_multiplier must be greater than min_multiplier")
        
        # パラメータ保存
        self.smoother_type = smoother_type
        self.smoother_period = smoother_period
        self.smoother_src_type = smoother_src_type
        self.atr_period = atr_period
        self.atr_tr_method = atr_tr_method
        self.atr_smoother_type = atr_smoother_type
        self.atr_src_type = atr_src_type
        self.adaptation_type = adaptation_type
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        
        # HyperER パラメータ
        self.hyper_er_period = hyper_er_period
        self.hyper_er_midline_period = hyper_er_midline_period
        self.hyper_er_src_type = hyper_er_src_type
        
        # HyperADX パラメータ
        self.hyper_adx_period = hyper_adx_period
        self.hyper_adx_midline_period = hyper_adx_midline_period
        
        # フィルタリングパラメータ
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_observation_noise = kalman_observation_noise
        self.use_roofing_filter = use_roofing_filter
        self.roofing_hp_cutoff = roofing_hp_cutoff
        self.roofing_ss_band_edge = roofing_ss_band_edge
        
        # スムーサー固有パラメータ
        self.smoother_kwargs = smoother_kwargs
        
        # コンポーネントの遅延初期化
        self._smoother = None
        self._ultimate_atr = None
        self._adaptation_indicator = None
        self._initialized = False
        
        # 結果キャッシュ
        self._result = None
        self._cache = {}
        self._cache_keys = []
        self._max_cache_size = 5
    
    def _initialize_components(self):
        """コンポーネントの遅延初期化"""
        if self._initialized:
            return
        
        try:
            # 1. Unified Smoother の初期化
            smoother_params = {
                'smoother_type': self.smoother_type,
                'period': self.smoother_period,
                'src_type': self.smoother_src_type,
                **self.smoother_kwargs
            }
            
            self._smoother = UnifiedSmoother(**smoother_params)
            self.logger.debug(f"Unified Smoother ({self.smoother_type}) を初期化しました")
            
            # 2. X_ATR の初期化
            atr_params = {
                'period': self.atr_period,
                'tr_method': self.atr_tr_method,
                'smoother_type': self.atr_smoother_type,
                'src_type': self.atr_src_type,
                'enable_kalman': self.use_kalman_filter,
                'kalman_type': self.kalman_filter_type if self.use_kalman_filter else 'unscented'
            }
            
            self._x_atr = XATR(**atr_params)
            self.logger.debug(f"X_ATR (period={self.atr_period}, tr_method={self.atr_tr_method}) を初期化しました")
            
            # 3. 動的適応インジケーターの初期化
            if self.adaptation_type == 'hyper_er':
                self._adaptation_indicator = HyperER(
                    period=self.hyper_er_period,
                    midline_period=self.hyper_er_midline_period,
                    er_src_type=self.hyper_er_src_type
                )
                self.logger.debug(f"HyperER (period={self.hyper_er_period}) を初期化しました")
            else:  # hyper_adx
                self._adaptation_indicator = HyperADX(
                    period=self.hyper_adx_period,
                    midline_period=self.hyper_adx_midline_period
                )
                self.logger.debug(f"HyperADX (period={self.hyper_adx_period}) を初期化しました")
            
            self._initialized = True
            
        except Exception as e:
            self.logger.error(f"コンポーネント初期化エラー: {e}")
            raise
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データハッシュを計算"""
        try:
            if isinstance(data, pd.DataFrame):
                shape_tuple = data.shape
                first_row = tuple(data.iloc[0]) if not data.empty else ()
                last_row = tuple(data.iloc[-1]) if not data.empty else ()
                data_hash_val = hash((shape_tuple, first_row, last_row))
            elif isinstance(data, np.ndarray):
                data_hash_val = hash(data.tobytes())
            else:
                data_hash_val = hash(str(data))
        except Exception:
            data_hash_val = hash(str(data))
        
        # パラメータハッシュ
        param_tuple = (
            self.smoother_type, self.smoother_period, self.smoother_src_type,
            self.atr_period, self.atr_src_type,
            self.adaptation_type, self.max_multiplier, self.min_multiplier,
            self.hyper_er_period, self.hyper_er_midline_period,
            self.hyper_adx_period, self.hyper_adx_midline_period,
            self.use_kalman_filter, self.kalman_filter_type,
            self.use_roofing_filter, self.roofing_hp_cutoff
        )
        param_hash = hash(param_tuple)
        
        return f"{data_hash_val}_{param_hash}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperChannelResult:
        """
        ハイパーチャネルを計算する
        
        Args:
            data: 価格データ (DataFrame または numpy配列)
            
        Returns:
            HyperChannelResult: 計算結果
        """
        try:
            # データ検証
            if data is None or (isinstance(data, pd.DataFrame) and data.empty) or \
               (isinstance(data, np.ndarray) and data.size == 0):
                self.logger.warning("入力データが空です")
                return self._empty_result()
            
            # キャッシュチェック
            current_hash = self._get_data_hash(data)
            if current_hash in self._cache:
                self._cache_keys.remove(current_hash)
                self._cache_keys.append(current_hash)
                self._result = self._cache[current_hash]
                return self._result
            
            # コンポーネント初期化
            self._initialize_components()
            
            # 1. Unified Smoother でミッドライン計算
            self.logger.debug("ミッドライン計算中...")
            smoother_result = self._smoother.calculate(data)
            if smoother_result is None or len(smoother_result.values) == 0:
                self.logger.error("スムーサー計算が失敗しました")
                return self._empty_result()
            
            midline = smoother_result.values
            raw_midline = smoother_result.raw_values
            
            # 2. X_ATR 計算
            self.logger.debug("X_ATR計算中...")
            atr_result = self._x_atr.calculate(data)
            if atr_result is None or len(atr_result.values) == 0:
                self.logger.error("X_ATR計算が失敗しました")
                return self._empty_result()
            
            x_atr_values = atr_result.values
            
            # 3. 動的適応値計算
            self.logger.debug(f"{self.adaptation_type}計算中...")
            adaptation_result = self._adaptation_indicator.calculate(data)
            if adaptation_result is None or len(adaptation_result.values) == 0:
                self.logger.error("動的適応値計算が失敗しました")
                return self._empty_result()
            
            adaptation_values = adaptation_result.values
            
            # 4. 動的乗数計算
            self.logger.debug("動的乗数計算中...")
            adaptation_np = np.asarray(adaptation_values, dtype=np.float64)
            dynamic_multiplier = calculate_dynamic_multiplier_vec(
                adaptation_np, 
                self.max_multiplier, 
                self.min_multiplier
            )
            
            # 5. チャネルバンド計算
            self.logger.debug("チャネルバンド計算中...")
            midline_np = np.asarray(midline, dtype=np.float64)
            x_atr_np = np.asarray(x_atr_values, dtype=np.float64)
            
            midline_final, upper_band, lower_band = calculate_hyper_channel_bands(
                midline_np, x_atr_np, dynamic_multiplier
            )
            
            # 6. 結果作成
            result = HyperChannelResult(
                midline=midline_final,
                upper_band=upper_band,
                lower_band=lower_band,
                x_atr=x_atr_np,
                dynamic_multiplier=dynamic_multiplier,
                adaptation_values=adaptation_values,
                smoother_type=self.smoother_type,
                adaptation_type=self.adaptation_type,
                filtered_midline=smoother_result.kalman_filtered_values,
                raw_midline=raw_midline
            )
            
            # キャッシュ更新
            self._cache[current_hash] = result
            self._cache_keys.append(current_hash)
            
            if len(self._cache_keys) > self._max_cache_size:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]
            
            self._result = result
            self._values = midline_final  # 基底クラス用
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ハイパーチャネル計算エラー: {error_msg}\\n{stack_trace}")
            return self._empty_result()
    
    def _empty_result(self) -> HyperChannelResult:
        """空の結果を返す"""
        empty_array = np.array([])
        return HyperChannelResult(
            midline=empty_array,
            upper_band=empty_array,
            lower_band=empty_array,
            x_atr=empty_array,
            dynamic_multiplier=empty_array,
            adaptation_values=empty_array,
            smoother_type=self.smoother_type,
            adaptation_type=self.adaptation_type
        )
    
    def get_bands(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        チャネルバンドを取得
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (ミッドライン, 上限バンド, 下限バンド)
        """
        if self._result is None:
            empty = np.array([])
            return empty, empty, empty
        
        return (
            self._result.midline.copy(),
            self._result.upper_band.copy(),
            self._result.lower_band.copy()
        )
    
    def get_midline(self) -> np.ndarray:
        """ミッドライン値を取得"""
        if self._result is None:
            return np.array([])
        return self._result.midline.copy()
    
    def get_x_atr(self) -> np.ndarray:
        """X_ATR値を取得"""
        if self._result is None:
            return np.array([])
        return self._result.x_atr.copy()
    
    def get_dynamic_multiplier(self) -> np.ndarray:
        """動的乗数を取得"""
        if self._result is None:
            return np.array([])
        return self._result.dynamic_multiplier.copy()
    
    def get_adaptation_values(self) -> np.ndarray:
        """動的適応値を取得"""
        if self._result is None:
            return np.array([])
        return self._result.adaptation_values.copy()
    
    def get_result(self) -> Optional[HyperChannelResult]:
        """完全な計算結果を取得"""
        return self._result
    
    def reset(self) -> None:
        """インジケーターをリセット"""
        super().reset()
        
        # 結果とキャッシュをクリア
        self._result = None
        self._cache.clear()
        self._cache_keys.clear()
        
        # コンポーネントをリセット
        for component_name in ['_smoother', '_x_atr', '_adaptation_indicator']:
            component = getattr(self, component_name, None)
            if component and hasattr(component, 'reset'):
                try:
                    component.reset()
                    self.logger.debug(f"{component_name}をリセットしました")
                except Exception as e:
                    self.logger.warning(f"{component_name}のリセット中にエラー: {e}")
        
        self._initialized = False
        self.logger.debug(f"ハイパーチャネル '{self.name}' をリセットしました")


# 便利関数
def calculate_hyper_channel(
    data: Union[pd.DataFrame, np.ndarray],
    smoother_type: str = 'frama',
    smoother_period: int = 21,
    atr_period: int = 14,
    adaptation_type: str = 'hyper_er',
    max_multiplier: float = 3.0,
    min_multiplier: float = 0.8,
    **kwargs
) -> HyperChannelResult:
    """
    ハイパーチャネルを計算する便利関数
    
    Args:
        data: 価格データ
        smoother_type: スムーサータイプ
        smoother_period: スムーサー期間
        atr_period: ATR期間
        adaptation_type: 動的適応タイプ
        max_multiplier: 最大乗数
        min_multiplier: 最小乗数
        **kwargs: その他のパラメータ
        
    Returns:
        HyperChannelResult: 計算結果
    """
    indicator = HyperChannel(
        smoother_type=smoother_type,
        smoother_period=smoother_period,
        atr_period=atr_period,
        adaptation_type=adaptation_type,
        max_multiplier=max_multiplier,
        min_multiplier=min_multiplier,
        **kwargs
    )
    
    return indicator.calculate(data)