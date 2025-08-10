#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import logging
from numba import njit, prange, vectorize
from position_sizing.position_sizing import PositionSizing, PositionSizingParams
from position_sizing.interfaces import IPositionManager
from indicators.volatility.x_atr import XATR
from indicators.trend_filter.hyper_er import HyperER
from indicators.hyper_trend_index import HyperTrendIndex


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_multiplier_vec(trigger_value: float, max_mult: float, min_mult: float) -> float:
    """
    トリガー値に基づいて動的なATR乗数を計算する（ベクトル化版）
    トリガーが高いときに小さくなり、低いときに大きくなる
    
    Args:
        trigger_value: トリガーの値（Hyper ER、Hyper Trend Indexなど）0-1の範囲
        max_mult: 最大乗数（トリガー値が低いときに使用）
        min_mult: 最小乗数（トリガー値が高いときに使用）
    
    Returns:
        動的な乗数の値
    """
    if np.isnan(trigger_value):
        return (max_mult + min_mult) / 2  # デフォルト値
    
    # 0-1の範囲にクリップ
    trigger_clipped = max(0.0, min(1.0, abs(trigger_value)))
    
    # トリガーが高い(1に近い)ときに最小乗数、低い(0に近い)ときに最大乗数
    return max_mult - trigger_clipped * (max_mult - min_mult)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_risk_ratio(trigger_value: float, max_risk: float, min_risk: float) -> float:
    """
    トリガー値に基づいて動的なリスク比率を計算する（ベクトル化版）
    トリガーが高いときに最大リスクに近づき、低いときに最小リスクに近づく
    
    Args:
        trigger_value: トリガーの値 0-1の範囲
        max_risk: 最大リスク比率（例：0.03）トリガー値が高いときに使用
        min_risk: 最小リスク比率（例：0.01）トリガー値が低いときに使用
    
    Returns:
        動的なリスク比率の値
    """
    if np.isnan(trigger_value):
        return (max_risk + min_risk) / 2  # デフォルト値
    
    # 0-1の範囲にクリップ
    trigger_clipped = max(0.0, min(1.0, abs(trigger_value)))
    
    # トリガーが高い(1に近い)ときに最大リスク、低い(0に近い)ときに最小リスク
    return min_risk + trigger_clipped * (max_risk - min_risk)


@njit(fastmath=True, cache=True)
def _calculate_position_size_numba(
    capital: float,
    risk_ratio: float,
    x_atr: float,
    multiplier: float,
    entry_price: float,
    unit_coefficient: float,
    leverage: float,
    max_position_percent: float
) -> float:
    """
    ポジションサイズを計算する（Numba最適化版）
    
    Args:
        capital: 資本額
        risk_ratio: リスク比率
        x_atr: X_ATR値
        multiplier: X_ATR乗数
        entry_price: エントリー価格
        unit_coefficient: 単位係数（効率比で調整済み）
        leverage: レバレッジ
        max_position_percent: 最大ポジションの比率
        
    Returns:
        float: ポジションサイズ（USD建て）
    """
    # ポジションサイズの計算: 資本 × リスク比率 ÷ (X_ATR × 乗数) × 価格 × 単位係数
    x_atr_risk = x_atr * multiplier
    position_size = capital * risk_ratio / (x_atr_risk / entry_price) * unit_coefficient
    
    # レバレッジの適用
    position_size *= leverage
    
    # 最大ポジションサイズの制限を適用
    max_position = capital * max_position_percent * leverage
    if position_size > max_position:
        position_size = max_position
        
    return position_size


@njit(fastmath=True, cache=True)
def _calculate_risk_amount_numba(position_size: float, x_atr: float, multiplier: float, entry_price: float) -> float:
    """
    リスク金額を計算する（Numba最適化版）
    
    Args:
        position_size: ポジションサイズ（USD建て）
        x_atr: X_ATR値
        multiplier: X_ATR乗数
        entry_price: エントリー価格
        
    Returns:
        float: リスク金額（USD建て）
    """
    x_atr_risk = x_atr * multiplier
    return position_size * x_atr_risk / entry_price


@njit(fastmath=True, parallel=True, cache=True)
def calculate_batch_position_sizes_numba(
    capitals: np.ndarray,
    risk_ratios: np.ndarray,
    x_atrs: np.ndarray, 
    multipliers: np.ndarray,
    entry_prices: np.ndarray,
    unit_coefficients: np.ndarray,
    leverages: np.ndarray,
    max_position_percents: np.ndarray
) -> np.ndarray:
    """
    複数のポジションサイズを一度に計算する（Numba並列最適化版）
    
    Args:
        capitals: 資本額の配列
        risk_ratios: リスク比率の配列
        x_atrs: X_ATR値の配列
        multipliers: X_ATR乗数の配列
        entry_prices: エントリー価格の配列
        unit_coefficients: 単位係数の配列（効率比で調整済み）
        leverages: レバレッジの配列
        max_position_percents: 最大ポジションの比率の配列
        
    Returns:
        np.ndarray: ポジションサイズの配列（USD建て）
    """
    n = len(capitals)
    position_sizes = np.zeros(n, dtype=np.float64)
    
    # 並列ループで高速に計算
    for i in prange(n):
        # ポジションサイズの計算: 資本 × リスク比率 ÷ (X_ATR × 乗数) × 価格 × 単位係数
        x_atr_risk = x_atrs[i] * multipliers[i]
        position_size = capitals[i] * risk_ratios[i] / (x_atr_risk / entry_prices[i]) * unit_coefficients[i]
        
        # レバレッジの適用
        position_size *= leverages[i]
        
        # 最大ポジションサイズの制限を適用
        max_position = capitals[i] * max_position_percents[i] * leverages[i]
        if position_size > max_position:
            position_size = max_position
            
        position_sizes[i] = position_size
        
    return position_sizes


class XATRPositionSizing(PositionSizing, IPositionManager):
    """
    X_ATRベースのポジションサイジング
    
    X_ATRベースのポジションサイジングで、複数のトリガー（Hyper ER、Hyper Trend Index）に基づく
    動的リスク調整と乗数調整が可能
    """
    
    def __init__(
        self, 
        base_risk_ratio: float = 0.01,  # 基本リスク比率（デフォルト2%）
        unit: float = 1.0,              # 基本単位係数（デフォルト1.0）
        max_position_percent: float = 0.3,  # 最大ポジションサイズの比率（デフォルト30%）
        leverage: float = 1.0,          # レバレッジ（デフォルト1倍）
        
        # X_ATRパラメータ
        x_atr_period: float = 25.0,
        x_atr_tr_method: str = 'atr',   # 'atr' または 'str'
        x_atr_smoother_type: str = 'laguerre',
        x_atr_enable_kalman: bool = True,
        x_atr_kalman_type: str = 'unscented',
        x_atr_period_mode: str = 'dynamic',
        x_atr_cycle_detector_type: str = 'dft_dominant',
        
        apply_dynamic_adjustment: bool = True,  # 動的調整を適用するか
        fixed_risk_percent: float = 0.01,  # 固定リスク率（資金の1%）
        
        # 動的適応のトリガータイプ
        trigger_type: str = 'hyper_trend_index',     # 'hyper_er' または 'hyper_trend_index'
        
        # 動的ATR乗数のパラメータ
        max_multiplier: float = 5.0,        # 最大ATR乗数
        min_multiplier: float = 2.0,        # 最小ATR乗数
        
        # 動的リスク比率のパラメータ
        max_risk_ratio: float = 0.02,   # 最大リスク比率（2%）
        min_risk_ratio: float = 0.005,  # 最小リスク比率（0.5%）
        
        # HyperER固有のパラメータ
        hyper_er_period: int = 14,
        hyper_er_midline_period: int = 100,
        hyper_er_er_period: int = 13,
        hyper_er_er_src_type: str = 'oc2',
        hyper_er_use_roofing_filter: bool = True,
        hyper_er_roofing_hp_cutoff: float = 55.0,
        hyper_er_roofing_ss_band_edge: float = 10.0,
        
        # HyperTrendIndex固有のパラメータ
        hyper_trend_period: int = 14,
        hyper_trend_midline_period: int = 100,
        hyper_trend_src_type: str = 'hlc3',
        hyper_trend_use_kalman_filter: bool = True,
        hyper_trend_kalman_filter_type: str = 'unscented',
        hyper_trend_use_dynamic_period: bool = True,
        hyper_trend_detector_type: str = 'dft_dominant',
        hyper_trend_use_roofing_filter: bool = False,
        hyper_trend_roofing_hp_cutoff: float = 55.0,
        hyper_trend_roofing_ss_band_edge: float = 10.0
    ):
        """
        初期化
        
        Args:
            base_risk_ratio: 基本リスク比率（資本に対する比率、例：0.02 = 2%）
            unit: 基本単位係数
            max_position_percent: 最大ポジションサイズの比率（資本に対する比率）
            leverage: レバレッジ
            
            x_atr_*: X_ATR固有のパラメータ
            
            apply_dynamic_adjustment: 動的調整を適用するか
            fixed_risk_percent: 固定リスク率（資金の何%をリスクとするか）
            trigger_type: 動的適応のトリガータイプ（'hyper_er', 'hyper_trend_index'）
            
            max_multiplier: 最大ATR乗数
            min_multiplier: 最小ATR乗数
            
            max_risk_ratio: 最大リスク比率
            min_risk_ratio: 最小リスク比率
            
            hyper_er_*: HyperER固有のパラメータ
            hyper_trend_*: HyperTrendIndex固有のパラメータ
        """
        super().__init__()
        
        # 全パラメータをインスタンス変数として保存
        self.base_risk_ratio = base_risk_ratio
        self.unit = unit
        self.max_position_percent = max_position_percent
        self.leverage = leverage
        self.apply_dynamic_adjustment = apply_dynamic_adjustment
        self.fixed_risk_percent = fixed_risk_percent
        self.trigger_type = trigger_type
        
        # X_ATRパラメータ
        self.x_atr_period = x_atr_period
        self.x_atr_tr_method = x_atr_tr_method
        self.x_atr_smoother_type = x_atr_smoother_type
        self.x_atr_enable_kalman = x_atr_enable_kalman
        self.x_atr_kalman_type = x_atr_kalman_type
        self.x_atr_period_mode = x_atr_period_mode
        self.x_atr_cycle_detector_type = x_atr_cycle_detector_type
        
        # 動的乗数のパラメータ
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        
        # 動的リスク比率のパラメータ
        self.max_risk_ratio = max_risk_ratio
        self.min_risk_ratio = min_risk_ratio
        
        # HyperERパラメータ
        self.hyper_er_period = hyper_er_period
        self.hyper_er_midline_period = hyper_er_midline_period
        self.hyper_er_er_period = hyper_er_er_period
        self.hyper_er_er_src_type = hyper_er_er_src_type
        self.hyper_er_use_roofing_filter = hyper_er_use_roofing_filter
        self.hyper_er_roofing_hp_cutoff = hyper_er_roofing_hp_cutoff
        self.hyper_er_roofing_ss_band_edge = hyper_er_roofing_ss_band_edge
        
        # HyperTrendIndexパラメータ
        self.hyper_trend_period = hyper_trend_period
        self.hyper_trend_midline_period = hyper_trend_midline_period
        self.hyper_trend_src_type = hyper_trend_src_type
        self.hyper_trend_use_kalman_filter = hyper_trend_use_kalman_filter
        self.hyper_trend_kalman_filter_type = hyper_trend_kalman_filter_type
        self.hyper_trend_use_dynamic_period = hyper_trend_use_dynamic_period
        self.hyper_trend_detector_type = hyper_trend_detector_type
        self.hyper_trend_use_roofing_filter = hyper_trend_use_roofing_filter
        self.hyper_trend_roofing_hp_cutoff = hyper_trend_roofing_hp_cutoff
        self.hyper_trend_roofing_ss_band_edge = hyper_trend_roofing_ss_band_edge
        
        # ロガーの設定
        self.logger = logging.getLogger(__name__)
        
        # X_ATR インスタンスを作成
        self.x_atr = XATR(
            period=x_atr_period,
            tr_method=x_atr_tr_method,
            smoother_type=x_atr_smoother_type,
            enable_kalman=x_atr_enable_kalman,
            kalman_type=x_atr_kalman_type,
            period_mode=x_atr_period_mode,
            cycle_detector_type=x_atr_cycle_detector_type
        )
        
        # トリガーインジケーターの初期化
        self._init_trigger_indicators()
    
    def _init_trigger_indicators(self):
        """トリガータイプに応じてインジケーターを初期化"""
        if self.trigger_type == 'hyper_er':
            # HyperER インスタンスを作成
            self.hyper_er = HyperER(
                period=self.hyper_er_period,
                midline_period=self.hyper_er_midline_period,
                er_period=self.hyper_er_er_period,
                er_src_type=self.hyper_er_er_src_type,
                use_roofing_filter=self.hyper_er_use_roofing_filter,
                roofing_hp_cutoff=self.hyper_er_roofing_hp_cutoff,
                roofing_ss_band_edge=self.hyper_er_roofing_ss_band_edge
            )
            self.trigger_indicator = self.hyper_er
            
        elif self.trigger_type == 'hyper_trend_index':
            # HyperTrendIndex インスタンスを作成
            self.hyper_trend_index = HyperTrendIndex(
                period=self.hyper_trend_period,
                midline_period=self.hyper_trend_midline_period,
                src_type=self.hyper_trend_src_type,
                use_kalman_filter=self.hyper_trend_use_kalman_filter,
                kalman_filter_type=self.hyper_trend_kalman_filter_type,
                use_dynamic_period=self.hyper_trend_use_dynamic_period,
                detector_type=self.hyper_trend_detector_type,
                use_roofing_filter=self.hyper_trend_use_roofing_filter,
                roofing_hp_cutoff=self.hyper_trend_roofing_hp_cutoff,
                roofing_ss_band_edge=self.hyper_trend_roofing_ss_band_edge
            )
            self.trigger_indicator = self.hyper_trend_index
            
        else:
            raise ValueError(f"サポートされていないトリガータイプ: {self.trigger_type}")
    
    def _get_trigger_value(self, history: pd.DataFrame) -> float:
        """
        トリガータイプに応じて適応値を取得
        
        Args:
            history: 履歴データ
            
        Returns:
            float: トリガー値
        """
        try:
            if self.trigger_type == 'hyper_er':
                # HyperERを計算
                trigger_result = self.hyper_er.calculate(history)
                if trigger_result is None or len(trigger_result.values) == 0:
                    return 0.5  # デフォルト値
                return trigger_result.values[-1]
                
            elif self.trigger_type == 'hyper_trend_index':
                # HyperTrendIndexを計算
                trigger_result = self.hyper_trend_index.calculate(history)
                if trigger_result is None or len(trigger_result.values) == 0:
                    return 0.5  # デフォルト値
                return trigger_result.values[-1]
                
            else:
                return 0.5  # デフォルト値
                
        except Exception as e:
            self.logger.warning(f"トリガー値計算中にエラー: {str(e)}、デフォルト値を使用します")
            return 0.5
    
    def _calculate_trigger_factor(self, trigger_value: float) -> float:
        """
        トリガー値から調整係数を計算
        
        Args:
            trigger_value: トリガーの値
            
        Returns:
            float: 調整係数
        """
        if self.trigger_type == 'hyper_er':
            # HyperERの場合：効率比が高い（トレンドが強い）ほど単位係数を大きくする
            return 0.5 + trigger_value  # 0.5〜1.5の範囲
            
        elif self.trigger_type == 'hyper_trend_index':
            # HyperTrendIndexの場合：トレンド値が高いほど単位係数を大きくする
            return 0.5 + trigger_value  # 0.5〜1.5の範囲
            
        else:
            return 1.0  # デフォルト値

    def can_enter(self) -> bool:
        """新規ポジションを取れるかどうか"""
        return True
    
    def calculate_position_size(self, price: float, capital: float) -> float:
        """
        シンプルなポジションサイズ計算（IPositionManagerインターフェース用）
        
        Args:
            price: 現在の価格
            capital: 現在の資金
            
        Returns:
            float: ポジションサイズ（USD建て）
        """
        params = PositionSizingParams(
            entry_price=price,
            stop_loss_price=None,  # X_ATR計算が内部で行われるので不要
            capital=capital,
            leverage=self.leverage,
            risk_per_trade=self.base_risk_ratio  # 基本リスク比率を正しく渡す
        )
        
        result = self.calculate(params)
        return result['position_size']
    
    def calculate(self, params: PositionSizingParams) -> Dict[str, Any]:
        """
        X_ATRベースのポジションサイズを計算
        
        Args:
            params: ポジションサイジングパラメータ
            
        Returns:
            Dict[str, Any]: 計算結果
        """
        # History要件の確認
        if params.historical_data is None:
            raise ValueError("履歴データが必要です")
            
        # 履歴データを取得
        history = params.historical_data
        
        # トリガー値を取得
        trigger_value = self._get_trigger_value(history) if self.apply_dynamic_adjustment else 0.5

        # X_ATRを計算する
        try:
            x_atr_result = self.x_atr.calculate(history)
            x_atr_value = x_atr_result.values[-1]
            
            # 値が無効な場合のみフォールバック処理
            if x_atr_value is None or np.isnan(x_atr_value):
                # 価格の0.01%を使用（小さな価格の通貨でも適切に機能するため）
                x_atr_value = params.entry_price * 0.0001
                self.logger.warning(f"X_ATR値が無効（None/NaN）、価格の0.01%をデフォルト値として使用: {x_atr_value}")
            
        except IndexError as e:
            # データが不足している場合のフォールバック
            x_atr_value = params.entry_price * 0.0001  # 価格の0.01%
            self.logger.warning(f"X_ATR計算中にインデックスエラー: {str(e)}、価格の0.01%をデフォルト値として使用: {x_atr_value}")
        except Exception as e:
            # その他のエラー
            x_atr_value = params.entry_price * 0.5  # 価格の0.5%
            self.logger.warning(f"X_ATR計算中にエラー: {str(e)}、価格の0.5%をデフォルト値として使用: {x_atr_value}")

        # 単位係数のベース値
        unit_coefficient = self.unit
        
        # 動的調整
        if self.apply_dynamic_adjustment:
            # トリガー値による調整係数の計算
            trigger_factor = self._calculate_trigger_factor(trigger_value)
            
            # トリガー値によるATR乗数の動的調整（簡素化）
            atr_multiplier = calculate_dynamic_multiplier_vec(
                trigger_value,
                self.max_multiplier,  # 最大乗数
                self.min_multiplier   # 最小乗数
            )
            
            # トリガー値によるリスク比率の動的調整
            dynamic_risk_ratio = calculate_dynamic_risk_ratio(
                trigger_value,
                self.max_risk_ratio,  # max_risk_ratio
                self.min_risk_ratio   # min_risk_ratio
            )
        else:
            # 調整なしの場合のデフォルト値
            trigger_factor = 1.0
            atr_multiplier = 1.5  # デフォルト値
            dynamic_risk_ratio = self.base_risk_ratio

        # Numba最適化関数を使用してポジションサイズを計算
        position_size_usd = _calculate_position_size_numba(
            capital=params.capital,
            risk_ratio=dynamic_risk_ratio,
            x_atr=x_atr_value,
            multiplier=atr_multiplier,
            entry_price=params.entry_price,
            unit_coefficient=unit_coefficient,
            leverage=self.leverage,
            max_position_percent=self.max_position_percent
        )

        # Numba最適化関数を使用してリスク金額を計算
        risk_amount = _calculate_risk_amount_numba(
            position_size=position_size_usd,
            x_atr=x_atr_value,
            multiplier=atr_multiplier,
            entry_price=params.entry_price
        )
        
        # 資産数量を計算（表示用）
        asset_quantity = position_size_usd / params.entry_price if params.entry_price > 0 else 0
        
        # 最大ポジションサイズ（情報表示用）
        max_position_size = params.capital * self.max_position_percent * self.leverage

        # 戻り値の構築
        return {
            'position_size': position_size_usd,
            'asset_quantity': asset_quantity,      # 資産数量（単位数）
            'risk_amount': risk_amount,
            'x_atr_value': x_atr_value,
            'atr_multiplier': atr_multiplier,
            'max_multiplier': self.max_multiplier,      # 動的最大乗数
            'min_multiplier': self.min_multiplier,      # 動的最小乗数
            'trigger_type': self.trigger_type,     # 使用したトリガータイプ
            'trigger_value': trigger_value,        # トリガー値
            'trigger_factor': trigger_factor,      # トリガー調整係数
            'unit': self.unit,                     # 元のunit値
            'unit_with_trigger': unit_coefficient, # トリガー調整後のunit値
            'risk_ratio': dynamic_risk_ratio,      # 動的リスク比率
            'max_position_size': max_position_size,
        }

    def calculate_stop_loss_price(
        self, 
        entry_price: float, 
        x_atr_value: float, 
        atr_multiplier: float, 
        is_long: bool = True
    ) -> float:
        """
        X_ATR乗数に基づいてストップロス価格を計算
        
        Args:
            entry_price: エントリー価格
            x_atr_value: X_ATR値
            atr_multiplier: X_ATR乗数
            is_long: ロングポジションかどうか（True=ロング、False=ショート）
            
        Returns:
            float: ストップロス価格
        """
        # X_ATRリスク（値幅）を計算
        x_atr_risk = x_atr_value * atr_multiplier
        
        # ポジションタイプに基づいてストップロス価格を計算
        if is_long:
            # ロングの場合はエントリー価格からX_ATRリスク分を引く
            stop_loss = entry_price - x_atr_risk
        else:
            # ショートの場合はエントリー価格にX_ATRリスク分を足す
            stop_loss = entry_price + x_atr_risk
            
        return stop_loss
    
    def calculate_position_size_with_fixed_risk(
        self, 
        entry_price: float, 
        capital: float, 
        historical_data: pd.DataFrame, 
        is_long: bool = True,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        固定リスク率（資金の1%）に基づいてポジションサイズを計算
        
        Args:
            entry_price: エントリー価格
            capital: 現在の資金額
            historical_data: 履歴データ
            is_long: ロングポジションかどうか（True=ロング、False=ショート）
            debug: デバッグ情報を出力するかどうか
            
        Returns:
            Dict[str, Any]: 計算結果
        """
        # 履歴データを取得
        history = historical_data
        
        # トリガー値を取得
        trigger_value = self._get_trigger_value(history) if self.apply_dynamic_adjustment else 0.5

        # X_ATRとX_ATR乗数を直接計算して取得する
        try:
            x_atr_result = self.x_atr.calculate(history)
            x_atr_value = x_atr_result.values[-1]
            
            # 値が無効な場合のみフォールバック処理
            if x_atr_value is None or np.isnan(x_atr_value):
                # 価格の0.01%を使用（小さな価格の通貨でも適切に機能するため）
                x_atr_value = entry_price * 0.01
                self.logger.warning(f"X_ATR値が無効（None/NaN）、価格の1%をデフォルト値として使用: {x_atr_value}")
            
        except IndexError as e:
            # データが不足している場合のフォールバック
            x_atr_value = entry_price * 0.01  # 価格の1%
            self.logger.warning(f"X_ATR計算中にインデックスエラー: {str(e)}、価格の1%をデフォルト値として使用: {x_atr_value}")
        except Exception as e:
            # その他のエラー
            x_atr_value = entry_price * 0.01  # 価格の1%
            self.logger.warning(f"X_ATR計算中にエラー: {str(e)}、価格の1%をデフォルト値として使用: {x_atr_value}")

        # 動的調整
        trigger_factor = 1.0
        
        if self.apply_dynamic_adjustment:
            # トリガー値による調整係数の計算
            trigger_factor = self._calculate_trigger_factor(trigger_value)
            
            # トリガー値によるATR乗数の動的調整（簡素化）
            atr_multiplier = calculate_dynamic_multiplier_vec(
                trigger_value,
                self.max_multiplier,  # 最大乗数
                self.min_multiplier   # 最小乗数
            )
        else:
            # 調整なしの場合のデフォルト値
            atr_multiplier = 1.5  # デフォルト値

        # ストップロス価格の計算
        stop_loss_price = self.calculate_stop_loss_price(
            entry_price=entry_price,
            x_atr_value=x_atr_value,
            atr_multiplier=atr_multiplier,
            is_long=is_long
        )
        
        # ポジションサイズを計算
        risk_amount = capital * self.fixed_risk_percent
        
        # エントリー価格とストップロス価格の差（ストップロス幅）
        stop_loss_distance = abs(entry_price - stop_loss_price)
        
        # デバッグ情報
        if debug:
            self.logger.info(f"==== ポジションサイズ計算デバッグ情報 ====")
            self.logger.info(f"エントリー価格: {entry_price}")
            self.logger.info(f"資金: {capital}")
            self.logger.info(f"固定リスク率: {self.fixed_risk_percent}")
            self.logger.info(f"X_ATR値: {x_atr_value}")
            self.logger.info(f"ATR乗数: {atr_multiplier}")
            self.logger.info(f"ストップロス価格: {stop_loss_price}")
            self.logger.info(f"ストップロス幅: {stop_loss_distance}")
            self.logger.info(f"リスク金額: {risk_amount}")
        
        # ストップロス幅に基づくポジションサイズ計算
        if stop_loss_distance > 0:
            # 正しい計算式: リスク金額 / (ストップロス幅 / エントリー価格)
            position_size_usd = risk_amount / (stop_loss_distance / entry_price)
        else:
            # ストップロス幅がゼロまたは負の場合（エラー状態）
            position_size_usd = 0
            self.logger.warning("ストップロス幅が無効です。ポジションサイズはゼロに設定されました。")
        
        # 最大ポジションサイズの制限を適用
        max_position_size = capital * self.max_position_percent * self.leverage
        
        if debug:
            self.logger.info(f"計算されたポジションサイズ: {position_size_usd}")
            self.logger.info(f"最大ポジションサイズ: {max_position_size}")
        
        # 最大値を超える場合は制限を適用
        if position_size_usd > max_position_size:
            if debug:
                self.logger.info(f"ポジションサイズが最大値を超えたため、{max_position_size}に制限されました")
            position_size_usd = max_position_size
        
        # 資産数量を計算（表示用）
        asset_quantity = position_size_usd / entry_price if entry_price > 0 else 0
        
        if debug:
            self.logger.info(f"最終ポジションサイズ: {position_size_usd}")
            self.logger.info(f"資産数量: {asset_quantity}")
            self.logger.info(f"========================================")
        
        # 結果の構築
        result = {
            'position_size': position_size_usd,
            'asset_quantity': asset_quantity,
            'stop_loss_price': stop_loss_price,
            'risk_amount': risk_amount,
            'stop_loss_distance': stop_loss_distance,
            'fixed_risk_percent': self.fixed_risk_percent,
            'x_atr_value': x_atr_value,
            'atr_multiplier': atr_multiplier,
            'trigger_type': self.trigger_type,        # 使用したトリガータイプ
            'trigger_value': trigger_value,           # トリガー値
            'trigger_factor': trigger_factor,         # トリガー調整係数
            'max_position_size': max_position_size,
        }
        
        return result

    @staticmethod
    def calculate_batch(
        capitals: np.ndarray,
        entry_prices: np.ndarray,
        historical_data_list: list,
        base_risk_ratios: np.ndarray = None,
        units: np.ndarray = None,
        max_position_percents: np.ndarray = None,
        leverages: np.ndarray = None,
        apply_dynamic_adjustments: np.ndarray = None,
        trigger_type: str = 'hyper_er'
    ) -> Dict[str, np.ndarray]:
        """
        バッチでX_ATRベースのポジションサイズを計算する
        
        Args:
            capitals: 資本額の配列
            entry_prices: エントリー価格の配列
            historical_data_list: 各ポジションの履歴データのリスト
            base_risk_ratios: 基本リスク比率の配列（省略時は0.02）
            units: 基本単位係数の配列（省略時は1.0）
            max_position_percents: 最大ポジションサイズの比率の配列（省略時は0.3）
            leverages: レバレッジの配列（省略時は1.0）
            apply_dynamic_adjustments: 動的調整を適用するかの配列（省略時はTrue）
            trigger_type: トリガータイプ
            
        Returns:
            Dict[str, np.ndarray]: 計算結果の辞書
        """
        n = len(capitals)
        if len(entry_prices) != n or len(historical_data_list) != n:
            raise ValueError("すべての入力配列の長さは同じである必要があります")
            
        # デフォルト値の設定
        if base_risk_ratios is None:
            base_risk_ratios = np.full(n, 0.02)
        if units is None:
            units = np.full(n, 1.0)
        if max_position_percents is None:
            max_position_percents = np.full(n, 0.3)
        if leverages is None:
            leverages = np.full(n, 1.0)
        if apply_dynamic_adjustments is None:
            apply_dynamic_adjustments = np.full(n, True, dtype=bool)
            
        # 結果配列の初期化
        position_sizes = np.zeros(n, dtype=np.float64)
        asset_quantities = np.zeros(n, dtype=np.float64)
        risk_amounts = np.zeros(n, dtype=np.float64)
        x_atr_values = np.zeros(n, dtype=np.float64)
        atr_multipliers = np.zeros(n, dtype=np.float64)
        trigger_values = np.zeros(n, dtype=np.float64)
        trigger_factors = np.zeros(n, dtype=np.float64)
        dynamic_risk_ratios = np.zeros(n, dtype=np.float64)
        
        # X_ATRとトリガーインジケーターのインスタンス作成（再利用）
        x_atr = XATR()
        
        if trigger_type == 'hyper_er':
            trigger_indicator = HyperER()
        elif trigger_type == 'hyper_trend_index':
            trigger_indicator = HyperTrendIndex()
        else:
            raise ValueError(f"サポートされていないトリガータイプ: {trigger_type}")
        
        # インプットデータの準備
        x_atrs = np.zeros(n, dtype=np.float64)
        multipliers = np.zeros(n, dtype=np.float64)
        dynamic_risks = np.zeros(n, dtype=np.float64)
        unit_coefficients = np.zeros(n, dtype=np.float64)
        
        # 各ポジションのデータを処理
        for i in range(n):
            history = historical_data_list[i]
            
            # トリガー値を計算
            try:
                if trigger_type == 'hyper_er':
                    trigger_result = trigger_indicator.calculate(history)
                    trigger_value = trigger_result.values[-1] if trigger_result is not None and len(trigger_result.values) > 0 else 0.5
                elif trigger_type == 'hyper_trend_index':
                    trigger_result = trigger_indicator.calculate(history)
                    trigger_value = trigger_result.values[-1] if trigger_result is not None and len(trigger_result.values) > 0 else 0.5
                else:
                    trigger_value = 0.5
            except Exception:
                trigger_value = 0.5
                
            # X_ATRを計算
            try:
                x_atr_result = x_atr.calculate(history)
                x_atr_value = x_atr_result.values[-1]
                
                # 値が無効な場合のみフォールバック処理
                if x_atr_value is None or np.isnan(x_atr_value):
                    # 価格の0.01%を使用（小さな価格の通貨でも適切に機能するため）
                    x_atr_value = entry_prices[i] * 0.0001
            except Exception:
                # 価格の0.01%をデフォルト値として使用
                x_atr_value = entry_prices[i] * 0.0001
                
            # トリガー値による調整
            if trigger_type == 'hyper_er':
                trigger_factor = 0.5 + trigger_value  # 0.5〜1.5の範囲
            elif trigger_type == 'hyper_trend_index':
                trigger_factor = 0.5 + trigger_value  # 0.5〜1.5の範囲
            else:
                trigger_factor = 1.0
                
            unit_coefficient = units[i]
            
            if apply_dynamic_adjustments[i]:
                unit_coefficient *= trigger_factor
                
                # トリガー値によるATR乗数の動的調整（簡素化）
                atr_multiplier = calculate_dynamic_multiplier_vec(
                    trigger_value,
                    6.0,  # max_multiplier
                    3.0   # min_multiplier
                )
                
                # トリガー値によるリスク比率の動的調整
                dynamic_risk_ratio = calculate_dynamic_risk_ratio(
                    trigger_value,
                    0.02,  # max_risk_ratio
                    0.005   # min_risk_ratio
                )
            else:
                atr_multiplier = 1.5  # デフォルト値
                dynamic_risk_ratio = base_risk_ratios[i]
                
            # 計算用配列に保存
            x_atrs[i] = x_atr_value
            multipliers[i] = atr_multiplier
            dynamic_risks[i] = dynamic_risk_ratio
            unit_coefficients[i] = unit_coefficient
            
            # 結果配列に保存
            x_atr_values[i] = x_atr_value
            atr_multipliers[i] = atr_multiplier
            trigger_values[i] = trigger_value
            trigger_factors[i] = trigger_factor
            dynamic_risk_ratios[i] = dynamic_risk_ratio
            
        # バッチ計算で高速にポジションサイズを計算
        position_sizes = calculate_batch_position_sizes_numba(
            capitals=capitals,
            risk_ratios=dynamic_risks,
            x_atrs=x_atrs,
            multipliers=multipliers,
            entry_prices=entry_prices,
            unit_coefficients=unit_coefficients,
            leverages=leverages,
            max_position_percents=max_position_percents
        )
        
        # 資産数量とリスク金額の計算
        for i in range(n):
            # 資産数量
            asset_quantities[i] = position_sizes[i] / entry_prices[i] if entry_prices[i] > 0 else 0
            
            # リスク金額
            x_atr_risk = x_atrs[i] * multipliers[i]
            risk_amounts[i] = position_sizes[i] * x_atr_risk / entry_prices[i]
            
        # 結果をまとめる
        return {
            'position_sizes': position_sizes,
            'asset_quantities': asset_quantities,
            'risk_amounts': risk_amounts,
            'x_atr_values': x_atr_values,
            'atr_multipliers': atr_multipliers,
            'trigger_values': trigger_values,
            'trigger_factors': trigger_factors,
            'dynamic_risk_ratios': dynamic_risk_ratios,
        }