#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import logging
from numba import njit, prange, vectorize
from position_sizing.position_sizing import PositionSizing, PositionSizingParams
from position_sizing.interfaces import IPositionManager
from indicators.z_atr import ZATR
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_multiplier_vec(cer: float, max_mult: float, min_mult: float) -> float:
    """
    サイクル効率比に基づいて動的なATR乗数を計算する（ベクトル化版）
    
    Args:
        cer: サイクル効率比の値
        max_mult: 最大乗数
        min_mult: 最小乗数
    
    Returns:
        動的な乗数の値
    """
    # CERが高い（トレンドが強い）ほど乗数は小さく、
    # CERが低い（トレンドが弱い）ほど乗数は大きくなる
    if np.isnan(cer):
        return (max_mult + min_mult) / 2  # デフォルト値
    cer_abs = abs(cer)
    return max_mult - cer_abs * (max_mult - min_mult)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_max_multiplier(cer: float, max_max_mult: float, min_max_mult: float) -> float:
    """
    サイクル効率比に基づいて動的な最大ATR乗数を計算する（ベクトル化版）
    
    Args:
        cer: サイクル効率比の値
        max_max_mult: 最大乗数の最大値（例：3.0）
        min_max_mult: 最大乗数の最小値（例：2.0）
    
    Returns:
        動的な最大乗数の値
    """
    if np.isnan(cer):
        return (max_max_mult + min_max_mult) / 2
    cer_abs = abs(cer)
    # CERが低い（トレンドが弱い）ほど最大乗数は大きく、
    # CERが高い（トレンドが強い）ほど最大乗数は小さくなる
    return max_max_mult - cer_abs * (max_max_mult - min_max_mult)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_min_multiplier(cer: float, max_min_mult: float, min_min_mult: float) -> float:
    """
    サイクル効率比に基づいて動的な最小ATR乗数を計算する（ベクトル化版）
    
    Args:
        cer: サイクル効率比の値
        max_min_mult: 最小乗数の最大値（例：1.5）
        min_min_mult: 最小乗数の最小値（例：0.5）
    
    Returns:
        動的な最小乗数の値
    """
    if np.isnan(cer):
        return (max_min_mult + min_min_mult) / 2
    cer_abs = abs(cer)
    # CERが低い（トレンドが弱い）ほど最小乗数は小さく、
    # CERが高い（トレンドが強い）ほど最小乗数は大きくなる
    return min_min_mult + cer_abs * (max_min_mult - min_min_mult)


@vectorize(['float64(float64, float64, float64)'], nopython=True, fastmath=True, cache=True)
def calculate_dynamic_risk_ratio(cer: float, max_risk: float, min_risk: float) -> float:
    """
    サイクル効率比に基づいて動的なリスク比率を計算する（ベクトル化版）
    
    Args:
        cer: サイクル効率比の値
        max_risk: 最大リスク比率（例：0.03）
        min_risk: 最小リスク比率（例：0.01）
    
    Returns:
        動的なリスク比率の値
    """
    if np.isnan(cer):
        return (max_risk + min_risk) / 2  # デフォルト値
    cer_abs = abs(cer)
    # CERが高い（トレンドが強い）ほどリスク比率は大きく、
    # CERが低い（トレンドが弱い）ほどリスク比率は小さくなる
    return min_risk + cer_abs * (max_risk - min_risk)


@njit(fastmath=True, cache=True)
def _calculate_position_size_numba(
    capital: float,
    risk_ratio: float,
    zatr: float,
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
        zatr: ZATR値
        multiplier: ZATR乗数
        entry_price: エントリー価格
        unit_coefficient: 単位係数（効率比で調整済み）
        leverage: レバレッジ
        max_position_percent: 最大ポジションの比率
        
    Returns:
        float: ポジションサイズ（USD建て）
    """
    # ポジションサイズの計算: 資本 × リスク比率 ÷ (ZATR × 乗数) × 価格 × 単位係数
    zatr_risk = zatr * multiplier
    position_size = capital * risk_ratio / (zatr_risk / entry_price) * unit_coefficient
    
    # レバレッジの適用
    position_size *= leverage
    
    # 最大ポジションサイズの制限を適用
    max_position = capital * max_position_percent * leverage
    if position_size > max_position:
        position_size = max_position
        
    return position_size


@njit(fastmath=True, cache=True)
def _calculate_risk_amount_numba(position_size: float, zatr: float, multiplier: float, entry_price: float) -> float:
    """
    リスク金額を計算する（Numba最適化版）
    
    Args:
        position_size: ポジションサイズ（USD建て）
        zatr: ZATR値
        multiplier: ZATR乗数
        entry_price: エントリー価格
        
    Returns:
        float: リスク金額（USD建て）
    """
    zatr_risk = zatr * multiplier
    return position_size * zatr_risk / entry_price


@njit(fastmath=True, parallel=True, cache=True)
def calculate_batch_position_sizes_numba(
    capitals: np.ndarray,
    risk_ratios: np.ndarray,
    zatrs: np.ndarray, 
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
        zatrs: ZATR値の配列
        multipliers: ZATR乗数の配列
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
        # ポジションサイズの計算: 資本 × リスク比率 ÷ (ZATR × 乗数) × 価格 × 単位係数
        zatr_risk = zatrs[i] * multipliers[i]
        position_size = capitals[i] * risk_ratios[i] / (zatr_risk / entry_prices[i]) * unit_coefficients[i]
        
        # レバレッジの適用
        position_size *= leverages[i]
        
        # 最大ポジションサイズの制限を適用
        max_position = capitals[i] * max_position_percents[i] * leverages[i]
        if position_size > max_position:
            position_size = max_position
            
        position_sizes[i] = position_size
        
    return position_sizes


class ZPositionSizing(PositionSizing, IPositionManager):
    """
    ZATRベースのポジションサイジング
    
    ZATRベースのポジションサイジングで、サイクル効率比（CER）に基づく
    動的リスク調整と乗数調整が可能
    """
    
    def __init__(
        self, 
        base_risk_ratio: float = 0.01,  # 基本リスク比率（デフォルト2%）
        unit: float = 1.0,              # 基本単位係数（デフォルト1.0）
        max_position_percent: float = 0.618,  # 最大ポジションサイズの比率（デフォルト50%）
        leverage: float = 1.0,          # レバレッジ（デフォルト1倍）
        zatr_detector_type: str = 'hody_e',  # ZATRの検出器タイプ
        zatr_max_period: int = 55,      # ZATR最大期間
        zatr_min_period: int = 5,       # ZATR最小期間
        apply_er_adjustment: bool = True,  # 効率比による調整を適用するか
        
        # 動的ATR乗数のパラメータ
        max_max_multiplier: float = 3.0,  # 最大乗数の最大値
        min_max_multiplier: float = 1.5,  # 最大乗数の最小値
        max_min_multiplier: float = 1.0,  # 最小乗数の最大値
        min_min_multiplier: float = 0.3,  # 最小乗数の最小値
        
        # 動的リスク比率のパラメータ
        max_risk_ratio: float = 0.03,   # 最大リスク比率（3%）
        min_risk_ratio: float = 0.003    # 最小リスク比率（0.3%）
    ):
        """
        初期化
        
        Args:
            base_risk_ratio: 基本リスク比率（資本に対する比率、例：0.02 = 2%）
            unit: 基本単位係数
            max_position_percent: 最大ポジションサイズの比率（資本に対する比率）
            leverage: レバレッジ
            zatr_detector_type: ZATRの検出器タイプ（hody, phac, dudi, dudi_e, hody_e, phac_e, dft）
            zatr_max_period: ZATR最大期間
            zatr_min_period: ZATR最小期間
            apply_er_adjustment: 効率比による調整を適用するか
            
            max_max_multiplier: 最大乗数の最大値
            min_max_multiplier: 最大乗数の最小値
            max_min_multiplier: 最小乗数の最大値
            min_min_multiplier: 最小乗数の最小値
            
            max_risk_ratio: 最大リスク比率
            min_risk_ratio: 最小リスク比率
        """
        super().__init__()
        self.base_risk_ratio = base_risk_ratio
        self.unit = unit
        self.max_position_percent = max_position_percent
        self.leverage = leverage
        self.apply_er_adjustment = apply_er_adjustment
        
        # 動的乗数のパラメータ
        self.max_max_multiplier = max_max_multiplier
        self.min_max_multiplier = min_max_multiplier
        self.max_min_multiplier = max_min_multiplier
        self.min_min_multiplier = min_min_multiplier
        
        # 動的リスク比率のパラメータ
        self.max_risk_ratio = max_risk_ratio
        self.min_risk_ratio = min_risk_ratio
        
        # ロガーの設定
        self.logger = logging.getLogger(__name__)
        
        # ZATR インスタンスを作成
        self.zatr = ZATR(
            detector_type=zatr_detector_type,
            max_dc_max_cycle=zatr_max_period,
            max_dc_min_cycle=zatr_min_period,
            smoother_type='alma'  # デフォルトのスムーサータイプ
        )
        
        # サイクル効率比（CER）インスタンスを作成
        self.cycle_er = CycleEfficiencyRatio(
            detector_type=zatr_detector_type,
            lp_period=5,
            hp_period=zatr_max_period,
            cycle_part=0.5
        )
    
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
            stop_loss_price=None,  # ATR計算が内部で行われるので不要
            capital=capital,
            leverage=self.leverage,
            risk_per_trade=self.base_risk_ratio  # 基本リスク比率を正しく渡す
        )
        
        result = self.calculate(params)
        return result['position_size']
    
    def calculate(self, params: PositionSizingParams) -> Dict[str, Any]:
        """
        ZATRベースのポジションサイズを計算
        
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
        
        # サイクル効率比（CER）を計算
        try:
            external_er = self.cycle_er.calculate(history)
            
            # 結果がNoneまたは空の配列の場合はフォールバックを使用
            if external_er is None or len(external_er) == 0:
                external_er = np.full(len(history), 0.5)
                self.logger.warning("CER計算結果が空のため、フォールバックCERを使用します")
                
        except Exception as e:
            # CER計算にエラーが発生した場合のフォールバック
            # すべての値が0.5のCERを作成（中立値）
            external_er = np.full(len(history), 0.5)
            self.logger.warning(f"CER計算中にエラー: {str(e)}、フォールバックCERを使用します")

        # ZATRを計算する
        try:
            self.zatr.calculate(history, external_er=external_er)
            zatr_value = self.zatr.get_absolute_atr()[-1]
            efficiency_ratio = self.zatr.get_efficiency_ratio()[-1]
            
            # 値が無効な場合のみフォールバック処理
            if zatr_value is None or np.isnan(zatr_value):
                # 価格の0.01%を使用（小さな価格の通貨でも適切に機能するため）
                zatr_value = params.entry_price * 0.0001
                self.logger.warning(f"ZATR値が無効（None/NaN）、価格の0.01%をデフォルト値として使用: {zatr_value}")
            
        except IndexError as e:
            # データが不足している場合のフォールバック
            zatr_value = params.entry_price * 0.0001  # 価格の0.01%
            efficiency_ratio = 0.5  # デフォルト値
            self.logger.warning(f"ZATR計算中にインデックスエラー: {str(e)}、価格の0.01%をデフォルト値として使用: {zatr_value}")
        except Exception as e:
            # その他のエラー
            zatr_value = params.entry_price * 0.0001  # 価格の0.01%
            efficiency_ratio = 0.5  # デフォルト値
            self.logger.warning(f"ZATR計算中にエラー: {str(e)}、価格の0.01%をデフォルト値として使用: {zatr_value}")

        # 単位係数のベース値
        unit_coefficient = self.unit
        
        # 効率比による調整
        er_factor = 1.0
        atr_multiplier = 1.5  # デフォルト値
        dynamic_risk_ratio = self.base_risk_ratio
        max_mult = 0.0
        min_mult = 0.0
        
        if self.apply_er_adjustment:
            # 効率比による単位係数の動的調整
            er_factor = self._calculate_er_factor(efficiency_ratio)
            unit_coefficient *= er_factor
            
            # 動的な最大・最小乗数の計算
            max_mult = calculate_dynamic_max_multiplier(
                efficiency_ratio,
                self.max_max_multiplier,
                self.min_max_multiplier
            )
            
            min_mult = calculate_dynamic_min_multiplier(
                efficiency_ratio,
                self.max_min_multiplier,
                self.min_min_multiplier
            )
            
            # 効率比によるATR乗数の動的調整
            atr_multiplier = calculate_dynamic_multiplier_vec(
                efficiency_ratio,
                max_mult,
                min_mult
            )
            
            # 効率比によるリスク比率の動的調整
            dynamic_risk_ratio = calculate_dynamic_risk_ratio(
                efficiency_ratio,
                self.max_risk_ratio,
                self.min_risk_ratio
            )
        else:
            # 調整なしの場合のデフォルト値
            atr_multiplier = 1.5  
            dynamic_risk_ratio = self.base_risk_ratio
            max_mult = self.max_max_multiplier
            min_mult = self.min_min_multiplier

        # Numba最適化関数を使用してポジションサイズを計算
        position_size_usd = _calculate_position_size_numba(
            capital=params.capital,
            risk_ratio=dynamic_risk_ratio,
            zatr=zatr_value,
            multiplier=atr_multiplier,
            entry_price=params.entry_price,
            unit_coefficient=unit_coefficient,
            leverage=self.leverage,
            max_position_percent=self.max_position_percent
        )

        # Numba最適化関数を使用してリスク金額を計算
        risk_amount = _calculate_risk_amount_numba(
            position_size=position_size_usd,
            zatr=zatr_value,
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
            'zatr_value': zatr_value,
            'atr_multiplier': atr_multiplier,
            'max_multiplier': max_mult,      # 動的最大乗数
            'min_multiplier': min_mult,      # 動的最小乗数
            'efficiency_ratio': efficiency_ratio,
            'er_factor': er_factor,
            'unit': self.unit,                     # 元のunit値
            'unit_with_er': unit_coefficient,      # 効率比調整後のunit値
            'risk_ratio': dynamic_risk_ratio,      # 動的リスク比率
            'max_position_size': max_position_size,
        }

    def _calculate_er_factor(self, efficiency_ratio: float) -> float:
        """
        効率比率から調整係数を計算
        
        Args:
            efficiency_ratio: 効率比率（0〜1の範囲）
            
        Returns:
            float: 調整係数
        """
        # 効率比が高い（トレンドが強い）ほど単位係数を大きくする
        # 効率比 0 → 係数 0.5
        # 効率比 0.5 → 係数 1.0
        # 効率比 1.0 → 係数 1.5
        return 0.5 + efficiency_ratio  # 0.5〜1.5の範囲 

    @staticmethod
    def calculate_batch(
        capitals: np.ndarray,
        entry_prices: np.ndarray,
        historical_data_list: list,
        base_risk_ratios: np.ndarray = None,
        units: np.ndarray = None,
        max_position_percents: np.ndarray = None,
        leverages: np.ndarray = None,
        apply_er_adjustments: np.ndarray = None,
        detector_type: str = 'hody'
    ) -> Dict[str, np.ndarray]:
        """
        バッチでZATRベースのポジションサイズを計算する
        
        Args:
            capitals: 資本額の配列
            entry_prices: エントリー価格の配列
            historical_data_list: 各ポジションの履歴データのリスト
            base_risk_ratios: 基本リスク比率の配列（省略時は0.02）
            units: 基本単位係数の配列（省略時は1.0）
            max_position_percents: 最大ポジションサイズの比率の配列（省略時は0.5）
            leverages: レバレッジの配列（省略時は1.0）
            apply_er_adjustments: 効率比による調整を適用するかの配列（省略時はTrue）
            detector_type: ZATRの検出器タイプ
            
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
            max_position_percents = np.full(n, 0.5)
        if leverages is None:
            leverages = np.full(n, 1.0)
        if apply_er_adjustments is None:
            apply_er_adjustments = np.full(n, True, dtype=bool)
            
        # 結果配列の初期化
        position_sizes = np.zeros(n, dtype=np.float64)
        asset_quantities = np.zeros(n, dtype=np.float64)
        risk_amounts = np.zeros(n, dtype=np.float64)
        zatr_values = np.zeros(n, dtype=np.float64)
        atr_multipliers = np.zeros(n, dtype=np.float64)
        efficiency_ratios = np.zeros(n, dtype=np.float64)
        er_factors = np.zeros(n, dtype=np.float64)
        dynamic_risk_ratios = np.zeros(n, dtype=np.float64)
        
        # ZATRとCERのインスタンス作成（再利用）
        zatr = ZATR(detector_type=detector_type)
        cycle_er = CycleEfficiencyRatio(detector_type=detector_type)
        
        # インプットデータの準備
        zatrs = np.zeros(n, dtype=np.float64)
        multipliers = np.zeros(n, dtype=np.float64)
        dynamic_risks = np.zeros(n, dtype=np.float64)
        unit_coefficients = np.zeros(n, dtype=np.float64)
        
        # 各ポジションのデータを処理
        for i in range(n):
            history = historical_data_list[i]
            
            # サイクル効率比（CER）を計算
            try:
                external_er = cycle_er.calculate(history)
                if external_er is None or len(external_er) == 0:
                    external_er = np.full(len(history), 0.5)
            except Exception:
                external_er = np.full(len(history), 0.5)
                
            # ZATRを計算
            try:
                zatr.calculate(history, external_er=external_er)
                zatr_value = zatr.get_absolute_atr()[-1]
                efficiency_ratio = zatr.get_efficiency_ratio()[-1]
                
                # 値が無効な場合のみフォールバック処理
                if zatr_value is None or np.isnan(zatr_value):
                    # 価格の0.01%を使用（小さな価格の通貨でも適切に機能するため）
                    zatr_value = entry_prices[i] * 0.0001
                    efficiency_ratio = 0.5
            except Exception:
                # 価格の0.01%をデフォルト値として使用
                zatr_value = entry_prices[i] * 0.0001
                efficiency_ratio = 0.5
                
            # 効率比による調整
            er_factor = 0.5 + efficiency_ratio  # 0.5〜1.5の範囲
            unit_coefficient = units[i]
            
            if apply_er_adjustments[i]:
                unit_coefficient *= er_factor
                
                # 動的な最大・最小乗数の計算
                max_mult = calculate_dynamic_max_multiplier(
                    efficiency_ratio,
                    2.5,  # max_max_multiplier
                    1.5   # min_max_multiplier
                )
                
                min_mult = calculate_dynamic_min_multiplier(
                    efficiency_ratio,
                    1.5,  # max_min_multiplier
                    0.5   # min_min_multiplier
                )
                
                # 効率比によるATR乗数の動的調整
                atr_multiplier = calculate_dynamic_multiplier_vec(
                    efficiency_ratio,
                    max_mult,
                    min_mult
                )
                
                # 効率比によるリスク比率の動的調整
                dynamic_risk_ratio = calculate_dynamic_risk_ratio(
                    efficiency_ratio,
                    0.03,  # max_risk_ratio
                    0.01   # min_risk_ratio
                )
            else:
                atr_multiplier = 1.5  # デフォルト値
                dynamic_risk_ratio = base_risk_ratios[i]
                
            # 計算用配列に保存
            zatrs[i] = zatr_value
            multipliers[i] = atr_multiplier
            dynamic_risks[i] = dynamic_risk_ratio
            unit_coefficients[i] = unit_coefficient
            
            # 結果配列に保存
            zatr_values[i] = zatr_value
            atr_multipliers[i] = atr_multiplier
            efficiency_ratios[i] = efficiency_ratio
            er_factors[i] = er_factor
            dynamic_risk_ratios[i] = dynamic_risk_ratio
            
        # バッチ計算で高速にポジションサイズを計算
        position_sizes = calculate_batch_position_sizes_numba(
            capitals=capitals,
            risk_ratios=dynamic_risks,
            zatrs=zatrs,
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
            zatr_risk = zatrs[i] * multipliers[i]
            risk_amounts[i] = position_sizes[i] * zatr_risk / entry_prices[i]
            
        # 結果をまとめる
        return {
            'position_sizes': position_sizes,
            'asset_quantities': asset_quantities,
            'risk_amounts': risk_amounts,
            'zatr_values': zatr_values,
            'atr_multipliers': atr_multipliers,
            'efficiency_ratios': efficiency_ratios,
            'er_factors': er_factors,
            'dynamic_risk_ratios': dynamic_risk_ratios,
        } 