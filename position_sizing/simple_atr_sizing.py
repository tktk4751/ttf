#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import logging
from numba import njit
from position_sizing.position_sizing import PositionSizing, PositionSizingParams
from position_sizing.interfaces import IPositionManager
from indicators.atr import ATR


@njit(fastmath=True)
def _calculate_position_size_simple_atr(
    capital: float,
    risk_ratio: float,
    atr: float,
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
        atr: ATR値
        multiplier: ATR乗数
        entry_price: エントリー価格
        unit_coefficient: 単位係数
        leverage: レバレッジ
        max_position_percent: 最大ポジションの比率
        
    Returns:
        float: ポジションサイズ（USD建て）
    """
    # ポジションサイズの計算: 資本 × リスク比率 ÷ (ATR × 乗数) × 価格 × 単位係数
    atr_risk = atr * multiplier
    position_size = capital * risk_ratio / (atr_risk / entry_price) * unit_coefficient
    
    # レバレッジの適用
    position_size *= leverage
    
    # 最大ポジションサイズの制限を適用
    max_position = capital * max_position_percent * leverage
    if position_size > max_position:
        position_size = max_position
        
    return position_size


@njit(fastmath=True)
def _calculate_risk_amount_simple_atr(position_size: float, atr: float, multiplier: float, entry_price: float) -> float:
    """
    リスク金額を計算する（Numba最適化版）
    
    Args:
        position_size: ポジションサイズ（USD建て）
        atr: ATR値
        multiplier: ATR乗数
        entry_price: エントリー価格
        
    Returns:
        float: リスク金額（USD建て）
    """
    atr_risk = atr * multiplier
    return position_size * atr_risk / entry_price


class SimpleATRPositionSizing(PositionSizing, IPositionManager):
    """
    シンプルなATRベースのポジションサイジング
    
    固定パラメータでATRベースのポジションサイジングを行う
    複雑なインジケータは使用せず、シンプルで高速な実装
    """
    
    def __init__(
        self, 
        base_risk_ratio: float = 0.02,  # 基本リスク比率（デフォルト1%）
        unit: float = 1.0,              # 基本単位係数（デフォルト1.0）
        max_position_percent: float = 0.3,  # 最大ポジションサイズの比率（デフォルト30%）
        leverage: float = 1.0,          # レバレッジ（デフォルト1倍）
        atr_period: int = 13,           # ATR期間（デフォルト14）
        atr_multiplier: float = 3.0,    # ATR乗数（デフォルト2.0）
        smoothing_method: str = 'hma',  # ATRスムージング方法
        fixed_risk_percent: float = 0.02,  # 固定リスク率（資金の1%）
    ):
        """
        初期化
        
        Args:
            base_risk_ratio: 基本リスク比率（資本に対する比率、例：0.01 = 1%）
            unit: 基本単位係数
            max_position_percent: 最大ポジションサイズの比率（資本に対する比率）
            leverage: レバレッジ
            atr_period: ATR計算期間
            atr_multiplier: ATR乗数（固定値）
            smoothing_method: ATRスムージング方法
            fixed_risk_percent: 固定リスク率（資金の何%をリスクとするか）
        """
        super().__init__()
        
        # 全パラメータをインスタンス変数として保存
        self.base_risk_ratio = base_risk_ratio
        self.unit = unit
        self.max_position_percent = max_position_percent
        self.leverage = leverage
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.smoothing_method = smoothing_method
        self.fixed_risk_percent = fixed_risk_percent
        
        # ロガーの設定
        self.logger = logging.getLogger(__name__)
        
        # ATR インスタンスを作成
        self.atr = ATR(
            period=atr_period,
            smoothing_method=smoothing_method,
            use_dynamic_period=False,  # 固定期間を使用
            slope_index=1,
            range_threshold=0.005
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
            risk_per_trade=self.base_risk_ratio
        )
        
        result = self.calculate(params)
        return result['position_size']
    
    def calculate(self, params: PositionSizingParams) -> Dict[str, Any]:
        """
        ATRベースのポジションサイズを計算
        
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

        # ATRを計算する
        try:
            atr_result = self.atr.calculate(history)
            atr_value = atr_result.values[-1]
            
            # 値が無効な場合のみフォールバック処理
            if atr_value is None or np.isnan(atr_value):
                # 価格の1%を使用
                atr_value = params.entry_price * 0.01
                self.logger.warning(f"ATR値が無効（None/NaN）、価格の1%をデフォルト値として使用: {atr_value}")
            
        except IndexError as e:
            # データが不足している場合のフォールバック
            atr_value = params.entry_price * 0.01  # 価格の1%
            self.logger.warning(f"ATR計算中にインデックスエラー: {str(e)}、価格の1%をデフォルト値として使用: {atr_value}")
        except Exception as e:
            # その他のエラー
            atr_value = params.entry_price * 0.01  # 価格の1%
            self.logger.warning(f"ATR計算中にエラー: {str(e)}、価格の1%をデフォルト値として使用: {atr_value}")

        # 単位係数のベース値（固定）
        unit_coefficient = self.unit
        
        # 固定のATR乗数とリスク比率を使用
        atr_multiplier = self.atr_multiplier
        dynamic_risk_ratio = self.base_risk_ratio

        # Numba最適化関数を使用してポジションサイズを計算
        position_size_usd = _calculate_position_size_simple_atr(
            capital=params.capital,
            risk_ratio=dynamic_risk_ratio,
            atr=atr_value,
            multiplier=atr_multiplier,
            entry_price=params.entry_price,
            unit_coefficient=unit_coefficient,
            leverage=self.leverage,
            max_position_percent=self.max_position_percent
        )

        # Numba最適化関数を使用してリスク金額を計算
        risk_amount = _calculate_risk_amount_simple_atr(
            position_size=position_size_usd,
            atr=atr_value,
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
            'atr_value': atr_value,
            'atr_multiplier': atr_multiplier,
            'unit': self.unit,                     # 元のunit値
            'unit_coefficient': unit_coefficient,  # 単位係数
            'risk_ratio': dynamic_risk_ratio,      # リスク比率
            'max_position_size': max_position_size,
            'atr_period': self.atr_period,
            'smoothing_method': self.smoothing_method,
        }

    def calculate_stop_loss_price(
        self, 
        entry_price: float, 
        atr_value: float, 
        atr_multiplier: float, 
        is_long: bool = True
    ) -> float:
        """
        ATR乗数に基づいてストップロス価格を計算
        
        Args:
            entry_price: エントリー価格
            atr_value: ATR値
            atr_multiplier: ATR乗数
            is_long: ロングポジションかどうか（True=ロング、False=ショート）
            
        Returns:
            float: ストップロス価格
        """
        # ATRリスク（値幅）を計算
        atr_risk = atr_value * atr_multiplier
        
        # ポジションタイプに基づいてストップロス価格を計算
        if is_long:
            # ロングの場合はエントリー価格からATRリスク分を引く
            stop_loss = entry_price - atr_risk
        else:
            # ショートの場合はエントリー価格にATRリスク分を足す
            stop_loss = entry_price + atr_risk
            
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

        # ATRとATR乗数を直接計算して取得する
        try:
            atr_result = self.atr.calculate(history)
            atr_value = atr_result.values[-1]
            
            # 値が無効な場合のみフォールバック処理
            if atr_value is None or np.isnan(atr_value):
                # 価格の1%を使用
                atr_value = entry_price * 0.01
                self.logger.warning(f"ATR値が無効（None/NaN）、価格の1%をデフォルト値として使用: {atr_value}")
            
        except IndexError as e:
            # データが不足している場合のフォールバック
            atr_value = entry_price * 0.01  # 価格の1%
            self.logger.warning(f"ATR計算中にインデックスエラー: {str(e)}、価格の1%をデフォルト値として使用: {atr_value}")
        except Exception as e:
            # その他のエラー
            atr_value = entry_price * 0.01  # 価格の1%
            self.logger.warning(f"ATR計算中にエラー: {str(e)}、価格の1%をデフォルト値として使用: {atr_value}")

        # 固定のATR乗数を使用
        atr_multiplier = self.atr_multiplier

        # ストップロス価格の計算
        stop_loss_price = self.calculate_stop_loss_price(
            entry_price=entry_price,
            atr_value=atr_value,
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
            self.logger.info(f"ATR値: {atr_value}")
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
            'atr_value': atr_value,
            'atr_multiplier': atr_multiplier,
            'max_position_size': max_position_size,
            'atr_period': self.atr_period,
            'smoothing_method': self.smoothing_method,
        }
        
        return result 