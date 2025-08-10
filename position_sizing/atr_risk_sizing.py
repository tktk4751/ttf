#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any
import numpy as np
import pandas as pd
import logging
from position_sizing.position_sizing import PositionSizing, PositionSizingParams
from position_sizing.interfaces import IPositionManager
from indicators.alpha_atr import AlphaATR
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio


class AlphaATRRiskSizing(PositionSizing, IPositionManager):
    """
    Alpha ATRベースのポジションサイジング
    
    ATRベースのポジションサイジングで、サイクル効率比（CER）に基づく動的調整が可能
    """
    
    def __init__(
        self, 
        risk_ratio: float = 0.01,  # リスク比率（デフォルト1%）
        unit: float = 1.0,         # 基本単位係数（デフォルト1.0）
        max_position_percent: float = 0.5,  # 最大ポジションサイズの比率（デフォルト50%）
        leverage: float = 1.0,     # レバレッジ（デフォルト1倍）
        atr_period: int = 55,      # ATR期間（デフォルト14）
        apply_er_adjustment: bool = True  # 効率比による調整を適用するか
    ):
        """
        初期化
        
        Args:
            risk_ratio: リスク比率（資本に対する比率、例：0.01 = 1%）
            unit: 基本単位係数
            max_position_percent: 最大ポジションサイズの比率（資本に対する比率）
            leverage: レバレッジ
            atr_period: ATR計算期間
            apply_er_adjustment: 効率比による調整を適用するか
        """
        super().__init__()
        self.risk_ratio = risk_ratio
        self.unit = unit
        self.max_position_percent = max_position_percent
        self.leverage = leverage
        self.period = atr_period
        self.max_atr_period = atr_period
        self.apply_er_adjustment = apply_er_adjustment
        
        # ロガーの設定
        self.logger = logging.getLogger(__name__)
        
        # Alpha ATRインスタンスを作成
        self.alpha_atr = AlphaATR(
            max_atr_period=atr_period,
            min_atr_period=max(3, atr_period // 3),  # 最小期間はATR期間の1/3（最小値は3）
            smoother_type='alma'  # デフォルトのスムーサータイプ
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
            risk_per_trade=self.risk_ratio  # リスク比率を正しく渡す
        )
        
        result = self.calculate(params)
        return result['position_size']
    
    def calculate(self, params: PositionSizingParams) -> Dict[str, Any]:
        """
        Alpha ATRベースのポジションサイズを計算
        
        Args:
            params: ポジションサイジングパラメータ
            
        Returns:
            Dict[str, Any]: 計算結果
        """
        # History要件の確認
        if params.historical_data is None or len(params.historical_data) < self.max_atr_period:
            raise ValueError(f"最低 {self.max_atr_period} 本の履歴データが必要です")
            
        # AlphaATRを計算し、最新のATR値を取得
        # 注意: AlphaATRはCER（Cycle Efficiency Ratio）が必須になっています
        history = params.historical_data
        
        # サイクル効率比（CER）を計算
        try:
            # CycleEfficiencyRatioのインスタンス化とCERの計算
            cycle_er = CycleEfficiencyRatio(
                detector_type='hody',
                lp_period=5,
                hp_period=144,
                cycle_part=0.5
            )
            external_er = cycle_er.calculate(history)
            
            # 結果がNoneまたは空の配列の場合はフォールバックを使用
            if external_er is None or len(external_er) == 0:
                external_er = np.full(len(history), 0.5)
                self.logger.warning("CER計算結果が空のため、フォールバックCERを使用します")
                
        except Exception as e:
            # CER計算にエラーが発生した場合のフォールバック
            # すべての値が0.5のCERを作成（中立値）
            external_er = np.full(len(history), 0.5)
            self.logger.warning(f"CER計算中にエラー: {str(e)}、フォールバックCERを使用します")

        if self.alpha_atr:
            try:
                self.alpha_atr.calculate(history, external_er=external_er)
                atr_value = self.alpha_atr.get_absolute_atr()[-1]
                efficiency_ratio = self.alpha_atr.get_efficiency_ratio()[-1]
                
                # ATR値が0または非常に小さい場合のフォールバック
                if atr_value is None or np.isnan(atr_value) or atr_value <= 0.000001:
                    # 価格の1%をデフォルトATR値として使用
                    atr_value = params.entry_price * 0.01
                    self.logger.warning(f"ATR値が無効（{atr_value}）、価格の1%をデフォルト値として使用: {atr_value}")
                
            except IndexError as e:
                # データが不足している場合のフォールバック
                # 価格の1%をデフォルトATR値として使用
                atr_value = params.entry_price * 0.01
                efficiency_ratio = 0.5  # デフォルト値
                self.logger.warning(f"ATR計算中にインデックスエラー: {str(e)}、デフォルトATR値を使用: {atr_value}")
            except Exception as e:
                # その他のエラー
                atr_value = params.entry_price * 0.01
                efficiency_ratio = 0.5  # デフォルト値
                self.logger.warning(f"ATR計算中にエラー: {str(e)}、デフォルトATR値を使用: {atr_value}")
        else:
            # AlphaATRが設定されていない場合
            atr_value = params.entry_price * 0.01  # 価格の1%をデフォルト値として使用
            efficiency_ratio = 0.5  # デフォルト値

        # 単位係数のベース値
        unit_coefficient = self.unit
        er_factor = 1.0

        # 効率比による調整（十分なデータがある場合のみ）
        if self.apply_er_adjustment and self.alpha_atr and len(history) >= self.max_atr_period:
            # 効率比による単位係数の動的調整
            er_factor = self._calculate_er_factor(efficiency_ratio)
            unit_coefficient *= er_factor

        # ポジションサイズの計算: 資本 × リスク比率 ÷ ATR × 価格 × 単位係数
        position_size_usd = params.capital * params.risk_per_trade / atr_value * params.entry_price * unit_coefficient

        # レバレッジの適用（パラメータからではなく、インスタンス変数を使用）
        position_size_usd *= self.leverage

        # 最大ポジションサイズの制限を適用
        max_position_size = params.capital * self.max_position_percent * self.leverage
        if position_size_usd > max_position_size:
            position_size_usd = max_position_size

        # リスク金額の計算：ポジションサイズ × ATR
        risk_amount = position_size_usd * atr_value / params.entry_price
        
        # 資産数量を計算（表示用）
        asset_quantity = position_size_usd / params.entry_price if params.entry_price > 0 else 0


        # 戻り値の構築
        return {
            'position_size': position_size_usd,
            'asset_quantity': asset_quantity,    # 資産数量（単位数）
            'risk_amount': risk_amount,
            'atr_value': atr_value,
            'efficiency_ratio': efficiency_ratio,
            'er_factor': er_factor,
            'unit': self.unit,           # 元のunit値を返す（効率比調整前）
            'unit_with_er': unit_coefficient,  # 効率比調整後のunit値
            'risk_ratio': params.risk_per_trade,  # 使用したリスク比率を明示的に含める
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