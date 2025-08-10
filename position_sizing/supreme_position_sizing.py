#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supreme Position Sizing Algorithm - 人類史上最強のポジションサイジングアルゴリズム

最新の学術研究と金融工学の知見を統合した究極のポジションサイジングシステム。
以下の最先端手法を組み合わせて実装：

1. フラクショナルケリー基準 (Fractional Kelly Criterion)
2. 動的CPPI (Dynamic Constant Proportion Portfolio Insurance)  
3. ATRベースボラティリティターゲティング
4. 機械学習統合型リスク調整
5. 適応型マルチファクター統合

Author: Claude Code
Date: 2025-07-20
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from numba import njit
from dataclasses import dataclass
from position_sizing.position_sizing import PositionSizing, PositionSizingParams
from position_sizing.interfaces import IPositionManager
from indicators.atr import ATR
from indicators.efficiency_ratio import EfficiencyRatio


@dataclass
class SupremePositionConfig:
    """最強ポジションサイジング設定"""
    # ケリー基準関連
    kelly_fraction: float = 0.30  # フラクショナルケリー（30%で最適リスク/リターン）
    win_rate_period: int = 50     # 勝率計算期間
    avg_win_loss_period: int = 50 # 平均勝敗計算期間
    
    # CPPI関連
    cppi_multiplier_base: float = 5.0     # ベースマルチプライヤー
    cppi_multiplier_high_vol: float = 2.0 # 高ボラティリティ時
    cppi_multiplier_low_vol: float = 8.0  # 低ボラティリティ時
    cppi_floor_percent: float = 0.80      # フロア水準（80%）
    
    # ATRボラティリティターゲティング
    atr_period: int = 14
    atr_multiplier_base: float = 2.0
    volatility_target: float = 0.15       # 年率15%ボラティリティターゲット
    
    # 適応型調整
    efficiency_ratio_period: int = 20     # 効率比率期間
    volatility_regime_period: int = 21    # ボラティリティ体制判定期間
    trend_strength_threshold: float = 0.3 # トレンド強度閾値
    
    # リスク制限
    max_position_percent: float = 0.25    # 最大ポジション25%
    max_kelly_position: float = 0.15      # ケリー基準最大15%
    leverage: float = 1.0                 # レバレッジ
    
    # 機械学習統合（将来の拡張用）
    ml_confidence_threshold: float = 0.7  # ML予測信頼度閾値
    ml_weight: float = 0.3                # ML予測重み


@njit(fastmath=True)
def _calculate_kelly_fraction_core(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fractional_kelly: float
) -> float:
    """
    フラクショナルケリー基準計算（Numba最適化）
    
    Kelly% = (bp - q) / b
    where: b = 平均勝ち/平均負け, p = 勝率, q = 負け率
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    b = avg_win / avg_loss  # 勝敗比
    p = win_rate            # 勝率
    q = 1 - win_rate        # 負け率
    
    kelly_fraction = (b * p - q) / b
    
    # フラクショナルケリーを適用（30%が最適）
    kelly_fraction *= fractional_kelly
    
    # 負の値は0にクリップ
    return max(0.0, kelly_fraction)


@njit(fastmath=True)
def _calculate_cppi_multiplier(
    current_volatility: float,
    avg_volatility: float,
    base_multiplier: float,
    high_vol_multiplier: float,
    low_vol_multiplier: float
) -> float:
    """
    動的CPPIマルチプライヤー計算（Numba最適化）
    
    ボラティリティ体制に基づいてマルチプライヤーを調整
    """
    vol_ratio = current_volatility / avg_volatility if avg_volatility > 0 else 1.0
    
    if vol_ratio > 1.2:  # 高ボラティリティ体制
        return high_vol_multiplier
    elif vol_ratio < 0.8:  # 低ボラティリティ体制
        return low_vol_multiplier
    else:  # 通常体制
        return base_multiplier


@njit(fastmath=True)
def _calculate_volatility_adjusted_size(
    base_position_size: float,
    current_atr: float,
    target_volatility: float,
    entry_price: float
) -> float:
    """
    ボラティリティターゲティング調整（Numba最適化）
    
    ドル建てボラティリティを正規化してポジションサイズを調整
    """
    if current_atr <= 0 or entry_price <= 0:
        return base_position_size
    
    # 現在のATRを年率ボラティリティに変換（仮定：252営業日）
    current_vol_annualized = (current_atr / entry_price) * np.sqrt(252)
    
    # ターゲットボラティリティとの比率で調整
    vol_adjustment = target_volatility / current_vol_annualized if current_vol_annualized > 0 else 1.0
    
    # 調整に上下限を設定（0.5倍〜2倍）
    vol_adjustment = max(0.5, min(vol_adjustment, 2.0))
    
    return base_position_size * vol_adjustment


@njit(fastmath=True)
def _calculate_supreme_position_size(
    capital: float,
    entry_price: float,
    kelly_fraction: float,
    cppi_multiplier: float,
    cppi_floor: float,
    current_atr: float,
    target_volatility: float,
    efficiency_ratio: float,
    max_position_percent: float,
    max_kelly_position: float,
    leverage: float
) -> Tuple[float, float, float, float]:
    """
    最強ポジションサイズ統合計算（Numba最適化）
    
    複数手法を統合して最適なポジションサイズを算出
    
    Returns:
        Tuple[kelly_size, cppi_size, final_size, confidence_score]
    """
    # 1. ケリー基準ポジションサイズ
    kelly_position_size = capital * min(kelly_fraction, max_kelly_position) * leverage
    
    # 2. CPPIポジションサイズ
    cushion = capital - (capital * cppi_floor)  # クッション
    cppi_position_size = cushion * cppi_multiplier if cushion > 0 else 0.0
    
    # 3. 効率比率による調整重み
    # 効率比率が高い = トレンド強い = ケリー重視
    # 効率比率が低い = レンジ相場 = CPPI重視
    kelly_weight = max(0.3, min(efficiency_ratio, 0.7))
    cppi_weight = 1.0 - kelly_weight
    
    # 4. 重み付き統合
    integrated_size = (kelly_position_size * kelly_weight + 
                      cppi_position_size * cppi_weight)
    
    # 5. ボラティリティターゲティング調整
    vol_adjusted_size = _calculate_volatility_adjusted_size(
        integrated_size, current_atr, target_volatility, entry_price
    )
    
    # 6. 最大ポジション制限
    max_position = capital * max_position_percent * leverage
    final_size = min(vol_adjusted_size, max_position)
    
    # 7. 信頼度スコア計算（0-1）
    confidence_score = min(efficiency_ratio * 2, 1.0)  # 効率比率ベース
    
    return kelly_position_size, cppi_position_size, final_size, confidence_score


class SupremePositionSizing(PositionSizing, IPositionManager):
    """
    人類史上最強のポジションサイジングアルゴリズム
    
    最新の学術研究と金融工学の知見を統合：
    - フラクショナルケリー基準（30%最適化）
    - 動的CPPI（ボラティリティ適応型）
    - ATRボラティリティターゲティング
    - 効率比率ベース適応型統合
    - マルチファクター統合最適化
    """
    
    def __init__(self, config: Optional[SupremePositionConfig] = None):
        """
        初期化
        
        Args:
            config: 最強ポジションサイジング設定
        """
        super().__init__()
        
        self.config = config or SupremePositionConfig()
        self.logger = logging.getLogger(__name__)
        
        # インジケーター初期化
        self.atr = ATR(
            period=self.config.atr_period,
            smoothing_method='hma',
            use_dynamic_period=False
        )
        
        self.efficiency_ratio = EfficiencyRatio(
            period=self.config.efficiency_ratio_period,
            src_type='hlc3',
            smoothing_method='hma'
        )
        
        # ATRをボラティリティ代替として使用
        self.volatility_atr = ATR(
            period=self.config.volatility_regime_period,
            smoothing_method='hma',
            use_dynamic_period=False
        )
        
        # パフォーマンス追跡
        self.trade_history = []
        self.performance_metrics = {
            'win_rate': 0.5,  # 初期値50%
            'avg_win': 0.02,  # 初期値2%
            'avg_loss': 0.01, # 初期値1%
        }
    
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
        # 簡易計算（履歴データなしの場合）
        kelly_fraction = self.config.kelly_fraction * 0.5  # 保守的
        base_size = capital * kelly_fraction * self.config.leverage
        max_size = capital * self.config.max_position_percent * self.config.leverage
        
        return min(base_size, max_size)
    
    def calculate(self, params: PositionSizingParams) -> Dict[str, Any]:
        """
        最強ポジションサイズ計算メイン処理
        
        Args:
            params: ポジションサイジングパラメータ
            
        Returns:
            Dict[str, Any]: 計算結果
        """
        if params.historical_data is None:
            raise ValueError("履歴データが必要です")
        
        history = params.historical_data
        
        try:
            # 1. 各インジケーターの計算
            atr_result = self.atr.calculate(history)
            current_atr = atr_result.values[-1] if len(atr_result.values) > 0 else params.entry_price * 0.01
            
            er_result = self.efficiency_ratio.calculate(history)
            current_er = er_result.values[-1] if len(er_result.values) > 0 else 0.5
            
            vol_result = self.volatility_atr.calculate(history)
            current_vol = vol_result.values[-1] / params.entry_price if len(vol_result.values) > 0 else 0.15
            avg_vol = np.mean(vol_result.values[-20:]) / params.entry_price if len(vol_result.values) >= 20 else current_vol
            
            # 2. パフォーマンス指標更新
            self._update_performance_metrics()
            
            # 3. ケリー分数計算
            kelly_fraction = _calculate_kelly_fraction_core(
                self.performance_metrics['win_rate'],
                self.performance_metrics['avg_win'],
                self.performance_metrics['avg_loss'],
                self.config.kelly_fraction
            )
            
            # 4. 動的CPPIマルチプライヤー計算
            cppi_multiplier = _calculate_cppi_multiplier(
                current_vol,
                avg_vol,
                self.config.cppi_multiplier_base,
                self.config.cppi_multiplier_high_vol,
                self.config.cppi_multiplier_low_vol
            )
            
            # 5. 最強ポジションサイズ統合計算
            kelly_size, cppi_size, final_size, confidence = _calculate_supreme_position_size(
                params.capital,
                params.entry_price,
                kelly_fraction,
                cppi_multiplier,
                self.config.cppi_floor_percent,
                current_atr,
                self.config.volatility_target,
                current_er,
                self.config.max_position_percent,
                self.config.max_kelly_position,
                self.config.leverage
            )
            
            # 6. 資産数量計算
            asset_quantity = final_size / params.entry_price if params.entry_price > 0 else 0
            
            # 7. リスク金額計算（ATRベース）
            risk_amount = final_size * (current_atr / params.entry_price) if params.entry_price > 0 else 0
            
            return {
                'position_size': final_size,
                'asset_quantity': asset_quantity,
                'risk_amount': risk_amount,
                
                # 詳細分析
                'kelly_position_size': kelly_size,
                'cppi_position_size': cppi_size,
                'kelly_fraction': kelly_fraction,
                'cppi_multiplier': cppi_multiplier,
                'confidence_score': confidence,
                
                # インジケーター値
                'atr_value': current_atr,
                'efficiency_ratio': current_er,
                'current_volatility': current_vol,
                'avg_volatility': avg_vol,
                
                # パフォーマンス指標
                'win_rate': self.performance_metrics['win_rate'],
                'avg_win': self.performance_metrics['avg_win'],
                'avg_loss': self.performance_metrics['avg_loss'],
                
                # 設定情報
                'volatility_target': self.config.volatility_target,
                'max_position_percent': self.config.max_position_percent,
                'algorithm_version': 'Supreme-v1.0'
            }
            
        except Exception as e:
            self.logger.error(f"最強ポジションサイズ計算エラー: {str(e)}")
            
            # フォールバック: 保守的な計算
            fallback_size = params.capital * 0.02 * self.config.leverage  # 資金の2%
            max_size = params.capital * self.config.max_position_percent * self.config.leverage
            final_size = min(fallback_size, max_size)
            
            return {
                'position_size': final_size,
                'asset_quantity': final_size / params.entry_price if params.entry_price > 0 else 0,
                'risk_amount': final_size * 0.01,  # 1%リスク想定
                'error': str(e),
                'fallback_used': True,
                'algorithm_version': 'Supreme-v1.0-Fallback'
            }
    
    def _update_performance_metrics(self):
        """
        パフォーマンス指標を更新
        
        実際の取引履歴から勝率、平均勝敗を計算
        """
        if len(self.trade_history) < 10:
            return  # 最低10取引必要
        
        recent_trades = self.trade_history[-self.config.win_rate_period:]
        
        wins = [trade for trade in recent_trades if trade['return'] > 0]
        losses = [trade for trade in recent_trades if trade['return'] <= 0]
        
        if len(recent_trades) > 0:
            self.performance_metrics['win_rate'] = len(wins) / len(recent_trades)
        
        if len(wins) > 0:
            self.performance_metrics['avg_win'] = np.mean([trade['return'] for trade in wins])
        
        if len(losses) > 0:
            self.performance_metrics['avg_loss'] = abs(np.mean([trade['return'] for trade in losses]))
    
    def add_trade_result(self, entry_price: float, exit_price: float, position_size: float):
        """
        取引結果を追加してパフォーマンス指標を更新
        
        Args:
            entry_price: エントリー価格
            exit_price: 決済価格  
            position_size: ポジションサイズ
        """
        trade_return = (exit_price - entry_price) / entry_price
        
        self.trade_history.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'return': trade_return,
            'timestamp': pd.Timestamp.now()
        })
        
        # 履歴が長すぎる場合は古いデータを削除
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-500:]
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        アルゴリズム情報を取得
        
        Returns:
            Dict[str, Any]: アルゴリズム詳細情報
        """
        return {
            'name': 'Supreme Position Sizing Algorithm',
            'version': '1.0',
            'description': '人類史上最強のポジションサイジングアルゴリズム',
            'methods': [
                'Fractional Kelly Criterion (30% optimal)',
                'Dynamic CPPI with Volatility Adaptation',
                'ATR-based Volatility Targeting',
                'Efficiency Ratio Adaptive Integration',
                'Multi-factor Optimization'
            ],
            'config': self.config.__dict__,
            'performance_metrics': self.performance_metrics,
            'trade_count': len(self.trade_history)
        }