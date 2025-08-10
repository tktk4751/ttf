#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Kalman Enhanced Smoothers - カルマン強化スムーサー** 🎯

スムーサーとカルマンフィルターを組み合わせて、
さらに高度なノイズ除去とトレンド追従を実現します。

📊 **機能:**
- ALMA + Kalman Filter
- HMA + Kalman Filter
- その他のスムーサー + Kalman Filter
- 動的パラメータ調整
- パフォーマンス最適化

🔧 **使用例:**
```python
# ALMA + Simple Kalman
smoother = KalmanEnhancedSmoother('alma', 'simple')
result = smoother.calculate(data)

# HMA + Quantum Adaptive Kalman
smoother = KalmanEnhancedSmoother('hma', 'quantum_adaptive')
result = smoother.calculate(data)
```
"""

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd

try:
    from ..indicator import Indicator
    from ..price_source import PriceSource
    from .unified_smoother import UnifiedSmoother
    from ..kalman.unified_kalman import UnifiedKalman
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from indicators.indicator import Indicator
    from indicators.price_source import PriceSource
    from indicators.smoother.unified_smoother import UnifiedSmoother
    from indicators.kalman.unified_kalman import UnifiedKalman


@dataclass
class KalmanEnhancedResult:
    """カルマン強化スムーサーの計算結果"""
    values: np.ndarray                      # 最終強化済み値
    smoother_values: np.ndarray             # スムーサー単独値
    kalman_values: np.ndarray               # カルマンフィルター単独値
    raw_values: np.ndarray                  # 元の価格データ
    smoother_type: str                      # 使用されたスムーサータイプ
    kalman_type: str                        # 使用されたカルマンタイプ
    smoother_params: Dict[str, Any]         # スムーサーパラメータ
    kalman_params: Dict[str, Any]           # カルマンパラメータ
    additional_data: Dict[str, np.ndarray]  # 追加データ


class KalmanEnhancedSmoother(Indicator):
    """
    カルマン強化スムーサー
    
    スムーサーの出力をカルマンフィルターでさらに強化。
    二段階のノイズ除去により、最高品質のスムージング結果を提供。
    """
    
    def __init__(
        self,
        smoother_type: str = 'alma',
        kalman_type: str = 'simple',
        src_type: str = 'close',
        combination_mode: str = 'sequential',
        **kwargs
    ):
        """
        コンストラクタ
        
        Args:
            smoother_type: スムーサータイプ
            kalman_type: カルマンフィルタータイプ
            src_type: 価格ソース
            combination_mode: 組み合わせモード ('sequential', 'weighted', 'adaptive')
            **kwargs: 各コンポーネントのパラメータ
        """
        super().__init__(f"KalmanEnhanced({smoother_type}+{kalman_type})")
        
        self.smoother_type = smoother_type.lower()
        self.kalman_type = kalman_type.lower()
        self.src_type = src_type.lower()
        self.combination_mode = combination_mode.lower()
        
        # パラメータ分離
        smoother_params = {}
        kalman_params = {}
        
        for key, value in kwargs.items():
            if key.startswith('smoother_'):
                smoother_params[key[9:]] = value  # 'smoother_' プレフィックスを除去
            elif key.startswith('kalman_'):
                kalman_params[key[7:]] = value    # 'kalman_' プレフィックスを除去
            else:
                # デフォルトはスムーサーパラメータ
                smoother_params[key] = value
        
        # コンポーネント作成
        self.smoother = UnifiedSmoother(
            smoother_type=self.smoother_type,
            src_type=self.src_type,
            **smoother_params
        )
        
        self.kalman = UnifiedKalman(
            filter_type=self.kalman_type,
            src_type=self.src_type,
            **kalman_params
        )
        
        # 内部状態
        self._last_result = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> KalmanEnhancedResult:
        """
        カルマン強化スムージングを計算
        
        Args:
            data: 価格データ
            
        Returns:
            KalmanEnhancedResult: 計算結果
        """
        try:
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            data_length = len(src_prices)
            
            if data_length < 2:
                return self._create_empty_result(data_length, src_prices)
            
            # ステップ1: スムーサーの計算
            smoother_result = self.smoother.calculate(data)
            smoother_values = smoother_result.values
            
            # ステップ2: カルマンフィルターの適用
            if self.combination_mode == 'sequential':
                # シーケンシャルモード: スムーサー → カルマン
                enhanced_values = self._apply_sequential_filtering(
                    data, smoother_values
                )
            elif self.combination_mode == 'weighted':
                # 重み付きモード: 両方を並行実行して重み付き平均
                enhanced_values = self._apply_weighted_combination(
                    data, smoother_values
                )
            elif self.combination_mode == 'adaptive':
                # 適応モード: 市況に応じて動的に組み合わせ
                enhanced_values = self._apply_adaptive_combination(
                    data, smoother_values
                )
            else:
                # デフォルトはシーケンシャル
                enhanced_values = self._apply_sequential_filtering(
                    data, smoother_values
                )
            
            # カルマン単独計算（比較用）
            kalman_result = self.kalman.calculate(data)
            kalman_values = kalman_result.values
            
            # 追加データの統合
            additional_data = {}
            additional_data.update(smoother_result.additional_data)
            additional_data.update(kalman_result.additional_data)
            additional_data['combination_mode'] = np.full(data_length, hash(self.combination_mode))
            
            # 結果の作成
            result = KalmanEnhancedResult(
                values=enhanced_values,
                smoother_values=smoother_values,
                kalman_values=kalman_values,
                raw_values=src_prices.copy(),
                smoother_type=self.smoother_type,
                kalman_type=self.kalman_type,
                smoother_params=smoother_result.parameters,
                kalman_params=kalman_result.parameters,
                additional_data=additional_data
            )
            
            # 基底クラス用の値設定
            self._values = enhanced_values
            self._last_result = result
            
            return result
            
        except Exception as e:
            error_msg = f"カルマン強化スムーサー計算中にエラー: {str(e)}"
            self.logger.error(error_msg)
            
            # エラー時は空の結果を返す
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                return self._create_empty_result(len(src_prices), src_prices)
            else:
                return self._create_empty_result(0, np.array([]))
    
    def _apply_sequential_filtering(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        smoother_values: np.ndarray
    ) -> np.ndarray:
        """シーケンシャルフィルタリング: スムーサー出力をカルマンで追加処理"""
        
        # スムーサー出力を新しいデータフレームとして構築
        if isinstance(data, pd.DataFrame):
            # DataFrameの場合: closeカラムをスムーサー値で置換
            enhanced_data = data.copy()
            enhanced_data['close'] = smoother_values
            
            # 他のOHLVカラムも調整（論理的整合性を保持）
            for i in range(len(enhanced_data)):
                if not np.isnan(smoother_values[i]):
                    close = smoother_values[i]
                    
                    # 元の比率を保持して調整
                    if not np.isnan(data.iloc[i]['close']) and data.iloc[i]['close'] != 0:
                        ratio = close / data.iloc[i]['close']
                        enhanced_data.iloc[i, enhanced_data.columns.get_loc('open')] *= ratio
                        enhanced_data.iloc[i, enhanced_data.columns.get_loc('high')] *= ratio
                        enhanced_data.iloc[i, enhanced_data.columns.get_loc('low')] *= ratio
        else:
            # NumPy配列の場合: closeカラム（最後の列）を置換
            enhanced_data = data.copy()
            if enhanced_data.ndim > 1 and enhanced_data.shape[1] >= 4:
                enhanced_data[:, 3] = smoother_values  # close列
            else:
                enhanced_data = smoother_values
        
        # カルマンフィルターを適用
        kalman_result = self.kalman.calculate(enhanced_data)
        return kalman_result.values
    
    def _apply_weighted_combination(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        smoother_values: np.ndarray
    ) -> np.ndarray:
        """重み付き組み合わせ: スムーサーとカルマンの重み付き平均"""
        
        # カルマンフィルターを元データに適用
        kalman_result = self.kalman.calculate(data)
        kalman_values = kalman_result.values
        
        # 動的重み計算（ボラティリティベース）
        window = min(20, len(smoother_values) // 4)
        weights = np.zeros(len(smoother_values))
        
        for i in range(len(smoother_values)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx > 1:
                # 最近のボラティリティを計算
                recent_prices = smoother_values[start_idx:end_idx]
                volatility = np.std(np.diff(recent_prices)) if len(recent_prices) > 1 else 0
                
                # 高ボラティリティ時はカルマンの重みを増加
                kalman_weight = min(0.8, 0.3 + volatility * 10)
                weights[i] = 1 - kalman_weight
            else:
                weights[i] = 0.7  # デフォルト：スムーサー重視
        
        # 重み付き平均
        valid_mask = ~(np.isnan(smoother_values) | np.isnan(kalman_values))
        enhanced_values = np.full_like(smoother_values, np.nan)
        
        enhanced_values[valid_mask] = (
            weights[valid_mask] * smoother_values[valid_mask] +
            (1 - weights[valid_mask]) * kalman_values[valid_mask]
        )
        
        return enhanced_values
    
    def _apply_adaptive_combination(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        smoother_values: np.ndarray
    ) -> np.ndarray:
        """適応的組み合わせ: 市況に応じて動的に最適化"""
        
        # カルマンフィルターを元データに適用
        kalman_result = self.kalman.calculate(data)
        kalman_values = kalman_result.values
        
        # 市況分析ウィンドウ
        analysis_window = min(50, len(smoother_values) // 3)
        enhanced_values = np.full_like(smoother_values, np.nan)
        
        for i in range(len(smoother_values)):
            if np.isnan(smoother_values[i]) or np.isnan(kalman_values[i]):
                continue
                
            start_idx = max(0, i - analysis_window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx > 10:
                # トレンド強度の計算
                recent_data = smoother_values[start_idx:end_idx]
                trend_strength = self._calculate_trend_strength(recent_data)
                
                # ノイズレベルの計算
                noise_level = self._calculate_noise_level(recent_data)
                
                # 適応的重み決定
                if trend_strength > 0.7:
                    # 強いトレンド時：スムーサー重視
                    weight = 0.8
                elif noise_level > 0.5:
                    # 高ノイズ時：カルマン重視
                    weight = 0.3
                else:
                    # 通常時：バランス
                    weight = 0.6
                
                enhanced_values[i] = (
                    weight * smoother_values[i] + 
                    (1 - weight) * kalman_values[i]
                )
            else:
                # データ不足時はスムーサー値を使用
                enhanced_values[i] = smoother_values[i]
        
        return enhanced_values
    
    def _calculate_trend_strength(self, values: np.ndarray) -> float:
        """トレンド強度を計算"""
        if len(values) < 5:
            return 0.0
        
        # 線形回帰の決定係数（R²）を使用
        x = np.arange(len(values))
        y = values
        
        # NaN除去
        valid_mask = ~np.isnan(y)
        if np.sum(valid_mask) < 3:
            return 0.0
        
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        # 決定係数計算
        correlation_matrix = np.corrcoef(x_valid, y_valid)
        if correlation_matrix.shape == (2, 2):
            correlation = correlation_matrix[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _calculate_noise_level(self, values: np.ndarray) -> float:
        """ノイズレベルを計算"""
        if len(values) < 3:
            return 0.0
        
        # 変化率の標準偏差を使用
        returns = np.diff(values) / values[:-1]
        valid_returns = returns[~np.isnan(returns)]
        
        if len(valid_returns) > 0:
            return min(1.0, np.std(valid_returns) * 100)  # 0-1に正規化
        
        return 0.0
    
    def _create_empty_result(self, length: int, raw_prices: np.ndarray) -> KalmanEnhancedResult:
        """空の結果を作成"""
        return KalmanEnhancedResult(
            values=np.full(length, np.nan),
            smoother_values=np.full(length, np.nan),
            kalman_values=np.full(length, np.nan),
            raw_values=raw_prices,
            smoother_type=self.smoother_type,
            kalman_type=self.kalman_type,
            smoother_params={},
            kalman_params={},
            additional_data={}
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """強化済み値を取得"""
        return self._values.copy() if self._values is not None else None
    
    def get_smoother_values(self) -> Optional[np.ndarray]:
        """スムーサー単独値を取得"""
        if self._last_result is not None:
            return self._last_result.smoother_values.copy()
        return None
    
    def get_kalman_values(self) -> Optional[np.ndarray]:
        """カルマン単独値を取得"""
        if self._last_result is not None:
            return self._last_result.kalman_values.copy()
        return None
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        if hasattr(self.smoother, 'reset'):
            self.smoother.reset()
        if hasattr(self.kalman, 'reset'):
            self.kalman.reset()
        self._last_result = None


# 便利関数
def kalman_enhanced_alma(
    data: Union[pd.DataFrame, np.ndarray],
    alma_length: int = 9,
    alma_offset: float = 0.85,
    alma_sigma: float = 6.0,
    kalman_type: str = 'simple',
    **kalman_params
) -> np.ndarray:
    """ALMA + Kalman Filter 組み合わせ（便利関数）"""
    smoother = KalmanEnhancedSmoother(
        smoother_type='alma',
        kalman_type=kalman_type,
        length=alma_length,
        offset=alma_offset,
        sigma=alma_sigma,
        **{f'kalman_{k}': v for k, v in kalman_params.items()}
    )
    result = smoother.calculate(data)
    return result.values


def kalman_enhanced_hma(
    data: Union[pd.DataFrame, np.ndarray],
    hma_length: int = 14,
    kalman_type: str = 'simple',
    **kalman_params
) -> np.ndarray:
    """HMA + Kalman Filter 組み合わせ（便利関数）"""
    smoother = KalmanEnhancedSmoother(
        smoother_type='hma',
        kalman_type=kalman_type,
        length=hma_length,
        **{f'kalman_{k}': v for k, v in kalman_params.items()}
    )
    result = smoother.calculate(data)
    return result.values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    print("=== カルマン強化スムーサーのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 100
    
    # 真の信号（サイン波 + トレンド）
    t = np.linspace(0, 4*np.pi, length)
    true_signal = 100 + 10 * np.sin(t) + 0.1 * t
    
    # 観測値（ノイズ付き）
    noise = np.random.normal(0, 3, length)
    observations = true_signal + noise
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(observations):
        high = close + abs(np.random.normal(0, 1))
        low = close - abs(np.random.normal(0, 1))
        open_price = observations[i-1] if i > 0 else close
        
        data.append({
            'open': open_price,
            'high': max(high, open_price, close),
            'low': min(low, open_price, close),
            'close': close,
            'volume': np.random.uniform(1000, 5000)
        })
    
    df = pd.DataFrame(data)
    start_date = datetime.now() - timedelta(days=length)
    df.index = [start_date + timedelta(hours=i) for i in range(length)]
    
    print(f"テストデータ: {len(df)}ポイント")
    
    # 異なる組み合わせでテスト
    test_combinations = [
        {'smoother': 'alma', 'kalman': 'simple', 'mode': 'sequential'},
        {'smoother': 'hma', 'kalman': 'simple', 'mode': 'weighted'},
        {'smoother': 'alma', 'kalman': 'quantum_adaptive', 'mode': 'adaptive'}
    ]
    
    for combo in test_combinations:
        print(f"\n{combo['smoother']} + {combo['kalman']} ({combo['mode']}) をテスト中...")
        
        try:
            smoother = KalmanEnhancedSmoother(
                smoother_type=combo['smoother'],
                kalman_type=combo['kalman'],
                combination_mode=combo['mode'],
                src_type='close'
            )
            result = smoother.calculate(df)
            
            # 統計計算
            valid_mask = ~np.isnan(result.values)
            if np.any(valid_mask):
                valid_enhanced = result.values[valid_mask]
                valid_true = true_signal[valid_mask]
                
                mae = np.mean(np.abs(valid_enhanced - valid_true))
                correlation = np.corrcoef(valid_enhanced, valid_true)[0, 1]
                
                print(f"  MAE: {mae:.4f}")
                print(f"  相関係数: {correlation:.4f}")
                print(f"  有効値数: {np.sum(valid_mask)}/{len(df)}")
                
        except Exception as e:
            print(f"  エラー: {e}")
    
    print("\n=== テスト完了 ===")