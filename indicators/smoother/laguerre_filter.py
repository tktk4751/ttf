#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Laguerre Filter - ラゲールフィルター** 🎯

John F. Ehlers氏の論文「TIME WARP – WITHOUT SPACE TRAVEL」に基づく実装。
- 時間軸を歪ませることで低周波成分により多くの遅延を適用
- 高周波成分には少ない遅延を適用
- 従来のFIRフィルターより優れたスムージング効果を提供

📊 **主な特徴:**
- Laguerre Transform を使用した時間軸歪み
- 全パス（All-pass）ネットワークによる可変遅延
- 最初にEMA低域フィルターを適用
- 後続のステージで全パス要素を使用
- ダンピングファクター（gamma）による調整可能

🔧 **パラメータ:**
- gamma (0-1): ダンピングファクター。大きいほどスムーシング強
- order: フィルターオーダー（通常4）
- coefficients: フィルター係数（デフォルト: [1, 2, 2, 1]/6）

📈 **使用例:**
```python
filter = LaguerreFilter(gamma=0.8, order=4)
result = filter.calculate(data)
```
"""

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any, List
import numpy as np
import pandas as pd
from numba import njit

try:
    from ..indicator import Indicator
    from ..price_source import PriceSource
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from indicators.indicator import Indicator
    from indicators.price_source import PriceSource


@dataclass
class LaguerreFilterResult:
    """ラゲールフィルターの計算結果"""
    values: np.ndarray              # フィルタリングされた値
    laguerre_stages: np.ndarray     # 各ステージの値 (shape: (n, order))
    gamma: float                    # 使用されたダンピングファクター
    coefficients: np.ndarray        # 使用された係数
    raw_values: np.ndarray          # 元の価格データ
    parameters: Dict[str, Any]      # 使用されたパラメータ


@njit
def laguerre_filter_dynamic_core(prices: np.ndarray, dynamic_gamma: np.ndarray, order: int, coefficients: np.ndarray) -> tuple:
    """
    動的ガンマ対応ラゲールフィルターのコア計算（Numba最適化）
    
    Args:
        prices: 価格データ
        dynamic_gamma: 動的ガンマ値の配列
        order: フィルターオーダー
        coefficients: フィルター係数
        
    Returns:
        tuple: (フィルタリング結果, 各ステージの値)
    """
    n = len(prices)
    
    # 各ステージの値を保存する配列
    stages = np.zeros((n, order))
    filtered_values = np.zeros(n)
    
    # 各ステージの前回値を保存
    prev_stages = np.zeros(order)
    
    for i in range(n):
        current_price = prices[i]
        gamma = dynamic_gamma[i] if not np.isnan(dynamic_gamma[i]) else 0.8
        
        # 第0ステージ: EMA (低域フィルター)
        if i == 0:
            stages[i, 0] = current_price
        else:
            stages[i, 0] = (1 - gamma) * current_price + gamma * prev_stages[0]
        
        # 第1〜order-1ステージ: 全パス要素
        for stage in range(1, order):
            if i == 0:
                stages[i, stage] = stages[i, 0]
            else:
                # 全パス要素の計算: -gamma*input + prev_input + gamma*prev_output
                input_val = stages[i, stage-1]
                prev_input = prev_stages[stage-1] if i > 0 else input_val
                prev_output = prev_stages[stage] if i > 0 else input_val
                
                stages[i, stage] = -gamma * input_val + prev_input + gamma * prev_output
        
        # フィルタリング結果の計算（係数の重み付き平均）
        if len(coefficients) == order:
            filtered_values[i] = np.sum(stages[i, :] * coefficients)
        else:
            # 等重み平均をフォールバック
            filtered_values[i] = np.mean(stages[i, :])
        
        # 前回値を更新
        prev_stages[:] = stages[i, :]
    
    return filtered_values, stages


@njit
def laguerre_filter_core(prices: np.ndarray, gamma: float, order: int, coefficients: np.ndarray) -> tuple:
    """
    ラゲールフィルターのコア計算（Numba最適化）
    
    Args:
        prices: 価格データ
        gamma: ダンピングファクター (0-1)
        order: フィルターオーダー
        coefficients: フィルター係数
        
    Returns:
        tuple: (フィルタリング結果, 各ステージの値)
    """
    n = len(prices)
    
    # 各ステージの値を保存する配列
    stages = np.zeros((n, order))
    filtered_values = np.zeros(n)
    
    # 各ステージの前回値を保存
    prev_stages = np.zeros(order)
    
    for i in range(n):
        current_price = prices[i]
        
        # 第0ステージ: EMA (低域フィルター)
        if i == 0:
            stages[i, 0] = current_price
        else:
            stages[i, 0] = (1 - gamma) * current_price + gamma * prev_stages[0]
        
        # 第1〜order-1ステージ: 全パス要素
        for stage in range(1, order):
            if i == 0:
                stages[i, stage] = stages[i, 0]
            else:
                # 全パス要素の計算: -gamma*input + prev_input + gamma*prev_output
                input_val = stages[i, stage-1]
                prev_input = prev_stages[stage-1] if i > 0 else input_val
                prev_output = prev_stages[stage] if i > 0 else input_val
                
                stages[i, stage] = -gamma * input_val + prev_input + gamma * prev_output
        
        # フィルタリング結果の計算（係数の重み付き平均）
        if len(coefficients) == order:
            filtered_values[i] = np.sum(stages[i, :] * coefficients)
        else:
            # 等重み平均をフォールバック
            filtered_values[i] = np.mean(stages[i, :])
        
        # 前回値を更新
        prev_stages[:] = stages[i, :]
    
    return filtered_values, stages


@njit
def validate_laguerre_params(gamma: float, order: int) -> bool:
    """ラゲールフィルターパラメータの検証"""
    if not (0.0 <= gamma <= 1.0):
        return False
    if order < 2 or order > 10:
        return False
    return True


class LaguerreFilter(Indicator):
    """
    ラゲールフィルター
    
    John F. Ehlers氏の論文「TIME WARP – WITHOUT SPACE TRAVEL」に基づく実装。
    時間軸を歪ませることで、低周波成分により多くの遅延を適用し、
    高周波成分には少ない遅延を適用する高度なスムージングフィルター。
    """
    
    def __init__(
        self,
        gamma: float = 0.5,
        order: int = 4,
        coefficients: Optional[List[float]] = None,
        src_type: str = 'close',
        period: int = 4,  # 互換性のため
        period_mode: str = 'fixed',
        cycle_detector_type: str = 'hody_e',
        cycle_part: float = 0.5,
        max_cycle: int = 124,
        min_cycle: int = 13,
        max_output: int = 124,
        min_output: int = 13,
        lp_period: int = 13,
        hp_period: int = 124
    ):
        """
        コンストラクタ
        
        Args:
            gamma: ダンピングファクター (0-1)。大きいほどスムーシング強
            order: フィルターオーダー（通常4）
            coefficients: フィルター係数。Noneの場合はデフォルト値使用
            src_type: 価格ソース ('close', 'hlc3', 'hl2', 'ohlc4', etc.)
            period: 期間（互換性維持のため、実際の計算には使用しない）
            period_mode: 期間モード ('fixed' または 'dynamic')
            cycle_detector_type: サイクル検出器タイプ（動的期間用）
            cycle_part: サイクル部分の倍率
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
        """
        super().__init__(f"LaguerreFilter(gamma={gamma}, order={order})")
        
        # パラメータ検証
        if not validate_laguerre_params(gamma, order):
            raise ValueError(f"無効なパラメータ: gamma={gamma} (0-1の範囲), order={order} (2-10の範囲)")
        
        self.gamma = float(gamma)
        self.order = int(order)
        self.src_type = src_type.lower()
        self.period = period
        self.period_mode = period_mode.lower()
        self.cycle_detector_type = cycle_detector_type
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.lp_period = lp_period
        self.hp_period = hp_period
        
        # 期間モード検証
        if self.period_mode not in ['fixed', 'dynamic']:
            raise ValueError(f"無効な期間モード: {period_mode}")
        
        # デフォルト係数の設定
        if coefficients is None:
            if order == 4:
                # 論文のデフォルト係数 [1, 2, 2, 1] / 6
                self.coefficients = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
            else:
                # 等重み平均
                self.coefficients = np.ones(order) / order
        else:
            self.coefficients = np.array(coefficients, dtype=np.float64)
            # 正規化（合計が1になるように）
            coeff_sum = np.sum(self.coefficients)
            if coeff_sum > 0:
                self.coefficients = self.coefficients / coeff_sum
        
        # 係数数がオーダーと一致するかチェック
        if len(self.coefficients) != order:
            self.logger.warning(
                f"係数数({len(self.coefficients)})がオーダー({order})と一致しません。"
                f"等重み平均を使用します。"
            )
            self.coefficients = np.ones(order) / order
        
        # ソースタイプ検証
        valid_sources = ['close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open','oc2']
        if self.src_type not in valid_sources:
            raise ValueError(f"無効なソースタイプ: {src_type}")
        
        # 内部状態
        self._last_result = None
        
        # 動的期間用のサイクル検出器
        self._cycle_detector = None
        if self.period_mode == 'dynamic':
            self._initialize_cycle_detector()
    
    def _initialize_cycle_detector(self):
        """サイクル検出器を初期化"""
        try:
            from ..cycle.ehlers_unified_dc import EhlersUnifiedDC
            self._cycle_detector = EhlersUnifiedDC(
                detector_type=self.cycle_detector_type,
                cycle_part=self.cycle_part,
                max_cycle=self.max_cycle,
                min_cycle=self.min_cycle,
                max_output=self.max_output,
                min_output=self.min_output,
                src_type=self.src_type,
                lp_period=self.lp_period,
                hp_period=self.hp_period
            )
        except ImportError as e:
            self.logger.warning(f"サイクル検出器のインポートに失敗: {e}。固定期間モードにフォールバック")
            self.period_mode = 'fixed'
            self._cycle_detector = None
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> LaguerreFilterResult:
        """
        ラゲールフィルターを計算
        
        Args:
            data: 価格データ
            
        Returns:
            LaguerreFilterResult: 計算結果
        """
        try:
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            data_length = len(src_prices)
            
            # 動的期間の計算
            dynamic_gamma = None
            if self.period_mode == 'dynamic' and self._cycle_detector is not None:
                try:
                    cycle_values = self._cycle_detector.calculate(data)
                    periods = np.clip(cycle_values, self.min_output, self.max_output)
                    # NaN値を期間のデフォルト値で埋める
                    nan_mask = np.isnan(periods)
                    if np.any(nan_mask):
                        periods[nan_mask] = self.period
                    # 期間に基づいて動的ガンマを計算
                    dynamic_gamma = self.gamma * (self.period / periods)
                    dynamic_gamma = np.clip(dynamic_gamma, 0.1, 0.9)  # 範囲制限
                except Exception as e:
                    self.logger.warning(f"動的期間計算に失敗: {e}。固定期間にフォールバック")
                    dynamic_gamma = None
            
            if data_length < 2:
                return self._create_empty_result(data_length, src_prices)
            
            # 有効な価格データのチェック
            if np.all(np.isnan(src_prices)):
                return self._create_empty_result(data_length, src_prices)
            
            # ラゲールフィルターの計算
            if dynamic_gamma is not None:
                # 動的ガンマモード
                filtered_values, stages = laguerre_filter_dynamic_core(
                    src_prices, dynamic_gamma, self.order, self.coefficients
                )
            else:
                # 固定ガンマモード
                filtered_values, stages = laguerre_filter_core(
                    src_prices, self.gamma, self.order, self.coefficients
                )
            
            # 結果の作成
            result = LaguerreFilterResult(
                values=filtered_values,
                laguerre_stages=stages,
                gamma=self.gamma,
                coefficients=self.coefficients.copy(),
                raw_values=src_prices.copy(),
                parameters={
                    'gamma': self.gamma,
                    'order': self.order,
                    'period': self.period,
                    'src_type': self.src_type,
                    'period_mode': self.period_mode,
                    'cycle_detector_type': self.cycle_detector_type if self.period_mode == 'dynamic' else None,
                    'dynamic_gamma': dynamic_gamma.tolist() if dynamic_gamma is not None else None,
                    'coefficients': self.coefficients.tolist()
                }
            )
            
            # 基底クラス用の値設定
            self._values = filtered_values
            self._last_result = result
            
            return result
            
        except Exception as e:
            error_msg = f"ラゲールフィルター計算中にエラー: {str(e)}"
            self.logger.error(error_msg)
            
            # エラー時は空の結果を返す
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                return self._create_empty_result(len(src_prices), src_prices)
            else:
                return self._create_empty_result(0, np.array([]))
    
    def _create_empty_result(self, length: int, raw_prices: np.ndarray) -> LaguerreFilterResult:
        """空の結果を作成"""
        return LaguerreFilterResult(
            values=np.full(length, np.nan),
            laguerre_stages=np.full((length, self.order), np.nan),
            gamma=self.gamma,
            coefficients=self.coefficients.copy(),
            raw_values=raw_prices,
            parameters={
                'gamma': self.gamma,
                'order': self.order,
                'period': self.period,
                'src_type': self.src_type,
                'period_mode': self.period_mode,
                'cycle_detector_type': self.cycle_detector_type if self.period_mode == 'dynamic' else None,
                'dynamic_gamma': None,
                'coefficients': self.coefficients.tolist()
            }
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """フィルタリングされた値を取得"""
        return self._values.copy() if self._values is not None else None
    
    def get_laguerre_stages(self) -> Optional[np.ndarray]:
        """各ラゲールステージの値を取得"""
        if self._last_result is not None:
            return self._last_result.laguerre_stages.copy()
        return None
    
    def get_stage(self, stage_index: int) -> Optional[np.ndarray]:
        """特定のステージの値を取得"""
        if self._last_result is not None and 0 <= stage_index < self.order:
            return self._last_result.laguerre_stages[:, stage_index].copy()
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """パラメータ情報を取得"""
        return {
            'gamma': self.gamma,
            'order': self.order,
            'coefficients': self.coefficients.tolist(),
            'src_type': self.src_type,
            'period': self.period,
            'period_mode': self.period_mode,
            'cycle_detector_type': self.cycle_detector_type,
            'supports_dynamic': self.period_mode == 'dynamic' and self._cycle_detector is not None
        }
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self._last_result = None
        if self._cycle_detector is not None:
            self._cycle_detector.reset()
    
    @staticmethod
    def create_fir_comparison(
        data: Union[pd.DataFrame, np.ndarray],
        gamma: float = 0.8,
        order: int = 4,
        src_type: str = 'close'
    ) -> Dict[str, np.ndarray]:
        """
        ラゲールフィルターと同等のFIRフィルターの比較を作成
        
        Args:
            data: 価格データ
            gamma: ダンピングファクター
            order: フィルターオーダー
            src_type: 価格ソース
            
        Returns:
            Dict[str, np.ndarray]: 'laguerre'と'fir'の結果
        """
        # ラゲールフィルターの計算
        laguerre = LaguerreFilter(gamma=gamma, order=order, src_type=src_type)
        laguerre_result = laguerre.calculate(data)
        
        # 同等のFIRフィルターの計算（論文の例）
        src_prices = PriceSource.calculate_source(data, src_type)
        
        if order == 4:
            # 論文のFIRフィルター係数 [1, 2, 2, 1] / 6
            fir_result = _calculate_fir_filter(src_prices, np.array([1.0, 2.0, 2.0, 1.0]) / 6.0)
        else:
            # 等重み移動平均
            fir_result = _calculate_simple_ma(src_prices, order)
        
        return {
            'laguerre': laguerre_result.values,
            'fir': fir_result,
            'raw': src_prices
        }


@njit
def _calculate_fir_filter(prices: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """FIRフィルターの計算（Numba最適化）"""
    n = len(prices)
    order = len(coefficients)
    result = np.zeros(n)
    
    for i in range(n):
        weighted_sum = 0.0
        for j in range(order):
            if i - j >= 0:
                weighted_sum += prices[i - j] * coefficients[j]
            else:
                # 足りない部分は最初の値で埋める
                weighted_sum += prices[0] * coefficients[j]
        result[i] = weighted_sum
    
    return result


@njit
def _calculate_simple_ma(prices: np.ndarray, period: int) -> np.ndarray:
    """単純移動平均の計算（Numba最適化）"""
    n = len(prices)
    result = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - period + 1)
        result[i] = np.mean(prices[start_idx:i+1])
    
    return result


# 便利関数
def laguerre_filter(
    data: Union[pd.DataFrame, np.ndarray],
    gamma: float = 0.8,
    order: int = 4,
    coefficients: Optional[List[float]] = None,
    src_type: str = 'close'
) -> np.ndarray:
    """
    ラゲールフィルターの計算（便利関数）
    
    Args:
        data: 価格データ
        gamma: ダンピングファクター
        order: フィルターオーダー
        coefficients: フィルター係数
        src_type: 価格ソース
        
    Returns:
        フィルタリングされた値
    """
    filter = LaguerreFilter(gamma=gamma, order=order, coefficients=coefficients, src_type=src_type)
    result = filter.calculate(data)
    return result.values


def laguerre_rsi(
    data: Union[pd.DataFrame, np.ndarray],
    gamma: float = 0.5,
    order: int = 4,
    src_type: str = 'close'
) -> np.ndarray:
    """
    ラゲールRSIの計算（論文の例）
    
    Args:
        data: 価格データ
        gamma: ダンピングファクター
        order: フィルターオーダー
        src_type: 価格ソース
        
    Returns:
        ラゲールRSI値 (0-1の範囲)
    """
    # ラゲールフィルターでステージを計算
    filter = LaguerreFilter(gamma=gamma, order=order, src_type=src_type)
    result = filter.calculate(data)
    stages = result.laguerre_stages
    
    n = len(stages)
    rsi_values = np.zeros(n)
    
    for i in range(n):
        cu = 0.0  # Closes Up
        cd = 0.0  # Closes Down
        
        # 隣接するステージ間の差分を計算
        for j in range(order - 1):
            if j + 1 < order:
                diff = stages[i, j] - stages[i, j + 1]
                if diff >= 0:
                    cu += diff
                else:
                    cd += abs(diff)
        
        # RSI計算
        if cu + cd > 0:
            rsi_values[i] = cu / (cu + cd)
        else:
            rsi_values[i] = 0.5  # 中性値
    
    return rsi_values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    print("=== ラゲールフィルターのテスト ===")
    
    # テストデータ生成（ノイズ付きトレンド）
    np.random.seed(42)
    length = 200
    base_price = 100.0
    trend = 0.001
    volatility = 0.03
    
    prices = [base_price]
    for i in range(1, length):
        # トレンド + ノイズ + 時々のジャンプ
        change = trend + np.random.normal(0, volatility)
        if np.random.random() < 0.02:  # 2%の確率でジャンプ
            change += np.random.choice([-1, 1]) * volatility * 3
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, volatility * close * 0.5))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, volatility * close * 0.2)
            open_price = prices[i-1] + gap
        
        # 論理的整合性の確保
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(data)
    start_date = datetime.now() - timedelta(days=length)
    df.index = [start_date + timedelta(hours=i) for i in range(length)]
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # ラゲールフィルターのテスト
    gamma_values = [0.5, 0.8]
    
    for gamma in gamma_values:
        print(f"\nGamma = {gamma} のテスト:")
        
        # ラゲールフィルター
        laguerre = LaguerreFilter(gamma=gamma, order=4, src_type='close')
        result = laguerre.calculate(df)
        
        # FIRフィルターとの比較
        comparison = LaguerreFilter.create_fir_comparison(df, gamma=gamma, order=4)
        
        print(f"  ラゲールフィルター平均: {np.nanmean(result.values):.4f}")
        print(f"  FIRフィルター平均: {np.nanmean(comparison['fir']):.4f}")
        print(f"  元価格平均: {np.nanmean(comparison['raw']):.4f}")
        
        # スムージング効果の測定（標準偏差の比較）
        raw_std = np.nanstd(comparison['raw'])
        laguerre_std = np.nanstd(result.values)
        fir_std = np.nanstd(comparison['fir'])
        
        print(f"  標準偏差 - 元: {raw_std:.4f}, ラゲール: {laguerre_std:.4f}, FIR: {fir_std:.4f}")
        print(f"  スムージング率 - ラゲール: {(1 - laguerre_std/raw_std)*100:.1f}%, FIR: {(1 - fir_std/raw_std)*100:.1f}%")
    
    # ラゲールRSIのテスト
    print(f"\nラゲールRSIのテスト:")
    rsi_values = laguerre_rsi(df, gamma=0.5, order=4)
    print(f"  RSI範囲: {np.nanmin(rsi_values):.3f} - {np.nanmax(rsi_values):.3f}")
    print(f"  RSI平均: {np.nanmean(rsi_values):.3f}")
    
    # 80%/20%レベルの交差回数
    over_80 = np.sum(rsi_values > 0.8)
    under_20 = np.sum(rsi_values < 0.2)
    print(f"  80%超え: {over_80}回, 20%未満: {under_20}回")
    
    print("\n=== テスト完了 ===")