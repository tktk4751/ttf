#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **HMA - Hull Moving Average** 🎯

Hull移動平均は、Alan Hullによって開発された移動平均で、
遅延を大幅に削減しながらスムージング効果を維持します。

📊 **特徴:**
- 低遅延設計
- 高速トレンド追従
- ノイズ除去効果
- 加重移動平均ベース

🔧 **計算式:**
1. WMA1 = WMA(src, length/2) * 2
2. WMA2 = WMA(src, length)
3. Diff = WMA1 - WMA2
4. HMA = WMA(Diff, sqrt(length))

🔧 **パラメータ:**
- length: 計算期間（デフォルト: 14）
- src_type: 価格ソース（デフォルト: 'close'）

🌟 **使用例:**
```python
hma = HMA(length=21)
result = hma.calculate(data)
```
"""

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import njit
import math

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
class HMAResult:
    """HMA計算結果"""
    values: np.ndarray              # HMA値
    wma1_values: np.ndarray         # 中間WMA1値
    wma2_values: np.ndarray         # 中間WMA2値
    diff_values: np.ndarray         # 差分値
    raw_values: np.ndarray          # 元の価格データ
    parameters: Dict[str, Any]      # 使用されたパラメータ


@njit
def wma_core(prices: np.ndarray, length: int) -> np.ndarray:
    """
    加重移動平均のコア計算（Numba最適化）
    
    Args:
        prices: 価格配列
        length: 期間
        
    Returns:
        WMA値
    """
    n = len(prices)
    wma_values = np.zeros(n)
    
    # 重みの合計を事前計算
    weight_sum = length * (length + 1) // 2
    
    for i in range(n):
        if i < length - 1:
            wma_values[i] = np.nan
        else:
            weighted_sum = 0.0
            
            for j in range(length):
                weight = j + 1  # 重みは1から始まる
                price_idx = i - (length - 1) + j
                weighted_sum += prices[price_idx] * weight
            
            wma_values[i] = weighted_sum / weight_sum
    
    return wma_values


@njit
def hma_dynamic_core_calculation(
    prices: np.ndarray,
    dynamic_length: np.ndarray
) -> tuple:
    """
    動的期間対応HMAコア計算（Numba最適化）
    
    Args:
        prices: 価格配列
        dynamic_length: 動的期間配列
        
    Returns:
        tuple: (HMA値, WMA1値, WMA2値, 差分値)
    """
    n = len(prices)
    hma_values = np.zeros(n)
    wma1_values = np.zeros(n)
    wma2_values = np.zeros(n)
    diff_values = np.zeros(n)
    
    for i in range(n):
        current_length = int(dynamic_length[i]) if not np.isnan(dynamic_length[i]) else 14
        current_length = max(1, min(current_length, 200))  # 範囲制限
        
        # 各ステップの期間計算
        half_length = max(1, current_length // 2)
        sqrt_length = max(1, int(math.sqrt(current_length)))
        
        # 最小データ数チェック
        min_required = current_length + sqrt_length
        if i < min_required - 1:
            hma_values[i] = np.nan
            wma1_values[i] = np.nan
            wma2_values[i] = np.nan
            diff_values[i] = np.nan
            continue
        
        # ステップ1: WMA(src, half_length) * 2
        if i >= half_length - 1:
            weight_sum1 = half_length * (half_length + 1) // 2
            weighted_sum1 = 0.0
            for j in range(half_length):
                weight = j + 1
                price_idx = i - (half_length - 1) + j
                weighted_sum1 += prices[price_idx] * weight
            wma1_values[i] = (weighted_sum1 / weight_sum1) * 2
        else:
            wma1_values[i] = np.nan
        
        # ステップ2: WMA(src, current_length)
        if i >= current_length - 1:
            weight_sum2 = current_length * (current_length + 1) // 2
            weighted_sum2 = 0.0
            for j in range(current_length):
                weight = j + 1
                price_idx = i - (current_length - 1) + j
                weighted_sum2 += prices[price_idx] * weight
            wma2_values[i] = weighted_sum2 / weight_sum2
        else:
            wma2_values[i] = np.nan
        
        # ステップ3: 差分計算
        if not np.isnan(wma1_values[i]) and not np.isnan(wma2_values[i]):
            diff_values[i] = wma1_values[i] - wma2_values[i]
        else:
            diff_values[i] = np.nan
        
        # ステップ4: WMA(差分, sqrt_length)
        if i >= sqrt_length - 1 and not np.isnan(diff_values[i]):
            # 差分値のWMA計算
            weight_sum3 = sqrt_length * (sqrt_length + 1) // 2
            weighted_sum3 = 0.0
            valid_count = 0
            
            for j in range(sqrt_length):
                diff_idx = i - (sqrt_length - 1) + j
                if not np.isnan(diff_values[diff_idx]):
                    weight = j + 1
                    weighted_sum3 += diff_values[diff_idx] * weight
                    valid_count += weight
            
            if valid_count > 0:
                hma_values[i] = weighted_sum3 / weight_sum3
            else:
                hma_values[i] = np.nan
        else:
            hma_values[i] = np.nan
    
    return hma_values, wma1_values, wma2_values, diff_values


@njit
def hma_core_calculation(
    prices: np.ndarray,
    length: int
) -> tuple:
    """
    HMAコア計算（Numba最適化）
    
    Args:
        prices: 価格配列
        length: 期間
        
    Returns:
        tuple: (HMA値, WMA1値, WMA2値, 差分値)
    """
    n = len(prices)
    
    # ステップ1: WMA(src, length/2) * 2
    half_length = max(1, length // 2)
    wma1_values = wma_core(prices, half_length) * 2
    
    # ステップ2: WMA(src, length)
    wma2_values = wma_core(prices, length)
    
    # ステップ3: 差分計算
    diff_values = wma1_values - wma2_values
    
    # ステップ4: WMA(差分, sqrt(length))
    sqrt_length = max(1, int(math.sqrt(length)))
    hma_values = wma_core(diff_values, sqrt_length)
    
    return hma_values, wma1_values, wma2_values, diff_values


@njit
def validate_hma_params(length: int) -> bool:
    """HMAパラメータの検証"""
    if length < 1 or length > 500:
        return False
    return True


class HMA(Indicator):
    """
    HMA - Hull Moving Average
    
    Alan Hullによって開発された低遅延移動平均。
    加重移動平均を組み合わせて遅延を削減。
    """
    
    def __init__(
        self,
        length: int = 14,
        src_type: str = 'close',
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
            length: 計算期間（固定期間モード用）
            src_type: 価格ソース
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
        super().__init__(f"HMA({length}, mode={period_mode})")
        
        # パラメータ検証
        if not validate_hma_params(length):
            raise ValueError(f"無効なパラメータ: length={length}")
        
        self.length = int(length)
        self.src_type = src_type.lower()
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
        
        # ソースタイプ検証
        valid_sources = ['close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open']
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
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HMAResult:
        """
        HMAを計算
        
        Args:
            data: 価格データ
            
        Returns:
            HMAResult: 計算結果
        """
        try:
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            data_length = len(src_prices)
            
            # 最小データ数の確認
            min_required = self.length + int(math.sqrt(self.length))
            if data_length < min_required:
                return self._create_empty_result(data_length, src_prices)
            
            # 有効な価格データのチェック
            if np.all(np.isnan(src_prices)):
                return self._create_empty_result(data_length, src_prices)
            
            # NaN値の処理（前方補間）
            clean_prices = src_prices.copy()
            last_valid = None
            
            for i in range(data_length):
                if not np.isnan(clean_prices[i]):
                    last_valid = clean_prices[i]
                elif last_valid is not None:
                    clean_prices[i] = last_valid
            
            # 最初の有効値が見つからない場合
            if last_valid is None:
                return self._create_empty_result(data_length, src_prices)
            
            # 最初の無効値を最初の有効値で埋める
            first_valid_idx = np.where(~np.isnan(src_prices))[0]
            if len(first_valid_idx) > 0:
                first_valid_value = src_prices[first_valid_idx[0]]
                for i in range(first_valid_idx[0]):
                    clean_prices[i] = first_valid_value
            
            # 動的期間の計算
            dynamic_length = None
            if self.period_mode == 'dynamic' and self._cycle_detector is not None:
                try:
                    cycle_values = self._cycle_detector.calculate(data)
                    periods = np.clip(cycle_values, self.min_output, self.max_output)
                    # NaN値を期間のデフォルト値で埋める
                    nan_mask = np.isnan(periods)
                    if np.any(nan_mask):
                        periods[nan_mask] = self.length
                    dynamic_length = periods.astype(int)
                except Exception as e:
                    self.logger.warning(f"動的期間計算に失敗: {e}。固定期間にフォールバック")
                    dynamic_length = None
            
            # HMAコア計算
            if dynamic_length is not None:
                # 動的期間モード
                hma_values, wma1_values, wma2_values, diff_values = hma_dynamic_core_calculation(
                    clean_prices, dynamic_length
                )
            else:
                # 固定期間モード
                hma_values, wma1_values, wma2_values, diff_values = hma_core_calculation(
                    clean_prices, self.length
                )
            
            # 結果の作成
            result = HMAResult(
                values=hma_values,
                wma1_values=wma1_values,
                wma2_values=wma2_values,
                diff_values=diff_values,
                raw_values=src_prices.copy(),
                parameters={
                    'length': self.length,
                    'src_type': self.src_type,
                    'period_mode': self.period_mode,
                    'cycle_detector_type': self.cycle_detector_type if self.period_mode == 'dynamic' else None,
                    'dynamic_length': dynamic_length.tolist() if dynamic_length is not None else None,
                    'half_length': max(1, self.length // 2),
                    'sqrt_length': max(1, int(math.sqrt(self.length)))
                }
            )
            
            # 基底クラス用の値設定
            self._values = hma_values
            self._last_result = result
            
            return result
            
        except Exception as e:
            error_msg = f"HMA計算中にエラー: {str(e)}"
            self.logger.error(error_msg)
            
            # エラー時は空の結果を返す
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                return self._create_empty_result(len(src_prices), src_prices)
            else:
                return self._create_empty_result(0, np.array([]))
    
    def _create_empty_result(self, length: int, raw_prices: np.ndarray) -> HMAResult:
        """空の結果を作成"""
        return HMAResult(
            values=np.full(length, np.nan),
            wma1_values=np.full(length, np.nan),
            wma2_values=np.full(length, np.nan),
            diff_values=np.full(length, np.nan),
            raw_values=raw_prices,
            parameters={
                'length': self.length,
                'src_type': self.src_type,
                'period_mode': self.period_mode,
                'cycle_detector_type': self.cycle_detector_type if self.period_mode == 'dynamic' else None,
                'dynamic_length': None,
                'half_length': max(1, self.length // 2),
                'sqrt_length': max(1, int(math.sqrt(self.length)))
            }
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """HMA値を取得"""
        return self._values.copy() if self._values is not None else None
    
    def get_wma1_values(self) -> Optional[np.ndarray]:
        """WMA1値を取得"""
        if self._last_result is not None:
            return self._last_result.wma1_values.copy()
        return None
    
    def get_wma2_values(self) -> Optional[np.ndarray]:
        """WMA2値を取得"""
        if self._last_result is not None:
            return self._last_result.wma2_values.copy()
        return None
    
    def get_diff_values(self) -> Optional[np.ndarray]:
        """差分値を取得"""
        if self._last_result is not None:
            return self._last_result.diff_values.copy()
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """パラメータ情報を取得"""
        return {
            'length': self.length,
            'src_type': self.src_type,
            'period_mode': self.period_mode,
            'cycle_detector_type': self.cycle_detector_type,
            'supports_dynamic': self.period_mode == 'dynamic' and self._cycle_detector is not None,
            'half_length': max(1, self.length // 2),
            'sqrt_length': max(1, int(math.sqrt(self.length)))
        }
    
    def reset(self) -> None:
        """状態をリセット"""
        super().reset()
        self._last_result = None
        if self._cycle_detector is not None:
            self._cycle_detector.reset()
    
    @staticmethod
    def get_recommended_params(market_type: str = 'normal') -> Dict[str, Any]:
        """
        推奨パラメータを取得
        
        Args:
            market_type: マーケットタイプ
            
        Returns:
            推奨パラメータ辞書
        """
        params_map = {
            'fast': {'length': 9},      # 高速応答
            'normal': {'length': 14},   # 標準設定
            'slow': {'length': 21},     # 低速応答
            'very_slow': {'length': 34} # 非常に低速
        }
        return params_map.get(market_type, params_map['normal'])


# 便利関数
def hma_filter(
    data: Union[pd.DataFrame, np.ndarray],
    length: int = 14,
    src_type: str = 'close'
) -> np.ndarray:
    """
    HMA計算（便利関数）
    
    Args:
        data: 価格データ
        length: 期間
        src_type: 価格ソース
        
    Returns:
        HMA値
    """
    hma = HMA(length=length, src_type=src_type)
    result = hma.calculate(data)
    return result.values


def adaptive_hma_length(
    data: Union[pd.DataFrame, np.ndarray],
    src_type: str = 'close',
    window: int = 20
) -> int:
    """
    データに基づいて適応的にHMA期間を推定
    
    Args:
        data: 価格データ
        src_type: 価格ソース
        window: ボラティリティ計算ウィンドウ
        
    Returns:
        推奨期間
    """
    src_prices = PriceSource.calculate_source(data, src_type)
    
    if len(src_prices) < window:
        return 14
    
    # 価格変化率のボラティリティを計算
    returns = np.diff(src_prices[-window:]) / src_prices[-window:-1]
    volatility = np.nanstd(returns)
    
    # ボラティリティに基づく期間調整
    if volatility < 0.01:
        return 21  # 低ボラ: 長期間
    elif volatility > 0.03:
        return 9   # 高ボラ: 短期間
    else:
        return 14  # 中ボラ: 標準期間


if __name__ == "__main__":
    """直接実行時のテスト"""
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    print("=== HMAのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 100
    
    # 真の信号（サイン波 + トレンド）
    t = np.linspace(0, 4*np.pi, length)
    true_signal = 100 + 10 * np.sin(t) + 0.1 * t
    
    # 観測値（ノイズ付き）
    noise = np.random.normal(0, 2, length)
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
    
    # 異なるパラメータでテスト
    test_params = [
        {'length': 9, 'label': '高速'},
        {'length': 14, 'label': '標準'},
        {'length': 21, 'label': '低速'}
    ]
    
    results = {}
    
    for params in test_params:
        print(f"\n{params['label']} (length={params['length']}) をテスト中...")
        
        hma = HMA(length=params['length'], src_type='close')
        result = hma.calculate(df)
        
        results[params['label']] = result
        
        # 統計計算
        valid_mask = ~np.isnan(result.values)
        if np.any(valid_mask):
            valid_hma = result.values[valid_mask]
            valid_true = true_signal[valid_mask]
            
            mae = np.mean(np.abs(valid_hma - valid_true))
            correlation = np.corrcoef(valid_hma, valid_true)[0, 1]
            
            print(f"  MAE: {mae:.4f}")
            print(f"  相関係数: {correlation:.4f}")
            print(f"  有効値数: {np.sum(valid_mask)}/{len(df)}")
    
    # 適応的期間のテスト
    print(f"\n適応的期間推定:")
    adaptive_length = adaptive_hma_length(df)
    print(f"  推奨期間: {adaptive_length}")
    
    print("\n=== テスト完了 ===")