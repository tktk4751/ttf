#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **ALMA - Arnaud Legoux Moving Average** 🎯

Arnaud Legoux移動平均は、ガウシアンフィルターを基盤とした
適応的移動平均で、遅延を最小化しながらノイズを効果的に除去します。

📊 **特徴:**
- ガウシアンフィルターベース
- 適応的な重み付け
- 低遅延とノイズ除去のバランス
- パラメータによる調整可能

🔧 **パラメータ:**
- length: 計算期間（デフォルト: 9）
- offset: オフセット係数（デフォルト: 0.85）
- sigma: シグマ係数（デフォルト: 6）
- src_type: 価格ソース（デフォルト: 'close'）

🌟 **使用例:**
```python
alma = ALMA(length=14, offset=0.85, sigma=6)
result = alma.calculate(data)
```
"""

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
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
class ALMAResult:
    """ALMA計算結果"""
    values: np.ndarray              # ALMA値
    weights: np.ndarray             # ガウシアン重み
    raw_values: np.ndarray          # 元の価格データ
    parameters: Dict[str, Any]      # 使用されたパラメータ


@njit
def calculate_gaussian_weights(length: int, offset: float, sigma: float) -> np.ndarray:
    """
    ガウシアン重みの計算（Numba最適化）
    
    Args:
        length: 期間
        offset: オフセット係数
        sigma: シグマ係数
        
    Returns:
        正規化されたガウシアン重み
    """
    weights = np.zeros(length)
    
    # オフセット位置の計算
    m = offset * (length - 1)
    
    # シグマの計算
    s = length / sigma
    
    # ガウシアン重みの計算
    sum_weights = 0.0
    for i in range(length):
        weight = np.exp(-((i - m) ** 2) / (2 * s * s))
        weights[i] = weight
        sum_weights += weight
    
    # 正規化
    if sum_weights > 0:
        for i in range(length):
            weights[i] /= sum_weights
    
    return weights


@njit
def alma_core_calculation(
    prices: np.ndarray,
    length: int,
    offset: float,
    sigma: float
) -> tuple:
    """
    ALMAコア計算（Numba最適化）
    
    Args:
        prices: 価格配列
        length: 期間
        offset: オフセット係数
        sigma: シグマ係数
        
    Returns:
        tuple: (ALMA値, 使用された重み)
    """
    n = len(prices)
    alma_values = np.zeros(n)
    
    # ガウシアン重みの計算
    weights = calculate_gaussian_weights(length, offset, sigma)
    
    # ALMA計算
    for i in range(n):
        if i < length - 1:
            # 初期期間: NaN
            alma_values[i] = np.nan
        else:
            # ALMA計算
            weighted_sum = 0.0
            
            for j in range(length):
                price_idx = i - (length - 1) + j
                weighted_sum += prices[price_idx] * weights[j]
            
            alma_values[i] = weighted_sum
    
    return alma_values, weights


@njit
def alma_dynamic_core_calculation(
    prices: np.ndarray,
    dynamic_length: np.ndarray,
    offset: float,
    sigma: float
) -> tuple:
    """
    動的期間対応ALMAコア計算（Numba最適化）
    
    Args:
        prices: 価格配列
        dynamic_length: 動的期間配列
        offset: オフセット係数
        sigma: シグマ係数
        
    Returns:
        tuple: (ALMA値, 使用された重み)
    """
    n = len(prices)
    alma_values = np.zeros(n)
    max_length = int(np.max(dynamic_length))
    weights = np.zeros(max_length)  # 最大長の重み配列
    
    # ALMA計算
    for i in range(n):
        current_length = int(dynamic_length[i]) if not np.isnan(dynamic_length[i]) else 9
        current_length = max(1, min(current_length, max_length))
        
        if i < current_length - 1:
            # 初期期間: NaN
            alma_values[i] = np.nan
        else:
            # ガウシアン重みの計算
            current_weights = calculate_gaussian_weights(current_length, offset, sigma)
            
            # ALMA計算
            weighted_sum = 0.0
            
            for j in range(current_length):
                price_idx = i - (current_length - 1) + j
                weighted_sum += prices[price_idx] * current_weights[j]
            
            alma_values[i] = weighted_sum
    
    return alma_values, weights


@njit
def validate_alma_params(length: int, offset: float, sigma: float) -> bool:
    """ALMAパラメータの検証"""
    if length < 1 or length > 500:
        return False
    if offset < 0.0 or offset > 1.0:
        return False
    if sigma <= 0.0 or sigma > 100.0:
        return False
    return True


class ALMA(Indicator):
    """
    ALMA - Arnaud Legoux Moving Average
    
    ガウシアンフィルターを基盤とした適応的移動平均。
    遅延を最小化しながらノイズを効果的に除去。
    """
    
    def __init__(
        self,
        length: int = 9,
        offset: float = 0.85,
        sigma: float = 6.0,
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
            offset: オフセット係数（0.0-1.0）
            sigma: シグマ係数（正の値）
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
        super().__init__(f"ALMA({length}, {offset}, {sigma}, mode={period_mode})")
        
        # パラメータ検証
        if not validate_alma_params(length, offset, sigma):
            raise ValueError(f"無効なパラメータ: length={length}, offset={offset}, sigma={sigma}")
        
        self.length = int(length)
        self.offset = float(offset)
        self.sigma = float(sigma)
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
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> ALMAResult:
        """
        ALMAを計算
        
        Args:
            data: 価格データ
            
        Returns:
            ALMAResult: 計算結果
        """
        try:
            # 価格データの抽出
            src_prices = PriceSource.calculate_source(data, self.src_type)
            data_length = len(src_prices)
            
            if data_length < self.length:
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
            
            # ALMAコア計算
            if dynamic_length is not None:
                # 動的期間モード
                alma_values, weights = alma_dynamic_core_calculation(
                    clean_prices, dynamic_length, self.offset, self.sigma
                )
            else:
                # 固定期間モード
                alma_values, weights = alma_core_calculation(
                    clean_prices, self.length, self.offset, self.sigma
                )
            
            # 結果の作成
            result = ALMAResult(
                values=alma_values,
                weights=weights,
                raw_values=src_prices.copy(),
                parameters={
                    'length': self.length,
                    'offset': self.offset,
                    'sigma': self.sigma,
                    'src_type': self.src_type,
                    'period_mode': self.period_mode,
                    'cycle_detector_type': self.cycle_detector_type if self.period_mode == 'dynamic' else None,
                    'dynamic_length': dynamic_length.tolist() if dynamic_length is not None else None
                }
            )
            
            # 基底クラス用の値設定
            self._values = alma_values
            self._last_result = result
            
            return result
            
        except Exception as e:
            error_msg = f"ALMA計算中にエラー: {str(e)}"
            self.logger.error(error_msg)
            
            # エラー時は空の結果を返す
            if isinstance(data, (pd.DataFrame, np.ndarray)) and len(data) > 0:
                src_prices = PriceSource.calculate_source(data, self.src_type)
                return self._create_empty_result(len(src_prices), src_prices)
            else:
                return self._create_empty_result(0, np.array([]))
    
    def _create_empty_result(self, length: int, raw_prices: np.ndarray) -> ALMAResult:
        """空の結果を作成"""
        return ALMAResult(
            values=np.full(length, np.nan),
            weights=np.zeros(self.length),
            raw_values=raw_prices,
            parameters={
                'length': self.length,
                'offset': self.offset,
                'sigma': self.sigma,
                'src_type': self.src_type,
                'period_mode': self.period_mode,
                'cycle_detector_type': self.cycle_detector_type if self.period_mode == 'dynamic' else None,
                'dynamic_length': None
            }
        )
    
    def get_values(self) -> Optional[np.ndarray]:
        """ALMA値を取得"""
        return self._values.copy() if self._values is not None else None
    
    def get_weights(self) -> Optional[np.ndarray]:
        """ガウシアン重みを取得"""
        if self._last_result is not None:
            return self._last_result.weights.copy()
        return None
    
    def get_parameters(self) -> Dict[str, Any]:
        """パラメータ情報を取得"""
        return {
            'length': self.length,
            'offset': self.offset,
            'sigma': self.sigma,
            'src_type': self.src_type,
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
    def get_recommended_params(volatility_regime: str = 'medium') -> Dict[str, Any]:
        """
        推奨パラメータを取得
        
        Args:
            volatility_regime: ボラティリティ環境
            
        Returns:
            推奨パラメータ辞書
        """
        params_map = {
            'low': {'length': 14, 'offset': 0.9, 'sigma': 8},      # 低ボラ: より敏感
            'medium': {'length': 9, 'offset': 0.85, 'sigma': 6},   # 中ボラ: バランス
            'high': {'length': 7, 'offset': 0.8, 'sigma': 4}       # 高ボラ: より応答性
        }
        return params_map.get(volatility_regime, params_map['medium'])


# 便利関数
def alma_filter(
    data: Union[pd.DataFrame, np.ndarray],
    length: int = 9,
    offset: float = 0.85,
    sigma: float = 6.0,
    src_type: str = 'close'
) -> np.ndarray:
    """
    ALMA計算（便利関数）
    
    Args:
        data: 価格データ
        length: 期間
        offset: オフセット係数
        sigma: シグマ係数
        src_type: 価格ソース
        
    Returns:
        ALMA値
    """
    alma = ALMA(length=length, offset=offset, sigma=sigma, src_type=src_type)
    result = alma.calculate(data)
    return result.values


def adaptive_alma_params(
    data: Union[pd.DataFrame, np.ndarray],
    src_type: str = 'close',
    window: int = 20
) -> Dict[str, Any]:
    """
    データに基づいて適応的にALMAパラメータを推定
    
    Args:
        data: 価格データ
        src_type: 価格ソース
        window: ボラティリティ計算ウィンドウ
        
    Returns:
        推奨パラメータ
    """
    src_prices = PriceSource.calculate_source(data, src_type)
    
    if len(src_prices) < window:
        return ALMA.get_recommended_params('medium')
    
    # 価格変化率のボラティリティを計算
    returns = np.diff(src_prices) / src_prices[:-1]
    volatility = np.nanstd(returns)
    
    # ボラティリティに基づくパラメータ調整
    if volatility < 0.01:
        regime = 'low'
    elif volatility > 0.03:
        regime = 'high'
    else:
        regime = 'medium'
    
    return ALMA.get_recommended_params(regime)


if __name__ == "__main__":
    """直接実行時のテスト"""
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    
    print("=== ALMAのテスト ===")
    
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
        {'length': 9, 'offset': 0.85, 'sigma': 6, 'label': 'デフォルト'},
        {'length': 14, 'offset': 0.9, 'sigma': 8, 'label': '低ボラ設定'},
        {'length': 7, 'offset': 0.8, 'sigma': 4, 'label': '高ボラ設定'}
    ]
    
    results = {}
    
    for params in test_params:
        print(f"\n{params['label']} をテスト中...")
        
        alma = ALMA(
            length=params['length'],
            offset=params['offset'],
            sigma=params['sigma'],
            src_type='close'
        )
        result = alma.calculate(df)
        
        results[params['label']] = result
        
        # 統計計算
        valid_mask = ~np.isnan(result.values)
        if np.any(valid_mask):
            valid_alma = result.values[valid_mask]
            valid_true = true_signal[valid_mask]
            
            mae = np.mean(np.abs(valid_alma - valid_true))
            correlation = np.corrcoef(valid_alma, valid_true)[0, 1]
            
            print(f"  MAE: {mae:.4f}")
            print(f"  相関係数: {correlation:.4f}")
            print(f"  有効値数: {np.sum(valid_mask)}/{len(df)}")
    
    # 適応的パラメータのテスト
    print(f"\n適応的パラメータ推定:")
    adaptive_params = adaptive_alma_params(df)
    print(f"  推奨パラメータ: {adaptive_params}")
    
    print("\n=== テスト完了 ===")