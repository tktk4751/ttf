#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import njit
import math

from ..indicator import Indicator
from ..price_source import PriceSource


@dataclass
class CorrelationCycleIndicatorResult:
    """Correlation Cycle Indicatorの計算結果"""
    real_component: np.ndarray            # Real成分（コサイン相関）
    imag_component: np.ndarray            # Imaginary成分（負サイン相関）
    angle: np.ndarray                     # フェーザー角度（-180度〜+180度）
    state: np.ndarray                     # 市場状態（+1: 上昇トレンド, 0: サイクル, -1: 下降トレンド）
    rate_of_change: np.ndarray            # 角度変化率（度/期間）
    cycle_mode: np.ndarray                # サイクルモード判定（True: サイクル, False: トレンド）
    ri_mode: np.ndarray                   # Real vs Imaginary判定（1: Real>Imag=トレンド, 0: Real<=Imag=サイクル）


@njit(fastmath=True, cache=True)
def calculate_correlation_cycle_numba(
    price: np.ndarray,
    period: int = 20,
    trend_threshold: float = 9.0
) -> tuple:
    """
    Correlation Cycle Indicatorを計算する（Numba最適化版）
    
    John Ehlersの論文に基づいて、価格とコサイン/負サイン波との相関を計算し、
    フェーザー角度と市場状態を判定
    
    Args:
        price: 価格配列
        period: 相関計算期間（デフォルト: 20）
        trend_threshold: トレンド判定閾値（角度変化率、デフォルト: 9.0度）
    
    Returns:
        Tuple[np.ndarray, ...]: Real成分, Imaginary成分, 角度, 状態, 変化率, サイクルモード, Real-Imaginary判定
    """
    data_length = len(price)
    
    # 変数の初期化
    real_component = np.zeros(data_length, dtype=np.float64)
    imag_component = np.zeros(data_length, dtype=np.float64)
    angle = np.zeros(data_length, dtype=np.float64)
    state = np.zeros(data_length, dtype=np.float64)
    rate_of_change = np.zeros(data_length, dtype=np.float64)
    cycle_mode = np.zeros(data_length, dtype=np.float64)
    ri_mode = np.zeros(data_length, dtype=np.float64)
    
    # 相関計算
    for i in range(period - 1, data_length):
        # Real成分：価格とコサイン波の相関
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        syy = 0.0
        
        for count in range(period):
            x = price[i - count]
            y = math.cos(2.0 * math.pi * count / period)
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
            syy += y * y
        
        if (period * sxx - sx * sx > 0.0) and (period * syy - sy * sy > 0.0):
            real_component[i] = (period * sxy - sx * sy) / math.sqrt((period * sxx - sx * sx) * (period * syy - sy * sy))
        
        # Imaginary成分：価格と負サイン波の相関
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        syy = 0.0
        
        for count in range(period):
            x = price[i - count]
            y = -math.sin(2.0 * math.pi * count / period)
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
            syy += y * y
        
        if (period * sxx - sx * sx > 0.0) and (period * syy - sy * sy > 0.0):
            imag_component[i] = (period * sxy - sx * sy) / math.sqrt((period * sxx - sx * sx) * (period * syy - sy * sy))
        
        # フェーザー角度の計算（Ehlersの論文に基づく）
        if imag_component[i] != 0.0:
            angle[i] = 90.0 + math.atan(real_component[i] / imag_component[i]) * 180.0 / math.pi
        else:
            angle[i] = angle[i-1] if i > 0 else 0.0
        
        # 象限の補正
        if imag_component[i] > 0.0:
            angle[i] = angle[i] - 180.0
        
        # 角度は逆向きに進まない制約（Ehlersの論文）
        if i > 0:
            angle_diff = angle[i-1] - angle[i]
            if angle_diff < 270.0 and angle[i] < angle[i-1]:
                angle[i] = angle[i-1]
        
        # 角度変化率の計算
        if i > 0:
            rate_of_change[i] = abs(angle[i] - angle[i-1])
        
        # 市場状態の判定（Ehlersの論文に基づく）
        state[i] = 0.0  # デフォルトはサイクルモード
        cycle_mode[i] = 1.0  # デフォルトはサイクルモード
        
        if rate_of_change[i] < trend_threshold:
            # 角度変化が小さい = トレンドモード
            cycle_mode[i] = 0.0  # トレンドモード
            if angle[i] < 0.0:
                state[i] = -1.0  # 下降トレンド
            elif angle[i] >= 0.0:
                state[i] = 1.0   # 上昇トレンド
        
        # Real vs Imaginary判定
        if abs(real_component[i]) > abs(imag_component[i]):
            ri_mode[i] = 1.0  # トレンドモード（Real成分が優勢）
        else:
            ri_mode[i] = 0.0  # サイクルモード（Imaginary成分が優勢）
    
    return real_component, imag_component, angle, state, rate_of_change, cycle_mode, ri_mode


class CorrelationCycleIndicator(Indicator):
    """
    Correlation Cycle Indicator
    
    John Ehlersの「CORRELATION AS A CYCLE INDICATOR」論文に基づいて実装された
    サイクル検出とトレンド・サイクルモード判定インジケーター。
    
    特徴:
    - 価格とコサイン波・負サイン波との相関でReal/Imaginary成分を計算
    - フェーザー角度による精密なサイクル位相検出
    - 角度変化率によるトレンド・サイクルモード自動判定
    - 2つの直交成分使用による6dBのSN比改善
    - Rate-of-Change指標による正確なエントリー・エグジットタイミング
    
    計算手順:
    1. 価格とコサイン波の相関でReal成分を計算
    2. 価格と負サイン波の相関でImaginary成分を計算
    3. arctangent関数でフェーザー角度を計算
    4. 角度変化率でトレンド・サイクルモードを判定
    5. 状態変数を生成（+1: 上昇トレンド, 0: サイクル, -1: 下降トレンド）
    """
    
    def __init__(
        self,
        period: int = 20,                     # 相関計算期間
        src_type: str = 'close',              # ソースタイプ
        trend_threshold: float = 9.0,         # トレンド判定閾値（角度変化率）
        use_theoretical_input: bool = False,  # 理論的入力を使用するか（テスト用）
        theoretical_period: int = 20          # 理論的サイン波の周期（テスト用）
    ):
        """
        コンストラクタ
        
        Args:
            period: 相関計算期間（デフォルト: 20）
            src_type: ソースタイプ（デフォルト: 'close'）
            trend_threshold: トレンド判定閾値（角度変化率、デフォルト: 9.0度）
            use_theoretical_input: 理論的入力を使用するか（デフォルト: False）
            theoretical_period: 理論的サイン波の周期（デフォルト: 20）
        """
        # インジケーター名の作成
        indicator_name = f"CorrelationCycle(period={period}, threshold={trend_threshold:.1f}, {src_type}"
        if use_theoretical_input:
            indicator_name += f", theoretical={theoretical_period}"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.period = period
        self.src_type = src_type.lower()
        self.trend_threshold = trend_threshold
        self.use_theoretical_input = use_theoretical_input
        self.theoretical_period = theoretical_period
        
        # ソースタイプの検証
        try:
            available_sources = PriceSource.get_available_sources()
            if self.src_type not in available_sources:
                raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(available_sources.keys())}")
        except AttributeError:
            # get_available_sources()がない場合は基本的なソースタイプのみチェック
            basic_sources = ['close', 'high', 'low', 'open', 'hl2', 'hlc3', 'ohlc4']
            if self.src_type not in basic_sources:
                raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(basic_sources)}")
        
        # パラメータ検証
        if period <= 0:
            raise ValueError("periodは正の整数である必要があります")
        if trend_threshold <= 0:
            raise ValueError("trend_thresholdは0より大きい必要があります")
        if use_theoretical_input and theoretical_period <= 0:
            raise ValueError("theoretical_periodは正の整数である必要があります")
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 20
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        try:
            # データ情報の取得
            if isinstance(data, pd.DataFrame):
                length = len(data)
                first_val = float(data.iloc[0].get('close', data.iloc[0, -1])) if length > 0 else 0.0
                last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1])) if length > 0 else 0.0
            else:
                length = len(data)
                if length > 0:
                    if data.ndim > 1:
                        first_val = float(data[0, -1])
                        last_val = float(data[-1, -1])
                    else:
                        first_val = float(data[0])
                        last_val = float(data[-1])
                else:
                    first_val = last_val = 0.0
            
            # パラメータ情報
            theoretical_sig = f"{self.theoretical_period}" if self.use_theoretical_input else "None"
            params_sig = f"{self.period}_{self.trend_threshold}_{self.src_type}_{theoretical_sig}"
            
            # 高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.period}_{self.trend_threshold}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> CorrelationCycleIndicatorResult:
        """
        Correlation Cycle Indicatorを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、OHLC + 選択したソースタイプに必要なカラムが必要
        
        Returns:
            CorrelationCycleIndicatorResult: Correlation Cycle Indicatorの計算結果
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ（高速化）
            data_hash = self._get_data_hash(data)
            
            # キャッシュにある場合は取得して返す
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新（最も新しく使われたキーを最後に）
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return CorrelationCycleIndicatorResult(
                    real_component=cached_result.real_component.copy(),
                    imag_component=cached_result.imag_component.copy(),
                    angle=cached_result.angle.copy(),
                    state=cached_result.state.copy(),
                    rate_of_change=cached_result.rate_of_change.copy(),
                    cycle_mode=cached_result.cycle_mode.copy(),
                    ri_mode=cached_result.ri_mode.copy()
                )
            
            # 価格ソースの計算
            if self.use_theoretical_input:
                # テスト用：理論的サイン波を生成
                data_length = len(data) if isinstance(data, pd.DataFrame) else len(data)
                price_source = np.array([
                    math.sin(2.0 * math.pi * i / self.theoretical_period) 
                    for i in range(data_length)
                ], dtype=np.float64)
            else:
                # 実際の価格データを使用
                price_source = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換（float64型で統一）
            price_source = np.asarray(price_source, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price_source)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < self.period + 10:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低{self.period + 10}点以上を推奨します。")
            
            # Correlation Cycle Indicatorの計算（Numba最適化関数を使用）
            real_component, imag_component, angle, state, rate_of_change, cycle_mode, ri_mode = calculate_correlation_cycle_numba(
                price_source, self.period, self.trend_threshold
            )
            
            # 結果の保存
            result = CorrelationCycleIndicatorResult(
                real_component=real_component.copy(),
                imag_component=imag_component.copy(),
                angle=angle.copy(),
                state=state.copy(),
                rate_of_change=rate_of_change.copy(),
                cycle_mode=cycle_mode.copy(),
                ri_mode=ri_mode.copy()
            )
            
            # キャッシュを更新
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = angle  # 基底クラスの要件を満たすため（角度を主要値とする）
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Correlation Cycle Indicator計算中にエラー: {error_msg}\\{stack_trace}")
            
            # エラー時は空の結果を返す
            empty_array = np.array([])
            return CorrelationCycleIndicatorResult(
                real_component=empty_array,
                imag_component=empty_array,
                angle=empty_array,
                state=empty_array,
                rate_of_change=empty_array,
                cycle_mode=empty_array,
                ri_mode=empty_array
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """フェーザー角度を取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.angle.copy()
    
    def get_real_component(self) -> Optional[np.ndarray]:
        """Real成分を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.real_component.copy()
    
    def get_imag_component(self) -> Optional[np.ndarray]:
        """Imaginary成分を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.imag_component.copy()
    
    def get_angle(self) -> Optional[np.ndarray]:
        """フェーザー角度を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.angle.copy()
    
    def get_state(self) -> Optional[np.ndarray]:
        """市場状態を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.state.copy()
    
    def get_rate_of_change(self) -> Optional[np.ndarray]:
        """角度変化率を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.rate_of_change.copy()
    
    def get_cycle_mode(self) -> Optional[np.ndarray]:
        """サイクルモード判定を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.cycle_mode.copy()
    
    def get_orthogonal_components(self) -> Optional[tuple]:
        """直交成分（Real, Imaginary）を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.real_component.copy(), result.imag_component.copy()
    
    def get_ri_mode(self) -> Optional[np.ndarray]:
        """Real vs Imaginaryモード判定を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.ri_mode.copy()
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'period': self.period,
            'src_type': self.src_type,
            'trend_threshold': self.trend_threshold,
            'use_theoretical_input': self.use_theoretical_input,
            'theoretical_period': self.theoretical_period if self.use_theoretical_input else None,
            'phase_range': '(-180° to +180°)',
            'sn_ratio_improvement': '6dB (due to orthogonal components)',
            'description': 'サイクル検出・トレンドモード判定インジケーター（John Ehlersの論文に基づく）'
        }
    
    def reset(self) -> None:
        """インディケーターの状態をリセットする"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_correlation_cycle(
    data: Union[pd.DataFrame, np.ndarray],
    period: int = 20,
    src_type: str = 'close',
    trend_threshold: float = 9.0,
    use_theoretical_input: bool = False,
    theoretical_period: int = 20,
    **kwargs
) -> np.ndarray:
    """
    Correlation Cycle Indicatorの計算（便利関数）
    
    Args:
        data: 価格データ
        period: 相関計算期間
        src_type: ソースタイプ
        trend_threshold: トレンド判定閾値
        use_theoretical_input: 理論的入力を使用するか
        theoretical_period: 理論的サイン波の周期
        **kwargs: その他のパラメータ
        
    Returns:
        フェーザー角度
    """
    indicator = CorrelationCycleIndicator(
        period=period,
        src_type=src_type,
        trend_threshold=trend_threshold,
        use_theoretical_input=use_theoretical_input,
        theoretical_period=theoretical_period,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.angle


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== Correlation Cycle Indicator インジケーターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # サイクルとトレンドが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 60:  # サイクル相場（20期間周期）
            cycle_component = 5.0 * math.sin(2.0 * math.pi * i / 20.0)
            noise = np.random.normal(0, 1.0)
            change = (cycle_component + noise) / base_price
        elif i < 120:  # 上昇トレンド
            change = 0.003 + np.random.normal(0, 0.01)
        elif i < 180:  # サイクル相場（15期間周期）
            cycle_component = 3.0 * math.sin(2.0 * math.pi * i / 15.0)
            noise = np.random.normal(0, 0.8)
            change = (cycle_component + noise) / base_price
        else:  # 下降トレンド
            change = -0.002 + np.random.normal(0, 0.008)
        
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # OHLC データの生成
    data = []
    for i, close in enumerate(prices):
        daily_range = abs(np.random.normal(0, close * 0.01))
        
        high = close + daily_range * np.random.uniform(0.3, 1.0)
        low = close - daily_range * np.random.uniform(0.3, 1.0)
        
        if i == 0:
            open_price = close
        else:
            gap = np.random.normal(0, close * 0.005)
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
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 基本版Correlation Cycle Indicatorをテスト
    print("\\n基本版Correlation Cycle Indicatorをテスト中...")
    cci_basic = CorrelationCycleIndicator(
        period=20,
        src_type='close',
        trend_threshold=9.0,
        use_theoretical_input=False
    )
    try:
        result_basic = cci_basic.calculate(df)
        print(f"  結果の型: {type(result_basic)}")
        print(f"  フェーザー角度配列の形状: {result_basic.angle.shape}")
        print(f"  Real成分配列の形状: {result_basic.real_component.shape}")
        print(f"  Imaginary成分配列の形状: {result_basic.imag_component.shape}")
        print(f"  市場状態配列の形状: {result_basic.state.shape}")
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()
        result_basic = None
    
    if result_basic is not None:
        valid_count = np.sum(~np.isnan(result_basic.angle))
        mean_angle = np.nanmean(result_basic.angle)
        max_angle = np.nanmax(result_basic.angle)
        min_angle = np.nanmin(result_basic.angle)
        
        uptrend_count = np.sum(result_basic.state == 1)
        downtrend_count = np.sum(result_basic.state == -1)
        cycle_count = np.sum(result_basic.state == 0)
        
        cycle_mode_count = np.sum(result_basic.cycle_mode == 1)
        trend_mode_count = np.sum(result_basic.cycle_mode == 0)
        
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  フェーザー角度 - 平均: {mean_angle:.2f}°, 範囲: {min_angle:.2f}° - {max_angle:.2f}°")
        print(f"  上昇トレンド: {uptrend_count}期間 ({uptrend_count/len(df)*100:.1f}%)")
        print(f"  下降トレンド: {downtrend_count}期間 ({downtrend_count/len(df)*100:.1f}%)")
        print(f"  サイクルモード: {cycle_count}期間 ({cycle_count/len(df)*100:.1f}%)")
        print(f"  実際のサイクルモード: {cycle_mode_count}期間 ({cycle_mode_count/len(df)*100:.1f}%)")
        print(f"  実際のトレンドモード: {trend_mode_count}期間 ({trend_mode_count/len(df)*100:.1f}%)")
    else:
        print("  基本版Correlation Cycle Indicatorの計算に失敗しました")
    
    # 理論値テスト版（Figure 1のテスト）
    print("\\n理論値テスト版Correlation Cycle Indicatorをテスト中...")
    cci_theoretical = CorrelationCycleIndicator(
        period=20,
        src_type='close',
        trend_threshold=9.0,
        use_theoretical_input=True,
        theoretical_period=20
    )
    try:
        result_theoretical = cci_theoretical.calculate(df)
        
        valid_count_theo = np.sum(~np.isnan(result_theoretical.angle))
        mean_angle_theo = np.nanmean(result_theoretical.angle)
        mean_real = np.nanmean(result_theoretical.real_component)
        mean_imag = np.nanmean(result_theoretical.imag_component)
        
        print(f"  有効値数: {valid_count_theo}/{len(df)}")
        print(f"  理論値フェーザー角度（平均）: {mean_angle_theo:.2f}°")
        print(f"  Real成分（平均）: {mean_real:.4f}")
        print(f"  Imaginary成分（平均）: {mean_imag:.4f}")
        
        # 理論値では完全な相関が期待される
        real_max = np.nanmax(np.abs(result_theoretical.real_component))
        imag_max = np.nanmax(np.abs(result_theoretical.imag_component))
        print(f"  最大Real成分: {real_max:.4f}")
        print(f"  最大Imaginary成分: {imag_max:.4f}")
        
    except Exception as e:
        print(f"  理論値テスト版でエラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n=== テスト完了 ===")