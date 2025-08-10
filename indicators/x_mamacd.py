#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple, Any
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, types
import traceback
import math

from .indicator import Indicator
from .price_source import PriceSource
from .x_mama import X_MAMA, XMAMAResult


@dataclass
class XMAMACDResult:
    """X_MAMACD（X_MAMAベースMACD）の計算結果"""
    mamacd: np.ndarray           # MAMACD値（MAMA - FAMA）
    signal: np.ndarray           # シグナルライン（MAMACDの指数移動平均）
    histogram: np.ndarray        # ヒストグラム（MAMACD - Signal）
    mama_values: np.ndarray      # X_MAMA値
    fama_values: np.ndarray      # X_FAMA値
    period_values: np.ndarray    # 計算されたPeriod値
    alpha_values: np.ndarray     # 計算されたAlpha値
    phase_values: np.ndarray     # Phase値
    i1_values: np.ndarray        # InPhase component
    q1_values: np.ndarray        # Quadrature component
    filtered_price: np.ndarray   # カルマンフィルター後の価格（使用した場合）


@njit(fastmath=True, cache=True)
def calculate_ema(values: np.ndarray, period: int) -> np.ndarray:
    """
    指数移動平均を計算する（Numba最適化版）
    
    Args:
        values: 計算対象の値配列
        period: EMA期間
    
    Returns:
        EMA値の配列
    """
    length = len(values)
    result = np.full(length, np.nan, dtype=np.float64)
    
    if length == 0 or period <= 0:
        return result
    
    # アルファ値（2/(period+1)）
    alpha = 2.0 / (period + 1.0)
    
    # 最初の有効な値を見つける
    first_valid_idx = -1
    for i in range(length):
        if not np.isnan(values[i]):
            result[i] = values[i]
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return result
    
    # EMAの計算
    for i in range(first_valid_idx + 1, length):
        if not np.isnan(values[i]):
            if not np.isnan(result[i-1]):
                result[i] = alpha * values[i] + (1.0 - alpha) * result[i-1]
            else:
                result[i] = values[i]
    
    return result


@njit(fastmath=True, cache=True)
def calculate_adaptive_ema(values: np.ndarray, alpha_values: np.ndarray) -> np.ndarray:
    """
    適応型指数移動平均を計算する（X_MAMAのアルファ値を使用）
    
    Args:
        values: 計算対象の値配列
        alpha_values: X_MAMAから得られるアルファ値配列
    
    Returns:
        適応型EMA値の配列
    """
    length = len(values)
    result = np.full(length, np.nan, dtype=np.float64)
    
    if length == 0 or len(alpha_values) != length:
        return result
    
    # 最初の有効な値を見つける
    first_valid_idx = -1
    for i in range(length):
        if not np.isnan(values[i]) and not np.isnan(alpha_values[i]):
            result[i] = values[i]
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return result
    
    # 適応型EMAの計算
    for i in range(first_valid_idx + 1, length):
        if not np.isnan(values[i]) and not np.isnan(alpha_values[i]):
            if not np.isnan(result[i-1]):
                # X_MAMAのアルファ値を使用してEMAを計算
                result[i] = alpha_values[i] * values[i] + (1.0 - alpha_values[i]) * result[i-1]
            else:
                result[i] = values[i]
    
    return result


class X_MAMACD(Indicator):
    """
    X_MAMACD (eXtended MAMA-based MACD) インジケーター
    
    X_MAMAとX_FAMAを使用したMACDの実装：
    - MAMACD = X_MAMA - X_FAMA
    - Signal Line = MAMACDの指数移動平均または適応型移動平均
    - Histogram = MAMACD - Signal Line
    
    特徴:
    - 市場サイクルの変化に適応
    - トレンド強度に応じて応答速度を調整
    - X_MAMAのアルファ値を使用した適応型シグナルライン
    - カルマンフィルターとゼロラグ処理の統合
    """
    
    def __init__(
        self,
        # X_MAMAパラメータ
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.05,              # 低速制限値
        src_type: str = 'hlc3',                # ソースタイプ
        # シグナルラインパラメータ
        signal_period: int = 9,                # シグナルライン期間
        use_adaptive_signal: bool = True,      # 適応型シグナルラインを使用するか
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,       # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,    # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True              # ゼロラグ処理を使用するか
    ):
        """
        コンストラクタ
        
        Args:
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ
            signal_period: シグナルライン期間（デフォルト: 9）
            use_adaptive_signal: 適応型シグナルラインを使用するか（デフォルト: True）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
        """
        # インジケーター名の作成
        indicator_name = f"X_MAMACD(fast={fast_limit}, slow={slow_limit}, {src_type}, signal={signal_period}"
        if use_adaptive_signal:
            indicator_name += ", adaptive_signal=True"
        if use_kalman_filter:
            indicator_name += f", kalman={kalman_filter_type}"
        if use_zero_lag:
            indicator_name += ", zero_lag=True"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        self.src_type = src_type.lower()
        self.signal_period = signal_period
        self.use_adaptive_signal = use_adaptive_signal
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_observation_noise = kalman_observation_noise
        self.use_zero_lag = use_zero_lag
        
        # パラメータ検証
        if signal_period <= 0:
            raise ValueError("signal_periodは0より大きい必要があります")
        
        # X_MAMAインジケーターの初期化
        self.x_mama = X_MAMA(
            fast_limit=fast_limit,
            slow_limit=slow_limit,
            src_type=src_type,
            use_kalman_filter=use_kalman_filter,
            kalman_filter_type=kalman_filter_type,
            kalman_process_noise=kalman_process_noise,
            kalman_observation_noise=kalman_observation_noise,
            use_zero_lag=use_zero_lag
        )
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 20
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """
        データのハッシュ値を計算してキャッシュに使用する（超高速版）
        
        Args:
            data: 価格データ
            
        Returns:
            データハッシュ文字列
        """
        # 超高速化のため最小限のサンプリング
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
            
            # 最小限のパラメータ情報
            kalman_sig = f"{self.kalman_filter_type}_{self.kalman_process_noise}" if self.use_kalman_filter else "None"
            zero_lag_sig = str(self.use_zero_lag)
            params_sig = f"{self.fast_limit}_{self.slow_limit}_{self.src_type}_{self.signal_period}_{self.use_adaptive_signal}_{kalman_sig}_{zero_lag_sig}"
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.fast_limit}_{self.slow_limit}_{self.signal_period}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> XMAMACDResult:
        """
        X_MAMACDを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、OHLC + 選択したソースタイプに必要なカラムが必要
        
        Returns:
            XMAMACDResult: X_MAMACDの値と計算中間値を含む結果
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
                return XMAMACDResult(
                    mamacd=cached_result.mamacd.copy(),
                    signal=cached_result.signal.copy(),
                    histogram=cached_result.histogram.copy(),
                    mama_values=cached_result.mama_values.copy(),
                    fama_values=cached_result.fama_values.copy(),
                    period_values=cached_result.period_values.copy(),
                    alpha_values=cached_result.alpha_values.copy(),
                    phase_values=cached_result.phase_values.copy(),
                    i1_values=cached_result.i1_values.copy(),
                    q1_values=cached_result.q1_values.copy(),
                    filtered_price=cached_result.filtered_price.copy()
                )
            
            # X_MAMAの計算
            x_mama_result = self.x_mama.calculate(data)
            
            # データ長の検証
            data_length = len(x_mama_result.mama_values)
            if data_length == 0:
                raise ValueError("X_MAMA計算結果が空です")
            
            # MAMACD値の計算（MAMA - FAMA）
            mamacd = x_mama_result.mama_values - x_mama_result.fama_values
            
            # シグナルラインの計算
            if self.use_adaptive_signal:
                # 適応型シグナルライン（X_MAMAのアルファ値を使用）
                signal = calculate_adaptive_ema(mamacd, x_mama_result.alpha_values)
                self.logger.debug("適応型シグナルラインを計算しました")
            else:
                # 標準EMAシグナルライン
                signal = calculate_ema(mamacd, self.signal_period)
                self.logger.debug(f"標準EMAシグナルライン（期間{self.signal_period}）を計算しました")
            
            # ヒストグラムの計算（MAMACD - Signal）
            histogram = mamacd - signal
            
            # 結果の保存（参照問題を避けるため必要な部分だけコピー）
            result = XMAMACDResult(
                mamacd=mamacd.copy(),
                signal=signal.copy(),
                histogram=histogram.copy(),
                mama_values=x_mama_result.mama_values.copy(),
                fama_values=x_mama_result.fama_values.copy(),
                period_values=x_mama_result.period_values.copy(),
                alpha_values=x_mama_result.alpha_values.copy(),
                phase_values=x_mama_result.phase_values.copy(),
                i1_values=x_mama_result.i1_values.copy(),
                q1_values=x_mama_result.q1_values.copy(),
                filtered_price=x_mama_result.filtered_price.copy()
            )
            
            # キャッシュを更新
            # キャッシュサイズ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                # 最も古いキャッシュを削除
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = mamacd  # 基底クラスの要件を満たすため（MAMACD値をメインとする）
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"X_MAMACD計算中にエラー: {error_msg}\\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = XMAMACDResult(
                mamacd=np.array([]),
                signal=np.array([]),
                histogram=np.array([]),
                mama_values=np.array([]),
                fama_values=np.array([]),
                period_values=np.array([]),
                alpha_values=np.array([]),
                phase_values=np.array([]),
                i1_values=np.array([]),
                q1_values=np.array([]),
                filtered_price=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """MAMACD値のみを取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.mamacd.copy()
    
    def get_mamacd_values(self) -> Optional[np.ndarray]:
        """
        MAMACD値を取得する
        
        Returns:
            np.ndarray: MAMACD値
        """
        return self.get_values()
    
    def get_signal_values(self) -> Optional[np.ndarray]:
        """
        シグナルライン値を取得する
        
        Returns:
            np.ndarray: シグナルライン値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.signal.copy()
    
    def get_histogram_values(self) -> Optional[np.ndarray]:
        """
        ヒストグラム値を取得する
        
        Returns:
            np.ndarray: ヒストグラム値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.histogram.copy()
    
    def get_mama_fama_values(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        X_MAMAとX_FAMA値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X_MAMA値, X_FAMA値)
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.mama_values.copy(), result.fama_values.copy()
    
    def get_all_values(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        すべての主要な値を取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (MAMACD, Signal, Histogram)
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.mamacd.copy(), result.signal.copy(), result.histogram.copy()
    
    def get_crossover_signals(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        MAMACD/Signalラインのクロスオーバーシグナルを取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (bullish_crossover, bearish_crossover)
                各要素はブール配列で、Trueの位置でクロスオーバーが発生
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
        
        mamacd = result.mamacd
        signal = result.signal
        
        length = len(mamacd)
        bullish_crossover = np.zeros(length, dtype=bool)
        bearish_crossover = np.zeros(length, dtype=bool)
        
        # クロスオーバーの検出
        for i in range(1, length):
            if not np.isnan(mamacd[i]) and not np.isnan(signal[i]) and not np.isnan(mamacd[i-1]) and not np.isnan(signal[i-1]):
                # Bullish crossover: MAMACD crosses above Signal
                if mamacd[i-1] <= signal[i-1] and mamacd[i] > signal[i]:
                    bullish_crossover[i] = True
                
                # Bearish crossover: MAMACD crosses below Signal
                elif mamacd[i-1] >= signal[i-1] and mamacd[i] < signal[i]:
                    bearish_crossover[i] = True
        
        return bullish_crossover, bearish_crossover
    
    def get_zero_line_crossover_signals(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        MAMACDのゼロラインクロスオーバーシグナルを取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (bullish_zero_cross, bearish_zero_cross)
                各要素はブール配列で、Trueの位置でクロスオーバーが発生
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
        
        mamacd = result.mamacd
        
        length = len(mamacd)
        bullish_zero_cross = np.zeros(length, dtype=bool)
        bearish_zero_cross = np.zeros(length, dtype=bool)
        
        # ゼロラインクロスオーバーの検出
        for i in range(1, length):
            if not np.isnan(mamacd[i]) and not np.isnan(mamacd[i-1]):
                # Bullish zero line crossover: MAMACD crosses above zero
                if mamacd[i-1] <= 0.0 and mamacd[i] > 0.0:
                    bullish_zero_cross[i] = True
                
                # Bearish zero line crossover: MAMACD crosses below zero
                elif mamacd[i-1] >= 0.0 and mamacd[i] < 0.0:
                    bearish_zero_cross[i] = True
        
        return bullish_zero_cross, bearish_zero_cross
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'fast_limit': self.fast_limit,
            'slow_limit': self.slow_limit,
            'src_type': self.src_type,
            'signal_period': self.signal_period,
            'use_adaptive_signal': self.use_adaptive_signal,
            'use_kalman_filter': self.use_kalman_filter,
            'kalman_filter_type': self.kalman_filter_type if self.use_kalman_filter else None,
            'kalman_process_noise': self.kalman_process_noise if self.use_kalman_filter else None,
            'kalman_observation_noise': self.kalman_observation_noise if self.use_kalman_filter else None,
            'use_zero_lag': self.use_zero_lag,
            'description': 'X_MAMAベースのMACD（適応型シグナルライン対応）'
        }
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        if hasattr(self.x_mama, 'reset'):
            self.x_mama.reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_x_mamacd(
    data: Union[pd.DataFrame, np.ndarray],
    fast_limit: float = 0.5,
    slow_limit: float = 0.05,
    src_type: str = 'hlc3',
    signal_period: int = 9,
    use_adaptive_signal: bool = True,
    use_kalman_filter: bool = False,
    kalman_filter_type: str = 'unscented',
    kalman_process_noise: float = 0.01,
    kalman_observation_noise: float = 0.001,
    use_zero_lag: bool = True,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X_MAMACDの計算（便利関数）
    
    Args:
        data: 価格データ
        fast_limit: 高速制限値
        slow_limit: 低速制限値
        src_type: ソースタイプ
        signal_period: シグナルライン期間
        use_adaptive_signal: 適応型シグナルラインを使用するか
        use_kalman_filter: カルマンフィルターを使用するか
        kalman_filter_type: カルマンフィルタータイプ
        kalman_process_noise: プロセスノイズ
        kalman_observation_noise: 観測ノイズ
        use_zero_lag: ゼロラグ処理を使用するか
        **kwargs: その他のパラメータ
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (MAMACD, Signal, Histogram)
    """
    indicator = X_MAMACD(
        fast_limit=fast_limit,
        slow_limit=slow_limit,
        src_type=src_type,
        signal_period=signal_period,
        use_adaptive_signal=use_adaptive_signal,
        use_kalman_filter=use_kalman_filter,
        kalman_filter_type=kalman_filter_type,
        kalman_process_noise=kalman_process_noise,
        kalman_observation_noise=kalman_observation_noise,
        use_zero_lag=use_zero_lag,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.mamacd, result.signal, result.histogram


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== X_MAMACD インジケーターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # トレンドとレンジが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 上昇トレンド相場
            change = 0.003 + np.random.normal(0, 0.01)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.008)
        elif i < 150:  # 下降トレンド相場
            change = -0.002 + np.random.normal(0, 0.015)
        else:  # レンジ相場
            change = np.random.normal(0, 0.006)
        
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
    
    # 基本版X_MAMACDをテスト
    print("\\n基本版X_MAMACDをテスト中...")
    x_mamacd_basic = X_MAMACD(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        signal_period=9,
        use_adaptive_signal=False,
        use_kalman_filter=False,
        use_zero_lag=False
    )
    try:
        result_basic = x_mamacd_basic.calculate(df)
        print(f"  X_MAMACD結果の型: {type(result_basic)}")
        print(f"  MAMACD配列の形状: {result_basic.mamacd.shape}")
        print(f"  Signal配列の形状: {result_basic.signal.shape}")
        print(f"  Histogram配列の形状: {result_basic.histogram.shape}")
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()
        result_basic = None
    
    if result_basic is not None:
        valid_count = np.sum(~np.isnan(result_basic.mamacd))
        mean_mamacd = np.nanmean(result_basic.mamacd)
        mean_signal = np.nanmean(result_basic.signal)
        mean_histogram = np.nanmean(result_basic.histogram)
        
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  平均MAMACD: {mean_mamacd:.6f}")
        print(f"  平均Signal: {mean_signal:.6f}")
        print(f"  平均Histogram: {mean_histogram:.6f}")
        
        # クロスオーバーシグナルをテスト
        bullish_cross, bearish_cross = result_basic.crossover_signals if hasattr(result_basic, 'crossover_signals') else x_mamacd_basic.get_crossover_signals()
        if bullish_cross is not None:
            print(f"  Bullishクロスオーバー: {np.sum(bullish_cross)}回")
            print(f"  Bearishクロスオーバー: {np.sum(bearish_cross)}回")
    else:
        print("  基本版X_MAMACDの計算に失敗しました")
    
    # 適応型シグナルライン版をテスト
    print("\\n適応型シグナルライン版X_MAMACDをテスト中...")
    x_mamacd_adaptive = X_MAMACD(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        signal_period=9,
        use_adaptive_signal=True,
        use_kalman_filter=False,
        use_zero_lag=True
    )
    try:
        result_adaptive = x_mamacd_adaptive.calculate(df)
        
        valid_count_adaptive = np.sum(~np.isnan(result_adaptive.mamacd))
        mean_mamacd_adaptive = np.nanmean(result_adaptive.mamacd)
        mean_signal_adaptive = np.nanmean(result_adaptive.signal)
        mean_histogram_adaptive = np.nanmean(result_adaptive.histogram)
        
        print(f"  有効値数: {valid_count_adaptive}/{len(df)}")
        print(f"  平均MAMACD（適応型）: {mean_mamacd_adaptive:.6f}")
        print(f"  平均Signal（適応型）: {mean_signal_adaptive:.6f}")
        print(f"  平均Histogram（適応型）: {mean_histogram_adaptive:.6f}")
        
        # 比較統計
        if result_basic is not None and valid_count > 0 and valid_count_adaptive > 0:
            min_length = min(valid_count, valid_count_adaptive)
            correlation = np.corrcoef(
                result_basic.mamacd[~np.isnan(result_basic.mamacd)][-min_length:],
                result_adaptive.mamacd[~np.isnan(result_adaptive.mamacd)][-min_length:]
            )[0, 1]
            print(f"  基本版と適応型版のMAMACD相関: {correlation:.4f}")
    except Exception as e:
        print(f"  適応型シグナルライン版でエラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n=== テスト完了 ===")