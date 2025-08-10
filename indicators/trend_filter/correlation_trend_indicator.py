#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import njit, float64
import math

from ..indicator import Indicator
from ..price_source import PriceSource


@dataclass
class CorrelationTrendIndicatorResult:
    """Correlation Trend Indicatorの計算結果"""
    values: np.ndarray                    # 相関係数値（-1から+1の範囲）
    trend_signal: np.ndarray             # トレンドシグナル（+1: 上昇, -1: 下降, 0: 横這い）
    trend_strength: np.ndarray           # トレンド強度（0から1の範囲）
    smoothed_values: np.ndarray          # 平滑化された相関値（オプション）


@njit(fastmath=True, cache=True)
def calculate_correlation_trend_numba(
    price: np.ndarray,
    length: int = 20
) -> tuple:
    """
    Correlation Trend Indicatorを計算する（Numba最適化版）
    
    John Ehlersの論文に基づいて、価格と正の傾きを持つ直線との相関係数を計算
    
    Args:
        price: 価格配列
        length: 相関計算期間（デフォルト: 20）
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 相関値, トレンドシグナル, トレンド強度
    """
    data_length = len(price)
    
    # 変数の初期化
    correlation = np.zeros(data_length, dtype=np.float64)
    trend_signal = np.zeros(data_length, dtype=np.float64)
    trend_strength = np.zeros(data_length, dtype=np.float64)
    
    # 相関計算
    for i in range(length - 1, data_length):
        # Spearman相関の変数初期化
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        syy = 0.0
        
        # 相関計算ループ（Ehlersのコードに基づく）
        for count in range(length):
            # X = 価格データ（過去から現在へ）
            x = price[i - count]
            # Y = 直線（負の傾き、時間は逆方向にカウント）
            y = -count
            
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
            syy += y * y
        
        # 相関係数の計算
        denominator_x = length * sxx - sx * sx
        denominator_y = length * syy - sy * sy
        
        if denominator_x > 0.0 and denominator_y > 0.0:
            numerator = length * sxy - sx * sy
            denominator = math.sqrt(denominator_x * denominator_y)
            
            if denominator != 0.0:
                correlation[i] = numerator / denominator
            else:
                correlation[i] = 0.0
        else:
            correlation[i] = 0.0
        
        # トレンドシグナルの生成
        corr_val = correlation[i]
        
        # トレンド判定（Ehlersの論文に基づく）
        if corr_val > 0.3:  # 明確な上昇トレンド
            trend_signal[i] = 1.0
            trend_strength[i] = min(1.0, abs(corr_val))
        elif corr_val < -0.3:  # 明確な下降トレンド
            trend_signal[i] = -1.0
            trend_strength[i] = min(1.0, abs(corr_val))
        else:  # 横這いまたは弱いトレンド
            trend_signal[i] = 0.0
            trend_strength[i] = abs(corr_val)
    
    return correlation, trend_signal, trend_strength


@njit(fastmath=True, cache=True)
def smooth_correlation_numba(
    correlation: np.ndarray,
    smooth_length: int = 5
) -> np.ndarray:
    """
    相関値を平滑化する（Numba最適化版）
    
    Args:
        correlation: 相関値配列
        smooth_length: 平滑化期間
    
    Returns:
        平滑化された相関値
    """
    data_length = len(correlation)
    smoothed = np.zeros(data_length, dtype=np.float64)
    
    for i in range(smooth_length - 1, data_length):
        sum_val = 0.0
        for j in range(smooth_length):
            sum_val += correlation[i - j]
        smoothed[i] = sum_val / smooth_length
    
    return smoothed


class CorrelationTrendIndicator(Indicator):
    """
    Correlation Trend Indicator
    
    John Ehlersの「CORRELATION AS A TREND INDICATOR」論文に基づいて実装された
    相関ベースのトレンド判定インジケーター。
    
    特徴:
    - 価格と正の傾きを持つ直線との相関係数を計算
    - -1から+1の範囲でトレンドを示す
    - +1に近い：強い上昇トレンド、-1に近い：強い下降トレンド、0に近い：横這い
    - ラグは相関期間の約半分
    - トレード期間に応じて相関期間を調整可能
    
    計算手順:
    1. 指定された期間の価格データを取得
    2. 正の傾きを持つ直線を生成（Y = -count）
    3. Spearman相関係数を計算
    4. トレンドシグナルと強度を判定
    5. オプションで平滑化を適用
    """
    
    def __init__(
        self,
        length: int = 20,                     # 相関計算期間
        src_type: str = 'close',              # ソースタイプ
        trend_threshold: float = 0.3,         # トレンド判定閾値
        enable_smoothing: bool = False,       # 平滑化を有効にするか
        smooth_length: int = 5                # 平滑化期間
    ):
        """
        コンストラクタ
        
        Args:
            length: 相関計算期間（デフォルト: 20）
            src_type: ソースタイプ（デフォルト: 'close'）
            trend_threshold: トレンド判定閾値（デフォルト: 0.3）
            enable_smoothing: 平滑化を有効にするか（デフォルト: False）
            smooth_length: 平滑化期間（デフォルト: 5）
        """
        # インジケーター名の作成
        indicator_name = f"CorrelationTrend(length={length}, threshold={trend_threshold:.1f}, {src_type}"
        if enable_smoothing:
            indicator_name += f", smooth={smooth_length}"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.length = length
        self.src_type = src_type.lower()
        self.trend_threshold = trend_threshold
        self.enable_smoothing = enable_smoothing
        self.smooth_length = smooth_length
        
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
        if length <= 0:
            raise ValueError("lengthは正の整数である必要があります")
        if not 0.0 <= trend_threshold <= 1.0:
            raise ValueError("trend_thresholdは0.0から1.0の間である必要があります")
        if enable_smoothing and smooth_length <= 0:
            raise ValueError("smooth_lengthは正の整数である必要があります")
        
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
            smooth_sig = f"{self.smooth_length}" if self.enable_smoothing else "None"
            params_sig = f"{self.length}_{self.trend_threshold}_{self.src_type}_{smooth_sig}"
            
            # 高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.length}_{self.trend_threshold}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> CorrelationTrendIndicatorResult:
        """
        Correlation Trend Indicatorを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、OHLC + 選択したソースタイプに必要なカラムが必要
        
        Returns:
            CorrelationTrendIndicatorResult: Correlation Trend Indicatorの計算結果
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
                return CorrelationTrendIndicatorResult(
                    values=cached_result.values.copy(),
                    trend_signal=cached_result.trend_signal.copy(),
                    trend_strength=cached_result.trend_strength.copy(),
                    smoothed_values=cached_result.smoothed_values.copy()
                )
            
            # 価格ソースの計算
            price_source = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換（float64型で統一）
            price_source = np.asarray(price_source, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price_source)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < self.length + 10:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低{self.length + 10}点以上を推奨します。")
            
            # Correlation Trend Indicatorの計算（Numba最適化関数を使用）
            correlation_values, trend_signal, trend_strength = calculate_correlation_trend_numba(
                price_source, self.length
            )
            
            # 平滑化（オプション）
            smoothed_values = np.zeros_like(correlation_values)
            if self.enable_smoothing:
                smoothed_values = smooth_correlation_numba(correlation_values, self.smooth_length)
            else:
                smoothed_values = correlation_values.copy()
            
            # 結果の保存
            result = CorrelationTrendIndicatorResult(
                values=correlation_values.copy(),
                trend_signal=trend_signal.copy(),
                trend_strength=trend_strength.copy(),
                smoothed_values=smoothed_values.copy()
            )
            
            # キャッシュを更新
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = correlation_values  # 基底クラスの要件を満たすため
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Correlation Trend Indicator計算中にエラー: {error_msg}\\{stack_trace}")
            
            # エラー時は空の結果を返す
            empty_array = np.array([])
            return CorrelationTrendIndicatorResult(
                values=empty_array,
                trend_signal=empty_array,
                trend_strength=empty_array,
                smoothed_values=empty_array
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """相関値を取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.values.copy()
    
    def get_trend_signal(self) -> Optional[np.ndarray]:
        """トレンドシグナルを取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.trend_signal.copy()
    
    def get_trend_strength(self) -> Optional[np.ndarray]:
        """トレンド強度を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.trend_strength.copy()
    
    def get_smoothed_values(self) -> Optional[np.ndarray]:
        """平滑化された相関値を取得する"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.smoothed_values.copy()
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'length': self.length,
            'src_type': self.src_type,
            'trend_threshold': self.trend_threshold,
            'enable_smoothing': self.enable_smoothing,
            'smooth_length': self.smooth_length if self.enable_smoothing else None,
            'lag_estimate': self.length // 2,  # 推定ラグ
            'description': '相関ベーストレンド判定インジケーター（John Ehlersの論文に基づく）'
        }
    
    def reset(self) -> None:
        """インディケーターの状態をリセットする"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_correlation_trend(
    data: Union[pd.DataFrame, np.ndarray],
    length: int = 20,
    src_type: str = 'close',
    trend_threshold: float = 0.3,
    enable_smoothing: bool = False,
    smooth_length: int = 5,
    **kwargs
) -> np.ndarray:
    """
    Correlation Trend Indicatorの計算（便利関数）
    
    Args:
        data: 価格データ
        length: 相関計算期間
        src_type: ソースタイプ
        trend_threshold: トレンド判定閾値
        enable_smoothing: 平滑化を有効にするか
        smooth_length: 平滑化期間
        **kwargs: その他のパラメータ
        
    Returns:
        相関値
    """
    indicator = CorrelationTrendIndicator(
        length=length,
        src_type=src_type,
        trend_threshold=trend_threshold,
        enable_smoothing=enable_smoothing,
        smooth_length=smooth_length,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== Correlation Trend Indicator インジケーターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # トレンドとレンジが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 上昇トレンド
            change = 0.002 + np.random.normal(0, 0.01)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.008)
        elif i < 150:  # 下降トレンド
            change = -0.003 + np.random.normal(0, 0.012)
        else:  # 上昇トレンド
            change = 0.004 + np.random.normal(0, 0.015)
        
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
    
    # 基本版Correlation Trend Indicatorをテスト
    print("\\n基本版Correlation Trend Indicatorをテスト中...")
    cti_basic = CorrelationTrendIndicator(
        length=20,
        src_type='close',
        trend_threshold=0.3,
        enable_smoothing=False
    )
    try:
        result_basic = cti_basic.calculate(df)
        print(f"  結果の型: {type(result_basic)}")
        print(f"  相関値配列の形状: {result_basic.values.shape}")
        print(f"  トレンドシグナル配列の形状: {result_basic.trend_signal.shape}")
        print(f"  トレンド強度配列の形状: {result_basic.trend_strength.shape}")
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()
        result_basic = None
    
    if result_basic is not None:
        valid_count = np.sum(~np.isnan(result_basic.values))
        mean_corr = np.nanmean(result_basic.values)
        max_corr = np.nanmax(result_basic.values)
        min_corr = np.nanmin(result_basic.values)
        uptrend_count = np.sum(result_basic.trend_signal == 1)
        downtrend_count = np.sum(result_basic.trend_signal == -1)
        sideways_count = np.sum(result_basic.trend_signal == 0)
        
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  相関値 - 平均: {mean_corr:.4f}, 範囲: {min_corr:.4f} - {max_corr:.4f}")
        print(f"  上昇トレンド: {uptrend_count}期間 ({uptrend_count/len(df)*100:.1f}%)")
        print(f"  下降トレンド: {downtrend_count}期間 ({downtrend_count/len(df)*100:.1f}%)")
        print(f"  横這い: {sideways_count}期間 ({sideways_count/len(df)*100:.1f}%)")
    else:
        print("  基本版Correlation Trend Indicatorの計算に失敗しました")
    
    # 平滑化版をテスト
    print("\\n平滑化版Correlation Trend Indicatorをテスト中...")
    cti_smooth = CorrelationTrendIndicator(
        length=20,
        src_type='close',
        trend_threshold=0.3,
        enable_smoothing=True,
        smooth_length=5
    )
    try:
        result_smooth = cti_smooth.calculate(df)
        
        valid_count_smooth = np.sum(~np.isnan(result_smooth.smoothed_values))
        mean_corr_smooth = np.nanmean(result_smooth.smoothed_values)
        
        print(f"  有効値数: {valid_count_smooth}/{len(df)}")
        print(f"  平滑化相関値（平均）: {mean_corr_smooth:.4f}")
        
        # 比較統計
        if result_basic is not None and valid_count > 0 and valid_count_smooth > 0:
            min_length = min(valid_count, valid_count_smooth)
            correlation = np.corrcoef(
                result_basic.values[~np.isnan(result_basic.values)][-min_length:],
                result_smooth.smoothed_values[~np.isnan(result_smooth.smoothed_values)][-min_length:]
            )[0, 1]
            print(f"  基本版と平滑化版の相関: {correlation:.4f}")
    except Exception as e:
        print(f"  平滑化版でエラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n=== テスト完了 ===")