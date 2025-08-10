#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Hyper Donchian - 80-20パーセンタイルベースドンチャンチャネル** 🎯

従来のドンチャンチャネル（最高値・最安値）ではなく、
80パーセンタイル高値と20パーセンタイル安値を使用した改良版ドンチャンチャネル。

🌟 **主要機能:**
1. **Min/Max範囲内80-20パーセンタイル**: n期間の最高値・最安値範囲内での80%・20%位置を計算
2. **HyperER動的適応**: 市場効率性に基づく期間適応
3. **ミッドライン**: (Min+80%Range値 + Min+20%Range値) / 2
4. **上部バンド**: Min + 0.8 * (Max - Min)
5. **下部バンド**: Min + 0.2 * (Max - Min)

📊 **特徴:**
- n期間のmin/max範囲を基準としたパーセンタイル計算
- 従来ドンチャンよりも安定したチャネル形成
- HyperER動的適応でマーケット効率性に応じた期間調整
"""

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import jit, njit, float64

from ..indicator import Indicator
from ..price_source import PriceSource


@dataclass
class HyperDonchianResult:
    """Hyperドンチャンの計算結果"""
    values: np.ndarray               # Hyperドンチャンミッドライン値
    upper_band: np.ndarray          # 上部バンド（80パーセンタイル）
    lower_band: np.ndarray          # 下部バンド（20パーセンタイル）
    band_width: np.ndarray          # バンド幅（上部-下部）


@njit(fastmath=True, cache=True)
def calculate_dynamic_period_vec(er_values: np.ndarray, period_min: float, period_max: float) -> np.ndarray:
    """
    HyperER値に基づいて動的にHyperドンチャン期間を計算する（ベクトル化版）
    
    Args:
        er_values: HyperER値の配列（0-1の範囲）
        period_min: 最小期間（ER高い時に使用）
        period_max: 最大期間（ER低い時に使用）
    
    Returns:
        動的期間値配列
    """
    length = len(er_values)
    dynamic_periods = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        er = er_values[i] if not np.isnan(er_values[i]) else 0.0
        
        # ER高い（効率的）→ 期間小さく（period_minに近づく）
        # ER低い（非効率）→ 期間大きく（period_maxに近づく）
        dynamic_periods[i] = period_min + (1.0 - er) * (period_max - period_min)
    
    return dynamic_periods


@njit(fastmath=True, cache=True)
def calculate_range_based_percentile_numba(values: np.ndarray, percentile: float) -> float:
    """
    n期間のmin/max範囲内での指定パーセンタイル値を計算する（Numba最適化版）
    
    Args:
        values: 値の配列
        percentile: パーセンタイル（0-100）
    
    Returns:
        min/max範囲内でのパーセンタイル値
    """
    if len(values) == 0:
        return np.nan
    
    # 有効な値のみを抽出
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return np.nan
    
    if len(valid_values) == 1:
        return valid_values[0]
    
    # n期間の最高値と最安値を取得
    period_max = np.max(valid_values)
    period_min = np.min(valid_values)
    
    # 範囲が0の場合は最高値/最安値を返す
    if period_max == period_min:
        return period_max
    
    # min/max範囲内でのパーセンタイル位置を計算
    # 80%の場合：min + 0.8 * (max - min)
    # 20%の場合：min + 0.2 * (max - min)
    percentile_ratio = percentile / 100.0
    range_based_value = period_min + percentile_ratio * (period_max - period_min)
    
    return range_based_value


@njit(fastmath=True, cache=True)
def calculate_hyper_donchian_hyper_er(
    prices: np.ndarray,
    dynamic_periods: np.ndarray
) -> tuple:
    """
    HyperER動的適応Hyperドンチャンを計算する（Numba最適化版）
    
    Args:
        prices: 価格配列
        dynamic_periods: 動的期間配列
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        (ミッドライン, 上部バンド, 下部バンド, バンド幅)
    """
    length = len(prices)
    midline = np.full(length, np.nan, dtype=np.float64)
    upper_band = np.full(length, np.nan, dtype=np.float64)
    lower_band = np.full(length, np.nan, dtype=np.float64)
    band_width = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        # 動的期間を取得（最小期間を確保）
        current_period = max(5, int(dynamic_periods[i])) if i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]) else 20
        
        if i >= current_period - 1:
            # 期間内の価格データを取得
            start_idx = i - current_period + 1
            period_prices = prices[start_idx:i + 1]
            
            # NaN値を除外
            valid_prices = period_prices[~np.isnan(period_prices)]
            
            if len(valid_prices) >= current_period // 2:  # 有効なデータが半分以上の場合
                # min/max範囲内での80パーセンタイルと20パーセンタイルを計算
                upper_80 = calculate_range_based_percentile_numba(valid_prices, 80.0)
                lower_20 = calculate_range_based_percentile_numba(valid_prices, 20.0)
                
                if not np.isnan(upper_80) and not np.isnan(lower_20):
                    # Hyperドンチャンミッドライン = (Min+80%Range値 + Min+20%Range値) / 2
                    midline[i] = (upper_80 + lower_20) / 2.0
                    upper_band[i] = upper_80
                    lower_band[i] = lower_20
                    band_width[i] = upper_80 - lower_20
    
    return midline, upper_band, lower_band, band_width


@njit(fastmath=True, cache=True)
def calculate_hyper_donchian_numba(
    prices: np.ndarray,
    period: int
) -> tuple:
    """
    Hyperドンチャンを計算する（Numba最適化版）
    
    Args:
        prices: 価格配列
        period: 計算期間
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        (ミッドライン, 上部バンド, 下部バンド, バンド幅)
    """
    length = len(prices)
    midline = np.full(length, np.nan, dtype=np.float64)
    upper_band = np.full(length, np.nan, dtype=np.float64)
    lower_band = np.full(length, np.nan, dtype=np.float64)
    band_width = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(period - 1, length):
        # 期間内の価格データを取得
        start_idx = i - period + 1
        period_prices = prices[start_idx:i + 1]
        
        # NaN値を除外
        valid_prices = period_prices[~np.isnan(period_prices)]
        
        if len(valid_prices) >= period // 2:  # 有効なデータが半分以上の場合
            # min/max範囲内での80パーセンタイルと20パーセンタイルを計算
            upper_80 = calculate_range_based_percentile_numba(valid_prices, 80.0)
            lower_20 = calculate_range_based_percentile_numba(valid_prices, 20.0)
            
            if not np.isnan(upper_80) and not np.isnan(lower_20):
                # Hyperドンチャンミッドライン = (Min+80%Range値 + Min+20%Range値) / 2
                midline[i] = (upper_80 + lower_20) / 2.0
                upper_band[i] = upper_80
                lower_band[i] = lower_20
                band_width[i] = upper_80 - lower_20
    
    return midline, upper_band, lower_band, band_width


class HyperDonchian(Indicator):
    """
    Hyperドンチャン（80-20パーセンタイル版ドンチャンチャネル）インジケーター
    
    従来の最高値・最安値ベースではなく、n期間のmin/max範囲内での80%・20%位置を使用。
    純粋なパーセンタイル値ではなく、期間内の範囲に基づいた計算で安定性を向上。
    
    特徴:
    - n期間のmin/max範囲内での80%・20%位置を計算
    - ミッドライン計算式：(Min+80%Range値 + Min+20%Range値) / 2
    - HyperER動的適応サポート
    - 従来ドンチャンチャネルよりも安定した動作
    """
    
    def __init__(
        self,
        period: int = 20,
        src_type: str = 'hlc3',
        # HyperER動的適応パラメータ
        enable_hyper_er_adaptation: bool = True,  # HyperER動的適応を有効にするか
        hyper_er_period: int = 14,                 # HyperER計算期間
        hyper_er_midline_period: int = 100,        # HyperERミッドライン期間
        period_min: float = 40.0,                  # 最小期間（ER高い時）
        period_max: float = 240.0                  # 最大期間（ER低い時）
    ):
        """
        コンストラクタ
        
        Args:
            period: Hyperドンチャン期間（デフォルト: 20）
            src_type: ソースタイプ（デフォルト: 'hlc3'）
            enable_hyper_er_adaptation: HyperER動的適応を有効にするか
            hyper_er_period: HyperER計算期間
            hyper_er_midline_period: HyperERミッドライン期間
            period_min: 最小期間（ER高い時）
            period_max: 最大期間（ER低い時）
        """
        # 動的適応文字列の作成
        adaptation_str = ""
        if enable_hyper_er_adaptation:
            adaptation_str = f"_hyper_er({hyper_er_period},{hyper_er_midline_period})"
        
        indicator_name = f"HyperDonchian({period}, src={src_type}{adaptation_str})"
        
        super().__init__(indicator_name)
        
        # パラメータ保存
        self.period = period
        self.src_type = src_type
        
        # HyperER動的適応パラメータ
        self.enable_hyper_er_adaptation = enable_hyper_er_adaptation
        self.hyper_er_period = hyper_er_period
        self.hyper_er_midline_period = hyper_er_midline_period
        self.period_min = period_min
        self.period_max = period_max
        
        # HyperERインジケーターの初期化（遅延インポート）
        self.hyper_er = None
        self._last_hyper_er_values = None
        self._hyper_er_initialized = False
        
        # パラメータ検証
        if self.period <= 0:
            raise ValueError("periodは0より大きい必要があります")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    first_val = float(data.iloc[0].get('close', data.iloc[0, -1]))
                    last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1]))
                else:
                    first_val = last_val = 0.0
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
            param_str = f"{self.period}_{self.src_type}"
            
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(param_str)}"
            
        except Exception:
            return f"{id(data)}_{self.period}"

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperDonchianResult:
        """
        Hyperドンチャンを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                必要なカラム: high, low, close, open
        
        Returns:
            HyperDonchianResult: Hyperドンチャンの計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # キャッシュヒット
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return HyperDonchianResult(
                    values=cached_result.values.copy(),
                    upper_band=cached_result.upper_band.copy(),
                    lower_band=cached_result.lower_band.copy(),
                    band_width=cached_result.band_width.copy()
                )
            
            # データの準備と検証
            if isinstance(data, pd.DataFrame):
                required_cols = ['high', 'low', 'close']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"必要なカラムが不足しています: {missing_cols}")
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
            
            # データ長の検証
            data_length = len(data)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < self.period:
                self.logger.warning(f"データ長（{data_length}）が必要な期間（{self.period}）より短いです")
            
            # 1. ソース価格データを取得
            source_prices = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換
            if not isinstance(source_prices, np.ndarray):
                source_prices = np.array(source_prices)
            if source_prices.dtype != np.float64:
                source_prices = source_prices.astype(np.float64)
            
            # 2. HyperER動的適応の計算（オプション）
            dynamic_periods = None
            if self.enable_hyper_er_adaptation:
                # 遅延インポートでHyperERを初期化
                if not self._hyper_er_initialized:
                    try:
                        from .hyper_er import HyperER
                        self.hyper_er = HyperER(
                            period=self.hyper_er_period,
                            midline_period=self.hyper_er_midline_period,
                            er_src_type=self.src_type
                        )
                        self._hyper_er_initialized = True
                    except Exception as e:
                        self.logger.warning(f"HyperERインジケーターの初期化に失敗しました: {e}")
                        self.enable_hyper_er_adaptation = False
                
                if self.hyper_er is not None:
                    try:
                        hyper_er_result = self.hyper_er.calculate(data)
                        if hyper_er_result is not None and hasattr(hyper_er_result, 'values'):
                            er_values = np.asarray(hyper_er_result.values, dtype=np.float64)
                            dynamic_periods = calculate_dynamic_period_vec(
                                er_values, self.period_min, self.period_max
                            )
                            self._last_hyper_er_values = er_values.copy()
                    except Exception as e:
                        self.logger.warning(f"HyperER動的適応計算に失敗しました: {e}")
                        # フォールバック: 前回の値を使用または固定値
                        if self._last_hyper_er_values is not None:
                            dynamic_periods = calculate_dynamic_period_vec(
                                self._last_hyper_er_values, self.period_min, self.period_max
                            )
            
            # 3. Hyperドンチャンを計算
            if self.enable_hyper_er_adaptation and dynamic_periods is not None:
                # HyperER動的適応版を使用
                midline, upper_band, lower_band, band_width = calculate_hyper_donchian_hyper_er(
                    source_prices, dynamic_periods
                )
            else:
                # 固定期間版を使用
                midline, upper_band, lower_band, band_width = calculate_hyper_donchian_numba(
                    source_prices, self.period
                )
            
            # 結果の作成
            result = HyperDonchianResult(
                values=midline.copy(),
                upper_band=upper_band.copy(),
                lower_band=lower_band.copy(),
                band_width=band_width.copy()
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            # 基底クラス用の値設定
            self._values = midline
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Hyperドンチャン計算中にエラー: {error_msg}\\n{stack_trace}")
            
            # エラー時は空の結果を返す
            empty_array = np.array([])
            return HyperDonchianResult(
                values=empty_array,
                upper_band=empty_array,
                lower_band=empty_array,
                band_width=empty_array
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """Hyperドンチャンミッドライン値を取得（後方互換性のため）"""
        if not self._result_cache:
            return None
        
        result = self._get_latest_result()
        return result.values.copy() if result else None
    
    def get_upper_band(self) -> Optional[np.ndarray]:
        """上部バンド値（80パーセンタイル）を取得"""
        result = self._get_latest_result()
        return result.upper_band.copy() if result else None
    
    def get_lower_band(self) -> Optional[np.ndarray]:
        """下部バンド値（20パーセンタイル）を取得"""
        result = self._get_latest_result()
        return result.lower_band.copy() if result else None
    
    def get_band_width(self) -> Optional[np.ndarray]:
        """バンド幅を取得"""
        result = self._get_latest_result()
        return result.band_width.copy() if result else None
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'period': self.period,
            'src_type': self.src_type,
            'description': 'Hyperドンチャンチャネル(Min/Max範囲内80-20%位置版)、安定したトレンドフィルター'
        }
    
    def _get_latest_result(self) -> Optional[HyperDonchianResult]:
        """最新の結果を取得"""
        if not self._result_cache:
            return None
        
        if self._cache_keys:
            return self._result_cache[self._cache_keys[-1]]
        else:
            return next(iter(self._result_cache.values()))
    
    def reset(self) -> None:
        """インディケーターの状態をリセット"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_hyper_donchian(
    data: Union[pd.DataFrame, np.ndarray],
    period: int = 20,
    src_type: str = 'hlc3',
    **kwargs
) -> np.ndarray:
    """
    Hyperドンチャンの計算（便利関数）
    
    Args:
        data: 価格データ
        period: Hyperドンチャン期間
        src_type: ソースタイプ
        **kwargs: その他のパラメータ
        
    Returns:
        Hyperドンチャンミッドライン値
    """
    indicator = HyperDonchian(
        period=period,
        src_type=src_type,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== Hyperドンチャン インジケーターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # トレンドとレンジが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 上昇トレンド
            change = 0.003 + np.random.normal(0, 0.008)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.012)
        elif i < 150:  # 強い上昇トレンド
            change = 0.005 + np.random.normal(0, 0.006)
        else:  # 下降トレンド
            change = -0.002 + np.random.normal(0, 0.010)
        
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
    
    # Hyperドンチャンを計算
    print("\\nHyperドンチャン（80-20パーセンタイル版）をテスト中...")
    hyper_donchian = HyperDonchian(period=20, src_type='hlc3')
    result = hyper_donchian.calculate(df)
    
    valid_count = np.sum(~np.isnan(result.values))
    mean_midline = np.nanmean(result.values)
    mean_band_width = np.nanmean(result.band_width)
    
    print(f"  有効値数: {valid_count}/{len(df)}")
    print(f"  平均ミッドライン: {mean_midline:.2f}")
    print(f"  平均バンド幅: {mean_band_width:.2f}")
    print(f"  Min+80%Range最高値: {np.nanmax(result.upper_band):.2f}")
    print(f"  Min+20%Range最低値: {np.nanmin(result.lower_band):.2f}")
    
    # 従来のドンチャンとの比較
    print("\\n従来のドンチャンミッドラインとの比較...")
    from .donchian_midline import DonchianMidline
    
    traditional_donchian = DonchianMidline(period=20, src_type='hlc3', enable_hyper_er_adaptation=False)
    traditional_result = traditional_donchian.calculate(df)
    
    trad_mean_midline = np.nanmean(traditional_result.values)
    trad_mean_band_width = np.nanmean(traditional_result.band_width)
    
    print(f"  従来版平均ミッドライン: {trad_mean_midline:.2f}")
    print(f"  従来版平均バンド幅: {trad_mean_band_width:.2f}")
    
    # バンド幅の安定性比較
    hyper_stability = np.nanstd(result.band_width) / np.nanmean(result.band_width) if np.nanmean(result.band_width) > 0 else float('inf')
    trad_stability = np.nanstd(traditional_result.band_width) / np.nanmean(traditional_result.band_width) if np.nanmean(traditional_result.band_width) > 0 else float('inf')
    
    print(f"\\n安定性比較（変動係数）:")
    print(f"  Hyperドンチャン: {hyper_stability:.4f}")
    print(f"  従来ドンチャン: {trad_stability:.4f}")
    print(f"  安定性改善: {((trad_stability - hyper_stability) / trad_stability * 100):.1f}%")
    
    print("\\n=== テスト完了 ===")