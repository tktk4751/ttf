#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, types
import traceback

from .indicator import Indicator
from .price_source import PriceSource
from .atr import ATR


@dataclass
class SupertrendResult:
    """スーパートレンドの計算結果"""
    values: np.ndarray           # スーパートレンドライン値
    upper_band: np.ndarray       # 上側のバンド価格
    lower_band: np.ndarray       # 下側のバンド価格
    trend: np.ndarray           # トレンド方向（1=上昇トレンド、-1=下降トレンド）
    atr_values: np.ndarray      # 使用されたATR値


@njit(fastmath=True, cache=True)
def calculate_supertrend_bands(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray, multiplier: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    スーパートレンドを計算する（参考コードベース）
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
        atr: ATRの配列
        multiplier: ATRの乗数
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 上側バンド、下側バンド、トレンド方向
    """
    length = len(close)
    
    # 基準となるバンドの計算
    hl_avg = (high + low) / 2.0
    final_upper_band = hl_avg + multiplier * atr
    final_lower_band = hl_avg - multiplier * atr
    
    # トレンド方向の配列を初期化
    trend = np.zeros(length, dtype=np.int8)
    upper_band = np.zeros(length, dtype=np.float64)
    lower_band = np.zeros(length, dtype=np.float64)
    
    # 最初の有効な値を見つける
    first_valid_idx = -1
    for i in range(length):
        if not np.isnan(final_upper_band[i]) and not np.isnan(final_lower_band[i]) and not np.isnan(close[i]):
            first_valid_idx = i
            break
    
    # 有効な値が見つからない場合は全てNaN/0を返す
    if first_valid_idx < 0:
        upper_band[:] = np.nan
        lower_band[:] = np.nan
        return upper_band, lower_band, trend
    
    # 最初の値を設定
    trend[first_valid_idx] = 1 if close[first_valid_idx] > final_upper_band[first_valid_idx] else -1
    
    # 最初の有効インデックスまでは無効値
    for i in range(first_valid_idx):
        upper_band[i] = np.nan
        lower_band[i] = np.nan
        trend[i] = 0
    
    # 最初の有効値のバンド設定
    if trend[first_valid_idx] == 1:
        upper_band[first_valid_idx] = np.nan
        lower_band[first_valid_idx] = final_lower_band[first_valid_idx]
    else:
        upper_band[first_valid_idx] = final_upper_band[first_valid_idx]
        lower_band[first_valid_idx] = np.nan
    
    # バンドとトレンドの計算
    for i in range(first_valid_idx + 1, length):
        # データが無効な場合は前の値を維持
        if np.isnan(close[i]) or np.isnan(final_upper_band[i]) or np.isnan(final_lower_band[i]):
            trend[i] = trend[i-1]
            upper_band[i] = upper_band[i-1]
            lower_band[i] = lower_band[i-1]
            continue
            
        # トレンド判定（参考コードと同じロジック）
        if close[i] > final_upper_band[i-1]:
            trend[i] = 1
        elif close[i] < final_lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
            # バンドの調整（参考コードと同じロジック）
            if trend[i] == 1 and final_lower_band[i] < final_lower_band[i-1]:
                final_lower_band[i] = final_lower_band[i-1]
            elif trend[i] == -1 and final_upper_band[i] > final_upper_band[i-1]:
                final_upper_band[i] = final_upper_band[i-1]
        
        # トレンドに基づいてバンドを設定（参考コードと同じロジック）
        if trend[i] == 1:
            upper_band[i] = np.nan
            lower_band[i] = final_lower_band[i]
        else:
            upper_band[i] = final_upper_band[i]
            lower_band[i] = np.nan
    
    return upper_band, lower_band, trend



@njit(fastmath=True, cache=True)
def calculate_supertrend_line(upper_band: np.ndarray, lower_band: np.ndarray, trend: np.ndarray) -> np.ndarray:
    """
    スーパートレンドラインを計算する（参考コードベース）
    
    Args:
        upper_band: 上側バンド（表示用）
        lower_band: 下側バンド（表示用）
        trend: トレンド方向
    
    Returns:
        スーパートレンドラインの配列
    """
    length = len(trend)
    supertrend = np.zeros(length, dtype=np.float64)
    
    for i in range(length):
        if trend[i] == 1:
            # 上昇トレンド時は下側バンドを使用
            supertrend[i] = lower_band[i] if not np.isnan(lower_band[i]) else np.nan
        elif trend[i] == -1:
            # 下降トレンド時は上側バンドを使用
            supertrend[i] = upper_band[i] if not np.isnan(upper_band[i]) else np.nan
        else:
            # トレンドが0の場合はNaN
            supertrend[i] = np.nan
    
    return supertrend


class Supertrend(Indicator):
    """
    スーパートレンドインジケーター（最新仕様版）
    
    ATRを使用して、トレンドの方向と強さを判断する適応型トレンドフォローイング指標：
    - ATRの複数のスムージング方法に対応（Wilder's, HMA, ALMA, ZLEMA）
    - 固定期間または動的期間（ドミナントサイクル）での計算に対応
    - 複数のプライスソースに対応
    
    特徴:
    - トレンドが強い時：明確なトレンドライン表示
    - 動的期間使用時：市場サイクルに応じて期間を自動調整
    """
    
    def __init__(
        self,
        period: int = 10,                      # ATR期間
        multiplier: float = 3.0,               # ATR乗数
        src_type: str = 'hlc3',                # ソースタイプ
        atr_smoothing: str = 'wilder',         # ATRスムージング方法
        use_dynamic_period: bool = False,      # 動的期間を使用するか
        cycle_part: float = 1.0,              # サイクル部分の倍率（動的期間用）
        detector_type: str = 'cycle_period2',  # 検出器タイプ（動的期間用）
        max_cycle: int = 233,                  # 最大サイクル期間（動的期間用）
        min_cycle: int = 13,                   # 最小サイクル期間（動的期間用）
        max_output: int = 144,                 # 最大出力値（動的期間用）
        min_output: int = 13,                  # 最小出力値（動的期間用）
        lp_period: int = 10,                   # ローパスフィルター期間（動的期間用）
        hp_period: int = 48                    # ハイパスフィルター期間（動的期間用）
    ):
        """
        コンストラクタ
        
        Args:
            period: ATR期間（デフォルト: 10）
            multiplier: ATR乗数（デフォルト: 3.0）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
                - 'close': 終値
                - 'hlc3': (高値 + 安値 + 終値) / 3（デフォルト）
                - 'hl2': (高値 + 安値) / 2
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
            atr_smoothing: ATRスムージング方法 ('wilder', 'hma', 'alma', 'zlema')
            use_dynamic_period: 動的期間を使用するかどうか
            cycle_part: サイクル部分の倍率（動的期間モード用）
            detector_type: 検出器タイプ（動的期間モード用）
            max_cycle: 最大サイクル期間（動的期間モード用）
            min_cycle: 最小サイクル期間（動的期間モード用）
            max_output: 最大出力値（動的期間モード用）
            min_output: 最小出力値（動的期間モード用）
            lp_period: ローパスフィルター期間（動的期間モード用、デフォルト: 10）
            hp_period: ハイパスフィルター期間（動的期間モード用、デフォルト: 48）
        """
        # 指標名の作成
        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_period else ""
        indicator_name = f"Supertrend(p={period}, mult={multiplier}, {src_type}, atr={atr_smoothing}{dynamic_str})"
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.period = period
        self.multiplier = multiplier
        self.src_type = src_type.lower()
        self.atr_smoothing = atr_smoothing.lower()
        self.use_dynamic_period = use_dynamic_period
        self.cycle_part = cycle_part
        self.detector_type = detector_type
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.lp_period = lp_period
        self.hp_period = hp_period
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # ATRインジケーターを初期化
        self._atr = ATR(
            period=self.period,
            smoothing_method=self.atr_smoothing,
            use_dynamic_period=self.use_dynamic_period,
            cycle_part=self.cycle_part,
            detector_type=self.detector_type,
            max_cycle=self.max_cycle,
            min_cycle=self.min_cycle,
            max_output=self.max_output,
            min_output=self.min_output,
            lp_period=self.lp_period,
            hp_period=self.hp_period
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
            params_sig = f"{self.period}_{self.multiplier}_{self.src_type}_{self.atr_smoothing}_{self.use_dynamic_period}"
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.period}_{self.multiplier}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> SupertrendResult:
        """
        スーパートレンドを計算する（最新仕様版）
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、OHLC + 選択したソースタイプに必要なカラムが必要
        
        Returns:
            SupertrendResult: スーパートレンドの値とトレンド情報を含む結果
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
                return SupertrendResult(
                    values=cached_result.values.copy(),
                    upper_band=cached_result.upper_band.copy(),
                    lower_band=cached_result.lower_band.copy(),
                    trend=cached_result.trend.copy(),
                    atr_values=cached_result.atr_values.copy()
                )
            
            # データの準備
            if isinstance(data, pd.DataFrame):
                # 必要なカラムの検証
                required_cols = ['high', 'low', 'close']
                if self.src_type == 'ohlc4':
                    required_cols.append('open')
                
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"必要なカラムが不足しています: {missing_cols}")
                
                high = data['high'].to_numpy()
                low = data['low'].to_numpy()
                close = data['close'].to_numpy()
            else:
                # NumPy配列の場合
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                high = data[:, 1]  # high
                low = data[:, 2]   # low  
                close = data[:, 3] # close
            
            # NumPy配列に変換（float64型で統一）
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            close = np.asarray(close, dtype=np.float64)
            
            # データ長の検証
            data_length = len(close)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            # ATRの計算（最新のATRクラスを使用）
            atr_result = self._atr.calculate(data)
            atr_values = atr_result.values
            
            # スーパートレンドの計算（バンド + トレンド一括計算）
            upper_band, lower_band, trend_direction = calculate_supertrend_bands(high, low, close, atr_values, self.multiplier)
            
            # スーパートレンドラインの計算
            supertrend_line = calculate_supertrend_line(upper_band, lower_band, trend_direction)
            
            # 結果の保存（参照問題を避けるため必要な部分だけコピー）
            result = SupertrendResult(
                values=supertrend_line.copy(),
                upper_band=upper_band.copy(), 
                lower_band=lower_band.copy(),
                trend=trend_direction.copy(),
                atr_values=atr_values.copy()
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
            
            self._values = supertrend_line  # 基底クラスの要件を満たすため
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Supertrend計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = SupertrendResult(
                values=np.array([]),
                upper_band=np.array([]),
                lower_band=np.array([]),
                trend=np.array([], dtype=np.int8),
                atr_values=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """スーパートレンドライン値のみを取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.values.copy()
    
    def get_supertrend_direction(self) -> Optional[np.ndarray]:
        """
        スーパートレンドの基本方向を取得する
        
        Returns:
            np.ndarray: スーパートレンドの基本方向（1=上昇、-1=下降）
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.trend.copy()
    
    def get_upper_band(self) -> Optional[np.ndarray]:
        """
        上側バンドを取得する
        
        Returns:
            np.ndarray: 上側バンドの値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.upper_band.copy()
    
    def get_lower_band(self) -> Optional[np.ndarray]:
        """
        下側バンドを取得する
        
        Returns:
            np.ndarray: 下側バンドの値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.lower_band.copy()
    
    def get_atr_values(self) -> Optional[np.ndarray]:
        """
        使用されたATR値を取得する
        
        Returns:
            np.ndarray: ATR値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.atr_values.copy()
    
    def get_dynamic_periods(self) -> np.ndarray:
        """
        動的期間の値を取得する（動的期間モードのみ）
        
        Returns:
            動的期間の配列
        """
        if not self.use_dynamic_period:
            return np.array([])
        
        # ATRインジケーターから動的期間を取得
        return self._atr.get_dynamic_periods()
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        self._atr.reset() 