#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Hyper Adaptive Supertrend - 最強のスーパートレンドインジケーター** 🎯

既存のスーパートレンドを大幅に進化させた最強版：
- unified_smootherをミッドラインとして使用
- unscented_kalman_filterによるソース価格フィルタリング（オプション）
- x_atrを使用した高度なボラティリティ計算
- 後のロジックは既存のスーパートレンドと同じ

🌟 **主要改良点:**
1. **高度なミッドライン**: unified_smootherによる多様なスムージング手法
2. **カルマンフィルタリング**: ソース価格のノイズ除去（オプション）
3. **X_ATR統合**: 拡張ATRによる精密なボラティリティ測定
4. **適応的パラメータ**: 動的期間調整対応

📊 **処理フロー:**
1. ソース価格抽出 → カルマンフィルター（オプション） → 統合スムーサー（ミッドライン）
2. X_ATR計算（ボラティリティ）
3. ミッドライン ± (X_ATR × 乗数) でバンド計算
4. 既存のスーパートレンドロジック適用
"""

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, types
import traceback

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .smoother.unified_smoother import UnifiedSmoother
    from .kalman.unscented_kalman_filter import UnscentedKalmanFilter
    from .volatility.x_atr import XATR
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    from indicators.indicator import Indicator
    from indicators.price_source import PriceSource
    from indicators.smoother.unified_smoother import UnifiedSmoother
    from indicators.kalman.unscented_kalman_filter import UnscentedKalmanFilter
    from indicators.volatility.x_atr import XATR


@dataclass
class HyperAdaptiveSupertrendResult:
    """Hyperアダプティブスーパートレンドの計算結果"""
    values: np.ndarray           # スーパートレンドライン値
    upper_band: np.ndarray       # 上側のバンド価格
    lower_band: np.ndarray       # 下側のバンド価格
    trend: np.ndarray           # トレンド方向（1=上昇トレンド、-1=下降トレンド）
    midline: np.ndarray         # ミッドライン（統合スムーサー結果）
    atr_values: np.ndarray      # 使用されたX_ATR値
    raw_source: np.ndarray      # 元のソース価格
    filtered_source: Optional[np.ndarray]  # カルマンフィルター後のソース価格（使用時のみ）
    smoother_type: str          # 使用されたスムーサータイプ
    atr_method: str            # 使用されたATR計算方法
    kalman_enabled: bool       # カルマンフィルター使用フラグ
    parameters: Dict[str, any] # 使用されたパラメータ


@njit(fastmath=True, cache=True)
def calculate_hyper_supertrend_bands(
    midline: np.ndarray, 
    close: np.ndarray, 
    atr: np.ndarray, 
    multiplier: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hyperアダプティブスーパートレンドを計算する
    
    Args:
        midline: 統合スムーサーで計算されたミッドライン
        close: 終値の配列
        atr: X_ATRの配列
        multiplier: ATRの乗数
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 上側バンド、下側バンド、トレンド方向
    """
    length = len(close)
    
    # 基準となるバンドの計算（ミッドラインベース）
    final_upper_band = midline + multiplier * atr
    final_lower_band = midline - multiplier * atr
    
    # トレンド方向の配列を初期化
    trend = np.zeros(length, dtype=np.int8)
    upper_band = np.zeros(length, dtype=np.float64)
    lower_band = np.zeros(length, dtype=np.float64)
    
    # 最初の有効な値を見つける
    first_valid_idx = -1
    for i in range(length):
        if (not np.isnan(final_upper_band[i]) and not np.isnan(final_lower_band[i]) 
            and not np.isnan(close[i]) and not np.isnan(midline[i])):
            first_valid_idx = i
            break
    
    # 有効な値が見つからない場合は全てNaN/0を返す
    if first_valid_idx < 0:
        upper_band[:] = np.nan
        lower_band[:] = np.nan
        return upper_band, lower_band, trend
    
    # 最初の値を設定（終値とミッドラインの比較で判定）
    trend[first_valid_idx] = 1 if close[first_valid_idx] > midline[first_valid_idx] else -1
    
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
        if (np.isnan(close[i]) or np.isnan(final_upper_band[i]) 
            or np.isnan(final_lower_band[i]) or np.isnan(midline[i])):
            trend[i] = trend[i-1]
            upper_band[i] = upper_band[i-1]
            lower_band[i] = lower_band[i-1]
            continue
            
        # トレンド判定（前のバンド値との比較）
        if close[i] > final_upper_band[i-1]:
            trend[i] = 1
        elif close[i] < final_lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
            # バンドの調整（既存のスーパートレンドと同じロジック）
            if trend[i] == 1 and final_lower_band[i] < final_lower_band[i-1]:
                final_lower_band[i] = final_lower_band[i-1]
            elif trend[i] == -1 and final_upper_band[i] > final_upper_band[i-1]:
                final_upper_band[i] = final_upper_band[i-1]
        
        # トレンドに基づいてバンドを設定
        if trend[i] == 1:
            upper_band[i] = np.nan
            lower_band[i] = final_lower_band[i]
        else:
            upper_band[i] = final_upper_band[i]
            lower_band[i] = np.nan
    
    return upper_band, lower_band, trend


@njit(fastmath=True, cache=True)
def calculate_hyper_supertrend_line(
    upper_band: np.ndarray, 
    lower_band: np.ndarray, 
    trend: np.ndarray
) -> np.ndarray:
    """
    Hyperアダプティブスーパートレンドラインを計算する
    
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


class HyperAdaptiveSupertrend(Indicator):
    """
    Hyperアダプティブスーパートレンドインジケーター（最強版）
    
    unified_smootherをミッドライン、x_atrをボラティリティ計算、
    unscented_kalman_filterをソース価格フィルタリングに使用した
    最強のスーパートレンドインジケーター：
    
    - 高度なミッドライン計算（統合スムーサー）
    - 精密なボラティリティ測定（X_ATR）
    - カルマンフィルターによるノイズ除去（オプション）
    - 既存のスーパートレンドロジック
    
    特徴:
    - 多様なスムージング手法による適応的ミッドライン
    - 拡張ATRによる高精度ボラティリティ測定
    - カルマンフィルターによるノイズ除去
    - 動的期間調整対応
    """
    
    def __init__(
        self,
        # ATR/ボラティリティパラメータ
        atr_period: float = 14.0,                      # X_ATR期間
        multiplier: float = 2.0,                       # ATR乗数
        atr_method: str = 'atr',                       # X_ATRの計算方法（'atr' or 'str'）
        atr_smoother_type: str = 'ultimate_smoother',  # X_ATRのスムーサー
        
        # ミッドライン（統合スムーサー）パラメータ
        midline_smoother_type: str = 'frama',          # ミッドラインスムーサータイプ
        midline_period: float = 21.0,                  # ミッドライン期間
        
        # ソース価格関連パラメータ
        src_type: str = 'hlc3',                        # ソースタイプ
        enable_kalman: bool = False,                   # カルマンフィルター使用フラグ
        kalman_alpha: float = 0.1,                     # UKFアルファパラメータ
        kalman_beta: float = 2.0,                      # UKFベータパラメータ
        kalman_kappa: float = 0.0,                     # UKFカッパパラメータ
        kalman_process_noise: float = 0.01,            # UKFプロセスノイズ
        
        # 動的期間調整パラメータ
        use_dynamic_period: bool = False,              # 動的期間を使用するか
        cycle_part: float = 1.0,                      # サイクル部分の倍率
        detector_type: str = 'absolute_ultimate',      # 検出器タイプ
        max_cycle: int = 233,                         # 最大サイクル期間
        min_cycle: int = 13,                          # 最小サイクル期間
        max_output: int = 144,                        # 最大出力値
        min_output: int = 13,                         # 最小出力値
        lp_period: int = 10,                          # ローパスフィルター期間
        hp_period: int = 48,                          # ハイパスフィルター期間
        
        # 追加パラメータ
        midline_smoother_params: Optional[Dict] = None,  # ミッドラインスムーサー固有パラメータ
        atr_smoother_params: Optional[Dict] = None,      # ATRスムーサー固有パラメータ
        atr_kalman_params: Optional[Dict] = None         # ATR用カルマンパラメータ
    ):
        """
        コンストラクタ
        
        Args:
            atr_period: X_ATR期間
            multiplier: ATR乗数
            atr_method: X_ATRの計算方法（'atr' または 'str'）
            atr_smoother_type: X_ATRのスムーサータイプ
            
            midline_smoother_type: ミッドラインスムーサータイプ
            midline_period: ミッドライン期間
            
            src_type: ソースタイプ
            enable_kalman: カルマンフィルター使用フラグ
            kalman_alpha: UKFアルファパラメータ
            kalman_beta: UKFベータパラメータ
            kalman_kappa: UKFカッパパラメータ
            kalman_process_noise: UKFプロセスノイズ
            
            use_dynamic_period: 動的期間を使用するか
            cycle_part: サイクル部分の倍率
            detector_type: 検出器タイプ
            max_cycle: 最大サイクル期間
            min_cycle: 最小サイクル期間
            max_output: 最大出力値
            min_output: 最小出力値
            lp_period: ローパスフィルター期間
            hp_period: ハイパスフィルター期間
            
            midline_smoother_params: ミッドラインスムーサー固有パラメータ
            atr_smoother_params: ATRスムーサー固有パラメータ
            atr_kalman_params: ATR用カルマンパラメータ
        """
        # 指標名の作成
        kalman_str = "_K" if enable_kalman else ""
        dynamic_str = f"_dynamic({detector_type})" if use_dynamic_period else ""
        indicator_name = (f"HyperAdaptiveSupertrend("
                         f"atr={atr_period}×{multiplier}_{atr_method}_{atr_smoother_type}, "
                         f"mid={midline_period}_{midline_smoother_type}, "
                         f"{src_type}{kalman_str}{dynamic_str})")
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.atr_method = atr_method.lower()
        self.atr_smoother_type = atr_smoother_type.lower()
        self.midline_smoother_type = midline_smoother_type.lower()
        self.midline_period = midline_period
        self.src_type = src_type.lower()
        self.enable_kalman = enable_kalman
        self.kalman_alpha = kalman_alpha
        self.kalman_beta = kalman_beta
        self.kalman_kappa = kalman_kappa
        self.kalman_process_noise = kalman_process_noise
        self.use_dynamic_period = use_dynamic_period
        self.cycle_part = cycle_part
        self.detector_type = detector_type
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.lp_period = lp_period
        self.hp_period = hp_period
        
        # パラメータ辞書の初期化
        self.midline_smoother_params = midline_smoother_params or {}
        self.atr_smoother_params = atr_smoother_params or {}
        self.atr_kalman_params = atr_kalman_params or {}
        
        # ソースタイプの検証
        if self.src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # ATR計算方法の検証
        if self.atr_method not in ['atr', 'str']:
            raise ValueError(f"無効なATR計算方法です: {atr_method}。'atr' または 'str' を指定してください。")
        
        # カルマンフィルターの初期化（必要時のみ）
        self._kalman_filter = None
        if self.enable_kalman:
            try:
                self._kalman_filter = UnscentedKalmanFilter(
                    src_type=self.src_type,
                    alpha=self.kalman_alpha,
                    beta=self.kalman_beta,
                    kappa=self.kalman_kappa,
                    process_noise_scale=self.kalman_process_noise
                )
                self.logger.info(f"カルマンフィルター初期化完了: UKF(α={self.kalman_alpha})")
            except Exception as e:
                self.logger.error(f"カルマンフィルターの初期化に失敗: {e}")
                self.enable_kalman = False
                self.logger.warning("カルマンフィルターを無効化しました")
        
        # ミッドライン用統合スムーサーの初期化
        midline_config = {
            **self.midline_smoother_params
        }
        
        # スムーサータイプに応じてパラメータ名を調整
        if self.midline_smoother_type in ['frama']:
            # FRAMAはperiodではなく異なるパラメータ名を使用
            # 期間は偶数である必要があるため調整
            period_val = int(self.midline_period)
            if period_val % 2 == 1:
                period_val += 1  # 奇数の場合は+1して偶数にする
            midline_config['period'] = period_val
        else:
            midline_config['period'] = self.midline_period
        
        # 動的期間パラメータの追加
        if self.use_dynamic_period:
            midline_config.update({
                'cycle_detector_type': self.detector_type,
                'cycle_part': self.cycle_part,
                'max_cycle': self.max_cycle,
                'min_cycle': self.min_cycle,
                'max_output': self.max_output,
                'min_output': self.min_output,
                'lp_period': self.lp_period,
                'hp_period': self.hp_period
            })
        
        self._midline_smoother = UnifiedSmoother(
            smoother_type=self.midline_smoother_type,
            src_type=self.src_type,
            period_mode='dynamic' if self.use_dynamic_period else 'fixed',
            **midline_config
        )
        
        # X_ATRインジケーターの初期化
        atr_config = {
            'period': self.atr_period,
            'tr_method': self.atr_method,
            'smoother_type': self.atr_smoother_type,
            'src_type': 'close',  # X_ATRは常にcloseベース
            'enable_kalman': False,  # ATR計算では独立したカルマンフィルターを使用
            'period_mode': 'dynamic' if self.use_dynamic_period else 'fixed',
            **self.atr_smoother_params
        }
        
        # 動的期間パラメータの追加
        if self.use_dynamic_period:
            atr_config.update({
                'cycle_detector_type': self.detector_type,
                'cycle_detector_cycle_part': self.cycle_part,
                'cycle_detector_max_cycle': self.max_cycle,
                'cycle_detector_min_cycle': self.min_cycle,
                'cycle_period_multiplier': 1.0,
                'cycle_detector_period_range': (self.min_output, self.max_output)
            })
        
        # カルマンパラメータの追加
        if self.atr_kalman_params:
            for key, value in self.atr_kalman_params.items():
                atr_config[f'kalman_{key}'] = value
        
        self._x_atr = XATR(**atr_config)
        
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
            params_sig = (f"{self.atr_period}_{self.multiplier}_{self.atr_method}_"
                         f"{self.atr_smoother_type}_{self.midline_smoother_type}_"
                         f"{self.midline_period}_{self.src_type}_{self.enable_kalman}_"
                         f"{self.use_dynamic_period}_{self.detector_type}")
            
            # ハッシュ計算
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.atr_period}_{self.multiplier}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> HyperAdaptiveSupertrendResult:
        """
        Hyperアダプティブスーパートレンドを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、OHLC + 選択したソースタイプに必要なカラムが必要
        
        Returns:
            HyperAdaptiveSupertrendResult: Hyperアダプティブスーパートレンドの値とトレンド情報を含む結果
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
                return HyperAdaptiveSupertrendResult(
                    values=cached_result.values.copy(),
                    upper_band=cached_result.upper_band.copy(),
                    lower_band=cached_result.lower_band.copy(),
                    trend=cached_result.trend.copy(),
                    midline=cached_result.midline.copy(),
                    atr_values=cached_result.atr_values.copy(),
                    raw_source=cached_result.raw_source.copy(),
                    filtered_source=cached_result.filtered_source.copy() if cached_result.filtered_source is not None else None,
                    smoother_type=cached_result.smoother_type,
                    atr_method=cached_result.atr_method,
                    kalman_enabled=cached_result.kalman_enabled,
                    parameters=cached_result.parameters.copy()
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
                
                close = data['close'].to_numpy()
            else:
                # NumPy配列の場合
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                close = data[:, 3]  # close
            
            # データ長の検証
            data_length = len(close)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            # ステップ1: ソース価格の抽出
            raw_source = PriceSource.calculate_source(data, self.src_type)
            
            # ステップ2: カルマンフィルター適用（オプション）
            filtered_source = None
            processed_data = data
            
            if self.enable_kalman and self._kalman_filter is not None:
                try:
                    kalman_result = self._kalman_filter.calculate(data)
                    filtered_source = kalman_result.filtered_values
                    
                    # フィルター済み価格で新しいデータを構築
                    if isinstance(data, pd.DataFrame):
                        processed_data = data.copy()
                        processed_data[self.src_type] = filtered_source
                        
                        # 論理的整合性を保持（close価格をベースに他のOHLVも比例調整）
                        if self.src_type == 'close':
                            for i in range(len(processed_data)):
                                if (not np.isnan(filtered_source[i]) and 
                                    not np.isnan(data.iloc[i]['close']) and 
                                    data.iloc[i]['close'] != 0):
                                    ratio = filtered_source[i] / data.iloc[i]['close']
                                    for col in ['open', 'high', 'low']:
                                        if col in processed_data.columns:
                                            processed_data.iloc[i, processed_data.columns.get_loc(col)] *= ratio
                    else:
                        # NumPy配列の場合
                        processed_data = data.copy()
                        if self.src_type == 'close':
                            processed_data[:, 3] = filtered_source  # close列
                    
                    self.logger.debug("カルマンフィルター適用完了")
                except Exception as e:
                    self.logger.error(f"カルマンフィルター適用中にエラー: {e}")
                    filtered_source = None
                    processed_data = data
            
            # ステップ3: ミッドライン計算（統合スムーサー）
            midline_result = self._midline_smoother.calculate(processed_data)
            midline = midline_result.values
            
            # ステップ4: X_ATR計算
            atr_result = self._x_atr.calculate(data)  # 元のデータを使用
            atr_values = atr_result.values
            
            # ステップ5: スーパートレンドバンド計算
            upper_band, lower_band, trend_direction = calculate_hyper_supertrend_bands(
                midline, close, atr_values, self.multiplier
            )
            
            # ステップ6: スーパートレンドライン計算
            supertrend_line = calculate_hyper_supertrend_line(upper_band, lower_band, trend_direction)
            
            # 結果の保存
            result = HyperAdaptiveSupertrendResult(
                values=supertrend_line.copy(),
                upper_band=upper_band.copy(),
                lower_band=lower_band.copy(),
                trend=trend_direction.copy(),
                midline=midline.copy(),
                atr_values=atr_values.copy(),
                raw_source=raw_source.copy(),
                filtered_source=filtered_source.copy() if filtered_source is not None else None,
                smoother_type=self.midline_smoother_type,
                atr_method=self.atr_method,
                kalman_enabled=self.enable_kalman,
                parameters={
                    'atr_period': self.atr_period,
                    'multiplier': self.multiplier,
                    'atr_method': self.atr_method,
                    'atr_smoother_type': self.atr_smoother_type,
                    'midline_smoother_type': self.midline_smoother_type,
                    'midline_period': self.midline_period,
                    'src_type': self.src_type,
                    'enable_kalman': self.enable_kalman,
                    'kalman_alpha': self.kalman_alpha,
                    'kalman_beta': self.kalman_beta,
                    'kalman_kappa': self.kalman_kappa,
                    'kalman_process_noise': self.kalman_process_noise,
                    'use_dynamic_period': self.use_dynamic_period,
                    'cycle_part': self.cycle_part,
                    'detector_type': self.detector_type,
                    'max_cycle': self.max_cycle,
                    'min_cycle': self.min_cycle,
                    'max_output': self.max_output,
                    'min_output': self.min_output,
                    'lp_period': self.lp_period,
                    'hp_period': self.hp_period,
                    'midline_smoother_params': self.midline_smoother_params.copy(),
                    'atr_smoother_params': self.atr_smoother_params.copy(),
                    'atr_kalman_params': self.atr_kalman_params.copy()
                }
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = supertrend_line  # 基底クラスの要件を満たすため
            
            self.logger.debug(f"Hyperアダプティブスーパートレンド計算完了 - "
                            f"ミッドライン: {self.midline_smoother_type}, "
                            f"ATR: {self.atr_method}_{self.atr_smoother_type}, "
                            f"カルマン: {self.enable_kalman}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"Hyperアダプティブスーパートレンド計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = HyperAdaptiveSupertrendResult(
                values=np.array([]),
                upper_band=np.array([]),
                lower_band=np.array([]),
                trend=np.array([], dtype=np.int8),
                midline=np.array([]),
                atr_values=np.array([]),
                raw_source=np.array([]),
                filtered_source=None,
                smoother_type=self.midline_smoother_type,
                atr_method=self.atr_method,
                kalman_enabled=self.enable_kalman,
                parameters={}
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
            result = next(iter(self._result_cache.values()))
            
        return result.values.copy()
    
    def get_supertrend_direction(self) -> Optional[np.ndarray]:
        """
        スーパートレンドの基本方向を取得する
        
        Returns:
            np.ndarray: スーパートレンドの基本方向（1=上昇、-1=下降）
        """
        result = self._get_latest_result()
        return result.trend.copy() if result else None
    
    def get_upper_band(self) -> Optional[np.ndarray]:
        """上側バンドを取得する"""
        result = self._get_latest_result()
        return result.upper_band.copy() if result else None
    
    def get_lower_band(self) -> Optional[np.ndarray]:
        """下側バンドを取得する"""
        result = self._get_latest_result()
        return result.lower_band.copy() if result else None
    
    def get_midline(self) -> Optional[np.ndarray]:
        """ミッドライン（統合スムーサー結果）を取得する"""
        result = self._get_latest_result()
        return result.midline.copy() if result else None
    
    def get_atr_values(self) -> Optional[np.ndarray]:
        """使用されたX_ATR値を取得する"""
        result = self._get_latest_result()
        return result.atr_values.copy() if result else None
    
    def get_raw_source(self) -> Optional[np.ndarray]:
        """元のソース価格を取得する"""
        result = self._get_latest_result()
        return result.raw_source.copy() if result else None
    
    def get_filtered_source(self) -> Optional[np.ndarray]:
        """カルマンフィルター後のソース価格を取得する（使用時のみ）"""
        result = self._get_latest_result()
        if result and result.filtered_source is not None:
            return result.filtered_source.copy()
        return None
    
    def get_dynamic_periods(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        動的期間の値を取得する（動的期間モードのみ）
        
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: 
            (ミッドライン動的期間, ATR動的期間)
        """
        if not self.use_dynamic_period:
            return None, None
        
        # ミッドラインスムーサーから動的期間を取得
        midline_periods = None
        if hasattr(self._midline_smoother, 'get_dynamic_periods'):
            try:
                midline_periods = self._midline_smoother.get_dynamic_periods()
            except:
                pass
        
        # X_ATRから動的期間を取得
        atr_periods = None
        if hasattr(self._x_atr, 'get_dynamic_periods'):
            try:
                atr_periods = self._x_atr.get_dynamic_periods()
            except:
                pass
        
        return midline_periods, atr_periods
    
    def get_metadata(self) -> Dict:
        """
        インジケーターのメタデータを取得する
        
        Returns:
            Dict: メタデータ情報
        """
        result = self._get_latest_result()
        
        metadata = {
            'indicator_type': 'Hyper Adaptive Supertrend',
            'version': '1.0.0',
            'components': {
                'midline_smoother': self.midline_smoother_type,
                'atr_calculator': f'X_ATR({self.atr_method}_{self.atr_smoother_type})',
                'kalman_filter': 'UKF' if self.enable_kalman else None
            },
            'parameters': {
                'atr_period': self.atr_period,
                'multiplier': self.multiplier,
                'midline_period': self.midline_period,
                'src_type': self.src_type,
                'use_dynamic_period': self.use_dynamic_period
            },
            'features': {
                'kalman_filtering': self.enable_kalman,
                'dynamic_periods': self.use_dynamic_period,
                'adaptive_midline': True,
                'enhanced_atr': True
            }
        }
        
        if result:
            metadata['data_info'] = {
                'data_points': len(result.values),
                'valid_values': np.sum(~np.isnan(result.values)),
                'trend_distribution': {
                    'uptrend': np.sum(result.trend == 1),
                    'downtrend': np.sum(result.trend == -1),
                    'undefined': np.sum(result.trend == 0)
                }
            }
        
        return metadata
    
    def _get_latest_result(self) -> Optional[HyperAdaptiveSupertrendResult]:
        """最新の結果を取得"""
        if not self._result_cache:
            return None
            
        if self._cache_keys:
            return self._result_cache[self._cache_keys[-1]]
        else:
            return next(iter(self._result_cache.values()))
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        
        # サブコンポーネントのリセット
        if hasattr(self._midline_smoother, 'reset'):
            self._midline_smoother.reset()
        
        if hasattr(self._x_atr, 'reset'):
            self._x_atr.reset()
        
        if self._kalman_filter and hasattr(self._kalman_filter, 'reset'):
            self._kalman_filter.reset()


# 便利関数
def calculate_hyper_adaptive_supertrend(
    data: Union[pd.DataFrame, np.ndarray],
    atr_period: float = 14.0,
    multiplier: float = 2.0,
    atr_method: str = 'atr',
    atr_smoother_type: str = 'ultimate_smoother',
    midline_smoother_type: str = 'frama',
    midline_period: float = 21.0,
    src_type: str = 'hlc3',
    enable_kalman: bool = False,
    use_dynamic_period: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Hyperアダプティブスーパートレンドの計算（便利関数）
    
    Args:
        data: 価格データ
        atr_period: X_ATR期間
        multiplier: ATR乗数
        atr_method: X_ATRの計算方法
        atr_smoother_type: X_ATRのスムーサータイプ
        midline_smoother_type: ミッドラインスムーサータイプ
        midline_period: ミッドライン期間
        src_type: ソースタイプ
        enable_kalman: カルマンフィルター使用フラグ
        use_dynamic_period: 動的期間を使用するか
        **kwargs: 追加パラメータ
        
    Returns:
        スーパートレンドライン値の配列
    """
    indicator = HyperAdaptiveSupertrend(
        atr_period=atr_period,
        multiplier=multiplier,
        atr_method=atr_method,
        atr_smoother_type=atr_smoother_type,
        midline_smoother_type=midline_smoother_type,
        midline_period=midline_period,
        src_type=src_type,
        enable_kalman=enable_kalman,
        use_dynamic_period=use_dynamic_period,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    print("=== Hyperアダプティブスーパートレンドのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    trend = 0.001
    volatility = 0.02
    
    prices = [base_price]
    for i in range(1, length):
        change = trend + np.random.normal(0, volatility)
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
    
    print(f"テストデータ: {len(df)}ポイント")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 各設定でのテスト
    test_configs = [
        {
            'name': 'ベーシック（FRAMA + ATR）',
            'midline_smoother_type': 'frama',
            'atr_method': 'atr',
            'enable_kalman': False,
            'use_dynamic_period': False
        },
        {
            'name': 'アドバンス（Ultimate Smoother + STR）',
            'midline_smoother_type': 'ultimate_smoother',
            'atr_method': 'str',
            'enable_kalman': False,
            'use_dynamic_period': False
        },
        {
            'name': 'カルマンフィルター付き',
            'midline_smoother_type': 'frama',
            'atr_method': 'atr',
            'enable_kalman': True,
            'use_dynamic_period': False
        },
        {
            'name': 'フル機能（動的期間 + カルマン）',
            'midline_smoother_type': 'ultimate_smoother',
            'atr_method': 'str',
            'enable_kalman': True,
            'use_dynamic_period': True
        }
    ]
    
    for config in test_configs:
        try:
            name = config.pop('name')
            print(f"\n{name}をテスト中...")
            
            indicator = HyperAdaptiveSupertrend(**config)
            result = indicator.calculate(df)
            
            mean_value = np.nanmean(result.values)
            valid_count = np.sum(~np.isnan(result.values))
            uptrend_count = np.sum(result.trend == 1)
            downtrend_count = np.sum(result.trend == -1)
            
            print(f"  平均スーパートレンド値: {mean_value:.4f}")
            print(f"  有効値数: {valid_count}/{len(df)}")
            print(f"  トレンド分布: 上昇={uptrend_count}, 下降={downtrend_count}")
            print(f"  ミッドライン平均: {np.nanmean(result.midline):.4f}")
            print(f"  ATR平均: {np.nanmean(result.atr_values):.4f}")
            
            if result.filtered_source is not None:
                print(f"  カルマンフィルター効果: 元={np.nanmean(result.raw_source):.4f}, "
                      f"フィルター後={np.nanmean(result.filtered_source):.4f}")
            
        except Exception as e:
            print(f"  エラー: {e}")
    
    print("\n=== テスト完了 ===")