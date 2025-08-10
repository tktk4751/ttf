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

# 条件付きインポート（オプション機能）
try:
    from .kalman.unified_kalman import UnifiedKalman
    UNIFIED_KALMAN_AVAILABLE = True
except ImportError:
    try:
        # 絶対インポートを試行（パス調整付き）
        import sys
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from indicators.kalman.unified_kalman import UnifiedKalman
        UNIFIED_KALMAN_AVAILABLE = True
    except ImportError:
        UnifiedKalman = None
        UNIFIED_KALMAN_AVAILABLE = False


@dataclass
class XMAMAResult:
    """X_MAMA/X_FAMAの計算結果"""
    mama_values: np.ndarray      # X_MAMAライン値
    fama_values: np.ndarray      # X_FAMAライン値
    period_values: np.ndarray    # 計算されたPeriod値
    alpha_values: np.ndarray     # 計算されたAlpha値
    phase_values: np.ndarray     # Phase値
    i1_values: np.ndarray        # InPhase component
    q1_values: np.ndarray        # Quadrature component
    filtered_price: np.ndarray   # カルマンフィルター後の価格（使用した場合）


@njit(fastmath=True, cache=True)
def calculate_zero_lag_processing(mama_values: np.ndarray, alpha_values: np.ndarray) -> np.ndarray:
    """
    X_MAMA値に対してゼロラグ処理を適用する（Numba最適化版）
    X_MAMA内部のアルファ値を使用してゼロラグ処理を実行
    
    Args:
        mama_values: X_MAMA値の配列
        alpha_values: X_MAMA内部のアルファ値の配列
    
    Returns:
        ゼロラグ処理後の値配列
    """
    length = len(mama_values)
    result = np.full(length, np.nan, dtype=np.float64)
    
    if length == 0:
        return result
    
    # EMA値の配列（X_MAMA値のEMA）
    ema_values = np.full(length, np.nan, dtype=np.float64)
    
    # ラグ除去データの配列
    lag_reduced_data = np.full(length, np.nan, dtype=np.float64)
    
    # 最初の有効な値を見つける
    first_valid_idx = -1
    for i in range(length):
        if not np.isnan(mama_values[i]):
            ema_values[i] = mama_values[i]
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return result
    
    # EMAの計算（X_MAMA値のEMA、X_MAMAのアルファ値を使用）
    for i in range(first_valid_idx + 1, length):
        if not np.isnan(mama_values[i]) and not np.isnan(alpha_values[i]):
            if not np.isnan(ema_values[i-1]):
                ema_values[i] = alpha_values[i] * mama_values[i] + (1.0 - alpha_values[i]) * ema_values[i-1]
            else:
                ema_values[i] = mama_values[i]
    
    # ラグ除去データの計算
    for i in range(length):
        if not np.isnan(mama_values[i]) and not np.isnan(ema_values[i]):
            lag_reduced_data[i] = 2.0 * mama_values[i] - ema_values[i]
    
    # ZLEMAの計算
    # 最初の値はラグ除去データと同じ
    start_idx = first_valid_idx
    if start_idx < length and not np.isnan(lag_reduced_data[start_idx]):
        result[start_idx] = lag_reduced_data[start_idx]
    
    # 以降はラグ除去データのEMAを計算（X_MAMAのアルファ値を使用）
    for i in range(start_idx + 1, length):
        if not np.isnan(lag_reduced_data[i]) and not np.isnan(alpha_values[i]):
            if not np.isnan(result[i-1]):
                result[i] = alpha_values[i] * lag_reduced_data[i] + (1.0 - alpha_values[i]) * result[i-1]
            else:
                result[i] = lag_reduced_data[i]
    
    return result


@njit(fastmath=True, cache=True)
def calculate_x_mama_fama(
    price: np.ndarray,
    fast_limit: float = 0.5,
    slow_limit: float = 0.05,
    use_zero_lag: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    X_MAMA/X_FAMAを計算する（Numba最適化版）
    
    Args:
        price: 価格配列（通常は(H+L)/2）
        fast_limit: 速いリミット（デフォルト: 0.5）
        slow_limit: 遅いリミット（デフォルト: 0.05）
        use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
    
    Returns:
        Tuple[np.ndarray, ...]: X_MAMA値, X_FAMA値, Period値, Alpha値, Phase値, I1値, Q1値
    """
    length = len(price)
    
    # 変数の初期化
    smooth = np.zeros(length, dtype=np.float64)
    detrender = np.zeros(length, dtype=np.float64)
    i1 = np.zeros(length, dtype=np.float64)
    q1 = np.zeros(length, dtype=np.float64)
    j_i = np.zeros(length, dtype=np.float64)
    j_q = np.zeros(length, dtype=np.float64)
    i2 = np.zeros(length, dtype=np.float64)
    q2 = np.zeros(length, dtype=np.float64)
    re = np.zeros(length, dtype=np.float64)
    im = np.zeros(length, dtype=np.float64)
    period = np.zeros(length, dtype=np.float64)
    smooth_period = np.zeros(length, dtype=np.float64)
    phase = np.zeros(length, dtype=np.float64)
    delta_phase = np.zeros(length, dtype=np.float64)
    alpha = np.zeros(length, dtype=np.float64)
    mama = np.zeros(length, dtype=np.float64)
    fama = np.zeros(length, dtype=np.float64)
    
    # 初期値設定 - すべて有効な値で初期化
    for i in range(min(7, length)):
        smooth[i] = price[i] if i < length else 100.0
        detrender[i] = 0.0
        i1[i] = 0.0
        q1[i] = 0.0
        j_i[i] = 0.0
        j_q[i] = 0.0
        i2[i] = 0.0
        q2[i] = 0.0
        re[i] = 0.0
        im[i] = 0.0
        period[i] = 20.0  # 初期値として有効な値を設定
        smooth_period[i] = 20.0
        phase[i] = 0.0
        delta_phase[i] = 1.0
        alpha[i] = 0.05  # slow_limitで初期化
        mama[i] = price[i] if i < length else 100.0
        fama[i] = price[i] if i < length else 100.0
    
    # CurrentBar > 5の条件 (インデックス5から開始、0ベースなので)
    for i in range(5, length):
        # 価格のスムージング: Smooth = (4*Price + 3*Price[1] + 2*Price[2] + Price[3]) / 10
        if i >= 3:  # 最低4つの価格が必要
            smooth[i] = (4.0 * price[i] + 3.0 * price[i-1] + 2.0 * price[i-2] + price[i-3]) / 10.0
        else:
            smooth[i] = price[i]  # フォールバック
            continue
        
        # 前回のPeriod値を取得（初回は20に設定）
        prev_period = period[i-1] if i > 6 and not np.isnan(period[i-1]) else 20.0
        
        # Detrender計算
        if i >= 6:
            detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] - 
                           0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * prev_period + 0.54)
        else:
            detrender[i] = 0.0  # 初期値として0を設定
            continue
        
        # InPhaseとQuadratureコンポーネントの計算
        if i >= 6:
            q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] - 
                    0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * prev_period + 0.54)
            i1[i] = detrender[i-3] if i >= 9 else 0.0
        else:
            q1[i] = 0.0
            i1[i] = 0.0
            continue
        
        # 90度位相を進める
        if i >= 6:
            j_i[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] - 
                     0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * prev_period + 0.54)
            j_q[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] - 
                     0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * prev_period + 0.54)
        else:
            j_i[i] = 0.0
            j_q[i] = 0.0
            continue
        
        # Phasor加算（3バー平均）
        i2[i] = i1[i] - j_q[i]
        q2[i] = q1[i] + j_i[i]
        
        # IとQコンポーネントのスムージング
        if i > 5:
            i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1]
            q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1]
        
        # Homodyne Discriminator
        if i > 5:
            re[i] = i2[i] * i2[i-1] + q2[i] * q2[i-1]
            im[i] = i2[i] * q2[i-1] - q2[i] * i2[i-1]
            
            # ReとImのスムージング
            re[i] = 0.2 * re[i] + 0.8 * re[i-1]
            im[i] = 0.2 * im[i] + 0.8 * im[i-1]
        else:
            re[i] = 0.0
            im[i] = 0.0
            continue
        
        # Period計算
        if not np.isnan(im[i]) and not np.isnan(re[i]) and im[i] != 0.0 and re[i] != 0.0:
            # ArcTangent計算 - atan2を使用してより安全に計算
            atan_result = math.atan2(im[i], re[i]) * 180.0 / math.pi
            if abs(atan_result) > 0.001:  # 0に近すぎる値を避ける
                period[i] = 360.0 / abs(atan_result)
            else:
                period[i] = period[i-1] if i > 6 and not np.isnan(period[i-1]) else 20.0
            
            # Period制限
            if i > 5 and not np.isnan(period[i-1]):
                if period[i] > 1.5 * period[i-1]:
                    period[i] = 1.5 * period[i-1]
                elif period[i] < 0.67 * period[i-1]:
                    period[i] = 0.67 * period[i-1]
            
            if period[i] < 6.0:
                period[i] = 6.0
            elif period[i] > 50.0:
                period[i] = 50.0
            
            # Periodのスムージング
            if i > 5 and not np.isnan(period[i-1]):
                period[i] = 0.2 * period[i] + 0.8 * period[i-1]
        else:
            period[i] = period[i-1] if i > 5 and not np.isnan(period[i-1]) else 20.0
        
        # SmoothPeriod計算
        if i > 5 and not np.isnan(smooth_period[i-1]):
            smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i-1]
        else:
            smooth_period[i] = period[i]
        
        # Phase計算
        if not np.isnan(i1[i]) and not np.isnan(q1[i]):
            if abs(i1[i]) > 1e-10:  # i1が0に近すぎない場合のみ計算
                phase[i] = math.atan2(q1[i], i1[i]) * 180.0 / math.pi
            else:
                phase[i] = phase[i-1] if i > 5 else 0.0
        else:
            phase[i] = phase[i-1] if i > 5 else 0.0
        
        # DeltaPhase計算
        if i > 5:
            delta_phase[i] = abs(phase[i-1] - phase[i])
            if delta_phase[i] < 1.0:
                delta_phase[i] = 1.0
        else:
            delta_phase[i] = 1.0
        
        # Alpha計算 - ゼロ除算を避ける
        if delta_phase[i] > 0:
            alpha[i] = fast_limit / delta_phase[i]
            if alpha[i] < slow_limit:
                alpha[i] = slow_limit
            elif alpha[i] > fast_limit:
                alpha[i] = fast_limit
        else:
            alpha[i] = slow_limit
        
        # X_MAMA計算
        if i > 5 and not np.isnan(mama[i-1]) and not np.isnan(alpha[i]):
            mama[i] = alpha[i] * price[i] + (1.0 - alpha[i]) * mama[i-1]
        else:
            mama[i] = price[i]  # 初期値として価格を使用
        
        # X_FAMA計算
        if i > 5 and not np.isnan(fama[i-1]) and not np.isnan(mama[i]) and not np.isnan(alpha[i]):
            fama[i] = 0.5 * alpha[i] * mama[i] + (1.0 - 0.5 * alpha[i]) * fama[i-1]
        else:
            fama[i] = mama[i]  # 初期値としてMAMA値を使用
    
    # ゼロラグ処理の適用（オプション）
    if use_zero_lag:
        # X_MAMAにゼロラグ処理を適用（X_MAMAのアルファ値を使用）
        mama_zero_lag = calculate_zero_lag_processing(mama, alpha)
        
        # X_FAMAにゼロラグ処理を適用（X_MAMAのアルファ値を使用）
        fama_zero_lag = calculate_zero_lag_processing(fama, alpha)
        
        # 有効な値のみを使用（NaN値は元の値を保持）
        for i in range(length):
            if not np.isnan(mama_zero_lag[i]):
                mama[i] = mama_zero_lag[i]
            
            if not np.isnan(fama_zero_lag[i]):
                fama[i] = fama_zero_lag[i]
    
    return mama, fama, period, alpha, phase, i1, q1


class X_MAMA(Indicator):
    """
    X_MAMA (eXtended Mother of Adaptive Moving Average) インジケーター
    
    標準MAMAインジケーターに以下の機能を追加した拡張版：
    - カルマンフィルターによる価格ソースの前処理（オプション）
    - ゼロラグ処理による応答性の向上（オプション）
    - より高度な適応性とノイズフィルタリング
    
    特徴:
    - 市場サイクルの変化に適応
    - トレンド強度に応じて応答速度を調整
    - ノイズフィルタリング機能
    - カルマンフィルターとゼロラグ処理の統合
    """
    
    def __init__(
        self,
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.05,              # 低速制限値
        src_type: str = 'hlc3',                # ソースタイプ
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
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
        """
        # インジケーター名の作成
        indicator_name = f"X_MAMA(fast={fast_limit}, slow={slow_limit}, {src_type}"
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
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_observation_noise = kalman_observation_noise
        self.use_zero_lag = use_zero_lag
        
        # ソースタイプの検証（PriceSourceから利用可能なタイプを取得）
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
        if fast_limit <= 0 or fast_limit > 1:
            raise ValueError("fast_limitは0より大きく1以下である必要があります")
        if slow_limit <= 0 or slow_limit > 1:
            raise ValueError("slow_limitは0より大きく1以下である必要があります")
        if slow_limit >= fast_limit:
            raise ValueError("slow_limitはfast_limitより小さい必要があります")
        if use_kalman_filter and kalman_process_noise <= 0:
            raise ValueError("kalman_process_noiseは0より大きい必要があります")
        
        # カルマンフィルターの初期化（オプション）
        self.kalman_filter = None
        if self.use_kalman_filter:
            if not UNIFIED_KALMAN_AVAILABLE:
                self.logger.error("統合カルマンフィルターが利用できません。indicators.kalman.unified_kalmanをインポートできません。")
                self.use_kalman_filter = False
                self.logger.warning("カルマンフィルター機能を無効にしました")
            else:
                try:
                    self.kalman_filter = UnifiedKalman(
                        filter_type=self.kalman_filter_type,
                        src_type=self.src_type,
                        process_noise_scale=self.kalman_process_noise,
                        observation_noise_scale=self.kalman_observation_noise
                    )
                    self.logger.info(f"統合カルマンフィルターを初期化しました: {self.kalman_filter_type}")
                except Exception as e:
                    self.logger.error(f"統合カルマンフィルターの初期化に失敗: {e}")
                    self.use_kalman_filter = False
                    self.logger.warning("カルマンフィルター機能を無効にしました")
        
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
            zero_lag_sig = f"{self.zero_lag_period}" if self.use_zero_lag else "None"
            params_sig = f"{self.fast_limit}_{self.slow_limit}_{self.src_type}_{kalman_sig}_{zero_lag_sig}"
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.fast_limit}_{self.slow_limit}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> XMAMAResult:
        """
        X_MAMA/X_FAMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、OHLC + 選択したソースタイプに必要なカラムが必要
        
        Returns:
            XMAMAResult: X_MAMA/X_FAMAの値と計算中間値を含む結果
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
                return XMAMAResult(
                    mama_values=cached_result.mama_values.copy(),
                    fama_values=cached_result.fama_values.copy(),
                    period_values=cached_result.period_values.copy(),
                    alpha_values=cached_result.alpha_values.copy(),
                    phase_values=cached_result.phase_values.copy(),
                    i1_values=cached_result.i1_values.copy(),
                    q1_values=cached_result.q1_values.copy(),
                    filtered_price=cached_result.filtered_price.copy()
                )
            
            # 価格ソースの計算
            price_source = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換（float64型で統一）
            price_source = np.asarray(price_source, dtype=np.float64)
            
            # カルマンフィルターによる前処理（オプション）
            filtered_price = price_source.copy()
            if self.use_kalman_filter and self.kalman_filter is not None:
                try:
                    kalman_result = self.kalman_filter.calculate(data)
                    
                    # カルマンフィルター結果の詳細なデバッグ
                    self.logger.debug(f"カルマンフィルター結果タイプ: {type(kalman_result)}")
                    
                    # カルマンフィルター結果の形式を確認
                    kalman_values = None
                    
                    # UnifiedKalmanResult専用の値抽出
                    if hasattr(kalman_result, 'values'):
                        # UnifiedKalmanResult or 他の標準化結果の場合
                        kalman_values = kalman_result.values
                        self.logger.debug(f"values属性を使用: {type(kalman_values)}")
                    elif hasattr(kalman_result, 'filtered_values'):
                        # UKFResult や他のフィルター結果の場合
                        kalman_values = kalman_result.filtered_values
                        self.logger.debug(f"filtered_values属性を使用: {type(kalman_values)}")
                    elif isinstance(kalman_result, (np.ndarray, list)):
                        # 直接配列の場合
                        kalman_values = kalman_result
                        self.logger.debug(f"直接配列を使用: {type(kalman_values)}")
                    else:
                        # その他の場合
                        kalman_values = kalman_result
                        self.logger.debug(f"その他の形式を使用: {type(kalman_values)}")
                    
                    
                    # Noneでない場合のみ処理続行
                    if kalman_values is not None:
                        # NumPy配列に変換（統一されたインターフェースで処理）
                        try:
                            # NumPy配列に変換
                            kalman_values = np.asarray(kalman_values, dtype=np.float64)
                            
                            # 配列の次元をチェック
                            if kalman_values.ndim == 0:
                                # スカラー値の場合はエラー
                                raise ValueError("カルマンフィルター結果がスカラー値です")
                            elif kalman_values.ndim > 1:
                                # 多次元配列の場合は1次元に変換
                                kalman_values = kalman_values.flatten()
                            
                            # フィルタリングされた価格の検証
                            if len(kalman_values) != len(price_source):
                                self.logger.warning(f"カルマンフィルター結果のサイズ不一致: {len(kalman_values)} != {len(price_source)}。元の価格を使用します。")
                                filtered_price = price_source.copy()
                            else:
                                # NaN値の処理
                                nan_mask = np.isnan(kalman_values)
                                if np.any(nan_mask):
                                    kalman_values[nan_mask] = price_source[nan_mask]
                                
                                filtered_price = kalman_values
                                self.logger.debug("カルマンフィルターによる価格前処理を適用しました")
                        except Exception as array_error:
                            self.logger.warning(f"カルマンフィルター結果の配列変換エラー: {array_error}。元の価格を使用します。")
                            filtered_price = price_source.copy()
                    else:
                        self.logger.warning("カルマンフィルター結果がNoneです。元の価格を使用します。")
                        filtered_price = price_source.copy()
                        
                except Exception as e:
                    self.logger.warning(f"カルマンフィルター適用中にエラー: {e}。元の価格を使用します。")
                    filtered_price = price_source.copy()
            
            # データ長の検証
            data_length = len(filtered_price)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < 10:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低10点以上を推奨します。")
            
            # X_MAMA/X_FAMAの計算（Numba最適化関数を使用）
            mama_values, fama_values, period_values, alpha_values, phase_values, i1_values, q1_values = calculate_x_mama_fama(
                filtered_price, self.fast_limit, self.slow_limit, self.use_zero_lag
            )
            
            # 結果の保存（参照問題を避けるため必要な部分だけコピー）
            result = XMAMAResult(
                mama_values=mama_values.copy(),
                fama_values=fama_values.copy(),
                period_values=period_values.copy(),
                alpha_values=alpha_values.copy(),
                phase_values=phase_values.copy(),
                i1_values=i1_values.copy(),
                q1_values=q1_values.copy(),
                filtered_price=filtered_price.copy()
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
            
            self._values = mama_values  # 基底クラスの要件を満たすため（X_MAMA値をメインとする）
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"X_MAMA/X_FAMA計算中にエラー: {error_msg}\\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = XMAMAResult(
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
        """X_MAMA値のみを取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.mama_values.copy()
    
    def get_mama_values(self) -> Optional[np.ndarray]:
        """
        X_MAMA値を取得する
        
        Returns:
            np.ndarray: X_MAMA値
        """
        return self.get_values()
    
    def get_fama_values(self) -> Optional[np.ndarray]:
        """
        X_FAMA値を取得する
        
        Returns:
            np.ndarray: X_FAMA値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.fama_values.copy()
    
    def get_period_values(self) -> Optional[np.ndarray]:
        """
        Period値を取得する
        
        Returns:
            np.ndarray: Period値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.period_values.copy()
    
    def get_alpha_values(self) -> Optional[np.ndarray]:
        """
        Alpha値を取得する
        
        Returns:
            np.ndarray: Alpha値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.alpha_values.copy()
    
    def get_phase_values(self) -> Optional[np.ndarray]:
        """
        Phase値を取得する
        
        Returns:
            np.ndarray: Phase値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.phase_values.copy()
    
    def get_i1_values(self) -> Optional[np.ndarray]:
        """
        I1値を取得する
        
        Returns:
            np.ndarray: I1値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.i1_values.copy()
    
    def get_q1_values(self) -> Optional[np.ndarray]:
        """
        Q1値を取得する
        
        Returns:
            np.ndarray: Q1値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.q1_values.copy()
    
    def get_inphase_quadrature(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        InPhaseとQuadratureコンポーネントを取得する
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (I1値, Q1値)
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.i1_values.copy(), result.q1_values.copy()
    
    def get_filtered_price(self) -> Optional[np.ndarray]:
        """
        カルマンフィルター後の価格を取得する
        
        Returns:
            np.ndarray: フィルタリングされた価格
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.filtered_price.copy()
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'fast_limit': self.fast_limit,
            'slow_limit': self.slow_limit,
            'src_type': self.src_type,
            'use_kalman_filter': self.use_kalman_filter,
            'kalman_filter_type': self.kalman_filter_type if self.use_kalman_filter else None,
            'kalman_process_noise': self.kalman_process_noise if self.use_kalman_filter else None,
            'kalman_observation_noise': self.kalman_observation_noise if self.use_kalman_filter else None,
            'use_zero_lag': self.use_zero_lag,
            'zero_lag_period': self.zero_lag_period if self.use_zero_lag else None,
            'description': '拡張適応型移動平均線（カルマンフィルター・ゼロラグ処理対応）'
        }
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        if self.kalman_filter:
            self.kalman_filter.reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_x_mama(
    data: Union[pd.DataFrame, np.ndarray],
    fast_limit: float = 0.5,
    slow_limit: float = 0.05,
    src_type: str = 'hlc3',
    use_kalman_filter: bool = False,
    kalman_filter_type: str = 'unscented',
    kalman_process_noise: float = 0.01,
    kalman_observation_noise: float = 0.001,
    use_zero_lag: bool = True,
    zero_lag_period: int = 10,
    **kwargs
) -> np.ndarray:
    """
    X_MAMAの計算（便利関数）
    
    Args:
        data: 価格データ
        fast_limit: 高速制限値
        slow_limit: 低速制限値
        src_type: ソースタイプ
        use_kalman_filter: カルマンフィルターを使用するか
        kalman_filter_type: カルマンフィルタータイプ
        kalman_process_noise: プロセスノイズ
        kalman_observation_noise: 観測ノイズ
        use_zero_lag: ゼロラグ処理を使用するか
        zero_lag_period: ゼロラグ処理の期間
        **kwargs: その他のパラメータ
        
    Returns:
        X_MAMA値
    """
    indicator = X_MAMA(
        fast_limit=fast_limit,
        slow_limit=slow_limit,
        src_type=src_type,
        use_kalman_filter=use_kalman_filter,
        kalman_filter_type=kalman_filter_type,
        kalman_process_noise=kalman_process_noise,
        kalman_observation_noise=kalman_observation_noise,
        use_zero_lag=use_zero_lag,
        zero_lag_period=zero_lag_period,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.mama_values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== X_MAMA インジケーターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # トレンドとレンジが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # トレンド相場
            change = 0.002 + np.random.normal(0, 0.01)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.008)
        elif i < 150:  # 強いトレンド相場
            change = 0.004 + np.random.normal(0, 0.015)
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
    
    # 基本版X_MAMAをテスト
    print("\\n基本版X_MAMAをテスト中...")
    x_mama_basic = X_MAMA(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        use_kalman_filter=False,
        use_zero_lag=False,  # ゼロラグ処理を無効にしてテスト
        zero_lag_period=20
    )
    try:
        result_basic = x_mama_basic.calculate(df)
        print(f"  X_MAMA結果の型: {type(result_basic)}")
        print(f"  MAMA配列の形状: {result_basic.mama_values.shape}")
        print(f"  FAMA配列の形状: {result_basic.fama_values.shape}")
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()
        result_basic = None
    
    if result_basic is not None:
        valid_count = np.sum(~np.isnan(result_basic.mama_values))
        mean_mama = np.nanmean(result_basic.mama_values)
        mean_fama = np.nanmean(result_basic.fama_values)
        
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  平均X_MAMA: {mean_mama:.4f}")
        print(f"  平均X_FAMA: {mean_fama:.4f}")
    else:
        print("  基本版X_MAMAの計算に失敗しました")
    
    # ゼロラグ処理版をテスト
    print("\\nゼロラグ処理版X_MAMAをテスト中...")
    x_mama_zero_lag = X_MAMA(
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='hlc3',
        use_kalman_filter=False,
        use_zero_lag=True,
        zero_lag_period=10  # 期間を短くしてテスト
    )
    try:
        result_zero_lag = x_mama_zero_lag.calculate(df)
        
        valid_count_zero_lag = np.sum(~np.isnan(result_zero_lag.mama_values))
        mean_mama_zero_lag = np.nanmean(result_zero_lag.mama_values)
        mean_fama_zero_lag = np.nanmean(result_zero_lag.fama_values)
        
        print(f"  有効値数: {valid_count_zero_lag}/{len(df)}")
        print(f"  平均X_MAMA（ゼロラグ）: {mean_mama_zero_lag:.4f}")
        print(f"  平均X_FAMA（ゼロラグ）: {mean_fama_zero_lag:.4f}")
        
        # 比較統計
        if result_basic is not None and valid_count > 0 and valid_count_zero_lag > 0:
            min_length = min(valid_count, valid_count_zero_lag)
            correlation = np.corrcoef(
                result_basic.mama_values[~np.isnan(result_basic.mama_values)][-min_length:],
                result_zero_lag.mama_values[~np.isnan(result_zero_lag.mama_values)][-min_length:]
            )[0, 1]
            print(f"  基本版とゼロラグ版の相関: {correlation:.4f}")
    except Exception as e:
        print(f"  ゼロラグ処理版でエラー: {e}")
        import traceback
        traceback.print_exc()
    
    # カルマンフィルター版をテスト（利用可能な場合）
    if UNIFIED_KALMAN_AVAILABLE:
        print("\\nカルマンフィルター機能は利用できますが、テストではスキップします")
    else:
        print("\\nカルマンフィルター機能は利用できません")
    
    print("\\n=== テスト完了 ===")