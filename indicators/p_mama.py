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
class PMAMAResult:
    """P_MAMA/P_FAMAの計算結果"""
    mama_values: np.ndarray      # P_MAMAライン値
    fama_values: np.ndarray      # P_FAMAライン値
    phase_values: np.ndarray     # フェーザー角度値
    real_values: np.ndarray      # Real component
    imag_values: np.ndarray      # Imaginary component
    alpha_values: np.ndarray     # 計算されたAlpha値
    state_values: np.ndarray     # トレンド状態値（+1: 上昇, 0: サイクリング, -1: 下降）
    instantaneous_period: np.ndarray  # 瞬間周期
    filtered_price: np.ndarray   # カルマンフィルター後の価格（使用した場合）


@njit(fastmath=True, cache=True)
def calculate_zero_lag_processing(mama_values: np.ndarray, alpha_values: np.ndarray) -> np.ndarray:
    """
    P_MAMA値に対してゼロラグ処理を適用する（Numba最適化版）
    P_MAMA内部のアルファ値を使用してゼロラグ処理を実行
    
    Args:
        mama_values: P_MAMA値の配列
        alpha_values: P_MAMA内部のアルファ値の配列
    
    Returns:
        ゼロラグ処理後の値配列
    """
    length = len(mama_values)
    result = np.full(length, np.nan, dtype=np.float64)
    
    if length == 0:
        return result
    
    # EMA値の配列（P_MAMA値のEMA）
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
    
    # EMAの計算（P_MAMA値のEMA、P_MAMAのアルファ値を使用）
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
    
    # 以降はラグ除去データのEMAを計算（P_MAMAのアルファ値を使用）
    for i in range(start_idx + 1, length):
        if not np.isnan(lag_reduced_data[i]) and not np.isnan(alpha_values[i]):
            if not np.isnan(result[i-1]):
                result[i] = alpha_values[i] * lag_reduced_data[i] + (1.0 - alpha_values[i]) * result[i-1]
            else:
                result[i] = lag_reduced_data[i]
    
    return result


@njit(fastmath=True, cache=True)
def calculate_phasor_correlation(signal: np.ndarray, period: int, start_idx: int) -> Tuple[float, float]:
    """
    シグナルとcos/sinの相関を計算する
    
    Args:
        signal: 入力シグナル
        period: 固定周期
        start_idx: 開始インデックス
    
    Returns:
        Tuple[float, float]: (Real, Imaginary) components
    """
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    
    # cosineとの相関（Real component）
    for count in range(period):
        idx = start_idx - count
        if idx >= 0 and idx < len(signal):
            x = signal[idx]
            y = math.cos(2.0 * math.pi * count / period)
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
            syy += y * y
    
    real = 0.0
    if (period * sxx - sx * sx > 0) and (period * syy - sy * sy > 0):
        real = (period * sxy - sx * sy) / math.sqrt((period * sxx - sx * sx) * (period * syy - sy * sy))
    
    # sineとの相関（Imaginary component）
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    
    for count in range(period):
        idx = start_idx - count
        if idx >= 0 and idx < len(signal):
            x = signal[idx]
            y = -math.sin(2.0 * math.pi * count / period)
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
            syy += y * y
    
    imag = 0.0
    if (period * sxx - sx * sx > 0) and (period * syy - sy * sy > 0):
        imag = (period * sxy - sx * sy) / math.sqrt((period * sxx - sx * sx) * (period * syy - sy * sy))
    
    return real, imag


@njit(fastmath=True, cache=True)
def calculate_p_mama_fama(
    price: np.ndarray,
    period: int = 28,
    fast_limit: float = 0.5,
    slow_limit: float = 0.05,
    use_zero_lag: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    P_MAMA/P_FAMAを計算する（Numba最適化版）
    John Ehlersのフェーザー分析を使用してアルファを計算
    
    Args:
        price: 価格配列
        period: 固定周期（デフォルト: 28）
        fast_limit: 速いリミット（デフォルト: 0.5）
        slow_limit: 遅いリミット（デフォルト: 0.05）
        use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
    
    Returns:
        Tuple[np.ndarray, ...]: P_MAMA値, P_FAMA値, Phase値, Real値, Imaginary値, Alpha値, State値, InstantaneousPeriod値
    """
    length = len(price)
    
    # 変数の初期化
    real = np.zeros(length, dtype=np.float64)
    imag = np.zeros(length, dtype=np.float64)
    angle = np.zeros(length, dtype=np.float64)
    instantaneous_period = np.full(length, 60.0, dtype=np.float64)  # 初期値は60に設定
    alpha = np.full(length, slow_limit, dtype=np.float64)
    mama = np.zeros(length, dtype=np.float64)
    fama = np.zeros(length, dtype=np.float64)
    state = np.zeros(length, dtype=np.float64)
    
    # 初期値設定
    for i in range(min(period, length)):
        mama[i] = price[i] if i < length else 100.0
        fama[i] = price[i] if i < length else 100.0
        angle[i] = 0.0
    
    # フェーザー分析の計算
    for i in range(period, length):
        # フェーザー相関の計算
        real_val, imag_val = calculate_phasor_correlation(price, period, i)
        real[i] = real_val
        imag[i] = imag_val
        
        # 角度の計算
        if real[i] != 0.0:
            angle[i] = 90.0 - math.atan2(imag[i], real[i]) * 180.0 / math.pi
        else:
            angle[i] = angle[i-1] if i > 0 else 0.0
        
        # 角度の補正（リアル部が負の場合）
        if real[i] < 0.0:
            angle[i] = angle[i] - 180.0
        
        # 角度のラップアラウンド補正
        if i > 0:
            if abs(angle[i-1]) - abs(angle[i] - 360.0) < angle[i] - angle[i-1] and angle[i] > 90.0 and angle[i-1] < -90.0:
                angle[i] = angle[i] - 360.0
        
        # 角度は逆向きに進まない
        if i > 0:
            if angle[i] < angle[i-1] and ((angle[i] > -135.0 and angle[i-1] < 135.0) or (angle[i] < -90.0 and angle[i-1] < -90.0)):
                angle[i] = angle[i-1]
        
        # 瞬間周期の計算（角度変化率から）
        if i > 0:
            delta_angle = angle[i] - angle[i-1]
            if delta_angle <= 0.0:
                delta_angle = 1.0  # 最小値を設定
            
            if delta_angle != 0.0:
                inst_period = 360.0 / delta_angle
                if inst_period > 60.0:
                    inst_period = 60.0
                elif inst_period < 6.0:
                    inst_period = 6.0
                instantaneous_period[i] = inst_period
            else:
                instantaneous_period[i] = instantaneous_period[i-1]
        
        # トレンド状態の判定
        if i > 0:
            delta_angle = angle[i] - angle[i-1]
            if delta_angle <= 6.0:  # トレンドモード
                if angle[i] >= 90.0 or angle[i] <= -90.0:
                    state[i] = 1.0  # 上昇トレンド
                elif angle[i] > -90.0 and angle[i] < 90.0:
                    state[i] = -1.0  # 下降トレンド
                else:
                    state[i] = 0.0  # サイクリング
            else:
                state[i] = 0.0  # サイクリング
        
        # アルファの計算（フェーザー角度に基づく）
        if instantaneous_period[i] > 0:
            # より滑らかなアルファ計算
            cycle_factor = 2.0 / (instantaneous_period[i] + 1.0)
            alpha[i] = cycle_factor
            
            # リミットの適用
            if alpha[i] < slow_limit:
                alpha[i] = slow_limit
            elif alpha[i] > fast_limit:
                alpha[i] = fast_limit
        else:
            alpha[i] = slow_limit
        
        # P_MAMA計算
        if i > 0 and not np.isnan(mama[i-1]) and not np.isnan(alpha[i]):
            mama[i] = alpha[i] * price[i] + (1.0 - alpha[i]) * mama[i-1]
        else:
            mama[i] = price[i]
        
        # P_FAMA計算
        if i > 0 and not np.isnan(fama[i-1]) and not np.isnan(mama[i]) and not np.isnan(alpha[i]):
            fama[i] = 0.5 * alpha[i] * mama[i] + (1.0 - 0.5 * alpha[i]) * fama[i-1]
        else:
            fama[i] = mama[i]
    
    # ゼロラグ処理の適用（オプション）
    if use_zero_lag:
        # P_MAMAにゼロラグ処理を適用
        mama_zero_lag = calculate_zero_lag_processing(mama, alpha)
        
        # P_FAMAにゼロラグ処理を適用
        fama_zero_lag = calculate_zero_lag_processing(fama, alpha)
        
        # 有効な値のみを使用（NaN値は元の値を保持）
        for i in range(length):
            if not np.isnan(mama_zero_lag[i]):
                mama[i] = mama_zero_lag[i]
            
            if not np.isnan(fama_zero_lag[i]):
                fama[i] = fama_zero_lag[i]
    
    return mama, fama, angle, real, imag, alpha, state, instantaneous_period


class P_MAMA(Indicator):
    """
    P_MAMA (Phasor-based Mother of Adaptive Moving Average) インジケーター
    
    John Ehlersのフェーザー分析論文に基づいて実装されたMAMAインジケーター。
    従来のMESAではなく、フェーザー分析を使用してアルファ値を計算する。
    
    特徴:
    - フェーザー分析による高精度な市場サイクル検出
    - トレンドとサイクルの自動判定
    - ゼロラグ処理による応答性の向上（オプション）
    - カルマンフィルターによる前処理（オプション）
    """
    
    def __init__(
        self,
        period: int = 28,                      # フェーザー分析の固定周期
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.05,              # 低速制限値
        src_type: str = 'close',               # ソースタイプ
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
            period: フェーザー分析の固定周期（デフォルト: 28）
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
        """
        # インジケーター名の作成
        indicator_name = f"P_MAMA(period={period}, fast={fast_limit}, slow={slow_limit}, {src_type}"
        if use_kalman_filter:
            indicator_name += f", kalman={kalman_filter_type}"
        if use_zero_lag:
            indicator_name += ", zero_lag=True"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.period = period
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        self.src_type = src_type.lower()
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        self.kalman_process_noise = kalman_process_noise
        self.kalman_observation_noise = kalman_observation_noise
        self.use_zero_lag = use_zero_lag
        
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
            zero_lag_sig = "True" if self.use_zero_lag else "False"
            params_sig = f"{self.period}_{self.fast_limit}_{self.slow_limit}_{self.src_type}_{kalman_sig}_{zero_lag_sig}"
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.period}_{self.fast_limit}_{self.slow_limit}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> PMAMAResult:
        """
        P_MAMA/P_FAMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、OHLC + 選択したソースタイプに必要なカラムが必要
        
        Returns:
            PMAMAResult: P_MAMA/P_FAMAの値と計算中間値を含む結果
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
                return PMAMAResult(
                    mama_values=cached_result.mama_values.copy(),
                    fama_values=cached_result.fama_values.copy(),
                    phase_values=cached_result.phase_values.copy(),
                    real_values=cached_result.real_values.copy(),
                    imag_values=cached_result.imag_values.copy(),
                    alpha_values=cached_result.alpha_values.copy(),
                    state_values=cached_result.state_values.copy(),
                    instantaneous_period=cached_result.instantaneous_period.copy(),
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
            
            if data_length < self.period + 10:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低{self.period + 10}点以上を推奨します。")
            
            # P_MAMA/P_FAMAの計算（Numba最適化関数を使用）
            mama_values, fama_values, phase_values, real_values, imag_values, alpha_values, state_values, instantaneous_period = calculate_p_mama_fama(
                filtered_price, self.period, self.fast_limit, self.slow_limit, self.use_zero_lag
            )
            
            # 結果の保存（参照問題を避けるため必要な部分だけコピー）
            result = PMAMAResult(
                mama_values=mama_values.copy(),
                fama_values=fama_values.copy(),
                phase_values=phase_values.copy(),
                real_values=real_values.copy(),
                imag_values=imag_values.copy(),
                alpha_values=alpha_values.copy(),
                state_values=state_values.copy(),
                instantaneous_period=instantaneous_period.copy(),
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
            
            self._values = mama_values  # 基底クラスの要件を満たすため（P_MAMA値をメインとする）
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"P_MAMA/P_FAMA計算中にエラー: {error_msg}\\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = PMAMAResult(
                mama_values=np.array([]),
                fama_values=np.array([]),
                phase_values=np.array([]),
                real_values=np.array([]),
                imag_values=np.array([]),
                alpha_values=np.array([]),
                state_values=np.array([]),
                instantaneous_period=np.array([]),
                filtered_price=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """P_MAMA値のみを取得する（後方互換性のため）"""
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
        """P_MAMA値を取得する"""
        return self.get_values()
    
    def get_fama_values(self) -> Optional[np.ndarray]:
        """P_FAMA値を取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.fama_values.copy()
    
    def get_phase_values(self) -> Optional[np.ndarray]:
        """フェーザー角度値を取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.phase_values.copy()
    
    def get_real_values(self) -> Optional[np.ndarray]:
        """Real component値を取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.real_values.copy()
    
    def get_imag_values(self) -> Optional[np.ndarray]:
        """Imaginary component値を取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.imag_values.copy()
    
    def get_alpha_values(self) -> Optional[np.ndarray]:
        """Alpha値を取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.alpha_values.copy()
    
    def get_state_values(self) -> Optional[np.ndarray]:
        """トレンド状態値を取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.state_values.copy()
    
    def get_instantaneous_period(self) -> Optional[np.ndarray]:
        """瞬間周期値を取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.instantaneous_period.copy()
    
    def get_filtered_price(self) -> Optional[np.ndarray]:
        """カルマンフィルター後の価格を取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.filtered_price.copy()
    
    def get_phasor_components(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """フェーザーのRealとImaginaryコンポーネントを取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.real_values.copy(), result.imag_values.copy()
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'period': self.period,
            'fast_limit': self.fast_limit,
            'slow_limit': self.slow_limit,
            'src_type': self.src_type,
            'use_kalman_filter': self.use_kalman_filter,
            'kalman_filter_type': self.kalman_filter_type if self.use_kalman_filter else None,
            'kalman_process_noise': self.kalman_process_noise if self.use_kalman_filter else None,
            'kalman_observation_noise': self.kalman_observation_noise if self.use_kalman_filter else None,
            'use_zero_lag': self.use_zero_lag,
            'description': 'フェーザー分析ベース適応型移動平均線（カルマンフィルター・ゼロラグ処理対応）'
        }
    
    def reset(self) -> None:
        """インディケーターの状態をリセットする"""
        super().reset()
        if self.kalman_filter:
            self.kalman_filter.reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_p_mama(
    data: Union[pd.DataFrame, np.ndarray],
    period: int = 28,
    fast_limit: float = 0.5,
    slow_limit: float = 0.05,
    src_type: str = 'close',
    use_kalman_filter: bool = False,
    kalman_filter_type: str = 'unscented',
    kalman_process_noise: float = 0.01,
    kalman_observation_noise: float = 0.001,
    use_zero_lag: bool = True,
    **kwargs
) -> np.ndarray:
    """
    P_MAMAの計算（便利関数）
    
    Args:
        data: 価格データ
        period: フェーザー分析の固定周期
        fast_limit: 高速制限値
        slow_limit: 低速制限値
        src_type: ソースタイプ
        use_kalman_filter: カルマンフィルターを使用するか
        kalman_filter_type: カルマンフィルタータイプ
        kalman_process_noise: プロセスノイズ
        kalman_observation_noise: 観測ノイズ
        use_zero_lag: ゼロラグ処理を使用するか
        **kwargs: その他のパラメータ
        
    Returns:
        P_MAMA値
    """
    indicator = P_MAMA(
        period=period,
        fast_limit=fast_limit,
        slow_limit=slow_limit,
        src_type=src_type,
        use_kalman_filter=use_kalman_filter,
        kalman_filter_type=kalman_filter_type,
        kalman_process_noise=kalman_process_noise,
        kalman_observation_noise=kalman_observation_noise,
        use_zero_lag=use_zero_lag,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.mama_values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== P_MAMA インジケーターのテスト ===")
    
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
    
    # 基本版P_MAMAをテスト
    print("\\n基本版P_MAMAをテスト中...")
    p_mama_basic = P_MAMA(
        period=28,
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='close',
        use_kalman_filter=False,
        use_zero_lag=False
    )
    try:
        result_basic = p_mama_basic.calculate(df)
        print(f"  P_MAMA結果の型: {type(result_basic)}")
        print(f"  MAMA配列の形状: {result_basic.mama_values.shape}")
        print(f"  FAMA配列の形状: {result_basic.fama_values.shape}")
        print(f"  Phase配列の形状: {result_basic.phase_values.shape}")
        print(f"  State配列の形状: {result_basic.state_values.shape}")
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()
        result_basic = None
    
    if result_basic is not None:
        valid_count = np.sum(~np.isnan(result_basic.mama_values))
        mean_mama = np.nanmean(result_basic.mama_values)
        mean_fama = np.nanmean(result_basic.fama_values)
        mean_phase = np.nanmean(result_basic.phase_values)
        uptrend_count = np.sum(result_basic.state_values == 1)
        downtrend_count = np.sum(result_basic.state_values == -1)
        
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  平均P_MAMA: {mean_mama:.4f}")
        print(f"  平均P_FAMA: {mean_fama:.4f}")
        print(f"  平均Phase: {mean_phase:.2f}°")
        print(f"  上昇トレンド: {uptrend_count}期間")
        print(f"  下降トレンド: {downtrend_count}期間")
    else:
        print("  基本版P_MAMAの計算に失敗しました")
    
    # ゼロラグ処理版をテスト
    print("\\nゼロラグ処理版P_MAMAをテスト中...")
    p_mama_zero_lag = P_MAMA(
        period=28,
        fast_limit=0.5,
        slow_limit=0.05,
        src_type='close',
        use_kalman_filter=False,
        use_zero_lag=True
    )
    try:
        result_zero_lag = p_mama_zero_lag.calculate(df)
        
        valid_count_zero_lag = np.sum(~np.isnan(result_zero_lag.mama_values))
        mean_mama_zero_lag = np.nanmean(result_zero_lag.mama_values)
        mean_fama_zero_lag = np.nanmean(result_zero_lag.fama_values)
        
        print(f"  有効値数: {valid_count_zero_lag}/{len(df)}")
        print(f"  平均P_MAMA（ゼロラグ）: {mean_mama_zero_lag:.4f}")
        print(f"  平均P_FAMA（ゼロラグ）: {mean_fama_zero_lag:.4f}")
        
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