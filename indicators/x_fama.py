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
class XFAMAResult:
    """X_FAMAの計算結果"""
    frama_values: np.ndarray         # X_FRAMA値
    fast_fama_values: np.ndarray     # 高速X_FAMA値（アルファが2倍）
    fractal_dimension: np.ndarray    # フラクタル次元
    alpha_values: np.ndarray         # アルファ値
    filtered_price: np.ndarray       # カルマンフィルター後の価格（使用した場合）


@njit(fastmath=True, cache=True)
def calculate_zero_lag_processing_fama(frama_values: np.ndarray, alpha_values: np.ndarray) -> np.ndarray:
    """
    X_FRAMA値に対してゼロラグ処理を適用する（Numba最適化版）
    
    Args:
        frama_values: X_FRAMA値の配列
        alpha_values: アルファ値の配列
    
    Returns:
        ゼロラグ処理後の値配列
    """
    length = len(frama_values)
    result = np.full(length, np.nan, dtype=np.float64)
    
    if length == 0:
        return result
    
    # EMA値の配列（X_FRAMA値のEMA）
    ema_values = np.full(length, np.nan, dtype=np.float64)
    
    # ラグ除去データの配列
    lag_reduced_data = np.full(length, np.nan, dtype=np.float64)
    
    # 最初の有効な値を見つける
    first_valid_idx = -1
    for i in range(length):
        if not np.isnan(frama_values[i]):
            ema_values[i] = frama_values[i]
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return result
    
    # EMAの計算（X_FRAMA値のEMA、アルファ値を使用）
    for i in range(first_valid_idx + 1, length):
        if not np.isnan(frama_values[i]) and not np.isnan(alpha_values[i]):
            if not np.isnan(ema_values[i-1]):
                ema_values[i] = alpha_values[i] * frama_values[i] + (1.0 - alpha_values[i]) * ema_values[i-1]
            else:
                ema_values[i] = frama_values[i]
    
    # ラグ除去データの計算
    for i in range(length):
        if not np.isnan(frama_values[i]) and not np.isnan(ema_values[i]):
            lag_reduced_data[i] = 2.0 * frama_values[i] - ema_values[i]
    
    # ZLEMAの計算
    # 最初の値はラグ除去データと同じ
    start_idx = first_valid_idx
    if start_idx < length and not np.isnan(lag_reduced_data[start_idx]):
        result[start_idx] = lag_reduced_data[start_idx]
    
    # 以降はラグ除去データのEMAを計算（アルファ値を使用）
    for i in range(start_idx + 1, length):
        if not np.isnan(lag_reduced_data[i]) and not np.isnan(alpha_values[i]):
            if not np.isnan(result[i-1]):
                result[i] = alpha_values[i] * lag_reduced_data[i] + (1.0 - alpha_values[i]) * result[i-1]
            else:
                result[i] = lag_reduced_data[i]
    
    return result


@njit(fastmath=True, cache=True)
def calculate_x_frama_core(
    price: np.ndarray, 
    high: np.ndarray, 
    low: np.ndarray, 
    period: int, 
    fc: int, 
    sc: int,
    use_zero_lag: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    X_FAMAを計算する（Numba最適化版）
    
    Args:
        price: 価格配列
        high: 高値配列
        low: 安値配列
        period: 期間（偶数である必要がある）
        fc: Fast Constant
        sc: Slow Constant
        use_zero_lag: ゼロラグ処理を使用するか
    
    Returns:
        Tuple[np.ndarray, ...]: X_FRAMA値, Fast X_FRAMA値, フラクタル次元, アルファ値
    """
    length = len(price)
    
    # 結果配列を初期化
    frama = np.zeros(length, dtype=np.float64)
    fast_fama = np.zeros(length, dtype=np.float64)
    dimension = np.zeros(length, dtype=np.float64)
    alpha = np.zeros(length, dtype=np.float64)
    
    # 初期値設定
    for i in range(length):
        frama[i] = np.nan
        fast_fama[i] = np.nan
        dimension[i] = np.nan
        alpha[i] = np.nan
    
    # 計算に必要な最小期間
    min_period = period
    
    # w = log(2/(SC+1))
    w = np.log(2.0 / (sc + 1))
    
    # 最初の期間は価格をそのまま使用
    for i in range(min(min_period, length)):
        if not np.isnan(price[i]):
            frama[i] = price[i]
            fast_fama[i] = price[i]
            alpha[i] = 1.0
    
    # メインループ
    for i in range(min_period, length):
        if np.isnan(price[i]):
            frama[i] = frama[i-1] if i > 0 else np.nan
            fast_fama[i] = fast_fama[i-1] if i > 0 else np.nan
            continue
        
        # len1 = len/2
        len1 = period // 2
        
        # H1 = highest(high,len1)
        # L1 = lowest(low,len1)
        h1 = -np.inf
        l1 = np.inf
        for j in range(len1):
            if i - j >= 0:
                if high[i - j] > h1:
                    h1 = high[i - j]
                if low[i - j] < l1:
                    l1 = low[i - j]
        
        # N1 = (H1-L1)/len1
        n1 = (h1 - l1) / len1
        
        # H2 = highest(high,len)[len1]
        # L2 = lowest(low,len)[len1]
        h2 = -np.inf
        l2 = np.inf
        for j in range(len1, period):
            if i - j >= 0:
                if high[i - j] > h2:
                    h2 = high[i - j]
                if low[i - j] < l2:
                    l2 = low[i - j]
        
        # N2 = (H2-L2)/len1
        n2 = (h2 - l2) / len1
        
        # H3 = highest(high,len)
        # L3 = lowest(low,len)
        h3 = -np.inf
        l3 = np.inf
        for j in range(period):
            if i - j >= 0:
                if high[i - j] > h3:
                    h3 = high[i - j]
                if low[i - j] < l3:
                    l3 = low[i - j]
        
        # N3 = (H3-L3)/len
        n3 = (h3 - l3) / period
        
        # dimen1 = (log(N1+N2)-log(N3))/log(2)
        # dimen = iff(N1>0 and N2>0 and N3>0,dimen1,nz(dimen1[1]))
        if n1 > 0 and n2 > 0 and n3 > 0:
            dimen1 = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
            dimen = dimen1
        else:
            dimen = dimension[i-1] if i > 0 else 1.0
        
        dimension[i] = dimen
        
        # alpha1 = exp(w*(dimen-1))
        alpha1 = np.exp(w * (dimen - 1.0))
        
        # oldalpha = alpha1>1?1:(alpha1<0.01?0.01:alpha1)
        if alpha1 > 1.0:
            oldalpha = 1.0
        elif alpha1 < 0.01:
            oldalpha = 0.01
        else:
            oldalpha = alpha1
        
        # oldN = (2-oldalpha)/oldalpha
        oldN = (2.0 - oldalpha) / oldalpha
        
        # N = (((SC-FC)*(oldN-1))/(SC-1))+FC
        N = (((sc - fc) * (oldN - 1.0)) / (sc - 1.0)) + fc
        
        # alpha_ = 2/(N+1)
        alpha_ = 2.0 / (N + 1.0)
        
        # alpha = alpha_<2/(SC+1)?2/(SC+1):(alpha_>1?1:alpha_)
        min_alpha = 2.0 / (sc + 1.0)
        if alpha_ < min_alpha:
            final_alpha = min_alpha
        elif alpha_ > 1.0:
            final_alpha = 1.0
        else:
            final_alpha = alpha_
        
        alpha[i] = final_alpha
        
        # X_FAMAの計算
        if i == min_period:
            frama[i] = price[i]
            fast_fama[i] = price[i]
        else:
            frama[i] = (1.0 - final_alpha) * frama[i-1] + final_alpha * price[i]
            # Fast FAMAは2倍のアルファ値を使用（上限は1.0）
            fast_alpha = min(2.0 * final_alpha, 1.0)
            fast_fama[i] = (1.0 - fast_alpha) * fast_fama[i-1] + fast_alpha * price[i]
    
    # ゼロラグ処理の適用（オプション）
    if use_zero_lag:
        # X_FAMAにゼロラグ処理を適用
        frama_zero_lag = calculate_zero_lag_processing_fama(frama, alpha)
        fast_fama_zero_lag = calculate_zero_lag_processing_fama(fast_fama, alpha)
        
        # 有効な値のみを使用（NaN値は元の値を保持）
        for i in range(length):
            if not np.isnan(frama_zero_lag[i]):
                frama[i] = frama_zero_lag[i]
            
            if not np.isnan(fast_fama_zero_lag[i]):
                fast_fama[i] = fast_fama_zero_lag[i]
    
    return frama, fast_fama, dimension, alpha


class X_FAMA(Indicator):
    """
    X_FAMA (eXtended Fractal Adaptive Moving Average) インジケーター
    
    標準FRAMAインジケーターに以下の機能を追加した拡張版：
    - カルマンフィルターによる価格ソースの前処理（オプション）
    - ゼロラグ処理による応答性の向上（オプション）
    - 高速FAMA（アルファ値を2倍にした第二の線）の追加
    - より高度な適応性とノイズフィルタリング
    
    特徴:
    - フラクタル次元に基づく適応的な移動平均
    - トレンド強度に応じて応答速度を調整
    - 高速線と通常線の2本表示
    - ノイズフィルタリング機能
    - カルマンフィルターとゼロラグ処理の統合
    """
    
    def __init__(
        self,
        period: int = 16,                      # 期間（偶数である必要がある）
        src_type: str = 'hl2',                 # ソースタイプ
        fc: int = 1,                           # Fast Constant
        sc: int = 198,                         # Slow Constant
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
            period: 期間（偶数である必要がある、デフォルト: 16）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            fc: Fast Constant（デフォルト: 1）
            sc: Slow Constant（デフォルト: 198）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
        """
        # インジケーター名の作成
        indicator_name = f"X_FAMA(period={period}, {src_type}, fc={fc}, sc={sc}"
        if use_kalman_filter:
            indicator_name += f", kalman={kalman_filter_type}"
        if use_zero_lag:
            indicator_name += ", zero_lag=True"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        
        # パラメータの検証
        if period < 2:
            raise ValueError("期間は2以上である必要があります")
        if period % 2 != 0:
            raise ValueError("期間は偶数である必要があります")
        if fc < 1:
            raise ValueError("FC（Fast Constant）は1以上である必要があります")
        if sc < fc:
            raise ValueError("SC（Slow Constant）はFC以上である必要があります")
        if use_kalman_filter and kalman_process_noise <= 0:
            raise ValueError("kalman_process_noiseは0より大きい必要があります")
        
        # パラメータを保存
        self.period = period
        self.src_type = src_type.lower()
        self.fc = fc
        self.sc = sc
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
            params_sig = f"{self.period}_{self.src_type}_{self.fc}_{self.sc}_{kalman_sig}_{zero_lag_sig}"
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.period}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> XFAMAResult:
        """
        X_FAMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、OHLC + 選択したソースタイプに必要なカラムが必要
        
        Returns:
            XFAMAResult: X_FAMA値と計算中間値を含む結果
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
                return XFAMAResult(
                    frama_values=cached_result.frama_values.copy(),
                    fast_fama_values=cached_result.fast_fama_values.copy(),
                    fractal_dimension=cached_result.fractal_dimension.copy(),
                    alpha_values=cached_result.alpha_values.copy(),
                    filtered_price=cached_result.filtered_price.copy()
                )
            
            # 価格ソースの計算
            price_source = PriceSource.calculate_source(data, self.src_type)
            
            # 高値・安値データの取得（FRAMAのフラクタル次元計算に必要）
            if isinstance(data, pd.DataFrame):
                if 'high' not in data.columns or 'low' not in data.columns:
                    raise ValueError("DataFrameには'high'と'low'カラムが必要です")
                high = data['high'].to_numpy()
                low = data['low'].to_numpy()
            else:
                # NumPy配列の場合
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                high = data[:, 1]  # high
                low = data[:, 2]   # low
            
            # NumPy配列に変換（float64型で統一）
            price_source = np.asarray(price_source, dtype=np.float64)
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            
            # カルマンフィルターによる前処理（オプション）
            filtered_price = price_source.copy()
            if self.use_kalman_filter and self.kalman_filter is not None:
                try:
                    kalman_result = self.kalman_filter.calculate(data)
                    
                    # カルマンフィルター結果の値抽出
                    kalman_values = None
                    
                    if hasattr(kalman_result, 'values'):
                        kalman_values = kalman_result.values
                    elif hasattr(kalman_result, 'filtered_values'):
                        kalman_values = kalman_result.filtered_values
                    elif isinstance(kalman_result, (np.ndarray, list)):
                        kalman_values = kalman_result
                    else:
                        kalman_values = kalman_result
                    
                    if kalman_values is not None:
                        try:
                            kalman_values = np.asarray(kalman_values, dtype=np.float64)
                            
                            if kalman_values.ndim == 0:
                                raise ValueError("カルマンフィルター結果がスカラー値です")
                            elif kalman_values.ndim > 1:
                                kalman_values = kalman_values.flatten()
                            
                            if len(kalman_values) != len(price_source):
                                self.logger.warning(f"カルマンフィルター結果のサイズ不一致: {len(kalman_values)} != {len(price_source)}。元の価格を使用します。")
                                filtered_price = price_source.copy()
                            else:
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
            
            if data_length < self.period:
                self.logger.warning(f"データ長({data_length})が期間({self.period})より短いです")
            
            # X_FAMAの計算（Numba最適化関数を使用）
            frama_values, fast_fama_values, fractal_dim, alpha_values = calculate_x_frama_core(
                filtered_price, high, low, self.period, self.fc, self.sc, self.use_zero_lag
            )
            
            # 結果の保存（参照問題を避けるため必要な部分だけコピー）
            result = XFAMAResult(
                frama_values=frama_values.copy(),
                fast_fama_values=fast_fama_values.copy(),
                fractal_dimension=fractal_dim.copy(),
                alpha_values=alpha_values.copy(),
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
            
            self._values = frama_values  # 基底クラスの要件を満たすため（X_FRAMA値をメインとする）
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"X_FRAMA計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = XFAMAResult(
                frama_values=np.array([]),
                fast_fama_values=np.array([]),
                fractal_dimension=np.array([]),
                alpha_values=np.array([]),
                filtered_price=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """X_FRAMA値のみを取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            # 直近に使用されたキャッシュがない場合は最初のキャッシュを使用
            result = next(iter(self._result_cache.values()))
            
        return result.frama_values.copy()
    
    def get_frama_values(self) -> Optional[np.ndarray]:
        """
        X_FRAMA値を取得する
        
        Returns:
            np.ndarray: X_FRAMA値
        """
        return self.get_values()
    
    def get_fast_fama_values(self) -> Optional[np.ndarray]:
        """
        高速X_FAMA値を取得する
        
        Returns:
            np.ndarray: 高速X_FAMA値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.fast_fama_values.copy()
    
    def get_fractal_dimension(self) -> Optional[np.ndarray]:
        """
        フラクタル次元を取得する
        
        Returns:
            np.ndarray: フラクタル次元
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.fractal_dimension.copy()
    
    def get_alpha_values(self) -> Optional[np.ndarray]:
        """
        アルファ値を取得する
        
        Returns:
            np.ndarray: アルファ値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.alpha_values.copy()
    
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
            result = next(iter(self._result_cache.values()))
            
        return result.filtered_price.copy()
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'period': self.period,
            'src_type': self.src_type,
            'fc': self.fc,
            'sc': self.sc,
            'use_kalman_filter': self.use_kalman_filter,
            'kalman_filter_type': self.kalman_filter_type if self.use_kalman_filter else None,
            'kalman_process_noise': self.kalman_process_noise if self.use_kalman_filter else None,
            'kalman_observation_noise': self.kalman_observation_noise if self.use_kalman_filter else None,
            'use_zero_lag': self.use_zero_lag,
            'description': '拡張フラクタル適応移動平均線（カルマンフィルター・ゼロラグ処理対応）'
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
def calculate_x_fama(
    data: Union[pd.DataFrame, np.ndarray],
    period: int = 16,
    src_type: str = 'hl2',
    fc: int = 1,
    sc: int = 198,
    use_kalman_filter: bool = False,
    kalman_filter_type: str = 'unscented',
    kalman_process_noise: float = 0.01,
    kalman_observation_noise: float = 0.001,
    use_zero_lag: bool = True,
    **kwargs
) -> np.ndarray:
    """
    X_FAMAの計算（便利関数）
    
    Args:
        data: 価格データ
        period: 期間
        src_type: ソースタイプ
        fc: Fast Constant
        sc: Slow Constant
        use_kalman_filter: カルマンフィルターを使用するか
        kalman_filter_type: カルマンフィルタータイプ
        kalman_process_noise: プロセスノイズ
        kalman_observation_noise: 観測ノイズ
        use_zero_lag: ゼロラグ処理を使用するか
        **kwargs: その他のパラメータ
        
    Returns:
        X_FRAMA値
    """
    indicator = X_FAMA(
        period=period,
        src_type=src_type,
        fc=fc,
        sc=sc,
        use_kalman_filter=use_kalman_filter,
        kalman_filter_type=kalman_filter_type,
        kalman_process_noise=kalman_process_noise,
        kalman_observation_noise=kalman_observation_noise,
        use_zero_lag=use_zero_lag,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.frama_values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== X_FRAMA インジケーターのテスト ===")
    
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
    
    # 基本版X_FAMAをテスト
    print("\n基本版X_FAMAをテスト中...")
    x_fama_basic = X_FAMA(
        period=16,
        src_type='hl2',
        fc=1,
        sc=198,
        use_kalman_filter=False,
        use_zero_lag=False
    )
    try:
        result_basic = x_fama_basic.calculate(df)
        print(f"  X_FRAMA結果の型: {type(result_basic)}")
        print(f"  FRAMA配列の形状: {result_basic.frama_values.shape}")
        print(f"  Fast FAMA配列の形状: {result_basic.fast_fama_values.shape}")
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()
        result_basic = None
    
    if result_basic is not None:
        valid_count = np.sum(~np.isnan(result_basic.frama_values))
        mean_frama = np.nanmean(result_basic.frama_values)
        mean_fast_fama = np.nanmean(result_basic.fast_fama_values)
        
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  平均X_FRAMA: {mean_frama:.4f}")
        print(f"  平均Fast X_FRAMA: {mean_fast_fama:.4f}")
    else:
        print("  基本版X_FAMAの計算に失敗しました")
    
    # ゼロラグ処理版をテスト
    print("\nゼロラグ処理版X_FAMAをテスト中...")
    x_fama_zero_lag = X_FAMA(
        period=16,
        src_type='hl2',
        fc=1,
        sc=198,
        use_kalman_filter=False,
        use_zero_lag=True
    )
    try:
        result_zero_lag = x_fama_zero_lag.calculate(df)
        
        valid_count_zero_lag = np.sum(~np.isnan(result_zero_lag.frama_values))
        mean_frama_zero_lag = np.nanmean(result_zero_lag.frama_values)
        mean_fast_fama_zero_lag = np.nanmean(result_zero_lag.fast_fama_values)
        
        print(f"  有効値数: {valid_count_zero_lag}/{len(df)}")
        print(f"  平均X_FRAMA（ゼロラグ）: {mean_frama_zero_lag:.4f}")
        print(f"  平均Fast X_FRAMA（ゼロラグ）: {mean_fast_fama_zero_lag:.4f}")
        
        # 比較統計
        if result_basic is not None and valid_count > 0 and valid_count_zero_lag > 0:
            min_length = min(valid_count, valid_count_zero_lag)
            correlation = np.corrcoef(
                result_basic.frama_values[~np.isnan(result_basic.frama_values)][-min_length:],
                result_zero_lag.frama_values[~np.isnan(result_zero_lag.frama_values)][-min_length:]
            )[0, 1]
            print(f"  基本版とゼロラグ版の相関: {correlation:.4f}")
    except Exception as e:
        print(f"  ゼロラグ処理版でエラー: {e}")
        import traceback
        traceback.print_exc()
    
    # カルマンフィルター版をテスト（利用可能な場合）
    if UNIFIED_KALMAN_AVAILABLE:
        print("\nカルマンフィルター機能は利用できますが、テストではスキップします")
    else:
        print("\nカルマンフィルター機能は利用できません")
    
    print("\n=== テスト完了 ===")