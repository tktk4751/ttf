#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import njit
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
        # 絶対インポートを試行
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
class MESAFRAMAResult:
    """MESA_FRAMAの計算結果"""
    values: np.ndarray               # MESA_FRAMA値
    fractal_dimension: np.ndarray    # フラクタル次元
    alpha: np.ndarray               # アルファ値
    dynamic_periods: np.ndarray     # MESA適応期間
    mesa_phase: np.ndarray          # MESA位相値
    mesa_alpha: np.ndarray          # MESA内部アルファ値
    filtered_price: np.ndarray      # カルマンフィルター後の価格（使用した場合）


@njit(fastmath=True, cache=True)
def calculate_mesa_period_detection(
    price: np.ndarray,
    fast_limit: float = 0.5,
    slow_limit: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MESA期間検出アルゴリズム（MAMAアルゴリズムから期間計算部分を抽出）
    
    Args:
        price: 価格配列
        fast_limit: 高速制限値（デフォルト: 0.5）
        slow_limit: 低速制限値（デフォルト: 0.05）
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 適応期間, Phase値, Alpha値
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
    phase = np.zeros(length, dtype=np.float64)
    delta_phase = np.zeros(length, dtype=np.float64)
    alpha = np.zeros(length, dtype=np.float64)
    
    # 初期値設定
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
        phase[i] = 0.0
        delta_phase[i] = 1.0
        alpha[i] = 0.05  # slow_limitで初期化
    
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
        
        # Phasor加算
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
        
        # Alpha計算
        if delta_phase[i] > 0:
            alpha[i] = fast_limit / delta_phase[i]
            if alpha[i] < slow_limit:
                alpha[i] = slow_limit
            elif alpha[i] > fast_limit:
                alpha[i] = fast_limit
        else:
            alpha[i] = slow_limit
    
    return period, phase, alpha


@njit(fastmath=True, cache=True)
def calculate_zero_lag_processing(frama_values: np.ndarray, alpha_values: np.ndarray) -> np.ndarray:
    """
    MESA_FRAMA値に対してゼロラグ処理を適用する（Numba最適化版）
    
    Args:
        frama_values: MESA_FRAMA値の配列
        alpha_values: MESA_FRAMA内部のアルファ値の配列
    
    Returns:
        ゼロラグ処理後の値配列
    """
    length = len(frama_values)
    result = np.full(length, np.nan, dtype=np.float64)
    
    if length == 0:
        return result
    
    # EMA値の配列（MESA_FRAMA値のEMA）
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
    
    # EMAの計算（MESA_FRAMA値のEMA、MESA_FRAMAのアルファ値を使用）
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
    
    # 以降はラグ除去データのEMAを計算（MESA_FRAMAのアルファ値を使用）
    for i in range(start_idx + 1, length):
        if not np.isnan(lag_reduced_data[i]) and not np.isnan(alpha_values[i]):
            if not np.isnan(result[i-1]):
                result[i] = alpha_values[i] * lag_reduced_data[i] + (1.0 - alpha_values[i]) * result[i-1]
            else:
                result[i] = lag_reduced_data[i]
    
    return result


@njit(fastmath=True, cache=True)
def calculate_mesa_frama_core(
    price: np.ndarray, 
    high: np.ndarray, 
    low: np.ndarray, 
    base_period: int, 
    fc: int, 
    sc: int,
    mesa_fast_limit: float = 0.5,
    mesa_slow_limit: float = 0.05,
    use_zero_lag: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MESA適応期間を使用したFRAMAを計算する（Numba最適化版）
    
    Args:
        price: 価格データ
        high: 高値データ
        low: 安値データ
        base_period: 基本期間
        fc: Fast Constant
        sc: Slow Constant
        mesa_fast_limit: MESA高速制限値
        mesa_slow_limit: MESA低速制限値
        use_zero_lag: ゼロラグ処理を使用するか
    
    Returns:
        Tuple[np.ndarray, ...]: MESA_FRAMA値, フラクタル次元, アルファ値, 動的期間, MESA位相, MESAアルファ
    """
    length = len(price)
    
    # MESA期間検出の実行
    dynamic_periods, mesa_phase, mesa_alpha = calculate_mesa_period_detection(
        price, mesa_fast_limit, mesa_slow_limit
    )
    
    # 結果配列を初期化
    frama = np.zeros(length, dtype=np.float64)
    dimension = np.zeros(length, dtype=np.float64)
    alpha = np.zeros(length, dtype=np.float64)
    
    # 初期値設定
    for i in range(length):
        frama[i] = np.nan
        dimension[i] = np.nan
        alpha[i] = np.nan
    
    # w = log(2/(SC+1))
    w = np.log(2.0 / (sc + 1))
    
    for i in range(length):
        # 動的期間の決定（MESA期間を使用）
        current_period = base_period
        if i < len(dynamic_periods) and not np.isnan(dynamic_periods[i]):
            # MESA期間を基本期間にマッピング（偶数に調整）
            current_period = max(4, int(dynamic_periods[i]))
            if current_period % 2 != 0:
                current_period += 1  # 奇数の場合は偶数にする
        
        len1 = current_period // 2
        min_period = max(current_period - 1, 0)
        
        if i < min_period:
            if not np.isnan(price[i]):
                frama[i] = price[i]
                alpha[i] = 1.0
            continue
        
        if np.isnan(price[i]):
            frama[i] = frama[i-1] if i > 0 else np.nan
            continue
        
        # N1 = (highest(high,len1)-lowest(low,len1))/len1
        h1 = -np.inf
        l1 = np.inf
        for j in range(len1):
            if i - j >= 0:
                if high[i - j] > h1:
                    h1 = high[i - j]
                if low[i - j] < l1:
                    l1 = low[i - j]
        
        n1 = (h1 - l1) / len1
        
        # N2 = (highest(high,len2)[len1]-lowest(low,len2)[len1])/len1
        h2 = -np.inf
        l2 = np.inf
        for j in range(len1, current_period):
            if i - j >= 0:
                if high[i - j] > h2:
                    h2 = high[i - j]
                if low[i - j] < l2:
                    l2 = low[i - j]
        
        n2 = (h2 - l2) / len1
        
        # N3 = (highest(high,len)-lowest(low,len))/len
        h3 = -np.inf
        l3 = np.inf
        for j in range(current_period):
            if i - j >= 0:
                if high[i - j] > h3:
                    h3 = high[i - j]
                if low[i - j] < l3:
                    l3 = low[i - j]
        
        n3 = (h3 - l3) / current_period
        
        # フラクタル次元の計算
        if n1 > 0 and n2 > 0 and n3 > 0:
            dimen1 = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
            dimen = dimen1
        else:
            dimen = dimension[i-1] if i > 0 else 1.0
        
        dimension[i] = dimen
        
        # アルファ計算
        alpha1 = np.exp(w * (dimen - 1.0))
        
        if alpha1 > 1.0:
            oldalpha = 1.0
        elif alpha1 < 0.01:
            oldalpha = 0.01
        else:
            oldalpha = alpha1
        
        oldN = (2.0 - oldalpha) / oldalpha
        N = (((sc - fc) * (oldN - 1.0)) / (sc - 1.0)) + fc
        alpha_ = 2.0 / (N + 1.0)
        
        min_alpha = 2.0 / (sc + 1.0)
        if alpha_ < min_alpha:
            final_alpha = min_alpha
        elif alpha_ > 1.0:
            final_alpha = 1.0
        else:
            final_alpha = alpha_
        
        alpha[i] = final_alpha
        
        # MESA_FRAMA計算
        if i == min_period:
            frama[i] = price[i]
        else:
            frama[i] = (1.0 - final_alpha) * frama[i-1] + final_alpha * price[i]
    
    # ゼロラグ処理の適用（オプション）
    if use_zero_lag:
        # MESA_FRAMAにゼロラグ処理を適用（フラクタルアルファ値を使用）
        frama_zero_lag = calculate_zero_lag_processing(frama, alpha)
        
        # 有効な値のみを使用（NaN値は元の値を保持）
        for i in range(length):
            if not np.isnan(frama_zero_lag[i]):
                frama[i] = frama_zero_lag[i]
    
    return frama, dimension, alpha, dynamic_periods, mesa_phase, mesa_alpha


class MESA_FRAMA(Indicator):
    """
    MESA_FRAMA - MESA適応期間を使用したFractal Adaptive Moving Average
    
    MAMAアルゴリズムの期間決定メカニズムを使用してFRAMAの期間を動的に適応：
    - MAMAのMESA期間検出アルゴリズムを使用
    - フラクタル次元ベースのアルファ値計算（FRAMAの特徴）
    - 動的期間による適応性の向上
    - カルマンフィルターとゼロラグ処理の統合（オプション）
    
    特徴:
    - 市場サイクルに応じた動的期間調整
    - フラクタル次元による応答性制御
    - トレンド時とレンジ時の適応動作
    - ノイズフィルタリング機能
    """
    
    def __init__(
        self,
        base_period: int = 16,                 # 基本期間（偶数である必要がある）
        src_type: str = 'hl2',                # ソースタイプ
        fc: int = 1,                          # Fast Constant
        sc: int = 198,                        # Slow Constant
        # MESA期間検出パラメータ
        mesa_fast_limit: float = 0.5,         # MESA高速制限値
        mesa_slow_limit: float = 0.05,        # MESA低速制限値
        # カルマンフィルターパラメータ
        use_kalman_filter: bool = False,      # カルマンフィルターを使用するか
        kalman_filter_type: str = 'unscented', # カルマンフィルタータイプ
        kalman_process_noise: float = 0.01,   # プロセスノイズ
        kalman_observation_noise: float = 0.001, # 観測ノイズ
        # ゼロラグ処理パラメータ
        use_zero_lag: bool = True             # ゼロラグ処理を使用するか
    ):
        """
        コンストラクタ
        
        Args:
            base_period: 基本期間（偶数である必要がある、デフォルト: 16）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            fc: Fast Constant（デフォルト: 1）
            sc: Slow Constant（デフォルト: 198）
            mesa_fast_limit: MESA高速制限値（デフォルト: 0.5）
            mesa_slow_limit: MESA低速制限値（デフォルト: 0.05）
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ（デフォルト: 'unscented'）
            kalman_process_noise: プロセスノイズ（デフォルト: 0.01）
            kalman_observation_noise: 観測ノイズ（デフォルト: 0.001）
            use_zero_lag: ゼロラグ処理を使用するか（デフォルト: True）
        """
        # インジケーター名の作成
        indicator_name = f"MESA_FRAMA(base={base_period}, src={src_type}, fc={fc}, sc={sc}"
        indicator_name += f", mesa_fast={mesa_fast_limit}, mesa_slow={mesa_slow_limit}"
        if use_kalman_filter:
            indicator_name += f", kalman={kalman_filter_type}"
        if use_zero_lag:
            indicator_name += ", zero_lag=True"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        
        # パラメータの検証
        if base_period < 2:
            raise ValueError("基本期間は2以上である必要があります")
        if base_period % 2 != 0:
            raise ValueError("基本期間は偶数である必要があります")
        if fc < 1:
            raise ValueError("FC（Fast Constant）は1以上である必要があります")
        if sc < fc:
            raise ValueError("SC（Slow Constant）はFC以上である必要があります")
        if mesa_fast_limit <= 0 or mesa_fast_limit > 1:
            raise ValueError("mesa_fast_limitは0より大きく1以下である必要があります")
        if mesa_slow_limit <= 0 or mesa_slow_limit > 1:
            raise ValueError("mesa_slow_limitは0より大きく1以下である必要があります")
        if mesa_slow_limit >= mesa_fast_limit:
            raise ValueError("mesa_slow_limitはmesa_fast_limitより小さい必要があります")
        
        # パラメータを保存
        self.base_period = base_period
        self.src_type = src_type.lower()
        self.fc = fc
        self.sc = sc
        self.mesa_fast_limit = mesa_fast_limit
        self.mesa_slow_limit = mesa_slow_limit
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
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
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
            kalman_sig = f"{self.kalman_filter_type}_{self.kalman_process_noise}" if self.use_kalman_filter else "None"
            params_sig = f"{self.base_period}_{self.src_type}_{self.fc}_{self.sc}"
            params_sig += f"_{self.mesa_fast_limit}_{self.mesa_slow_limit}_{kalman_sig}_{self.use_zero_lag}"
            
            # ハッシュ計算
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.base_period}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> MESAFRAMAResult:
        """
        MESA_FRAMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            MESAFRAMAResult: MESA_FRAMA値、フラクタル次元、アルファ値等を含む結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # キャッシュキーの順序を更新
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return MESAFRAMAResult(
                    values=cached_result.values.copy(),
                    fractal_dimension=cached_result.fractal_dimension.copy(),
                    alpha=cached_result.alpha.copy(),
                    dynamic_periods=cached_result.dynamic_periods.copy(),
                    mesa_phase=cached_result.mesa_phase.copy(),
                    mesa_alpha=cached_result.mesa_alpha.copy(),
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
                        kalman_values = np.asarray(kalman_values, dtype=np.float64)
                        
                        if kalman_values.ndim > 1:
                            kalman_values = kalman_values.flatten()
                        
                        if len(kalman_values) == len(price_source):
                            # NaN値の処理
                            nan_mask = np.isnan(kalman_values)
                            if np.any(nan_mask):
                                kalman_values[nan_mask] = price_source[nan_mask]
                            
                            filtered_price = kalman_values
                            self.logger.debug("カルマンフィルターによる価格前処理を適用しました")
                        else:
                            self.logger.warning(f"カルマンフィルター結果のサイズ不一致。元の価格を使用します。")
                            filtered_price = price_source.copy()
                    else:
                        self.logger.warning("カルマンフィルター結果がNoneです。元の価格を使用します。")
                        filtered_price = price_source.copy()
                        
                except Exception as e:
                    self.logger.warning(f"カルマンフィルター適用中にエラー: {e}。元の価格を使用します。")
                    filtered_price = price_source.copy()
            
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
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            
            # データ長の検証
            data_length = len(filtered_price)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < self.base_period:
                self.logger.warning(f"データ長({data_length})が基本期間({self.base_period})より短いです")
            
            # MESA_FRAMAの計算
            frama_values, fractal_dim, alpha_values, dynamic_periods, mesa_phase, mesa_alpha = calculate_mesa_frama_core(
                filtered_price, high, low, self.base_period, self.fc, self.sc,
                self.mesa_fast_limit, self.mesa_slow_limit, self.use_zero_lag
            )
            
            # 結果の保存
            result = MESAFRAMAResult(
                values=frama_values.copy(),
                fractal_dimension=fractal_dim.copy(),
                alpha=alpha_values.copy(),
                dynamic_periods=dynamic_periods.copy(),
                mesa_phase=mesa_phase.copy(),
                mesa_alpha=mesa_alpha.copy(),
                filtered_price=filtered_price.copy()
            )
            
            # キャッシュを更新
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = frama_values  # 基底クラスの要件を満たすため
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"MESA_FRAMA計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = MESAFRAMAResult(
                values=np.array([]),
                fractal_dimension=np.array([]),
                alpha=np.array([]),
                dynamic_periods=np.array([]),
                mesa_phase=np.array([]),
                mesa_alpha=np.array([]),
                filtered_price=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """MESA_FRAMA値のみを取得する"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.values.copy()
    
    def get_fractal_dimension(self) -> Optional[np.ndarray]:
        """
        フラクタル次元を取得する
        
        Returns:
            np.ndarray: フラクタル次元の値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.fractal_dimension.copy()
    
    def get_alpha(self) -> Optional[np.ndarray]:
        """
        フラクタルアルファ値を取得する
        
        Returns:
            np.ndarray: フラクタルアルファ値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.alpha.copy()
    
    def get_dynamic_periods(self) -> Optional[np.ndarray]:
        """
        MESA適応期間を取得する
        
        Returns:
            np.ndarray: MESA適応期間
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.dynamic_periods.copy()
    
    def get_mesa_phase(self) -> Optional[np.ndarray]:
        """
        MESA位相値を取得する
        
        Returns:
            np.ndarray: MESA位相値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.mesa_phase.copy()
    
    def get_mesa_alpha(self) -> Optional[np.ndarray]:
        """
        MESAアルファ値を取得する
        
        Returns:
            np.ndarray: MESAアルファ値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.mesa_alpha.copy()
    
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
def calculate_mesa_frama(
    data: Union[pd.DataFrame, np.ndarray],
    base_period: int = 16,
    src_type: str = 'hl2',
    fc: int = 1,
    sc: int = 198,
    mesa_fast_limit: float = 0.5,
    mesa_slow_limit: float = 0.05,
    use_kalman_filter: bool = False,
    kalman_filter_type: str = 'unscented',
    kalman_process_noise: float = 0.01,
    kalman_observation_noise: float = 0.001,
    use_zero_lag: bool = True,
    **kwargs
) -> np.ndarray:
    """
    MESA_FRAMAの計算（便利関数）
    
    Args:
        data: 価格データ
        base_period: 基本期間
        src_type: ソースタイプ
        fc: Fast Constant
        sc: Slow Constant
        mesa_fast_limit: MESA高速制限値
        mesa_slow_limit: MESA低速制限値
        use_kalman_filter: カルマンフィルターを使用するか
        kalman_filter_type: カルマンフィルタータイプ
        kalman_process_noise: プロセスノイズ
        kalman_observation_noise: 観測ノイズ
        use_zero_lag: ゼロラグ処理を使用するか
        **kwargs: その他のパラメータ
        
    Returns:
        MESA_FRAMA値
    """
    indicator = MESA_FRAMA(
        base_period=base_period,
        src_type=src_type,
        fc=fc,
        sc=sc,
        mesa_fast_limit=mesa_fast_limit,
        mesa_slow_limit=mesa_slow_limit,
        use_kalman_filter=use_kalman_filter,
        kalman_filter_type=kalman_filter_type,
        kalman_process_noise=kalman_process_noise,
        kalman_observation_noise=kalman_observation_noise,
        use_zero_lag=use_zero_lag,
        **kwargs
    )
    result = indicator.calculate(data)
    return result.values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== MESA_FRAMA インジケーターのテスト ===")
    
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
    
    # 基本版MESA_FRAMAをテスト
    print("\n基本版MESA_FRAMAをテスト中...")
    mesa_frama_basic = MESA_FRAMA(
        base_period=16,
        src_type='hl2',
        fc=1,
        sc=198,
        mesa_fast_limit=0.5,
        mesa_slow_limit=0.05,
        use_kalman_filter=False,
        use_zero_lag=False
    )
    try:
        result_basic = mesa_frama_basic.calculate(df)
        print(f"  MESA_FRAMA結果の型: {type(result_basic)}")
        print(f"  値配列の形状: {result_basic.values.shape}")
        print(f"  動的期間配列の形状: {result_basic.dynamic_periods.shape}")
        
        valid_count = np.sum(~np.isnan(result_basic.values))
        mean_value = np.nanmean(result_basic.values)
        mean_period = np.nanmean(result_basic.dynamic_periods)
        
        print(f"  有効値数: {valid_count}/{len(df)}")
        print(f"  平均MESA_FRAMA: {mean_value:.4f}")
        print(f"  平均動的期間: {mean_period:.2f}")
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()
        result_basic = None
    
    # ゼロラグ処理版をテスト
    print("\nゼロラグ処理版MESA_FRAMAをテスト中...")
    mesa_frama_zero_lag = MESA_FRAMA(
        base_period=16,
        src_type='hl2',
        fc=1,
        sc=198,
        mesa_fast_limit=0.5,
        mesa_slow_limit=0.05,
        use_kalman_filter=False,
        use_zero_lag=True
    )
    try:
        result_zero_lag = mesa_frama_zero_lag.calculate(df)
        
        valid_count_zero_lag = np.sum(~np.isnan(result_zero_lag.values))
        mean_value_zero_lag = np.nanmean(result_zero_lag.values)
        
        print(f"  有効値数: {valid_count_zero_lag}/{len(df)}")
        print(f"  平均MESA_FRAMA（ゼロラグ）: {mean_value_zero_lag:.4f}")
        
        # 比較統計
        if result_basic is not None and valid_count > 0 and valid_count_zero_lag > 0:
            min_length = min(valid_count, valid_count_zero_lag)
            correlation = np.corrcoef(
                result_basic.values[~np.isnan(result_basic.values)][-min_length:],
                result_zero_lag.values[~np.isnan(result_zero_lag.values)][-min_length:]
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