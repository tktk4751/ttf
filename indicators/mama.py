#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, prange, vectorize, njit, float64, types
import traceback
import math

from .indicator import Indicator
from .price_source import PriceSource


@dataclass
class MAMAResult:
    """MAMA/FAMAの計算結果"""
    mama_values: np.ndarray      # MAMAライン値
    fama_values: np.ndarray      # FAMAライン値
    period_values: np.ndarray    # 計算されたPeriod値
    alpha_values: np.ndarray     # 計算されたAlpha値
    phase_values: np.ndarray     # Phase値
    i1_values: np.ndarray        # InPhase component
    q1_values: np.ndarray        # Quadrature component


@njit(fastmath=True, cache=True)
def calculate_mama_fama(
    price: np.ndarray,
    fast_limit: float = 0.5,
    slow_limit: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MAMA/FAMAを計算する（Numba最適化版）
    
    Args:
        price: 価格配列（通常は(H+L)/2）
        fast_limit: 速いリミット（デフォルト: 0.5）
        slow_limit: 遅いリミット（デフォルト: 0.05）
    
    Returns:
        Tuple[np.ndarray, ...]: MAMA値, FAMA値, Period値, Alpha値, Phase値, I1値, Q1値
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
        
        # MAMA計算
        if i > 5 and not np.isnan(mama[i-1]) and not np.isnan(alpha[i]):
            mama[i] = alpha[i] * price[i] + (1.0 - alpha[i]) * mama[i-1]
        else:
            mama[i] = price[i]  # 初期値として価格を使用
        
        # FAMA計算
        if i > 5 and not np.isnan(fama[i-1]) and not np.isnan(mama[i]) and not np.isnan(alpha[i]):
            fama[i] = 0.5 * alpha[i] * mama[i] + (1.0 - 0.5 * alpha[i]) * fama[i-1]
        else:
            fama[i] = mama[i]  # 初期値としてMAMA値を使用
    
    return mama, fama, period, alpha, phase, i1, q1


class MAMA(Indicator):
    """
    MAMA (Mother of Adaptive Moving Average) / FAMA (Following Adaptive Moving Average) インジケーター
    
    適応型移動平均線で、市場のサイクルに応じて自動的に期間を調整します：
    - MAMA: 主要な適応型移動平均線
    - FAMA: MAMAをフォローする適応型移動平均線
    - Ehlers's MESA (Maximum Entropy Spectrum Analysis) アルゴリズムベース
    
    特徴:
    - 市場サイクルの変化に適応
    - トレンド強度に応じて応答速度を調整
    - ノイズフィルタリング機能
    """
    
    def __init__(
        self,
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.05,              # 低速制限値
        src_type: str = 'ukf_hlc3',            # ソースタイプ
        ukf_params: Optional[Dict] = None      # UKFパラメータ（UKFソース使用時）
    ):
        """
        コンストラクタ
        
        Args:
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
                UKFソース: 'ukf', 'ukf_close', 'ukf_hlc3', 'ukf_hl2', 'ukf_ohlc4'
                - 'ukf_hlc3': UKFフィルター適用のHLC3（デフォルト、推奨）
                - 'hl2': (高値 + 安値) / 2
                - 'close': 終値
                - 'hlc3': (高値 + 安値 + 終値) / 3
                - 'ohlc4': (始値 + 高値 + 安値 + 終値) / 4
            ukf_params: UKFパラメータ（UKFソース使用時のオプション）
                alpha: UKFのalpha値（デフォルト: 0.001）
                beta: UKFのbeta値（デフォルト: 2.0）
                kappa: UKFのkappa値（デフォルト: 0.0）
                process_noise_scale: プロセスノイズスケール（デフォルト: 0.001）
                volatility_window: ボラティリティ計算ウィンドウ（デフォルト: 10）
                adaptive_noise: 適応ノイズの使用（デフォルト: True）
        """
        # インジケーター名の作成
        indicator_name = f"MAMA(fast={fast_limit}, slow={slow_limit}, {src_type})"
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        self.src_type = src_type.lower()
        self.ukf_params = ukf_params
        
        # ソースタイプの検証（PriceSourceから利用可能なタイプを取得）
        available_sources = PriceSource.get_available_sources()
        if self.src_type not in available_sources:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(available_sources.keys())}")
        
        # パラメータ検証
        if fast_limit <= 0 or fast_limit > 1:
            raise ValueError("fast_limitは0より大きく1以下である必要があります")
        if slow_limit <= 0 or slow_limit > 1:
            raise ValueError("slow_limitは0より大きく1以下である必要があります")
        if slow_limit >= fast_limit:
            raise ValueError("slow_limitはfast_limitより小さい必要があります")
        
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
            ukf_sig = str(self.ukf_params) if self.ukf_params else "None"
            params_sig = f"{self.fast_limit}_{self.slow_limit}_{self.src_type}_{ukf_sig}"
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.fast_limit}_{self.slow_limit}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> MAMAResult:
        """
        MAMA/FAMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、OHLC + 選択したソースタイプに必要なカラムが必要
        
        Returns:
            MAMAResult: MAMA/FAMAの値と計算中間値を含む結果
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
                return MAMAResult(
                    mama_values=cached_result.mama_values.copy(),
                    fama_values=cached_result.fama_values.copy(),
                    period_values=cached_result.period_values.copy(),
                    alpha_values=cached_result.alpha_values.copy(),
                    phase_values=cached_result.phase_values.copy(),
                    i1_values=cached_result.i1_values.copy(),
                    q1_values=cached_result.q1_values.copy()
                )
            
            # 価格ソースの計算
            price_source = PriceSource.calculate_source(data, self.src_type, self.ukf_params)
            
            # NumPy配列に変換（float64型で統一）
            price_source = np.asarray(price_source, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price_source)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < 10:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低10点以上を推奨します。")
            
            # MAMA/FAMAの計算（Numba最適化関数を使用）
            mama_values, fama_values, period_values, alpha_values, phase_values, i1_values, q1_values = calculate_mama_fama(
                price_source, self.fast_limit, self.slow_limit
            )
            
            # 結果の保存（参照問題を避けるため必要な部分だけコピー）
            result = MAMAResult(
                mama_values=mama_values.copy(),
                fama_values=fama_values.copy(),
                period_values=period_values.copy(),
                alpha_values=alpha_values.copy(),
                phase_values=phase_values.copy(),
                i1_values=i1_values.copy(),
                q1_values=q1_values.copy()
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
            
            self._values = mama_values  # 基底クラスの要件を満たすため（MAMA値をメインとする）
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"MAMA/FAMA計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = MAMAResult(
                mama_values=np.array([]),
                fama_values=np.array([]),
                period_values=np.array([]),
                alpha_values=np.array([]),
                phase_values=np.array([]),
                i1_values=np.array([]),
                q1_values=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """MAMA値のみを取得する（後方互換性のため）"""
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
        MAMA値を取得する
        
        Returns:
            np.ndarray: MAMA値
        """
        return self.get_values()
    
    def get_fama_values(self) -> Optional[np.ndarray]:
        """
        FAMA値を取得する
        
        Returns:
            np.ndarray: FAMA値
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
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result_cache = {}
        self._cache_keys = [] 