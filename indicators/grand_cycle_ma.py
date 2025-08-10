#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from numba import njit
import traceback
import math

from .indicator import Indicator
from .price_source import PriceSource
try:
    from .cycle.ehlers_unified_dc import EhlersUnifiedDC
except ImportError:
    try:
        from indicators.cycle.ehlers_unified_dc import EhlersUnifiedDC
    except ImportError:
        import sys
        import os
        cycle_path = os.path.join(os.path.dirname(__file__), 'cycle')
        sys.path.insert(0, cycle_path)
        from .cycle.ehlers_unified_dc import EhlersUnifiedDC

try:
    from .smoother.unified_smoother import UnifiedSmoother
except ImportError:
    try:
        from indicators.smoother.unified_smoother import UnifiedSmoother
    except ImportError:
        import sys
        import os
        smoother_path = os.path.join(os.path.dirname(__file__), 'smoother')
        sys.path.insert(0, smoother_path)
        from .smoother.unified_smoother import UnifiedSmoother

try:
    from .kalman.unified_kalman import UnifiedKalman
except ImportError:
    try:
        from indicators.kalman.unified_kalman import UnifiedKalman
    except ImportError:
        import sys
        import os
        kalman_path = os.path.join(os.path.dirname(__file__), 'kalman')
        sys.path.insert(0, kalman_path)
        from .kalman.unified_kalman import UnifiedKalman


@dataclass
class GrandCycleMAResult:
    """グランドサイクルMAの計算結果"""
    grand_mama_values: np.ndarray     # グランドサイクルMAMA値
    grand_fama_values: np.ndarray     # グランドサイクルFAMA値
    cycle_period: np.ndarray          # サイクル周期
    alpha_values: np.ndarray          # アルファ値
    phase_values: np.ndarray          # 位相値
    detector_values: np.ndarray       # サイクル検出器出力


@njit(fastmath=True, cache=True)
def calculate_grand_cycle_ma_core(
    price: np.ndarray,
    cycle_period: np.ndarray,
    fast_limit: float = 0.5,
    slow_limit: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    グランドサイクルMAを計算する（Numba最適化版）
    
    注意: 価格のスムージングは事前に外部で実行済みであることを前提とする
    
    Args:
        price: 事前にスムージング済みの価格配列
        cycle_period: サイクル周期配列
        fast_limit: 速いリミット（デフォルト: 0.5）
        slow_limit: 遅いリミット（デフォルト: 0.05）
    
    Returns:
        Tuple[np.ndarray, ...]: グランドMAMA値, グランドFAMA値, Alpha値, Phase値
    """
    length = len(price)
    
    # 変数の初期化
    phase = np.zeros(length, dtype=np.float64)
    delta_phase = np.zeros(length, dtype=np.float64)
    alpha = np.zeros(length, dtype=np.float64)
    grand_mama = np.zeros(length, dtype=np.float64)
    grand_fama = np.zeros(length, dtype=np.float64)
    
    # 初期値設定
    for i in range(min(5, length)):
        phase[i] = 0.0
        delta_phase[i] = 1.0
        alpha[i] = slow_limit
        grand_mama[i] = price[i] if i < length else 100.0
        grand_fama[i] = price[i] if i < length else 100.0
    
    # メインループ
    for i in range(5, length):
        # サイクル周期を使った位相計算
        current_period = cycle_period[i] if not np.isnan(cycle_period[i]) and cycle_period[i] > 0 else 20.0
        
        # 位相の計算（簡略化されたHilbert変換）
        if i >= 4:
            phase[i] = math.atan2(
                price[i] - price[i-4],
                price[i-2]
            ) * 180.0 / math.pi
        else:
            phase[i] = phase[i-1] if i > 0 else 0.0
        
        # DeltaPhase計算
        if i > 0:
            delta_phase[i] = abs(phase[i-1] - phase[i])
            if delta_phase[i] < 1.0:
                delta_phase[i] = 1.0
        else:
            delta_phase[i] = 1.0
        
        # サイクル周期に基づいたAlpha調整
        # より長いサイクル周期の場合はより低速に、短い場合は高速に
        cycle_factor = 20.0 / current_period if current_period > 0 else 1.0
        adjusted_fast_limit = fast_limit * cycle_factor
        adjusted_fast_limit = min(adjusted_fast_limit, 1.0)
        adjusted_fast_limit = max(adjusted_fast_limit, slow_limit)
        
        # Alpha計算
        if delta_phase[i] > 0:
            alpha[i] = adjusted_fast_limit / delta_phase[i]
            if alpha[i] < slow_limit:
                alpha[i] = slow_limit
            elif alpha[i] > adjusted_fast_limit:
                alpha[i] = adjusted_fast_limit
        else:
            alpha[i] = slow_limit
        
        # グランドサイクルMAMA計算
        if i > 0 and not np.isnan(grand_mama[i-1]) and not np.isnan(alpha[i]):
            grand_mama[i] = alpha[i] * price[i] + (1.0 - alpha[i]) * grand_mama[i-1]
        else:
            grand_mama[i] = price[i]
        
        # グランドサイクルFAMA計算
        if i > 0 and not np.isnan(grand_fama[i-1]) and not np.isnan(grand_mama[i]) and not np.isnan(alpha[i]):
            grand_fama[i] = 0.5 * alpha[i] * grand_mama[i] + (1.0 - 0.5 * alpha[i]) * grand_fama[i-1]
        else:
            grand_fama[i] = grand_mama[i]
    
    return grand_mama, grand_fama, alpha, phase


class GrandCycleMA(Indicator):
    """
    グランドサイクルMA - 様々なサイクル検出器を使用できる適応型移動平均線
    
    MAMAの計算ロジックをベースに、EhlersUnifiedDCのサイクル検出器を使用して
    市場のサイクルに適応する移動平均線を実装します。
    
    特徴:
    - 15種類以上のサイクル検出器から選択可能
    - サイクル周期に基づいた適応的なアルファ値調整
    - MAMAとFAMAの両方を提供
    - ノイズフィルタリング機能
    """
    
    def __init__(
        self,
        detector_type: str = 'hody',           # サイクル検出器のタイプ
        fast_limit: float = 0.5,               # 高速制限値
        slow_limit: float = 0.05,              # 低速制限値
        src_type: str = 'hlc3',                # ソースタイプ
        cycle_part: float = 0.5,               # サイクル部分
        max_cycle: int = 50,                   # 最大サイクル期間
        min_cycle: int = 6,                    # 最小サイクル期間
        max_output: int = 34,                  # 最大出力値
        min_output: int = 1,                   # 最小出力値
        # 新しいフィルタリング・スムージングパラメータ
        use_kalman_filter: bool = False,       # カルマンフィルターの使用
        kalman_filter_type: str = 'adaptive',  # カルマンフィルタータイプ
        kalman_params: Optional[Dict] = None,  # カルマンフィルターパラメータ
        use_smoother: bool = True,             # スムーサーの使用
        smoother_type: str = 'ultimate_smoother',          # スムーサータイプ
        smoother_params: Optional[Dict] = 10, # スムーサーパラメータ
        # サイクル検出器固有のパラメータ
        alpha: float = 0.07,
        bandwidth: float = 0.6,
        center_period: float = 15.0,
        avg_length: float = 3.0,
        window: int = 50,
        period_range: Tuple[int, int] = (5, 124),
        use_kalman_filter_legacy: bool = False,  # 旧式パラメータ（後方互換性）
        kalman_measurement_noise: float = 1.0,
        kalman_process_noise: float = 0.01,
        kalman_n_states: int = 5
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: 使用するサイクル検出器のタイプ
                基本検出器:
                - 'hody': ホモダイン判別機
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積
                高度な検出器:
                - 'cycle_period': サイクル周期検出器
                - 'cycle_period2': 改良サイクル周期検出器
                - 'bandpass_zero': バンドパスゼロクロッシング
                - 'autocorr_perio': 自己相関ピリオドグラム
                - 'dft_dominant': DFTドミナントサイクル
                - 'multi_bandpass': 複数バンドパス
                - 'absolute_ultimate': 絶対的究極サイクル
                - 'ultra_supreme_stability': 究極安定性サイクル
                - 'refined': 洗練されたサイクル検出器
                - 'practical': 実践的サイクル検出器
            fast_limit: 高速制限値（デフォルト: 0.5）
            slow_limit: 低速制限値（デフォルト: 0.05）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4')
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 50）
            min_cycle: 最小サイクル期間（デフォルト: 6）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 1）
            
            # 新しいフィルタリング・スムージングパラメータ
            use_kalman_filter: カルマンフィルターを使用するか（デフォルト: False）
            kalman_filter_type: カルマンフィルタータイプ
                - 'adaptive': 適応カルマンフィルター（デフォルト）
                - 'multivariate': 多変量カルマンフィルター
                - 'quantum_adaptive': 量子適応カルマンフィルター
                - 'unscented': 無香料カルマンフィルター
                - 'unscented_v2': 無香料カルマンフィルターV2
            kalman_params: カルマンフィルター固有のパラメータ辞書
            use_smoother: スムーサーを使用するか（デフォルト: True）
            smoother_type: スムーサータイプ
                - 'frama': FRAMA（フラクタル適応移動平均、デフォルト）
                - 'super_smoother': スーパースムーサー
                - 'ultimate_smoother': 究極スムーサー
                - 'zero_lag_ema': ゼロラグEMA
            smoother_params: スムーサー固有のパラメータ辞書
            
            # サイクル検出器固有のパラメータ
            alpha: アルファパラメータ（特定の検出器用）
            bandwidth: 帯域幅（bandpass_zero用）
            center_period: 中心周期（bandpass_zero用）
            avg_length: 平均長（autocorr_perio用）
            window: 分析ウィンドウ長（dft_dominant用）
            period_range: 周期範囲のタプル（特定の検出器用）
            
            # レガシーパラメータ（後方互換性）
            use_kalman_filter_legacy: 旧式カルマンフィルター使用フラグ
            kalman_measurement_noise: 旧式測定ノイズ
            kalman_process_noise: 旧式プロセスノイズ
            kalman_n_states: 旧式状態数
        """
        # インジケーター名の作成
        indicator_name = f"GrandCycleMA(det={detector_type}, fast={fast_limit}, slow={slow_limit}, src={src_type})"
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.detector_type = detector_type
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        self.src_type = src_type.lower()
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.center_period = center_period
        self.avg_length = avg_length
        self.window = window
        self.period_range = period_range
        
        # 新しいフィルタリング・スムージングパラメータ
        self.use_kalman_filter = use_kalman_filter or use_kalman_filter_legacy  # 後方互換性
        self.kalman_filter_type = kalman_filter_type
        self.kalman_params = kalman_params or {}
        self.use_smoother = use_smoother
        self.smoother_type = smoother_type
        self.smoother_params = smoother_params or {}
        
        # レガシーパラメータの統合
        if use_kalman_filter_legacy and not kalman_params:
            self.kalman_params = {
                'measurement_noise': kalman_measurement_noise,
                'process_noise': kalman_process_noise,
                'n_states': kalman_n_states
            }
        
        # パラメータ検証
        if fast_limit <= 0 or fast_limit > 1:
            raise ValueError("fast_limitは0より大きく1以下である必要があります")
        if slow_limit <= 0 or slow_limit > 1:
            raise ValueError("slow_limitは0より大きく1以下である必要があります")
        if slow_limit >= fast_limit:
            raise ValueError("slow_limitはfast_limitより小さい必要があります")
        
        # ソースタイプの検証
        available_sources = PriceSource.get_available_sources()
        if self.src_type not in available_sources:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(available_sources.keys())}")
        
        # フィルタリング・スムージングコンポーネントの初期化
        self.kalman_filter = None
        self.smoother = None
        
        # カルマンフィルターの初期化
        if self.use_kalman_filter:
            try:
                self.kalman_filter = UnifiedKalman(
                    filter_type=self.kalman_filter_type,
                    src_type=self.src_type,
                    **self.kalman_params
                )
            except Exception as e:
                self.logger.warning(f"カルマンフィルター初期化エラー: {e}。カルマンフィルターを無効化します。")
                self.use_kalman_filter = False
        
        # スムーサーの初期化
        if self.use_smoother:
            try:
                self.smoother = UnifiedSmoother(
                    smoother_type=self.smoother_type,
                    src_type=self.src_type,
                    **self.smoother_params
                )
            except Exception as e:
                self.logger.warning(f"スムーサー初期化エラー: {e}。スムーサーを無効化します。")
                self.use_smoother = False
        
        # サイクル検出器の初期化
        self.cycle_detector = EhlersUnifiedDC(
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            use_kalman_filter=False,  # サイクル検出器では無効化（独自処理のため）
            alpha=alpha,
            bandwidth=bandwidth,
            center_period=center_period,
            avg_length=avg_length,
            window=window,
            period_range=period_range
        )
        
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
            params_sig = f"{self.detector_type}_{self.fast_limit}_{self.slow_limit}_{self.src_type}_{self.cycle_part}"
            
            # ハッシュ計算
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.detector_type}_{self.fast_limit}_{self.slow_limit}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> GrandCycleMAResult:
        """
        グランドサイクルMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            GrandCycleMAResult: グランドサイクルMAの値と計算中間値を含む結果
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
                return GrandCycleMAResult(
                    grand_mama_values=cached_result.grand_mama_values.copy(),
                    grand_fama_values=cached_result.grand_fama_values.copy(),
                    cycle_period=cached_result.cycle_period.copy(),
                    alpha_values=cached_result.alpha_values.copy(),
                    phase_values=cached_result.phase_values.copy(),
                    detector_values=cached_result.detector_values.copy()
                )
            
            # 1. 価格ソースの計算
            price_source = PriceSource.calculate_source(data, self.src_type)
            price_source = np.asarray(price_source, dtype=np.float64)
            
            # データ長の検証
            data_length = len(price_source)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            if data_length < 10:
                self.logger.warning(f"データが短すぎます（{data_length}点）。最低10点以上を推奨します。")
            
            # 2. カルマンフィルターによるノイズ除去（オプション）
            processed_price = price_source
            if self.use_kalman_filter and self.kalman_filter:
                try:
                    kalman_result = self.kalman_filter.calculate(data)
                    processed_price = kalman_result.values
                    if len(processed_price) != data_length:
                        self.logger.warning("カルマンフィルター結果の長さが一致しません。元データを使用します。")
                        processed_price = price_source
                except Exception as e:
                    self.logger.warning(f"カルマンフィルター計算エラー: {e}。元データを使用します。")
                    processed_price = price_source
            
            # 3. スムージング処理（オプション）
            smoothed_price = processed_price
            if self.use_smoother and self.smoother:
                try:
                    # スムーサー用にDataFrameまたは配列を準備
                    if isinstance(data, pd.DataFrame):
                        # 処理済み価格でcloseカラムを更新したDataFrameを作成
                        temp_data = data.copy()
                        temp_data['close'] = processed_price
                        smoother_result = self.smoother.calculate(temp_data)
                    else:
                        # NumPy配列の場合、価格データとして直接使用
                        smoother_result = self.smoother.calculate(processed_price.reshape(-1, 1))
                    
                    smoothed_price = smoother_result.values
                    if len(smoothed_price) != data_length:
                        self.logger.warning("スムーサー結果の長さが一致しません。処理済みデータを使用します。")
                        smoothed_price = processed_price
                except Exception as e:
                    self.logger.warning(f"スムーサー計算エラー: {e}。処理済みデータを使用します。")
                    smoothed_price = processed_price
            
            # 4. サイクル検出器でサイクル周期を計算
            cycle_detector_result = self.cycle_detector.calculate(data)
            cycle_period = cycle_detector_result
            
            # 5. グランドサイクルMAの計算（Numba最適化関数を使用）
            grand_mama_values, grand_fama_values, alpha_values, phase_values = calculate_grand_cycle_ma_core(
                smoothed_price, cycle_period, self.fast_limit, self.slow_limit
            )
            
            # 結果の保存
            result = GrandCycleMAResult(
                grand_mama_values=grand_mama_values.copy(),
                grand_fama_values=grand_fama_values.copy(),
                cycle_period=cycle_period.copy(),
                alpha_values=alpha_values.copy(),
                phase_values=phase_values.copy(),
                detector_values=cycle_detector_result.copy()
            )
            
            # キャッシュを更新
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = grand_mama_values  # 基底クラスの要件を満たすため
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"グランドサイクルMA計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            error_result = GrandCycleMAResult(
                grand_mama_values=np.array([]),
                grand_fama_values=np.array([]),
                cycle_period=np.array([]),
                alpha_values=np.array([]),
                phase_values=np.array([]),
                detector_values=np.array([])
            )
            return error_result
    
    def get_values(self) -> Optional[np.ndarray]:
        """グランドサイクルMAMA値のみを取得する（後方互換性のため）"""
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.grand_mama_values.copy()
    
    def get_grand_mama_values(self) -> Optional[np.ndarray]:
        """
        グランドサイクルMAMA値を取得する
        
        Returns:
            np.ndarray: グランドサイクルMAMA値
        """
        return self.get_values()
    
    def get_grand_fama_values(self) -> Optional[np.ndarray]:
        """
        グランドサイクルFAMA値を取得する
        
        Returns:
            np.ndarray: グランドサイクルFAMA値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.grand_fama_values.copy()
    
    def get_cycle_period(self) -> Optional[np.ndarray]:
        """
        サイクル周期を取得する
        
        Returns:
            np.ndarray: サイクル周期
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.cycle_period.copy()
    
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
    
    def get_phase_values(self) -> Optional[np.ndarray]:
        """
        位相値を取得する
        
        Returns:
            np.ndarray: 位相値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.phase_values.copy()
    
    def get_detector_values(self) -> Optional[np.ndarray]:
        """
        サイクル検出器の出力値を取得する
        
        Returns:
            np.ndarray: サイクル検出器の出力値
        """
        if not self._result_cache:
            return None
            
        # 最新のキャッシュを使用
        if self._cache_keys:
            result = self._result_cache[self._cache_keys[-1]]
        else:
            result = next(iter(self._result_cache.values()))
            
        return result.detector_values.copy()
    
    @classmethod
    def get_available_detectors(cls) -> Dict[str, str]:
        """
        利用可能なサイクル検出器とその説明を返す
        
        Returns:
            Dict[str, str]: 検出器名とその説明の辞書
        """
        return EhlersUnifiedDC.get_available_detectors()
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        if hasattr(self.cycle_detector, 'reset'):
            self.cycle_detector.reset()
        if self.kalman_filter and hasattr(self.kalman_filter, 'reset'):
            self.kalman_filter.reset()
        if self.smoother and hasattr(self.smoother, 'reset'):
            self.smoother.reset()