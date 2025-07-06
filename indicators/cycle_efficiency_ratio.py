#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Dict, Optional, List
import numpy as np
import pandas as pd
from numba import njit

from .indicator import Indicator
from .ehlers_unified_dc import EhlersUnifiedDC
from .price_source import PriceSource
from .kalman_filter import KalmanFilter
from .efficiency_ratio import calculate_efficiency_ratio_for_period
from .alma import ALMA


class CycleEfficiencyRatio(Indicator):
    """
    サイクル効率比(CER)インジケーター
    
    ドミナントサイクルを使用して動的なウィンドウサイズでの効率比を計算します。
    計算に使用する価格ソースを選択可能で、オプションでカルマンフィルターを適用できます。
    また、ALMAによるスムージングも可能です。
    セルフアダプティブモードでは、効率比に応じてスムージング期間を動的に調整します。
    """
    
    def __init__(
        self,
        detector_type: str = 'phac_e',
        lp_period: int = 5,
        hp_period: int = 144,
        cycle_part: float = 0.5,
        max_cycle: int = 144,
        min_cycle: int = 5,
        max_output: int = 55,
        min_output: int = 5,
        src_type: str = 'hlc3',
        use_kalman_filter: bool = False,
        kalman_measurement_noise: float = 1.0,
        kalman_process_noise: float = 0.01,
        kalman_n_states: int = 5,
        smooth_er: bool = True,
        er_alma_period: int = 5,
        er_alma_offset: float = 0.85,
        er_alma_sigma: float = 6,
        self_adaptive: bool = False,
        # 新しい検出器用のパラメータ
        alpha: float = 0.07,
        bandwidth: float = 0.6,
        center_period: float = 15.0,
        avg_length: float = 3.0,
        window: int = 50
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: ドミナントサイクル検出器タイプ
                - 'hody': ホモダイン判別機
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積（デフォルト）
                - 'cycle_period': サイクル周期検出器
                - 'cycle_period2': 改良サイクル周期検出器
                - 'bandpass_zero': バンドパスゼロクロッシング検出器
                - 'autocorr_perio': 自己相関ピリオドグラム検出器
                - 'dft_dominant': DFTドミナントサイクル検出器
                - 'multi_bandpass': 複数バンドパス検出器
            lp_period: ドミナントサイクル用ローパスフィルター期間
            hp_period: ドミナントサイクル用ハイパスフィルター期間
            cycle_part: ドミナントサイクル計算用サイクル部分
            max_cycle: ドミナントサイクル最大期間
            min_cycle: ドミナントサイクル最小期間
            max_output: ドミナントサイクル最大出力値
            min_output: ドミナントサイクル最小出力値
            src_type: 価格ソース ('close', 'hlc3', etc.)
            use_kalman_filter: ソース価格にカルマンフィルターを適用するかどうか
            kalman_measurement_noise: カルマンフィルター測定ノイズ
            kalman_process_noise: カルマンフィルタープロセスノイズ
            kalman_n_states: カルマンフィルター状態数
            smooth_er: 効率比にALMAスムージングを適用するかどうか
            er_alma_period: ALMAスムージングの期間
            er_alma_offset: ALMAスムージングのオフセット
            er_alma_sigma: ALMAスムージングのシグマ
            self_adaptive: セルフアダプティブモードを有効にするかどうか (スムージングが有効時のみ機能)
            alpha: アルファパラメータ（cycle_period、cycle_period2用）
            bandwidth: 帯域幅（bandpass_zero用）
            center_period: 中心周期（bandpass_zero用）
            avg_length: 平均長（autocorr_perio用）
            window: 分析ウィンドウ長（dft_dominant用）
        """
        kalman_str = f"_kalman={'Y' if use_kalman_filter else 'N'}" if use_kalman_filter else ""
        smooth_str = f"_smooth={'Y' if smooth_er else 'N'}" if smooth_er else ""
        adaptive_str = f"_adaptive={'Y' if self_adaptive else 'N'}" if self_adaptive and smooth_er else ""
        indicator_name = f"CER(det={detector_type},part={cycle_part},src={src_type}{kalman_str}{smooth_str}{adaptive_str})"
        super().__init__(indicator_name)
        
        # パラメータ保存
        self.detector_type = detector_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = min_output
        self.src_type = src_type
        self.use_kalman_filter = use_kalman_filter
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_process_noise = kalman_process_noise
        self.kalman_n_states = kalman_n_states
        
        # ALMAスムージングパラメータ
        self.smooth_er = smooth_er
        self.er_alma_period = er_alma_period
        self.er_alma_offset = er_alma_offset
        self.er_alma_sigma = er_alma_sigma
        self.self_adaptive = self_adaptive
        self.adaptive_min_period = 1
        self.adaptive_max_period = 5
        
        # 新しい検出器用のパラメータ
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.center_period = center_period
        self.avg_length = avg_length
        self.window = window
        
        # PriceSourceとKalmanFilterの初期化
        self.price_source_extractor = PriceSource()
        self.kalman_filter = None
        if self.use_kalman_filter:
            self.kalman_filter = KalmanFilter(
                price_source=self.src_type,
                measurement_noise=self.kalman_measurement_noise,
                process_noise=self.kalman_process_noise,
                n_states=self.kalman_n_states
            )
        
        # ドミナントサイクル検出器
        self.dc_detector = EhlersUnifiedDC(
            detector_type=detector_type,
            cycle_part=cycle_part,
            max_cycle=max_cycle,
            min_cycle=min_cycle,
            max_output=max_output,
            min_output=min_output,
            src_type=src_type,
            use_kalman_filter=False,  # CERレベルでKalmanを適用するため、DCレベルでは無効化
            lp_period=lp_period,
            hp_period=hp_period,
            alpha=alpha,
            bandwidth=bandwidth,
            center_period=center_period,
            avg_length=avg_length,
            window=window
        )
        
        # ALMAスムーザーの初期化（有効な場合）
        self.er_alma_smoother = None
        if self.smooth_er:
            self.er_alma_smoother = ALMA(
                period=self.er_alma_period,
                offset=self.er_alma_offset,
                sigma=self.er_alma_sigma,
                use_dynamic_period=False,
                src_type='close',  # 直接ERの値を渡すので、ソースタイプは関係ない

            )
        
        # 結果キャッシュ
        self._values = None  # 生のER値
        self._smoothed_values = None  # スムージングされたER値
        self._data_hash = None
        self._adaptive_periods = None  # セルフアダプティブモード用の動的期間配列
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとパラメータに基づいてハッシュ値を計算する (ALMA参考)"""
        # src_typeに基づいて必要なカラムを決定 (ALMAと同様のロジック)
        required_cols = set()
        src_lower = self.src_type.lower()
        if src_lower == 'open':
            required_cols.add('open')
        elif src_lower == 'high':
            required_cols.add('high')
        elif src_lower == 'low':
            required_cols.add('low')
        elif src_lower == 'close':
            required_cols.add('close')
        elif src_lower == 'hl2':
            required_cols.update(['high', 'low'])
        elif src_lower == 'hlc3':
            required_cols.update(['high', 'low', 'close'])
        elif src_lower == 'ohlc4':
            required_cols.update(['open', 'high', 'low', 'close'])
        elif src_lower == 'hlcc4': # ALMAの例に合わせる
            required_cols.update(['high', 'low', 'close'])
        elif src_lower == 'weighted_close': # ALMAの例に合わせる
            required_cols.update(['high', 'low', 'close'])
        else:
            required_cols.add('close') # デフォルトはclose

        if isinstance(data, pd.DataFrame):
            present_cols = [col for col in data.columns if col.lower() in required_cols]
            if not present_cols:
                # 必要なカラムがない場合、基本的な情報でハッシュ
                try:
                    shape_tuple = data.shape
                    first_row = tuple(data.iloc[0]) if len(data) > 0 else ()
                    last_row = tuple(data.iloc[-1]) if len(data) > 0 else ()
                    data_repr_tuple = (shape_tuple, first_row, last_row)
                    data_hash_val = hash(data_repr_tuple)
                except Exception:
                    data_hash_val = hash(str(data)) # フォールバック
            else:
                # 関連するカラムの値でハッシュ
                data_values = data[present_cols].values
                data_hash_val = hash(data_values.tobytes())

        elif isinstance(data, np.ndarray):
            # NumPy配列の場合、形状や一部の値でハッシュ (簡略化)
            try:
                 shape_tuple = data.shape
                 first_row = tuple(data[0]) if len(data) > 0 else ()
                 last_row = tuple(data[-1]) if len(data) > 0 else ()
                 mean_val = np.mean(data) if data.size > 0 else 0.0
                 data_repr_tuple = (shape_tuple, first_row, last_row, mean_val)
                 data_hash_val = hash(data_repr_tuple)
            except Exception:
                 data_hash_val = hash(data.tobytes()) # フォールバック
        else:
            data_hash_val = hash(str(data)) # その他の型

        # パラメータ文字列を作成
        param_str = (
            f"det={self.detector_type}_lp={self.lp_period}_hp={self.hp_period}_"
            f"part={self.cycle_part}_maxC={self.max_cycle}_minC={self.min_cycle}_"
            f"maxO={self.max_output}_minO={self.min_output}_src={self.src_type}_"
            f"kalman={self.use_kalman_filter}_{self.kalman_measurement_noise}_"
            f"{self.kalman_process_noise}_{self.kalman_n_states}_"
            f"smooth={self.smooth_er}_{self.er_alma_period}_{self.er_alma_offset}_{self.er_alma_sigma}_"
            f"adaptive={self.self_adaptive}_{self.adaptive_min_period}_{self.adaptive_max_period}_"
            f"alpha={self.alpha}_bw={self.bandwidth}_cp={self.center_period}_"
            f"avgLen={self.avg_length}_win={self.window}"
        )
        return f"{data_hash_val}_{param_str}"
    
    def _calculate_adaptive_periods(self, er_values: np.ndarray) -> np.ndarray:
        """
        効率比に基づいて動的なALMA期間を計算
        
        Args:
            er_values: 効率比の値
        
        Returns:
            np.ndarray: 各データポイント用の動的ALMA期間（整数）
        """
        if not self.smooth_er or not self.self_adaptive:
            # セルフアダプティブが無効な場合は固定値を返す
            return np.full(len(er_values), self.er_alma_period)
            
        # 効率比の値を0-1の範囲に正規化（NaNは除外）
        valid_er = er_values[~np.isnan(er_values)]
        if len(valid_er) == 0:
            return np.full(len(er_values), self.er_alma_period)
            
        min_er = np.min(valid_er)
        max_er = np.max(valid_er)
        
        # 値の範囲が狭すぎる場合のガード
        if max_er - min_er < 0.001:
            return np.full(len(er_values), self.adaptive_min_period + 
                          (self.adaptive_max_period - self.adaptive_min_period) // 2)
        
        # 正規化（高いER値→小さい期間、低いER値→大きい期間）
        normalized_er = (er_values - min_er) / (max_er - min_er)
        
        # 期間を反転して計算（ER値が高いほど小さい期間になるように）
        # normalize_er: 0→1, period: max→min
        periods = self.adaptive_max_period - (normalized_er * 
                 (self.adaptive_max_period - self.adaptive_min_period))
        
        # 整数値に丸める
        return np.round(periods).astype(int)
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        サイクル効率比を計算
        
        Args:
            data: OHLC価格データ（DataFrameまたはNumpy配列）
        
        Returns:
            np.ndarray: サイクル効率比の値（スムージングが有効な場合はスムージングされた値）
        """
        try:
            # ハッシュチェックでキャッシュ利用
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._values is not None:
                return self._smoothed_values if self.smooth_er else self._values
            
            # --- PriceSourceの静的メソッドを使ってソース価格取得 ---
            src_prices = PriceSource.calculate_source(data, self.src_type)

            # --- Optional Kalman Filtering ---
            effective_prices = src_prices
            if self.use_kalman_filter and self.kalman_filter:
                # 元データを渡してKalmanFilter内部でソース取得・計算させる
                filtered_prices = self.kalman_filter.calculate(data)
                if filtered_prices is not None and len(filtered_prices) == len(effective_prices):
                    effective_prices = filtered_prices
                else:
                    self.logger.warning("カルマンフィルター計算失敗またはサイズ不一致。元のソース価格を使用します。")

            # データ長の検証
            data_length = len(effective_prices)
            if data_length == 0:
                self.logger.warning("価格データが空です。空の配列を返します。")
                self._values = np.array([])
                self._smoothed_values = np.array([])
                self._data_hash = data_hash # 空でもキャッシュする
                return self._values

            # ドミナントサイクルの計算 (元のデータを渡す)
            # 注意: dc_detector は内部で 'hlc3' 等の固定ソースを使う可能性がある
            dc_values = self.dc_detector.calculate(data)

            # ドミナントサイクル期間の計算
            valid_dc = dc_values[~np.isnan(dc_values) & (dc_values > 0)]
            period = int(np.mean(valid_dc)) if len(valid_dc) > 0 else 14 # デフォルト14
            period = max(8, min(period, 34)) # 期間を 8-34 に制限

            # 効率比の計算 (有効な価格データと期間を使用)
            er_values = calculate_efficiency_ratio_for_period(effective_prices, period)
            
            # セルフアダプティブモードでの動的期間計算
            self._adaptive_periods = None
            if self.smooth_er and self.self_adaptive:
                self._adaptive_periods = self._calculate_adaptive_periods(er_values)
            
            # ALMAによるスムージング（有効な場合）
            smoothed_er_values = er_values.copy()  # デフォルトはスムージングなし
            if self.smooth_er:
                try:
                    if self.self_adaptive and self._adaptive_periods is not None:
                        # セルフアダプティブモード: 各ポイントごとに異なる期間でスムージング
                        smoothed_er_values = np.full_like(er_values, np.nan)
                        
                        # 適応期間配列をキャッシュ
                        periods = self._adaptive_periods
                        
                        # ポイントごとにALMAを適用
                        for i in range(len(er_values)):
                            # i番目のポイントに対応する期間で新しいALMAを作成
                            current_period = periods[i]
                            current_alma = ALMA(
                                period=current_period,
                                offset=self.er_alma_offset,
                                sigma=self.er_alma_sigma,
                                src_type='close',
                                use_kalman_filter=False
                            )
                            
                            # ALMAスムージングを適用（i番目までのデータでスムージング）
                            slice_data = er_values[:i+1]
                            if len(slice_data) >= current_period:
                                try:
                                    # データの形状を確認し、適切に整形
                                    if slice_data.ndim > 1:
                                        slice_data_1d = slice_data.flatten()
                                    else:
                                        slice_data_1d = slice_data
                                    
                                    # ALMA計算を実行
                                    smoothed_slice = current_alma.calculate(slice_data_1d)
                                    
                                    # 結果の形状を確認
                                    if hasattr(smoothed_slice, 'values'):
                                        smoothed_slice = smoothed_slice.values
                                    elif isinstance(smoothed_slice, (list, tuple)) and len(smoothed_slice) > 0:
                                        # タプルやリストの場合は最初の要素を使用
                                        smoothed_slice = smoothed_slice[0] if hasattr(smoothed_slice[0], '__len__') else smoothed_slice
                                    
                                    if not np.isnan(smoothed_slice[-1]):
                                        smoothed_er_values[i] = smoothed_slice[-1]
                                    else:
                                        smoothed_er_values[i] = er_values[i]
                                except Exception as e:
                                    # ALMA計算でエラーが発生した場合は元の値を使用
                                    smoothed_er_values[i] = er_values[i]
                            else:
                                smoothed_er_values[i] = er_values[i]
                    else:
                        # 通常のスムージング: 固定期間でALMA適用
                        if self.er_alma_smoother is None:
                            self.er_alma_smoother = ALMA(
                                period=self.er_alma_period,
                                offset=self.er_alma_offset,
                                sigma=self.er_alma_sigma,
                                src_type='close',
                                use_kalman_filter=False
                            )
                            
                        # 1次元配列として渡す
                        try:
                            # データの形状を確認し、適切に整形
                            if er_values.ndim > 1:
                                # 多次元配列の場合は最初の次元のみを使用
                                er_values_1d = er_values.flatten()
                            else:
                                er_values_1d = er_values
                            
                            # ALMA計算を実行
                            smoothed_values = self.er_alma_smoother.calculate(er_values_1d)
                            
                            # 結果の形状を確認
                            if hasattr(smoothed_values, 'values'):
                                smoothed_values = smoothed_values.values
                            elif isinstance(smoothed_values, (list, tuple)) and len(smoothed_values) > 0:
                                # タプルやリストの場合は最初の要素を使用
                                smoothed_values = smoothed_values[0] if hasattr(smoothed_values[0], '__len__') else smoothed_values
                            
                            # NaNの処理（最初の数ポイントはNaNになるため、元の値で埋める）
                            nan_indices = np.isnan(smoothed_values)
                            smoothed_er_values = smoothed_values.copy()
                            smoothed_er_values[nan_indices] = er_values_1d[nan_indices]
                        except Exception as e:
                            # ALMA計算でエラーが発生した場合は元の値を使用
                            self.logger.warning(f"ALMA計算中にエラー: {str(e)}。元のER値を使用します。")
                            smoothed_er_values = er_values.copy()
                except Exception as e:
                    self.logger.error(f"ER値のスムージング中にエラー: {str(e)}。生のER値を使用します。")
                    smoothed_er_values = er_values.copy()  # エラー時は元の値を使用

            # 結果を保存してキャッシュ
            self._values = er_values
            self._smoothed_values = smoothed_er_values
            self._data_hash = data_hash

            # スムージングが有効な場合はスムージングされた値を返す
            return self._smoothed_values if self.smooth_er else self._values
            
        except Exception as e:
            import traceback
            error_msg = f"サイクル効率比計算中にエラー: {str(e)}"
            stack_trace = traceback.format_exc()
            self.logger.error(f"{error_msg}\n{stack_trace}")
            data_len = len(data) if hasattr(data, '__len__') else 0
            self._values = np.full(data_len, np.nan) # エラー時はNaN配列
            self._smoothed_values = np.full(data_len, np.nan)
            self._data_hash = None # エラー時はキャッシュクリア
            return self._values if not self.smooth_er else self._smoothed_values
    
    def get_raw_values(self) -> np.ndarray:
        """
        スムージングされていない生のサイクル効率比の値を取得する
        
        Returns:
            np.ndarray: 生のサイクル効率比の値
        """
        return self._values if self._values is not None else np.array([])
    
    def get_smoothed_values(self) -> np.ndarray:
        """
        スムージングされたサイクル効率比の値を取得する（スムージングが有効な場合のみ）
        
        Returns:
            np.ndarray: スムージングされたサイクル効率比の値
        """
        return self._smoothed_values if self._smoothed_values is not None else np.array([])
    
    def get_adaptive_periods(self) -> np.ndarray:
        """
        セルフアダプティブモードで使用された動的期間を取得する
        
        Returns:
            np.ndarray: 各ポイントに使用された動的ALMA期間（セルフアダプティブモード時のみ有効）
        """
        return self._adaptive_periods if self._adaptive_periods is not None else np.array([])
    
    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._values = None
        self._smoothed_values = None
        self._adaptive_periods = None
        self._data_hash = None
        if self.kalman_filter and hasattr(self.kalman_filter, 'reset'):
            self.kalman_filter.reset()
        if self.er_alma_smoother and hasattr(self.er_alma_smoother, 'reset'):
            self.er_alma_smoother.reset()
    
    def __str__(self) -> str:
        """文字列表現"""
        kalman_str = f", kalman={self.use_kalman_filter}" if self.use_kalman_filter else ""
        smooth_str = f", smooth={self.smooth_er}" if self.smooth_er else ""
        adaptive_str = f", adaptive={self.self_adaptive}" if self.self_adaptive and self.smooth_er else ""
        return f"CER(det={self.detector_type}, part={self.cycle_part}, src={self.src_type}{kalman_str}{smooth_str}{adaptive_str})" 