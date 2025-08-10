#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, Type, Tuple
import numpy as np
import pandas as pd

from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult

from .ehlers_hody_dc import EhlersHoDyDC
from .ehlers_phac_dc import EhlersPhAcDC
from .ehlers_dudi_dc import EhlersDuDiDC
from .ehlers_dudi_dce import EhlersDuDiDCE
from .ehlers_hody_dce import EhlersHoDyDCE
from .ehlers_phac_dce import EhlersPhAcDCE
# 新しい検出器のインポート
from .ehlers_cycle_period import EhlersCyclePeriod # サイクル周期検出器
from .ehlers_cycle_period2 import EhlersCyclePeriod2 # 改良サイクル周期検出器
from .ehlers_bandpass_zero_crossings import EhlersBandpassZeroCrossings # バンドパスゼロクロッシング検出器
from .ehlers_autocorrelation_periodogram import EhlersAutocorrelationPeriodogram # 自己相関ピリオドグラム検出器
from .ehlers_dft_dominant_cycle import EhlersDFTDominantCycle # DFTドミナントサイクル検出器
from .ehlers_multiple_bandpass import EhlersMultipleBandpass # 複数バンドパス検出器
from .ehlers_absolute_ultimate_cycle import EhlersAbsoluteUltimateCycle # 絶対的究極サイクル検出器
from .ehlers_neural_quantum_fractal_cycle import EhlersUltraSupremeStabilityCycle # 究極安定性サイクル検出器
# EhlersRefinedCycleDetector は関数内でインポートして循環インポートを回避
from .ehlers_practical_cycle_detector import EhlersPracticalCycleDetector # 実践的サイクル検出器

# 🚀🧠 革新的次世代サイクル検出器
try:
    from .ehlers_ultra_supreme_dft_cycle import EhlersUltraSupremeDFTCycle # 🚀 Ultra Supreme DFT Cycle
except ImportError:
    EhlersUltraSupremeDFTCycle = None

# 追加のサイクル検出器
try:
    from .ehlers_adaptive_ensemble_cycle import EhlersAdaptiveEnsembleCycle
except ImportError:
    EhlersAdaptiveEnsembleCycle = None

try:
    from .ehlers_adaptive_unified_dc import EhlersAdaptiveUnifiedDC
except ImportError:
    EhlersAdaptiveUnifiedDC = None

try:
    from .ehlers_quantum_adaptive_cycle import EhlersQuantumAdaptiveCycle
except ImportError:
    EhlersQuantumAdaptiveCycle = None

try:
    from .ehlers_supreme_ultimate_cycle import EhlersSupremeUltimateCycle
except ImportError:
    EhlersSupremeUltimateCycle = None

try:
    from .ehlers_ultimate_cycle import EhlersUltimateCycle
except ImportError:
    EhlersUltimateCycle = None

try:
    from .supreme_cycle_detector import SupremeCycleDetector
except ImportError:
    SupremeCycleDetector = None

from ..price_source import PriceSource


class EhlersUnifiedDC(EhlersDominantCycle):
    """
    改良版 エーラーズ統合サイクル検出器
    
    このクラスは複数のエーラーズサイクル検出アルゴリズムを統合し、
    単一のインターフェースで利用可能にします。
    
    特徴:
    - 複数のサイクル検出アルゴリズムを選択可能
    - 計算に使用する価格ソースを選択可能 ('close', 'hlc3', 'oc2', etc.)
    - オプションで価格ソースにカルマンフィルターを適用可能
    
    対応検出器:
    - 'hody': ホモダイン判別機 (Homodyne Discriminator)
    - 'phac': 位相累積 (Phase Accumulation)
    - 'dudi': 二重微分 (Dual Differentiator)
    - 'dudi_e': 拡張二重微分 (Enhanced Dual Differentiator)
    - 'hody_e': 拡張ホモダイン判別機 (Enhanced Homodyne Discriminator)
    - 'phac_e': 拡張位相累積 (Enhanced Phase Accumulation)
    - 'cycle_period': サイクル周期検出器 (Cycle Period Dominant Cycle)
    - 'cycle_period2': 改良サイクル周期検出器 (Enhanced Cycle Period)
    - 'bandpass_zero': バンドパスゼロクロッシング検出器 (Bandpass Zero Crossings)
    - 'autocorr_perio': 自己相関ピリオドグラム検出器 (Autocorrelation Periodogram)
    - 'dft_dominant': DFTドミナントサイクル検出器 (DFT Dominant Cycle)
    - 'multi_bandpass': 複数バンドパス検出器 (Multiple Bandpass)
    - 'absolute_ultimate': 絶対的究極サイクル検出器 (Absolute Ultimate Cycle)
    """
    
    # 利用可能な検出器の定義
    _DETECTORS = {
        # コア検出器
        'hody': EhlersHoDyDC,
        'phac': EhlersPhAcDC,
        'dudi': EhlersDuDiDC,
        'dudi_e': EhlersDuDiDCE,
        'hody_e': EhlersHoDyDCE,
        'phac_e': EhlersPhAcDCE,
        
        # 基本サイクル検出器
        'cycle_period': EhlersCyclePeriod,
        'cycle_period2': EhlersCyclePeriod2,
        'bandpass_zero': EhlersBandpassZeroCrossings,
        'autocorr_perio': EhlersAutocorrelationPeriodogram,
        'dft_dominant': EhlersDFTDominantCycle,
        'multi_bandpass': EhlersMultipleBandpass,
        'absolute_ultimate': EhlersAbsoluteUltimateCycle,
        'ultra_supreme_stability': EhlersUltraSupremeStabilityCycle,
        'practical': EhlersPracticalCycleDetector,
        
        # 🚀🧠 革新的次世代検出器
        'ultra_supreme_dft': EhlersUltraSupremeDFTCycle if EhlersUltraSupremeDFTCycle else None,
        
        # 高度な検出器 (インポートに失敗してもスキップ)
        # 'refined': EhlersRefinedCycleDetector,  # 循環インポート回避のため関数内でインポート
    }
    
    # オプショナル検出器 (インポートに失敗してもスキップ)
    _OPTIONAL_DETECTORS = {
        'adaptive_ensemble': EhlersAdaptiveEnsembleCycle,
        'adaptive_unified': EhlersAdaptiveUnifiedDC,
        'quantum_adaptive': EhlersQuantumAdaptiveCycle,
        'supreme_ultimate': EhlersSupremeUltimateCycle,
        'ultimate': EhlersUltimateCycle,
        'supreme': SupremeCycleDetector
    }
    
    # 検出器の説明
    _DETECTOR_DESCRIPTIONS = {
        # コア検出器
        'hody': 'ホモダイン判別機（Homodyne Discriminator）',
        'phac': '位相累積（Phase Accumulation）',
        'dudi': '二重微分（Dual Differentiator）',
        'dudi_e': '拡張二重微分（Enhanced Dual Differentiator）',
        'hody_e': '拡張ホモダイン判別機（Enhanced Homodyne Discriminator）',
        'phac_e': '拡張位相累積（Enhanced Phase Accumulation）',
        
        # 基本サイクル検出器
        'cycle_period': 'サイクル周期検出器（Cycle Period Dominant Cycle）',
        'cycle_period2': '改良サイクル周期検出器（Enhanced Cycle Period）',
        'bandpass_zero': 'バンドパスゼロクロッシング検出器（Bandpass Zero Crossings）',
        'autocorr_perio': '自己相関ピリオドグラム検出器（Autocorrelation Periodogram）',
        'dft_dominant': 'DFTドミナントサイクル検出器（DFT Dominant Cycle）',
        'multi_bandpass': '複数バンドパス検出器（Multiple Bandpass）',
        'absolute_ultimate': '絶対的究極サイクル検出器（Absolute Ultimate Cycle）',
        'ultra_supreme_stability': '究極安定性サイクル検出器（Ultra Supreme Stability Cycle）',
        'practical': '実践的サイクル検出器（Practical Cycle Detector）',
        
        # 🚀🧠 革新的次世代検出器
        'ultra_supreme_dft': '🚀🧠 Ultra Supreme DFT Cycle（究極至高DFT・次世代高性能）',
        
        # 高度な検出器
        'refined': '洗練されたサイクル検出器（Refined Cycle Detector）',
        'adaptive_ensemble': '適応型アンサンブルサイクル（Adaptive Ensemble Cycle）',
        'adaptive_unified': '適応型統合サイクル（Adaptive Unified Dominant Cycle）',
        'quantum_adaptive': '量子適応型サイクル（Quantum Adaptive Cycle）',
        'supreme_ultimate': '最高究極サイクル（Supreme Ultimate Cycle）',
        'ultimate': '究極サイクル（Ultimate Cycle）',
        'supreme': '最高サイクル検出器（Supreme Cycle Detector）'
    }
    
    def __init__(
        self,
        detector_type: str = 'hody_e',
        cycle_part: float = 0.5,
        max_cycle: int = 124,
        min_cycle: int = 13,
        max_output: int = 124,
        min_output: int = 13,
        src_type: str = 'oc2',
        use_kalman_filter: bool = False,
        kalman_measurement_noise: float = 1.0,
        kalman_process_noise: float = 0.01,
        kalman_n_states: int = 5,
        lp_period: int = 13,
        hp_period: int = 124,
        # サイクル検出器パラメータ
        alpha: float = 0.07,
        bandwidth: float = 0.6,
        center_period: float = 15.0,
        avg_length: float = 3.0,
        window: int = 50,
        period_range: Tuple[int, int] = (13, 124),
        # 高度な検出器用パラメータ
        entropy_window: int = 20,
        dft_window: int = 50,
        use_ukf: bool = True,
        ukf_alpha: float = 0.001,
        smoothing_factor: float = 0.1,
        weight_lookback: int = 20,
        adaptive_params: bool = True,
        ultimate_smoother_period: float = 20.0,
        use_ultimate_smoother: bool = True,
        kalman_filter_type: str = 'unscented'
    ):
        """
        コンストラクタ
        
        Args:
            detector_type: 使用する検出器のタイプ
                コア検出器:
                - 'hody': ホモダイン判別機
                - 'phac': 位相累積
                - 'dudi': 二重微分
                - 'dudi_e': 拡張二重微分
                - 'hody_e': 拡張ホモダイン判別機
                - 'phac_e': 拡張位相累積
                基本検出器:
                - 'cycle_period': サイクル周期検出器
                - 'cycle_period2': 改良サイクル周期検出器
                - 'bandpass_zero': バンドパスゼロクロッシング検出器
                - 'autocorr_perio': 自己相関ピリオドグラム検出器
                - 'dft_dominant': DFTドミナントサイクル検出器
                - 'multi_bandpass': 複数バンドパス検出器
                - 'absolute_ultimate': 絶対的究極サイクル検出器
                - 'ultra_supreme_stability': 究極安定性サイクル検出器
                - 'practical': 実践的サイクル検出器
                🚀🧠 革新的次世代検出器:
                - 'ultra_supreme_dft': Ultra Supreme DFT Cycle（最高性能・推奨）
                高度な検出器:
                - 'refined': 洗練されたサイクル検出器
                - 'adaptive_ensemble': 適応型アンサンブルサイクル
                - 'adaptive_unified': 適応型統合サイクル
                - 'quantum_adaptive': 量子適応型サイクル
                - 'supreme_ultimate': 最高究極サイクル
                - 'ultimate': 究極サイクル
                - 'supreme': 最高サイクル検出器
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 50）
            min_cycle: 最小サイクル期間（デフォルト: 6）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 1）
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4', 'oc2', etc.)
            use_kalman_filter: ソース価格にカルマンフィルターを適用するかどうか
            kalman_measurement_noise: カルマンフィルターの測定ノイズ
            kalman_process_noise: カルマンフィルターのプロセスノイズ
            kalman_n_states: カルマンフィルターの状態数
            lp_period: ローパスフィルターの期間（拡張検出器用）
            hp_period: ハイパスフィルターの期間（拡張検出器用）
            alpha: アルファパラメータ（cycle_period、cycle_period2用）
            bandwidth: 帯域幅（bandpass_zero用）
            center_period: 中心周期（bandpass_zero用）
            avg_length: 平均長（autocorr_perio用）
            window: 分析ウィンドウ長（dft_dominant用）
            period_range: 周期範囲のタプル（absolute_ultimate、ultra_supreme_stability用）
            entropy_window: エントロピーウィンドウ（adaptive_ensemble用）
            dft_window: DFTウィンドウ（supreme用）
            use_ukf: UKFフィルタリングを使用するか（supreme用）
            ukf_alpha: UKFのアルファ値（supreme用）
            smoothing_factor: 最終平滑化係数（supreme用）
            weight_lookback: 重み計算の評価期間（supreme用）
            adaptive_params: パラメータの動的調整を行うか（supreme用）
            ultimate_smoother_period: Ultimate Smoother期間（refined用）
            use_ultimate_smoother: Ultimate Smootherを使用するか（refined用）
            kalman_filter_type: Kalmanフィルターのタイプ
        """
        # 検出器名を小文字に変換して正規化
        detector_type = detector_type.lower()
        
        # 利用可能な検出器を統合
        available_detectors = {}
        # 基本検出器を追加（Noneでないもののみ）
        for key, detector_class in self._DETECTORS.items():
            if detector_class is not None:
                available_detectors[key] = detector_class
        
        # オプショナル検出器を追加（インポート済みのみ）
        for key, detector_class in self._OPTIONAL_DETECTORS.items():
            if detector_class is not None:
                available_detectors[key] = detector_class
        
        # 検出器が有効かチェック ('refined' は関数内でインポートするため別途処理)
        valid_detectors = list(available_detectors.keys()) + ['refined']
        if detector_type not in valid_detectors:
            raise ValueError(f"無効な検出器タイプです: {detector_type}。有効なオプション: {', '.join(valid_detectors)}")
        
        # 親クラスの初期化
        name = f"EhlersUnifiedDC(det={detector_type}, src={src_type}, kalman={'Y' if use_kalman_filter else 'N'})"
        super().__init__(
            name,
            cycle_part,
            max_cycle,
            min_cycle,
            max_output,
            min_output
        )
        
        # 検出器タイプとパラメータを保存
        self.detector_type = detector_type
        self.src_type = src_type
        self.use_kalman_filter = use_kalman_filter
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_process_noise = kalman_process_noise
        self.kalman_n_states = kalman_n_states
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.center_period = center_period
        self.avg_length = avg_length
        self.window = window
        self.period_range = period_range
        self.entropy_window = entropy_window
        self.dft_window = dft_window
        self.use_ukf = use_ukf
        self.ukf_alpha = ukf_alpha
        self.smoothing_factor = smoothing_factor
        self.weight_lookback = weight_lookback
        self.adaptive_params = adaptive_params
        self.ultimate_smoother_period = ultimate_smoother_period
        self.use_ultimate_smoother = use_ultimate_smoother
        self.kalman_filter_type = kalman_filter_type
        
        # 統合した検出器辞書を保存
        self.available_detectors = available_detectors
        
        # PriceSourceユーティリティ
        self.price_source_extractor = PriceSource()
        
        # カルマンフィルターは各検出器で個別に処理
        
        # 検出器の初期化
        if detector_type in ['dudi_e', 'hody_e', 'phac_e']:
            # 拡張検出器はローパスとハイパスのパラメータが必要
            self.detector = available_detectors[detector_type](
                lp_period=lp_period,
                hp_period=hp_period,
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type
            )
        elif detector_type in ['cycle_period', 'cycle_period2']:
            # サイクル周期検出器
            self.detector = available_detectors[detector_type](
                alpha=alpha,
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type
            )
        elif detector_type == 'bandpass_zero':
            # バンドパスゼロクロッシング検出器
            self.detector = available_detectors[detector_type](
                bandwidth=bandwidth,
                center_period=center_period,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type
            )
        elif detector_type == 'autocorr_perio':
            # 自己相関ピリオドグラム検出器
            self.detector = available_detectors[detector_type](
                avg_length=avg_length,
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type
            )
        elif detector_type == 'dft_dominant':
            # DFTドミナントサイクル検出器
            self.detector = available_detectors[detector_type](
                window=window,
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type
            )
        elif detector_type == 'multi_bandpass':
            # 複数バンドパス検出器
            self.detector = available_detectors[detector_type](
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type
            )
        elif detector_type == 'absolute_ultimate':
            # 絶対的究極サイクル検出器
            self.detector = available_detectors[detector_type](
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                period_range=period_range,
                src_type=src_type,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type
            )
        elif detector_type == 'ultra_supreme_stability':
            # 究極安定性サイクル検出器
            self.detector = available_detectors[detector_type](
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                period_range=period_range,
                src_type=src_type,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type
            )
        elif detector_type == 'ultra_supreme_dft':
            # 🚀🧠 Ultra Supreme DFT Cycle 検出器（次世代高性能）
            if available_detectors[detector_type] is not None:
                self.detector = available_detectors[detector_type](
                    base_window=window,  # DFT分析窓長
                    cycle_part=cycle_part,
                    max_output=max_output,
                    min_output=min_output,
                    src_type=src_type,
                    # 高度設定
                    adaptive_window=True,  # 適応窓長有効
                    prediction_enabled=True,  # 予測処理有効
                    spectral_optimization=True,  # スペクトル最適化有効
                    # カルマンフィルター設定
                    use_kalman_filter=use_kalman_filter,
                    kalman_filter_type=kalman_filter_type,  # 統合カルマンフィルター使用
                    kalman_pre_filter=True,  # 事前フィルタリング
                    kalman_post_refinement=True,  # 事後洗練
                    # 性能調整
                    quality_threshold=0.6,
                    confidence_boost=1.2,
                    refinement_strength=0.8
                )
            else:
                raise ImportError("EhlersUltraSupremeDFTCycle がインポートされていません")
        elif detector_type == 'adaptive_ensemble':
            # 適応型アンサンブルサイクル検出器
            self.detector = available_detectors[detector_type](
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                entropy_window=entropy_window,
                period_range=period_range,
                src_type=src_type,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type
            )
        elif detector_type == 'supreme':
            # 最高サイクル検出器
            self.detector = available_detectors[detector_type](
                lp_period=lp_period,
                hp_period=hp_period,
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type,
                dft_window=dft_window,
                use_ukf=use_ukf,
                ukf_alpha=ukf_alpha,
                smoothing_factor=smoothing_factor,
                weight_lookback=weight_lookback,
                adaptive_params=adaptive_params,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type
            )
        elif detector_type == 'refined':
            # 洗練されたサイクル検出器 - 循環インポート回避のため関数内でインポート
            try:
                from .ehlers_refined_cycle_detector import EhlersRefinedCycleDetector
            except ImportError:
                from ehlers_refined_cycle_detector import EhlersRefinedCycleDetector
            
            self.detector = EhlersRefinedCycleDetector(
                cycle_part=cycle_part,
                max_output=max_output,
                min_output=min_output,
                period_range=(float(period_range[0]), float(period_range[1])),
                alpha=alpha,
                src_type=src_type,
                ultimate_smoother_period=ultimate_smoother_period,
                use_ultimate_smoother=use_ultimate_smoother,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type
            )
        elif detector_type in available_detectors:
            # その他の高度な検出器（オプショナル検出器を含む）
            try:
                # 汎用的なパラメータで初期化を試みる
                self.detector = available_detectors[detector_type](
                    cycle_part=cycle_part,
                    max_output=max_output,
                    min_output=min_output,
                    src_type=src_type,
                    use_kalman_filter=use_kalman_filter,
                    kalman_filter_type=kalman_filter_type
                )
            except TypeError:
                # パラメータが合わない場合はデフォルトで初期化
                self.detector = available_detectors[detector_type]()
        else:
            # 標準検出器（コア検出器）
            self.detector = self._DETECTORS[detector_type](
                cycle_part=cycle_part,
                max_cycle=max_cycle,
                min_cycle=min_cycle,
                max_output=max_output,
                min_output=min_output,
                src_type=src_type,
                use_kalman_filter=use_kalman_filter,
                kalman_filter_type=kalman_filter_type
            )
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        指定された検出器を使用してドミナントサイクルを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                DataFrameの場合、src_type および (オプションで) HLC カラムが必要
        
        Returns:
            ドミナントサイクルの値
        """
        try:
            # キャッシュチェック - 同じデータの場合は計算をスキップ
            original_data = data # 元データを保持
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result.values
            
            self._data_hash = data_hash  # 新しいハッシュを保存
            
            # --- Run Calculation ---
            # 選択された検出器で計算を実行（カルマンフィルタリングは各検出器内で処理）
            dom_cycle = self.detector.calculate(original_data)
            
            # 結果を直接設定（get_result()を使用しない）
            from .ehlers_dominant_cycle import DominantCycleResult
            # raw_periodとsmooth_periodの設定（ここでは同じ値を使用）
            length = len(dom_cycle)
            raw_period = np.full(length, self.max_cycle)
            smooth_period = np.full(length, self.max_cycle)
            self._result = DominantCycleResult(dom_cycle, raw_period, smooth_period)
            self._values = dom_cycle
            
            return dom_cycle
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"{self.name} 計算中にエラー: {error_msg}\n{stack_trace}")
            data_len = len(original_data) if hasattr(original_data, '__len__') else 0
            self._values = np.full(data_len, np.nan)
            self._data_hash = None # エラー時はキャッシュ無効化
            return self._values
    
    @classmethod
    def get_available_detectors(cls) -> Dict[str, str]:
        """
        利用可能な検出器とその説明を返す
        
        Returns:
            Dict[str, str]: 検出器名とその説明の辞書
        """
        available_descriptions = dict(cls._DETECTOR_DESCRIPTIONS)
        
        # インポート可能なオプショナル検出器のみ追加
        for key, detector_class in cls._OPTIONAL_DETECTORS.items():
            if detector_class is not None and key in cls._DETECTOR_DESCRIPTIONS:
                available_descriptions[key] = cls._DETECTOR_DESCRIPTIONS[key]
        
        return available_descriptions
    
    def reset(self) -> None:
        """インジケータの状態をリセットする"""
        super().reset()
        self._data_hash = None
        if hasattr(self.detector, 'reset'):
            self.detector.reset()

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        # src_typeに基づいて必要なカラムを決定
        required_cols = set()
        if self.src_type == 'open':
            required_cols.add('open')
        elif self.src_type == 'high':
            required_cols.add('high')
        elif self.src_type == 'low':
            required_cols.add('low')
        elif self.src_type == 'close':
            required_cols.add('close')
        elif self.src_type == 'hl2':
            required_cols.update(['high', 'low'])
        elif self.src_type == 'hlc3':
            required_cols.update(['high', 'low', 'close'])
        elif self.src_type == 'ohlc4':
            required_cols.update(['open', 'high', 'low', 'close'])
        elif self.src_type == 'hlcc4':
            required_cols.update(['high', 'low', 'close'])
        elif self.src_type == 'weighted_close':
            required_cols.update(['high', 'low', 'close'])
        elif self.src_type == 'oc2':
            required_cols.update(['open', 'close'])
        else:
            # 不明なsrc_typeの場合はデフォルトでcloseを使用
            required_cols.add('close')

        # EhlersのDC計算は内部でフィルタリング等を行うため、
        # 安全のためにOHLCを要求する場合がある (detectorによる)
        # ここでは、src_typeに必要なものだけをハッシュ対象とする

        if isinstance(data, pd.DataFrame):
            relevant_cols = [col for col in data.columns if col.lower() in required_cols]
            # Check if all *required* columns are present, warn if not?
            # For hashing, only use present relevant columns
            present_cols = [col for col in relevant_cols if col in data.columns]
            if len(present_cols) < len(required_cols):
                 # Log a warning maybe, but proceed with hash of available data
                 pass
            if not present_cols:
                 # If no relevant columns found, hash the whole DataFrame shape? Or raise error?
                 # Let's hash based on shape and first/last row as fallback
                 try:
                     shape_tuple = data.shape
                     first_row = tuple(data.iloc[0]) if len(data) > 0 else ()
                     last_row = tuple(data.iloc[-1]) if len(data) > 0 else ()
                     data_repr_tuple = (shape_tuple, first_row, last_row)
                     data_hash_val = hash(data_repr_tuple)
                 except Exception:
                     data_hash_val = hash(str(data))
            else:
                 data_values = data[present_cols].values # Get only relevant columns
                 data_hash_val = hash(data_values.tobytes())

        elif isinstance(data, np.ndarray):
            # Determine column index based on src_type (assuming OHLC(V) order)
            col_indices = []
            if 'open' in required_cols: col_indices.append(0)
            if 'high' in required_cols: col_indices.append(1)
            if 'low' in required_cols: col_indices.append(2)
            if 'close' in required_cols: col_indices.append(3)
            # Add more mappings for hl2, hlc3, ohlc4 if needed, though they are harder with numpy

            if data.ndim == 2 and data.shape[1] > max(col_indices if col_indices else [-1]):
                data_values = data[:, col_indices]
                data_hash_val = hash(data_values.tobytes())
            else:
                data_hash_val = hash(data.tobytes()) # Fallback
        else:
            data_hash_val = hash(str(data))

        # Include relevant parameters
        param_str = (
            f"det={self.detector_type}_src={self.src_type}_"
            f"kalman={self.use_kalman_filter}_{self.kalman_measurement_noise}_{self.kalman_process_noise}_{self.kalman_n_states}_"
            f"cycPart={self.cycle_part}_maxC={self.max_cycle}_minC={self.min_cycle}_"
            f"maxOut={self.max_output}_minOut={self.min_output}_"
            f"lp={self.lp_period}_hp={self.hp_period}_"
            f"alpha={self.alpha}_bw={self.bandwidth}_cp={self.center_period}_"
            f"avgLen={self.avg_length}_win={self.window}_"
            f"periodRange={self.period_range}"
            # Add other specific detector params if they vary significantly and affect output
        )
        return f"{data_hash_val}_{param_str}" 