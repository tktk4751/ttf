#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import logging

# 既存の検出器とベースクラスをインポート
from .ehlers_dominant_cycle import EhlersDominantCycle, DominantCycleResult
from .ehlers_hody_dc import EhlersHoDyDC
from .ehlers_phac_dc import EhlersPhAcDC
from .ehlers_dudi_dc import EhlersDuDiDC
from .ehlers_dudi_dce import EhlersDuDiDCE
from .ehlers_hody_dce import EhlersHoDyDCE
from .ehlers_phac_dce import EhlersPhAcDCE
try:
    from indicators.kalman.unified_kalman import UnifiedKalman
except ImportError:
    try:
        from ..kalman.unified_kalman import UnifiedKalman
    except ImportError:
        UnifiedKalman = None

class EhlersAdaptiveUnifiedDC(EhlersDominantCycle):
    """
    エーラーズのサイクル検出器を統合し、最大/最小サイクル期間を動的に適応させるインジケーター

    このクラスは、メインのサイクル検出に加え、最大サイクル期間と最小サイクル期間を
    それぞれ別の指定されたエーラーズサイクル検出器を用いて動的に計算します。
    計算された動的な期間を使用して、メインのサイクル検出を実行します。

    利用可能な検出器タイプは EhlersUnifiedDC と同じです。
    """

    _DETECTORS = {
        'hody': EhlersHoDyDC,
        'phac': EhlersPhAcDC,
        'dudi': EhlersDuDiDC,
        'dudi_e': EhlersDuDiDCE,
        'hody_e': EhlersHoDyDCE,
        'phac_e': EhlersPhAcDCE
    }

    _DETECTOR_DESCRIPTIONS = {
        'hody': 'ホモダイン判別機（Homodyne Discriminator）',
        'phac': '位相累積（Phase Accumulation）',
        'dudi': '二重微分（Dual Differentiator）',
        'dudi_e': '拡張二重微分（Enhanced Dual Differentiator）',
        'hody_e': '拡張ホモダイン判別機（Enhanced Homodyne Discriminator）',
        'phac_e': '拡張位相累積（Enhanced Phase Accumulation）'
    }

    def __init__(
        self,
        main_detector_type: str = 'hody',
        max_cycle_detector_type: str = 'phac',
        min_cycle_detector_type: str = 'dudi',
        cycle_part: float = 0.5,
        # max_cycle と min_cycle は動的に決定されるため削除
        max_output: int = 34,
        min_output: int = 1,
        src_type: str = 'close',
        # 拡張検出器用の共通パラメータ
        lp_period: int = 5,
        hp_period: int = 55,
        # 各検出器固有のパラメータを受け付ける可能性も考慮 (将来拡張)
        main_detector_params: Optional[Dict] = None,
        max_cycle_detector_params: Optional[Dict] = None,
        min_cycle_detector_params: Optional[Dict] = None,
        use_kalman_filter: bool = False,
        kalman_filter_type: str = 'adaptive'
    ):
        """
        コンストラクタ

        Args:
            main_detector_type: メインのサイクル計算に使用する検出器のタイプ
            max_cycle_detector_type: 最大サイクル期間の計算に使用する検出器のタイプ
            min_cycle_detector_type: 最小サイクル期間の計算に使用する検出器のタイプ
            cycle_part: サイクル部分の倍率 (メイン検出器用)
            max_output: 最大出力値 (メイン検出器用)
            min_output: 最小出力値 (メイン検出器用)
            src_type: ソースタイプ ('close', 'hlc3', 'hl2', 'ohlc4') (全検出器共通)
            lp_period: ローパスフィルターの期間（拡張検出器用、全検出器共通）
            hp_period: ハイパスフィルターの期間（拡張検出器用、全検出器共通）
            main_detector_params: メイン検出器固有のパラメータ (オプション)
            max_cycle_detector_params: 最大サイクル検出器固有のパラメータ (オプション)
            min_cycle_detector_params: 最小サイクル検出器固有のパラメータ (オプション)
            use_kalman_filter: Kalmanフィルターを使用するか（デフォルト: False）
            kalman_filter_type: Kalmanフィルターのタイプ（デフォルト: 'adaptive'）
        """
        main_detector_type = main_detector_type.lower()
        max_cycle_detector_type = max_cycle_detector_type.lower()
        min_cycle_detector_type = min_cycle_detector_type.lower()

        # 親クラスの初期化 (max_cycle, min_cycle はダミー値または平均的な値で初期化)
        # calculateで動的に決まるため、ここの値は直接は使われない
        # しかし、EhlersDominantCycle の __init__ が要求するので仮の値を入れる
        initial_max_cycle = 50 # 仮の値
        initial_min_cycle = 6  # 仮の値
        super().__init__(
            f"EhlersAdaptiveUnifiedDC({main_detector_type}, max:{max_cycle_detector_type}, min:{min_cycle_detector_type})",
            cycle_part,
            initial_max_cycle,
            initial_min_cycle,
            max_output,
            min_output
        )

        self.main_detector_type = main_detector_type
        self.max_cycle_detector_type = max_cycle_detector_type
        self.min_cycle_detector_type = min_cycle_detector_type
        self.src_type = src_type
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.cycle_part = cycle_part # メイン計算で使うため保持
        self.max_output = max_output # メイン計算で使うため保持
        self.min_output = min_output # メイン計算で使うため保持
        self.use_kalman_filter = use_kalman_filter
        self.kalman_filter_type = kalman_filter_type
        
        # Kalmanフィルターの初期化
        self.kalman_filter = None
        if self.use_kalman_filter:
            self.kalman_filter = UnifiedKalman(
                filter_type=kalman_filter_type,
                src_type='close'  # 内部的にはcloseで使用
            )

        # ロガーの設定
        self.logger = logging.getLogger(__name__)
        # 必要に応じてロガーのレベルやハンドラを設定
        # 例: logging.basicConfig(level=logging.INFO) を呼び出し元で行うか、
        #     ここでハンドラを設定する
        if not self.logger.hasHandlers():
             handler = logging.StreamHandler()
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             self.logger.addHandler(handler)
             self.logger.setLevel(logging.INFO) # デフォルトレベル設定


        # 各検出器のパラメータを準備
        # max/min検出器にも cycle_part, max_output, min_output が必要か確認し、共通パラメータに含める
        # EhlersDominantCycle を継承するクラスはこれらを持つため含めるのが安全
        common_params = {
            'src_type': src_type,
            'lp_period': lp_period,
            'hp_period': hp_period,
            'cycle_part': cycle_part,
            'max_output': max_output,
            'min_output': min_output,
            'max_cycle': initial_max_cycle, # 仮の値
            'min_cycle': initial_min_cycle, # 仮の値
        }

        main_params = (main_detector_params or {}).copy()
        max_params = (max_cycle_detector_params or {}).copy()
        min_params = (min_cycle_detector_params or {}).copy()

        # 検出器インスタンスを作成
        self.main_detector = self._create_detector(main_detector_type, {**common_params, **main_params})
        self.max_cycle_detector = self._create_detector(max_cycle_detector_type, {**common_params, **max_params})
        self.min_cycle_detector = self._create_detector(min_cycle_detector_type, {**common_params, **min_params})

        # calculateメソッドで使う動的max/minサイクル期間のプレースホルダ
        self._dynamic_max_cycle = None
        self._dynamic_min_cycle = None


    def _create_detector(self, detector_type: str, params: Dict) -> EhlersDominantCycle:
        """指定されたタイプの検出器インスタンスを作成するヘルパーメソッド"""
        if detector_type not in self._DETECTORS:
            valid_detectors = ", ".join(self._DETECTORS.keys())
            raise ValueError(f"無効な検出器タイプです: {detector_type}。有効なオプション: {valid_detectors}")

        detector_class = self._DETECTORS[detector_type]
        
        # インスタンス化に必要なパラメータを特定
        # ここでは、共通パラメータと個別パラメータをマージしたものを基に、
        # 検出器クラスのコンストラクタが受け入れる可能性のあるパラメータをフィルタリングする
        # (より堅牢にするには、inspectモジュールでコンストラクタの引数を調べる方法もある)
        
        possible_params = {}
        if detector_type in ['dudi_e', 'hody_e', 'phac_e']:
             # 拡張検出器のパラメータ
             possible_params = {
                'lp_period', 'hp_period', 'cycle_part', 'max_output', 'min_output', 'src_type'
            }
        else:
            # 標準検出器のパラメータ
            possible_params = {
                 'cycle_part', 'max_cycle', 'min_cycle', 'max_output', 'min_output', 'src_type'
            }
            
        # 利用可能なパラメータのみを抽出して渡す
        init_params = {k: v for k, v in params.items() if k in possible_params}

        try:
            # デフォルト値を補完する必要があるかもしれない
            # 例: 標準検出器に lp_period が渡されても無視される
            #     拡張検出器に max_cycle が渡されても無視される
            return detector_class(**init_params)
        except TypeError as e:
            self.logger.error(f"検出器 {detector_type} の初期化に失敗しました。試行したパラメータ: {init_params}, エラー: {e}")
            # 不足しているパラメータなど、より詳細なエラー情報を提供できると良い
            import inspect
            sig = inspect.signature(detector_class.__init__)
            available_params = list(sig.parameters.keys())
            self.logger.error(f"{detector_type} が期待するパラメータ: {available_params}")
            raise

    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        動的な最大/最小サイクル期間を使用してドミナントサイクルを計算する

        Args:
            data: 価格データ（DataFrameまたはNumPy配列）

        Returns:
            ドミナントサイクルの値 (NumPy配列)
        """
        try:
            # データ長を取得
            data_length = len(data) if isinstance(data, (pd.DataFrame, np.ndarray)) else 0
            if data_length == 0:
                return np.array([])

            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                # キャッシュされた _values は numpy 配列のはず
                return self._values if isinstance(self._values, np.ndarray) else np.array([])

            self._data_hash = data_hash

            # --- 1. 最大サイクル期間の計算 ---
            self.max_cycle_detector.calculate(data)
            
            # 安全な結果取得
            if hasattr(self.max_cycle_detector, 'get_result') and callable(getattr(self.max_cycle_detector, 'get_result')):
                max_cycle_result = self.max_cycle_detector.get_result()
            elif hasattr(self.max_cycle_detector, '_result'):
                max_cycle_result = self.max_cycle_detector._result
            else:
                # フォールバック: 計算結果を直接使用
                if hasattr(self.max_cycle_detector, '_values') and self.max_cycle_detector._values is not None:
                    # DominantCycleResult を手動作成
                    values = self.max_cycle_detector._values
                    max_cycle_result = DominantCycleResult(
                        values=values,
                        raw_period=values,  # 同じ値を使用
                        smooth_period=values
                    )
                else:
                    self.logger.error("最大サイクル検出器から結果を取得できませんでした。")
                    return np.full(data_length, np.nan)
            
            if max_cycle_result is None:
                 self.logger.error("最大サイクル期間の計算結果がNoneです。")
                 # エラー時はNaN配列を返す
                 return np.full(data_length, np.nan)
            # smooth_period を使用
            dynamic_max_cycle_period = max_cycle_result.smooth_period.copy() # copy() で参照渡しを避ける
            
            # --- 2. 最小サイクル期間の計算 ---
            self.min_cycle_detector.calculate(data)
            
            # 安全な結果取得
            if hasattr(self.min_cycle_detector, 'get_result') and callable(getattr(self.min_cycle_detector, 'get_result')):
                min_cycle_result = self.min_cycle_detector.get_result()
            elif hasattr(self.min_cycle_detector, '_result'):
                min_cycle_result = self.min_cycle_detector._result
            else:
                # フォールバック: 計算結果を直接使用
                if hasattr(self.min_cycle_detector, '_values') and self.min_cycle_detector._values is not None:
                    # DominantCycleResult を手動作成
                    values = self.min_cycle_detector._values
                    min_cycle_result = DominantCycleResult(
                        values=values,
                        raw_period=values,  # 同じ値を使用
                        smooth_period=values
                    )
                else:
                    self.logger.error("最小サイクル検出器から結果を取得できませんでした。")
                    return np.full(data_length, np.nan)
            
            if min_cycle_result is None:
                 self.logger.error("最小サイクル期間の計算結果がNoneです。")
                 return np.full(data_length, np.nan)
            dynamic_min_cycle_period = min_cycle_result.smooth_period.copy()

            # --- 3. 動的期間のサニタイズと調整 ---
            # NaNを処理: 検出器のデフォルト境界値で埋める
            default_max = self.max_cycle_detector.max_cycle if hasattr(self.max_cycle_detector, 'max_cycle') else 50
            default_min = self.min_cycle_detector.min_cycle if hasattr(self.min_cycle_detector, 'min_cycle') else 6
            
            dynamic_max_cycle_period = np.nan_to_num(dynamic_max_cycle_period, nan=default_max)
            dynamic_min_cycle_period = np.nan_to_num(dynamic_min_cycle_period, nan=default_min)

            # 物理的な制約を適用
            dynamic_min_cycle_period = np.maximum(dynamic_min_cycle_period, 1) # 最小期間は1以上
            # max > min の制約を保証 (min >= max の場合、min を max-1 に調整)
            mask = dynamic_min_cycle_period >= dynamic_max_cycle_period
            dynamic_min_cycle_period[mask] = np.maximum(1, dynamic_max_cycle_period[mask] - 1)
            # 念のため、maxもmin以上であることを確認（上記調整でmax<1になるケースを防ぐ）
            dynamic_max_cycle_period = np.maximum(dynamic_max_cycle_period, dynamic_min_cycle_period + 1)


            self._dynamic_max_cycle = dynamic_max_cycle_period
            self._dynamic_min_cycle = dynamic_min_cycle_period

            # --- 4. メインのサイクル計算と動的境界の適用 ---
            # メイン検出器で初期（固定）境界を使って計算
            main_dom_cycle_values = self.main_detector.calculate(data)
            
            # 安全な結果取得
            if hasattr(self.main_detector, 'get_result') and callable(getattr(self.main_detector, 'get_result')):
                main_result = self.main_detector.get_result()
            elif hasattr(self.main_detector, '_result'):
                main_result = self.main_detector._result
            else:
                # フォールバック: 計算結果を直接使用
                if hasattr(self.main_detector, '_values') and self.main_detector._values is not None:
                    # DominantCycleResult を手動作成
                    values = self.main_detector._values
                    main_result = DominantCycleResult(
                        values=values,
                        raw_period=values,  # 同じ値を使用
                        smooth_period=values
                    )
                else:
                    self.logger.error("メイン検出器から結果を取得できませんでした。")
                    return np.full(data_length, np.nan)
            
            if main_result is None:
                self.logger.error("メインのサイクル計算結果がNoneです。")
                return np.full(data_length, np.nan)

            # 計算された期間を動的境界でクリッピング
            # main_resultの期間もNaNを含む可能性があるので処理
            smooth_period_raw = np.nan_to_num(main_result.smooth_period, nan=default_max) # NaNは仮にmaxで埋める
            raw_period_raw = np.nan_to_num(main_result.raw_period, nan=default_max)

            clipped_smooth_period = np.clip(
                smooth_period_raw,
                self._dynamic_min_cycle,
                self._dynamic_max_cycle
            )
            clipped_raw_period = np.clip(
                raw_period_raw,
                self._dynamic_min_cycle,
                self._dynamic_max_cycle
            )

            # 最終的なサイクル値はメイン検出器の計算結果をそのまま使用する
            # (期間情報のみ動的に調整されたものとして扱う)
            final_dom_cycle_values = main_dom_cycle_values

             # 結果をDominantCycleResultとして保存
            # 結果の配列長がデータ長と一致しているか確認
            if len(final_dom_cycle_values) != data_length or \
               len(clipped_raw_period) != data_length or \
               len(clipped_smooth_period) != data_length:
                self.logger.error(f"結果の配列長がデータ長({data_length})と一致しません。"
                                  f"Values: {len(final_dom_cycle_values)}, "
                                  f"RawP: {len(clipped_raw_period)}, "
                                  f"SmoothP: {len(clipped_smooth_period)}")
                # 長さを合わせるために NaN で埋めるなどの処理が必要かもしれない
                # ここではエラーログを出力し、不完全な結果を返す可能性がある
                # より安全なのは NaN 配列を返すこと
                return np.full(data_length, np.nan)


            self._result = DominantCycleResult(
                final_dom_cycle_values,
                clipped_raw_period,
                clipped_smooth_period
            )
            self._values = final_dom_cycle_values # キャッシュ用

            self.logger.info(f"動的サイクル計算完了。データ長: {data_length}")
            # デバッグログは大量に出力される可能性があるので注意
            # self.logger.debug(f"Dynamic Max Cycle (tail): {self._dynamic_max_cycle[-5:]}")
            # self.logger.debug(f"Dynamic Min Cycle (tail): {self._dynamic_min_cycle[-5:]}")
            # self.logger.debug(f"Clipped Smooth Period (tail): {clipped_smooth_period[-5:]}")

            return self._values

        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"EhlersAdaptiveUnifiedDC計算中にエラー: {error_msg}\n{stack_trace}")
            data_length = len(data) if isinstance(data, (pd.DataFrame, np.ndarray)) else 0
            return np.full(data_length, np.nan) # エラー時は NaN 配列を返す

    def get_dynamic_cycles(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        最後に計算された動的な最大サイクル期間と最小サイクル期間の配列を返す

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: (max_cycle_period, min_cycle_period) のタプル、または計算前は None
        """
        if self._dynamic_max_cycle is not None and self._dynamic_min_cycle is not None:
            return self._dynamic_max_cycle, self._dynamic_min_cycle
        else:
            return None

    @classmethod
    def get_available_detectors(cls) -> Dict[str, str]:
        """
        利用可能な検出器とその説明を返す

        Returns:
            Dict[str, str]: 検出器名とその説明の辞書
        """
        return cls._DETECTOR_DESCRIPTIONS 