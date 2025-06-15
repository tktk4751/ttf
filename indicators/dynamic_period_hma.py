#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Optional
import numpy as np
import pandas as pd
import traceback

# 依存関係のインポート
from .indicator import Indicator
from .price_source import PriceSource
from .hma import HMA
from .ehlers_unified_dc import EhlersUnifiedDC


class DynamicPeriodHMA(Indicator):
    """
    動的期間HMA (Dynamic Period Hull Moving Average) インジケーター
    
    特徴:
    - 既存のHMAクラスとEhlersUnifiedDCを組み合わせ
    - 各時点でドミナントサイクルから期間を決定してHMAを計算
    - 固定期間モードと適応期間モードを選択可能
    - 既存のHMA実装を最大限活用
    
    使用方法:
    - 市場サイクルに適応するトレンドフォロー
    - 動的なトレンドライン
    - サイクル対応のエントリー・エグジット
    """
    
    def __init__(
        self,
        # --- 期間設定 ---
        period_mode: str = 'adaptive',           # 'adaptive' または 'fixed'
        fixed_period: int = 9,                   # 固定期間モード時の期間
        # --- ドミナントサイクル検出器パラメータ（適応モード時） ---
        detector_type: str = 'phac_e',           # 検出器タイプ
        cycle_part: float = 0.5,                # サイクル部分の倍率
        lp_period: int = 13,
        hp_period: int = 50,
        max_cycle: int = 55,                    # 最大サイクル期間
        min_cycle: int = 5,                     # 最小サイクル期間
        max_output: int = 34,                   # 最大出力値
        min_output: int = 2,                    # 最小出力値
        # --- 価格ソースパラメータ ---
        src_type: str = 'close'                 # 価格ソース
    ):
        """
        コンストラクタ
        
        Args:
            period_mode: 期間モード
                - 'adaptive': ドミナントサイクル検出器から適応期間を決定
                - 'fixed': 固定期間を使用
            fixed_period: 固定期間モード時の期間（デフォルト: 9）
            detector_type: 検出器タイプ（適応モード時）
            cycle_part: サイクル部分の倍率（デフォルト: 0.5）
            max_cycle: 最大サイクル期間（デフォルト: 55）
            min_cycle: 最小サイクル期間（デフォルト: 5）
            max_output: 最大出力値（デフォルト: 34）
            min_output: 最小出力値（デフォルト: 2）
            src_type: 計算に使用する価格ソース（デフォルト: 'close'）
        """
        mode_str = f"{period_mode}_{fixed_period}" if period_mode == 'fixed' else f"{period_mode}_{detector_type}"
        super().__init__(f"DynamicPeriodHMA({mode_str},src={src_type})")
        
        # パラメータの保存
        self.period_mode = period_mode.lower()
        self.fixed_period = fixed_period
        self.detector_type = detector_type
        self.cycle_part = cycle_part
        self.lp_period = lp_period
        self.hp_period = hp_period
        self.max_cycle = max_cycle
        self.min_cycle = min_cycle
        self.max_output = max_output
        self.min_output = max(min_output, 2)  # HMAには最低2が必要
        self.src_type = src_type.lower()
        
        # パラメータ検証
        if self.period_mode not in ['adaptive', 'fixed']:
            raise ValueError(f"無効なperiod_mode: {period_mode}。'adaptive'または'fixed'を指定してください。")
        if self.fixed_period < 2:
            raise ValueError(f"固定期間は2以上である必要があります: {fixed_period}")
        
        # ドミナントサイクル検出器を初期化（適応モード時のみ）
        self.dc_detector = None
        if self.period_mode == 'adaptive':
            self.dc_detector = EhlersUnifiedDC(
                detector_type=self.detector_type,
                cycle_part=self.cycle_part,
                lp_period=self.lp_period,
                hp_period=self.hp_period,
                max_cycle=self.max_cycle,
                min_cycle=self.min_cycle,
                max_output=self.max_output,
                min_output=self.min_output
            )
        
        # HMAインスタンスのキャッシュ（期間ごと）
        self.hma_cache = {}
        
        # 依存ツールの初期化
        self.price_source_extractor = PriceSource()
        
        # 結果キャッシュ
        self._result = None
        self._data_hash = None
        self._periods_used = None
        
    def _get_hma_instance(self, period: int) -> HMA:
        """
        指定された期間のHMAインスタンスを取得（キャッシュ機能付き）
        
        Args:
            period: HMAの期間
        
        Returns:
            HMAインスタンス
        """
        if period not in self.hma_cache:
            self.hma_cache[period] = HMA(
                period=period,
                src_type=self.src_type,
                use_kalman_filter=False  # カルマンフィルターは使用しない
            )
        return self.hma_cache[period]
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    close_val = float(data.iloc[0].get('close', data.iloc[0, 3]))
                    last_close = float(data.iloc[-1].get('close', data.iloc[-1, 3]))
                    data_signature = (length, close_val, last_close)
                else:
                    data_signature = (0, 0.0, 0.0)
            else:
                length = len(data)
                if length > 0 and data.ndim > 1 and data.shape[1] >= 4:
                    data_signature = (length, float(data[0, 3]), float(data[-1, 3]))
                else:
                    data_signature = (0, 0.0, 0.0)
            
            # パラメータのハッシュ
            params_sig = f"{self.period_mode}_{self.fixed_period}_{self.src_type}"
            if self.period_mode == 'adaptive':
                params_sig += f"_{self.detector_type}_{self.max_output}_{self.min_output}"
            
            return f"{hash(data_signature)}_{hash(params_sig)}"
            
        except Exception:
            return f"{id(data)}_{self.period_mode}_{self.fixed_period}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        動的期間HMAを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            動的期間HMA値の配列
        """
        try:
            current_data_len = len(data) if hasattr(data, '__len__') else 0
            if current_data_len == 0:
                self.logger.warning("入力データが空です。")
                return np.array([])

            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            if data_hash == self._data_hash and self._result is not None:
                return self._result

            self._data_hash = data_hash

            # 期間の決定
            if self.period_mode == 'adaptive':
                # ドミナントサイクルから期間を決定
                if self.dc_detector is None:
                    self.logger.error("適応モードですが、ドミナントサイクル検出器が初期化されていません。")
                    return np.full(current_data_len, np.nan)
                
                dc_values = self.dc_detector.calculate(data)
                periods = np.asarray(dc_values, dtype=np.int32)
                
                # 期間の有効性チェック
                valid_periods = periods >= 2
                periods = np.where(valid_periods, periods, 2)  # 最小値2
                
            else:
                # 固定期間を使用
                periods = np.full(current_data_len, self.fixed_period, dtype=np.int32)

            # 結果配列の初期化
            result = np.full(current_data_len, np.nan)
            
            # 各ユニーク期間に対してHMAを計算
            unique_periods = np.unique(periods[~np.isnan(periods.astype(float))])
            hma_results = {}
            
            for period in unique_periods:
                period_int = int(period)
                if period_int < 2:
                    continue
                    
                try:
                    # 該当期間のHMAインスタンスを取得
                    hma_instance = self._get_hma_instance(period_int)
                    
                    # HMAを計算
                    hma_values = hma_instance.calculate(data)
                    hma_results[period_int] = hma_values
                    
                except Exception as e:
                    self.logger.warning(f"期間{period_int}のHMA計算でエラー: {e}")
                    continue
            
            # 各時点で適切な期間のHMA値を選択
            for i in range(current_data_len):
                period_i = int(periods[i]) if not np.isnan(periods[i]) else self.fixed_period
                
                if period_i in hma_results and i < len(hma_results[period_i]):
                    result[i] = hma_results[period_i][i]
            
            # 結果の保存
            self._result = result
            self._periods_used = periods.astype(float)
            self._values = result
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"動的期間HMA計算中にエラー: {error_msg}\n{stack_trace}")
            return np.full(current_data_len, np.nan)

    def get_periods_used(self) -> np.ndarray:
        """
        計算に使用された期間の配列を取得する
        
        Returns:
            np.ndarray: 使用された期間の配列
        """
        if self._periods_used is None:
            return np.array([])
        return self._periods_used.copy()
    
    def get_hma_cache_info(self) -> dict:
        """
        HMAキャッシュの情報を取得する
        
        Returns:
            dict: キャッシュされているHMAインスタンスの期間リスト
        """
        return {
            'cached_periods': list(self.hma_cache.keys()),
            'cache_size': len(self.hma_cache)
        }
    
    def clear_hma_cache(self) -> None:
        """
        HMAキャッシュをクリアする
        """
        for hma_instance in self.hma_cache.values():
            if hasattr(hma_instance, 'reset'):
                hma_instance.reset()
        self.hma_cache.clear()
        self.logger.debug("HMAキャッシュをクリアしました。")
    
    def reset(self) -> None:
        """
        インディケーターの状態をリセットする
        """
        super().reset()
        self._result = None
        self._data_hash = None
        self._periods_used = None
        
        # ドミナントサイクル検出器のリセット
        if self.dc_detector and hasattr(self.dc_detector, 'reset'):
            self.dc_detector.reset()
        
        # HMAキャッシュのリセット
        self.clear_hma_cache()
        
        self.logger.debug(f"インジケータ '{self.name}' がリセットされました。") 