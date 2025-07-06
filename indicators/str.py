#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Tuple, Dict, Optional, List, NamedTuple
import numpy as np
import pandas as pd
from numba import jit, njit, float64
import traceback

try:
    from .indicator import Indicator
    from .price_source import PriceSource
    from .ultimate_smoother import UltimateSmoother
    # EhlersUnifiedDC は関数内でインポートして循環インポートを回避
except ImportError:
    # スタンドアロン実行時の対応
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from indicator import Indicator
    from price_source import PriceSource
    from ultimate_smoother import UltimateSmoother
    # EhlersUnifiedDC は関数内でインポートして循環インポートを回避


@dataclass
class STRResult:
    """STR（Smooth True Range）の計算結果"""
    values: np.ndarray           # STR値
    true_range: np.ndarray       # True Range値
    true_high: np.ndarray        # True High値
    true_low: np.ndarray         # True Low値


@njit(fastmath=True, cache=True)
def calculate_true_range_values(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    True High、True Low、True Rangeを計算する
    
    Args:
        high: 高値の配列
        low: 安値の配列
        close: 終値の配列
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: True High、True Low、True Range
    """
    length = len(close)
    true_high = np.zeros(length, dtype=np.float64)
    true_low = np.zeros(length, dtype=np.float64)
    true_range = np.zeros(length, dtype=np.float64)
    
    # 最初の値は現在の高値/安値を使用
    true_high[0] = high[0]
    true_low[0] = low[0]
    true_range[0] = high[0] - low[0]
    
    for i in range(1, length):
        # True High = Close[1] > High ? Close[1] : High
        if close[i-1] > high[i]:
            true_high[i] = close[i-1]
        else:
            true_high[i] = high[i]
        
        # True Low = Close[1] < Low ? Close[1] : Low
        if close[i-1] < low[i]:
            true_low[i] = close[i-1]
        else:
            true_low[i] = low[i]
        
        # True Range = True High - True Low
        true_range[i] = true_high[i] - true_low[i]
    
    return true_high, true_low, true_range


class STR(Indicator):
    """
    STR（Smooth True Range）インジケーター - 動的適応対応版
    
    John Ehlersの論文「ULTIMATE CHANNEL and ULTIMATE BANDS」に基づく実装：
    - True Rangeを計算してUltimate Smootherで平滑化
    - 従来のATRよりも低遅延を実現
    - Ultimate ChannelとUltimate Bandsの基礎となる指標
    - 動的適応期間対応（EhlersUnifiedDCによるサイクル検出）
    
    特徴:
    - 超低遅延: Ultimate Smootherによる最小限の遅延
    - 高精度: True Rangeの正確な計算
    - 適応性: 固定期間または動的期間対応
    - サイクル検出: 複数のEhlersサイクル検出器による期間自動調整
    """
    
    def __init__(
        self,
        period: float = 20.0,                 # STR期間
        src_type: str = 'ukf_hlc3',               # プライスソース（True Range計算には影響しないが一貫性のため）
        ukf_params: Optional[Dict] = None,     # UKFパラメータ（UKFソース使用時）
        # 動的適応パラメータ
        period_mode: str = 'dynamic',            # 期間モード ('fixed' or 'dynamic')
        # サイクル検出器パラメータ
        cycle_detector_type: str = 'absolute_ultimate',
        cycle_detector_cycle_part: float = 0.5,
        cycle_detector_max_cycle: int = 55,
        cycle_detector_min_cycle: int = 5,
        cycle_period_multiplier: float = 1.0,
        cycle_detector_period_range: Tuple[int, int] = (5, 120)
    ):
        """
        コンストラクタ
        
        Args:
            period: STR期間（デフォルト: 20.0）
            src_type: プライスソースタイプ（一貫性のため）
                基本ソース: 'close', 'hlc3', 'hl2', 'ohlc4', 'high', 'low', 'open'
                UKFソース: 'ukf', 'ukf_close', 'ukf_hlc3', 'ukf_hl2', 'ukf_ohlc4'
            ukf_params: UKFパラメータ（UKFソース使用時のオプション）
                alpha: UKFのalpha値（デフォルト: 0.001）
                beta: UKFのbeta値（デフォルト: 2.0）
                kappa: UKFのkappa値（デフォルト: 0.0）
                process_noise_scale: プロセスノイズスケール（デフォルト: 0.001）
                volatility_window: ボラティリティ計算ウィンドウ（デフォルト: 10）
                adaptive_noise: 適応ノイズの使用（デフォルト: True）
            
            # 動的適応パラメータ
            period_mode: 期間モード ('fixed' or 'dynamic')
            
            # サイクル検出器パラメータ
            cycle_detector_type: サイクル検出器タイプ ('hody', 'phac', 'dudi', etc.)
            cycle_detector_cycle_part: サイクル検出器のサイクル部分倍率（デフォルト: 1.0）
            cycle_detector_max_cycle: サイクル検出器の最大サイクル期間（デフォルト: 120）
            cycle_detector_min_cycle: サイクル検出器の最小サイクル期間（デフォルト: 5）
            cycle_period_multiplier: サイクル期間の乗数（デフォルト: 1.0）
            cycle_detector_period_range: サイクル検出器の周期範囲（デフォルト: (5, 120)）
        """
        # 指標名の作成
        indicator_name = f"STR(period={period}({period_mode}), {src_type}, cycle={cycle_detector_type})"
        super().__init__(indicator_name)
        
        # パラメータを保存
        self.period = period
        self.src_type = src_type.lower()
        self.ukf_params = ukf_params
        
        # 動的適応パラメータ
        self.period_mode = period_mode.lower()
        
        # サイクル検出器パラメータ
        self.cycle_detector_type = cycle_detector_type
        self.cycle_detector_cycle_part = cycle_detector_cycle_part
        self.cycle_detector_max_cycle = cycle_detector_max_cycle
        self.cycle_detector_min_cycle = cycle_detector_min_cycle
        self.cycle_period_multiplier = cycle_period_multiplier
        self.cycle_detector_period_range = cycle_detector_period_range
        
        # パラメータ検証
        if self.period <= 0:
            raise ValueError("periodは0より大きい必要があります")
        if self.period < 2:
            raise ValueError("periodは2以上である必要があります（フィルター安定性のため）")
        
        if self.period_mode not in ['fixed', 'dynamic']:
            raise ValueError(f"無効なperiod_mode: {self.period_mode}. 'fixed' または 'dynamic' を指定してください。")
        
        # ソースタイプの検証（PriceSourceから利用可能なタイプを取得）
        available_sources = PriceSource.get_available_sources()
        if self.src_type not in available_sources:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(available_sources.keys())}")
        
        # 動的適応が必要な場合のみEhlersUnifiedDCを初期化
        self.cycle_detector = None
        
        if self.period_mode == 'dynamic':
            try:
                # 循環インポートを回避するため、関数内でインポート
                try:
                    from .ehlers_unified_dc import EhlersUnifiedDC
                except ImportError:
                    from ehlers_unified_dc import EhlersUnifiedDC
                    
                self.cycle_detector = EhlersUnifiedDC(
                    detector_type=self.cycle_detector_type,
                    cycle_part=self.cycle_detector_cycle_part,
                    max_cycle=self.cycle_detector_max_cycle,
                    min_cycle=self.cycle_detector_min_cycle,
                    src_type=self.src_type,
                    period_range=self.cycle_detector_period_range
                )
                self.logger.info(f"動的適応サイクル検出器を初期化: {self.cycle_detector_type}")
            except Exception as e:
                self.logger.error(f"サイクル検出器の初期化に失敗: {e}")
                # フォールバックとして固定モードに変更
                self.period_mode = 'fixed'
                self.logger.warning("動的適応モードの初期化に失敗したため、固定モードにフォールバックしました。")
        
        # Ultimate Smootherの初期化（動的期間対応）
        self._ultimate_smoother = UltimateSmoother(
            period=self.period,
            src_type='close',
            ukf_params=self.ukf_params,
            period_mode=self.period_mode,
            cycle_detector_type=self.cycle_detector_type,
            cycle_detector_cycle_part=self.cycle_detector_cycle_part,
            cycle_detector_max_cycle=self.cycle_detector_max_cycle,
            cycle_detector_min_cycle=self.cycle_detector_min_cycle,
            cycle_period_multiplier=self.cycle_period_multiplier,
            cycle_detector_period_range=self.cycle_detector_period_range
        )
        
        # 結果キャッシュ（サイズ制限付き）
        self._result_cache = {}
        self._max_cache_size = 20
        self._cache_keys = []
    
    def _get_dynamic_periods(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        動的適応期間を計算する
        
        Args:
            data: 価格データ
            
        Returns:
            np.ndarray: 動的期間配列
        """
        data_length = len(data) if hasattr(data, '__len__') else 0
        
        # デフォルト値で初期化
        periods = np.full(data_length, self.period, dtype=np.float64)
        
        # 動的適応期間の計算
        if self.period_mode == 'dynamic' and self.cycle_detector is not None:
            try:
                # ドミナントサイクルを計算
                dominant_cycles = self.cycle_detector.calculate(data)
                
                if dominant_cycles is not None and len(dominant_cycles) == data_length:
                    # サイクル期間に乗数を適用
                    adjusted_cycles = dominant_cycles * self.cycle_period_multiplier
                    
                    # サイクル期間を適切な範囲にクリップ
                    periods = np.clip(adjusted_cycles, 
                                     self.cycle_detector_min_cycle, 
                                     self.cycle_detector_max_cycle)
                    
                    self.logger.debug(f"動的期間計算完了 - 期間範囲: [{np.min(periods):.1f}-{np.max(periods):.1f}]")
                else:
                    self.logger.warning("ドミナントサイクルの計算結果が無効です。固定期間を使用します。")
                    
            except Exception as e:
                self.logger.error(f"動的期間計算中にエラー: {e}")
                # エラー時は固定期間を使用
        
        return periods
    
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
            params_sig = (f"{self.period}_{self.period_mode}_{self.src_type}_{ukf_sig}_"
                         f"{self.cycle_detector_type}_{self.cycle_detector_cycle_part}_"
                         f"{self.cycle_detector_max_cycle}_{self.cycle_detector_min_cycle}_"
                         f"{self.cycle_period_multiplier}_{self.cycle_detector_period_range}")
            
            # 超高速ハッシュ
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.period}_{self.period_mode}_{self.src_type}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> STRResult:
        """
        STRを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
                必要なカラム: high, low, close
        
        Returns:
            STRResult: STRの値とTrue Range情報を含む結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return STRResult(
                    values=cached_result.values.copy(),
                    true_range=cached_result.true_range.copy(),
                    true_high=cached_result.true_high.copy(),
                    true_low=cached_result.true_low.copy()
                )
            
            # データの準備
            if isinstance(data, pd.DataFrame):
                required_cols = ['high', 'low', 'close']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(f"必要なカラムが不足しています: {missing_cols}")
                
                high = data['high'].to_numpy()
                low = data['low'].to_numpy()
                close = data['close'].to_numpy()
            else:
                if data.ndim != 2 or data.shape[1] < 4:
                    raise ValueError("NumPy配列は2次元で、少なくとも4列（OHLC）が必要です")
                high = data[:, 1]   # high
                low = data[:, 2]    # low  
                close = data[:, 3]  # close
            
            # NumPy配列に変換
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            close = np.asarray(close, dtype=np.float64)
            
            # データ長の検証
            data_length = len(close)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            # True Range値を計算
            true_high, true_low, true_range = calculate_true_range_values(high, low, close)
            
            # True RangeをDataFrame形式に変換してUltimate Smootherに渡す
            tr_df = pd.DataFrame({'close': true_range})
            
            # 動的期間のログ出力
            mode_info = f"モード:{self.period_mode}"
            if self.period_mode == 'dynamic':
                mode_info += f", サイクル検出器:{self.cycle_detector_type}"
            
            # Ultimate Smootherを適用してSTRを計算
            smoother_result = self._ultimate_smoother.calculate(tr_df)
            str_values = smoother_result.values
            
            # 結果の保存
            result = STRResult(
                values=str_values.copy(),
                true_range=true_range.copy(),
                true_high=true_high.copy(),
                true_low=true_low.copy()
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            self._values = str_values  # 基底クラスの要件
            
            self.logger.debug(f"STR 計算完了 - {mode_info}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"STR計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            return STRResult(
                values=np.array([]),
                true_range=np.array([]),
                true_high=np.array([]),
                true_low=np.array([])
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """STR値のみを取得する（後方互換性のため）"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.values.copy()
    
    def get_true_range(self) -> Optional[np.ndarray]:
        """True Range値を取得する"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.true_range.copy()
    
    def get_true_high(self) -> Optional[np.ndarray]:
        """True High値を取得する"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.true_high.copy()
    
    def get_true_low(self) -> Optional[np.ndarray]:
        """True Low値を取得する"""
        if not self._result_cache or not self._cache_keys:
            return None
        
        result = self._result_cache[self._cache_keys[-1]]
        return result.true_low.copy()
    
    def get_dynamic_periods_info(self) -> dict:
        """動的適応期間の情報を取得する"""
        info = {
            'period_mode': self.period_mode,
            'cycle_detector_available': self.cycle_detector is not None
        }
        
        # サイクル検出器の情報
        if self.cycle_detector is not None:
            info.update({
                'cycle_detector_type': self.cycle_detector_type,
                'cycle_detector_cycle_part': self.cycle_detector_cycle_part,
                'cycle_detector_max_cycle': self.cycle_detector_max_cycle,
                'cycle_detector_min_cycle': self.cycle_detector_min_cycle,
                'cycle_period_multiplier': self.cycle_period_multiplier,
                'cycle_detector_period_range': self.cycle_detector_period_range
            })
        
        return info

    def reset(self) -> None:
        """インディケーターの状態をリセットする"""
        super().reset()
        self._result_cache = {}
        self._cache_keys = []
        self._ultimate_smoother.reset()
        if self.cycle_detector is not None:
            self.cycle_detector.reset() 