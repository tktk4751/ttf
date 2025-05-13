#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple, Optional
import numpy as np
import pandas as pd
from numba import njit, prange
import traceback

# --- 依存関係のインポート ---
try:
    from strategies.base.signal_generator import BaseSignalGenerator
    from signals.implementations.x_channel.breakout_entry import XChannelBreakoutEntrySignal # XChannelシグナルをインポート
    from indicators.x_channel import XChannel # インジケータ自体も参照する場合
except ImportError:
    # フォールバック
    print("Warning: Could not import from relative path. Assuming base classes/functions are available.")
    class BaseSignalGenerator:
        def __init__(self, name): self.name = name; self.logger = self._get_logger()
        def _get_logger(self): import logging; return logging.getLogger(self.__class__.__name__)
    class XChannelBreakoutEntrySignal: # Dummy
        def __init__(self, **kwargs): pass
        def generate(self, data): return type('obj', (object,), {'buy': np.zeros(len(data)), 'sell': np.zeros(len(data))})()
        def get_bands(self): return np.array([]), np.array([]), np.array([])
        def get_trigger_values(self): return np.array([])
        def get_dynamic_multiplier(self): return np.array([])
        def get_catr(self): return np.array([])
        def reset(self): pass
    class XChannel: # Dummy
         def __init__(self, **kwargs): pass
         def get_bands(self): return np.array([]), np.array([]), np.array([])
         def get_trigger_values(self): return np.array([])
         def get_dynamic_multiplier(self): return np.array([])
         def get_catr(self): return np.array([])
         def reset(self): pass


class XCSimpleSignalGenerator(BaseSignalGenerator):
    """
    Xチャネルのシグナル生成クラス（シンプル版）

    エントリー条件:
    - ロング: Xチャネルのブレイクアウトで買いシグナル
    - ショート: Xチャネルのブレイクアウトで売りシグナル

    エグジット条件:
    - ロング: Xチャネルの売りシグナル
    - ショート: Xチャネルの買いシグナル
    """

    def __init__(
        self,
        # XChannelBreakoutEntrySignalに必要なパラメータを渡す
        x_channel_params: Dict[str, Any], # ネストされたXChannelパラメータ
        use_close_confirmation: bool = True,
        # ... (他のパラメータがあれば追加)
    ):
        """初期化"""
        super().__init__("XCSimpleSignalGenerator")

        # パラメータを保存 (戦略クラスでの参照用)
        self._params = {
            'x_channel_params': x_channel_params,
            'use_close_confirmation': use_close_confirmation,
        }

        # Xチャネルブレイクアウトシグナルの初期化
        # XChannelBreakoutEntrySignalに、x_channel_paramsとして完全なXChannelパラメータを渡す
        self.x_channel_signal = XChannelBreakoutEntrySignal(
            x_channel_params=x_channel_params,
            use_close_confirmation=use_close_confirmation,
        )

        # キャッシュ用の変数
        self._data_len = 0
        self._signals: Optional[np.ndarray] = None # Combined signals (1=buy, -1=sell, 0=none)
        self._data_hash: Optional[str] = None

    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データとジェネレータのパラメータに基づいてハッシュ値を計算する。"""
        # XChannelBreakoutEntrySignalのハッシュ計算ロジックを借用または参照
        return self.x_channel_signal._get_data_hash(data)

    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（内部キャッシュ用）"""
        current_len = len(data) if hasattr(data, '__len__') else 0
        if current_len == 0: return # データが空なら何もしない

        data_hash = self._get_data_hash(data)

        # データ長またはハッシュが変わった場合のみ再計算
        if self._signals is None or current_len != self._data_len or data_hash != self._data_hash:
            try:
                # XChannelBreakoutEntrySignal の public generate メソッドを呼び出す
                signals = self.x_channel_signal.generate(data)
                self._signals = signals
                self._data_len = current_len
                self._data_hash = data_hash
            except Exception as e:
                self.logger.error(f"シグナル計算中にエラー: {e}", exc_info=True)
                # エラー時はゼロシグナルを設定
                self._signals = np.zeros(current_len, dtype=np.int8)
                self._data_len = current_len
                self._data_hash = None # エラー時はハッシュをクリア

    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        エントリーシグナルを取得（1: 買い, -1: 売り, 0: なし）
        """
        self.calculate_signals(data) # 必要なら内部で計算/キャッシュ更新
        if self._signals is None:
             return np.zeros(len(data), dtype=np.int8) # 計算失敗時

        return self._signals.copy() # generate already returns the combined signal

    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """
        エグジットシグナル生成
        - ロングポジションの場合、売りシグナルが出たらエグジット
        - ショートポジションの場合、買いシグナルが出たらエグジット
        """
        self.calculate_signals(data) # 必要なら内部で計算/キャッシュ更新

        if self._signals is None:
            return False # 計算失敗時

        if index == -1:
            index = self._data_len - 1

        if 0 <= index < self._data_len:
            if position == 1:  # ロングポジション
                # Exit long if sell signal (-1) occurs
                return bool(self._signals[index] == -1)
            elif position == -1:  # ショートポジション
                # Exit short if buy signal (1) occurs
                return bool(self._signals[index] == 1)
        return False

    # --- ヘルパーメソッド (XChannelの情報を取得) ---
    #     これらのメソッドは、内部の x_channel_signal を通じて
    #     さらにその内部の x_channel インジケータの情報を取得する

    def get_indicator_instance(self) -> Optional[XChannel]:
        """内部で使用しているXChannelインジケータのインスタンスを取得する"""
        if hasattr(self.x_channel_signal, 'x_channel'):
            return self.x_channel_signal.x_channel
        return None

    def get_band_values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Xチャネルのバンド値を取得"""
        indicator = self.get_indicator_instance()
        if indicator and hasattr(indicator, 'get_bands') and indicator._result is not None:
            return indicator.get_bands()
        return np.array([]), np.array([]), np.array([])

    def get_trigger_values(self) -> np.ndarray:
        """乗数計算に使用されたトリガー値を取得"""
        indicator = self.get_indicator_instance()
        if indicator and hasattr(indicator, 'get_trigger_values') and indicator._result is not None:
            return indicator.get_trigger_values()
        return np.array([])

    def get_dynamic_multiplier(self) -> np.ndarray:
        """動的ATR乗数の値を取得"""
        indicator = self.get_indicator_instance()
        if indicator and hasattr(indicator, 'get_dynamic_multiplier') and indicator._result is not None:
            return indicator.get_dynamic_multiplier()
        return np.array([])

    def get_catr(self) -> np.ndarray:
        """CATR値（金額ベース）を取得"""
        indicator = self.get_indicator_instance()
        if indicator and hasattr(indicator, 'get_catr') and indicator._result is not None:
            return indicator.get_catr()
        return np.array([])

    def reset(self):
        """シグナルジェネレータの状態をリセット"""
        super().reset()
        self._data_len = 0
        self._signals = None
        self._data_hash = None
        if hasattr(self.x_channel_signal, 'reset'):
            self.x_channel_signal.reset()
        self.logger.debug(f"Signal Generator '{self.name}' reset.") 