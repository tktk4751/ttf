#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from numba import njit, prange

from ...base.signal_generator import BaseSignalGenerator
from signals.implementations.cosmic_universal_adaptive_channel.breakout_entry import CosmicUniversalAdaptiveChannelBreakoutEntrySignal


class CosmicUniversalSignalGenerator(BaseSignalGenerator):
    """
    宇宙統一適応チャネルのシグナル生成クラス（両方向・高速化版）
    
    特徴:
    - 量子統計熱力学エンジンによる動的適応性
    - フラクタル液体力学システムによる市場フロー解析
    - ヒルベルト・ウェーブレット多重解像度解析
    - 適応カオス理論センターライン
    - 宇宙統計エントロピーフィルター
    - 多次元ベイズ適応システム
    
    エントリー条件:
    - ロング: 宇宙統一適応チャネルのブレイクアウトで買いシグナル
    - ショート: 宇宙統一適応チャネルのブレイクアウトで売りシグナル
    
    エグジット条件:
    - ロング: 宇宙統一適応チャネルの売りシグナル
    - ショート: 宇宙統一適応チャネルの買いシグナル
    """
    
    def __init__(
        self,
        # 基本パラメータ
        channel_lookback: int = 1,
        
        # 宇宙チャネルのパラメータ
        quantum_window: int = 34,
        fractal_window: int = 21,
        chaos_window: int = 55,
        entropy_window: int = 21,
        bayesian_window: int = 34,
        base_multiplier: float = 2.0,
        src_type: str = 'hlc3',
        volume_src: str = 'volume'
    ):
        """初期化"""
        super().__init__("CosmicUniversalSignalGenerator")
        
        # パラメータの設定
        self._params = {
            # 基本パラメータ
            'channel_lookback': channel_lookback,
            
            # 宇宙チャネルのパラメータ
            'quantum_window': quantum_window,
            'fractal_window': fractal_window,
            'chaos_window': chaos_window,
            'entropy_window': entropy_window,
            'bayesian_window': bayesian_window,
            'base_multiplier': base_multiplier,
            'src_type': src_type,
            'volume_src': volume_src
        }
        
        # 宇宙統一適応チャネルブレイクアウトシグナルの初期化
        self.cosmic_channel_signal = CosmicUniversalAdaptiveChannelBreakoutEntrySignal(
            channel_lookback=channel_lookback,
            cosmic_channel_params={
                'quantum_window': quantum_window,
                'fractal_window': fractal_window,
                'chaos_window': chaos_window,
                'entropy_window': entropy_window,
                'bayesian_window': bayesian_window,
                'base_multiplier': base_multiplier,
                'src_type': src_type,
                'volume_src': volume_src
            }
        )
        
        # キャッシュ用の変数
        self._data_len = 0
        self._signals = None
        self._cosmic_channel_signals = None
    
    def calculate_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> None:
        """シグナル計算（高速化版）"""
        try:
            current_len = len(data)
            
            # データ長が変わった場合のみ再計算
            if self._signals is None or current_len != self._data_len:
                # データフレームの作成（必要な列のみ）
                if isinstance(data, pd.DataFrame):
                    df = data.copy()
                    # volume列が存在しない場合は1で埋める
                    if self._params['volume_src'] not in df.columns:
                        df[self._params['volume_src']] = 1.0
                else:
                    # NumPyの場合、OHLCとして解釈してvolume列を追加
                    if data.ndim == 2 and data.shape[1] >= 4:
                        df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                        if data.shape[1] > 4:
                            df[self._params['volume_src']] = data[:, 4]
                        else:
                            df[self._params['volume_src']] = 1.0
                    else:
                        df = pd.DataFrame({'close': data})
                        df['open'] = df['close']
                        df['high'] = df['close'] 
                        df['low'] = df['close']
                        df[self._params['volume_src']] = 1.0
                
                # 宇宙統一適応チャネルシグナルの計算
                try:
                    cosmic_channel_signals = self.cosmic_channel_signal.generate(df)
                    
                    # シンプルなシグナル
                    self._signals = cosmic_channel_signals
                    
                    # エグジット用のシグナルを事前計算
                    self._cosmic_channel_signals = cosmic_channel_signals
                except Exception as e:
                    self.logger.error(f"宇宙統一適応チャネルシグナル計算中にエラー: {str(e)}")
                    # エラー時はゼロシグナルを設定
                    self._signals = np.zeros(current_len, dtype=np.int8)
                    self._cosmic_channel_signals = np.zeros(current_len, dtype=np.int8)
                
                self._data_len = current_len
        except Exception as e:
            self.logger.error(f"calculate_signals全体でエラー: {str(e)}")
            # エラー時はゼロシグナルを設定
            if data is not None:
                self._signals = np.zeros(len(data), dtype=np.int8)
                self._cosmic_channel_signals = np.zeros(len(data), dtype=np.int8)
                self._data_len = len(data)
    
    def get_entry_signals(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """エントリーシグナル取得（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        return self._signals
    
    def get_exit_signals(self, data: Union[pd.DataFrame, np.ndarray], position: int, index: int = -1) -> bool:
        """エグジットシグナル生成（高速化版）"""
        if self._signals is None or len(data) != self._data_len:
            self.calculate_signals(data)
        
        if index == -1:
            index = len(data) - 1
        
        # キャッシュされたシグナルを使用
        if position == 1:  # ロングポジション
            return bool(self._cosmic_channel_signals[index] == -1)
        elif position == -1:  # ショートポジション
            return bool(self._cosmic_channel_signals[index] == 1)
        return False
    
    def get_channel_values(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        宇宙統一適応チャネルのチャネル値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (中心線, 上部チャネル, 下部チャネル)のタプル
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.cosmic_channel_signal.get_channel_values()
        except Exception as e:
            self.logger.error(f"チャネル値取得中にエラー: {str(e)}")
            # エラー時は空の配列を返す
            empty = np.array([])
            return empty, empty, empty
    
    def get_cosmic_intelligence_report(self, data: Union[pd.DataFrame, np.ndarray] = None) -> Dict:
        """
        宇宙知能レポートを取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            Dict: 宇宙知能レポート
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.cosmic_channel_signal.get_cosmic_intelligence_report()
        except Exception as e:
            self.logger.error(f"宇宙知能レポート取得中にエラー: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_quantum_entanglement(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        量子もつれ強度の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 量子もつれ強度の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.cosmic_channel_signal.get_quantum_entanglement()
        except Exception as e:
            self.logger.error(f"量子もつれ強度取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_fractal_dimension(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        フラクタル次元の値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: フラクタル次元の値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.cosmic_channel_signal.get_fractal_dimension()
        except Exception as e:
            self.logger.error(f"フラクタル次元取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_cosmic_phase(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        宇宙フェーズの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 宇宙フェーズの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.cosmic_channel_signal.get_cosmic_phase()
        except Exception as e:
            self.logger.error(f"宇宙フェーズ取得中にエラー: {str(e)}")
            return np.array([])
    
    def get_omniscient_confidence(self, data: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        全知信頼度スコアの値を取得
        
        Args:
            data: オプションの価格データ。指定された場合は計算を実行します。
            
        Returns:
            np.ndarray: 全知信頼度スコアの値
        """
        try:
            if data is not None:
                self.calculate_signals(data)
                
            return self.cosmic_channel_signal.get_omniscient_confidence()
        except Exception as e:
            self.logger.error(f"全知信頼度スコア取得中にエラー: {str(e)}")
            return np.array([]) 