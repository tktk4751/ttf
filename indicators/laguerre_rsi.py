#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Union, Optional, Dict, Any
import numpy as np
import pandas as pd
from numba import njit, float64

from .indicator import Indicator
from .price_source import PriceSource
from .smoother.roofing_filter import RoofingFilter


@dataclass
class LaguerreRSIResult:
    """ラゲールRSIの計算結果"""
    values: np.ndarray           # ラゲールRSI値（0-1の範囲）
    l0_values: np.ndarray        # L0値（ラゲール係数）
    l1_values: np.ndarray        # L1値（ラゲール係数）
    l2_values: np.ndarray        # L2値（ラゲール係数）
    l3_values: np.ndarray        # L3値（ラゲール係数）
    cu_values: np.ndarray        # CU値（上昇の累積）
    cd_values: np.ndarray        # CD値（下降の累積）
    roofing_values: np.ndarray   # ルーフィングフィルター値（オプション）
    filtered_source: np.ndarray  # フィルタリング済みソース価格


@njit(fastmath=True, cache=True)
def calculate_laguerre_filter_core(prices: np.ndarray, gamma: float) -> tuple:
    """
    ラゲールフィルターを計算する（Numba最適化版）
    
    Args:
        prices: 価格配列
        gamma: ガンマパラメータ（-0.1から0.9の範囲）
    
    Returns:
        tuple: (L0, L1, L2, L3) ラゲール係数の配列
    """
    length = len(prices)
    
    # ラゲール係数の初期化
    l0 = np.full(length, np.nan, dtype=np.float64)
    l1 = np.full(length, np.nan, dtype=np.float64)
    l2 = np.full(length, np.nan, dtype=np.float64)
    l3 = np.full(length, np.nan, dtype=np.float64)
    
    # 初期値設定
    if length > 0 and not np.isnan(prices[0]):
        l0[0] = prices[0]
        l1[0] = prices[0]
        l2[0] = prices[0]
        l3[0] = prices[0]
    
    # ラゲールフィルターの計算
    for i in range(1, length):
        if np.isnan(prices[i]):
            # NaN値の場合は前の値をそのまま使用
            if i > 0:
                l0[i] = l0[i-1] if not np.isnan(l0[i-1]) else prices[i]
                l1[i] = l1[i-1] if not np.isnan(l1[i-1]) else l0[i]
                l2[i] = l2[i-1] if not np.isnan(l2[i-1]) else l1[i]
                l3[i] = l3[i-1] if not np.isnan(l3[i-1]) else l2[i]
            continue
        
        # L0の計算: L0 = (1-gamma) * close + gamma * L0[1]
        prev_l0 = l0[i-1] if not np.isnan(l0[i-1]) else prices[i]
        l0[i] = (1 - gamma) * prices[i] + gamma * prev_l0
        
        # L1の計算: L1 = -gamma * L0 + L0[1] + gamma * L1[1]
        prev_l1 = l1[i-1] if not np.isnan(l1[i-1]) else l0[i]
        l1[i] = -gamma * l0[i] + prev_l0 + gamma * prev_l1
        
        # L2の計算: L2 = -gamma * L1 + L1[1] + gamma * L2[1]
        prev_l2 = l2[i-1] if not np.isnan(l2[i-1]) else l1[i]
        l2[i] = -gamma * l1[i] + prev_l1 + gamma * prev_l2
        
        # L3の計算: L3 = -gamma * L2 + L2[1] + gamma * L3[1]
        prev_l3 = l3[i-1] if not np.isnan(l3[i-1]) else l2[i]
        l3[i] = -gamma * l2[i] + prev_l2 + gamma * prev_l3
    
    return l0, l1, l2, l3


@njit(fastmath=True, cache=True)
def calculate_laguerre_rsi_core(l0: np.ndarray, l1: np.ndarray, l2: np.ndarray, l3: np.ndarray) -> tuple:
    """
    ラゲールRSIを計算する（Numba最適化版）
    
    Args:
        l0: L0値の配列
        l1: L1値の配列
        l2: L2値の配列
        l3: L3値の配列
    
    Returns:
        tuple: (RSI値, CU値, CD値) の配列
    """
    length = len(l0)
    rsi_values = np.full(length, np.nan, dtype=np.float64)
    cu_values = np.full(length, np.nan, dtype=np.float64)
    cd_values = np.full(length, np.nan, dtype=np.float64)
    
    for i in range(length):
        if (np.isnan(l0[i]) or np.isnan(l1[i]) or 
            np.isnan(l2[i]) or np.isnan(l3[i])):
            continue
        
        # CU（上昇の累積）の計算
        cu = 0.0
        if l0[i] >= l1[i]:
            cu += l0[i] - l1[i]
        if l1[i] >= l2[i]:
            cu += l1[i] - l2[i]
        if l2[i] >= l3[i]:
            cu += l2[i] - l3[i]
        
        # CD（下降の累積）の計算
        cd = 0.0
        if l0[i] < l1[i]:
            cd += l1[i] - l0[i]
        if l1[i] < l2[i]:
            cd += l2[i] - l1[i]
        if l2[i] < l3[i]:
            cd += l3[i] - l2[i]
        
        cu_values[i] = cu
        cd_values[i] = cd
        
        # RSIの計算
        total = cu + cd
        if total != 0.0:
            rsi_values[i] = cu / total
        else:
            rsi_values[i] = 0.0
    
    return rsi_values, cu_values, cd_values


class LaguerreRSI(Indicator):
    """
    ラゲールRSI（Laguerre-based RSI）インジケーター
    
    John Ehlers's Laguerre transform filterを使用したRSI指標。
    従来のRSIよりも価格変動に敏感で、短期間のデータでも効果的なシグナルを生成できる。
    
    特徴:
    - ラゲール変換フィルターによる高度な価格変動検出
    - 従来のRSIより感度が高い
    - 短期間のデータでも有効
    - オプションでルーフィングフィルターによる前処理
    
    計算フロー:
    1. ソース価格データを取得
    2. ルーフィングフィルターを適用（オプション）
    3. ラゲールフィルター（L0, L1, L2, L3）を計算
    4. CU（上昇）、CD（下降）を計算
    5. RSI = CU / (CU + CD)
    """
    
    def __init__(
        self,
        gamma: float = 0.9,                      # ガンマパラメータ
        src_type: str = 'oc2',                 # ソースタイプ
        # ルーフィングフィルターパラメータ（オプション）
        use_roofing_filter: bool = True,        # ルーフィングフィルターを使用するか
        roofing_hp_cutoff: float = 48.0,         # ルーフィングフィルターのHighPassカットオフ
        roofing_ss_band_edge: float = 10.0       # ルーフィングフィルターのSuperSmootherバンドエッジ
    ):
        """
        コンストラクタ
        
        Args:
            gamma: ガンマパラメータ（-0.1から0.9の範囲、デフォルト: 0.5）
            src_type: ソースタイプ（デフォルト: 'close'）
            use_roofing_filter: ルーフィングフィルターを使用するか（デフォルト: False）
            roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ（デフォルト: 48.0）
            roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ（デフォルト: 10.0）
        """
        indicator_name = f"LaguerreRSI(gamma={gamma}, {src_type}"
        if use_roofing_filter:
            indicator_name += f", roofing(hp={roofing_hp_cutoff}, ss={roofing_ss_band_edge})"
        indicator_name += ")"
        
        super().__init__(indicator_name)
        
        # パラメータの検証
        if not -0.1 <= gamma <= 0.999:
            raise ValueError("gammaは-0.1から0.999の範囲である必要があります")
        if src_type not in self.SRC_TYPES:
            raise ValueError(f"無効なソースタイプです: {src_type}。有効なオプション: {', '.join(self.SRC_TYPES)}")
        
        # パラメータの保存
        self.gamma = gamma
        self.src_type = src_type
        self.use_roofing_filter = use_roofing_filter
        self.roofing_hp_cutoff = roofing_hp_cutoff
        self.roofing_ss_band_edge = roofing_ss_band_edge
        
        # ルーフィングフィルターの初期化（オプション）
        self.roofing_filter = None
        if self.use_roofing_filter:
            try:
                self.roofing_filter = RoofingFilter(
                    src_type=self.src_type,
                    hp_cutoff=self.roofing_hp_cutoff,
                    ss_band_edge=self.roofing_ss_band_edge
                )
                self.logger.info(f"ルーフィングフィルターを初期化しました: hp={self.roofing_hp_cutoff}, ss={self.roofing_ss_band_edge}")
            except Exception as e:
                self.logger.error(f"ルーフィングフィルターの初期化に失敗: {e}")
                self.use_roofing_filter = False
                self.logger.warning("ルーフィングフィルター機能を無効にしました")
        
        # 結果キャッシュ
        self._result_cache = {}
        self._max_cache_size = 10
        self._cache_keys = []
    
    def _get_data_hash(self, data: Union[pd.DataFrame, np.ndarray]) -> str:
        """データのハッシュ値を計算してキャッシュに使用する"""
        try:
            if isinstance(data, pd.DataFrame):
                length = len(data)
                if length > 0:
                    first_val = float(data.iloc[0].get('close', data.iloc[0, -1]))
                    last_val = float(data.iloc[-1].get('close', data.iloc[-1, -1]))
                else:
                    first_val = last_val = 0.0
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
            params_sig = f"{self.gamma}_{self.src_type}_{self.use_roofing_filter}_{self.roofing_hp_cutoff}_{self.roofing_ss_band_edge}"
            
            # ハッシュ計算
            data_sig = (length, first_val, last_val)
            return f"{hash(data_sig)}_{hash(params_sig)}"
            
        except Exception:
            # フォールバック
            return f"{id(data)}_{self.gamma}_{self.src_type}_{self.use_roofing_filter}"
    
    def calculate(self, data: Union[pd.DataFrame, np.ndarray]) -> LaguerreRSIResult:
        """
        ラゲールRSIを計算する
        
        Args:
            data: 価格データ（DataFrameまたはNumPy配列）
        
        Returns:
            LaguerreRSIResult: ラゲールRSIの計算結果
        """
        try:
            # キャッシュチェック
            data_hash = self._get_data_hash(data)
            
            if data_hash in self._result_cache:
                # キャッシュヒット
                if data_hash in self._cache_keys:
                    self._cache_keys.remove(data_hash)
                self._cache_keys.append(data_hash)
                cached_result = self._result_cache[data_hash]
                return LaguerreRSIResult(
                    values=cached_result.values.copy(),
                    l0_values=cached_result.l0_values.copy(),
                    l1_values=cached_result.l1_values.copy(),
                    l2_values=cached_result.l2_values.copy(),
                    l3_values=cached_result.l3_values.copy(),
                    cu_values=cached_result.cu_values.copy(),
                    cd_values=cached_result.cd_values.copy(),
                    roofing_values=cached_result.roofing_values.copy(),
                    filtered_source=cached_result.filtered_source.copy()
                )
            
            # データの検証
            data_length = len(data)
            if data_length == 0:
                raise ValueError("入力データが空です")
            
            # 1. ソース価格データを取得
            source_prices = PriceSource.calculate_source(data, self.src_type)
            
            # NumPy配列に変換（float64型で統一）
            if not isinstance(source_prices, np.ndarray):
                source_prices = np.array(source_prices)
            if source_prices.dtype != np.float64:
                source_prices = source_prices.astype(np.float64)
            
            # 2. ルーフィングフィルターによる前処理（オプション）
            filtered_source = source_prices
            roofing_values = np.full_like(source_prices, np.nan)
            
            if self.use_roofing_filter and self.roofing_filter is not None:
                try:
                    roofing_result = self.roofing_filter.calculate(data)
                    roofing_values = roofing_result.values
                    
                    # ルーフィングフィルターの結果を使用
                    # 有効値が十分にある場合のみフィルターを適用
                    valid_roofing = np.sum(~np.isnan(roofing_values))
                    if valid_roofing > len(roofing_values) * 0.3:  # 有効値が30%以上の場合
                        filtered_source = roofing_values
                    else:
                        self.logger.debug("ルーフィングフィルターの有効値が不十分。元の価格を使用します")
                        filtered_source = source_prices
                        
                except Exception as e:
                    self.logger.warning(f"ルーフィングフィルター処理中にエラー: {e}。元の値を使用します。")
                    filtered_source = source_prices
                    roofing_values = np.full_like(source_prices, np.nan)
            
            # NumPy配列として確保
            if not isinstance(filtered_source, np.ndarray):
                filtered_source = np.array(filtered_source)
            if filtered_source.dtype != np.float64:
                filtered_source = filtered_source.astype(np.float64)
            
            # 3. ラゲールフィルター（L0, L1, L2, L3）を計算
            l0, l1, l2, l3 = calculate_laguerre_filter_core(filtered_source, self.gamma)
            
            # 4. ラゲールRSIを計算
            rsi_values, cu_values, cd_values = calculate_laguerre_rsi_core(l0, l1, l2, l3)
            
            # 結果の作成
            result = LaguerreRSIResult(
                values=rsi_values.copy(),
                l0_values=l0.copy(),
                l1_values=l1.copy(),
                l2_values=l2.copy(),
                l3_values=l3.copy(),
                cu_values=cu_values.copy(),
                cd_values=cd_values.copy(),
                roofing_values=roofing_values.copy(),
                filtered_source=filtered_source.copy()
            )
            
            # キャッシュ管理
            if len(self._result_cache) >= self._max_cache_size and self._cache_keys:
                oldest_key = self._cache_keys.pop(0)
                if oldest_key in self._result_cache:
                    del self._result_cache[oldest_key]
            
            self._result_cache[data_hash] = result
            self._cache_keys.append(data_hash)
            
            # 基底クラス用の値設定
            self._values = rsi_values
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.logger.error(f"ラゲールRSI計算中にエラー: {error_msg}\n{stack_trace}")
            
            # エラー時は空の結果を返す
            empty_array = np.array([])
            return LaguerreRSIResult(
                values=empty_array,
                l0_values=empty_array,
                l1_values=empty_array,
                l2_values=empty_array,
                l3_values=empty_array,
                cu_values=empty_array,
                cd_values=empty_array,
                roofing_values=empty_array,
                filtered_source=empty_array
            )
    
    def get_values(self) -> Optional[np.ndarray]:
        """ラゲールRSI値を取得（後方互換性のため）"""
        if not self._result_cache:
            return None
        
        result = self._get_latest_result()
        return result.values.copy() if result else None
    
    def get_l0_values(self) -> Optional[np.ndarray]:
        """L0値を取得"""
        result = self._get_latest_result()
        return result.l0_values.copy() if result else None
    
    def get_l1_values(self) -> Optional[np.ndarray]:
        """L1値を取得"""
        result = self._get_latest_result()
        return result.l1_values.copy() if result else None
    
    def get_l2_values(self) -> Optional[np.ndarray]:
        """L2値を取得"""
        result = self._get_latest_result()
        return result.l2_values.copy() if result else None
    
    def get_l3_values(self) -> Optional[np.ndarray]:
        """L3値を取得"""
        result = self._get_latest_result()
        return result.l3_values.copy() if result else None
    
    def get_cu_values(self) -> Optional[np.ndarray]:
        """CU値（上昇累積）を取得"""
        result = self._get_latest_result()
        return result.cu_values.copy() if result else None
    
    def get_cd_values(self) -> Optional[np.ndarray]:
        """CD値（下降累積）を取得"""
        result = self._get_latest_result()
        return result.cd_values.copy() if result else None
    
    def get_roofing_values(self) -> Optional[np.ndarray]:
        """ルーフィングフィルター値を取得"""
        result = self._get_latest_result()
        return result.roofing_values.copy() if result else None
    
    def get_filtered_source(self) -> Optional[np.ndarray]:
        """フィルタリング済みソース価格を取得"""
        result = self._get_latest_result()
        return result.filtered_source.copy() if result else None
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """インジケーター情報を取得"""
        return {
            'name': self.name,
            'gamma': self.gamma,
            'src_type': self.src_type,
            'use_roofing_filter': self.use_roofing_filter,
            'roofing_hp_cutoff': self.roofing_hp_cutoff if self.use_roofing_filter else None,
            'roofing_ss_band_edge': self.roofing_ss_band_edge if self.use_roofing_filter else None,
            'description': 'ラゲール変換フィルターベースのRSI（価格変動に高感度、短期データ有効、オプションルーフィングフィルター対応）'
        }
    
    def _get_latest_result(self) -> Optional[LaguerreRSIResult]:
        """最新の結果を取得"""
        if not self._result_cache:
            return None
        
        if self._cache_keys:
            return self._result_cache[self._cache_keys[-1]]
        else:
            return next(iter(self._result_cache.values()))
    
    def reset(self) -> None:
        """インディケーターの状態をリセット"""
        super().reset()
        if self.roofing_filter:
            self.roofing_filter.reset()
        self._result_cache = {}
        self._cache_keys = []


# 便利関数
def calculate_laguerre_rsi(
    data: Union[pd.DataFrame, np.ndarray],
    gamma: float = 0.5,
    src_type: str = 'close',
    use_roofing_filter: bool = False,
    roofing_hp_cutoff: float = 48.0,
    roofing_ss_band_edge: float = 10.0
) -> np.ndarray:
    """
    ラゲールRSIの計算（便利関数）
    
    Args:
        data: 価格データ
        gamma: ガンマパラメータ
        src_type: ソースタイプ
        use_roofing_filter: ルーフィングフィルターを使用するか
        roofing_hp_cutoff: ルーフィングフィルターのHighPassカットオフ
        roofing_ss_band_edge: ルーフィングフィルターのSuperSmootherバンドエッジ
        
    Returns:
        ラゲールRSI値
    """
    indicator = LaguerreRSI(
        gamma=gamma,
        src_type=src_type,
        use_roofing_filter=use_roofing_filter,
        roofing_hp_cutoff=roofing_hp_cutoff,
        roofing_ss_band_edge=roofing_ss_band_edge
    )
    result = indicator.calculate(data)
    return result.values


if __name__ == "__main__":
    """直接実行時のテスト"""
    import numpy as np
    import pandas as pd
    
    print("=== ラゲールRSI インジケーターのテスト ===")
    
    # テストデータ生成
    np.random.seed(42)
    length = 200
    base_price = 100.0
    
    # トレンドとレンジが混在するデータを生成
    prices = [base_price]
    for i in range(1, length):
        if i < 50:  # 上昇トレンド
            change = 0.002 + np.random.normal(0, 0.008)
        elif i < 100:  # レンジ相場
            change = np.random.normal(0, 0.010)
        elif i < 150:  # 下降トレンド
            change = -0.002 + np.random.normal(0, 0.008)
        else:  # 再びレンジ相場
            change = np.random.normal(0, 0.010)
        
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
    
    # 基本版ラゲールRSIをテスト
    print("\n基本版ラゲールRSIをテスト中...")
    laguerre_rsi = LaguerreRSI(gamma=0.5, src_type='close')
    result = laguerre_rsi.calculate(df)
    
    valid_count = np.sum(~np.isnan(result.values))
    mean_rsi = np.nanmean(result.values)
    overbought_ratio = np.sum(result.values > 0.8) / valid_count if valid_count > 0 else 0
    oversold_ratio = np.sum(result.values < 0.2) / valid_count if valid_count > 0 else 0
    
    print(f"  有効値数: {valid_count}/{len(df)}")
    print(f"  平均RSI: {mean_rsi:.4f}")
    print(f"  買われすぎ比率 (>0.8): {overbought_ratio:.2%}")
    print(f"  売られすぎ比率 (<0.2): {oversold_ratio:.2%}")
    
    # ルーフィングフィルター版をテスト
    print("\nルーフィングフィルター版ラゲールRSIをテスト中...")
    laguerre_rsi_roofing = LaguerreRSI(
        gamma=0.5,
        src_type='close',
        use_roofing_filter=True,
        roofing_hp_cutoff=48.0,
        roofing_ss_band_edge=10.0
    )
    result_roofing = laguerre_rsi_roofing.calculate(df)
    
    valid_count_roofing = np.sum(~np.isnan(result_roofing.values))
    mean_rsi_roofing = np.nanmean(result_roofing.values)
    
    print(f"  有効値数: {valid_count_roofing}/{len(df)}")
    print(f"  平均RSI（ルーフィングフィルター版）: {mean_rsi_roofing:.4f}")
    
    # 比較統計
    if valid_count > 0 and valid_count_roofing > 0:
        correlation = np.corrcoef(
            result.values[~np.isnan(result.values)][-min(valid_count, valid_count_roofing):],
            result_roofing.values[~np.isnan(result_roofing.values)][-min(valid_count, valid_count_roofing):]
        )[0, 1]
        print(f"  基本版とルーフィング版の相関: {correlation:.4f}")
    
    # 異なるガンマ値でのテスト
    print("\n異なるガンマ値でのテスト...")
    for gamma_val in [0.2, 0.7]:
        test_rsi = LaguerreRSI(gamma=gamma_val, src_type='close')
        test_result = test_rsi.calculate(df)
        test_valid = np.sum(~np.isnan(test_result.values))
        test_mean = np.nanmean(test_result.values)
        print(f"  gamma={gamma_val}: 有効値数={test_valid}, 平均RSI={test_mean:.4f}")
    
    print("\n=== テスト完了 ===")