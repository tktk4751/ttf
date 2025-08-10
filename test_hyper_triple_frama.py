#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HyperTripleFRAMAのシンプルテスト
"""

import pandas as pd
import numpy as np
from indicators.hyper_triple_frama import HyperTripleFRAMA

def test_hyper_triple_frama():
    """HyperTripleFRAMAの基本的なテスト"""
    print("=== HyperTripleFRAMA基本テスト ===")
    
    # テストデータ生成（SOLUSTDの実際の価格に似せたデータ）
    np.random.seed(42)
    data_length = 100
    base_price = 100.0
    
    # OHLCデータを生成
    prices = []
    for i in range(data_length):
        # トレンドとノイズを加えた価格生成
        trend = i * 0.1  # 上昇トレンド
        noise = np.random.normal(0, 2.0)  # ノイズ
        price = base_price + trend + noise
        
        # OHLC生成
        high = price + abs(np.random.normal(0, 1.0))
        low = price - abs(np.random.normal(0, 1.0))
        open_price = price + np.random.normal(0, 0.5)
        close_price = price + np.random.normal(0, 0.5)
        
        prices.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': 1000 + np.random.randint(0, 500)
        })
    
    df = pd.DataFrame(prices)
    print(f"テストデータ生成完了: {len(df)}行")
    print(f"価格範囲: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # HyperTripleFRAMAインスタンス作成
    print("\n=== HyperTripleFRAMAインスタンス作成 ===")
    indicator = HyperTripleFRAMA(
        period=16,
        src_type='hl2',
        fc=1,
        sc=198,
        alpha_multiplier1=1.0,   # 1本目（通常FRAMA）
        alpha_multiplier2=0.5,   # 2本目（デフォルト）
        alpha_multiplier3=0.1,   # 3本目（デフォルト）
        enable_indicator_adaptation=False,  # 基本テストでは無効
        smoothing_mode='none'     # 基本テストでは平滑化なし
    )
    print(f"指標名: {indicator.name}")
    
    # 計算実行
    print("\n=== 計算実行 ===")
    try:
        result = indicator.calculate(df)
        print("計算成功!")
        
        # 結果の確認
        print(f"\n=== 結果確認 ===")
        print(f"1本目FRAMA値数: {len(result.frama_values)}")
        print(f"2本目FRAMA値数: {len(result.second_frama_values)}")
        print(f"3本目FRAMA値数: {len(result.third_frama_values)}")
        print(f"フラクタル次元数: {len(result.fractal_dimension)}")
        print(f"1本目アルファ値数: {len(result.alpha)}")
        print(f"2本目アルファ値数: {len(result.second_alpha)}")
        print(f"3本目アルファ値数: {len(result.third_alpha)}")
        print(f"平滑化方法: {result.smoothing_applied}")
        
        # 有効な値の確認（最後の10個）
        valid_indices = ~np.isnan(result.frama_values)
        if np.any(valid_indices):
            valid_count = np.sum(valid_indices)
            print(f"有効な値の数: {valid_count}")
            
            # 最後の有効な値を表示
            last_valid_idx = np.where(valid_indices)[0][-1]
            print(f"\n=== 最後の有効な値（インデックス {last_valid_idx}） ===")
            print(f"価格: {df['close'].iloc[last_valid_idx]:.4f}")
            print(f"1本目FRAMA: {result.frama_values[last_valid_idx]:.4f}")
            print(f"2本目FRAMA: {result.second_frama_values[last_valid_idx]:.4f}")
            print(f"3本目FRAMA: {result.third_frama_values[last_valid_idx]:.4f}")
            print(f"フラクタル次元: {result.fractal_dimension[last_valid_idx]:.4f}")
            print(f"1本目アルファ: {result.alpha[last_valid_idx]:.4f}")
            print(f"2本目アルファ: {result.second_alpha[last_valid_idx]:.4f}")
            print(f"3本目アルファ: {result.third_alpha[last_valid_idx]:.4f}")
            
            # 最後の5個の値を比較
            print(f"\n=== 最後の5個の値の比較 ===")
            end_idx = len(result.frama_values)
            start_idx = max(0, end_idx - 5)
            
            for i in range(start_idx, end_idx):
                if not np.isnan(result.frama_values[i]):
                    print(f"[{i:2d}] 価格:{df['close'].iloc[i]:7.2f} | " + 
                          f"FRAMA1:{result.frama_values[i]:7.2f} | " +
                          f"FRAMA2:{result.second_frama_values[i]:7.2f} | " +
                          f"FRAMA3:{result.third_frama_values[i]:7.2f}")
        else:
            print("有効な値が見つかりません")
            
        # getterメソッドのテスト
        print(f"\n=== getterメソッドテスト ===")
        frama1 = indicator.get_frama_values()
        frama2 = indicator.get_second_frama_values()
        frama3 = indicator.get_third_frama_values()
        fractal_dim = indicator.get_fractal_dimension()
        alpha1 = indicator.get_alpha()
        alpha2 = indicator.get_second_alpha()
        alpha3 = indicator.get_third_alpha()
        
        print(f"get_frama_values(): {len(frama1) if frama1 is not None else 'None'}")
        print(f"get_second_frama_values(): {len(frama2) if frama2 is not None else 'None'}")
        print(f"get_third_frama_values(): {len(frama3) if frama3 is not None else 'None'}")
        print(f"get_fractal_dimension(): {len(fractal_dim) if fractal_dim is not None else 'None'}")
        print(f"get_alpha(): {len(alpha1) if alpha1 is not None else 'None'}")
        print(f"get_second_alpha(): {len(alpha2) if alpha2 is not None else 'None'}")
        print(f"get_third_alpha(): {len(alpha3) if alpha3 is not None else 'None'}")
        
        print("\n=== HyperTripleFRAMA基本テスト完了 ===")
        return True
        
    except Exception as e:
        print(f"計算エラー: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hyper_triple_frama_params():
    """異なるパラメータでのテスト"""
    print("\n=== パラメータ別テスト ===")
    
    # 簡単なテストデータ
    np.random.seed(123)
    data_length = 50
    
    prices = []
    for i in range(data_length):
        price = 100 + i * 0.5 + np.random.normal(0, 1.0)
        prices.append({
            'open': price,
            'high': price + 1,
            'low': price - 1,
            'close': price,
            'volume': 1000
        })
    
    df = pd.DataFrame(prices)
    
    # パラメータのテストケース
    test_cases = [
        {
            'name': 'デフォルト設定',
            'params': {
                'alpha_multiplier1': 1.0,
                'alpha_multiplier2': 0.5,
                'alpha_multiplier3': 0.1
            }
        },
        {
            'name': 'カスタム設定1',
            'params': {
                'alpha_multiplier1': 1.0,
                'alpha_multiplier2': 0.7,
                'alpha_multiplier3': 0.3
            }
        },
        {
            'name': 'カスタム設定2',
            'params': {
                'alpha_multiplier1': 0.8,
                'alpha_multiplier2': 0.4,
                'alpha_multiplier3': 0.2
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        try:
            indicator = HyperTripleFRAMA(
                period=12,
                enable_indicator_adaptation=False,
                smoothing_mode='none',
                **test_case['params']
            )
            
            result = indicator.calculate(df)
            
            # 最後の有効な値を表示
            valid_indices = ~np.isnan(result.frama_values)
            if np.any(valid_indices):
                last_valid_idx = np.where(valid_indices)[0][-1]
                print(f"価格: {df['close'].iloc[last_valid_idx]:.2f}")
                print(f"FRAMA1 (α={test_case['params']['alpha_multiplier1']}): {result.frama_values[last_valid_idx]:.2f}")
                print(f"FRAMA2 (α={test_case['params']['alpha_multiplier2']}): {result.second_frama_values[last_valid_idx]:.2f}")
                print(f"FRAMA3 (α={test_case['params']['alpha_multiplier3']}): {result.third_frama_values[last_valid_idx]:.2f}")
            
            print("✅ 成功")
            
        except Exception as e:
            print(f"❌ エラー: {e}")


if __name__ == "__main__":
    print("HyperTripleFRAMAテスト開始")
    
    # 基本テスト
    success = test_hyper_triple_frama()
    
    if success:
        # パラメータテスト
        test_hyper_triple_frama_params()
        print("\n🎉 全テスト完了!")
    else:
        print("\n❌ 基本テストに失敗しました")