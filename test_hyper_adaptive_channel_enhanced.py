#!/usr/bin/env python3
"""
Hyper Adaptive Channel 拡張パラメーターテスト

すべてのパラメーターを含むバージョンのテスト
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('.')

def test_enhanced_parameters():
    """拡張パラメーターテスト"""
    
    print("=== Hyper Adaptive Channel 拡張パラメーターテスト ===")
    
    try:
        from indicators.hyper_adaptive_channel import HyperAdaptiveChannel
        
        # テストデータ作成
        np.random.seed(42)
        n = 200
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1H'),
            'open': 100 + np.cumsum(np.random.randn(n) * 0.02),
            'high': 101 + np.cumsum(np.random.randn(n) * 0.02),
            'low': 99 + np.cumsum(np.random.randn(n) * 0.02),
            'close': 100 + np.cumsum(np.random.randn(n) * 0.02),
            'volume': np.random.randint(1000, 5000, n)
        })
        
        print(f"テストデータ作成: {len(data)}件")
        
        # カスタマイズされたパラメーターでテスト
        print("\n--- カスタマイズパラメーターテスト ---")
        
        indicator = HyperAdaptiveChannel(
            period=20,
            midline_smoother="super_smoother",
            multiplier_mode="dynamic",
            fixed_multiplier=3.0,
            
            # SuperSmoother カスタマイズ
            super_smoother_length=18,
            super_smoother_num_poles=3,
            super_smoother_src_type='hlc3',
            
            # X_ATR カスタマイズ  
            x_atr_period=16.0,
            x_atr_tr_method='atr',
            x_atr_enable_kalman=True,
            x_atr_smoother_type='ultimate_ma',
            
            # HyperER カスタマイズ
            hyper_er_period=12,
            hyper_er_er_period=15,
            hyper_er_use_roofing_filter=True,
            hyper_er_use_smoothing=True,
            hyper_er_smoother_period=8,
            
            # その他カスタマイズ
            enable_signals=True,
            enable_percentile=True,
            percentile_period=80
        )
        
        result = indicator.calculate(data)
        
        print("✓ カスタマイズパラメーター計算成功")
        print(f"  - Midline有効: {np.sum(~np.isnan(result.midline))}/{len(data)}")
        print(f"  - Upper Band有効: {np.sum(~np.isnan(result.upper_band))}/{len(data)}")
        print(f"  - Lower Band有効: {np.sum(~np.isnan(result.lower_band))}/{len(data)}")
        print(f"  - 乗数範囲: {np.nanmin(result.multiplier_values):.2f} - {np.nanmax(result.multiplier_values):.2f}")
        
        if result.er_values is not None:
            print(f"  - ER範囲: {np.nanmin(result.er_values):.3f} - {np.nanmax(result.er_values):.3f}")
        
        # HyperFRAMA with フル動的適応
        print("\n--- HyperFRAMA 動的適応テスト ---")
        
        indicator_frama = HyperAdaptiveChannel(
            period=14,
            midline_smoother="hyper_frama",
            multiplier_mode="dynamic",
            
            # HyperFRAMA 動的適応パラメーター
            hyper_frama_period_mode='dynamic',
            hyper_frama_enable_indicator_adaptation=True,
            hyper_frama_adaptation_indicator='hyper_er',
            hyper_frama_fc_min=1.0,
            hyper_frama_fc_max=12.0,
            hyper_frama_sc_min=40.0,
            hyper_frama_sc_max=300.0,
            hyper_frama_period_min=6,
            hyper_frama_period_max=60,
            
            # Cycle detector
            hyper_frama_cycle_detector_type='hody_e',
            hyper_frama_max_cycle=120,
            hyper_frama_min_cycle=6
        )
        
        result_frama = indicator_frama.calculate(data)
        
        print("✓ HyperFRAMA 動的適応計算成功")
        print(f"  - Midline有効: {np.sum(~np.isnan(result_frama.midline))}/{len(data)}")
        print(f"  - 乗数範囲: {np.nanmin(result_frama.multiplier_values):.2f} - {np.nanmax(result_frama.multiplier_values):.2f}")
        
        # UltimateMA with フル動的適応
        print("\n--- UltimateMA 動的適応テスト ---")
        
        try:
            indicator_uma = HyperAdaptiveChannel(
                period=14,
                midline_smoother="ultimate_ma",
                multiplier_mode="fixed",
                fixed_multiplier=2.0,
                
                # UltimateMA 動的適応パラメーター
                ultimate_ma_src_type='hlc3',  # ukf_hlc3は問題があるため
                ultimate_ma_zero_lag_period_mode='dynamic',
                ultimate_ma_realtime_window_mode='dynamic',
                ultimate_ma_use_adaptive_kalman=True,
                ultimate_ma_kalman_process_variance=5e-6,
                ultimate_ma_kalman_measurement_variance=0.005,
                
                # Cycle detectors
                ultimate_ma_zl_cycle_detector_type='absolute_ultimate',
                ultimate_ma_rt_cycle_detector_type='absolute_ultimate',
                ultimate_ma_zl_cycle_detector_max_cycle=100,
                ultimate_ma_rt_cycle_detector_max_cycle=80
            )
            
            result_uma = indicator_uma.calculate(data)
            
            print("✓ UltimateMA 動的適応計算成功")
            print(f"  - Midline有効: {np.sum(~np.isnan(result_uma.midline))}/{len(data)}")
            
        except Exception as e:
            print(f"! UltimateMA 動的適応エラー: {e}")
        
        # パラメーター値の確認
        print("\n--- パラメーター確認 ---")
        
        # HyperFRAMAパラメーター確認
        frama_params = indicator_frama.hyper_frama_params
        print(f"HyperFRAMA FC範囲: {frama_params['fc_min']} - {frama_params['fc_max']}")
        print(f"HyperFRAMA SC範囲: {frama_params['sc_min']} - {frama_params['sc_max']}")
        print(f"HyperFRAMA Period範囲: {frama_params['period_min']} - {frama_params['period_max']}")
        print(f"HyperFRAMA 動的適応: {frama_params['enable_indicator_adaptation']}")
        print(f"HyperFRAMA 適応インジケーター: {frama_params['adaptation_indicator']}")
        
        # X_ATRパラメーター確認
        atr_params = indicator.x_atr_params
        print(f"X_ATR TR方法: {atr_params['tr_method']}")
        print(f"X_ATR カルマン使用: {atr_params['enable_kalman']}")
        print(f"X_ATR スムーサー: {atr_params['smoother_type']}")
        print(f"X_ATR パーセンタイル分析: {atr_params['enable_percentile_analysis']}")
        
        # HyperERパラメーター確認
        er_params = indicator.hyper_er_params
        print(f"HyperER ルーフィング使用: {er_params['use_roofing_filter']}")
        print(f"HyperER カルマン使用: {er_params['use_kalman_filter']}")
        print(f"HyperER 平滑化使用: {er_params['use_smoothing']}")
        print(f"HyperER 平滑化タイプ: {er_params['smoother_type']}")
        
    except Exception as e:
        print(f"✗ テストエラー: {e}")
        import traceback
        traceback.print_exc()


def test_all_smoothers_with_custom_params():
    """すべてのスムーザーをカスタムパラメーターでテスト"""
    
    print("\n=== 全スムーザー カスタムパラメーターテスト ===")
    
    # テストデータ
    np.random.seed(42)
    n = 150
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='2H'),
        'open': 100 + np.cumsum(np.random.randn(n) * 0.01),
        'high': 101 + np.cumsum(np.random.randn(n) * 0.01),
        'low': 99 + np.cumsum(np.random.randn(n) * 0.01), 
        'close': 100 + np.cumsum(np.random.randn(n) * 0.01),
        'volume': np.random.randint(1000, 3000, n)
    })
    
    smoothers = ["hyper_frama", "laguerre_filter", "super_smoother"]
    
    for smoother in smoothers:
        print(f"\n--- {smoother} カスタムパラメーターテスト ---")
        
        try:
            # 各スムーザー専用のカスタマイズパラメーター
            if smoother == "hyper_frama":
                params = {
                    "midline_smoother": smoother,
                    "hyper_frama_fc": 2,
                    "hyper_frama_sc": 150,
                    "hyper_frama_alpha_multiplier": 0.7,
                    "hyper_frama_src_type": "hlc3"
                }
            elif smoother == "laguerre_filter":
                params = {
                    "midline_smoother": smoother,
                    "laguerre_gamma": 0.8,
                    "laguerre_order": 6,
                    "laguerre_src_type": "hl2"
                }
            elif smoother == "super_smoother":
                params = {
                    "midline_smoother": smoother,
                    "super_smoother_length": 12,
                    "super_smoother_num_poles": 3,
                    "super_smoother_src_type": "ohlc4"
                }
            
            indicator = HyperAdaptiveChannel(
                period=16,
                multiplier_mode="dynamic",
                **params
            )
            
            result = indicator.calculate(data)
            
            print(f"✓ {smoother} カスタム計算成功")
            print(f"  - Midline有効: {np.sum(~np.isnan(result.midline))}/{len(data)}")
            print(f"  - 乗数範囲: {np.nanmin(result.multiplier_values):.2f} - {np.nanmax(result.multiplier_values):.2f}")
            
        except Exception as e:
            print(f"✗ {smoother} カスタムエラー: {e}")
    
    print("\n✓ 拡張パラメーターテスト完了")


if __name__ == "__main__":
    test_enhanced_parameters()
    test_all_smoothers_with_custom_params()