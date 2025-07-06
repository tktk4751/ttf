#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌟 Ultra Quantum Adaptive Trend-Range Discriminator (UQATRD) デモンストレーション

John Ehlersの革新的な量子アルゴリズムによるトレンド/レンジ判別インジケーターの実演
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import time

# プロジェクトルートを追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 出力ディレクトリの作成
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

def load_sample_data():
    """サンプルデータの読み込み"""
    try:
        # 既存のデータファイルから読み込み
        data_files = [
            project_root / "data" / "sample_data.csv",
            project_root / "examples" / "sample_data.csv"
        ]
        
        for file_path in data_files:
            if file_path.exists():
                print(f"📊 データファイル読み込み: {file_path}")
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    return df
        
        # データファイルが見つからない場合、サンプルデータを生成
        print("🔄 サンプルデータを生成中...")
        return generate_sample_data()
        
    except Exception as e:
        print(f"⚠️ データ読み込みエラー: {e}")
        return generate_sample_data()


def generate_sample_data(n_points=1000):
    """複雑な市場データを生成"""
    np.random.seed(42)
    
    # 基本トレンド
    trend = np.cumsum(np.random.randn(n_points) * 0.1)
    
    # サイクル成分
    cycle1 = 10 * np.sin(np.linspace(0, 10*np.pi, n_points))
    cycle2 = 5 * np.sin(np.linspace(0, 20*np.pi, n_points))
    
    # ランダムウォーク
    random_walk = np.cumsum(np.random.randn(n_points) * 0.5)
    
    # 価格の構築
    base_price = 100 + trend + cycle1 + cycle2 + random_walk
    
    # OHLC生成
    high = base_price + np.abs(np.random.randn(n_points) * 0.8)
    low = base_price - np.abs(np.random.randn(n_points) * 0.8)
    open_price = base_price + np.random.randn(n_points) * 0.3
    close = base_price + np.random.randn(n_points) * 0.3
    
    # 日付インデックス
    dates = pd.date_range('2023-01-01', periods=n_points, freq='D')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, n_points)
    })
    
    return df


def run_uqatrd_analysis():
    """UQATRD分析の実行"""
    print("🚀 Ultra Quantum Adaptive Trend-Range Discriminator 分析開始")
    print("=" * 80)
    
    # データの読み込み
    df = load_sample_data()
    print(f"✅ データロード完了: {len(df)} rows")
    
    # UQATRDインジケーターの初期化
    try:
        from indicators.ultra_quantum_adaptive_trend_range_discriminator import UltraQuantumAdaptiveTrendRangeDiscriminator
        
        # 異なるパラメータ設定でのインスタンス作成
        uqatrd_standard = UltraQuantumAdaptiveTrendRangeDiscriminator(
            coherence_window=21,
            entanglement_window=34,
            efficiency_window=21,
            uncertainty_window=14,
            src_type='hlc3',
            sensitivity=1.0
        )
        
        uqatrd_sensitive = UltraQuantumAdaptiveTrendRangeDiscriminator(
            coherence_window=14,
            entanglement_window=21,
            efficiency_window=14,
            uncertainty_window=10,
            src_type='close',
            sensitivity=1.5
        )
        
        print("✅ UQATRD インジケーター初期化完了")
        
    except Exception as e:
        print(f"❌ インジケーター初期化エラー: {e}")
        return
    
    # 分析実行
    print("\n🔬 量子アルゴリズム分析実行中...")
    
    start_time = time.time()
    
    # 標準パラメータ
    result_standard = uqatrd_standard.calculate(df)
    
    # 高感度パラメータ  
    result_sensitive = uqatrd_sensitive.calculate(df)
    
    calculation_time = time.time() - start_time
    
    print(f"⚡ 計算時間: {calculation_time:.4f}秒")
    
    # 結果の分析
    print("\n📊 分析結果サマリー")
    print("-" * 40)
    
    # 標準パラメータ結果
    trend_signal = result_standard.trend_range_signal
    signal_strength = result_standard.signal_strength
    confidence = result_standard.confidence_score
    
    print(f"🎯 トレンド/レンジ判定:")
    print(f"   - 平均信号: {np.mean(trend_signal):.3f}")
    print(f"   - 信号強度: {np.mean(signal_strength):.3f}")
    print(f"   - 信頼度: {np.mean(confidence):.3f}")
    
    # 各量子アルゴリズムの結果
    print(f"\n🔬 量子アルゴリズム詳細:")
    print(f"   - 量子コヒーレンス: {np.mean(result_standard.quantum_coherence):.3f}")
    print(f"   - トレンド持続性: {np.mean(result_standard.trend_persistence):.3f}")
    print(f"   - 効率スペクトラム: {np.mean(result_standard.efficiency_spectrum):.3f}")
    print(f"   - 不確定性レンジ: {np.mean(result_standard.uncertainty_range):.3f}")
    
    # 可視化
    print("\n📈 可視化作成中...")
    create_comprehensive_visualization(df, result_standard, result_sensitive)
    
    # 動的閾値分析
    print("\n🎯 動的適応閾値分析")
    analyze_adaptive_threshold(uqatrd_standard, df)
    
    # パフォーマンス分析
    print("\n🏆 パフォーマンス分析")
    analyze_performance(result_standard, df)
    
    return result_standard, result_sensitive


def create_comprehensive_visualization(df, result_standard, result_sensitive):
    """包括的な可視化の作成"""
    
    # 価格データの準備
    prices = df['close'].values
    dates = df['timestamp'] if 'timestamp' in df.columns else range(len(df))
    
    # 図のスタイル設定
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. メイン価格チャート + トレンド/レンジ判定
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(dates, prices, 'k-', linewidth=0.8, alpha=0.7, label='価格')
    
    # トレンド/レンジ判定のカラーマップ
    trend_signal = result_standard.trend_range_signal
    confidence = result_standard.confidence_score
    
    # 信号に基づく色付け（0=レンジ、1=トレンド）
    for i in range(len(prices)-1):
        if trend_signal[i] > 0.7:
            color = 'green'  # 強いトレンド
            alpha = min(0.8, confidence[i] + 0.2)
        elif trend_signal[i] > 0.4:
            color = 'yellow'  # 弱いトレンド
            alpha = min(0.6, confidence[i] + 0.1)
        else:
            color = 'red'  # レンジ
            alpha = min(0.8, confidence[i] + 0.2)
        
        ax1.axvspan(dates[i], dates[i+1], color=color, alpha=alpha*0.2)
    
    ax1.set_title('🎯 UQATRD トレンド/レンジ判定', fontsize=14, fontweight='bold')
    ax1.set_ylabel('価格', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 量子コヒーレンス
    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(dates, result_standard.quantum_coherence, 'b-', linewidth=1.5, label='量子コヒーレンス')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(dates, 0, result_standard.quantum_coherence, alpha=0.3, color='blue')
    ax2.set_title('🌀 量子コヒーレンス方向性測定', fontsize=14, fontweight='bold')
    ax2.set_ylabel('コヒーレンス', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. トレンド持続性
    ax3 = plt.subplot(4, 2, 3)
    ax3.plot(dates, result_standard.trend_persistence, 'r-', linewidth=1.5, label='トレンド持続性')
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='レンジ/トレンド境界')
    ax3.fill_between(dates, 0.5, result_standard.trend_persistence, 
                     alpha=0.3, color='green', where=(result_standard.trend_persistence > 0.5))
    ax3.fill_between(dates, 0, result_standard.trend_persistence, 
                     alpha=0.3, color='red', where=(result_standard.trend_persistence <= 0.5))
    ax3.set_title('🔗 量子エンタングルメントトレンド持続性', fontsize=14, fontweight='bold')
    ax3.set_ylabel('トレンド持続性', fontsize=12)
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 効率スペクトラム
    ax4 = plt.subplot(4, 2, 4)
    ax4.plot(dates, result_standard.efficiency_spectrum, 'g-', linewidth=1.5, label='効率スペクトラム')
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax4.fill_between(dates, 0, result_standard.efficiency_spectrum, alpha=0.3, color='green')
    ax4.set_title('📊 量子効率スペクトラム', fontsize=14, fontweight='bold')
    ax4.set_ylabel('効率性', fontsize=12)
    ax4.set_ylim(0, 1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 不確定性レンジ
    ax5 = plt.subplot(4, 2, 5)
    ax5.plot(dates, result_standard.uncertainty_range, 'purple', linewidth=1.5, label='不確定性レンジ')
    ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax5.fill_between(dates, 0, result_standard.uncertainty_range, alpha=0.3, color='purple')
    ax5.set_title('🎯 量子不確定性レンジ検出', fontsize=14, fontweight='bold')
    ax5.set_ylabel('不確定性', fontsize=12)
    ax5.set_ylim(0, 1)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 信頼度スコア
    ax6 = plt.subplot(4, 2, 6)
    ax6.plot(dates, result_standard.confidence_score, 'orange', linewidth=1.5, label='信頼度スコア')
    ax6.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='閾値')
    ax6.fill_between(dates, 0, result_standard.confidence_score, alpha=0.3, color='orange')
    ax6.set_title('📈 信頼度スコア', fontsize=14, fontweight='bold')
    ax6.set_ylabel('信頼度', fontsize=12)
    ax6.set_ylim(0, 1)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 最終統合信号
    ax7 = plt.subplot(4, 2, 7)
    ax7.plot(dates, result_standard.trend_range_signal, 'black', linewidth=2, label='UQATRD信号')
    ax7.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='レンジ/トレンド境界')
    ax7.axhline(y=0.7, color='green', linestyle=':', alpha=0.5, label='強いトレンド閾値')
    ax7.axhline(y=0.3, color='orange', linestyle=':', alpha=0.5, label='弱いトレンド閾値')
    ax7.fill_between(dates, 0.5, result_standard.trend_range_signal, 
                     alpha=0.3, color='green', where=(result_standard.trend_range_signal > 0.5))
    ax7.fill_between(dates, 0, result_standard.trend_range_signal, 
                     alpha=0.3, color='red', where=(result_standard.trend_range_signal <= 0.5))
    ax7.set_title('🎯 最終統合信号 (0=レンジ, 1=トレンド)', fontsize=14, fontweight='bold')
    ax7.set_ylabel('トレンド/レンジ信号', fontsize=12)
    ax7.set_ylim(0, 1)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 感度比較
    ax8 = plt.subplot(4, 2, 8)
    ax8.plot(dates, result_standard.trend_range_signal, 'blue', linewidth=1.5, label='標準感度')
    ax8.plot(dates, result_sensitive.trend_range_signal, 'red', linewidth=1.5, label='高感度')
    ax8.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='レンジ/トレンド境界')
    ax8.set_title('⚡ 感度比較 (0=レンジ, 1=トレンド)', fontsize=14, fontweight='bold')
    ax8.set_ylabel('トレンド/レンジ信号', fontsize=12)
    ax8.set_ylim(0, 1)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / "uqatrd_comprehensive_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 総合分析チャート保存: {output_path}")
    
    plt.show()


def analyze_adaptive_threshold(indicator, df):
    """動的適応閾値分析"""
    
    # 閾値情報の取得
    threshold_info = indicator.get_threshold_info()
    adaptive_threshold = indicator.get_adaptive_threshold()
    classification = indicator.get_trend_range_classification()
    
    if threshold_info is None:
        print("❌ 閾値情報を取得できませんでした")
        return
    
    # 閾値統計情報の表示
    print(f"📊 動的適応閾値統計:")
    print(f"   - 平均閾値: {threshold_info['mean_threshold']:.3f}")
    print(f"   - 標準偏差: {threshold_info['std_threshold']:.3f}")
    print(f"   - 最小閾値: {threshold_info['min_threshold']:.3f}")
    print(f"   - 最大閾値: {threshold_info['max_threshold']:.3f}")
    print(f"   - 中央値: {threshold_info['median_threshold']:.3f}")
    print(f"   - 現在の閾値: {threshold_info['current_threshold']:.3f}")
    
    # 動的閾値による分類統計
    if classification is not None:
        trend_count = np.sum(classification == 1.0)
        range_count = np.sum(classification == 0.0)
        total_count = len(classification)
        
        print(f"\n🎯 動的閾値による分類結果:")
        print(f"   - トレンド判定: {trend_count}点 ({trend_count/total_count*100:.1f}%)")
        print(f"   - レンジ判定: {range_count}点 ({range_count/total_count*100:.1f}%)")
    
    # 閾値の適応性分析
    if adaptive_threshold is not None:
        threshold_changes = np.abs(np.diff(adaptive_threshold))
        avg_change = np.mean(threshold_changes)
        max_change = np.max(threshold_changes)
        
        print(f"\n⚡ 閾値適応性分析:")
        print(f"   - 平均変化量: {avg_change:.4f}")
        print(f"   - 最大変化量: {max_change:.4f}")
        print(f"   - 閾値安定性: {1.0 - avg_change:.3f}")
        
        # 閾値の分布
        threshold_bins = np.histogram(adaptive_threshold, bins=5)
        print(f"   - 閾値分布:")
        for i, (count, bin_edge) in enumerate(zip(threshold_bins[0], threshold_bins[1][:-1])):
            next_edge = threshold_bins[1][i+1]
            print(f"     [{bin_edge:.2f}-{next_edge:.2f}]: {count}点 ({count/len(adaptive_threshold)*100:.1f}%)")
    
    # 動的閾値の可視化
    create_adaptive_threshold_visualization(df, indicator)


def create_adaptive_threshold_visualization(df, indicator):
    """動的閾値可視化"""
    
    # データの準備
    result = indicator._result_cache[indicator._cache_keys[-1]]
    prices = df['close'].values
    dates = df['timestamp'] if 'timestamp' in df.columns else range(len(df))
    
    # 図の作成
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # 1. 価格チャート + 動的閾値による分類
    ax1 = axes[0]
    ax1.plot(dates, prices, 'k-', linewidth=1.0, alpha=0.7, label='価格')
    
    # 動的閾値による分類の色付け
    classification = indicator.get_trend_range_classification()
    if classification is not None:
        for i in range(len(prices)-1):
            if classification[i] == 1.0:
                color = 'green'  # トレンド
                alpha = 0.2
            else:
                color = 'red'    # レンジ
                alpha = 0.2
            
            ax1.axvspan(dates[i], dates[i+1], color=color, alpha=alpha)
    
    ax1.set_title('🎯 動的閾値によるトレンド/レンジ分類', fontsize=14, fontweight='bold')
    ax1.set_ylabel('価格', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. UQATRD信号 + 動的閾値
    ax2 = axes[1]
    ax2.plot(dates, result.trend_range_signal, 'blue', linewidth=2, label='UQATRD信号')
    ax2.plot(dates, result.adaptive_threshold, 'red', linewidth=2, label='動的閾値')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='固定閾値 (0.5)')
    
    # 閾値を超えた部分の色付け
    mask_above = result.trend_range_signal >= result.adaptive_threshold
    ax2.fill_between(dates, 0, result.trend_range_signal, 
                     alpha=0.3, color='green', where=mask_above, label='トレンド領域')
    ax2.fill_between(dates, 0, result.trend_range_signal, 
                     alpha=0.3, color='red', where=~mask_above, label='レンジ領域')
    
    ax2.set_title('📈 UQATRD信号 vs 動的適応閾値', fontsize=14, fontweight='bold')
    ax2.set_ylabel('値', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 閾値の時系列変化
    ax3 = axes[2]
    ax3.plot(dates, result.adaptive_threshold, 'red', linewidth=2, label='動的閾値')
    ax3.axhline(y=np.mean(result.adaptive_threshold), color='orange', 
                linestyle='--', alpha=0.7, label=f'平均閾値 ({np.mean(result.adaptive_threshold):.3f})')
    ax3.fill_between(dates, 0.4, 0.6, alpha=0.1, color='gray', label='閾値範囲 (0.4-0.6)')
    
    ax3.set_title('⚡ 動的閾値の時系列変化', fontsize=14, fontweight='bold')
    ax3.set_ylabel('閾値', fontsize=12)
    ax3.set_xlabel('時間', fontsize=12)
    ax3.set_ylim(0.35, 0.65)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / "uqatrd_adaptive_threshold_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"🎯 動的閾値分析チャート保存: {output_path}")
    
    plt.show()


def analyze_performance(result, df):
    """パフォーマンス分析"""
    
    # トレンド/レンジ判定の精度分析
    trend_signal = result.trend_range_signal
    confidence = result.confidence_score
    
    # 高信頼度ポイントの抽出
    high_confidence_mask = confidence > 0.7
    high_confidence_signals = trend_signal[high_confidence_mask]
    
    # 統計サマリー
    print(f"📊 統計サマリー:")
    print(f"   - 高信頼度ポイント: {np.sum(high_confidence_mask)}/{len(confidence)} ({np.sum(high_confidence_mask)/len(confidence)*100:.1f}%)")
    print(f"   - 平均信頼度: {np.mean(confidence):.3f}")
    print(f"   - 信頼度範囲: {np.min(confidence):.3f} - {np.max(confidence):.3f}")
    
    # トレンド/レンジの分布（0=レンジ、1=トレンド）
    strong_trend_count = np.sum(trend_signal > 0.7)
    weak_trend_count = np.sum((trend_signal > 0.4) & (trend_signal <= 0.7))
    range_count = np.sum(trend_signal <= 0.4)
    
    print(f"   - 強いトレンド判定: {strong_trend_count} ({strong_trend_count/len(trend_signal)*100:.1f}%)")
    print(f"   - 弱いトレンド判定: {weak_trend_count} ({weak_trend_count/len(trend_signal)*100:.1f}%)")
    print(f"   - レンジ判定: {range_count} ({range_count/len(trend_signal)*100:.1f}%)")
    
    # 量子アルゴリズムの安定性
    algorithms = {
        'Quantum Coherence': result.quantum_coherence,
        'Trend Persistence': result.trend_persistence,
        'Efficiency Spectrum': result.efficiency_spectrum,
        'Uncertainty Range': result.uncertainty_range
    }
    
    print(f"\n🔬 量子アルゴリズム安定性:")
    for name, values in algorithms.items():
        std_dev = np.std(values)
        mean_val = np.mean(values)
        stability = 1.0 - std_dev / (abs(mean_val) + 1e-10)
        print(f"   - {name}: 平均={mean_val:.3f}, 標準偏差={std_dev:.3f}, 安定性={stability:.3f}")


if __name__ == "__main__":
    print("🌟 Ultra Quantum Adaptive Trend-Range Discriminator (UQATRD)")
    print("=" * 70)
    print("John Ehlersの革新的量子アルゴリズム実演")
    print("=" * 70)
    
    try:
        result_standard, result_sensitive = run_uqatrd_analysis()
        
        print("\n🎯 分析完了!")
        print(f"結果は {output_dir}/uqatrd_comprehensive_analysis.png に保存されました。")
        
    except Exception as e:
        print(f"❌ 分析エラー: {e}")
        import traceback
        traceback.print_exc() 