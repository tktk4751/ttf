#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 **Kalman Filter Unified Demo V1.0** 🎯

統合カルマンフィルターシステムのデモンストレーション
- 全フィルターの比較テスト
- パフォーマンス分析
- 結果の可視化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import yaml

from indicators.kalman_filter_unified import KalmanFilterUnified
# データローダー関連のインポート
from data.data_loader import DataLoader, CSVDataSource
from data.data_processor import DataProcessor
from data.binance_data_source import BinanceDataSource

plt.style.use('dark_background')


def create_synthetic_data(n_points: int = 1000) -> pd.DataFrame:
    """
    合成価格データを作成（テスト用）
    """
    np.random.seed(42)
    
    # 基本トレンド
    t = np.linspace(0, 10, n_points)
    trend = 100 + 10 * np.sin(t * 0.5) + 0.5 * t
    
    # サイクル成分
    cycle1 = 5 * np.sin(t * 2)
    cycle2 = 3 * np.cos(t * 3)
    
    # ノイズ
    noise = np.random.normal(0, 2, n_points)
    
    # 突発的なスパイク
    spikes = np.zeros(n_points)
    spike_indices = np.random.choice(n_points, size=20, replace=False)
    spikes[spike_indices] = np.random.normal(0, 10, 20)
    
    # 最終価格
    close_prices = trend + cycle1 + cycle2 + noise + spikes
    
    # OHLC生成
    high_prices = close_prices + np.abs(np.random.normal(0, 1, n_points))
    low_prices = close_prices - np.abs(np.random.normal(0, 1, n_points))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # ボリューム
    volume = np.random.lognormal(10, 0.5, n_points)
    
    # 日時
    start_date = datetime.now() - timedelta(days=n_points)
    dates = [start_date + timedelta(days=i) for i in range(n_points)]
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    })
    
    # datetimeをインデックスに設定
    df.set_index('datetime', inplace=True)
    return df


def test_all_filters(data: pd.DataFrame) -> dict:
    """
    全フィルターをテストして結果を比較
    """
    filters = KalmanFilterUnified.get_available_filters()
    results = {}
    
    print("🧪 全カルマンフィルターのテスト実行中...")
    
    for filter_type, description in filters.items():
        print(f"   📊 {filter_type}: {description}")
        
        try:
            # フィルター初期化
            kalman_filter = KalmanFilterUnified(
                filter_type=filter_type,
                src_type='close',
                base_process_noise=0.001,
                base_measurement_noise=0.01,
                volatility_window=10
            )
            
            # 計算実行
            result = kalman_filter.calculate(data)
            
            # パフォーマンス評価
            performance = evaluate_filter_performance(
                original=data['close'].values,
                filtered=result.filtered_values,
                confidence=result.confidence_scores
            )
            
            results[filter_type] = {
                'result': result,
                'performance': performance,
                'metadata': kalman_filter.get_filter_metadata(),
                'description': description
            }
            
            print(f"      ✅ 成功 - 性能スコア: {performance['total_score']:.3f}")
            
        except Exception as e:
            print(f"      ❌ エラー: {e}")
            import traceback
            print(f"         詳細: {traceback.format_exc()}")
            results[filter_type] = None
    
    return results


def evaluate_filter_performance(original: np.ndarray, filtered: np.ndarray, confidence: np.ndarray) -> dict:
    """
    フィルターの性能を評価
    """
    if len(original) != len(filtered) or len(original) < 10:
        return {'total_score': 0.0}
    
    # 1. ノイズ除去効果（差分の標準偏差比較）
    original_noise = np.std(np.diff(original))
    filtered_noise = np.std(np.diff(filtered))
    noise_reduction = max(0, 1 - filtered_noise / original_noise) if original_noise > 0 else 0
    
    # 2. 価格追従性（平均絶対誤差）
    tracking_error = np.mean(np.abs(filtered - original))
    price_std = np.std(original)
    tracking_score = max(0, 1 - tracking_error / price_std) if price_std > 0 else 0
    
    # 3. 滑らかさ（二次差分の分散）
    filtered_smooth = np.var(np.diff(filtered, n=2)) if len(filtered) > 2 else 1.0
    original_smooth = np.var(np.diff(original, n=2)) if len(original) > 2 else 1.0
    smoothness_score = max(0, 1 - filtered_smooth / original_smooth) if original_smooth > 0 else 0
    
    # 4. 信頼度平均
    confidence_score = np.nanmean(confidence) if len(confidence) > 0 else 0
    
    # 5. 遅延評価（ピーク検出での遅延測定）
    delay_score = calculate_delay_score(original, filtered)
    
    # 6. 総合スコア（重み付き平均）
    weights = [0.25, 0.25, 0.2, 0.15, 0.15]  # [ノイズ除去, 追従性, 滑らかさ, 信頼度, 遅延]
    scores = [noise_reduction, tracking_score, smoothness_score, confidence_score, delay_score]
    total_score = sum(w * s for w, s in zip(weights, scores))
    
    return {
        'noise_reduction': noise_reduction,
        'tracking_score': tracking_score,
        'smoothness_score': smoothness_score,
        'confidence_score': confidence_score,
        'delay_score': delay_score,
        'total_score': total_score
    }


def calculate_delay_score(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    遅延スコアを計算（ピーク検出による）
    """
    try:
        from scipy.signal import find_peaks
        
        # ピーク検出
        orig_peaks, _ = find_peaks(original, height=np.percentile(original, 70))
        filt_peaks, _ = find_peaks(filtered, height=np.percentile(filtered, 70))
        
        if len(orig_peaks) < 2 or len(filt_peaks) < 2:
            return 0.5  # デフォルトスコア
        
        # 最も近いピーク間の遅延を計算
        delays = []
        for op in orig_peaks[:10]:  # 最初の10ピークのみ
            distances = np.abs(filt_peaks - op)
            min_delay = np.min(distances)
            delays.append(min_delay)
        
        avg_delay = np.mean(delays)
        max_acceptable_delay = len(original) * 0.05  # 5%までの遅延は許容
        
        delay_score = max(0, 1 - avg_delay / max_acceptable_delay)
        return min(1.0, delay_score)
        
    except ImportError:
        # scipy無い場合は簡易評価
        correlation = np.corrcoef(original[1:], filtered[:-1])[0, 1]
        return max(0, correlation) if not np.isnan(correlation) else 0.5


def visualize_comparison(data: pd.DataFrame, results: dict, save_path: str = None):
    """
    全フィルターの比較結果を可視化
    """
    # 有効な結果のみ抽出
    valid_results = {k: v for k, v in results.items() if v is not None}
    n_filters = len(valid_results)
    
    if n_filters == 0:
        print("表示可能な結果がありません")
        return
    
    # 図の設定
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('black')
    
    # グリッド設定
    rows = 3
    cols = 2
    
    # 1. 価格比較プロット
    ax1 = plt.subplot(rows, cols, 1)
    ax1.plot(data.index, data['close'], 'white', alpha=0.7, linewidth=1, label='Original')
    
    colors = ['cyan', 'yellow', 'lime', 'magenta', 'orange', 'red']
    for i, (filter_type, result_data) in enumerate(valid_results.items()):
        if result_data and result_data['result']:
            ax1.plot(data.index, result_data['result'].filtered_values, 
                    colors[i % len(colors)], alpha=0.8, linewidth=2, 
                    label=f"{filter_type}")
    
    ax1.set_title('🎯 Kalman Filters Comparison', fontsize=14, color='white', fontweight='bold')
    ax1.set_ylabel('Price', color='white')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 性能スコア比較
    ax2 = plt.subplot(rows, cols, 2)
    filter_names = []
    total_scores = []
    
    for filter_type, result_data in valid_results.items():
        if result_data and result_data['performance']:
            filter_names.append(filter_type)
            total_scores.append(result_data['performance']['total_score'])
    
    bars = ax2.bar(filter_names, total_scores, color=colors[:len(filter_names)], alpha=0.8)
    ax2.set_title('📊 Performance Scores', fontsize=14, color='white', fontweight='bold')
    ax2.set_ylabel('Score', color='white')
    ax2.set_ylim(0, 1)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # スコア値をバーの上に表示
    for bar, score in zip(bars, total_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', color='white', fontsize=10)
    
    ax2.grid(True, alpha=0.3)
    
    # 3. カルマンゲイン比較
    ax3 = plt.subplot(rows, cols, 3)
    for i, (filter_type, result_data) in enumerate(valid_results.items()):
        if result_data and result_data['result']:
            ax3.plot(data.index, result_data['result'].kalman_gains,
                    colors[i % len(colors)], alpha=0.7, linewidth=1.5,
                    label=f"{filter_type}")
    
    ax3.set_title('⚙️ Kalman Gains', fontsize=14, color='white', fontweight='bold')
    ax3.set_ylabel('Kalman Gain', color='white')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 信頼度比較
    ax4 = plt.subplot(rows, cols, 4)
    for i, (filter_type, result_data) in enumerate(valid_results.items()):
        if result_data and result_data['result']:
            ax4.plot(data.index, result_data['result'].confidence_scores,
                    colors[i % len(colors)], alpha=0.7, linewidth=1.5,
                    label=f"{filter_type}")
    
    ax4.set_title('📈 Confidence Scores', fontsize=14, color='white', fontweight='bold')
    ax4.set_ylabel('Confidence', color='white')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. 詳細性能メトリクス
    ax5 = plt.subplot(rows, cols, 5)
    metrics = ['noise_reduction', 'tracking_score', 'smoothness_score', 'confidence_score', 'delay_score']
    metric_labels = ['Noise\nReduction', 'Price\nTracking', 'Smoothness', 'Confidence', 'Low\nDelay']
    
    # 最高性能フィルターを特定
    best_filter = None
    best_score = 0
    for filter_type, result_data in valid_results.items():
        if result_data and result_data['performance']['total_score'] > best_score:
            best_score = result_data['performance']['total_score']
            best_filter = filter_type
    
    if best_filter and valid_results[best_filter]:
        perf = valid_results[best_filter]['performance']
        values = [perf[metric] for metric in metrics]
        
        bars = ax5.bar(metric_labels, values, color='lime', alpha=0.8)
        ax5.set_title(f'🏆 Best Filter: {best_filter}', fontsize=14, color='white', fontweight='bold')
        ax5.set_ylabel('Score', color='white')
        ax5.set_ylim(0, 1)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', color='white', fontsize=10)
        
        ax5.grid(True, alpha=0.3)
    
    # 6. ランキング表
    ax6 = plt.subplot(rows, cols, 6)
    ax6.axis('off')
    
    # ランキング作成
    ranking = sorted([(k, v['performance']['total_score']) for k, v in valid_results.items() 
                     if v and v['performance']], key=lambda x: x[1], reverse=True)
    
    ranking_text = "🏆 Filter Ranking:\n\n"
    for i, (filter_name, score) in enumerate(ranking):
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        ranking_text += f"{emoji} {filter_name}: {score:.3f}\n"
    
    ax6.text(0.1, 0.9, ranking_text, transform=ax6.transAxes, fontsize=12, 
            color='white', verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, facecolor='black', edgecolor='none', dpi=300, bbox_inches='tight')
        print(f"📊 比較結果を保存しました: {save_path}")
    
    plt.show()


def detailed_filter_analysis(filter_type: str, data: pd.DataFrame):
    """
    特定フィルターの詳細分析
    """
    print(f"\n🔍 詳細分析: {filter_type}")
    print("=" * 50)
    
    # フィルター実行
    try:
        kalman_filter = KalmanFilterUnified(filter_type=filter_type, src_type='close')
        result = kalman_filter.calculate(data)
        metadata = kalman_filter.get_filter_metadata()
    except Exception as e:
        print(f"❌ フィルター実行エラー: {e}")
        return
    
    # 分析結果表示
    print(f"📊 基本統計:")
    print(f"   データポイント数: {metadata.get('data_points', 'N/A')}")
    print(f"   平均信頼度: {metadata.get('avg_confidence', 0):.3f}")
    print(f"   平均カルマンゲイン: {metadata.get('avg_kalman_gain', 0):.3f}")
    print(f"   平均イノベーション: {metadata.get('avg_innovation', 0):.3f}")
    
    # フィルター固有の情報
    if metadata.get('avg_quantum_coherence'):
        print(f"   平均量子コヒーレンス: {metadata['avg_quantum_coherence']:.3f}")
    if metadata.get('avg_uncertainty'):
        print(f"   平均不確実性: {metadata['avg_uncertainty']:.3f}")
    if metadata.get('avg_trend_estimate'):
        print(f"   平均トレンド推定: {metadata['avg_trend_estimate']:.3f}")
    
    # 性能評価
    performance = evaluate_filter_performance(
        data['close'].values, result.filtered_values, result.confidence_scores
    )
    
    print(f"\n🎯 性能評価:")
    print(f"   ノイズ除去: {performance['noise_reduction']:.3f}")
    print(f"   価格追従性: {performance['tracking_score']:.3f}")
    print(f"   滑らかさ: {performance['smoothness_score']:.3f}")
    print(f"   信頼度: {performance['confidence_score']:.3f}")
    print(f"   遅延スコア: {performance['delay_score']:.3f}")
    print(f"   総合スコア: {performance['total_score']:.3f}")


def load_data_from_config(config_path: str = 'config.yaml') -> pd.DataFrame:
    """
    config.yamlから実際の相場データを読み込む（z_adaptive_trend_chart.py参考）
    """
    try:
        print(f"📡 {config_path} からデータを読み込み中...")
        
        # 設定ファイルの読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # データの準備
        binance_config = config.get('binance_data', {})
        data_dir = binance_config.get('data_dir', 'data/binance')
        binance_data_source = BinanceDataSource(data_dir)
        
        # CSVデータソースはダミーとして渡す（Binanceデータソースのみを使用）
        dummy_csv_source = CSVDataSource("dummy")
        data_loader = DataLoader(
            data_source=dummy_csv_source,
            binance_data_source=binance_data_source
        )
        data_processor = DataProcessor()
        
        # データの読み込みと処理
        raw_data = data_loader.load_data_from_config(config)
        processed_data = {
            symbol: data_processor.process(df)
            for symbol, df in raw_data.items()
        }
        
        # 最初のシンボルのデータを取得
        first_symbol = next(iter(processed_data))
        data = processed_data[first_symbol]
        
        print(f"✅ 実データ読み込み完了: {first_symbol}")
        print(f"📊 データ数: {len(data)}")
        return data
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        print("🔄 合成データを使用します")
        return create_synthetic_data(500)


def main():
    """
    メイン実行関数
    """
    print("🚀 Kalman Filter Unified Demo V1.0")
    print("=" * 50)
    
    # データ準備
    print("\n1️⃣ データ準備")
    
    # 実データを試し、失敗したら合成データ
    data = load_data_from_config('config.yaml')
    
    print(f"📊 データ概要:")
    print(f"   期間: {len(data)} データポイント")
    print(f"   価格範囲: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   平均価格: ${data['close'].mean():.2f}")
    
    # 全フィルターテスト
    print("\n2️⃣ 全フィルターテスト実行")
    results = test_all_filters(data)
    
    # 結果比較・可視化
    print("\n3️⃣ 結果の可視化")
    output_path = os.path.join('output', 'kalman_filter_comparison.png')
    os.makedirs('output', exist_ok=True)
    visualize_comparison(data, results, output_path)
    
    # 詳細分析
    print("\n4️⃣ 詳細分析")
    
    # 最高性能フィルターの詳細分析
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_filter = max(valid_results.keys(), 
                         key=lambda k: valid_results[k]['performance']['total_score'])
        detailed_filter_analysis(best_filter, data)
    
    # 総合レポート
    print("\n📋 総合レポート")
    print("=" * 50)
    
    valid_count = len(valid_results)
    total_filters = len(KalmanFilterUnified.get_available_filters())
    
    print(f"✅ テスト成功: {valid_count}/{total_filters} フィルター")
    
    if valid_results:
        # トップ3
        ranking = sorted([(k, v['performance']['total_score']) for k, v in valid_results.items()], 
                        key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 トップ3フィルター:")
        for i, (name, score) in enumerate(ranking[:3]):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            print(f"   {emoji} {name}: {score:.3f}")
        
        # 推奨用途
        print(f"\n💡 推奨用途:")
        if ranking[0][0] == 'adaptive':
            print("   - 汎用的な用途には adaptive フィルターがお勧め")
        elif ranking[0][0] == 'quantum_adaptive':
            print("   - 高精度が必要な場合は quantum_adaptive がお勧め")
        elif ranking[0][0] == 'triple_ensemble':
            print("   - 安定性重視なら triple_ensemble がお勧め")
        else:
            print(f"   - {ranking[0][0]} フィルターが最適です")
    
    print(f"\n🎉 デモ完了!")
    print(f"📊 結果は {output_path} に保存されました")


if __name__ == "__main__":
    main() 