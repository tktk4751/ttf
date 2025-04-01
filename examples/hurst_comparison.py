#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
from datetime import datetime, timedelta
import matplotlib as mpl
from pathlib import Path
import csv
import seaborn as sns

# 親ディレクトリをパスに追加してインポートできるようにする
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# インジケーターのインポート
from indicators.hurst_exponent import HurstExponent
from indicators.z_hurst_exponent import ZHurstExponent
from indicators.cycle_efficiency_ratio import CycleEfficiencyRatio

# 日本語フォント設定
plt.rcParams['font.family'] = 'sans-serif'
# Ubuntuの場合
plt.rcParams['font.sans-serif'] = ['IPAGothic', 'IPAPGothic', 'VL Gothic', 'Noto Sans CJK JP', 'Takao']

# フォントが見つからない場合のフォールバック
import matplotlib.font_manager as fm
fonts = set([f.name for f in fm.fontManager.ttflist])
if not any(font in fonts for font in plt.rcParams['font.sans-serif']):
    # フォールバック: 日本語をASCIIで置き換える
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    
    # 警告を無効化
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    
    # 日本語のタイトルや軸ラベルの場合はASCIIに置き換える
    USE_ASCII_LABELS = True
else:
    USE_ASCII_LABELS = False


def generate_synthetic_data(days=200, seed=42):
    """
    異なる市場状態を持つ合成データを生成する
    
    Args:
        days: 生成する日数
        seed: 乱数シード
        
    Returns:
        pd.DataFrame: 生成されたOHLCデータとトレンド/レンジ状態のラベル
    """
    np.random.seed(seed)
    
    # 日付インデックスの作成
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 初期値
    base_price = 100.0
    
    # 異なる市場状態
    prices = []
    current_price = base_price
    
    # ランダムウォーク部分（最初の20%）
    random_days = int(days * 0.2)
    for _ in range(random_days):
        volatility = np.random.uniform(0.5, 1.5)
        change = np.random.normal(0, volatility)
        current_price += change
        prices.append(current_price)
    
    # トレンド部分（次の20%）
    trend_days = int(days * 0.2)
    trend_strength = np.random.choice([0.5, -0.5])  # 上昇または下降トレンド
    for _ in range(trend_days):
        volatility = np.random.uniform(0.3, 1.0)
        change = trend_strength + np.random.normal(0, volatility)
        current_price += change
        prices.append(current_price)
    
    # レンジ相場部分（次の20%）
    range_days = int(days * 0.2)
    range_center = current_price
    for _ in range(range_days):
        # レンジ相場では中心に引き戻される力が働く
        mean_reversion = (range_center - current_price) * 0.1
        volatility = np.random.uniform(0.3, 0.8)
        change = mean_reversion + np.random.normal(0, volatility)
        current_price += change
        prices.append(current_price)
    
    # 強いトレンド部分（次の20%）
    strong_trend_days = int(days * 0.2)
    strong_trend = np.random.choice([1.0, -1.0])  # 強い上昇または下降トレンド
    for _ in range(strong_trend_days):
        volatility = np.random.uniform(0.2, 0.7)
        change = strong_trend + np.random.normal(0, volatility)
        current_price += change
        prices.append(current_price)
    
    # レンジ相場部分（残り）
    remaining_days = days - random_days - trend_days - range_days - strong_trend_days
    range_center = current_price
    for _ in range(remaining_days):
        mean_reversion = (range_center - current_price) * 0.15
        volatility = np.random.uniform(0.2, 0.6)
        change = mean_reversion + np.random.normal(0, volatility)
        current_price += change
        prices.append(current_price)
    
    # リストからndarrayに変換
    prices = np.array(prices)
    
    # OHLCデータの生成
    data = pd.DataFrame(index=date_range[:len(prices)])
    data['close'] = prices
    
    # 高値と安値の生成
    daily_volatility = np.random.uniform(0.5, 2.0, size=len(prices))
    data['high'] = data['close'] + daily_volatility
    data['low'] = data['close'] - daily_volatility
    
    # 始値の生成（前日の終値からランダムに変動）
    open_changes = np.random.normal(0, 0.5, size=len(prices))
    data['open'] = data['close'].shift(1) + open_changes
    data.loc[data.index[0], 'open'] = data['close'].iloc[0] - np.random.uniform(0, 1)
    
    # 列の順序を調整
    data = data[['open', 'high', 'low', 'close']]
    
    # 市場状態ラベルの追加（1=トレンド、0=レンジ）
    market_state = np.zeros(len(prices), dtype=int)
    market_state_labels = np.empty(len(prices), dtype=object)
    
    # ランダムウォーク部分
    market_state[:random_days] = 0
    market_state_labels[:random_days] = 'ランダム'
    
    # トレンド部分
    market_state[random_days:random_days+trend_days] = 1
    market_state_labels[random_days:random_days+trend_days] = '中程度トレンド'
    
    # レンジ相場部分
    market_state[random_days+trend_days:random_days+trend_days+range_days] = 0
    market_state_labels[random_days+trend_days:random_days+trend_days+range_days] = 'レンジ相場'
    
    # 強いトレンド部分
    start_idx = random_days+trend_days+range_days
    end_idx = start_idx+strong_trend_days
    market_state[start_idx:end_idx] = 1
    market_state_labels[start_idx:end_idx] = '強いトレンド'
    
    # 最後のレンジ相場部分
    market_state[end_idx:] = 0
    market_state_labels[end_idx:] = 'レンジ相場2'
    
    data['is_trend'] = market_state  # 1=トレンド、0=レンジ
    data['market_state'] = market_state_labels  # 市場状態の詳細ラベル
    
    return data


def calculate_indicators(df, window=100, min_lag=10, max_lag=50, step=5):
    """
    通常のハースト指数とZハースト指数を計算する
    
    Args:
        df: 価格データ
        window: ハースト指数の計算ウィンドウ
        min_lag: 最小ラグ
        max_lag: 最大ラグ
        step: ステップサイズ
        
    Returns:
        tuple: 標準ハースト指数値, Zハースト指数値, サイクル効率比
    """
    print("インジケーターを計算中...")
    
    # サイクル効率比を計算（Zハースト指数で使用）
    cer = CycleEfficiencyRatio(
        cycle_detector_type='dudi_dce',
        lp_period=10,
        hp_period=48,
        cycle_part=0.5
    )
    cer_values = cer.calculate(df)
    
    # 標準ハースト指数の計算
    hurst = HurstExponent(
        window=window,
        min_lag=min_lag,
        max_lag=max_lag,
        step=step
    )
    hurst_values = hurst.calculate(df)
    
    try:
        # Zハースト指数の計算
        z_hurst = ZHurstExponent(
            max_window_dc_cycle_part=0.75,
            max_window_dc_max_cycle=144,
            max_window_dc_min_cycle=8,
            max_window_dc_max_output=200,
            max_window_dc_min_output=50,
            
            min_window_dc_cycle_part=0.5,
            min_window_dc_max_cycle=55,
            min_window_dc_min_cycle=5,
            min_window_dc_max_output=50,
            min_window_dc_min_output=20,
            
            max_lag_ratio=0.5,
            min_lag_ratio=0.1,
            
            max_step=10,
            min_step=2,
            
            cycle_detector_type='dudi_dce',
            lp_period=10,
            hp_period=48,
            cycle_part=0.5,
            
            max_threshold=0.7,
            min_threshold=0.55
        )
        z_hurst_values = z_hurst.calculate(df)
    except Exception as e:
        print(f"Zハースト指数計算中にエラー: {str(e)}")
        print("標準ハースト指数と同じ長さのNaN配列を使用します")
        # エラーが発生した場合、標準ハースト指数と同じ長さのNaN配列を生成
        z_hurst_values = np.full_like(hurst_values, np.nan)
    
    return hurst_values, z_hurst_values, cer_values


def analyze_performance(df, hurst_values, z_hurst_values, output_dir='output'):
    """
    ハースト指数とZハースト指数の性能を分析して、統計情報を出力する
    
    Args:
        df: 価格データとトレンド/レンジのラベル付きデータフレーム
        hurst_values: 標準ハースト指数値
        z_hurst_values: Zハースト指数値
        output_dir: 出力ディレクトリ
    """
    print("性能分析中...")
    
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 配列長を確認し、一致させる
    min_length = min(len(df), len(hurst_values), len(z_hurst_values))
    
    # データを切り詰める
    df = df.iloc[:min_length].copy()
    hurst_values = hurst_values[:min_length]
    z_hurst_values = z_hurst_values[:min_length]
    
    # 分析用のデータフレームを作成
    analysis_df = pd.DataFrame({
        'date': df.index,
        'close': df['close'],
        'is_trend': df['is_trend'],
        'market_state': df['market_state'],
        'hurst': hurst_values,
        'z_hurst': z_hurst_values
    })
    
    # 初期値をスキップ（NaN値）
    start_idx = 100  # ウィンドウサイズと同じくらいをスキップ
    analysis_df = analysis_df.iloc[start_idx:].copy()
    
    # NaN値の処理
    analysis_df = analysis_df.dropna()
    
    # サンプルが少ない場合はエラーメッセージを表示して空の結果を返す
    if len(analysis_df) < 10:
        print("有効なデータが足りません。分析を中止します。")
        return analysis_df, []
    
    # しきい値を設定
    hurst_threshold = 0.6  # 通常のハースト指数の閾値
    z_hurst_adaptive_thresholds = analysis_df['z_hurst'] > 0.6  # 簡略化のため固定閾値を使用
    
    # 予測を生成
    analysis_df['hurst_trend_pred'] = (analysis_df['hurst'] > hurst_threshold).astype(int)
    analysis_df['z_hurst_trend_pred'] = z_hurst_adaptive_thresholds.astype(int)
    
    # 正解率の計算
    hurst_accuracy = (analysis_df['hurst_trend_pred'] == analysis_df['is_trend']).mean()
    z_hurst_accuracy = (analysis_df['z_hurst_trend_pred'] == analysis_df['is_trend']).mean()
    
    # 各市場状態ごとの正解率
    market_states = analysis_df['market_state'].unique()
    state_accuracies = []
    
    for state in market_states:
        state_mask = analysis_df['market_state'] == state
        if state_mask.sum() > 0:
            hurst_state_acc = (analysis_df.loc[state_mask, 'hurst_trend_pred'] == 
                              analysis_df.loc[state_mask, 'is_trend']).mean()
            z_hurst_state_acc = (analysis_df.loc[state_mask, 'z_hurst_trend_pred'] == 
                                analysis_df.loc[state_mask, 'is_trend']).mean()
            state_accuracies.append({
                'market_state': state,
                'samples': state_mask.sum(),
                'hurst_accuracy': hurst_state_acc,
                'z_hurst_accuracy': z_hurst_state_acc,
                'improvement': z_hurst_state_acc - hurst_state_acc
            })
    
    # 統計情報の出力
    with open(output_path / 'hurst_comparison_stats.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Standard Hurst', 'Z-Hurst', 'Improvement'])
        writer.writerow(['Overall Accuracy', f'{hurst_accuracy:.4f}', f'{z_hurst_accuracy:.4f}', 
                         f'{z_hurst_accuracy - hurst_accuracy:.4f}'])
        
        for state_acc in state_accuracies:
            writer.writerow([
                f"Accuracy ({state_acc['market_state']})", 
                f"{state_acc['hurst_accuracy']:.4f}", 
                f"{state_acc['z_hurst_accuracy']:.4f}",
                f"{state_acc['improvement']:.4f}"
            ])
    
    # 詳細な分析結果をJSONで保存
    analysis_df.to_csv(output_path / 'hurst_comparison_data.csv', index=False)
    
    print(f"全体の正解率 - 標準ハースト指数: {hurst_accuracy:.4f}, Zハースト指数: {z_hurst_accuracy:.4f}")
    print(f"改善率: {(z_hurst_accuracy - hurst_accuracy) * 100:.2f}%")
    
    return analysis_df, state_accuracies


def plot_comparison(df, hurst_values, z_hurst_values, analysis_df, state_accuracies, output_dir='output'):
    """
    ハースト指数とZハースト指数の比較プロットを作成
    
    Args:
        df: 元のデータフレーム
        hurst_values: 標準ハースト指数値
        z_hurst_values: Zハースト指数値
        analysis_df: 分析用データフレーム
        state_accuracies: 市場状態ごとの正解率
        output_dir: 出力ディレクトリ
    """
    print("比較プロットを作成中...")
    
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 配列長の調整
    min_length = min(len(df), len(hurst_values), len(z_hurst_values))
    df = df.iloc[:min_length].copy()
    hurst_values = hurst_values[:min_length]
    z_hurst_values = z_hurst_values[:min_length]
    
    # 検証 - 市場状態に関する統計がない場合はプロットしない
    if not state_accuracies:
        print("市場状態に関する統計がないため、プロットは作成されません。")
        return
    
    # 初期の値（最初の100日）を除外
    start_idx = 100
    if min_length <= start_idx:
        start_idx = min_length // 2  # データが少ない場合は半分だけスキップ
    
    # プロットの作成
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True, 
                             gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # 価格チャート (初期値を除外)
    axes[0].plot(df.index[start_idx:], df['close'][start_idx:], 'k-', linewidth=1.5, 
                label='価格' if not USE_ASCII_LABELS else 'Price')
    axes[0].set_ylabel('価格' if not USE_ASCII_LABELS else 'Price')
    axes[0].set_title('価格チャート' if not USE_ASCII_LABELS else 'Price Chart')
    axes[0].grid(True)
    axes[0].legend(loc='upper left')
    
    # 市場状態に応じた背景色
    colors = {
        'ランダム': 'lightyellow', 
        '中程度トレンド': 'lightgreen', 
        'レンジ相場': 'lightblue', 
        '強いトレンド': 'lightcoral',
        'レンジ相場2': 'lightskyblue'
    }
    
    if USE_ASCII_LABELS:
        # 英語の市場状態ラベル
        color_map = {
            'ランダム': 'Random',
            '中程度トレンド': 'Moderate Trend',
            'レンジ相場': 'Range',
            '強いトレンド': 'Strong Trend',
            'レンジ相場2': 'Range 2'
        }
        # 色のマッピングを英語に変更
        new_colors = {}
        for k, v in colors.items():
            new_colors[color_map.get(k, k)] = v
        colors = new_colors
        
        # データフレームの市場状態ラベルも英語に変換
        df['market_state'] = df['market_state'].map(lambda x: color_map.get(x, x))
    
    try:
        current_state = df['market_state'].iloc[start_idx]
        state_start = df.index[start_idx]
        
        for i, (idx, row) in enumerate(df.iloc[start_idx:].iterrows()):
            if row['market_state'] != current_state or i == len(df.iloc[start_idx:]) - 1:
                for ax in axes:
                    ax.axvspan(state_start, idx, alpha=0.3, color=colors.get(current_state, 'lightgray'))
                current_state = row['market_state']
                state_start = idx
    except Exception as e:
        print(f"市場状態の背景塗りつぶしエラー: {str(e)}")
    
    # ハースト指数の比較プロット
    axes[1].plot(df.index[start_idx:], hurst_values[start_idx:], 'b-', linewidth=1.5, 
                label='標準ハースト指数' if not USE_ASCII_LABELS else 'Standard Hurst Exponent')
    axes[1].plot(df.index[start_idx:], z_hurst_values[start_idx:], 'r-', linewidth=1.5, 
                label='Zハースト指数' if not USE_ASCII_LABELS else 'Z Hurst Exponent')
    axes[1].axhline(y=0.5, color='k', linestyle='--', alpha=0.7)
    axes[1].axhline(y=0.6, color='g', linestyle='--', alpha=0.7, 
                   label='閾値 (0.6)' if not USE_ASCII_LABELS else 'Threshold (0.6)')
    axes[1].set_ylabel('ハースト指数値' if not USE_ASCII_LABELS else 'Hurst Exponent Value')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)
    axes[1].legend(loc='upper left')
    
    # 正解率の棒グラフ
    x = np.arange(len(state_accuracies))
    width = 0.35
    
    state_names = [acc['market_state'] for acc in state_accuracies]
    hurst_accs = [acc['hurst_accuracy'] for acc in state_accuracies]
    z_hurst_accs = [acc['z_hurst_accuracy'] for acc in state_accuracies]
    
    axes[2].bar(x - width/2, hurst_accs, width, label='標準ハースト指数' if not USE_ASCII_LABELS else 'Standard Hurst')
    axes[2].bar(x + width/2, z_hurst_accs, width, label='Zハースト指数' if not USE_ASCII_LABELS else 'Z Hurst')
    
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(state_names)
    axes[2].set_ylabel('正解率' if not USE_ASCII_LABELS else 'Accuracy')
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis='y')
    axes[2].legend(loc='upper left')
    
    # X軸の設定
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    
    # レイアウトの調整とタイトル
    plt.tight_layout()
    title = 'ハースト指数とZハースト指数の比較' if not USE_ASCII_LABELS else 'Standard Hurst vs Z-Hurst Comparison'
    plt.suptitle(title, fontsize=16, y=1.02)
    
    # 保存
    plt.savefig(output_path / 'hurst_comparison.png', bbox_inches='tight')
    
    try:
        # トレンド/レンジの識別パフォーマンスのヒートマップ
        plt.figure(figsize=(10, 6))
        
        # 混同行列の作成
        from sklearn.metrics import confusion_matrix
        hurst_cm = confusion_matrix(analysis_df['is_trend'], analysis_df['hurst_trend_pred'])
        z_hurst_cm = confusion_matrix(analysis_df['is_trend'], analysis_df['z_hurst_trend_pred'])
        
        # 混同行列を正規化
        hurst_cm_norm = hurst_cm.astype('float') / hurst_cm.sum(axis=1)[:, np.newaxis]
        z_hurst_cm_norm = z_hurst_cm.astype('float') / z_hurst_cm.sum(axis=1)[:, np.newaxis]
        
        # Zハースト指数の改善率のヒートマップ
        improvement = z_hurst_cm_norm - hurst_cm_norm
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # ラベル設定
        labels = ['レンジ', 'トレンド'] if not USE_ASCII_LABELS else ['Range', 'Trend']
        
        # 標準ハースト指数の混同行列
        sns.heatmap(hurst_cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[0], 
                    xticklabels=labels, yticklabels=labels)
        axes[0].set_xlabel('予測' if not USE_ASCII_LABELS else 'Predicted')
        axes[0].set_ylabel('実際' if not USE_ASCII_LABELS else 'Actual')
        axes[0].set_title('標準ハースト指数' if not USE_ASCII_LABELS else 'Standard Hurst')
        
        # Zハースト指数の混同行列
        sns.heatmap(z_hurst_cm_norm, annot=True, fmt='.2f', cmap='Reds', ax=axes[1], 
                    xticklabels=labels, yticklabels=labels)
        axes[1].set_xlabel('予測' if not USE_ASCII_LABELS else 'Predicted')
        axes[1].set_ylabel('実際' if not USE_ASCII_LABELS else 'Actual')
        axes[1].set_title('Zハースト指数' if not USE_ASCII_LABELS else 'Z Hurst')
        
        # 改善率のヒートマップ
        sns.heatmap(improvement, annot=True, fmt='.2f', cmap='RdBu', center=0, ax=axes[2], 
                    xticklabels=labels, yticklabels=labels)
        axes[2].set_xlabel('予測' if not USE_ASCII_LABELS else 'Predicted')
        axes[2].set_ylabel('実際' if not USE_ASCII_LABELS else 'Actual')
        axes[2].set_title('改善率 (Z - 標準)' if not USE_ASCII_LABELS else 'Improvement (Z - Standard)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'hurst_confusion_matrices.png', bbox_inches='tight')
    except Exception as e:
        print(f"混同行列の作成エラー: {str(e)}")
    
    # プロットを表示（オプション）
    plt.show()


def main():
    """メイン関数"""
    # 出力ディレクトリ
    output_dir = 'examples/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 合成データの生成
    print("合成データを生成中...")
    df = generate_synthetic_data(days=300, seed=42)
    
    # ハースト指数とZハースト指数の計算
    hurst_values, z_hurst_values, cer_values = calculate_indicators(df)
    
    # 性能分析
    analysis_df, state_accuracies = analyze_performance(df, hurst_values, z_hurst_values, output_dir)
    
    # 比較プロット
    plot_comparison(df, hurst_values, z_hurst_values, analysis_df, state_accuracies, output_dir)
    
    print(f"処理が完了しました。結果は '{output_dir}' に保存されました。")


if __name__ == "__main__":
    main() 