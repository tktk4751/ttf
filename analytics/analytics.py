import numpy as np
from typing import List, Optional, Tuple, Dict
from datetime import datetime
from backtesting.trade import Trade


class Analytics:
    def __init__(self, trades: List[Trade], initial_capital: float):
        """トレード結果の分析器を初期化"""
        self.trades = sorted(trades, key=lambda x: x.entry_date)
        self.initial_capital = initial_capital
        self.final_capital = trades[-1].balance if trades else initial_capital
        
        # 損益データを配列として保持（高速化）
        self.profits = np.array([t.profit_loss for t in trades if t.profit_loss > 0])
        self.losses = np.array([t.profit_loss for t in trades if t.profit_loss < 0])
        self.returns = np.array([t.profit_loss / t.position_size for t in trades])
        
        # 辞書形式のトレードデータを内部で保持
        self._trades_data = [{
            'profit_loss': t.profit_loss,
            'profit_loss_pct': t.profit_loss / t.position_size,
            'balance': t.balance,
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'position_type': t.position_type
        } for t in trades]

    def calculate_total_return(self) -> float:
        """総リターンを計算"""
        if not self.trades:
            return 0.0
        return (self.final_capital / self.initial_capital - 1) * 100

    def calculate_cagr(self, position_type: Optional[str] = None) -> float:
        """年率複利収益率を計算"""
        if not self.trades:
            return 0.0
            
        filtered_trades = self._filter_trades_by_position_type(position_type)
        if not filtered_trades:
            return 0.0
            
        # トレード日数を取得
        start_date = filtered_trades[0].entry_date
        end_date = filtered_trades[-1].exit_date
        trading_days = (end_date - start_date).days
        years = trading_days / 365.25  # 営業日数で除算
        
        if years == 0:
            return 0.0
            
        # 累積リターンを計算
        cumulative_return = self.final_capital / self.initial_capital
        
        # CAGRを計算
        return ((cumulative_return) ** (1/years) - 1) * 100

    def calculate_win_rate(self) -> float:
        """勝率を計算"""
        if not self.trades:
            return 0.0
        return (len(self.profits) / len(self.trades)) * 100

    def calculate_average_bars(self) -> float:
        """平均保有期間を計算"""
        if not self.trades:
            return 0.0
        holding_periods = [(t.exit_date - t.entry_date).total_seconds() / 3600 for t in self.trades]
        return sum(holding_periods) / len(holding_periods)

    def calculate_total_profit(self) -> float:
        """総利益を計算"""
        return np.sum(self.profits) if len(self.profits) > 0 else 0.0

    def calculate_total_loss(self) -> float:
        """総損失を計算"""
        return np.sum(self.losses) if len(self.losses) > 0 else 0.0

    def calculate_net_profit_loss(self, position_type: Optional[str] = None) -> float:
        """純損益を計算"""
        filtered_trades = self._filter_trades_by_position_type(position_type)
        return sum(t.profit_loss for t in filtered_trades)

    def calculate_number_of_trades(self, position_type: Optional[str] = None) -> int:
        """トレード数を計算"""
        filtered_trades = self._filter_trades_by_position_type(position_type)
        return len(filtered_trades)

    def calculate_max_win_loss(self) -> Tuple[float, float]:
        """最大の勝ち負けを計算"""
        max_profit = np.max(self.profits) if len(self.profits) > 0 else 0.0
        max_loss = np.min(self.losses) if len(self.losses) > 0 else 0.0
        return max_profit, max_loss

    def calculate_average_profit_loss(self) -> Tuple[float, float]:
        """平均の勝ち負けを計算"""
        avg_profit = np.mean(self.profits) if len(self.profits) > 0 else 0.0
        avg_loss = np.mean(self.losses) if len(self.losses) > 0 else 0.0
        return avg_profit, avg_loss

    def calculate_max_drawdown(self) -> Tuple[float, Optional[datetime], Optional[datetime]]:
        """最大ドローダウンを計算"""
        if not self._trades_data:
            return 0.0, None, None

        equity_curve = [self.initial_capital] + [t['balance'] for t in self._trades_data]
        dates = [self.trades[0].entry_date] + [t['exit_date'] for t in self._trades_data]
        
        max_drawdown = 0
        max_dd_start = None
        max_dd_end = None
        peak = equity_curve[0]
        peak_idx = 0
        
        for i in range(1, len(equity_curve)):
            if equity_curve[i] > peak:
                peak = equity_curve[i]
                peak_idx = i
            drawdown = (peak - equity_curve[i]) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_dd_start = dates[peak_idx]
                max_dd_end = dates[i]

        return max_drawdown * 100, max_dd_start, max_dd_end

    def calculate_drawdown_periods(self) -> List[Tuple[float, int, datetime, datetime]]:
        """すべてのドローダウン期間を計算"""
        if not self._trades_data:
            return []

        equity_curve = [self.initial_capital] + [t['balance'] for t in self._trades_data]
        dates = [self.trades[0].entry_date] + [t['exit_date'] for t in self._trades_data]
        
        drawdown_periods = []
        peak = equity_curve[0]
        peak_idx = 0
        in_drawdown = False
        dd_start = None
        
        for i in range(1, len(equity_curve)):
            if equity_curve[i] > peak:
                if in_drawdown:
                    drawdown = (peak - equity_curve[i-1]) / peak * 100
                    duration = (dates[i-1] - dates[dd_start]).days
                    drawdown_periods.append((drawdown, duration, dates[dd_start], dates[i-1]))
                    in_drawdown = False
                peak = equity_curve[i]
                peak_idx = i
            elif not in_drawdown and equity_curve[i] < peak:
                in_drawdown = True
                dd_start = peak_idx
        
        if in_drawdown:
            drawdown = (peak - equity_curve[-1]) / peak * 100
            duration = (dates[-1] - dates[dd_start]).days
            drawdown_periods.append((drawdown, duration, dates[dd_start], dates[-1]))
        
        return sorted(drawdown_periods, key=lambda x: x[0], reverse=True)

    def calculate_sharpe_ratio(self) -> float:
        """シャープレシオを計算"""
        if not self.trades:
            return 0.0
        risk_free_rate = 0.02  # 年率2%と仮定
        excess_returns = self.returns - (risk_free_rate / 365.25)  # 日次リターンに変換
        volatility = np.std(self.returns, ddof=1) * np.sqrt(365.25)
        if volatility == 0:
            return 0.0
        return np.mean(excess_returns) * np.sqrt(365.25) / volatility

    def calculate_sortino_ratio(self) -> float:
        """ソルティノレシオを計算"""
        if not self.trades:
            return 0.0
        risk_free_rate = 0.02
        excess_returns = self.returns - (risk_free_rate / 365.25)
        downside_returns = self.returns[self.returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        downside_volatility = np.std(downside_returns, ddof=1) * np.sqrt(365.25)
        if downside_volatility == 0:
            return 0.0
        return np.mean(excess_returns) * np.sqrt(365.25) / downside_volatility

    def calculate_calmar_ratio(self) -> float:
        """カルマーレシオを計算
        
        CAGRを小数点表記（パーセンテージではなく）で使用し、
        最大ドローダウンも小数点表記で計算します。
        
        Returns:
            float: カルマーレシオ
        """
        max_dd, _, _ = self.calculate_max_drawdown()
        if max_dd == 0:
            return 0.0
        # CAGRをパーセンテージから小数点表記に変換
        return (self.calculate_cagr() / 100) / (max_dd / 100)
    
    def calculate_calmar_ratio_v2(self) -> float:
        """調整済みリターンでカルマーレシオを計算"""
        if not self.trades:
            return 0.0
        risk_free_rate = 0.02  # 年率2%と仮定
        excess_returns = self.returns - (risk_free_rate / 365.25)  # 日次リターンに変換
        max_dd, _, _ = self.calculate_max_drawdown()
        if max_dd == 0:
            return 0.0
        return np.mean(excess_returns) / (max_dd / 100)
    
    def calculate_drawdown_recovery_efficiency(self) -> float:
        """ドローダウン回復効率を計算
        
        最大ドローダウンからの回復速度を0-1のスケールで評価します。
        - 1に近いほど回復が早い
        - 0に近いほど回復が遅い
        
        計算方法：
        1. 最大ドローダウン期間を取得
        2. 回復日数を計算
        3. exp(-回復日数/365)で0-1の値に変換（1年で約0.37、2年で約0.14）
        
        Returns:
            float: ドローダウン回復効率（0-1）
        """
        if not self.trades:
            return 0.0
            
        max_dd, start_date, end_date = self.calculate_max_drawdown()
        if max_dd == 0 or not start_date or not end_date:
            return 1.0  # ドローダウンがない場合は最高効率
            
        # 回復日数を計算
        recovery_days = (end_date - start_date).days
        if recovery_days <= 0:
            return 1.0
            
        # 指数関数で0-1の値に変換（回復日数が長いほど小さい値に）
        return np.exp(-recovery_days / 365)

    def calculate_value_at_risk(self, confidence: float = 0.95) -> float:
        """バリューアットリスクを計算
        
        指定された信頼水準での最大予想損失率を計算します。
        例えば、VaR(95%) = -10%の場合、95%の確率で損失は10%を超えないことを意味します。
        
        Args:
            confidence: 信頼水準（デフォルト: 0.95）
            
        Returns:
            float: VaR（パーセンテージ）
        """
        if not self.trades:
            return 0.0
            
        # 累積リターンの配列を作成
        cumulative_returns = []
        current_balance = self.initial_capital
        
        for trade in sorted(self.trades, key=lambda x: x.entry_date):
            return_pct = (trade.profit_loss / current_balance) * 100
            cumulative_returns.append(return_pct)
            current_balance = trade.balance
        
        # 指定された信頼水準でのパーセンタイルを計算
        var = np.percentile(cumulative_returns, (1 - confidence) * 100)
        
        return var

    def calculate_expected_shortfall(self, confidence: float = 0.95) -> float:
        """期待ショートフォール（条件付きVaR）を計算
        
        VaRを超える損失が発生した場合の平均損失率を計算します。
        例えば、ES(95%) = -15%の場合、VaRを超える損失が発生した際の
        平均的な損失は15%であることを意味します。
        
        Args:
            confidence: 信頼水準（デフォルト: 0.95）
            
        Returns:
            float: 期待ショートフォール（パーセンテージ）
        """
        if not self.trades:
            return 0.0
            
        # 累積リターンの配列を作成
        cumulative_returns = []
        current_balance = self.initial_capital
        
        for trade in sorted(self.trades, key=lambda x: x.entry_date):
            return_pct = (trade.profit_loss / current_balance) * 100
            cumulative_returns.append(return_pct)
            current_balance = trade.balance
        
        # VaRを計算
        var = np.percentile(cumulative_returns, (1 - confidence) * 100)
        
        # VaRを超える損失のみを抽出
        tail_losses = [r for r in cumulative_returns if r <= var]
        
        if not tail_losses:
            return var
        
        # VaRを超える損失の平均を計算
        return np.mean(tail_losses)

    def calculate_tail_risk_ratio(self) -> float:
        """テールリスク比率を計算"""
        if not self.trades:
            return 0.0
        var = self.calculate_value_at_risk()
        es = self.calculate_expected_shortfall()
        if es == 0:
            return 0.0
        return var / es

    def calculate_payoff_ratio(self) -> float:
        """ペイオフレシオを計算"""
        avg_profit, avg_loss = self.calculate_average_profit_loss()
        if abs(avg_loss) == 0:
            return float('inf') if avg_profit > 0 else 0.0
        return abs(avg_profit / avg_loss)

    def calculate_expected_value(self) -> float:
        """期待値を計算
        
        期待値 = (勝率 * ペイオフレシオ - (1 - 勝率)) / 投資単位あたりの平均損失
        
        Returns:
            float: 期待値
        """
        if not self.trades or len(self.losses) == 0:
            return 0.0
            
        win_rate = self.calculate_win_rate() / 100
        payoff_ratio = self.calculate_payoff_ratio()
        
        # 投資単位あたりの平均損失を計算
        avg_loss_per_unit = abs(np.mean([t.profit_loss / t.position_size for t in self.trades if t.profit_loss < 0]))
        
        if avg_loss_per_unit == 0:
            return 0.0
            
        return (win_rate * payoff_ratio - (1 - win_rate)) / avg_loss_per_unit

    # def calculate_common_sense_ratio(self) -> float:
    #     """コモンセンスレシオを計算"""
    #     if len(self.profits) == 0 or len(self.losses) == 0:
    #         return 0.0
    #     avg_profit = np.mean(self.profits)
    #     avg_loss = abs(np.mean(self.losses))
    #     if avg_loss == 0:
    #         return 0.0
    #     return avg_profit / avg_loss

    def calculate_profit_factor(self) -> float:
        """プロフィットファクターを計算"""
        total_loss = abs(self.calculate_total_loss())
        if total_loss == 0:
            return float('inf') if self.calculate_total_profit() > 0 else 0.0
        return self.calculate_total_profit() / total_loss

    def calculate_pessimistic_return_ratio(self) -> float:
        """悲観的リターンレシオを計算"""
        if not self.trades or len(self.losses) == 0:
            return 0.0
        
        winning_count = len(self.profits)
        losing_count = len(self.losses)
        
        if winning_count == 0 or losing_count == 0:
            return 0.0
        
        total_profit = self.calculate_total_profit()
        total_loss = abs(self.calculate_total_loss())
        
        adjusted_profit = (winning_count - np.sqrt(winning_count)) * (total_profit / winning_count)
        adjusted_loss = (losing_count + np.sqrt(losing_count)) * (total_loss / losing_count)
        
        if adjusted_loss == 0:
            return 0.0
        
        return adjusted_profit / adjusted_loss

    def calculate_geometric_mean_return(self) -> float:
        """リターンの幾何平均を計算
        
        日次リターンの幾何平均を計算します。
        
        Returns:
            float: 幾何平均リターン（パーセンテージ）
        """
        if not self.trades:
            return 0.0
            
        # 1を加えて小数点表記に変換
        returns = self.returns + 1
        
        # 幾何平均を計算
        geometric_mean = np.exp(np.mean(np.log(returns))) - 1
        
        return geometric_mean * 100

    def calculate_alpha_score(self) -> float:
        """アルファスコアを計算 (ゼロ値置換)

        以下の要素を幾何平均で組み合わせた総合的なパフォーマンス指標：

        1. カルマーレシオ (25%): ドローダウンに対するリターンの効率性
        2. ソルティノレシオ (30%): ダウンサイドリスクに対するリターン
        3. 悲観的リターンレシオ (20%): 保守的な収益性評価
        4. 最大ドローダウン (15%): リスク管理の効率性
        5. CAGR (10%): 年間リターンの効率性

        Returns:
            float: 0-100のスケールでのスコア。高いほど良い。
        """
        if not self.trades:
            return 0.0

        # 各指標を0-1にスケール
        calmar = min(max(self.calculate_calmar_ratio_v2(), 0), 3) / 3    # 0-1にスケール
        sortino = min(max(self.calculate_sortino_ratio(), 0), 7) / 7  # 0-1にスケール
        prr = min(max(self.calculate_pessimistic_return_ratio(), 0), 3) / 3  # 0-1にスケール
        max_dd = self.calculate_max_drawdown()[0]
        max_dd_score = max(0, 1 - (max_dd / 100))  # ドローダウンが小さいほど高スコア
        cagr = min(max(self.calculate_cagr(), 0), 400) / 400  # 0-1にスケール（400%を超える場合は1に丸める）     

        # ゼロ値置換: 各指標が0の場合、小さな値に置き換え
        replacement_value = 0.01
        calmar = calmar if calmar > 0 else replacement_value
        sortino = sortino if sortino > 0 else replacement_value
        prr = prr if prr > 0 else replacement_value
        max_dd_score = max_dd_score if max_dd_score > 0 else replacement_value
        cagr = cagr if cagr > 0 else replacement_value

        # 各指標の重要度に応じて指数を設定
        score = (
            calmar ** 0.25 *         # カルマーレシオ (25%)
            sortino ** 0.30 *        # ソルティノレシオ (30%)
            prr ** 0.20 *            # 悲観的リターンレシオ (20%)
            max_dd_score ** 0.15 *       # 最大ドローダウン (15%)
            cagr ** 0.10   # cagr (10%)
        )

        # 0-100のスケールに変換 (補正不要)
        return score * 100
    
    def calculate_cagr_dd_score(self) -> float:
       
        if not self.trades:
            return 0.0

        # 各指標を0-1にスケール
        
        max_dd = self.calculate_max_drawdown()[0]
        max_dd_score = max(0, 1 - (max_dd / 100))  # ドローダウンが小さいほど高スコア
        cagr = min(max(self.calculate_cagr(), 0), 400) / 400  # 0-1にスケール（400%を超える場合は1に丸める）     

        # ゼロ値置換: 各指標が0の場合、小さな値に置き換え
        replacement_value = 0.01
        max_dd_score = max_dd_score if max_dd_score > 0 else replacement_value
        cagr = cagr if cagr > 0 else replacement_value

        # 各指標の重要度に応じて指数を設定
        score = (
            max_dd_score ** 0.60 *      
            cagr ** 0.40 
        )

        # 0-100のスケールに変換 (補正不要)
        return score * 100

    def calculate_sqn(self) -> float:
        """SQN（System Quality Number）スコアを計算
        
        SQNは以下の計算式で求められます：
        SQN = √N * (平均R / 標準偏差R)
        
        ここで：
        - N：トレード数
        - R：各トレードのR倍数（profit_loss / position_size）
        - 平均R：全トレードのR倍数の平均
        - 標準偏差R：全トレードのR倍数の標準偏差
        
        Returns:
            float: SQNスコア。高いほど良い。
            - 1.6-1.9: Below average
            - 2.0-2.4: Average
            - 2.5-2.9: Good
            - 3.0-5.0: Excellent
            - 5.1-6.9: Superb
            - 7.0+: Holy Grail
        """
        if not self.trades:
            return 0.0
            
        # R倍数の配列を取得（すでにself.returnsとして保持）
        n = len(self.trades)
        mean_r = np.mean(self.returns)
        std_r = np.std(self.returns, ddof=1)  # 不偏標準偏差を使用
        
        if std_r == 0:
            return 0.0
            
        return np.sqrt(n) * (mean_r / std_r)

    def _filter_trades_by_position_type(self, position_type: Optional[str] = None) -> List[Trade]:
        """ポジションタイプでトレードをフィルタリング"""
        if position_type is None:
            return self.trades
        return [t for t in self.trades if t.position_type == position_type.upper()]

    def get_full_analysis(self) -> Dict:
        """すべての分析結果を取得"""
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.calculate_total_return(),
            'total_trades': len(self.trades),
            'winning_trades': len(self.profits),
            'losing_trades': len(self.losses),
            'win_rate': self.calculate_win_rate(),
            'total_profit': self.calculate_total_profit(),
            'total_loss': self.calculate_total_loss(),
            'net_profit_loss': self.calculate_net_profit_loss(),
            'max_drawdown': self.calculate_max_drawdown()[0],
            'drawdown_recovery_efficiency': self.calculate_drawdown_recovery_efficiency(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'value_at_risk': self.calculate_value_at_risk(),
            'expected_shortfall': self.calculate_expected_shortfall(),
            'tail_risk_ratio': self.calculate_tail_risk_ratio(),
            'payoff_ratio': self.calculate_payoff_ratio(),
            'expected_value': self.calculate_expected_value(),
            'common_sense_ratio': self.calculate_common_sense_ratio(),
            'profit_factor': self.calculate_profit_factor(),
            'pessimistic_return_ratio': self.calculate_pessimistic_return_ratio(),
            'alpha_score': self.calculate_alpha_score(),
            'sqn': self.calculate_sqn(),
            'average_bars': self.calculate_average_bars(),
            
            # ポジションタイプ別の分析
            'long': {
                'trade_count': self.calculate_number_of_trades('LONG'),
                'net_profit_loss': self.calculate_net_profit_loss('LONG'),
                'cagr': self.calculate_cagr('LONG')
            },
            'short': {
                'trade_count': self.calculate_number_of_trades('SHORT'),
                'net_profit_loss': self.calculate_net_profit_loss('SHORT'),
                'cagr': self.calculate_cagr('SHORT')
            }
        }
    def get_avg_bars_winning_trades(self):
        """勝ちトレードの平均バー数を取得"""
        winning_trades = [t for t in self.trades if t.profit_loss > 0]
        if not winning_trades:
            return 0
        holding_periods = [(t.exit_date - t.entry_date) / np.timedelta64(1, 'D') for t in winning_trades]
        return sum(holding_periods) / len(holding_periods)

    def get_avg_bars_losing_trades(self):
        """負けトレードの平均バー数を取得"""
        losing_trades = [t for t in self.trades if t.profit_loss < 0]
        if not losing_trades:
            return 0
        holding_periods = [(t.exit_date - t.entry_date) / np.timedelta64(1, 'D') for t in losing_trades]
        return sum(holding_periods) / len(holding_periods)

    def get_avg_bars_all_trades(self):
        """全トレードの平均バー数を取得"""
        if not self.trades:
            return 0
        holding_periods = [(t.exit_date - t.entry_date) / np.timedelta64(1, 'D') for t in self.trades]
        return sum(holding_periods) / len(holding_periods)

    def get_long_total_profit(self):
        """ロングトレードの総利益を取得"""
        return sum(t.profit_loss for t in self.trades if t.position_type == 'LONG' and t.profit_loss > 0)

    def get_long_total_profit_percentage(self):
        """ロングトレードの総利益をパーセンテージで取得"""
        return (self.get_long_total_profit() / self.initial_capital) * 100

    def get_long_total_loss(self):
        """ロングトレードの総損失を取得"""
        return sum(t.profit_loss for t in self.trades if t.position_type == 'LONG' and t.profit_loss < 0)

    def get_long_total_loss_percentage(self):
        """ロングトレードの総損失をパーセンテージで取得"""
        return (self.get_long_total_loss() / self.initial_capital) * 100

    def get_long_net_profit(self):
        """ロングトレードの純利益を取得"""
        return self.get_long_total_profit() + self.get_long_total_loss()

    def get_long_net_profit_percentage(self):
        """ロングトレードの純利益をパーセンテージで取得"""
        return (self.get_long_net_profit() / self.initial_capital) * 100

    def get_long_trade_count(self):
        """ロングトレードの数を取得"""
        return len([t for t in self.trades if t.position_type == 'LONG'])

    def get_long_win_rate(self):
        """ロングトレードの勝率を取得"""
        long_trades = [t for t in self.trades if t.position_type == 'LONG']
        if not long_trades:
            return 0
        winning_trades = len([t for t in long_trades if t.profit_loss > 0])
        return (winning_trades / len(long_trades)) * 100

    def get_long_max_win(self):
        """ロングトレードの最大勝ちトレード額を取得"""
        long_profits = [t.profit_loss for t in self.trades if t.position_type == 'LONG' and t.profit_loss > 0]
        return max(long_profits) if long_profits else 0

    def get_long_max_win_percentage(self):
        """ロングトレードの最大勝ちトレード額をパーセンテージで取得"""
        return (self.get_long_max_win() / self.initial_capital) * 100

    def get_long_max_loss(self):
        """ロングトレードの最大負けトレード額を取得"""
        long_losses = [t.profit_loss for t in self.trades if t.position_type == 'LONG' and t.profit_loss < 0]
        return min(long_losses) if long_losses else 0

    def get_long_max_loss_percentage(self):
        """ロングトレードの最大負けトレード額をパーセンテージで取得"""
        return (self.get_long_max_loss() / self.initial_capital) * 100

    def get_short_total_profit(self):
        """ショートトレードの総利益を取得"""
        return sum(t.profit_loss for t in self.trades if t.position_type == 'SHORT' and t.profit_loss > 0)

    def get_short_total_profit_percentage(self):
        """ショートトレードの総利益をパーセンテージで取得"""
        return (self.get_short_total_profit() / self.initial_capital) * 100

    def get_short_total_loss(self):
        """ショートトレードの総損失を取得"""
        return sum(t.profit_loss for t in self.trades if t.position_type == 'SHORT' and t.profit_loss < 0)

    def get_short_total_loss_percentage(self):
        """ショートトレードの総損失をパーセンテージで取得"""
        return (self.get_short_total_loss() / self.initial_capital) * 100

    def get_short_net_profit(self):
        """ショートトレードの純利益を取得"""
        return self.get_short_total_profit() + self.get_short_total_loss()

    def get_short_net_profit_percentage(self):
        """ショートトレードの純利益をパーセンテージで取得"""
        return (self.get_short_net_profit() / self.initial_capital) * 100

    def get_short_max_win(self):
        """ショートトレードの最大勝ちトレード額を取得"""
        short_profits = [t.profit_loss for t in self.trades if t.position_type == 'SHORT' and t.profit_loss > 0]
        return max(short_profits) if short_profits else 0

    def get_short_max_win_percentage(self):
        """ショートトレードの最大勝ちトレード額をパーセンテージで取得"""
        return (self.get_short_max_win() / self.initial_capital) * 100

    def get_short_max_loss(self):
        """ショートトレードの最大負けトレード額を取得"""
        short_losses = [t.profit_loss for t in self.trades if t.position_type == 'SHORT' and t.profit_loss < 0]
        return min(short_losses) if short_losses else 0

    def get_short_max_loss_percentage(self):
        """ショートトレードの最大負けトレード額をパーセンテージで取得"""
        return (self.get_short_max_loss() / self.initial_capital) * 100

    def get_short_trade_count(self):
        """ショートトレードの数を取得"""
        return len([t for t in self.trades if t.position_type == 'SHORT'])

    def get_short_win_rate(self):
        """ショートトレードの勝率を取得"""
        short_trades = [t for t in self.trades if t.position_type == 'SHORT']
        if not short_trades:
            return 0
        winning_trades = len([t for t in short_trades if t.profit_loss > 0])
        return (winning_trades / len(short_trades)) * 100

    def get_winning_trades(self) -> int:
        """勝ちトレード数を取得"""
        return len(self.profits)
    
    def get_losing_trades(self) -> int:
        """負けトレード数を取得"""
        return len(self.losses)

    def print_backtest_results(self) -> None:
        """バックテスト結果の詳細を出力"""
        print("\n=== 基本統計 ===")
        print(f"初期資金: {self.initial_capital:.2f} USD")
        print(f"最終残高: {self.final_capital:.2f} USD")
        print(f"総リターン: {self.calculate_total_return():.2f}%")
        print(f"CAGR: {self.calculate_cagr():.2f}%")
        print(f"1トレードあたりの幾何平均リターン: {self.calculate_geometric_mean_return():.2f}%")
        print(f"勝率: {self.calculate_win_rate():.2f}%")
        print(f"総トレード数: {len(self.trades)}")
        print(f"勝ちトレード数: {self.get_winning_trades()}")
        print(f"負けトレード数: {self.get_losing_trades()}")
        print(f"最大連勝数: {self.calculate_max_consecutive_wins()}")
        print(f"最大連敗数: {self.calculate_max_consecutive_losses()}")
        print(f"平均保有期間（日）: {self.get_avg_bars_all_trades():.2f}")
        print(f"勝ちトレード平均保有期間（日）: {self.get_avg_bars_winning_trades():.2f}")
        print(f"負けトレード平均保有期間（日）: {self.get_avg_bars_losing_trades():.2f}")
        print(f"平均保有バー数: {self.get_avg_bars_all_trades() * 6:.2f}")  # 4時間足なので1日6バー
        print(f"勝ちトレード平均保有バー数: {self.get_avg_bars_winning_trades() * 6:.2f}")
        print(f"負けトレード平均保有バー数: {self.get_avg_bars_losing_trades() * 6:.2f}")

        # 損益統計の出力
        print("\n=== 損益統計 ===")
        print(f"総利益: {self.calculate_total_profit():.2f}")
        print(f"総損失: {self.calculate_total_loss():.2f}")
        print(f"純損益: {self.calculate_net_profit_loss():.2f}")
        max_profit, max_loss = self.calculate_max_win_loss()
        print(f"最大利益: {max_profit:.2f}")
        print(f"最大損失: {max_loss:.2f}")
        avg_profit, avg_loss = self.calculate_average_profit_loss()
        print(f"平均利益: {avg_profit:.2f}")
        print(f"平均損失: {avg_loss:.2f}")

        # ポジションタイプ別の分析
        print("\n=== ポジションタイプ別の分析 ===")
        print("LONG:")
        print(f"トレード数: {self.get_long_trade_count()}")
        print(f"勝率: {self.get_long_win_rate():.2f}%")
        print(f"総利益: {self.get_long_total_profit():.2f}")
        print(f"総損失: {self.get_long_total_loss():.2f}")
        print(f"純損益: {self.get_long_net_profit():.2f}")
        print(f"最大利益: {self.get_long_max_win():.2f}")
        print(f"最大損失: {self.get_long_max_loss():.2f}")
        print(f"総利益率: {self.get_long_total_profit_percentage():.2f}%")
        print(f"総損失率: {self.get_long_total_loss_percentage():.2f}%")
        print(f"純損益率: {self.get_long_net_profit_percentage():.2f}%")

        print("\nSHORT:")
        print(f"トレード数: {self.get_short_trade_count()}")
        print(f"勝率: {self.get_short_win_rate():.2f}%")
        print(f"総利益: {self.get_short_total_profit():.2f}")
        print(f"総損失: {self.get_short_total_loss():.2f}")
        print(f"純損益: {self.get_short_net_profit():.2f}")
        print(f"最大利益: {self.get_short_max_win():.2f}")
        print(f"最大損失: {self.get_short_max_loss():.2f}")
        print(f"総利益率: {self.get_short_total_profit_percentage():.2f}%")
        print(f"総損失率: {self.get_short_total_loss_percentage():.2f}%")
        print(f"純損益率: {self.get_short_net_profit_percentage():.2f}%")
        
        # リスク指標
        print("\n=== リスク指標 ===")
        max_dd, max_dd_start, max_dd_end = self.calculate_max_drawdown()
        print(f"最大ドローダウン: {max_dd:.2f}%")
        if max_dd_start and max_dd_end:
            print(f"最大ドローダウン期間: {max_dd_start.strftime('%Y-%m-%d %H:%M')} → {max_dd_end.strftime('%Y-%m-%d %H:%M')}")
            print(f"最大ドローダウン期間（日数）: {(max_dd_end - max_dd_start).days}日")
        
        # 全ドローダウン期間の表示
        print("\n=== ドローダウン期間 ===")
        drawdown_periods = self.calculate_drawdown_periods()
        for i, (dd_percent, dd_days, start_date, end_date) in enumerate(drawdown_periods[:5], 1):
            print(f"\nドローダウン {i}:")
            print(f"ドローダウン率: {dd_percent:.2f}%")
            print(f"期間: {start_date.strftime('%Y-%m-%d %H:%M')} → {end_date.strftime('%Y-%m-%d %H:%M')} ({dd_days}日)")
        
        print(f"\nシャープレシオ: {self.calculate_sharpe_ratio():.2f}")
        print(f"ソルティノレシオ: {self.calculate_sortino_ratio():.2f}")
        print(f"カルマーレシオ: {self.calculate_calmar_ratio():.2f}")
        print(f"カルマーレシオ（調整済み）: {self.calculate_calmar_ratio_v2():.2f}")
        print(f"VaR (95%): {self.calculate_value_at_risk():.2f}%")
        print(f"期待ショートフォール (95%): {self.calculate_expected_shortfall():.2f}%")
        print(f"ドローダウン回復効率: {self.calculate_drawdown_recovery_efficiency():.2f}")
        
        # トレード効率指標
        print("\n=== トレード効率指標 ===")
        print(f"プロフィットファクター: {self.calculate_profit_factor():.2f}")
        print(f"ペイオフレシオ: {self.calculate_payoff_ratio():.2f}")
        print(f"期待値: {self.calculate_expected_value():.2f}")
        print(f"悲観的リターンレシオ: {self.calculate_pessimistic_return_ratio():.2f}")
        print(f"アルファスコア: {self.calculate_alpha_score():.2f}")
        print(f"SQNスコア: {self.calculate_sqn():.2f}")

    def calculate_max_consecutive_wins(self) -> int:
        """最大連勝数を計算

        Returns:
            int: 最大連勝数
        """
        if not self.trades:
            return 0

        max_streak = current_streak = 0
        for trade in self.trades:
            if trade.profit_loss > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def calculate_max_consecutive_losses(self) -> int:
        """最大連敗数を計算

        Returns:
            int: 最大連敗数
        """
        if not self.trades:
            return 0

        max_streak = current_streak = 0
        for trade in self.trades:
            if trade.profit_loss < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak