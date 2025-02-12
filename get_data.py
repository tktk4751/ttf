from binance_historical_data import BinanceDataDumper

tickers = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT", "ATOMUSDT", "UNIUSDT","ARBUSDT","OPUSDT","PEPEUSDT","SEIUSDT","BNBUSDT","SUIUSDT","TIAUSDT","LINKUSDT","WIFUSDT","INJUSDT","AAVEUSDT","RENDERUSDT","JUPUSDT","TAOUSDT","TONUSDT","DOGEUSDT","WLDUSDT","XRPUSDT","NEARUSDT","DOTUSDT"]

durations = ["1h", "2h", "4h", "6h", "8h", "12h"]

if __name__ == '__main__':

  
    for duration in durations:
        data_dumper = BinanceDataDumper(
            path_dir_where_to_dump="./data",
            asset_class="spot",  # spot, um, cm
            data_type="klines",  # aggTrades, klines, trades
            data_frequency=duration,
        )

        data_dumper.dump_data(
            tickers=tickers,
            date_start=None,
            date_end=None,
            is_to_update_existing=False,
            tickers_to_exclude=["UST"],
        )