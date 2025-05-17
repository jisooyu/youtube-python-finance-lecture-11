import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
from typing import List, Union
import logging

'''
다우존스지표에 포함된 주식을 대상으로 성과가 좋은 주식을 포트폴리오에 포함시켜 포트폴리오 성과가 다우존스 지표를 추월하도록 하는 전략

'''
# type annotation을 사용하여 fetch_data를 정의. type annotation을 사용하면 어떤 data type을 사용해야 하는 것이 명확
def fetch_data(
    tickers: List[str], start_date: Union[str, dt.datetime], end_date: Union[str, dt.datetime], interval: str = '1mo'
) -> pd.DataFrame:
    """
    주어진 기간에 대해 주식의 데이터를 다운 받음

    Parameters:
        tickers (List[str]): 다운 받을 주식 ticker
        start_date (Union[str, dt.datetime]): 기간의 시작 시점
        end_date (Union[str, dt.datetime]): 기간의 종료시점
        interval (str): 데이터의 간격격 (default: '1mo').

    Returns:
        pd.DataFrame: 주식 데이터의 DataFrame
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval)
        return data.dropna(axis=0, how="all")
    except Exception as e:
        logging.error(f"Failed to fetch the data: {e}")
        return pd.DataFrame()
    
# function to calculate portfolio return iteratively
'''
Dow Jones에 포함된 주식으로 portfolio를 구성
월수익률이 가장 낮은 주식은 portfolio에서 제거. 월수익률이 더 나은 다른 주식을 추가
portfolio에 포함된 주식들의의 월평균수익률을 계산하여 monthly_return list에 저장
'''
def pflio(df, m, x):
    """
    포트롤리오의 월별평균수익율을 return
    DF = 다운 받은 모든 주식의 월별 수익율 dataframe
    m = portfolio에 있는 주식의 숫자
    x = 매월 제거하는 저성과 주식의 숫자
    """
    portfolio = []
    monthly_ret = [0]
    for i in range(len(df)):
        if portfolio:
            # 월별 포트롤리오의 평균수익율 을 monthly_ret에 첨부
            monthly_ret.append(df[portfolio].iloc[i,:].mean())
            # 저성과 주식을 X 개만큼 추출출
            bad_stocks = df[portfolio].iloc[i,:].sort_values(ascending=True)[:x].index.tolist()
            # 저성과 주식을 제외한 포트폴리오를 새로 만듬
            portfolio = [t for t in portfolio if t not in bad_stocks]
        # 포트롤리오에 새로 추가할 주식의 수를 계산 (fill)
        fill = m - len(portfolio)
        # 포트롤리오에 새로 추가할 주식을 fill개 만큼 선택택
        new_picks = [t for t in df.iloc[i, :].nlargest(fill).index if t not in portfolio] 
        # fill 개의 선택한 주식을 포트폴리오에 포함
        portfolio.extend(new_picks)
        print(portfolio)
    # monthly_ret를 DataFrame으로 만듬
    monthly_ret_df = pd.DataFrame(np.array(monthly_ret),columns=["mon_ret"])
    return monthly_ret_df

def CAGR(df):
    "투자전략(포트폴리오 전략)의 연간누적수익율 계산"
    df["cum_return"] = (1 + df["mon_ret"]).cumprod()
    n = len(df)/12
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(df, period = 12):
    '''
    연간변동성(risk)을 계산. 표준화를 위해 연간으로 변동성을 변환
    period = 일일주가를 사용했다면 np.sqrt(252), 월별주가의 경우에는 np.sqrt(12)
    '''
    vol = df["mon_ret"].std() * np.sqrt(period) #
    return vol

def sharpe(df,rf):
    "sharpe ratio - 포트폴리오 연간수익율에서 risk free(미국국채)의 수익율을 제거한 후 변동성으로 나눔. 미국국채 수익율 이상 얻은 추가 수익률과 변동성 비교. 높을 수록 좋은 주식식"
    sr = (CAGR(df) - rf)/volatility(df)
    return sr

def max_dd(df):
    "특정기간 중 가장 큰 하락 % 측정. 최고가에 사서 최저가에 매각하는 경우. 낮을 수록 좋은 주식"
    df["cum_return"] = (1 + df["mon_ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd

def plot_return(df, DJI_ret):
    #visualization
    fig, ax = plt.subplots()
    plt.plot((1+pflio(df,6,3)).cumprod())
    plt.plot((1+DJI_ret["mon_ret"].reset_index(drop=True)).cumprod())
    plt.title("Index Return vs Strategy Return")
    plt.ylabel("cumulative return")
    plt.xlabel("months")
    ax.legend(["Strategy Return","Index Return"])
    fig.set_facecolor('#CDC1FF')

def plot_bar_kpi(kpi_pflio, kpi_dji):
    fig, ax = plt.subplots()

    # Define the positions for the bars
    categories = list(kpi_pflio.keys())  # KPI names
    x = np.arange(len(categories))  # Positions for each KPI
    width = 0.35  # Width of the bars

    # Plot the bars for kpi_pflio and kpi_dji
    bars_pflio = ax.bar(x - width / 2, kpi_pflio.values(), width, label='Portfolio')
    bars_dji = ax.bar(x + width / 2, kpi_dji.values(), width, label='DJI')

    # Add labels, title, and legend
    ax.set_xlabel('KPIs')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of KPIs: Portfolio vs DJI')
    ax.set_xticks(x)  # Set the x-axis positions
    ax.set_xticklabels(categories)  # Set the x-axis labels
    ax.legend()

    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on top of each bar
    for bar in bars_pflio:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=9)

    for bar in bars_dji:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=9)
        
    fig.set_facecolor('#CDC1FF')


if __name__ =='__main__':
    '''
    과거의 시작시기로 돌아가 당시 DJ에 포함된 주식을 선택하여 backtesting을 해야 함. 아니면 survivor ship bias위험 
    '''
    # Define tickers and date range
    tickers = ["MSFT","NVDA","META","AMD","QCOM","MU","NVDA","CRWD", "FTNT", "PANW"]
    # tickers = ["MSFT", "IBM", "HD", "GS", "XOM", "DIS", "DWDP", "KO", "CSCO", "CVX", "CAT", "BA", "AAPL", "AXP", "MMM"]
    start = (dt.datetime.today() - dt.timedelta(days=1825)).strftime('%Y-%m-%d')
    end = dt.datetime.today().strftime('%Y-%m-%d')

    # Fetch and clean data
    df_ohlcv = fetch_data(tickers, start, end, interval='1mo')

    if df_ohlcv.empty:
        logging.error("No valid data fetched for any tickers.")
        print("tickers 데이터가 비어 있어서 종료")
    else:
        df_cls = df_ohlcv['Close'].dropna(how="all")
        df_cls_ret = df_cls.pct_change(fill_method=None).fillna(0)
        print("tickers 데이터처리완료.")

    # fetch DJI data
    DJI_mon = fetch_data(['^DJI'], start, end, interval='1mo')

    if DJI_mon.empty:
        logging.error("No data available. Exiting the script.")
        print("DJI 데이터가 비어 있어서 종료")

    else:
        df_DJI = DJI_mon['Close'].dropna(how='all')
        # calculate DJI percent change
        df_DJI_mon = df_DJI.pct_change(fill_method=None).fillna(0)
        # to standardize the format of the data before passing it to methods like plot_return
        df_DJI_ret = pd.DataFrame(np.array(df_DJI_mon), columns=['mon_ret'])
        print("DJI 데이터처리완료.")
 
     # visualize DJI and porfolio performance
    plot_return(df_cls_ret, df_DJI_ret)

    '''
    포트폴리오 전략과 DJI의 KPI 비교 
    '''
    #calculating KPIs for portfolio strategy
    cagr_pflio = CAGR(pflio(df_cls_ret,6,3))
    sharpe_pflio = sharpe(pflio(df_cls_ret,6,3),0.025)
    max_dd_pflio = max_dd(pflio(df_cls_ret,6,3)) 
    print(f"cagr, sharpe, and max_dd for pflio are {cagr_pflio:.2f}, {sharpe_pflio:.2f}, {max_dd_pflio:.2f}")
  
    #calculating KPIs for DJI Index
    cagr_dji = CAGR(df_DJI_ret)
    sharpe_dji  = sharpe(df_DJI_ret,0.025)
    max_dd_dji = max_dd(df_DJI_ret)
    print(f"cagr, sharpe, and max_dd for dji are {cagr_dji:.2f}, {sharpe_dji:.2f}, {max_dd_dji:.2f}")

    # Populate the dictionary
    kpi_pflio = {
        'CAGR': cagr_pflio,
        'sharpe': sharpe_pflio,
        'max_dd': max_dd_pflio
    }

    # # Calculate KPIs for DJI
    kpi_dji = {
        'CAGR': cagr_dji ,
        'sharpe': sharpe_dji,
        'max_dd':max_dd_dji
    }
 
    # Create a bar chart with bars side by side
    plot_bar_kpi(kpi_pflio, kpi_dji)