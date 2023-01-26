
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px


st.header('portfolio-opt')
st.subheader('Portfolio Optimization app')
st.write(
    """
    A Python application to help portfolio managers optimize their asset mix using streamlit and yfinance.

    Modern Portfolio Theory suggests that investors should diversify by 
    investing in a range of different assets in order to reduce the overall portfolio risk.

    By diversifying and analysing their data,
    investors are able to **limit risk exposure to an asset class** or sector,
    **minimize the amount of volatility** they experience, and **reduce market fluctuation
    impact** in their investments.

    This app aims to help investors **ideate a balanced portfolio mix**
    allowing you to **enter a range of instruments and show you interactive insights**. 
    You could use this f.e. to help prevent being overly weighted in a single asset class or sector, 
    while maintaining or improving your `expected returns`.

    This app could ultimately help portfolio managers **plan to good expected return, respective to their risk apetite**.
    """
)
st.subheader('Get Started')
st.write('Pass a list of symbols and a number of experiments to start. You need to look for the tickers in https://finance.yahoo.com/')
products = st.text_input("Enter symbols to look for (f.e.): okta msft ^gspc imkta ^ixic gold si=f dax aapl goog", value="amzn aapl goog")

col1, col2 = st.columns(2)
with col1:
    start_date = col1.text_input("Enter a start date (YYYY-MM-DD)", value="2021-01-01")
    num_ports = col1.number_input('No. of simulations: ', 1000, 100000, value=10000)

with col2:
    end_date = col2.text_input("Enter an end date: (YYYY-MM-DD)", value="2023-01-31")
    sd = st.number_input('Seed: ', 1, 999, value=101)
    np.random.seed(sd)

if col2.button('Run portfolio optimization!'):
    st.write('---')
    data = yf.download(products, start=start_date, end=end_date)
    stocks = data.Close
    mean_daily_ret = stocks.pct_change(1).mean()
    stock_normed = stocks/stocks.iloc[0]

    st.subheader("Normalized performance compared")
    st.line_chart(stock_normed)
    
    stock_daily_ret = stocks.pct_change(1)
    log_ret = np.log(stocks/stocks.shift(1))

    st.subheader('Daily Returns Statitics')
    st.write(stock_daily_ret.describe().T)
    st.bar_graph(stock_daily_ret)

    weights = np.array(np.random.random(stocks.shape[1]))
    weights = weights / np.sum(weights)
    exp_ret = np.sum(log_ret.mean() * weights) *252
    exp_vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    all_weights = np.zeros((num_ports,len(stocks.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for ind in range(num_ports):
        weights = np.array(np.random.random(stocks.shape[1])) # Create Random Weights
        weights = weights / np.sum(weights)
        all_weights[ind,:] = weights

        ret_arr[ind] = np.sum((log_ret.mean() * weights) *252)
        vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
        sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

    weights_df = pd.DataFrame(all_weights, columns = stocks.columns)
    results = pd.concat([(weights_df*100).round(2), pd.Series(vol_arr, name='volatility'), pd.Series(ret_arr, name='returns')], axis=1)
    results['sharpe_ratio'] = results.returns/results.volatility

    st.subheader('Pick the best mix for your risk apetite')
    st.write("""
            This is the core result of this application. The idea is that you find a mix of the stocks you picked,
            which gives you the most returns (Y axis) for a volatility (X axis) that you as an investor feel comfortable with.

            For example, in case you feel alright with a 20% volatility (X=0.2) in your portfolio, you shoud pick the mix on the
            top return(maximum Y). This would give you a mix with the most expected returns, for the expected volatility you're aiming for. 
            """)
    fig = px.scatter(results, x='volatility', y='returns', color='sharpe_ratio', hover_data=stocks.columns)
    st.plotly_chart(fig)

    st.write(
        """
        **Disclaimer:** **Not financial advise, always should do your own research before investing**. 
        By using this app, you acknowledge that you are sole responsible for your own investment decisions. 
        and therefore, the author can't be made responsible for their outcomes.
        """
    )
