import streamlit as st
import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.google import Gemini
import yfinance as yf
import plotly.graph_objects as go
load_dotenv()
def compare_stock(symbols):
    data={}
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        history = stock.history(period="1y")
        if history.empty:
            print(f"No data found for {symbol}")
            continue
        data[symbol] = history['Close'].pct_change().sum()

    return data

analyst_agent = Agent(
    model = Gemini(id="gemini-2.5-pro"),
    description="AI agent that thouroughly analyses stock and compare its perfromance over time",
    instructions=["Retrieve and compare stock perfromance from yahoo finance."
                  "Calculate the percentage change over a 12-month period.",
                  "Rank Stocks based on their relative performance."],
    show_tool_calls=True,
    markdown=True
)

def market_analysis(symbols):
    data = compare_stock(symbols)
    if not data:
        return "No data available for the provided stock symbols."
    analyzed_data = analyst_agent.run(f"Compare these stock performances: {data}")
    return analyzed_data.content

def company_info(symbol):
    stock = yf.Ticker(symbol)
    return {
        "name": stock.info.get("longName", "N/A"),
        "sector": stock.info.get("sector", "N/A"),
        "market_cap": stock.info.get("marketCap", "N/A"),
        "summary": stock.info.get("longBusinessSummary", "N/A")
        }

def company_news(symbol):
    stock = yf.Ticker(symbol)
    news = stock.news[:5]
    return news

company_researcher  = Agent(
    model = Gemini(id="gemini-2.5-pro"),
    description = "AI agent that fetches company profiles, financials and latest news.",
    instructions = [
        "Retrieve company profile, financials and latest news from Yahoo Finance.",
        "Provide a summary of the company, its sector, market cap and a brief business summary.",
        "Fetch and summarizes the latest news articles related to the company and are relevant to investors."
    ],
    show_tool_calls=True,
    markdown=True
)
def company_analysis(symbol):
    info = company_info(symbol)
    news = company_news(symbol)
    response = company_researcher.run(
        f"Provide an analysis of the company {info['name']} in the {info['sector']} sector.\n"
        f"Market cap: {info['market_cap']} \n" 
        f"Summary: {info['summary']} \n"
        f"Latest news: {news}"
    )
    return response.content

stock_agent = Agent(
    model = Gemini(id = "gemini-2.5-pro"),
    description = "AI agent that provides stock market analysis and insights.",
    instructions = [
        "Analyze stock performance, Company Fundamentals and latest news related to that stock that can affect the price.",
        "Evaluate risk-reward potential, compare stocks and industry trends.",
        "provide top stock recommendations based on the analysis for investors that can have good profit potential.",
    ],
    show_tool_calls=True,
    markdown=True
)
def stock_recommendations(symbols):
    market_data = market_analysis(symbols)
    data = {}
    for symbol in symbols:
        data[symbol] = company_analysis(symbol)

    analysis = stock_agent.run(
        f"Based on the market analysis: {market_data}\n"
        f"And company news: {data}\n"
        f"Provide stock recommendations for these symbols: {symbols}"
    )
    return analysis.content

team_lead = Agent(
    model = Gemini(id="gemini-2.5-pro"),
    description = "Aggregates stock analysis, comapany research and investment strategies.",
    instructions = [
        "Compile stock performance, company analysis, recommendations and investment strategies.",
        "Ensure all insights are structured in an investor-friendly report.",
        "Rank the top stocks based on combined analysis and provide actionable insights."
    ],
    markdown=True
)

def final_report(symbols):
    market_data = market_analysis(symbols)
    company_data = [company_analysis(symbol) for symbol in symbols]
    stock_recs = stock_recommendations(symbols)

    final_repo = team_lead.run(
        f"Market Analysis: {market_data}\n"
        f"Company Analysis: {company_data}\n"
        f"Stock Recommendations: {stock_recs}\n"
        f"Provide the full analysis of each stock with Fundamentals and market news\n"
        f"Generate a final ranked list in ascending order on which should I buy."
    )
    return final_repo.content

st.set_page_config(page_title="Investment Assistant",page_icon="📈")

st.title("📊 Investment Assistant")

st.markdown("""
### ℹ️ Important Stock Ticker Format

- For **NSE** stocks, append `.NS` (e.g., `TCS.NS`, `INFY.NS`)
- For **BSE** stocks, append `.BO` (e.g., `TCS.BO`, `INFY.BO`)
- For US stocks, just use the ticker (e.g., `NVDA`, `AAPL`)
""")
input_symbols = st.text_input("Enter stock symbols (comma-separated, e.g., AAPL, GOOGL, TSLA):")
stocks_symbols = [symbol.strip() for symbol in input_symbols.split(",")]

if st.button("Generate Report"):
    if not stocks_symbols or stocks_symbols == [""]:
        st.error("Please enter valid stock symbols.")
    else:
        report = final_report(stocks_symbols)
        st.markdown(report)
        st.info("This report provides detailed insights, including market performance, company analysis, and investment recommendations.")
        st.markdown("### 📈 Stock Performance over past 12 months")
        stock_data = yf.download(stocks_symbols, period="1y")['Close']
        fig = go.Figure()
        for symbol in stocks_symbols:
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[symbol], mode='lines', name=symbol))

        fig.update_layout(title="Stock Performance Over the Last 12 Months",
                          xaxis_title="Date",
                          yaxis_title="Price (in USD)",
                          template="plotly_dark")
        st.plotly_chart(fig)