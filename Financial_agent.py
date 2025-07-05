import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.tavily import TavilyTools
from phi.tools.duckduckgo import DuckDuckGo
import os

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = groq_api_key
TAVILY_API_KEY = "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"

web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama3-70b-8192"),
    tools=[
        DuckDuckGo(),
        TavilyTools(api_key=TAVILY_API_KEY)
    ],
    instructions=["Always Include Sources","Use markdown"],
    show_tools_calls=True,
    markdown=True
)

finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-70b-8192"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        )
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id="llama3-70b-8192"),
    instructions=["Use markdown", "Always include sources", "Avoid repeating previous tasks","Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

st.set_page_config(page_title="Finance & Web Chatbot", page_icon="📈")
st.title("📊 Finance + Web Assistant")
st.write("A Financial chatbot where u can ask anything realted to any stock" \
" whether an analyst recommendation, investment strategies, stock behaviour/pattern" \
" or any latest news related to that stock or company")
st.markdown("""
### ℹ️ Important Stock Ticker Format

- For **NSE** stocks, append `.NS` (e.g., `TCS.NS`, `INFY.NS`)
- For **BSE** stocks, append `.BO` (e.g., `TCS.BO`, `INFY.BO`)
- For US stocks, just use the ticker (e.g., `NVDA`, `AAPL`)
""")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask about a stock, financial topic, or market info...")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input:
    if "its" in user_input.lower() and st.session_state.last_stock_symbol:
        user_input = user_input.replace("its", st.session_state.last_stock_symbol)

    for word in user_input.upper().split():
        if word.endswith(".NS") or word.endswith(".BO") or word.isalpha():
            st.session_state.last_stock_symbol = word.upper()

    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    response_text = ""
    with st.chat_message("assistant"):
        response_box = st.empty()
        for chunk in multi_ai_agent.run(
            message=user_input,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history
            ],
            stream=True,
        ):
            if chunk and hasattr(chunk, "content") and chunk.content:
                response_text += chunk.content
                response_box.markdown(response_text)

    st.session_state.chat_history.append({"role": "assistant", "content": response_text})