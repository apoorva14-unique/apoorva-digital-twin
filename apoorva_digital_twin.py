import os
import re
import math
import datetime as dt
from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from a local .env file if present

# ============================================================
# 1. API KEY — read from an environment variable, NOT hardcoded
#    (this keeps your key out of GitHub)
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Create a .env file (see .env.example) "
        "with GROQ_API_KEY=your_key_here, or set it as an environment "
        "variable before running this script."
    )

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import gradio as gr

# ============================================================
# 2. KNOWLEDGE BASE — your resume + bio, as plain text
#    (Edit this directly to update facts, add projects, etc.)
# ============================================================
KNOWLEDGE_TEXT = """
ABOUT ME

I'm Apoorva (Usurupati Apoorva), a final-year B.Tech student in Computer Science &
Engineering with a specialization in AI & ML at MLR Institute of Technology, Hyderabad,
graduating in 2027. My CGPA is 8.30/10.0. I'm based in Hyderabad, India.
Contact: usurupatiapoorva17@gmail.com | 6301276492

WHAT I DO

I'm a data analyst at heart — I like taking messy, real-world data and turning it into
something a business can actually act on. My sweet spot is the full pipeline: cleaning
and transforming raw data with Python and SQL, running statistical analysis, and then
building interactive Power BI dashboards that surface the story hiding in the numbers.
I'm comfortable with Python, SQL, Power BI, Tableau, Pandas, NumPy, Matplotlib,
Scikit-learn, Excel, MySQL, Git/GitHub, and Jupyter.

CURRENT ROLE

I'm currently a Data Analytics Intern at Zaalima Development Pvt. Ltd. (Apr 2026 - July
2026, virtual). I clean, transform, and analyze structured datasets using Python and SQL,
translating raw data into business-ready insights for stakeholder reporting, and I build
dashboards and KPI reports that help teams make faster, data-driven decisions.

Before that, I was a Data Science Intern at Main Flow Services and Technologies Pvt. Ltd.
(Feb-Mar 2025, virtual), where I built and evaluated three ML models end-to-end: a House
Price Prediction regression model, a Digit Recognition classifier, and a Customer
Segmentation model using K-Means — covering everything from EDA to model evaluation,
using Python, Pandas, Scikit-learn, and Seaborn.

PROJECT: ALPHAPULSE — Financial Analytics & Stock Market Dashboard

This is my flagship project from my Zaalima internship. Stack: Python, Pandas, NumPy,
Matplotlib, Power BI, yfinance API. I pulled 5 years (2020-2024) of stock data for 11
assets using the yfinance API, then ran returns analysis, a correlation matrix, Value at
Risk, and a Monte Carlo Simulation — landing on a portfolio average risk score of 0.02%.
I engineered a long-format dataset of 13,827 rows with 20-day and 50-day moving averages,
rolling volatility, cumulative return percentages, and trend signals. From that I built an
interactive Power BI dashboard with 4 slicers, 5 KPI cards (Average Portfolio Return of
137.34%, Investment Growth of 2.39K%), and 6 visualizations. I also delivered a Moving
Average Strategy, stock forecasting, portfolio optimization, and an automated_pipeline.py
script for one-command reproducible execution across all 4 weekly project milestones. I
iterated the dashboard through about five rounds of feedback, fixing things like switching
to a log scale for cumulative returns, correcting aggregations from Sum to Average, and
reshaping the dataset so the slicers actually worked interactively.

PROJECT: CONSUMER360 — Customer Analytics & Segmentation Engine

Stack: Python, Pandas, Power BI, Excel. I cleaned and analyzed 536,000+ retail transaction
records spanning 4,338 customers and $8.91M in revenue. I applied RFM Analysis, Customer
Lifetime Value, Market Basket Analysis, and Cohort Analysis to segment customers into 4
groups, and found that just two of those segments — Champions and Loyal Customers — were
driving 80% of total revenue. I built an interactive Power BI dashboard with segment,
country, and recency/monetary filters so stakeholders could explore the patterns
themselves.

CERTIFICATIONS & TRAINING

- Deloitte Australia - Data Analytics Job Simulation, Forage (Jun 2026)
- Tata GenAI-Powered Data Analytics Job Simulation, Forage (Aug 2025)
- Introduction to Data Science, Cisco Networking Academy (Apr 2025)
- Introduction to Internet of Things, NPTEL - Elite, 81% (2025)

EDUCATION

B.Tech, Computer Science & Engineering (AI & ML), MLR Institute of Technology, Hyderabad
(2023-2027), CGPA 8.30/10.0.
Intermediate (XII), SV Junior College, Telangana Board of Intermediate Education
(2021-2023), 92%.

WHAT I'M LOOKING FOR

I'm actively looking for Data Analyst internship and full-time roles where I can work with
real business data end-to-end — from cleaning and analysis through to dashboarding and
reporting. I'm particularly drawn to roles where the output actually gets used by
stakeholders to make decisions, not just analysis for its own sake.

HOW I'D DESCRIBE MYSELF

[EDIT THIS SECTION IN YOUR OWN WORDS before running for the demo:]
- Working style / personality: (2-3 sentences)
- What got me into data analytics: (2-3 sentences)
- A challenge I solved on AlphaPulse or Consumer360: (2-3 sentences)
- Outside of academics/internships: (2-3 sentences)
"""

# ============================================================
# 3. BUILD THE VECTOR STORE (runs automatically at startup)
# ============================================================
def build_vector_store():
    doc = Document(page_content=KNOWLEDGE_TEXT)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents([doc])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print(f"Knowledge base ready: {len(chunks)} chunks embedded.")
    return vector_store.as_retriever(search_kwargs={"k": 4})


RETRIEVER = build_vector_store()

# ============================================================
# 4. TOOLS
# ============================================================

@tool
def get_background_info(query: str) -> str:
    """
    Retrieve facts about Apoorva's background: education, work experience,
    projects (AlphaPulse, Consumer360), technical skills, and certifications.
    Use this for any question about who Apoorva is or what she has done.
    """
    results = RETRIEVER.invoke(query)
    if not results:
        return "No relevant background information found."
    return "\n\n".join(d.page_content for d in results)


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression, e.g. '15% of 2400' or '(137.34 - 100) / 100'.
    Supports +, -, *, /, %, parentheses, and common math functions like sqrt, round.
    """
    try:
        pct_match = re.match(r"\s*([\d.]+)\s*%\s*of\s*([\d.]+)\s*$", expression, re.IGNORECASE)
        if pct_match:
            pct, base = float(pct_match.group(1)), float(pct_match.group(2))
            return str((pct / 100) * base)

        allowed_names = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
        allowed_names.update({"abs": abs, "round": round})
        expression = expression.replace("%", "/100")
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Could not evaluate that expression: {e}"


@tool
def days_until_deadline(target_date: str) -> str:
    """
    Calculate how many days remain between today and a target date.
    target_date must be in YYYY-MM-DD format, e.g. '2026-08-15'.
    """
    try:
        today = dt.date.today()
        target = dt.datetime.strptime(target_date, "%Y-%m-%d").date()
        delta = (target - today).days
        if delta > 0:
            return f"Today is {today.isoformat()}. {delta} day(s) remain until {target_date}."
        elif delta == 0:
            return f"Today is {today.isoformat()}, which IS the target date {target_date}."
        else:
            return f"Today is {today.isoformat()}. {target_date} was {abs(delta)} day(s) ago."
    except ValueError:
        return "Please provide the date in YYYY-MM-DD format."


web_search = DuckDuckGoSearchRun(
    name="web_search",
    description="Search the web for current information — recent news, company info, or anything not in the resume/bio.",
)

TOOLS = [get_background_info, calculator, days_until_deadline, web_search]

# ============================================================
# 5. AGENT
# ============================================================
SYSTEM_PROMPT = """You are Apoorva's Digital Twin — an AI agent that represents
Usurupati Apoorva, a final-year B.Tech CSE (AI & ML) student and Data Analyst.

Rules:
1. Always answer in the FIRST PERSON, as if you are Apoorva herself.
2. For any question about background, skills, projects, internships, education,
   or certifications, ALWAYS call get_background_info first to ground your
   answer in real facts. Never invent details.
3. Speak in a confident, concise, results-oriented tone, referencing specific
   metrics where relevant (e.g. "536K+ transaction records", "137.34% average
   portfolio return").
4. Use calculator, days_until_deadline, and web_search whenever a question
   calls for them, even if unrelated to background.
5. If you don't know something and no tool can help, say so honestly.
"""

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    max_tokens=2048,
    api_key=GROQ_API_KEY,
)

agent = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)


def ask_digital_twin(message, history=None):
    response = agent.invoke({"messages": [{"role": "user", "content": message}]})
    return response["messages"][-1].content


# ============================================================
# 6. GRADIO UI — custom theme + styling
# ============================================================

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
    body_background_fill_dark="linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
    block_background_fill="#111827",
    block_background_fill_dark="#111827",
    block_border_color="#1e293b",
    block_title_text_color="#e2e8f0",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    button_primary_background_fill="linear-gradient(90deg, #14b8a6 0%, #0d9488 100%)",
    button_primary_text_color="#0f172a",
    chatbot_text_size="md",
)

CUSTOM_CSS = """
#header {
    text-align: center;
    padding: 28px 16px 8px 16px;
}
#header h1 {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #2dd4bf, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
#header p {
    color: #94a3b8;
    font-size: 0.95rem;
}
#badges {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
    margin-bottom: 6px;
}
.badge {
    background: #1e293b;
    color: #2dd4bf;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
}
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}
/* Chat bubbles render on a light background — force dark, readable text inside them */
#chat-window .message,
#chat-window .message *,
#chat-window .prose,
#chat-window .prose * {
    color: #0f172a !important;
}
#chat-window h1, #chat-window h2, #chat-window h3,
#chat-window h4, #chat-window h5, #chat-window h6 {
    color: #0f172a !important;
    font-weight: 700 !important;
}
#chat-window code {
    color: #0d9488 !important;
    background: #e2e8f0 !important;
}
#chat-window .bot,
#chat-window .user {
    background: #f8fafc !important;
}
"""

HEADER_HTML = """
<div id="header">
  <h1>⚡ Apoorva's Digital Twin</h1>
  <p>Data Analyst · AI &amp; ML Engineer · Ask me about my work, or hand me a quick task</p>
  <div id="badges">
    <span class="badge">📊 Power BI</span>
    <span class="badge">🐍 Python &amp; SQL</span>
    <span class="badge">🧮 Calculator Tool</span>
    <span class="badge">📅 Deadline Tracker</span>
    <span class="badge">🔎 Web Search</span>
  </div>
</div>
"""

if __name__ == "__main__":
    with gr.Blocks(theme=THEME, css=CUSTOM_CSS, title="Apoorva's Digital Twin") as demo:
        gr.HTML(HEADER_HTML)
        gr.ChatInterface(
            fn=ask_digital_twin,
            chatbot=gr.Chatbot(
                height=520,
                avatar_images=(None, "🤖"),
                show_label=False,
                elem_id="chat-window",
            ),
            examples=[
                "Tell me about yourself",
                "Walk me through the AlphaPulse project",
                "What's 15% of 2400?",
                "How many days until 2026-12-31?",
                "What's the latest news on Power BI?",
            ],
        )

    is_render = os.environ.get("RENDER") is not Non
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=not is_render,
    )