from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import numpy as np

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open for hackathon
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FinancialQuery(BaseModel):
    query: str

# Embedded training dataset
TRAINING_DATA = [
    ("budget R5000 for monthly expenses", "budget"),
    ("how to manage my budget", "budget"),
    ("spending plan for R3000 income", "budget"),
    ("create a budget for family", "budget"),
    ("pay off credit card debt", "debt"),
    ("manage R10000 loan", "debt"),
    ("reduce debt quickly", "debt"),
    ("owe R5000 on credit", "debt"),
    ("save for emergency fund", "savings"),
    ("build savings R2000 monthly", "savings"),
    ("stokvel for community savings", "savings"),
    ("save R10000 in a year", "savings"),
    ("invest R5000 safely", "investment"),
    ("stocks for beginners", "investment"),
    ("etf investment options", "investment"),
    ("invest in bonds", "investment"),
    ("improve my credit score", "credit_score"),
    ("check credit rating", "credit_score"),
    ("bad credit help", "credit_score"),
    ("credit report advice", "credit_score"),
    ("join a stokvel with 10 people", "stokvel"),
    ("stokvel savings R500 monthly", "stokvel"),
    ("group savings plan", "stokvel"),
    ("community savings advice", "stokvel"),
    ("retire with R1000000", "retirement"),
    ("pension plan for age 30", "retirement"),
    ("retirement fund options", "retirement"),
    ("save for retirement at 65", "retirement"),
]

# Advice templates with AR emphasis
ADVICE_TEMPLATES = {
    "budget": """
**Budgeting Advice (Enhanced by AR Engagement)**
- **50/30/20 Rule**: Allocate 50% to needs (rent, groceries), 30% to wants (entertainment), 20% to savings/debt ([Investopedia](https://www.investopedia.com)).
- **Track Expenses**: Use [22seven](https://www.22seven.com) or [YNAB](https://www.ynab.com).
- **Automate Savings**: Set up transfers to a [SARS tax-free account](https://www.sars.gov.za).
- **AR Integration**: Use the AR camera scan to visualize spending goals in real-world contexts.
""",
    "debt": """
**Debt Management (Enhanced by AR Engagement)**
- **Snowball Method**: Pay smallest debts first ([Debt.org](https://www.debt.org)).
- **Avalanche Method**: Prioritize high-interest debts ([MoneySmart.gov](https://www.moneysmart.gov)).
- **Negotiate**: Contact creditors for lower rates ([Nedbank](https://www.nedbank.co.za)).
- **AR Integration**: Visualize debt reduction progress via AR camera scan.
""",
    "savings": """
**Savings Plan (Enhanced by AR Engagement)**
- **Emergency Fund**: Save 3-6 months of expenses ([MoneySmart.gov](https://www.moneysmart.gov)).
- **Automate**: Schedule transfers to a high-yield account ([FNB](https://www.fnb.co.za)).
- **Stokvel**: Join a community savings group ([Fin24](https://www.fin24.com)).
- **AR Integration**: Use AR to project savings goals in your environment.
""",
    "investment": """
**Investment Options (Enhanced by AR Engagement)**
- **Low-Risk**: Start with ETFs via a tax-free account ([SARS.gov.za](https://www.sars.gov.za)).
- **Diversify**: Spread across stocks, bonds, property ([Investopedia](https://www.investopedia.com)).
- **Learn**: Use [EasyEquities](https://www.easyequities.co.za) for beginners.
- **AR Integration**: Visualize investment growth with AR camera features.
""",
    "credit_score": """
**Credit Score Improvement (Enhanced by AR Engagement)**
- **Check Score**: Free report from [TransUnion](https://www.transunion.com).
- **Pay on Time**: Timely payments boost your score.
- **Reduce Debt**: Keep card balances below 30% of limits.
- **AR Integration**: Use AR to track credit score progress visually.
""",
    "stokvel": """
**Stokvel Guide (Enhanced by AR Engagement)**
- **Join**: Contribute monthly to a group pool ([NASASA](https://www.nasasa.co.za)).
- **Set Rules**: Agree on contributions and payouts.
- **Bank Account**: Use [FNB Stokvel Account](https://www.fnb.co.za).
- **AR Integration**: Visualize stokvel contributions via AR camera scan.
""",
    "retirement": """
**Retirement Planning (Enhanced by AR Engagement)**
- **Start Early**: Contribute to a retirement annuity ([Sanlam](https://www.sanlam.co.za)).
- **Tax Benefits**: Use [SARS tax-free accounts](https://www.sars.gov.za).
- **Employer Plans**: Maximize pension contributions.
- **AR Integration**: Project retirement goals with AR visualization.
""",
}

# Contextual advice generator
def generate_contextual_advice(category: str, context: dict) -> str:
    base_advice = ADVICE_TEMPLATES.get(category, "")
    if not context:
        return base_advice
    if category == "budget" and context.get("income"):
        return f"""
{base_advice}
**Personalized Budget Plan**
- **Income**: {context['income']:.2f} ZAR
- **Allocation**:
  - Needs (50%): {context['income'] * 0.5:.2f} ZAR
  - Wants (30%): {context['income'] * 0.3:.2f} ZAR
  - Savings/Debt (20%): {context['income'] * 0.2:.2f} ZAR
- **AR Tip**: Use the AR camera to scan your environment and visualize budgeting goals.
"""
    elif category == "debt" and context.get("debt_amount"):
        return f"""
{base_advice}
**Debt Strategy**
- **Amount**: {context['debt_amount']:.2f} ZAR
- **Type**: {context.get('debt_type', 'General').capitalize()}
- **Action**: Focus on {context.get('debt_type', 'debts')} with highest interest; contact [Capitec](https://www.capitec.co.za).
- **AR Tip**: Visualize debt reduction progress with AR scan.
"""
    elif category == "savings" and context.get("savings_goal"):
        months = context.get('timeline_months', 12)
        return f"""
{base_advice}
**Savings Plan**
- **Goal**: {context['savings_goal']:.2f} ZAR
- **Timeline**: {context.get('timeline', '1 year')}
- **Monthly Savings**: {context['savings_goal'] / max(1, months):.2f} ZAR
- **AR Tip**: Use AR to project savings goals in your environment.
"""
    elif category == "investment" and context.get("investment_amount"):
        return f"""
{base_advice}
**Investment Plan**
- **Amount**: {context['investment_amount']:.2f} ZAR
- **Risk**: {context.get('risk_profile', 'Moderate')}
- **Recommendation**: Start with [Satrix MSCI World ETF](https://satrix.co.za).
- **AR Tip**: Visualize investment growth with AR camera.
"""
    elif category == "stokvel" and context.get("contribution"):
        return f"""
{base_advice}
**Stokvel Plan**
- **Contribution**: {context['contribution']:.2f} ZAR/month
- **Group Size**: {context.get('group_size', 10)} members
- **Pool**: {context['contribution'] * context.get('group_size', 10):.2f} ZAR/month
- **Payout**: Every {context.get('payout_cycle', 12)} months
- **AR Tip**: Use AR to visualize stokvel contributions.
"""
    elif category == "retirement" and context.get("retirement_goal"):
        years = context.get('retirement_age', 65) - context.get('age', 30)
        return f"""
{base_advice}
**Retirement Plan**
- **Goal**: {context['retirement_goal']:.2f} ZAR
- **Age**: {context.get('age', 'Unknown')}
- **Monthly Contribution**: {context['retirement_goal'] / max(1, years * 12):.2f} ZAR
- **AR Tip**: Project retirement goals with AR visualization.
"""
    return base_advice

# Extract context from query
def extract_context(query: str) -> dict:
    context = {}
    # Income (e.g., "budget R5000")
    income_match = re.search(r'\b(R?\d{1,6}(?:\.\d{2})?)\b', query, re.IGNORECASE)
    if income_match:
        context['income'] = float(income_match.group(1).replace('R', ''))
    # Debt
    debt_match = re.search(r'\b(debt|owe)\s+R?(\d{1,6})\b', query, re.IGNORECASE)
    if debt_match:
        context['debt_amount'] = float(debt_match.group(2))
        context['debt_type'] = 'credit card' if 'card' in query.lower() else 'loan'
    # Savings
    savings_match = re.search(r'\b(save|saving)\s+R?(\d{1,6})\b', query, re.IGNORECASE)
    if savings_match:
        context['savings_goal'] = float(savings_match.group(2))
    # Timeline
    timeline_match = re.search(r'\b(\d+)\s*(month|year)s?\b', query, re.IGNORECASE)
    if timeline_match:
        context['timeline'] = timeline_match.group(0)
        context['timeline_months'] = int(timeline_match.group(1)) * (12 if 'year' in timeline_match.group(2).lower() else 1)
    # Investment
    invest_match = re.search(r'\b(invest|investment)\s+R?(\d{1,6})\b', query, re.IGNORECASE)
    if invest_match:
        context['investment_amount'] = float(invest_match.group(2))
        context['risk_profile'] = 'Low' if 'safe' in query.lower() else 'Moderate'
    # Stokvel
    stokvel_match = re.search(r'\b(stokvel|group)\s+R?(\d{1,5})\b', query, re.IGNORECASE)
    if stokvel_match:
        context['contribution'] = float(stokvel_match.group(2))
        context['group_size'] = int(re.search(r'\b(\d+)\s*(member|people)s?\b', query, re.IGNORECASE).group(1)) if re.search(r'\b(\d+)\s*(member|people)s?\b', query, re.IGNORECASE) else 10
        context['payout_cycle'] = int(re.search(r'\b(\d+)\s*month', query, re.IGNORECASE).group(1)) if re.search(r'\b(\d+)\s*month', query, re.IGNORECASE) else 12
    # Retirement
    age_match = re.search(r'\b(\d{1,2})\s*(year|yr)s?\s*old\b', query, re.IGNORECASE)
    if age_match:
        context['age'] = int(age_match.group(1))
    retirement_goal_match = re.search(r'\bretire\s+R?(\d{1,7})\b', query, re.IGNORECASE)
    if retirement_goal_match:
        context['retirement_goal'] = float(retirement_goal_match.group(1))
        context['retirement_age'] = int(re.search(r'\bretire\s+at\s+(\d{1,2})\b', query, re.IGNORECASE).group(1)) if re.search(r'\bretire\s+at\s+(\d{1,2})\b', query, re.IGNORECASE) else 65
    return context

# Train model
def train_financial_model():
    queries, labels = zip(*TRAINING_DATA)
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(queries)
    y = labels
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X, y)
    return model, vectorizer

# Global model and vectorizer
MODEL, VECTORIZER = train_financial_model()

def financial_advisor(query: str) -> str:
    """
    ML-based financial advisor using logistic regression to categorize queries and generate SA-specific advice with AR integration.
    """
    if not query or not query.strip():
        return """
**Welcome to Financial World Quest**
- **Get Started with AR**: Use the AR camera scan to start your financial quest.
- **Ask Questions**: Query about budgeting, debt, savings, or stokvels.
- **Resources**: [MyMoney.gov](https://www.mymoney.gov), [SARS.gov.za](https://www.sars.gov.za).
- **Example**: "Budget R5000" or "Manage credit card debt".
"""

    # Predict category
    query_vec = VECTORIZER.transform([query.lower()])
    category = MODEL.predict(query_vec)[0]
    context = extract_context(query)
    return generate_contextual_advice(category, context)

@app.post("/financial-advice")
async def get_financial_advice(query: FinancialQuery):
    response = financial_advisor(query.query)
    return {"advice": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)