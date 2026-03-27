# Trader Behavior & Bitcoin Market Sentiment Analysis

## Junior Data Scientist Assignment - PrimeTrade.AI

### 📊 Project Overview

This project analyzes the relationship between **trader performance on Hyperliquid** and **Bitcoin market sentiment** (Fear/Greed Index) to uncover hidden patterns that can drive smarter trading strategies.

### 🎯 Key Findings

| Metric | Value |
|--------|-------|
| Total Trades Analyzed | 211,218 |
| Unique Traders | 32 |
| Total Trading Volume | $1.19 Billion |
| Total Net PnL | $10.25 Million |
| Overall Win Rate | 83.20% |
| Smart Money Traders Identified | 23 |

### 📈 Key Insights

#### 1. Sentiment-Performance Correlation
- **Win rate during Fear periods: 84.42%** (higher than Greed: 82.45%)
- Chi-square test confirms trading direction significantly correlates with sentiment (p < 0.001)
- Smart money traders show consistent profitability across all sentiment regimes

#### 2. Smart Money Behavior
- Top 23 "Smart Money" traders account for **$9.75M** of the total $10.25M profits
- These traders maintain >50% win rate, >10 trades, and profit factor >1.5
- Smart money performs best during **Extreme Greed** ($63.97 avg PnL per trade)

#### 3. Trading Direction Insights
- Long/short ratios shift significantly with market sentiment
- During Fear: Traders tend to go long more frequently (contrarian opportunity)
- During Greed: More balanced long/short distribution

#### 4. Coin Analysis
- **BTC, HYPE, SOL, ETH** are the most traded assets
- **@107 token** shows highest total PnL ($2.78M)
- **TRUMP** shows significant losses (-$364K)

### 🔧 Trading Recommendations

**Risk Management:**
- Reduce position sizes during extreme sentiment (Fear < 25 or Greed > 75)
- Set tighter stop-losses during fear periods
- Monitor smart money behavior as a leading indicator

**Strategy Development:**
- Consider mean-reversion strategies during extreme fear
- Trend-following may work better during greed phases
- Adapt long/short bias based on current sentiment regime

**Portfolio Optimization:**
- Diversify across coins to reduce single-asset risk
- Allocate more capital during neutral sentiment (lower volatility)
- Study and potentially mirror smart money positioning

### 📁 Project Structure

```
├── data/
│   ├── historical_trader_data.csv    # Hyperliquid trader data
│   ├── fear_greed_index.csv          # Bitcoin Fear/Greed Index
│   ├── analysis_summary.csv          # Summary statistics
│   └── fig*.png                      # Visualization outputs
├── notebooks/
│   ├── trader_sentiment_analysis.ipynb  # Jupyter notebook
│   └── trader_sentiment_analysis.py     # Python script
├── README.md
└── requirements.txt
```

### 🛠️ Setup & Installation

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Run the analysis
cd notebooks
python trader_sentiment_analysis.py
# OR open the Jupyter notebook
jupyter notebook trader_sentiment_analysis.ipynb
```

### 📊 Visualizations

The analysis generates 7 key visualizations:

1. **Sentiment Overview** - Distribution & timeline of Fear/Greed Index
2. **PnL Analysis** - Performance metrics across sentiment categories
3. **Side Analysis** - Long vs Short positions by sentiment
4. **Trader Distribution** - PnL, win rate, and profit factor distributions
5. **Smart Money Comparison** - Top traders vs regular traders behavior
6. **Time Series** - Daily PnL correlated with sentiment
7. **Coin Analysis** - Volume and PnL by trading pair

### 📝 Technical Details

**Data Processing:**
- Merged 211K+ trades with daily sentiment data
- Handled missing values and data type conversions
- Created derived features (sentiment scores, win rate, profit factor)

**Statistical Tests:**
- T-test: Fear vs Greed PnL comparison
- Chi-square: Trading direction vs sentiment
- Pearson correlation: Sentiment value vs PnL

### 👤 Author

**Chandan** - Junior Data Scientist Candidate

### 📧 Contact

Application submitted to:
- saami@primetrade.ai
- nagasai@primetrade.ai
- chetan@primetrade.ai
- CC: sonika@primetrade.ai

---

*This analysis was completed as part of the PrimeTrade.AI Data Science hiring process.*
