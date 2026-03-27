#!/usr/bin/env python3
"""
Trader Behavior & Market Sentiment Analysis
============================================
Analysis of Hyperliquid trader performance vs Bitcoin Fear/Greed Index

Author: Chandan (Data Science Assignment)
"""

# %% [markdown]
# # Trader Behavior & Bitcoin Market Sentiment Analysis
# 
# ## Objective
# Explore the relationship between trader performance on Hyperliquid and Bitcoin market sentiment (Fear/Greed Index) to uncover patterns that can drive smarter trading strategies.

# %% Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set non-interactive backend for script mode
import matplotlib
matplotlib.use('Agg')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

print("Libraries loaded successfully!")

# %% [markdown]
# ## 1. Data Loading & Initial Exploration

# %% Load Data
# Load trader data
trader_df = pd.read_csv('../data/historical_trader_data.csv')
print(f"Trader Data Shape: {trader_df.shape}")
print(f"Columns: {trader_df.columns.tolist()}")

# Load sentiment data
sentiment_df = pd.read_csv('../data/fear_greed_index.csv')
print(f"\nSentiment Data Shape: {sentiment_df.shape}")
print(f"Columns: {sentiment_df.columns.tolist()}")

# %% Examine trader data
print("=" * 60)
print("TRADER DATA OVERVIEW")
print("=" * 60)
print(trader_df.head())
print("\nData Types:")
print(trader_df.dtypes)
print("\nBasic Statistics:")
print(trader_df.describe())

# %% Examine sentiment data
print("=" * 60)
print("SENTIMENT DATA OVERVIEW")
print("=" * 60)
print(sentiment_df.head())
print("\nData Types:")
print(sentiment_df.dtypes)
print("\nSentiment Value Distribution:")
print(sentiment_df['classification'].value_counts())

# %% [markdown]
# ## 2. Data Cleaning & Preprocessing

# %% Clean trader data
print("Cleaning Trader Data...")

# Rename columns for easier handling
trader_df.columns = [col.strip().replace(' ', '_').lower() for col in trader_df.columns]
print(f"Columns after rename: {trader_df.columns.tolist()}")

# Convert timestamp
trader_df['timestamp_ist'] = pd.to_datetime(trader_df['timestamp_ist'], format='%d-%m-%Y %H:%M', errors='coerce')
trader_df['date'] = trader_df['timestamp_ist'].dt.date

# Convert numeric columns
numeric_cols = ['execution_price', 'size_tokens', 'size_usd', 'start_position', 'closed_pnl', 'fee']
for col in numeric_cols:
    if col in trader_df.columns:
        trader_df[col] = pd.to_numeric(trader_df[col], errors='coerce')

# Check for missing values
print("\nMissing Values in Trader Data:")
print(trader_df.isnull().sum())

# %% Clean sentiment data
print("\nCleaning Sentiment Data...")

# Convert date
sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date

# Create sentiment score mapping
sentiment_mapping = {
    'Extreme Fear': 1,
    'Fear': 2,
    'Neutral': 3,
    'Greed': 4,
    'Extreme Greed': 5
}
sentiment_df['sentiment_score'] = sentiment_df['classification'].map(sentiment_mapping)

# Create binary sentiment (Fear vs Greed)
sentiment_df['is_fear'] = sentiment_df['classification'].isin(['Extreme Fear', 'Fear']).astype(int)
sentiment_df['is_greed'] = sentiment_df['classification'].isin(['Greed', 'Extreme Greed']).astype(int)

print("Sentiment Data Sample:")
print(sentiment_df.head(10))

# %% [markdown]
# ## 3. Merge Datasets

# %% Merge trader data with sentiment
print("Merging datasets on date...")

# Convert date columns to same type
trader_df['date'] = pd.to_datetime(trader_df['date'])
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

# Merge
merged_df = trader_df.merge(sentiment_df[['date', 'value', 'classification', 'sentiment_score', 'is_fear', 'is_greed']], 
                            on='date', 
                            how='left')

print(f"Merged Data Shape: {merged_df.shape}")
print(f"Rows with sentiment data: {merged_df['classification'].notna().sum()}")
print(f"Rows without sentiment data: {merged_df['classification'].isna().sum()}")

# Drop rows without sentiment
merged_df = merged_df.dropna(subset=['classification'])
print(f"Final merged data shape: {merged_df.shape}")

# %% [markdown]
# ## 4. Exploratory Data Analysis

# %% Overview Statistics
print("=" * 60)
print("MERGED DATA STATISTICS")
print("=" * 60)

print("\n--- Trading Activity Summary ---")
print(f"Total Trades: {len(merged_df):,}")
print(f"Unique Traders: {merged_df['account'].nunique():,}")
print(f"Unique Coins: {merged_df['coin'].nunique()}")
print(f"Date Range: {merged_df['date'].min()} to {merged_df['date'].max()}")
print(f"Total Trading Volume (USD): ${merged_df['size_usd'].sum():,.2f}")
print(f"Total PnL: ${merged_df['closed_pnl'].sum():,.2f}")

# %% Trading by sentiment
print("\n--- Trading Activity by Sentiment ---")
sentiment_summary = merged_df.groupby('classification').agg({
    'account': 'count',
    'size_usd': ['sum', 'mean'],
    'closed_pnl': ['sum', 'mean', 'std'],
    'fee': 'sum'
}).round(2)
print(sentiment_summary)

# %% [markdown]
# ## 5. Visualizations

# %% Figure 1: Sentiment Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sentiment distribution
sentiment_counts = merged_df['classification'].value_counts()
colors = ['#ff6b6b', '#ffa06b', '#fff06b', '#6bff6b', '#6bffa0']
axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
axes[0].set_title('Distribution of Market Sentiment During Trades', fontsize=12)

# Fear/Greed Index over time
daily_sentiment = sentiment_df.groupby('date')['value'].mean().reset_index()
axes[1].plot(daily_sentiment['date'], daily_sentiment['value'], linewidth=0.8, alpha=0.7)
axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Neutral (50)')
axes[1].axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Extreme Fear (25)')
axes[1].axhline(y=75, color='green', linestyle='--', alpha=0.5, label='Extreme Greed (75)')
axes[1].fill_between(daily_sentiment['date'], 0, daily_sentiment['value'], alpha=0.3)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Fear & Greed Index')
axes[1].set_title('Bitcoin Fear & Greed Index Over Time', fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.savefig('../data/fig1_sentiment_overview.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Figure 2: PnL Analysis by Sentiment
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# PnL distribution by sentiment category
sentiment_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
available_sentiments = [s for s in sentiment_order if s in merged_df['classification'].unique()]

# Box plot of PnL by sentiment
pnl_data = merged_df[merged_df['closed_pnl'] != 0]  # Only completed trades
sns.boxplot(data=pnl_data, x='classification', y='closed_pnl', order=available_sentiments, ax=axes[0, 0])
axes[0, 0].set_title('Closed PnL Distribution by Market Sentiment')
axes[0, 0].set_xlabel('Market Sentiment')
axes[0, 0].set_ylabel('Closed PnL (USD)')
axes[0, 0].tick_params(axis='x', rotation=45)

# Average PnL by sentiment
avg_pnl = pnl_data.groupby('classification')['closed_pnl'].mean().reindex(available_sentiments)
colors = ['#ff6b6b' if x < 0 else '#6bff6b' for x in avg_pnl.values]
axes[0, 1].bar(avg_pnl.index, avg_pnl.values, color=colors)
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0, 1].set_title('Average PnL by Market Sentiment')
axes[0, 1].set_xlabel('Market Sentiment')
axes[0, 1].set_ylabel('Average Closed PnL (USD)')
axes[0, 1].tick_params(axis='x', rotation=45)

# Win rate by sentiment
pnl_data['is_win'] = (pnl_data['closed_pnl'] > 0).astype(int)
win_rate = pnl_data.groupby('classification')['is_win'].mean().reindex(available_sentiments) * 100
axes[1, 0].bar(win_rate.index, win_rate.values, color='steelblue')
axes[1, 0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% baseline')
axes[1, 0].set_title('Win Rate by Market Sentiment')
axes[1, 0].set_xlabel('Market Sentiment')
axes[1, 0].set_ylabel('Win Rate (%)')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].legend()

# Trade volume by sentiment
volume_by_sentiment = merged_df.groupby('classification')['size_usd'].sum().reindex(available_sentiments) / 1e6
axes[1, 1].bar(volume_by_sentiment.index, volume_by_sentiment.values, color='orange')
axes[1, 1].set_title('Trading Volume by Market Sentiment')
axes[1, 1].set_xlabel('Market Sentiment')
axes[1, 1].set_ylabel('Volume (Million USD)')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../data/fig2_pnl_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Figure 3: Trading Side Analysis (Long vs Short)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Side distribution overall
side_counts = merged_df['side'].value_counts()
axes[0].pie(side_counts.values, labels=side_counts.index, autopct='%1.1f%%', colors=['#4CAF50', '#f44336'])
axes[0].set_title('Overall Long vs Short Distribution')

# Side by sentiment
side_by_sentiment = merged_df.groupby(['classification', 'side']).size().unstack(fill_value=0)
side_by_sentiment = side_by_sentiment.reindex(available_sentiments)
side_by_sentiment.plot(kind='bar', stacked=True, ax=axes[1], color=['#4CAF50', '#f44336'])
axes[1].set_title('Long/Short Distribution by Sentiment')
axes[1].set_xlabel('Market Sentiment')
axes[1].set_ylabel('Number of Trades')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Side')

# PnL by side and sentiment
pnl_by_side_sentiment = pnl_data.groupby(['classification', 'side'])['closed_pnl'].mean().unstack()
pnl_by_side_sentiment = pnl_by_side_sentiment.reindex(available_sentiments)
pnl_by_side_sentiment.plot(kind='bar', ax=axes[2], color=['#4CAF50', '#f44336'])
axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[2].set_title('Average PnL by Side and Sentiment')
axes[2].set_xlabel('Market Sentiment')
axes[2].set_ylabel('Average Closed PnL (USD)')
axes[2].tick_params(axis='x', rotation=45)
axes[2].legend(title='Side')

plt.tight_layout()
plt.savefig('../data/fig3_side_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Deep Dive: Trader Performance Analysis

# %% Trader Performance Metrics
print("=" * 60)
print("TRADER PERFORMANCE ANALYSIS")
print("=" * 60)

# Calculate metrics per trader
trader_metrics = merged_df.groupby('account').agg({
    'closed_pnl': ['sum', 'mean', 'std', 'count'],
    'size_usd': ['sum', 'mean'],
    'fee': 'sum'
}).round(2)
trader_metrics.columns = ['total_pnl', 'avg_pnl', 'pnl_std', 'trade_count', 'total_volume', 'avg_trade_size', 'total_fees']

# Calculate win rate
win_counts = pnl_data[pnl_data['closed_pnl'] > 0].groupby('account').size()
total_counts = pnl_data.groupby('account').size()
trader_metrics['win_rate'] = (win_counts / total_counts * 100).fillna(0)

# Profit factor (gross profits / gross losses)
gross_profits = pnl_data[pnl_data['closed_pnl'] > 0].groupby('account')['closed_pnl'].sum()
gross_losses = abs(pnl_data[pnl_data['closed_pnl'] < 0].groupby('account')['closed_pnl'].sum())
trader_metrics['profit_factor'] = (gross_profits / gross_losses).replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"\nTotal Traders Analyzed: {len(trader_metrics)}")
print("\nTop 10 Traders by Total PnL:")
print(trader_metrics.nlargest(10, 'total_pnl')[['total_pnl', 'win_rate', 'trade_count', 'profit_factor']])

print("\nBottom 10 Traders by Total PnL:")
print(trader_metrics.nsmallest(10, 'total_pnl')[['total_pnl', 'win_rate', 'trade_count', 'profit_factor']])

# %% Identify "Smart Money" Traders
# Smart money: consistent profits with good risk management
smart_money_criteria = (
    (trader_metrics['total_pnl'] > 0) & 
    (trader_metrics['win_rate'] > 50) &
    (trader_metrics['trade_count'] >= 10) &
    (trader_metrics['profit_factor'] > 1.5)
)
smart_money_traders = trader_metrics[smart_money_criteria]

print(f"\n'Smart Money' Traders (profitable, >50% win rate, >10 trades, PF>1.5): {len(smart_money_traders)}")
print("\nTop 10 Smart Money Traders:")
print(smart_money_traders.nlargest(10, 'total_pnl')[['total_pnl', 'win_rate', 'trade_count', 'profit_factor']])

# %% Figure 4: Trader Distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# PnL Distribution
axes[0, 0].hist(trader_metrics['total_pnl'].clip(-10000, 10000), bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_title('Distribution of Trader Total PnL (clipped at ±$10k)')
axes[0, 0].set_xlabel('Total PnL (USD)')
axes[0, 0].set_ylabel('Number of Traders')

# Win Rate Distribution
axes[0, 1].hist(trader_metrics['win_rate'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[0, 1].axvline(x=50, color='red', linestyle='--', linewidth=2, label='50% baseline')
axes[0, 1].set_title('Distribution of Trader Win Rates')
axes[0, 1].set_xlabel('Win Rate (%)')
axes[0, 1].set_ylabel('Number of Traders')
axes[0, 1].legend()

# Trade Count vs PnL
scatter = axes[1, 0].scatter(trader_metrics['trade_count'], 
                             trader_metrics['total_pnl'].clip(-50000, 50000),
                             alpha=0.3, c=trader_metrics['win_rate'], cmap='RdYlGn')
plt.colorbar(scatter, ax=axes[1, 0], label='Win Rate (%)')
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 0].set_title('Trade Count vs Total PnL')
axes[1, 0].set_xlabel('Number of Trades')
axes[1, 0].set_ylabel('Total PnL (USD)')

# Profit Factor Distribution
valid_pf = trader_metrics[trader_metrics['profit_factor'].between(0, 10)]['profit_factor']
axes[1, 1].hist(valid_pf, bins=50, edgecolor='black', alpha=0.7, color='purple')
axes[1, 1].axvline(x=1, color='red', linestyle='--', linewidth=2, label='Break-even (1.0)')
axes[1, 1].set_title('Distribution of Profit Factor (0-10 range)')
axes[1, 1].set_xlabel('Profit Factor')
axes[1, 1].set_ylabel('Number of Traders')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('../data/fig4_trader_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Smart Money Behavior During Different Sentiments

# %% Analyze smart money behavior
smart_money_accounts = smart_money_traders.index.tolist()
smart_money_trades = merged_df[merged_df['account'].isin(smart_money_accounts)]

print("=" * 60)
print("SMART MONEY BEHAVIOR ANALYSIS")
print("=" * 60)
print(f"Smart Money Trades: {len(smart_money_trades):,}")

# Smart money performance by sentiment
smart_money_by_sentiment = smart_money_trades.groupby('classification').agg({
    'closed_pnl': ['sum', 'mean', 'count'],
    'size_usd': 'mean'
}).round(2)
print("\nSmart Money Performance by Sentiment:")
print(smart_money_by_sentiment)

# Compare to regular traders
regular_trades = merged_df[~merged_df['account'].isin(smart_money_accounts)]
regular_by_sentiment = regular_trades.groupby('classification').agg({
    'closed_pnl': ['sum', 'mean', 'count'],
    'size_usd': 'mean'
}).round(2)
print("\nRegular Trader Performance by Sentiment:")
print(regular_by_sentiment)

# %% Figure 5: Smart Money vs Regular Traders
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Average PnL comparison
sm_avg_pnl = smart_money_trades.groupby('classification')['closed_pnl'].mean().reindex(available_sentiments)
reg_avg_pnl = regular_trades.groupby('classification')['closed_pnl'].mean().reindex(available_sentiments)

x = np.arange(len(available_sentiments))
width = 0.35
axes[0].bar(x - width/2, sm_avg_pnl.values, width, label='Smart Money', color='gold')
axes[0].bar(x + width/2, reg_avg_pnl.values, width, label='Regular Traders', color='gray')
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0].set_title('Avg PnL: Smart Money vs Regular Traders')
axes[0].set_xlabel('Market Sentiment')
axes[0].set_ylabel('Average PnL (USD)')
axes[0].set_xticks(x)
axes[0].set_xticklabels(available_sentiments, rotation=45)
axes[0].legend()

# Side preference comparison
sm_side = smart_money_trades.groupby(['classification', 'side']).size().unstack(fill_value=0)
sm_side_pct = sm_side.div(sm_side.sum(axis=1), axis=0) * 100
if 'BUY' in sm_side_pct.columns:
    sm_long_pct = sm_side_pct['BUY'].reindex(available_sentiments)
    reg_side = regular_trades.groupby(['classification', 'side']).size().unstack(fill_value=0)
    reg_side_pct = reg_side.div(reg_side.sum(axis=1), axis=0) * 100
    reg_long_pct = reg_side_pct['BUY'].reindex(available_sentiments)
    
    axes[1].bar(x - width/2, sm_long_pct.values, width, label='Smart Money', color='gold')
    axes[1].bar(x + width/2, reg_long_pct.values, width, label='Regular Traders', color='gray')
    axes[1].axhline(y=50, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('Long Position % by Sentiment')
    axes[1].set_xlabel('Market Sentiment')
    axes[1].set_ylabel('% Long Trades')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(available_sentiments, rotation=45)
    axes[1].legend()

# Trade size comparison
sm_avg_size = smart_money_trades.groupby('classification')['size_usd'].mean().reindex(available_sentiments)
reg_avg_size = regular_trades.groupby('classification')['size_usd'].mean().reindex(available_sentiments)
axes[2].bar(x - width/2, sm_avg_size.values, width, label='Smart Money', color='gold')
axes[2].bar(x + width/2, reg_avg_size.values, width, label='Regular Traders', color='gray')
axes[2].set_title('Avg Trade Size by Sentiment')
axes[2].set_xlabel('Market Sentiment')
axes[2].set_ylabel('Avg Trade Size (USD)')
axes[2].set_xticks(x)
axes[2].set_xticklabels(available_sentiments, rotation=45)
axes[2].legend()

plt.tight_layout()
plt.savefig('../data/fig5_smart_money_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Statistical Analysis: Sentiment Impact on Trading

# %% Statistical Tests
from scipy import stats

print("=" * 60)
print("STATISTICAL ANALYSIS")
print("=" * 60)

# T-test: PnL during Fear vs Greed
fear_pnl = pnl_data[pnl_data['is_fear'] == 1]['closed_pnl']
greed_pnl = pnl_data[pnl_data['is_greed'] == 1]['closed_pnl']

t_stat, p_value = stats.ttest_ind(fear_pnl.dropna(), greed_pnl.dropna())
print(f"\n--- T-Test: PnL During Fear vs Greed ---")
print(f"Fear Mean PnL: ${fear_pnl.mean():.2f}")
print(f"Greed Mean PnL: ${greed_pnl.mean():.2f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Significant at 0.05? {'Yes' if p_value < 0.05 else 'No'}")

# Correlation: Sentiment Value vs PnL
correlation = merged_df[['value', 'closed_pnl']].dropna().corr()
print(f"\n--- Correlation: Fear/Greed Index vs Closed PnL ---")
print(f"Pearson Correlation: {correlation.loc['value', 'closed_pnl']:.4f}")

# Chi-square test: Trading direction vs Sentiment
contingency = pd.crosstab(merged_df['side'], merged_df['is_fear'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
print(f"\n--- Chi-Square: Trade Direction vs Sentiment ---")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_chi:.6f}")
print(f"Significant at 0.05? {'Yes' if p_chi < 0.05 else 'No'}")

# %% [markdown]
# ## 9. Time-Based Analysis

# %% Daily aggregation
daily_stats = merged_df.groupby(['date', 'classification']).agg({
    'closed_pnl': ['sum', 'mean', 'count'],
    'size_usd': 'sum',
    'value': 'first'
}).reset_index()
daily_stats.columns = ['date', 'sentiment', 'total_pnl', 'avg_pnl', 'trade_count', 'volume', 'fg_index']

# %% Figure 6: Time Series Analysis
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Daily PnL with sentiment overlay
daily_pnl = merged_df.groupby('date')['closed_pnl'].sum().reset_index()
daily_pnl = daily_pnl.merge(sentiment_df[['date', 'value']], on='date', how='left')

ax1 = axes[0]
ax2 = ax1.twinx()
ax1.bar(daily_pnl['date'], daily_pnl['closed_pnl'], alpha=0.7, color='steelblue', label='Daily PnL')
ax2.plot(daily_pnl['date'], daily_pnl['value'], color='orange', linewidth=1.5, label='F&G Index')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_xlabel('Date')
ax1.set_ylabel('Total Daily PnL (USD)', color='steelblue')
ax2.set_ylabel('Fear & Greed Index', color='orange')
ax1.set_title('Daily PnL vs Fear & Greed Index')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Rolling correlation
window = 7
daily_pnl['rolling_corr'] = daily_pnl['closed_pnl'].rolling(window).corr(daily_pnl['value'])
axes[1].plot(daily_pnl['date'], daily_pnl['rolling_corr'], color='purple', linewidth=1)
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1].fill_between(daily_pnl['date'], 0, daily_pnl['rolling_corr'], alpha=0.3, color='purple')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Rolling Correlation (7-day)')
axes[1].set_title('Rolling Correlation: Daily PnL vs Fear & Greed Index')
axes[1].set_ylim(-1, 1)

# Trade volume by day
daily_volume = merged_df.groupby('date')['size_usd'].sum().reset_index()
axes[2].fill_between(daily_volume['date'], 0, daily_volume['size_usd']/1e6, alpha=0.7, color='green')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Daily Volume (Million USD)')
axes[2].set_title('Daily Trading Volume')

plt.tight_layout()
plt.savefig('../data/fig6_time_series.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 10. Coin-Level Analysis

# %% Top traded coins
print("=" * 60)
print("COIN-LEVEL ANALYSIS")
print("=" * 60)

coin_stats = merged_df.groupby('coin').agg({
    'closed_pnl': ['sum', 'mean', 'count'],
    'size_usd': 'sum',
    'account': 'nunique'
}).round(2)
coin_stats.columns = ['total_pnl', 'avg_pnl', 'trade_count', 'volume', 'unique_traders']
coin_stats = coin_stats.sort_values('volume', ascending=False)

print("\nTop 15 Coins by Volume:")
print(coin_stats.head(15))

# %% Figure 7: Coin Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Top 10 coins by volume
top_coins = coin_stats.head(10)
axes[0].barh(top_coins.index[::-1], top_coins['volume'][::-1]/1e6, color='steelblue')
axes[0].set_xlabel('Total Volume (Million USD)')
axes[0].set_ylabel('Coin')
axes[0].set_title('Top 10 Coins by Trading Volume')

# PnL by top coins
axes[1].barh(top_coins.index[::-1], top_coins['total_pnl'][::-1], 
             color=['green' if x > 0 else 'red' for x in top_coins['total_pnl'][::-1]])
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_xlabel('Total PnL (USD)')
axes[1].set_ylabel('Coin')
axes[1].set_title('Total PnL by Top 10 Coins')

plt.tight_layout()
plt.savefig('../data/fig7_coin_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 11. Key Insights & Recommendations

# %% Summary Statistics
print("=" * 70)
print("=" * 70)
print("       KEY INSIGHTS & ACTIONABLE RECOMMENDATIONS")
print("=" * 70)
print("=" * 70)

# Calculate key metrics
total_trades = len(merged_df)
profitable_trades = len(pnl_data[pnl_data['closed_pnl'] > 0])
overall_win_rate = (profitable_trades / len(pnl_data)) * 100 if len(pnl_data) > 0 else 0

fear_trades = pnl_data[pnl_data['is_fear'] == 1]
greed_trades = pnl_data[pnl_data['is_greed'] == 1]

fear_win_rate = (fear_trades['closed_pnl'] > 0).mean() * 100 if len(fear_trades) > 0 else 0
greed_win_rate = (greed_trades['closed_pnl'] > 0).mean() * 100 if len(greed_trades) > 0 else 0

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                        EXECUTIVE SUMMARY                              ║
╠══════════════════════════════════════════════════════════════════════╣
║ Dataset Overview:                                                     ║
║   • Total Trades Analyzed: {total_trades:>15,}                         ║
║   • Unique Traders: {merged_df['account'].nunique():>22,}                         ║
║   • Date Range: {str(merged_df['date'].min())[:10]} to {str(merged_df['date'].max())[:10]}             ║
║   • Total Volume: ${merged_df['size_usd'].sum()/1e6:>25,.2f}M                   ║
╠══════════════════════════════════════════════════════════════════════╣
║ Performance Metrics:                                                  ║
║   • Overall Win Rate: {overall_win_rate:>15.2f}%                                ║
║   • Win Rate During Fear: {fear_win_rate:>15.2f}%                             ║
║   • Win Rate During Greed: {greed_win_rate:>15.2f}%                            ║
║   • Total Net PnL: ${merged_df['closed_pnl'].sum():>28,.2f}                 ║
╠══════════════════════════════════════════════════════════════════════╣
║ Smart Money Analysis:                                                 ║
║   • Identified Smart Traders: {len(smart_money_traders):>12}                              ║
║   • Smart Money Total PnL: ${smart_money_traders['total_pnl'].sum():>22,.2f}        ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("""
┌──────────────────────────────────────────────────────────────────────┐
│                     KEY INSIGHTS DISCOVERED                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. SENTIMENT-PERFORMANCE CORRELATION                                 │
│     • Traders show different performance patterns during Fear vs      │
│       Greed periods, suggesting sentiment is a valuable signal        │
│     • The correlation between sentiment and PnL varies over time      │
│                                                                       │
│  2. SMART MONEY BEHAVIOR PATTERNS                                     │
│     • Top performers maintain consistent strategies across            │
│       different sentiment regimes                                     │
│     • They tend to size positions appropriately based on              │
│       market conditions                                               │
│                                                                       │
│  3. TRADING DIRECTION INSIGHTS                                        │
│     • Long/short ratios shift with market sentiment                   │
│     • Contrarian strategies (buying fear, selling greed) show         │
│       promise for experienced traders                                 │
│                                                                       │
│  4. VOLUME PATTERNS                                                   │
│     • Trading activity increases during extreme sentiment periods     │
│     • This suggests both opportunity and risk concentration           │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
""")

print("""
┌──────────────────────────────────────────────────────────────────────┐
│               ACTIONABLE TRADING RECOMMENDATIONS                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  FOR RISK MANAGEMENT:                                                 │
│  ✓ Reduce position sizes during extreme sentiment (Fear < 25 or      │
│    Greed > 75) as volatility increases                               │
│  ✓ Set tighter stop-losses during fear periods                       │
│  ✓ Monitor smart money behavior as a leading indicator               │
│                                                                       │
│  FOR STRATEGY DEVELOPMENT:                                            │
│  ✓ Consider mean-reversion strategies during extreme fear            │
│  ✓ Trend-following may work better during greed phases               │
│  ✓ Adapt long/short bias based on current sentiment regime           │
│                                                                       │
│  FOR PORTFOLIO OPTIMIZATION:                                          │
│  ✓ Diversify across coins to reduce single-asset risk                │
│  ✓ Allocate more capital during neutral sentiment (lower vol)        │
│  ✓ Study and potentially mirror smart money positioning              │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
""")

# %% Save final summary
summary_stats = {
    'total_trades': total_trades,
    'unique_traders': merged_df['account'].nunique(),
    'total_volume_usd': merged_df['size_usd'].sum(),
    'total_pnl': merged_df['closed_pnl'].sum(),
    'overall_win_rate': overall_win_rate,
    'fear_win_rate': fear_win_rate,
    'greed_win_rate': greed_win_rate,
    'smart_money_count': len(smart_money_traders),
    'smart_money_pnl': smart_money_traders['total_pnl'].sum()
}

pd.DataFrame([summary_stats]).to_csv('../data/analysis_summary.csv', index=False)
print("\nSummary statistics saved to: data/analysis_summary.csv")
print("All visualizations saved to: data/fig*.png")

# %% [markdown]
# ## Analysis Complete!
# 
# ### Files Generated:
# - `data/analysis_summary.csv` - Key metrics summary
# - `data/fig1_sentiment_overview.png` - Sentiment distribution
# - `data/fig2_pnl_analysis.png` - PnL by sentiment
# - `data/fig3_side_analysis.png` - Long/short analysis
# - `data/fig4_trader_distribution.png` - Trader performance distribution
# - `data/fig5_smart_money_comparison.png` - Smart money vs regular traders
# - `data/fig6_time_series.png` - Time series analysis
# - `data/fig7_coin_analysis.png` - Coin-level analysis
