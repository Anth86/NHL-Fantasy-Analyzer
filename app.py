import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================================
# 1. LEAGUE CONFIGURATION
# ==========================================
LEAGUE_SIZE = 12
ROSTER_LAYOUT = {"C": 2, "LW": 2, "RW": 2, "D": 4, "G": 2, "UTIL": 1, "BENCH": 3}
SKATER_CATEGORIES = ['g', 'a', 'plus_minus', 'ppp', 'sog', 'hit', 'blk']
GOALIE_CATEGORIES = ['w', 'ga', 'sv', 'sho']

st.set_page_config(page_title="NHL Fantasy Architect", page_icon="🏒", layout="wide")

# ==========================================
# 2. DATA CLEANING & UNIFICATION
# ==========================================
def standardize_data(df):
    """Standardizes column headers and player names."""
    df.columns = [c.lower().strip().replace(" ", "_").replace("\xa0", "_") for c in df.columns]
    if 'player' in df.columns:
        df['player'] = df['player'].astype(str).str.strip().str.title()
    return df

@st.cache_data
def load_and_merge_data():
    try:
        nst = standardize_data(pd.read_csv("nst_data.csv"))
        pdo = standardize_data(pd.read_csv("pdo_data.csv"))
        ozs = standardize_data(pd.read_csv("ozs.data.csv"))
        adp = standardize_data(pd.read_csv("adp_data.csv"))
        
        # Subsetting to ensure unique merge keys and prevent suffix errors
        pdo_subset = pdo[['player', 'pdo']]
        ozs_subset = ozs[['player', 'off._zone_start_%']]
        adp_subset = adp[['player', 'adp']]
        
        # Sequential Merge
        merged = pd.merge(nst, pdo_subset, on="player", how="left")
        merged = pd.merge(merged, ozs_subset, on="player", how="left")
        merged = pd.merge(merged, adp_subset, on="player", how="left")
        
        # Clean specific columns
        merged = merged.rename(columns={'off._zone_start_%': 'ozs_pct'})
        merged['ozs_pct'] = pd.to_numeric(merged['ozs_pct'], errors='coerce').fillna(50.0) / 100.0
        merged['pdo'] = pd.to_numeric(merged['pdo'], errors='coerce').fillna(1.000)
        
        return merged
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return pd.DataFrame()

# ==========================================
# 3. MATH ENGINE
# ==========================================
@st.cache_data
def compute_value_gap_matrix(df, weights, mode):
    rename_dict = {'goals': 'g', 'total_assists': 'a', 'shots': 'sog', 'hits': 'hit', 'blocked': 'blk'}
    df = df.rename(columns=rename_dict)
    
    # Calculate Z-Scores
    df['total_z'] = 0.0
    for pos in df['position'].unique():
        mask = df['position'] == pos
        for cat in SKATER_CATEGORIES:
            if cat in df.columns:
                mean, std = df.loc[mask, cat].mean(), df.loc[mask, cat].std()
                if std > 0:
                    df.loc[mask, 'total_z'] += ((df.loc[mask, cat] - mean) / std) * weights.get(cat, 1.0)
    
    # REFINED MATH: Non-linear ADP penalty
    df['adp_penalty'] = (df['adp'].fillna(100) / 100.0) ** 1.5
    df['value_gap'] = df['total_z'] - df['adp_penalty']
    
    # Boost for elite talent (Top 5 ADP)
    df.loc[df['adp'] <= 5, 'value_gap'] += 3.0
    
    df['action_tier'] = pd.cut(df['value_gap'], bins=[-float('inf'), -1.5, 0.1, 3.0, float('inf')], 
                               labels=["🛑 Do Not Draft", "⚠️ Avoid / Overpay", "✅ Draft as Needed", "🎯 Target Immediately"])
    df['pdo_alert'] = np.where(df['pdo'] < 0.975, "📈 Buy Low", np.where(df['pdo'] > 1.025, "📉 Sell High", "🔄 Stable"))
    
    # Mode Toggle: Sort by Value Gap or Total Production
    sort_col = 'value_gap' if mode == "Value Hunting" else 'total_z'
    return df.sort_values(sort_col, ascending=False)

# ==========================================
# 4. MAIN UI
# ==========================================
def main():
    st.title("🏒 NHL Fantasy Draft Architect")
    df = load_and_merge_data()
    
    if df.empty:
        st.warning("Ensure your CSV files (nst_data, pdo_data, ozs.data, adp_data) are in the project folder.")
        return
    
    # Add Mode Toggle to Sidebar
    mode = st.sidebar.radio("Analysis Mode:", ["Value Hunting", "Talent Ranking"])
    
    weights = {cat: 1.0 for cat in SKATER_CATEGORIES}
    processed = compute_value_gap_matrix(df, weights, mode)
    
    tab1, tab3 = st.tabs(["Optimization Matrix", "Regression Alerts"])
    
    with tab1:
        st.subheader(f"Current View: {mode}")
        st.data_editor(processed, column_config={
            "ozs_pct": st.column_config.ProgressColumn("OZS% Intent", format="%.2f", min_value=0.0, max_value=1.0),
            "value_gap": st.column_config.NumberColumn("Value Gap", format="%.2f")
        }, use_container_width=True, disabled=True)
        
    with tab3:
        st.dataframe(processed[['player', 'position', 'pdo', 'pdo_alert', 'ozs_pct']], use_container_width=True)

if __name__ == "__main__":
    main()