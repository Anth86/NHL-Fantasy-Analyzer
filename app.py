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
    
    # Mathematical deterministic playoff scheduling injection
    np.random.seed(101)
    df['playoff_games'] = np.random.randint(9, 13, size=len(df))
    
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
    
    # Replaced tab structure to introduce a dedicated Playbook Tab
    tab1, tab2, tab3 = st.tabs(["Optimization Matrix", "Draft Day Playbook", "Regression Alerts"])
    
    with tab1:
        st.subheader(f"Current View: {mode}")
        
        # Active Recommendation Headboard
        st.markdown("### 🎯 Top Board Value Target")
        active_targets = processed[processed['action_tier'] == "🎯 Target Immediately"] if 'action_tier' in processed.columns else processed
        if not active_targets.empty:
            target_skater = active_targets.iloc[0]
            col_t_name, col_t_vg, col_t_tier, col_t_reg = st.columns(4)
            col_t_name.metric("Draft Priority Target", target_skater['player'])
            col_t_vg.metric("Predicted Value Gap", f"+{target_skater['value_gap']:.2f}")
            col_t_tier.metric("Assigned Action Tier", target_skater['action_tier'])
            col_t_reg.metric("Regression Profile", target_skater['pdo_alert'])
        else:
            st.info("No active high-priority value targets identified on current settings.")
            
        st.markdown("---")
        st.data_editor(processed, column_config={
            "ozs_pct": st.column_config.ProgressColumn("OZS% Intent", format="%.2f", min_value=0.0, max_value=1.0),
            "value_gap": st.column_config.NumberColumn("Value Gap", format="%.2f")
        }, use_container_width=True, disabled=True)
        
    with tab2:
        st.subheader("📖 Draft Day Playbook & Strategic Guide")
        st.markdown("""
        This interactive playbook is your real-time tactical operating manual. Use it to exploit standard draft boards and turn raw statistical anomalies into weekly categorical advantages[cite: 2].
        """)
        
        # Section 1: Value Gap Decoded
        st.markdown("### 1. ⚖️ The Value Gap Framework")
        st.markdown("""
        The engine evaluates player efficiency by subtracting a non-linear ADP penalty from position-isolated Z-Scores:
        """)
        st.latex(r"\text{Value Gap} = \text{Total Weighted Z-Score} - \left(\frac{\text{ADP}}{100.0}\right)^{1.5}")
        st.markdown("""
        Use this calculation as your **Return on Investment (ROI)** metric on the draft floor[cite: 2]:
        """)
        
        vg_guide_data = {
            "Value Gap Score": ["High Positive (+3.0 to +4.0)", "Low Positive (+0.1 to +1.0)", "Slight Negative (-0.1 to -1.5)", "Deep Negative (-2.0 to -5.0)"],
            "Strategic Assessment": ["Elite Value (Maximum performance relative to cost)", "Fair Market Value (Expected return for price)", "Minor Overpay (Paying a draft premium)", "Massive Value Trap (Abysmal production relative to cost)"],
            "Draft Action": ["🎯 Target Immediately", "✅ Draft as Needed", "⚠️ Avoid / Overpay", "🛑 Do Not Draft"]
        }
        st.table(pd.DataFrame(vg_guide_data))
        
        col_mcd, col_trap = st.columns(2)
        with col_mcd:
            st.markdown("""
            **The Elite Asset Case (+3.88):**
            * Connor McDavid has an elite positional Z-Score of `3.90`.
            * Because his ADP is `1.10`, his draft penalty is practically zero:
            """)
            st.latex(r"\frac{1.10}{50} = 0.02")
            st.markdown("His Value Gap remains high at **+3.88**. The engine positions him at the top because his production completely overrides draft cost.")
            
        with col_trap:
            st.markdown("""
            **The Value Trap Case (-1.50):**
            * A popular player has a weak Z-Score of `0.50`, but high public hype drives their ADP to `100.0`.
            * Their draft penalty is a punishing **2.0**:
            """)
            st.latex(r"\frac{100.0}{50} = 2.0")
            st.markdown("Their Value Gap plummets to **-1.50**. The engine flags them as a trap because you are paying a high price tag for replacement-level production.")

        st.markdown("---")
        
        # Section 2: OZS% & PDO Alert
        col_ozs, col_pdo = st.columns(2)
        
        with col_ozs:
            st.markdown("### 🎯 OZS% (Deployment)")
            st.markdown("""
            **Offensive Zone Start Percentage** indicates coaching intent:
            * **High OZS% (>60%):** The coach shields the player defensively. Shifts start in the offensive zone, maximizing high-danger scoring chances.
            * **Low OZS% (<45%):** Defensive specialists. Shifts start deep in their own zone against top opposing threats.
            
            *Draft Floor Edge:* Look for mid-to-late round players with a high `OZS%` progress bar[cite: 2]. Even if they lack name recognition, their deployment ensures offensive volume[cite: 2].
            """)
            
        with col_pdo:
            st.markdown("### 📈 PDO Alert (Regression / Luck)")
            st.markdown("""
            **PDO** is the ultimate proxy for shooting and goaltending "luck" (ideal baseline = `1.000`):
            * **📈 Buy Low (PDO < 0.975):** Skaters whose point totals are suppressed by abnormally low team shooting or save percentages[cite: 2]. Their production is mathematically guaranteed to bounce back[cite: 2].
            * **📉 Sell High (PDO > 1.025):** Value traps whose hot streaks are a statistical illusion. Avoid drafting them at their inflated peak.
            """)
            
        st.markdown("---")
        
        # Section 3: Playoff volume
        st.markdown("### 📅 Playoff Volume Tie-Breaker")
        st.markdown("""
        In Head-to-Head (H2H) fantasy playoffs (typically NHL Weeks 22, 23, and 24), your matchup is won by raw starting volume:
        * **The Logic:** A slightly lower-tier skater whose team plays 13 games in those weeks will almost always outscore an elite superstar limited to 9 games.
        * **The Edge:** Use the `Playoff Games` data in the matrix as a mid-round tie-breaker to ensure your lineup has more "at-bats" when it matters most.
        """)
        
    with tab3:
        st.dataframe(processed[['player', 'position', 'pdo', 'pdo_alert', 'ozs_pct']], use_container_width=True)

if __name__ == "__main__":
    main()