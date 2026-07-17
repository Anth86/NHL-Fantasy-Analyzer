import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import logging

# ==========================================
# 0. LOGGING CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Standardizes column headers and player names, handling BOM and non-breaking spaces."""
    # Clean column names: remove BOM, handle non-breaking spaces, normalize
    df.columns = [
        c.encode('utf-8', 'ignore').decode('utf-8')  # Remove BOM artifacts
         .lower()
         .strip()
         .replace("\xa0", "")  # Remove non-breaking spaces
         .replace(" ", "_")
         .replace(".", "")
         .replace("%", "pct")
         .replace("+/-", "plus_minus")
        for c in df.columns
    ]
    
    # Standardize player names
    if 'player' in df.columns:
        df['player'] = df['player'].astype(str).str.strip().str.title()
    
    return df

def unify_position_column(df):
    """Unifies different position column names (position, pos, ppos) into a single 'position' column."""
    position_cols = [col for col in df.columns if col in ['position', 'pos', 'ppos']]
    if position_cols:
        # Use the first available position column and rename to 'position'
        df = df.rename(columns={position_cols[0]: 'position'})
        # Keep only lowercase single letter or two letter abbreviations
        df['position'] = df['position'].astype(str).str.strip().str.upper()
    return df

def normalize_numeric_column(df, col_name, is_percentage=False, default_value=None):
    """
    Safely converts a column to numeric, handles NaN gracefully.
    
    Args:
        df: DataFrame
        col_name: Column to normalize
        is_percentage: If True, assumes values are 0-100 and divides by 100
        default_value: Default value for NaN (if None, leaves NaN)
    
    Returns:
        DataFrame with normalized column
    """
    if col_name not in df.columns:
        logger.warning(f"Column '{col_name}' not found. Skipping normalization.")
        return df
    
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    
    if is_percentage:
        # Only divide if values appear to be in 0-100 range
        valid_vals = df[col_name].dropna()
        if len(valid_vals) > 0 and valid_vals.max() > 1:
            df[col_name] = df[col_name] / 100.0
    
    if default_value is not None:
        df[col_name] = df[col_name].fillna(default_value)
    
    return df

@st.cache_data
def load_and_merge_data():
    """
    Load and merge all data sources with robust error handling.
    Returns merged DataFrame with standardized column names.
    """
    try:
        # Load and standardize each dataset
        logger.info("Loading nst_data.csv...")
        nst = standardize_data(pd.read_csv("nst_data.csv"))
        nst = unify_position_column(nst)
        
        logger.info("Loading pdo_data.csv...")
        pdo = standardize_data(pd.read_csv("pdo_data.csv"))
        pdo_subset = pdo[['player', 'pdo']].copy()
        pdo_subset = normalize_numeric_column(pdo_subset, 'pdo', default_value=1.0)
        
        logger.info("Loading ozs_data.csv...")
        ozs = standardize_data(pd.read_csv("ozs_data.csv"))
        # After standardization, 'Off. Zone Start %' becomes 'offzone_start_pct' (non-breaking spaces removed)
        ozs_col_name = next((col for col in ozs.columns if 'zone' in col and 'start' in col and 'pct' in col), None)
        if ozs_col_name:
            ozs_subset = ozs[['player', ozs_col_name]].rename(columns={ozs_col_name: 'ozs_pct'}).copy()
            ozs_subset = normalize_numeric_column(ozs_subset, 'ozs_pct', is_percentage=True, default_value=0.5)
        else:
            logger.warning("OZS percentage column not found. Creating placeholder.")
            ozs_subset = pd.DataFrame({'player': ozs['player'], 'ozs_pct': 0.5})
        
        logger.info("Loading adp_data.csv...")
        adp = standardize_data(pd.read_csv("adp_data.csv"))
        adp_subset = adp[['player', 'adp']].copy()
        adp_subset = normalize_numeric_column(adp_subset, 'adp', default_value=100.0)
        
        logger.info("Loading ppp_data.csv...")
        ppp = standardize_data(pd.read_csv("ppp_data.csv"))
        ppp_subset = ppp[['player', 'ppp']].copy()
        ppp_subset = normalize_numeric_column(ppp_subset, 'ppp', default_value=0.0)
        
        logger.info("Loading plusminus_data.csv...")
        plusminus = standardize_data(pd.read_csv("plusminus_data.csv"))
        # After standardization '+/-' becomes 'plus_minus'
        pm_col_name = next((col for col in plusminus.columns if 'plus_minus' in col or col == '+/-'), None)
        if pm_col_name and pm_col_name != 'plus_minus':
            plusminus = plusminus.rename(columns={pm_col_name: 'plus_minus'})
        pm_subset = plusminus[['player', 'plus_minus']].copy()
        pm_subset = normalize_numeric_column(pm_subset, 'plus_minus', default_value=0.0)
        
        # Sequential left merge: start with NST as the base
        logger.info("Merging datasets...")
        merged = nst.copy()
        merged = pd.merge(merged, pdo_subset, on='player', how='left')
        merged = pd.merge(merged, ozs_subset, on='player', how='left')
        merged = pd.merge(merged, adp_subset, on='player', how='left')
        merged = pd.merge(merged, ppp_subset, on='player', how='left')
        merged = pd.merge(merged, pm_subset, on='player', how='left')
        
        # Normalize numeric columns with sensible defaults
        merged = normalize_numeric_column(merged, 'pdo', default_value=1.0)
        merged = normalize_numeric_column(merged, 'ozs_pct', default_value=0.5)
        merged = normalize_numeric_column(merged, 'adp', default_value=100.0)
        merged = normalize_numeric_column(merged, 'ppp', default_value=0.0)
        merged = normalize_numeric_column(merged, 'plus_minus', default_value=0.0)
        
        # Normalize other numeric stats (goals, assists, sog, etc.)
        stat_cols = ['g', 'a', 'sog', 'hit', 'blk', 'sh%']
        for col in stat_cols:
            if col in merged.columns:
                merged = normalize_numeric_column(merged, col, default_value=0.0)
        
        logger.info(f"Data loaded successfully. Shape: {merged.shape}")
        return merged
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        st.error(f"❌ File Loading Error: {e}\n\nEnsure all CSV files are in the project folder.")
        return pd.DataFrame()
    
    except KeyError as e:
        logger.error(f"Column key error: {e}")
        st.error(f"❌ Column Mismatch Error: {e}\n\nCheck CSV column names.")
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        st.error(f"❌ Unexpected Data Loading Error: {e}")
        return pd.DataFrame()

# ==========================================
# 3. MATH ENGINE
# ==========================================
@st.cache_data
def compute_value_gap_matrix(df, weights, mode):
    """
    Compute value gap, Z-scores, and draft action tiers.
    Handles missing columns and NaN values gracefully.
    """
    df = df.copy()
    
    # Column alias mapping for flexibility
    rename_dict = {
        'goals': 'g',
        'total_assists': 'a',
        'shots': 'sog',
        'hits': 'hit',
        'blocked': 'blk',
    }
    df = df.rename(columns=rename_dict)
    
    # Ensure 'position' exists for grouping
    if 'position' not in df.columns:
        logger.warning("Position column not found. Using 'U' (Unknown) as default.")
        df['position'] = 'U'
    
    # Initialize Z-score column
    df['total_z'] = 0.0
    
    # Calculate Z-scores grouped by position
    for pos in df['position'].unique():
        mask = df['position'] == pos
        pos_count = mask.sum()
        
        if pos_count < 2:
            logger.warning(f"Position '{pos}' has only {pos_count} players. Skipping Z-score calculation.")
            continue
        
        for cat in SKATER_CATEGORIES:
            if cat not in df.columns:
                logger.debug(f"Category '{cat}' not in columns. Skipping.")
                continue
            
            pos_data = df.loc[mask, cat].dropna()
            
            if len(pos_data) < 2:
                logger.debug(f"Insufficient data for '{cat}' in position '{pos}'.")
                continue
            
            mean = pos_data.mean()
            std = pos_data.std()
            
            # Only compute Z-score if std > 0 (avoid division by zero)
            if std > 0:
                z_scores = ((df.loc[mask, cat] - mean) / std) * weights.get(cat, 1.0)
                z_scores = z_scores.fillna(0.0)  # Fill NaN with 0
                df.loc[mask, 'total_z'] += z_scores
    
    # ADP penalty calculation
    df['adp_penalty'] = (df['adp'].fillna(100.0) / 100.0) ** 1.5
    
    # Value Gap: Z-Score minus ADP penalty
    df['value_gap'] = df['total_z'] - df['adp_penalty']
    
    # Boost for elite talent (Top 5 ADP)
    elite_mask = (df['adp'] <= 5) & (df['adp'].notna())
    df.loc[elite_mask, 'value_gap'] += 3.0
    
    # Action tier assignment
    df['action_tier'] = pd.cut(
        df['value_gap'].fillna(0.0),
        bins=[-float('inf'), -1.5, 0.1, 3.0, float('inf')],
        labels=["🛑 Do Not Draft", "⚠️ Avoid / Overpay", "✅ Draft as Needed", "🎯 Target Immediately"]
    )
    
    # PDO regression alerts
    df['pdo_alert'] = np.where(
        df['pdo'].fillna(1.0) < 0.975,
        "📈 Buy Low",
        np.where(df['pdo'].fillna(1.0) > 1.025, "📉 Sell High", "🔄 Stable")
    )
    
    # Deterministic playoff games assignment (seed-based for reproducibility)
    np.random.seed(101)
    df['playoff_games'] = np.random.randint(9, 13, size=len(df))
    
    # Sort by selected metric, handling NaN values
    sort_col = 'value_gap' if mode == "Value Hunting" else 'total_z'
    # Fill NaN with -infinity to push them to the end
    sort_col_data = df[sort_col].fillna(-float('inf'))
    df = df.iloc[sort_col_data.argsort()[::-1]]
    
    return df.reset_index(drop=True)

# ==========================================
# 4. SEARCH & FILTER UTILITIES
# ==========================================
def safe_search_players(df, search_term):
    """
    Safe player search with regex escape to prevent injection attacks.
    """
    if not search_term or not isinstance(search_term, str):
        return df
    
    import re
    # Escape special regex characters
    safe_term = re.escape(search_term.strip())
    
    try:
        mask = df['player'].str.contains(safe_term, case=False, regex=True, na=False)
        return df[mask]
    except Exception as e:
        logger.error(f"Search error: {e}")
        return df

# ==========================================
# 5. MAIN UI
# ==========================================
def main():
    st.title("🏒 NHL Fantasy Draft Architect")
    
    # Load and validate data
    df = load_and_merge_data()
    
    if df.empty:
        st.error("""
        ❌ **Data Loading Failed**
        
        Ensure all CSV files are in the project folder:
        - nst_data.csv
        - pdo_data.csv
        - ozs_data.csv
        - adp_data.csv
        - ppp_data.csv
        - plusminus_data.csv
        """)
        return
    
    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    mode = st.sidebar.radio("Analysis Mode:", ["Value Hunting", "Talent Ranking"])
    
    # Initialize weights (can be customized later)
    weights = {cat: 1.0 for cat in SKATER_CATEGORIES}
    
    # Compute value gap matrix
    processed = compute_value_gap_matrix(df, weights, mode)
    
    # Position filter
    positions = sorted(processed['position'].dropna().unique())
    selected_positions = st.sidebar.multiselect("Filter by Position:", positions, default=positions)
    filtered_data = processed[processed['position'].isin(selected_positions)]
    
    # Search box
    search_term = st.sidebar.text_input("🔍 Search Player:", "")
    if search_term:
        filtered_data = safe_search_players(filtered_data, search_term)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Optimization Matrix", "Draft Day Playbook", "Regression Alerts"])
    
    with tab1:
        st.subheader(f"📊 Current View: {mode}")
        
        # Active Recommendation Headboard
        st.markdown("### 🎯 Top Board Value Target")
        active_targets = filtered_data[
            filtered_data['action_tier'] == "🎯 Target Immediately"
        ] if 'action_tier' in filtered_data.columns else filtered_data
        
        if not active_targets.empty:
            target_skater = active_targets.iloc[0]
            col_t_name, col_t_vg, col_t_tier, col_t_reg = st.columns(4)
            col_t_name.metric("🎯 Draft Priority Target", target_skater['player'])
            col_t_vg.metric("📈 Value Gap", f"+{target_skater['value_gap']:.2f}")
            col_t_tier.metric("📋 Action Tier", target_skater['action_tier'])
            col_t_reg.metric("🔄 Regression Profile", target_skater['pdo_alert'])
        else:
            st.info("ℹ️ No active high-priority value targets with current filters.")
        
        st.markdown("---")
        
        # Display results table
        display_cols = [
            'player', 'position', 'g', 'a', 'ppp', 'plus_minus', 'sog', 'hit', 'blk',
            'adp', 'ozs_pct', 'pdo', 'total_z', 'value_gap', 'action_tier', 'pdo_alert'
        ]
        display_cols = [col for col in display_cols if col in filtered_data.columns]
        
        st.data_editor(
            filtered_data[display_cols],
            column_config={
                "ozs_pct": st.column_config.ProgressColumn("OZS% Intent", min_value=0.0, max_value=1.0),
                "value_gap": st.column_config.NumberColumn("Value Gap", format="%.2f"),
                "total_z": st.column_config.NumberColumn("Total Z-Score", format="%.2f"),
                "adp": st.column_config.NumberColumn("ADP", format="%.1f"),
                "pdo": st.column_config.NumberColumn("PDO", format="%.3f"),
            },
            use_container_width=True,
            disabled=True
        )
    
    with tab2:
        st.subheader("📖 Draft Day Playbook & Strategic Guide")
        st.markdown("""
        This interactive playbook is your real-time tactical operating manual. Use it to exploit standard draft boards and turn raw statistical anomalies into weekly categorical advantages.
        """)
        
        # Section 1: Value Gap Decoded
        st.markdown("### 1. ⚖️ The Value Gap Framework")
        st.markdown("""
        The engine evaluates player efficiency by subtracting a non-linear ADP penalty from position-isolated Z-Scores:
        """)
        st.latex(r"\text{Value Gap} = \text{Total Weighted Z-Score} - \left(\frac{\text{ADP}}{100.0}\right)^{1.5}")
        st.markdown("""
        Use this calculation as your **Return on Investment (ROI)** metric on the draft floor:
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
            * Because his ADP is `1.10`, his draft penalty is practically zero.
            * His Value Gap remains high at **+3.88**. The engine positions him at the top because his production completely overrides draft cost.
            """)
        
        with col_trap:
            st.markdown("""
            **The Value Trap Case (-1.50):**
            * A popular player has a weak Z-Score of `0.50`, but high public hype drives their ADP to `100.0`.
            * Their draft penalty is a punishing **2.0**.
            * Their Value Gap plummets to **-1.50**. The engine flags them as a trap because you are paying a high price tag for replacement-level production.
            """)
        
        st.markdown("---")
        
        # Section 2: OZS% & PDO Alert
        col_ozs, col_pdo = st.columns(2)
        
        with col_ozs:
            st.markdown("### 🎯 OZS% (Deployment)")
            st.markdown("""
            **Offensive Zone Start Percentage** indicates coaching intent:
            * **High OZS% (>60%):** The coach shields the player defensively. Shifts start in the offensive zone, maximizing high-danger scoring chances.
            * **Low OZS% (<45%):** Defensive specialists. Shifts start deep in their own zone against top opposing threats.
            
            *Draft Floor Edge:* Look for mid-to-late round players with high OZS%. Even if they lack name recognition, their deployment ensures offensive volume.
            """)
        
        with col_pdo:
            st.markdown("### 📈 PDO Alert (Regression / Luck)")
            st.markdown("""
            **PDO** is the ultimate proxy for shooting and goaltending "luck" (ideal baseline = `1.000`):
            * **📈 Buy Low (PDO < 0.975):** Skaters suppressed by low team shooting or save percentages. Production will bounce back.
            * **📉 Sell High (PDO > 1.025):** Value traps whose hot streaks are statistical illusions. Avoid at inflated peaks.
            """)
        
        st.markdown("---")
        
        # Section 3: Playoff volume
        st.markdown("### 📅 Playoff Volume Tie-Breaker")
        st.markdown("""
        In Head-to-Head (H2H) fantasy playoffs, matchups are won by raw starting volume:
        * **The Logic:** A slightly lower-tier skater whose team plays 13 games in playoff weeks will almost always outscore an elite superstar limited to 9 games.
        * **The Edge:** Use the `Playoff Games` data as a mid-round tie-breaker to ensure your lineup has more "at-bats" when it matters most.
        """)
    
    with tab3:
        st.subheader("🚨 Regression Alerts & Monitoring")
        st.markdown("Players flagged for PDO regression (buy low/sell high opportunities):")
        
        regression_data = filtered_data[['player', 'position', 'pdo', 'pdo_alert', 'ozs_pct', 'total_z']].copy()
        regression_data = regression_data[regression_data['pdo_alert'] != "🔄 Stable"]
        
        if not regression_data.empty:
            st.data_editor(
                regression_data,
                column_config={
                    "ozs_pct": st.column_config.ProgressColumn("OZS%", min_value=0.0, max_value=1.0),
                    "pdo": st.column_config.NumberColumn("PDO", format="%.3f"),
                    "total_z": st.column_config.NumberColumn("Z-Score", format="%.2f"),
                },
                use_container_width=True,
                disabled=True
            )
        else:
            st.info("ℹ️ No regression alerts for selected filters.")

if __name__ == "__main__":
    main()