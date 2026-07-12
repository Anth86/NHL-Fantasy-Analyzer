import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# 1. LEAGUE CONFIGURATION
# ==========================================
ROSTER_LAYOUT = {"C": 2, "LW": 2, "RW": 2, "D": 4, "G": 2, "UTIL": 1, "BENCH": 3}
SKATER_CATEGORIES = ['g', 'a', 'plus_minus', 'ppp', 'sog', 'hit', 'blk']

st.set_page_config(page_title="NHL Draft Engine", layout="wide")

# ==========================================
# 2. STATE & DATA LOADING (FIXED MERGE)
# ==========================================
@st.cache_data
def load_data():
    try:
        # Load raw files
        nst = pd.read_csv('nst_data.csv')
        adp = pd.read_csv('adp_data.csv')
        pm = pd.read_csv('plusminus.data.csv')
        pdo = pd.read_csv('pdo_data.csv')
        
        # Standardize 'player' key to lower case
        for df in [nst, adp, pdo, pm]:
            df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
            if 'player' in df.columns: 
                df['player'] = df['player'].astype(str).str.strip().str.lower()
        
        # 1. Start with master NST, and rename the plus/minus column
        pm = pm.rename(columns={'+/-': 'plus_minus'})
        
        # 2. Build the merge chain (Only keep Key + Required Stats)
        df = nst
        df = pd.merge(df, pdo[['player', 'pdo']], on='player', how='left')
        df = pd.merge(df, pm[['player', 'plus_minus']], on='player', how='left')
        
        # 3. Handle ADP (select the rank/adp column specifically)
        possible_adp = ['adp', 'avg', 'yahoo', 'espn', 'cbs']
        adp_col = next((c for c in adp.columns if c in possible_adp), None)
        if adp_col:
            df = pd.merge(df, adp[['player', adp_col]].rename(columns={adp_col: 'adp'}), on='player', how='left')
        
        # 4. Safe Numerical Conversions
        df['adp'] = pd.to_numeric(df['adp'], errors='coerce').fillna(250.0)
        df['pdo'] = pd.to_numeric(df['pdo'], errors='coerce').fillna(1.0)
        df['plus_minus'] = pd.to_numeric(df['plus_minus'], errors='coerce').fillna(0.0)
        
        # 5. Math Engine (Z-Scores)
        cols_to_z = [c for c in ['g', 'a', 'sog', 'hit', 'blk'] if c in df.columns]
        for col in cols_to_z:
            df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()
        
        df['total_z'] = df[[c for c in df.columns if c.endswith('_z')]].sum(axis=1)
        df['value_gap'] = df['total_z'] - (df['adp'] / 50.0)
        
        return df.set_index('player'), None
    except Exception as e:
        return None, f"Merge Error: {str(e)}"

# ==========================================
# 3. INTERFACE
# ==========================================
def main():
    st.title("🏒 NHL Fantasy Draft Architect")
    df, error = load_data()
    if error: st.error(error); return

    tab1, tab2, tab3 = st.tabs(["Draft Board", "Compare Players", "Team Rating"])
    
    with tab1:
        st.dataframe(df, use_container_width=True)
        
    with tab2:
        c1, c2 = st.columns(2)
        p1 = c1.selectbox("Player 1", df.index.unique(), key="p1")
        p2 = c2.selectbox("Player 2", df.index.unique(), key="p2")
        st.dataframe(df.loc[[p1, p2]], use_container_width=True)
        
    with tab3:
        st.subheader("Team Roster Evaluation")
        if 'my_roster' not in st.session_state: st.session_state.my_roster = {pos: [] for pos in ROSTER_LAYOUT.keys()}
        all_players = [p for players in st.session_state.my_roster.values() for p in players]
        
        if not all_players:
            st.info("Draft players to see your team rating.")
        else:
            team_df = df.loc[df.index.isin(all_players)]
            st.metric("Total Roster Value Gap", f"{team_df['value_gap'].sum():.2f}")
            st.dataframe(team_df, use_container_width=True)

if __name__ == "__main__":
    main()