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
TOTAL_ROSTER_SIZE = sum(ROSTER_LAYOUT.values())
CATEGORY_LABELS = {
    'g': 'Goals', 'a': 'Assists', 'plus_minus': 'Plus/Minus', 'ppp': 'Powerplay Points',
    'sog': 'Shots on Goal', 'hit': 'Hits', 'blk': 'Blocks',
}
# Goals/Assists are already the entire focus of Diamond Score v2 (via Points),
# so Category Specialists covers the other five. HIT/BLK/SOG accrue in every
# game situation, so per-60-of-TOI is a clean rate (same logic as Points/60
# elsewhere). PPP only accrues on the power play specifically, and Plus/Minus
# is defined as an even-strength, non-shorthanded stat by the NHL itself --
# for both, the only TOI available here is all-situations, so dividing by it
# would mismatch the stat's actual context and understate real specialists.
# Those two use a raw position-relative percentile instead of a rate.
CATEGORY_SPECIALIST_OPTIONS = {
    'hit': {'label': 'Hits', 'method': 'rate'},
    'blk': {'label': 'Blocks', 'method': 'rate'},
    'sog': {'label': 'Shots on Goal', 'method': 'rate'},
    'ppp': {'label': 'Powerplay Points', 'method': 'raw'},
    'plus_minus': {'label': 'Plus/Minus', 'method': 'raw'},
}

# --- Positional VORP configuration ---
# Draftable skater pool = every starting skater slot across the league
# (excludes goalies, bench, and UTIL isn't position-specific so it's included
# in the pool size but not given its own replacement-level bucket below).
DRAFTABLE_SKATER_POOL_SIZE = (
    ROSTER_LAYOUT.get('C', 0) + ROSTER_LAYOUT.get('LW', 0) +
    ROSTER_LAYOUT.get('RW', 0) + ROSTER_LAYOUT.get('D', 0) +
    ROSTER_LAYOUT.get('UTIL', 0)
) * LEAGUE_SIZE
# The data's position column uses base codes (C, D, L, R) rather than
# LW/RW, and multi-eligible players are labeled with combo strings like
# "C, L" -- VORP groups every player by their PRIMARY (first-listed)
# position so a "C, L" player is compared against full-time Cs, consistent
# with how a real draft only lets that body fill one slot at a time.
VORP_REPLACEMENT_RANK = {
    'C': ROSTER_LAYOUT.get('C', 0) * LEAGUE_SIZE + 1,
    'L': ROSTER_LAYOUT.get('LW', 0) * LEAGUE_SIZE + 1,
    'R': ROSTER_LAYOUT.get('RW', 0) * LEAGUE_SIZE + 1,
    'D': ROSTER_LAYOUT.get('D', 0) * LEAGUE_SIZE + 1,
}
# Each position gets its OWN draft pool (2x its replacement rank, for
# comfortable margin), rather than competing for a single shared top-132
# pool ranked by cross-position total_z. Defensemen naturally post lower
# total_z than forwards (fewer skater categories favor D-specific play), so
# a shared pool silently under-fills D and RW -- confirmed empirically: a
# shared top-132 pool contained only 42 D against a target replacement rank
# of 49, and only 24 R against a target of 25, both falling back to
# whichever player happened to be last in a too-small pool instead of
# hitting the actual intended rank.
VORP_POSITION_POOL_SIZE = {pos: rank * 2 for pos, rank in VORP_REPLACEMENT_RANK.items()}

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

def add_base_position(df):
    """
    Adds 'base_position' = primary (first-listed) position, e.g. "C, L" -> "C".
    Every STATISTICAL grouping in this app (Z-scores, percentiles, replacement
    levels) must group by this, never by the raw 'position' column.
    
    Reason: a handful of players carry multi-position combo labels like
    "C, L" or "L, R" (8 in the current dataset). Grouping by the raw combo
    string treats each unique combo as its own tiny, separate population --
    confirmed empirically at just 2 players sharing "C, R" against a real C
    population of 233+ and a real R population of 114+. With a group of 2,
    one player is mathematically guaranteed a 100th-percentile score and the
    other a 50th, regardless of actual talent -- this produced a real,
    visible failure: Ethen Frank showed a perfect 100.0 Diamond Score (best
    in the dataset) while the ORIGINAL Value Gap engine, hitting the same
    bug independently, rated the same player "Do Not Draft." Neither number
    reflected anything real; both were coin-flip artifacts of a 2-person
    group. 'position' (the full combo string) is still kept and displayed
    everywhere in the UI so dual-eligibility remains visible -- only the
    grouping key changes.
    """
    df = df.copy()
    if 'position' in df.columns:
        df['base_position'] = df['position'].astype(str).str.split(',').str[0].str.strip()
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

def dedupe_by_player(df, prefer_col=None):
    """
    Guarantees at most one row per 'player' before any merge.
    
    Real NHL data occasionally splits one identity across multiple rows under
    the same name (e.g. a skater whose season got exported once per position
    played), or two different people can share a name. Left on its own,
    pd.merge(..., on='player') treats every duplicate as a match, so a single
    2-row name compounds into a cross-product explosion across a chain of
    sequential merges (2 rows -> 4 -> 8 -> ... after N merges) and silently
    pollutes every downstream percentile/Z-score with garbage rows.
    
    If prefer_col is given, keeps the row with the larger value in that
    column (e.g. more games played = more likely the "main" stat line).
    Otherwise keeps the first occurrence. Either way, exactly one row per
    player survives, so every merge downstream is safe.
    """
    if 'player' not in df.columns:
        return df
    dupe_names = df['player'][df['player'].duplicated(keep=False)].unique()
    if len(dupe_names) > 0:
        logger.warning(f"Duplicate player name(s) found, collapsing to one row each: {list(dupe_names)}")
    if prefer_col and prefer_col in df.columns:
        df = df.sort_values(prefer_col, ascending=False, na_position='last')
    return df.drop_duplicates(subset='player', keep='first').reset_index(drop=True)

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
        if 'g' in nst.columns and 'a' in nst.columns:
            nst['_prod_tiebreak'] = pd.to_numeric(nst['g'], errors='coerce').fillna(0) + pd.to_numeric(nst['a'], errors='coerce').fillna(0)
            nst = dedupe_by_player(nst, prefer_col='_prod_tiebreak').drop(columns=['_prod_tiebreak'])
        else:
            nst = dedupe_by_player(nst)
        
        logger.info("Loading pdo_data.csv...")
        pdo = standardize_data(pd.read_csv("pdo_data.csv"))
        pdo_subset = pdo[['player', 'pdo']].copy()
        pdo_subset = normalize_numeric_column(pdo_subset, 'pdo', default_value=1.0)
        pdo_subset = dedupe_by_player(pdo_subset)
        
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
        ozs_subset = dedupe_by_player(ozs_subset)
        
        logger.info("Loading adp_data.csv...")
        adp = standardize_data(pd.read_csv("adp_data.csv"))
        adp_subset = adp[['player', 'adp']].copy()
        # Leave missing/unparseable ADP as NaN -- a player outside the ADP list has
        # no market price (they go undrafted in typical leagues). Baking a fake 100.0
        # into the column made that invented number display as if real. The Value Gap
        # penalty applies its own neutral fillna(100.0) inline, so every computed
        # number (penalty, value_gap, tiers) is unchanged by this -- display only.
        adp_subset = normalize_numeric_column(adp_subset, 'adp')
        adp_subset = dedupe_by_player(adp_subset, prefer_col='adp')  # lower ADP = more prominent; sort desc then keep-first still picks one deterministically, exact winner doesn't matter since duplicates are rare
        
        logger.info("Loading ppp_data.csv...")
        ppp = standardize_data(pd.read_csv("ppp_data.csv"))
        ppp_subset = ppp[['player', 'ppp']].copy()
        ppp_subset = normalize_numeric_column(ppp_subset, 'ppp', default_value=0.0)
        ppp_subset = dedupe_by_player(ppp_subset, prefer_col='ppp')
        
        logger.info("Loading plusminus_data.csv...")
        plusminus = standardize_data(pd.read_csv("plusminus_data.csv"))
        # After standardization '+/-' becomes 'plus_minus'
        pm_col_name = next((col for col in plusminus.columns if 'plus_minus' in col or col == '+/-'), None)
        if pm_col_name and pm_col_name != 'plus_minus':
            plusminus = plusminus.rename(columns={pm_col_name: 'plus_minus'})
        pm_subset = plusminus[['player', 'plus_minus']].copy()
        pm_subset = normalize_numeric_column(pm_subset, 'plus_minus', default_value=0.0)
        pm_subset = dedupe_by_player(pm_subset)
        
        logger.info("Loading finddiamonds_data.csv...")
        try:
            breakout = pd.read_csv("finddiamonds_data.csv")
            # Drop stray index column some exports include
            if breakout.columns[0].lower().startswith('unnamed'):
                breakout = breakout.drop(columns=[breakout.columns[0]])
            breakout = standardize_data(breakout)
            breakout = unify_position_column(breakout)
            # Standardize column names for breakout data
            if 'name' in breakout.columns:
                breakout = breakout.rename(columns={'name': 'player'})
            
            # Dynamic detection: handles 'toi' (finddiamonds_data.csv) or
            # 'evtoi' (legacy hockey_breakout_data.csv format)
            toi_col_name = next((col for col in breakout.columns if col in ['toi', 'evtoi']), None)
            cf_col_name = next((col for col in breakout.columns if col == 'cfpct'), None)
            # Optional columns for Diamond Score v2 (only in finddiamonds_data.csv,
            # not the legacy hockey_breakout_data.csv format)
            gp_col_name = next((col for col in breakout.columns if col == 'gp'), None)
            xgf_col_name = next((col for col in breakout.columns if col == 'xgfpct'), None)
            hdcf_col_name = next((col for col in breakout.columns if col == 'hdcfpct'), None)
            
            if toi_col_name and cf_col_name:
                cols_to_pull = ['player', toi_col_name, cf_col_name]
                rename_map = {toi_col_name: 'toi_min', cf_col_name: 'cf_pct'}
                
                for src_col, dest_col in [(gp_col_name, 'gp'), (xgf_col_name, 'xgf_pct'), (hdcf_col_name, 'hdcf_pct')]:
                    if src_col:
                        cols_to_pull.append(src_col)
                        rename_map[src_col] = dest_col
                
                breakout_subset = breakout[cols_to_pull].copy()
                breakout_subset = breakout_subset.rename(columns=rename_map)
                breakout_subset = normalize_numeric_column(breakout_subset, 'toi_min', default_value=0.0)
                breakout_subset = normalize_numeric_column(breakout_subset, 'cf_pct', default_value=50.0)
                if 'gp' in breakout_subset.columns:
                    breakout_subset = normalize_numeric_column(breakout_subset, 'gp', default_value=0.0)
                if 'xgf_pct' in breakout_subset.columns:
                    breakout_subset = normalize_numeric_column(breakout_subset, 'xgf_pct', default_value=50.0)
                if 'hdcf_pct' in breakout_subset.columns:
                    breakout_subset = normalize_numeric_column(breakout_subset, 'hdcf_pct', default_value=50.0)
                
                missing_for_v2 = [name for name, col in [('GP', gp_col_name), ('xGF%', xgf_col_name), ('HDCF%', hdcf_col_name)] if not col]
                if missing_for_v2:
                    logger.warning(f"Diamond Score v2 will be limited: missing {missing_for_v2} in finddiamonds_data.csv.")
                breakout_subset = dedupe_by_player(breakout_subset, prefer_col='toi_min')
                logger.info(f"Loaded {len(breakout_subset)} players with advanced stats for Diamond modules.")
            else:
                logger.warning(f"TOI or CF% column not found. Columns available: {breakout.columns.tolist()}")
                breakout_subset = pd.DataFrame({'player': [], 'toi_min': [], 'cf_pct': []})
        except FileNotFoundError:
            logger.warning("finddiamonds_data.csv not found. Module 1 (Diamonds) will be disabled.")
            breakout_subset = pd.DataFrame({'player': [], 'toi_min': [], 'cf_pct': []})
        
        # Sequential left merge: start with NST as the base
        logger.info("Merging datasets...")
        merged = nst.copy()
        merged = pd.merge(merged, pdo_subset, on='player', how='left')
        merged = pd.merge(merged, ozs_subset, on='player', how='left')
        merged = pd.merge(merged, adp_subset, on='player', how='left')
        merged = pd.merge(merged, ppp_subset, on='player', how='left')
        merged = pd.merge(merged, pm_subset, on='player', how='left')
        merged = pd.merge(merged, breakout_subset, on='player', how='left')
        
        # Normalize numeric columns with sensible defaults
        merged = normalize_numeric_column(merged, 'pdo', default_value=1.0)
        merged = normalize_numeric_column(merged, 'ozs_pct', default_value=0.5)
        merged = normalize_numeric_column(merged, 'adp')  # NaN preserved for undrafted -- see note above
        merged = normalize_numeric_column(merged, 'ppp', default_value=0.0)
        merged = normalize_numeric_column(merged, 'plus_minus', default_value=0.0)
        
        # Normalize other numeric stats (goals, assists, sog, etc.)
        stat_cols = ['g', 'a', 'sog', 'hit', 'blk', 'sh%']
        for col in stat_cols:
            if col in merged.columns:
                merged = normalize_numeric_column(merged, col, default_value=0.0)
        
        merged = add_base_position(merged)
        
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
    
    # base_position (primary position, e.g. "C, L" -> "C") is the correct
    # grouping key -- see add_base_position(). Compute it defensively here
    # too in case this function is ever called on a dataframe that skipped
    # load_and_merge_data().
    if 'base_position' not in df.columns:
        df = add_base_position(df)
    
    # Initialize Z-score column, plus one raw (unweighted) per-category Z column
    # so individual category strength/weakness can be analyzed later (e.g. for
    # a drafted team's category profile) independent of the draft-priority weights.
    df['total_z'] = 0.0
    for cat in SKATER_CATEGORIES:
        df[f'z_{cat}'] = 0.0
    
    # Calculate Z-scores grouped by PRIMARY position, not the raw combo
    # string -- grouping by "C, L" as its own bucket splinters ~8 dual-
    # eligible players into groups of 2-4, where percentile-style extremes
    # (100th, 50th) are mathematical guarantees regardless of talent, not
    # signal. See add_base_position() for the confirmed real-world case
    # this caused.
    for pos in df['base_position'].unique():
        mask = df['base_position'] == pos
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
                raw_z = ((df.loc[mask, cat] - mean) / std).fillna(0.0)
                df.loc[mask, f'z_{cat}'] = raw_z
                df.loc[mask, 'total_z'] += raw_z * weights.get(cat, 1.0)
    
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
        "📈 Undervalued",
        np.where(df['pdo'].fillna(1.0) > 1.025, "📉 Overvalued", "🔄 Fairly Valued")
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
# 4B. MODULE 1 v2: DIAMOND SCORE (WEIGHTED, ADVANCED)
# ==========================================
LEAGUE_BASELINE_SH_PCT = 10.0   # League-average individual shooting %, used as the
                                  # regression baseline for the Shooting Luck factor.
PDO_BASELINE = 1.000             # Neutral on-ice PDO; below this = on-ice bad luck
                                  # (own SH% + goalie SV% while player is on ice).
OZS_DRAG_FULL_CREDIT = 0.35      # At/below this OZS%, Deployment Drag gets full (100%) credit.
OZS_DRAG_NO_CREDIT = 0.55        # At/above this OZS%, Deployment Drag gets zero credit.
                                  # Linear ramp between the two, replacing a hard 45% cliff
                                  # that left most of the eligible pool flatlined at exactly 0.

@st.cache_data
def calculate_diamond_scores_v2(df, gp_threshold=20, w_shooting=0.35, w_quality=0.35, w_deployment=0.30):
    """
    Diamond Score v2: a weighted composite (0-100), not a strict multi-factor
    AND filter. Requiring all factors to clear a bar simultaneously tends to
    return zero players on real data (as Module 1's original design did before
    the TOI-eligibility fix) -- a weighted average always yields a full,
    rankable list instead.

    Three factors, each converted to a 0-100 percentile rank computed WITHIN
    POSITION GROUP and only among players meeting the GP threshold:

      1. Shooting Luck: blends two regression signals -- individual iSh%
         sitting below the ~10% league baseline, AND on-ice PDO sitting below
         1.000 (own shooting % + goalie SV% while on the ice). Two different
         levels of "results are worse than the process," averaged together
         after each is independently percentile-ranked.
      2. Underlying Quality: blends CF%, xGF%, and HDCF% -- whichever are
         available -- each measured relative to the player's own team average
         (not the league), so it reflects who's outproducing their actual
         linemates/deployment context, not just playing on a dominant team.
      3. Zone Deployment Drag: points-per-60 percentile, scaled by a
         continuous credit multiplier based on OZS% -- full credit at/below
         35%, zero credit at/above 55%, linear in between. Replaces a hard
         45% cutoff that left most of the eligible pool flatlined at exactly
         0 with no graduation.

    The GP threshold is applied FIRST, before any percentile is computed, so
    call-ups and injury-shortened seasons never distort the eligible pool's
    baseline -- matching the fix already applied in Module 1 for TOI.
    """
    df = df.copy()
    
    df['gp'] = df['gp'].fillna(0) if 'gp' in df.columns else 0.0
    df['gp_eligible'] = df['gp'] >= gp_threshold
    
    # base_position (primary position) is the correct grouping key for every
    # percentile below -- see add_base_position(). Defensive fallback in
    # case this is ever called on a dataframe that skipped it upstream.
    if 'base_position' not in df.columns and 'position' in df.columns:
        df = add_base_position(df)
    
    has_xgf = 'xgf_pct' in df.columns and df['xgf_pct'].notna().any()
    has_hdcf = 'hdcf_pct' in df.columns and df['hdcf_pct'].notna().any()
    
    # Work only within the GP-eligible pool for every ranking computation
    elig = df[df['gp_eligible']].copy()
    
    result_cols = ['sub_shooting_luck', 'sub_underlying_quality', 'sub_deployment_drag',
                   'diamond_score_v2', 'points_per_60_v2']
    
    if elig.empty or 'base_position' not in elig.columns:
        for col in result_cols:
            df[col] = np.nan
        return df
    
    # --- Points per 60 (needed for factor 3) ---
    elig['points_v2'] = elig['g'].fillna(0) + elig['a'].fillna(0)
    elig['points_per_60_v2'] = np.where(
        elig['toi_min'].fillna(0) > 0,
        (elig['points_v2'] / elig['toi_min']) * 60,
        0.0
    )
    
    # --- Factor 1: Shooting Luck (individual iSh% + on-ice PDO) ---
    sh_col = 'shpct' if 'shpct' in elig.columns else None
    has_pdo = 'pdo' in elig.columns and elig['pdo'].notna().any()
    
    luck_parts = []
    if sh_col:
        elig['sh_pct_gap'] = LEAGUE_BASELINE_SH_PCT - pd.to_numeric(elig[sh_col], errors='coerce').fillna(LEAGUE_BASELINE_SH_PCT)
        elig['sh_luck_pctile'] = elig.groupby('base_position')['sh_pct_gap'].rank(pct=True) * 100
        luck_parts.append(elig['sh_luck_pctile'])
    if has_pdo:
        elig['pdo_gap'] = PDO_BASELINE - pd.to_numeric(elig['pdo'], errors='coerce').fillna(PDO_BASELINE)
        elig['pdo_luck_pctile'] = elig.groupby('base_position')['pdo_gap'].rank(pct=True) * 100
        luck_parts.append(elig['pdo_luck_pctile'])
    
    if luck_parts:
        elig['sub_shooting_luck'] = sum(luck_parts) / len(luck_parts)
    else:
        elig['sub_shooting_luck'] = np.nan
    
    # --- Factor 2: Underlying Quality (team-relative CF% / xGF% / HDCF%) ---
    has_cf = 'cf_pct' in df.columns and df['cf_pct'].notna().any()
    if has_xgf or has_hdcf or has_cf:
        blend_parts = []
        if has_cf:
            elig['cf_pct'] = pd.to_numeric(elig['cf_pct'], errors='coerce').fillna(50.0)
            team_avg_cf = elig.groupby('team')['cf_pct'].transform('mean')
            elig['cf_vs_team'] = elig['cf_pct'] - team_avg_cf
            blend_parts.append(elig['cf_vs_team'])
        if has_xgf:
            elig['xgf_pct'] = pd.to_numeric(elig['xgf_pct'], errors='coerce').fillna(50.0)
            team_avg_xgf = elig.groupby('team')['xgf_pct'].transform('mean')
            elig['xgf_vs_team'] = elig['xgf_pct'] - team_avg_xgf
            blend_parts.append(elig['xgf_vs_team'])
        if has_hdcf:
            elig['hdcf_pct'] = pd.to_numeric(elig['hdcf_pct'], errors='coerce').fillna(50.0)
            team_avg_hdcf = elig.groupby('team')['hdcf_pct'].transform('mean')
            elig['hdcf_vs_team'] = elig['hdcf_pct'] - team_avg_hdcf
            blend_parts.append(elig['hdcf_vs_team'])
        elig['quality_blend'] = sum(blend_parts) / len(blend_parts)
        elig['sub_underlying_quality'] = elig.groupby('base_position')['quality_blend'].rank(pct=True) * 100
    else:
        elig['sub_underlying_quality'] = np.nan
    
    # --- Factor 3: Zone Deployment Drag (continuous, not a binary cutoff) ---
    elig['pts60_percentile'] = elig.groupby('base_position')['points_per_60_v2'].rank(pct=True) * 100
    ozs = elig['ozs_pct'].fillna(0.5)
    elig['drag_multiplier'] = (
        (OZS_DRAG_NO_CREDIT - ozs) / (OZS_DRAG_NO_CREDIT - OZS_DRAG_FULL_CREDIT)
    ).clip(lower=0.0, upper=1.0)
    elig['sub_deployment_drag'] = elig['pts60_percentile'] * elig['drag_multiplier']
    
    # --- Weighted composite ---
    # Renormalize weights over whichever factors actually have data, so a
    # missing xGF%/HDCF% source doesn't silently zero out 35% of every score.
    active_weights = {}
    if elig['sub_shooting_luck'].notna().any():
        active_weights['sub_shooting_luck'] = w_shooting
    if elig['sub_underlying_quality'].notna().any():
        active_weights['sub_underlying_quality'] = w_quality
    active_weights['sub_deployment_drag'] = w_deployment  # always computable
    
    weight_sum = sum(active_weights.values()) or 1.0
    elig['diamond_score_v2'] = 0.0
    for col, w in active_weights.items():
        elig['diamond_score_v2'] += elig[col].fillna(0) * (w / weight_sum)
    
    # Merge results back onto the full (unfiltered) dataframe
    merge_cols = ['player'] + result_cols
    elig_results = elig[merge_cols].copy()
    df = df.merge(elig_results, on='player', how='left', suffixes=('', ''))
    
    return df


@st.cache_data
def calculate_category_specialists(df, category, gp_threshold=20):
    """
    Category-specific "hidden gem" finder for H2H Categories leagues, separate
    from the points-focused Diamond Score above. Goals/Assists are already
    the entire focus of that model, so this covers HIT, BLK, SOG, PPP, and
    Plus/Minus -- categories that count just as much in a Yahoo Categories
    league but where a player can be legitimately elite while being
    completely invisible to a points-based ranking.
    
    Unlike the regression-driven Diamond Score, this isn't a "due for
    positive regression" story -- hit/block rates are stable, repeatable
    skills, not luck. The "hidden" part comes from being overlooked in ADP
    despite elite category production, not from being unlucky.
    
    See CATEGORY_SPECIALIST_OPTIONS for which categories use a true per-60
    rate (HIT/BLK/SOG accrue in every situation, so all-situations TOI is a
    clean denominator) versus a raw position-relative percentile (PPP and
    Plus/Minus are tied to a specific game situation that the available TOI
    doesn't isolate -- dividing by it would understate real specialists).
    """
    df = df.copy()
    df['gp'] = df['gp'].fillna(0) if 'gp' in df.columns else 0.0
    df['gp_eligible'] = df['gp'] >= gp_threshold
    
    if 'base_position' not in df.columns and 'position' in df.columns:
        df = add_base_position(df)
    
    elig = df[df['gp_eligible']].copy()
    result_cols = ['category_rate_value', 'category_percentile']
    
    if elig.empty or category not in elig.columns or 'base_position' not in elig.columns:
        for col in result_cols:
            df[col] = np.nan
        return df
    
    method = CATEGORY_SPECIALIST_OPTIONS.get(category, {}).get('method', 'raw')
    cat_values = pd.to_numeric(elig[category], errors='coerce').fillna(0)
    
    if method == 'rate':
        elig['category_rate_value'] = np.where(
            elig['toi_min'].fillna(0) > 0,
            (cat_values / elig['toi_min']) * 60,
            0.0
        )
    else:
        elig['category_rate_value'] = cat_values
    
    elig['category_percentile'] = elig.groupby('base_position')['category_rate_value'].rank(pct=True) * 100
    
    merge_cols = ['player'] + result_cols
    df = df.merge(elig[merge_cols], on='player', how='left')
    
    return df


@st.cache_data
def calculate_positional_vorp(df):
    """
    Positional VORP (Value Over Replacement Player) -- a separate, self-
    contained metric alongside Value Gap, not a replacement for it. Value Gap
    answers "is the market undervaluing this player relative to their ADP."
    VORP answers a different question: "how much better is this player than
    the worst startable option at their position," independent of ADP
    entirely. Verified empirically that these produce meaningfully different
    rankings (positional scarcity at D, for instance, pulls defensemen higher
    in VORP than they rank in Value Gap).

    Three-step process, run SEPARATELY per position (see note below on why):
      1. Draft pool: for each position, rank players at that position by
         their EXISTING total_z (already position-adjusted, already
         verified) and take the top VORP_POSITION_POOL_SIZE for that
         position specifically. Using the full ~900+ player universe to set
         a Z-score baseline pulls the mean/std toward hundreds of players
         nobody would ever roster in a 12-team league -- confirmed this
         shifts category means by 40-180%+ depending on the stat.
      2. Within that position's draft pool, recompute FRESH per-category
         Z-scores and sum into a draft-pool-relative raw value.
      3. Subtract replacement level: within the same position's pool, find
         the value of the last startable player at that position (25th C,
         25th LW, 25th RW, 49th D, per VORP_REPLACEMENT_RANK) and subtract
         it from every player's raw value at that position -- including
         players outside the pool, who by construction score below
         replacement.

    Why per-position pools instead of one shared cross-position pool: an
    earlier version selected one top-132 pool across ALL positions by raw
    total_z, then split it by position afterward. Defensemen naturally post
    lower total_z than forwards (fewer skater categories favor D-specific
    play), so that shared pool silently under-filled D and RW -- confirmed
    empirically at only 42 D against a target replacement rank of 49, and
    24 R against a target of 25, both silently falling back to whichever
    player was last in a too-small pool instead of the actual intended rank.
    Running each position independently, sized with 2x margin over its own
    replacement rank, guarantees the real target rank is always reachable
    (938 total players easily supports it: 323 D, 303 C, 163 L, 149 R exist
    league-wide against pool needs of at most 98).

    Does not touch total_z, value_gap, or any other existing column --
    this is purely additive.
    """
    df = df.copy()
    
    if 'position' not in df.columns or 'total_z' not in df.columns:
        df['base_position'] = np.nan
        df['vorp_raw_value'] = np.nan
        df['positional_vorp'] = np.nan
        return df
    
    # base_position is computed centrally in add_base_position() and should
    # already be present via load_and_merge_data() -- recompute defensively
    # only if it's somehow missing.
    if 'base_position' not in df.columns:
        df = add_base_position(df)
    df['vorp_raw_value'] = 0.0
    df['positional_vorp'] = np.nan
    
    for pos, replacement_rank in VORP_REPLACEMENT_RANK.items():
        pos_all = df[df['base_position'] == pos]
        if pos_all.empty:
            continue
        
        # Step 1: this position's OWN draft pool, sized with margin so the
        # replacement rank is always reachable rather than falling back.
        pool_size = VORP_POSITION_POOL_SIZE.get(pos, replacement_rank * 2)
        pos_pool = pos_all.nlargest(min(pool_size, len(pos_all)), 'total_z')
        
        # Step 2: fresh per-category Z-scores using ONLY this position's pool
        pool_raw_value = pd.Series(0.0, index=pos_pool.index)
        cat_stats = {}
        for cat in SKATER_CATEGORIES:
            if cat not in pos_pool.columns:
                continue
            vals = pd.to_numeric(pos_pool[cat], errors='coerce').dropna()
            if len(vals) < 2:
                continue
            mean, std = vals.mean(), vals.std()
            cat_stats[cat] = (mean, std)
            if std > 0:
                z = (pd.to_numeric(pos_pool[cat], errors='coerce') - mean) / std
                pool_raw_value += z.fillna(0.0)
        
        # Apply the SAME pool-derived mean/std to EVERY player at this
        # position (not just the pool), so players outside it get a
        # coherent, typically-negative value on the same scale.
        pos_all_raw_value = pd.Series(0.0, index=pos_all.index)
        for cat, (mean, std) in cat_stats.items():
            if std <= 0:
                continue
            z = (pd.to_numeric(pos_all[cat], errors='coerce') - mean) / std
            pos_all_raw_value += z.fillna(0.0)
        df.loc[pos_all.index, 'vorp_raw_value'] = pos_all_raw_value
        
        # Step 3: replacement level, found within this position's pool at
        # its specified rank, subtracted from every player at that position.
        pool_raw_value_sorted = pool_raw_value.sort_values(ascending=False)
        rank_idx = min(replacement_rank, len(pool_raw_value_sorted)) - 1
        replacement_value = pool_raw_value_sorted.iloc[rank_idx]
        df.loc[pos_all.index, 'positional_vorp'] = df.loc[pos_all.index, 'vorp_raw_value'] - replacement_value
    
    return df


# ==========================================
# 5. SEARCH & FILTER UTILITIES
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
# 6. LIVE DRAFT STATE MANAGEMENT
# ==========================================
STRENGTH_Z_THRESHOLD = 0.5    # avg per-category Z at/above this = roster strength
WEAKNESS_Z_THRESHOLD = -0.5   # avg per-category Z at/below this = roster weakness

def initialize_draft_state():
    """One-time setup of session state for the live draft tracker. Must run
    before anything else in main() touches these keys."""
    if 'drafted_players' not in st.session_state:
        st.session_state.drafted_players = set()      # every player off the board (mine + other teams')
    if 'my_team' not in st.session_state:
        st.session_state.my_team = []                  # ordered list of player dicts I've drafted
    if 'pending_draft_player' not in st.session_state:
        st.session_state.pending_draft_player = None    # player name awaiting a Draft/Release decision

@st.dialog("Draft Decision")
def show_draft_dialog(player_name, source_df):
    """
    Modal shown when a player is clicked on the draft board.
    - Draft: adds the player to My Team and removes them from the board.
    - Release: removes them from the board without adding to My Team (i.e.
      taken by another team in the draft) -- either choice frees up the next
      best available player for the Top Board Value Target.
    """
    match = source_df[source_df['player'] == player_name]
    if match.empty:
        # Player already left the pool (e.g. a stale click) -- just close.
        st.session_state.pending_draft_player = None
        st.rerun()
        return
    
    row = match.iloc[0]
    st.markdown(f"### {player_name}")
    detail_bits = [str(row['position'])] if 'position' in row and pd.notna(row['position']) else []
    if 'adp' in row and pd.notna(row['adp']):
        detail_bits.append(f"ADP {row['adp']:.1f}")
    if 'value_gap' in row and pd.notna(row['value_gap']):
        detail_bits.append(f"Value Gap {row['value_gap']:+.2f}")
    if detail_bits:
        st.caption(" • ".join(detail_bits))
    st.caption("Double-check this is the player you meant before confirming.")
    
    st.markdown("---")
    col_draft, col_release = st.columns(2)
    with col_draft:
        st.markdown("**Draft**")
        st.caption("Adds this player to My Team.")
        if st.button("✅ Draft to My Team", use_container_width=True, type="primary"):
            st.session_state.my_team.append(row.to_dict())
            st.session_state.drafted_players.add(player_name)
            st.session_state.pending_draft_player = None
            st.session_state.draft_player_selectbox = ""
            st.rerun()
    with col_release:
        st.markdown("**Release**")
        st.caption("Off the board — taken by another team.")
        if st.button("❌ Release", use_container_width=True):
            st.session_state.drafted_players.add(player_name)
            st.session_state.pending_draft_player = None
            st.session_state.draft_player_selectbox = ""
            st.rerun()
    
    if st.button("Cancel", use_container_width=True):
        st.session_state.pending_draft_player = None
        st.session_state.draft_player_selectbox = ""
        st.rerun()

# ==========================================
# 7. MAIN UI
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
        - finddiamonds_data.csv (required for Module 1)
        """)
        return
    
    initialize_draft_state()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    mode = st.sidebar.radio("Analysis Mode:", ["Value Hunting", "Talent Ranking"])
    
    # Initialize weights (can be customized later)
    weights = {cat: 1.0 for cat in SKATER_CATEGORIES}
    
    # Compute value gap matrix
    processed = compute_value_gap_matrix(df, weights, mode)
    
    # Positional VORP: a separate, additive metric alongside Value Gap --
    # does not modify total_z, value_gap, or anything else computed above.
    processed = calculate_positional_vorp(processed)
    
    # Roster target: goalie slots only count if goalie data actually exists
    # in the pool. None of the currently-loaded CSVs contain goalie stats
    # (they're all skater exports), so by default this drops 2 slots and
    # says so explicitly rather than silently leaving "complete" unreachable.
    has_goalie_pool = (processed['position'] == 'G').any()
    effective_roster_size = TOTAL_ROSTER_SIZE - (0 if has_goalie_pool else ROSTER_LAYOUT.get('G', 0))
    
    # Position filter
    positions = sorted(processed['position'].dropna().unique())
    selected_positions = st.sidebar.multiselect("Filter by Position:", positions, default=positions)
    filtered_data = processed[processed['position'].isin(selected_positions)]
    
    # Search box
    search_term = st.sidebar.text_input("🔍 Search Player:", "")
    if search_term:
        filtered_data = safe_search_players(filtered_data, search_term)
    
    # ---- My Team / Draft Tracker (sidebar, left side of screen) ----
    st.sidebar.markdown("---")
    st.sidebar.header("🏒 My Team")
    
    drafted_count = len(st.session_state.my_team)
    st.sidebar.progress(min(drafted_count / effective_roster_size, 1.0))
    progress_caption = f"{drafted_count} / {effective_roster_size} drafted"
    if not has_goalie_pool:
        progress_caption += "  (goalie slots excluded — no goalie data loaded)"
    st.sidebar.caption(progress_caption)
    
    if st.session_state.my_team:
        team_preview = pd.DataFrame(st.session_state.my_team)[['player', 'position']].copy()
        team_preview.index = range(1, len(team_preview) + 1)
        st.sidebar.dataframe(team_preview, use_container_width=True, height=min(38 * len(team_preview) + 38, 350))
        
        remove_name = st.sidebar.selectbox(
            "Wrong pick? Remove it:",
            options=[""] + [p['player'] for p in st.session_state.my_team],
            format_func=lambda x: "— Select a player —" if x == "" else x,
            key="remove_pick_selectbox",
        )
        if remove_name and st.sidebar.button("🗑️ Remove from My Team", use_container_width=True):
            st.session_state.my_team = [p for p in st.session_state.my_team if p['player'] != remove_name]
            st.session_state.drafted_players.discard(remove_name)
            st.session_state.remove_pick_selectbox = ""
            st.rerun()
    else:
        st.sidebar.caption("No players drafted yet — use the search box on Optimization Matrix to start.")
    
    if st.sidebar.button("🔄 Reset Entire Draft", use_container_width=True):
        st.session_state.drafted_players = set()
        st.session_state.my_team = []
        st.session_state.pending_draft_player = None
        st.rerun()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Optimization Matrix", "Draft Day Playbook", "Regression Alerts",
        "🔬 Diamond Score v2", "🏆 My Team"
    ])
    
    with tab1:
        st.subheader(f"📊 Current View: {mode}")
        
        # Only players nobody has claimed yet -- drafting or releasing a
        # player removes them from this pool, so the Top Board Value Target
        # and the board below always reflect who's actually still available.
        available_data = filtered_data[~filtered_data['player'].isin(st.session_state.drafted_players)]
        
        # Active Recommendation Headboard
        st.markdown("### 🎯 Top Board Value Target")
        active_targets = available_data[
            available_data['action_tier'] == "🎯 Target Immediately"
        ] if 'action_tier' in available_data.columns else available_data
        
        if not active_targets.empty:
            target_skater = active_targets.iloc[0]
            col_t_name, col_t_vg, col_t_tier, col_t_reg = st.columns(4)
            col_t_name.metric("🎯 Draft Priority Target", target_skater['player'])
            col_t_vg.metric("📈 Value Gap", f"+{target_skater['value_gap']:.2f}")
            col_t_tier.metric("📋 Action Tier", target_skater['action_tier'])
            col_t_reg.metric("🔄 Regression Profile", target_skater['pdo_alert'])
            if st.button(f"⚡ Quick Draft/Release {target_skater['player']}", key="quick_draft_top_target"):
                st.session_state.pending_draft_player = target_skater['player']
                st.rerun()
        elif not available_data.empty and 'value_gap' in available_data.columns:
            # Nobody left hits the top tier specifically, but players remain --
            # fall back to the single best Value Gap so this metric keeps
            # pointing at "who should I take next" for the whole draft, not
            # just the first round or two.
            fallback = available_data.sort_values('value_gap', ascending=False).iloc[0]
            st.caption("No remaining player hits '🎯 Target Immediately' — showing best available by Value Gap.")
            col_t_name, col_t_vg, col_t_tier, col_t_reg = st.columns(4)
            col_t_name.metric("🎯 Draft Priority Target", fallback['player'])
            col_t_vg.metric("📈 Value Gap", f"{fallback['value_gap']:+.2f}")
            col_t_tier.metric("📋 Action Tier", fallback.get('action_tier', '—'))
            col_t_reg.metric("🔄 Regression Profile", fallback.get('pdo_alert', '—'))
            if st.button(f"⚡ Quick Draft/Release {fallback['player']}", key="quick_draft_fallback_target"):
                st.session_state.pending_draft_player = fallback['player']
                st.rerun()
        else:
            st.success("🏁 Every player matching your current filters has been drafted or released.")
        
        st.markdown("---")
        st.caption(
            "ℹ️ **Value Gap** vs. **Positional VORP**: Value Gap rewards production the market is "
            "underpricing (ADP-aware). Positional VORP ignores ADP entirely and measures pure talent-pool "
            "scarcity — how far above the last startable player at that position (25th C/LW/RW, 49th D, "
            "within a draft pool of the top 132 skaters) this player sits. They're independent signals; "
            "sort by whichever column you want to lead with by clicking its header. "
            "**A blank ADP** means the player isn't drafted in typical leagues — no market price exists, "
            "so their Value Gap uses a neutral default and their Action Tier is a rougher estimate; lean "
            "on VORP and Diamond Score for those players."
        )
        
        # Display results table
        display_cols = [
            'player', 'position', 'g', 'a', 'ppp', 'plus_minus', 'sog', 'hit', 'blk',
            'adp', 'ozs_pct', 'pdo', 'total_z', 'value_gap', 'positional_vorp', 'action_tier', 'pdo_alert'
        ]
        display_cols = [col for col in display_cols if col in available_data.columns]
        board_df = available_data[display_cols].reset_index(drop=True)
        
        # NOTE: player selection is driven by name (selectbox), not by
        # clicking a table row. st.dataframe's row-click selection is reported
        # by positional index against the data passed in, but the table also
        # lets a user sort by clicking any column header -- and Streamlit has
        # a confirmed, team-acknowledged bug (streamlit/streamlit#11345) where
        # sorting doesn't reset the selection but leaves the OLD index in
        # place, silently pointing at a different row than the one visually
        # highlighted. On a wide sortable stats table, sorting before picking
        # is a completely natural thing to do, so that failure mode is not an
        # edge case here. A name-driven selectbox can't misfire this way
        # regardless of how the table is sorted.
        st.caption("💡 Hit ⚡ Quick Draft above for the recommended target, or search for any other player below.")
        player_options = [""] + board_df['player'].tolist()
        selected_name = st.selectbox(
            "Select a player:",
            options=player_options,
            format_func=lambda x: "— Type to search a player —" if x == "" else x,
            key="draft_player_selectbox",
            label_visibility="collapsed",
        )
        
        if selected_name and st.session_state.pending_draft_player is None:
            st.session_state.pending_draft_player = selected_name
            st.rerun()
        
        if st.session_state.pending_draft_player is not None:
            show_draft_dialog(st.session_state.pending_draft_player, available_data)
        
        st.dataframe(
            board_df,
            column_config={
                "ozs_pct": st.column_config.ProgressColumn("OZS% Intent", min_value=0.0, max_value=1.0),
                "value_gap": st.column_config.NumberColumn("Value Gap", format="%.2f"),
                "positional_vorp": st.column_config.NumberColumn("Positional VORP", format="%.2f"),
                "total_z": st.column_config.NumberColumn("Total Z-Score", format="%.2f"),
                "adp": st.column_config.NumberColumn("ADP", format="%.1f"),
                "pdo": st.column_config.NumberColumn("PDO", format="%.3f"),
            },
            use_container_width=True,
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
            **The Elite Asset Case (real data):**
            * Connor McDavid's weighted Z-Score across all 7 categories is **+17.87** — 
              summing several strong-to-elite categories adds up fast.
            * His ADP of **1.5** makes the draft penalty microscopic: (1.5/100)^1.5 ≈ **0.002**.
            * Because his ADP is ≤ 5, the engine also adds the **+3.0 elite-talent boost**.
            * Value Gap = 17.87 − 0.002 + 3.0 = **+20.87**. The engine ranks him at the top 
              because production this far above position-average completely swamps the 
              (already tiny) cost of a top-5 pick.
            """)
        
        with col_trap:
            st.markdown("""
            **The Value Trap Case (real data):**
            * Tommy Novak has a modest weighted Z-Score of **+0.74** — decent, not special.
            * He's outside the ADP data entirely, so the engine defaults him to **ADP 100** 
              (its fallback for "unranked / very late").
            * Penalty = (100/100)^1.5 = **1.00** — a full point taken off for that price tag.
            * Value Gap = 0.74 − 1.00 = **−0.26**. The engine flags a minor overpay: modest 
              production isn't bad, but there's no ADP discount left to justify reaching for it.
            """)
        
        st.caption(
            "Both figures above are pulled live from the current formula and dataset, not "
            "hardcoded illustrations — they'll shift if the underlying CSVs change."
        )
        
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
            **PDO** is the ultimate proxy for shooting and goaltending "luck" (ideal baseline = `1.000`). 
            It flags whether a player's current stat line is a fair read on their talent, or a mirage:
            * **📈 Undervalued (PDO < 0.975):** Suppressed by low team shooting or save percentages — 
              their box score is *underselling* them. Target them; the numbers should catch up to the talent.
            * **📉 Overvalued (PDO > 1.025):** A hot streak inflated by unsustainable luck — their box score 
              is *overselling* them. Be cautious drafting them at a price that assumes it continues.
            * **🔄 Fairly Valued:** PDO in the normal range — current production is a reliable read on talent.
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
        st.markdown("Players whose PDO says their current stats are misleading — 📈 Undervalued (due to bounce back) or 📉 Overvalued (running hot on luck):")
        
        regression_data = filtered_data[['player', 'position', 'pdo', 'pdo_alert', 'ozs_pct', 'total_z']].copy()
        regression_data = regression_data[regression_data['pdo_alert'] != "🔄 Fairly Valued"]
        
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
    
    with tab4:
        st.subheader("🔬 Diamond Score v2 — Weighted Composite Model")
        st.markdown("""
        Finds breakout candidates by blending **three independent regression/context signals** 
        into one weighted 0-100 score — every eligible player gets ranked, rather than 
        requiring all signals to fire at once (which tends to return zero results).
        """)
        
        gp_col_available = 'gp' in filtered_data.columns and filtered_data['gp'].notna().any() and filtered_data['gp'].sum() > 0
        
        with st.expander("🔍 Data diagnostics (open if GP/xGF%/HDCF% seem missing)"):
            st.write(f"**All columns currently loaded:** {sorted(filtered_data.columns.tolist())}")
            for col_key, label in [('gp', 'GP'), ('xgf_pct', 'xGF%'), ('hdcf_pct', 'HDCF%'), ('toi_min', 'TOI'), ('cf_pct', 'CF%')]:
                if col_key in filtered_data.columns:
                    non_null = filtered_data[col_key].notna().sum()
                    sample = filtered_data[col_key].dropna().head(3).tolist()
                    st.write(f"- `{label}` → found as column `{col_key}`, {non_null}/{len(filtered_data)} non-null, sample values: {sample}")
                else:
                    st.write(f"- `{label}` → **not present** in the merged dataframe")
            st.caption(
                "If a column shows as missing here but you've confirmed it's in finddiamonds_data.csv, "
                "the running app is likely on stale code or a stale cache. Stop the Streamlit server, "
                "restart with `streamlit run app_module1.py`, and if the banner still appears, open the "
                "hamburger menu → 'Clear cache', then rerun."
            )
        
        if not gp_col_available:
            st.warning("""
            ⚠️ **Games Played (GP) data not found or empty.** Diamond Score v2 requires the `GP` 
            column from `finddiamonds_data.csv`. Check the diagnostics panel above — if `GP` shows 
            as "not present" there despite being in your CSV, restart the Streamlit server (stopping 
            and re-running `streamlit run app_module1.py` picks up code/data changes that a browser 
            refresh alone won't).
            """)
        else:
            has_xgf_data = 'xgf_pct' in filtered_data.columns and filtered_data['xgf_pct'].notna().any()
            has_hdcf_data = 'hdcf_pct' in filtered_data.columns and filtered_data['hdcf_pct'].notna().any()
            if not (has_xgf_data or has_hdcf_data):
                st.info("""
                ℹ️ xGF%/HDCF% not found in your data — the **Underlying Quality** factor 
                will be skipped and its weight redistributed across the other two factors.
                """)
            
            st.markdown("#### ⚙️ Settings")
            col_gp_slider, col_w1, col_w2, col_w3 = st.columns(4)
            
            with col_gp_slider:
                gp_threshold = st.slider(
                    "Minimum Games Played", min_value=1, max_value=82, value=20, step=1,
                    help="Players below this GP are excluded before any percentile is computed — "
                         "prevents call-ups/injury-shortened seasons from skewing the rankings."
                )
            with col_w1:
                w_shooting = st.slider("Weight: Shooting Luck", 0, 100, 35, step=5) / 100.0
            with col_w2:
                w_quality = st.slider("Weight: Underlying Quality", 0, 100, 35, step=5) / 100.0
            with col_w3:
                w_deployment = st.slider("Weight: Deployment Drag", 0, 100, 30, step=5) / 100.0
            
            diamond_v2 = calculate_diamond_scores_v2(
                filtered_data, gp_threshold=gp_threshold,
                w_shooting=w_shooting, w_quality=w_quality, w_deployment=w_deployment
            )
            
            eligible_count_v2 = diamond_v2['gp_eligible'].sum() if 'gp_eligible' in diamond_v2.columns else 0
            ranked = diamond_v2[diamond_v2['diamond_score_v2'].notna()].sort_values('diamond_score_v2', ascending=False)
            
            st.markdown("---")
            col_v2_count, col_v2_top = st.columns(2)
            with col_v2_count:
                st.metric("Players Meeting GP Threshold", int(eligible_count_v2), f"of {len(diamond_v2)} filtered players")
            with col_v2_top:
                if not ranked.empty:
                    top_player = ranked.iloc[0]
                    st.metric("🏆 Top Diamond Score", top_player['player'], f"{top_player['diamond_score_v2']:.1f} / 100")
                else:
                    st.metric("🏆 Top Diamond Score", "N/A")
            
            st.markdown("---")
            
            if not ranked.empty:
                show_count = st.slider(
                    "How many players to show:",
                    min_value=10, max_value=max(min(len(ranked), 200), 10),
                    value=min(20, len(ranked)), step=10,
                    key="diamond_v2_show_count",
                    help=f"Your eligible pool has {len(ranked)} ranked players at the current GP threshold."
                )
                shown_diamonds = ranked.head(show_count).copy()
                
                # Raw season totals for context alongside the composite score --
                # 'g'/'a' already flow through untouched from nst_data, so this
                # is pure display, no change to the scoring engine itself.
                shown_diamonds['g'] = pd.to_numeric(shown_diamonds['g'], errors='coerce').fillna(0)
                shown_diamonds['a'] = pd.to_numeric(shown_diamonds['a'], errors='coerce').fillna(0)
                shown_diamonds['points_total'] = shown_diamonds['g'] + shown_diamonds['a']
                
                display_cols_v2 = ['player', 'position', 'gp', 'g', 'a', 'points_total', 'shpct',
                                    'xgf_pct', 'hdcf_pct', 'ozs_pct', 'points_per_60_v2', 'sub_shooting_luck',
                                    'sub_underlying_quality', 'sub_deployment_drag', 'diamond_score_v2']
                display_cols_v2 = [c for c in display_cols_v2 if c in shown_diamonds.columns]
                
                st.markdown(f"### 🏆 Top {len(shown_diamonds)} Diamonds by Weighted Score")
                st.data_editor(
                    shown_diamonds[display_cols_v2],
                    column_config={
                        "player": st.column_config.TextColumn("Player", width="medium"),
                        "position": st.column_config.TextColumn("Pos", width="small"),
                        "gp": st.column_config.NumberColumn("GP", format="%.0f", width="small"),
                        "g": st.column_config.NumberColumn("G", format="%.0f", width="small"),
                        "a": st.column_config.NumberColumn("A", format="%.0f", width="small"),
                        "points_total": st.column_config.NumberColumn("PTS", format="%.0f", width="small"),
                        "shpct": st.column_config.NumberColumn("iSh%", format="%.1f", width="small"),
                        "xgf_pct": st.column_config.NumberColumn("xGF%", format="%.1f", width="small"),
                        "hdcf_pct": st.column_config.NumberColumn("HDCF%", format="%.1f", width="small"),
                        "ozs_pct": st.column_config.ProgressColumn("OZS%", min_value=0.0, max_value=1.0, width="small"),
                        "points_per_60_v2": st.column_config.NumberColumn("Pts/60", format="%.2f", width="small"),
                        "sub_shooting_luck": st.column_config.NumberColumn("Luck %ile", format="%.0f", width="small"),
                        "sub_underlying_quality": st.column_config.NumberColumn("Quality %ile", format="%.0f", width="small"),
                        "sub_deployment_drag": st.column_config.NumberColumn("Drag %ile", format="%.0f", width="small"),
                        "diamond_score_v2": st.column_config.NumberColumn("Diamond Score", format="%.1f", width="medium"),
                    },
                    use_container_width=True,
                    disabled=True
                )
                
                st.markdown("---")
                st.markdown("### 📊 How to Read the Three Factors")
                st.markdown(f"""
                **Shooting Luck** — blends individual iSh% below the {LEAGUE_BASELINE_SH_PCT:.0f}% league 
                baseline with on-ice PDO below {PDO_BASELINE:.3f}. Higher percentile = colder than expected 
                on both fronts, i.e. due for positive regression.
                
                **Underlying Quality** — blends CF%, xGF%, and HDCF%, each relative to their *own team's* 
                average, not the league. Higher percentile = generating better shot volume and quality 
                than their teammates, regardless of whether the team overall is strong or weak.
                
                **Deployment Drag** — points-per-60 percentile, scaled by a credit multiplier: full credit 
                at OZS% ≤ {OZS_DRAG_FULL_CREDIT*100:.0f}%, zero credit at OZS% ≥ {OZS_DRAG_NO_CREDIT*100:.0f}%, 
                sliding in between. Higher percentile = scoring well *despite* tough zone deployment; 
                players deployed offensively get little or no credit here since deployment isn't holding 
                them back.
                
                **Deferred:** Primary/Secondary assist ratio isn't included — `nst_data.csv` only has 
                combined assists, no split. Add that column later to extend this model.
                """)
            else:
                st.info(f"ℹ️ No players meet the {gp_threshold}-game minimum with current filters. Try lowering the GP threshold or removing position filters.")
            
            st.markdown("---")
            st.markdown("### 🏒 Category Specialists")
            st.markdown("""
            The Diamond Score above is points-focused (Goals + Assists). In an H2H Categories league, a 
            player who's quietly elite in **Hits**, **Blocks**, **Shots on Goal**, **Powerplay Points**, 
            or **Plus/Minus** — while completely off the radar in points and ADP — can be just as valuable 
            as a points diamond. This isn't a regression/luck story like the Diamond Score above; hit and 
            block rates are stable, repeatable skills. The "hidden" part is being overlooked in ADP despite 
            elite category production, not being unlucky.
            """)
            
            col_cat_select, col_cat_threshold = st.columns(2)
            with col_cat_select:
                category_choice = st.selectbox(
                    "Category:",
                    options=list(CATEGORY_SPECIALIST_OPTIONS.keys()),
                    format_func=lambda k: CATEGORY_SPECIALIST_OPTIONS[k]['label'],
                    key="category_specialist_select",
                )
            with col_cat_threshold:
                specialist_threshold = st.slider(
                    "Minimum percentile to qualify:", min_value=50, max_value=99, value=80, step=5,
                    key="category_specialist_threshold",
                )
            
            is_rate_based = CATEGORY_SPECIALIST_OPTIONS[category_choice]['method'] == 'rate'
            cat_label = CATEGORY_SPECIALIST_OPTIONS[category_choice]['label']
            st.caption(
                f"Ranked by **{cat_label} per 60 minutes**, position-adjusted."
                if is_rate_based else
                f"Ranked by **raw season {cat_label}**, position-adjusted — see note below on why this "
                f"one isn't a per-60 rate."
            )
            
            specialists_df = calculate_category_specialists(filtered_data, category_choice, gp_threshold=gp_threshold)
            specialist_pool = specialists_df[
                specialists_df['category_percentile'].notna() &
                (specialists_df['category_percentile'] >= specialist_threshold)
            ].sort_values('category_percentile', ascending=False)
            
            if not specialist_pool.empty:
                display_specialist_cols = ['player', 'position', 'gp']
                if is_rate_based:
                    display_specialist_cols += [category_choice, 'category_rate_value']
                else:
                    display_specialist_cols += ['category_rate_value']
                display_specialist_cols += ['category_percentile', 'adp']
                display_specialist_cols = [c for c in display_specialist_cols if c in specialist_pool.columns]
                
                rate_col_label = f"{cat_label}/60" if is_rate_based else f"Season {cat_label}"
                st.dataframe(
                    specialist_pool[display_specialist_cols].head(25),
                    column_config={
                        "player": st.column_config.TextColumn("Player", width="medium"),
                        "position": st.column_config.TextColumn("Pos", width="small"),
                        "gp": st.column_config.NumberColumn("GP", format="%.0f", width="small"),
                        category_choice: st.column_config.NumberColumn(f"{cat_label} (Season)", format="%.0f", width="small"),
                        "category_rate_value": st.column_config.NumberColumn(
                            rate_col_label, format="%.2f" if is_rate_based else "%.0f", width="small"),
                        "category_percentile": st.column_config.NumberColumn("Percentile", format="%.0f", width="small"),
                        "adp": st.column_config.NumberColumn("ADP", format="%.1f", width="small"),
                    },
                    use_container_width=True,
                )
                st.caption(
                    f"💡 Look for a high ADP (late-round or blank/undrafted) here — that's the actual "
                    f"'hidden' part. Elite {cat_label} production at an ADP past pick 150 is a near-free "
                    f"source of that category all season."
                )
            else:
                st.info(f"No players meet the {specialist_threshold}th percentile threshold for {cat_label} with current filters.")
            
            if not is_rate_based:
                st.caption(
                    f"ℹ️ {cat_label} isn't shown as a per-60 rate because it only accrues in a specific "
                    f"game situation (power play for PPP; even-strength non-shorthanded for Plus/Minus), "
                    f"while the only TOI available here covers all situations combined — dividing by it "
                    f"would understate players who get real opportunity in that specific situation."
                )
    
    with tab5:
        st.subheader("🏆 My Team — Strengths & Weaknesses")
        
        drafted_count = len(st.session_state.my_team)
        
        if drafted_count == 0:
            st.info("ℹ️ You haven't drafted anyone yet. Go to **Optimization Matrix** and click a player to get started.")
        else:
            team_df = pd.DataFrame(st.session_state.my_team)
            is_complete = drafted_count >= effective_roster_size
            
            if is_complete:
                st.success(f"🎉 Roster complete! ({drafted_count}/{effective_roster_size} picks made)")
            else:
                st.info(f"📋 Draft in progress: {drafted_count}/{effective_roster_size} picks made. "
                        f"Here's your profile so far — it'll keep updating as you draft.")
            
            st.markdown("---")
            
            # Position breakdown
            st.markdown("### 📍 Roster Composition")
            pos_counts = team_df['position'].value_counts().rename_axis('Position').reset_index(name='Drafted')
            st.dataframe(pos_counts, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Category strengths/weaknesses via average per-category Z-score
            st.markdown("### 📊 Category Profile")
            st.caption(
                f"Average position-adjusted Z-score across your drafted skaters. "
                f"Above +{STRENGTH_Z_THRESHOLD} = strength, below {WEAKNESS_Z_THRESHOLD} = weakness."
            )
            
            category_rows = []
            for cat in SKATER_CATEGORIES:
                z_col = f'z_{cat}'
                if z_col in team_df.columns:
                    avg_z = pd.to_numeric(team_df[z_col], errors='coerce').mean()
                    if pd.notna(avg_z):
                        category_rows.append({'category': CATEGORY_LABELS.get(cat, cat), 'avg_z': avg_z})
            
            if category_rows:
                cat_df = pd.DataFrame(category_rows).sort_values('avg_z', ascending=True)
                cat_df['classification'] = cat_df['avg_z'].apply(
                    lambda z: 'Strength' if z > STRENGTH_Z_THRESHOLD
                    else ('Weakness' if z < WEAKNESS_Z_THRESHOLD else 'Balanced')
                )
                
                fig = px.bar(
                    cat_df, x='avg_z', y='category', orientation='h',
                    color='classification',
                    color_discrete_map={'Strength': '#2ecc71', 'Weakness': '#e74c3c', 'Balanced': '#95a5a6'},
                    labels={'avg_z': 'Avg Position-Adjusted Z-Score', 'category': ''},
                )
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                strengths = cat_df[cat_df['classification'] == 'Strength'].sort_values('avg_z', ascending=False)
                weaknesses = cat_df[cat_df['classification'] == 'Weakness'].sort_values('avg_z')
                
                col_s, col_w = st.columns(2)
                with col_s:
                    st.markdown("**💪 Strengths**")
                    if not strengths.empty:
                        for _, r in strengths.iterrows():
                            st.write(f"- {r['category']} ({r['avg_z']:+.2f})")
                    else:
                        st.caption("No standout strengths yet.")
                with col_w:
                    st.markdown("**⚠️ Weaknesses**")
                    if not weaknesses.empty:
                        for _, r in weaknesses.iterrows():
                            st.write(f"- {r['category']} ({r['avg_z']:+.2f})")
                    else:
                        st.caption("No glaring weaknesses yet.")
                
                st.markdown("---")
                st.markdown("### 🎯 Draft Strategy Takeaway")
                if not weaknesses.empty:
                    weak_names = ", ".join(weaknesses['category'].tolist())
                    verb = "Target" if not is_complete else "Stream free agents or trade for"
                    st.markdown(f"**{verb} {weak_names}** — these categories are running below a typical roster.")
                if not strengths.empty:
                    strong_names = ", ".join(strengths['category'].tolist())
                    st.markdown(f"You can afford to **deal from your {strong_names} depth** to shore up weaker categories.")
                if strengths.empty and weaknesses.empty:
                    st.markdown("Your category profile is balanced across the board so far — no glaring gaps.")
            else:
                st.info("Category Z-score data not available for your drafted players yet.")
            
            if not has_goalie_pool and ROSTER_LAYOUT.get('G', 0) > 0:
                st.caption(
                    "ℹ️ Goaltending categories (Wins, GA, Saves, Shutouts) aren't covered here — "
                    "no goalie data source is currently loaded."
                )
            
            st.markdown("---")
            st.markdown("### 📋 Full Roster")
            roster_display_cols = ['player', 'position', 'g', 'a', 'ppp', 'plus_minus', 'sog', 'hit', 'blk', 'adp', 'value_gap']
            roster_display_cols = [c for c in roster_display_cols if c in team_df.columns]
            st.dataframe(team_df[roster_display_cols], use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()