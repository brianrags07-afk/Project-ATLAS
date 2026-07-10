
from copy import deepcopy


TEAM_CARD_TEMPLATE = {

    # ============================================================
    # METADATA
    # ============================================================

    "metadata": {},
    "career_summary": {},
    "timeline": [],

    # ============================================================
    # TEAM HISTORY
    # ============================================================

    "roster_history": {},
    "manager_history": {},
    "lineup_history": {},
    "rotation_history": {},
    "bullpen_history": {},

    # ============================================================
    # OFFENSE
    # ============================================================

    "offense_profile": {},
    "contact_profile": {},
    "discipline_profile": {},
    "baserunning_profile": {},

    # ============================================================
    # DEFENSE
    # ============================================================

    "defense_profile": {},

    # ============================================================
    # PITCHING
    # ============================================================

    "starting_pitching_profile": {},
    "bullpen_profile": {},
    "pitch_type_profile": {},

    # ============================================================
    # CONTEXT
    # ============================================================

    "home_road_profile": {},
    "park_profile": {},
    "weather_profile": {},
    "travel_profile": {},
    "rest_profile": {},
    "series_profile": {},
    "day_night_profile": {},
    "inning_profile": {},
    "one_run_profile": {},
    "blowout_profile": {},
    "late_game_profile": {},

    # ============================================================
    # MATCHUPS
    # ============================================================

    "pitcher_archetype_profile": {},
    "offensive_archetype_profile": {},
    "defensive_archetype_profile": {},
    "bullpen_archetype_profile": {},
    "opponent_profile": {},
    "lineup_profile": {},
    "umpire_profile": {},

    # ============================================================
    # BETTING
    # ============================================================

    "moneyline_profile": {},
    "runline_profile": {},
    "totals_profile": {},
    "yrfi_nrfi_profile": {},
    "first5_profile": {},
    "player_prop_environment": {},

    # ============================================================
    # INTELLIGENCE
    # ============================================================

    "identity": {},
    "identity_history": [],
    "interaction_history": [],
    "learning": {},
    "confidence": {},
    "volatility": {},

    # ============================================================
    # PROVENANCE
    # ============================================================

    "provenance": {}
}


def blank_team_card():
    return deepcopy(TEAM_CARD_TEMPLATE)
