
from atlas.questions.question_schema import make_question


def core_team_questions():
    questions = []

    questions.append(make_question(
        question_id="Q_TEAM_ML_0001",
        subject_type="team",
        subject_scope="team_overall",
        target="moneyline",
        outcome="team_win",
        question_text="WHEN does this team win?",
        contexts=[
            "home_away",
            "opponent",
            "starter_identity",
            "bullpen_state",
            "park",
            "weather",
            "rest",
            "travel",
            "series_game"
        ],
        priority="tier_1"
    ))

    questions.append(make_question(
        question_id="Q_TEAM_TOTALS_0001",
        subject_type="team",
        subject_scope="team_offense",
        target="totals",
        outcome="team_runs_scored",
        question_text="WHEN does this team score runs?",
        contexts=[
            "home_away",
            "opponent_pitching_identity",
            "pitch_type_profile",
            "park",
            "weather",
            "rest",
            "travel",
            "series_game"
        ],
        priority="tier_1"
    ))

    questions.append(make_question(
        question_id="Q_TEAM_TOTALS_0002",
        subject_type="team",
        subject_scope="team_run_environment",
        target="totals",
        outcome="game_total_runs",
        question_text="WHEN do this team's games create overs or unders?",
        contexts=[
            "starter_matchup",
            "bullpen_state",
            "park",
            "weather",
            "offense_identity",
            "opponent_identity",
            "umpire"
        ],
        priority="tier_1"
    ))

    questions.append(make_question(
        question_id="Q_TEAM_RL_0001",
        subject_type="team",
        subject_scope="team_margin",
        target="runline",
        outcome="win_or_lose_by_margin",
        question_text="WHEN does this team win comfortably or lose badly?",
        contexts=[
            "home_away",
            "starter_advantage",
            "bullpen_advantage",
            "offense_identity",
            "opponent_identity",
            "park",
            "weather"
        ],
        priority="tier_1"
    ))

    questions.append(make_question(
        question_id="Q_TEAM_BULLPEN_0001",
        subject_type="team",
        subject_scope="bullpen",
        target="moneyline_totals",
        outcome="late_game_run_prevention",
        question_text="WHEN is this bullpen reliable or vulnerable?",
        contexts=[
            "reliever_availability",
            "bullpen_rest",
            "recent_usage",
            "opponent_offense_identity",
            "park",
            "weather",
            "leverage",
            "manager_usage"
        ],
        priority="tier_1"
    ))

    return questions


def core_pitcher_questions():
    questions = []

    questions.append(make_question(
        question_id="Q_PITCHER_ML_0001",
        subject_type="pitcher",
        subject_scope="starter",
        target="moneyline",
        outcome="starter_run_prevention",
        question_text="WHEN does this pitcher suppress runs?",
        contexts=[
            "opponent_offense_identity",
            "pitch_type_matchup",
            "home_away",
            "park",
            "weather",
            "rest",
            "umpire"
        ],
        priority="tier_1"
    ))

    questions.append(make_question(
        question_id="Q_PITCHER_YRFI_0001",
        subject_type="pitcher",
        subject_scope="starter_first_inning",
        target="yrfi_nrfi",
        outcome="first_inning_run_allowed",
        question_text="WHEN does this pitcher allow or prevent first-inning scoring?",
        contexts=[
            "opponent_top_order",
            "home_away",
            "park",
            "weather",
            "rest",
            "pitch_mix",
            "command_profile"
        ],
        priority="tier_1"
    ))

    questions.append(make_question(
        question_id="Q_PITCHER_PROP_0001",
        subject_type="pitcher",
        subject_scope="starter_strikeouts",
        target="pitcher_props",
        outcome="strikeouts",
        question_text="WHEN does this pitcher exceed or fall short of strikeout expectations?",
        contexts=[
            "opponent_contact_profile",
            "opponent_chase_profile",
            "umpire",
            "pitch_count_profile",
            "rest",
            "park",
            "weather"
        ],
        priority="tier_1"
    ))

    return questions


def core_batter_questions():
    questions = []

    questions.append(make_question(
        question_id="Q_BATTER_PROP_0001",
        subject_type="batter",
        subject_scope="batter_total_bases",
        target="batter_props",
        outcome="total_bases",
        question_text="WHEN does this batter create total bases?",
        contexts=[
            "pitcher_pitch_type_profile",
            "pitcher_handedness",
            "velocity",
            "movement",
            "park",
            "weather",
            "lineup_spot"
        ],
        priority="tier_1"
    ))

    return questions


def core_similarity_questions():
    questions = []

    questions.append(make_question(
        question_id="Q_SIM_PITCHER_0001",
        subject_type="similarity_group",
        subject_scope="pitcher_style",
        target="all_targets",
        outcome="style_outcome_effect",
        question_text="WHEN do teams perform differently against this pitcher style?",
        contexts=[
            "pitch_mix",
            "velocity",
            "movement",
            "command",
            "opponent_offense_identity",
            "park",
            "weather"
        ],
        priority="tier_2"
    ))

    return questions


def build_core_question_library():
    questions = []
    questions.extend(core_team_questions())
    questions.extend(core_pitcher_questions())
    questions.extend(core_batter_questions())
    questions.extend(core_similarity_questions())
    return questions
