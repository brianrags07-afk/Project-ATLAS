
from copy import deepcopy
from datetime import datetime


QUESTION_EXPANSION_VERSION = "1.0.0"


BASIC_CONTEXT_VALUES = {
    "home_away": ["home", "away"],
    "pitcher_handedness": ["vs_lhp", "vs_rhp"],
    "park_type": ["pitcher_friendly", "neutral", "hitter_friendly"],
    "weather_temp": ["cold", "mild", "hot"],
    "weather_wind": ["wind_in", "neutral_wind", "wind_out"],
    "rest": ["short_rest", "normal_rest", "extra_rest"],
    "series_game": ["game_1", "game_2", "game_3", "game_4", "getaway_day"],
}


QUESTION_CONTEXT_PLAN = {
    "moneyline": ["home_away", "rest", "series_game"],
    "totals": ["home_away", "park_type", "weather_temp", "weather_wind", "series_game"],
    "runline": ["home_away", "rest", "park_type"],
    "yrfi_nrfi": ["home_away", "pitcher_handedness", "weather_temp", "weather_wind"],
    "pitcher_props": ["home_away", "pitcher_handedness", "rest", "park_type"],
    "batter_props": ["home_away", "pitcher_handedness", "park_type", "weather_temp"],
    "all_targets": ["home_away", "park_type", "weather_temp", "weather_wind"],
}


def expand_question(question):
    expanded = []

    target = question.get("target")
    context_keys = QUESTION_CONTEXT_PLAN.get(target, [])

    for context_key in context_keys:
        values = BASIC_CONTEXT_VALUES.get(context_key, [])

        for value in values:
            q = deepcopy(question)

            q["parent_question_id"] = question["question_id"]
            q["question_id"] = f"{question['question_id']}__{context_key.upper()}__{value.upper()}"
            q["question_type"] = "expanded_context"
            q["expanded_context"] = {
                "context_key": context_key,
                "context_value": value,
            }
            q["status"] = "unanswered"
            q["evidence_status"] = "not_collected"
            q["created_at"] = datetime.utcnow().isoformat()
            q["updated_at"] = datetime.utcnow().isoformat()

            expanded.append(q)

    return expanded


def expand_question_library(core_questions, include_core=True):
    all_questions = []

    if include_core:
        all_questions.extend(core_questions)

    for question in core_questions:
        all_questions.extend(expand_question(question))

    return all_questions
