
PITCHER_ENGINE_VERSION = "2.0.0"
PITCHER_CARD_VERSION = "2.0.0"

HIT_EVENTS = {
    "single",
    "double",
    "triple",
    "home_run",
}

WALK_EVENTS = {
    "walk",
    "intent_walk",
}

STRIKEOUT_EVENTS = {
    "strikeout",
    "strikeout_double_play",
}

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
}

FOUL_DESCRIPTIONS = {
    "foul",
    "foul_tip",
    "foul_bunt",
}

PITCH_COUNT_BINS = [
    0,
    15,
    30,
    45,
    60,
    75,
    90,
    105,
    float("inf"),
]

PITCH_COUNT_LABELS = [
    "001_015",
    "016_030",
    "031_045",
    "046_060",
    "061_075",
    "076_090",
    "091_105",
    "106_plus",
]
