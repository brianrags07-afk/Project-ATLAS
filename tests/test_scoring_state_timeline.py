from pathlib import Path
import json

import pandas as pd


ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

TIMELINE_PATH = (
    ROOT
    / "data"
    / "game_intelligence"
    / "scoring_timelines"
    / "2024"
    / "scoring_state_timelines.parquet"
)

AUDIT_PATH = (
    ROOT
    / "data"
    / "game_intelligence"
    / "scoring_timelines"
    / "2024"
    / "scoring_state_timeline_audit.parquet"
)

FAILURE_PATH = (
    ROOT
    / "data"
    / "game_intelligence"
    / "scoring_timelines"
    / "2024"
    / "scoring_state_timeline_failures.parquet"
)

METADATA_PATH = (
    ROOT
    / "data"
    / "game_intelligence"
    / "scoring_timelines"
    / "2024"
    / "scoring_state_timeline_metadata.json"
)

OUTCOME_PATH = (
    ROOT
    / "data"
    / "game_intelligence"
    / "outcomes"
    / "2024"
    / "game_outcomes.parquet"
)


def _timeline():
    dataframe = pd.read_parquet(
        TIMELINE_PATH
    )

    dataframe["game_pk"] = pd.to_numeric(
        dataframe["game_pk"],
        errors="raise",
    ).astype("int64")

    return dataframe


def _audit():
    dataframe = pd.read_parquet(
        AUDIT_PATH
    )

    dataframe["game_pk"] = pd.to_numeric(
        dataframe["game_pk"],
        errors="raise",
    ).astype("int64")

    return dataframe


def _outcomes():
    dataframe = pd.read_parquet(
        OUTCOME_PATH
    )

    dataframe["game_pk"] = pd.to_numeric(
        dataframe["game_pk"],
        errors="raise",
    ).astype("int64")

    return dataframe


def test_full_2024_scoring_timeline_artifact():
    timeline = _timeline()

    assert len(timeline) == 16_450
    assert timeline["game_pk"].nunique() == 2_428

    assert not timeline.duplicated(
        subset=[
            "game_pk",
            "scoring_event_number",
        ]
    ).any()


def test_full_2024_scoring_timeline_audit():
    audit = _audit()

    assert len(audit) == 2_428
    assert audit["game_pk"].nunique() == 2_428
    assert audit["audit_pass"].all()
    assert audit["final_score_matches"].all()
    assert audit["score_continuity_home"].all()
    assert audit["score_continuity_away"].all()
    assert audit["provenance_pass"].all()


def test_no_saved_scoring_timeline_failures():
    failures = pd.read_parquet(
        FAILURE_PATH
    )

    assert failures.empty


def test_known_scoring_transition_counts():
    timeline = _timeline()

    assert int(
        timeline[
            "score_state_repaired"
        ].sum()
    ) == 54

    assert int(
        timeline[
            "score_change_within_plate_appearance"
        ].sum()
    ) == 114

    assert int(
        timeline[
            "delayed_score_update"
        ].sum()
    ) == 1

    assert int(
        timeline[
            "scoring_attribution_repaired"
        ].sum()
    ) == 1


def test_score_state_continuity():
    timeline = _timeline().sort_values(
        [
            "game_pk",
            "scoring_event_number",
        ],
        kind="stable",
    )

    previous_home = (
        timeline.groupby(
            "game_pk",
            sort=False,
        )[
            "post_home_score"
        ].shift(1)
    )

    previous_away = (
        timeline.groupby(
            "game_pk",
            sort=False,
        )[
            "post_away_score"
        ].shift(1)
    )

    assert (
        previous_home.isna()
        | timeline[
            "pre_home_score"
        ].eq(previous_home)
    ).all()

    assert (
        previous_away.isna()
        | timeline[
            "pre_away_score"
        ].eq(previous_away)
    ).all()


def test_positive_one_side_scoring_transitions():
    timeline = _timeline()

    assert timeline[
        "runs_on_play"
    ].gt(0).all()

    assert (
        timeline[
            "home_runs_on_play"
        ].gt(0)
        ^ timeline[
            "away_runs_on_play"
        ].gt(0)
    ).all()

    assert timeline[
        "runs_on_play"
    ].eq(
        timeline[
            "home_runs_on_play"
        ]
        + timeline[
            "away_runs_on_play"
        ]
    ).all()


def test_canonical_batting_and_scoring_sides_agree():
    timeline = _timeline()

    assert timeline[
        "batting_side"
    ].eq(
        timeline[
            "scoring_side"
        ]
    ).all()


def test_delayed_score_update_game_747004():
    timeline = _timeline()

    game = timeline[
        timeline["game_pk"].eq(
            747004
        )
    ].sort_values(
        "scoring_event_number",
        kind="stable",
    )

    delayed = game[
        game[
            "delayed_score_update"
        ]
    ]

    assert len(delayed) == 1

    row = delayed.iloc[0]

    assert int(
        row["scoring_event_number"]
    ) == 2

    assert int(
        row["source_inning"]
    ) == 1

    assert str(
        row["source_inning_half"]
    ).lower().startswith(
        "bot"
    )

    assert row[
        "source_batting_side"
    ] == "HOME"

    assert int(
        row["inning"]
    ) == 1

    assert str(
        row["inning_half"]
    ).lower().startswith(
        "top"
    )

    assert row[
        "batting_side"
    ] == "AWAY"

    assert row[
        "scoring_side"
    ] == "AWAY"

    assert row[
        "scoring_team"
    ] == "WSH"

    assert bool(
        row[
            "scoring_attribution_repaired"
        ]
    )


def test_ordinary_road_shutout_game_744795():
    timeline = _timeline()

    game = timeline[
        timeline["game_pk"].eq(
            744795
        )
    ].sort_values(
        "scoring_event_number",
        kind="stable",
    )

    assert len(game) == 2

    assert game[
        "scoring_team"
    ].eq("KC").all()

    assert (
        int(
            game[
                "post_home_score"
            ].iloc[-1]
        ),
        int(
            game[
                "post_away_score"
            ].iloc[-1]
        ),
    ) == (0, 3)


def test_extra_inning_walkoff_game_745039():
    timeline = _timeline()

    game = timeline[
        timeline["game_pk"].eq(
            745039
        )
    ].sort_values(
        "scoring_event_number",
        kind="stable",
    )

    final_row = game.iloc[-1]

    assert (
        int(
            final_row[
                "post_home_score"
            ]
        ),
        int(
            final_row[
                "post_away_score"
            ]
        ),
    ) == (4, 3)

    assert int(
        final_row["inning"]
    ) == 10

    assert str(
        final_row["inning_half"]
    ).lower().startswith(
        "bot"
    )

    assert final_row[
        "scoring_team"
    ] == "TEX"

    assert bool(
        final_row[
            "terminal_scoring_event"
        ]
    )


def test_multi_run_walkoff_game_746576():
    timeline = _timeline()

    game = timeline[
        timeline["game_pk"].eq(
            746576
        )
    ].sort_values(
        "scoring_event_number",
        kind="stable",
    )

    final_row = game.iloc[-1]

    assert (
        int(
            final_row[
                "post_home_score"
            ]
        ),
        int(
            final_row[
                "post_away_score"
            ]
        ),
    ) == (10, 7)

    assert int(
        final_row[
            "runs_on_play"
        ]
    ) == 4

    assert final_row[
        "scoring_team"
    ] == "COL"

    assert bool(
        final_row[
            "terminal_scoring_event"
        ]
    )


def test_within_plate_appearance_transitions_exist():
    timeline = _timeline()

    within_pa = timeline[
        timeline[
            "score_change_within_plate_appearance"
        ]
    ]

    assert len(within_pa) == 114
    assert within_pa[
        "game_pk"
    ].nunique() > 0

    counts = (
        timeline.groupby(
            [
                "game_pk",
                "at_bat_number",
            ],
            sort=False,
        )
        .size()
    )

    flagged_keys = set(
        zip(
            within_pa[
                "game_pk"
            ].astype(int),
            within_pa[
                "at_bat_number"
            ].astype(int),
        )
    )

    for key in flagged_keys:
        assert int(
            counts.loc[key]
        ) > 1


def test_raw_score_repairs_preserve_source_provenance():
    timeline = _timeline()

    repaired = timeline[
        timeline[
            "score_state_repaired"
        ]
    ]

    assert len(repaired) == 54

    assert repaired[
        "raw_pre_score_matches_canonical"
    ].eq(False).all()

    assert (
        repaired[
            "raw_pre_home_score"
        ].ne(
            repaired[
                "pre_home_score"
            ]
        )
        | repaired[
            "raw_pre_away_score"
        ].ne(
            repaired[
                "pre_away_score"
            ]
        )
    ).all()


def test_final_scores_match_frozen_outcomes():
    timeline = _timeline()
    outcomes = _outcomes()

    terminal = timeline[
        timeline[
            "terminal_scoring_event"
        ]
    ][
        [
            "game_pk",
            "post_home_score",
            "post_away_score",
        ]
    ].copy()

    terminal = terminal.rename(
        columns={
            "post_home_score":
                "timeline_home_score",

            "post_away_score":
                "timeline_away_score",
        }
    )

    merged = outcomes.merge(
        terminal,
        on="game_pk",
        how="left",
        validate="one_to_one",
    )

    assert len(merged) == 2_428

    assert merged[
        "timeline_home_score"
    ].notna().all()

    assert merged[
        "timeline_away_score"
    ].notna().all()

    assert merged[
        "home_score"
    ].eq(
        merged[
            "timeline_home_score"
        ]
    ).all()

    assert merged[
        "away_score"
    ].eq(
        merged[
            "timeline_away_score"
        ]
    ).all()


def test_one_terminal_scoring_event_per_game():
    timeline = _timeline()

    terminal_counts = (
        timeline.groupby(
            "game_pk",
            sort=False,
        )[
            "terminal_scoring_event"
        ].sum()
    )

    assert terminal_counts.eq(1).all()


def test_no_prediction_identity_explanation_or_future_data():
    timeline = _timeline()

    assert (
        ~timeline[
            "prediction_created"
        ]
    ).all()

    assert (
        ~timeline[
            "identity_updated"
        ]
    ).all()

    assert (
        ~timeline[
            "explanation_created"
        ]
    ).all()

    assert (
        ~timeline[
            "future_games_used"
        ]
    ).all()


def test_phase_2c_metadata():
    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    assert int(
        metadata[
            "verified_games"
        ]
    ) == 2_428

    assert int(
        metadata[
            "games_built"
        ]
    ) == 2_428

    assert int(
        metadata[
            "scoring_state_rows"
        ]
    ) == 16_450

    assert int(
        metadata[
            "build_failures"
        ]
    ) == 0

    assert int(
        metadata[
            "audit_failures"
        ]
    ) == 0

    assert int(
        metadata[
            "duplicate_scoring_event_rows"
        ]
    ) == 0

    assert int(
        metadata[
            "raw_score_state_repairs"
        ]
    ) == 54

    assert int(
        metadata[
            "within_pa_score_changes"
        ]
    ) == 114

    assert bool(
        metadata[
            "phase_2c2_pass"
        ]
    )
