from pathlib import Path
import json

import pandas as pd


ROOT = Path(
    "/content/drive/MyDrive/Project_Atlas"
)

ROLE_DIR = (
    ROOT
    / "data"
    / "game_intelligence"
    / "scoring_event_roles"
    / "2024"
)

TEAM_FLOW_DIR = (
    ROOT
    / "data"
    / "game_intelligence"
    / "team_game_flow"
    / "2024"
)

LEAD_DIR = (
    ROOT
    / "data"
    / "game_intelligence"
    / "lead_protection"
    / "2024"
)

RESPONSE_DIR = (
    ROOT
    / "data"
    / "game_intelligence"
    / "response_recovery"
    / "2024"
)

FACT_DIR = (
    ROOT
    / "data"
    / "game_intelligence"
    / "game_flow_facts"
    / "2024"
)


def _roles():
    return pd.read_parquet(
        ROLE_DIR
        / "scoring_event_roles.parquet"
    )


def _team_flow():
    return pd.read_parquet(
        TEAM_FLOW_DIR
        / "team_game_flow.parquet"
    )


def _lead():
    return pd.read_parquet(
        LEAD_DIR
        / "team_lead_protection.parquet"
    )


def _response():
    return pd.read_parquet(
        RESPONSE_DIR
        / "team_response_recovery.parquet"
    )


def _facts():
    return pd.read_parquet(
        FACT_DIR
        / "team_game_flow_facts.parquet"
    )


def test_full_2024_scoring_event_role_artifact():
    roles = _roles()

    assert len(roles) == 16_450
    assert roles["game_pk"].nunique() == 2_428

    assert not roles.duplicated(
        subset=[
            "game_pk",
            "scoring_event_number",
        ]
    ).any()


def test_scoring_event_role_distribution():
    roles = _roles()

    counts = (
        roles[
            "primary_scoring_role"
        ]
        .value_counts()
        .to_dict()
    )

    assert counts == {
        "lead_extension": 7_015,
        "deficit_reduction": 3_352,
        "opening_score": 2_428,
        "go_ahead_score": 2_063,
        "tying_score": 1_592,
    }


def test_exactly_one_role_per_scoring_event():
    roles = _roles()

    columns = [
        "opening_score",
        "tying_score",
        "go_ahead_score",
        "lead_extension",
        "deficit_reduction",
    ]

    assert roles[
        columns
    ].sum(
        axis=1
    ).eq(1).all()


def test_one_opening_and_decisive_event_per_game():
    roles = _roles()

    assert (
        roles.groupby(
            "game_pk",
            sort=False,
        )[
            "opening_score"
        ].sum().eq(1).all()
    )

    assert (
        roles.groupby(
            "game_pk",
            sort=False,
        )[
            "decisive_scoring_event"
        ].sum().eq(1).all()
    )


def test_decisive_event_is_scored_by_winner():
    roles = _roles()

    decisive = roles[
        roles[
            "decisive_scoring_event"
        ]
    ]

    assert decisive[
        "scoring_team"
    ].eq(
        decisive[
            "winner_team"
        ]
    ).all()


def test_delayed_update_game_747004_survives_role_layer():
    roles = _roles()

    game = roles[
        roles[
            "game_pk"
        ].eq(747004)
    ]

    delayed = game[
        game[
            "delayed_score_update"
        ]
    ]

    assert len(delayed) == 1

    row = delayed.iloc[0]

    assert row["scoring_team"] == "WSH"
    assert row["batting_side"] == "AWAY"
    assert row["scoring_side"] == "AWAY"
    assert bool(
        row[
            "scoring_attribution_repaired"
        ]
    )


def test_team_flow_artifact():
    flow = _team_flow()

    assert len(flow) == 4_856
    assert flow["game_pk"].nunique() == 2_428
    assert flow["team"].nunique() == 30

    assert (
        flow.groupby(
            "game_pk",
            sort=False,
        ).size().eq(2).all()
    )

    assert not flow.duplicated(
        subset=[
            "game_pk",
            "team",
        ]
    ).any()


def test_run_line_distribution():
    flow = _team_flow()

    assert int(
        flow[
            "failed_minus_1_5_as_winner"
        ].sum()
    ) == 675

    assert int(
        flow[
            "won_by_2_plus"
        ].sum()
    ) == 1_753

    assert int(
        flow[
            "won_by_3_plus"
        ].sum()
    ) == 1_316

    assert int(
        flow[
            "covered_minus_1_5"
        ].sum()
    ) == 1_753


def test_run_line_math():
    flow = _team_flow()

    assert flow[
        "covered_minus_1_5"
    ].eq(
        flow[
            "run_differential"
        ].ge(2)
    ).all()

    assert flow[
        "covered_plus_1_5"
    ].eq(
        flow[
            "run_differential"
        ].ge(-1)
    ).all()

    assert flow[
        "failed_minus_1_5_as_winner"
    ].eq(
        flow[
            "won"
        ]
        & flow[
            "run_differential"
        ].eq(1)
    ).all()


def test_lead_protection_artifact():
    lead = _lead()

    assert len(lead) == 4_856
    assert lead["game_pk"].nunique() == 2_428
    assert lead["team"].nunique() == 30

    assert not lead.duplicated(
        subset=[
            "game_pk",
            "team",
        ]
    ).any()


def test_lead_protection_known_counts():
    lead = _lead()

    assert int(
        lead[
            "ever_led_by_2"
        ].sum()
    ) == 2_591

    assert int(
        lead[
            "two_run_lead_held_to_final"
        ].sum()
    ) == 1_753

    assert int(
        lead[
            "led_by_2_but_failed_minus_1_5"
        ].sum()
    ) == 838

    assert int(
        lead[
            "led_by_2_but_lost"
        ].sum()
    ) == 491

    assert int(
        lead[
            "winner_failed_to_separate"
        ].sum()
    ) == 675


def test_lead_protection_math():
    lead = _lead()

    assert lead[
        "two_run_lead_held_to_final"
    ].eq(
        lead[
            "ever_led_by_2"
        ]
        & lead[
            "final_run_differential"
        ].ge(2)
    ).all()

    assert lead[
        "led_but_lost"
    ].eq(
        lead[
            "ever_led"
        ]
        & lead[
            "lost"
        ]
    ).all()


def test_response_recovery_artifact():
    response = _response()

    assert len(response) == 4_856
    assert response["game_pk"].nunique() == 2_428
    assert response["team"].nunique() == 30

    assert not response.duplicated(
        subset=[
            "game_pk",
            "team",
        ]
    ).any()


def test_response_recovery_known_counts():
    response = _response()

    assert int(
        response[
            "won_after_allowing_first_score"
        ].sum()
    ) == 765

    assert int(
        response[
            "lost_after_scoring_first"
        ].sum()
    ) == 765

    assert int(
        response[
            "same_inning_responses"
        ].sum()
    ) == 2_269

    assert int(
        response[
            "tying_responses"
        ].sum()
    ) == 1_523

    assert int(
        response[
            "go_ahead_responses"
        ].sum()
    ) == 2_617


def test_response_rates_are_bounded():
    response = _response()

    assert response[
        "immediate_response_rate"
    ].dropna().between(
        0,
        1,
    ).all()

    assert response[
        "eventual_response_rate"
    ].dropna().between(
        0,
        1,
    ).all()


def test_consolidated_fact_table():
    facts = _facts()

    assert len(facts) == 4_856
    assert len(facts.columns) == 121
    assert facts["game_pk"].nunique() == 2_428
    assert facts["team"].nunique() == 30

    assert not facts.duplicated(
        subset=[
            "game_pk",
            "team",
        ]
    ).any()


def test_consolidated_cross_layer_checks():
    facts = _facts()

    assert facts[
        "all_cross_layer_checks_pass"
    ].all()

    check_columns = [
        "shared_game_date_match",
        "shared_season_match",
        "shared_opponent_match",
        "shared_home_away_match",
        "shared_score_match",
        "shared_margin_match",
        "shared_win_loss_match",
        "shared_run_line_match",
        "score_math_match",
        "winner_margin_match",
        "loser_margin_match",
        "minus_1_5_math_match",
        "plus_1_5_math_match",
    ]

    assert facts[
        check_columns
    ].all(
        axis=1
    ).all()


def test_phase_2b_compatibility_aliases():
    facts = _facts()

    assert facts[
        "outcome__one_run_win"
    ].eq(
        facts[
            "won"
        ]
        & facts[
            "run_differential"
        ].eq(1)
    ).all()

    assert facts[
        "outcome__one_run_loss"
    ].eq(
        facts[
            "lost"
        ]
        & facts[
            "run_differential"
        ].eq(-1)
    ).all()

    assert facts[
        "outcome__win_by_2_plus"
    ].eq(
        facts[
            "won"
        ]
        & facts[
            "run_differential"
        ].ge(2)
    ).all()

    assert facts[
        "outcome__loss_by_2_plus"
    ].eq(
        facts[
            "lost"
        ]
        & facts[
            "run_differential"
        ].le(-2)
    ).all()


def test_representative_multi_run_walkoff():
    facts = _facts()

    game = facts[
        facts[
            "game_pk"
        ].eq(746576)
    ]

    colorado = game[
        game[
            "team"
        ].eq("COL")
    ].iloc[0]

    tampa = game[
        game[
            "team"
        ].eq("TB")
    ].iloc[0]

    assert bool(
        colorado[
            "won"
        ]
    )

    assert int(
        colorado[
            "run_differential"
        ]
    ) == 3

    assert bool(
        colorado[
            "covered_minus_1_5"
        ]
    )

    assert bool(
        colorado[
            "outcome__walkoff_win"
        ]
    )

    assert bool(
        tampa[
            "lead__led_but_lost"
        ]
    )


def test_representative_one_run_walkoff():
    facts = _facts()

    game = facts[
        facts[
            "game_pk"
        ].eq(745039)
    ]

    texas = game[
        game[
            "team"
        ].eq("TEX")
    ].iloc[0]

    assert bool(
        texas[
            "won"
        ]
    )

    assert int(
        texas[
            "run_differential"
        ]
    ) == 1

    assert not bool(
        texas[
            "covered_minus_1_5"
        ]
    )

    assert bool(
        texas[
            "flow__failed_minus_1_5_as_winner"
        ]
    )

    assert bool(
        texas[
            "outcome__walkoff_win"
        ]
    )


def test_saved_audits_and_failures():
    audit_paths = [
        ROLE_DIR
        / "scoring_event_role_audit.parquet",

        TEAM_FLOW_DIR
        / "team_game_flow_audit.parquet",

        LEAD_DIR
        / "team_lead_protection_audit.parquet",

        RESPONSE_DIR
        / "team_response_recovery_audit.parquet",

        FACT_DIR
        / "team_game_flow_fact_audit.parquet",
    ]

    failure_paths = [
        ROLE_DIR
        / "scoring_event_role_failures.parquet",

        TEAM_FLOW_DIR
        / "team_game_flow_failures.parquet",

        LEAD_DIR
        / "team_lead_protection_failures.parquet",

        RESPONSE_DIR
        / "team_response_recovery_failures.parquet",

        FACT_DIR
        / "team_game_flow_fact_failures.parquet",
    ]

    expected_audit_rows = [
        2_428,
        2_428,
        2_428,
        2_428,
        2_428,
    ]

    for path, expected_rows in zip(
        audit_paths,
        expected_audit_rows,
    ):
        audit = pd.read_parquet(
            path
        )

        assert len(audit) == expected_rows
        assert audit[
            "audit_pass"
        ].all()

    for path in failure_paths:
        failures = pd.read_parquet(
            path
        )

        assert failures.empty


def test_no_prediction_identity_explanation_or_future_data():
    tables = [
        _roles(),
        _team_flow(),
        _lead(),
        _response(),
        _facts(),
    ]

    for table in tables:
        assert (
            ~table[
                "prediction_created"
            ]
        ).all()

        assert (
            ~table[
                "identity_updated"
            ]
        ).all()

        assert (
            ~table[
                "future_games_used"
            ]
        ).all()

        if (
            "explanation_created"
            in table.columns
        ):
            assert (
                ~table[
                    "explanation_created"
                ]
            ).all()


def test_phase_2d_metadata():
    with (
        ROLE_DIR
        / "scoring_event_role_metadata.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        role_metadata = json.load(
            handle
        )

    with (
        FACT_DIR
        / "team_game_flow_fact_metadata.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        fact_metadata = json.load(
            handle
        )

    assert role_metadata[
        "phase_2d2_pass"
    ]

    assert int(
        role_metadata[
            "role_rows"
        ]
    ) == 16_450

    assert int(
        role_metadata[
            "decisive_scoring_events"
        ]
    ) == 2_428

    assert fact_metadata[
        "phase_2d6_pass"
    ]

    assert int(
        fact_metadata[
            "team_game_rows"
        ]
    ) == 4_856

    assert int(
        fact_metadata[
            "columns"
        ]
    ) == 121

    assert int(
        fact_metadata[
            "cross_layer_failures"
        ]
    ) == 0
