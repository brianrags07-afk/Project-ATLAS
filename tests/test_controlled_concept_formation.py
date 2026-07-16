


def test_target_specific_analogue_keys_do_not_cross_contaminate():
    analogue_pairs = {
        (
            "target_team_win_by_2_plus",
            "identity__identity_edge__outcome__win_by_2_plus",
        ),
    }

    member_for_other_target = (
        "target_team_win",
        "identity__identity_edge__outcome__win_by_2_plus",
    )

    member_for_same_target = (
        "target_team_win_by_2_plus",
        "identity__identity_edge__outcome__win_by_2_plus",
    )

    assert member_for_other_target not in analogue_pairs
    assert member_for_same_target in analogue_pairs

