"""The braille line chart, which is arithmetic and therefore testable alone.

:mod:`tinydiffusion.tui.chart` imports neither Textual nor Rich, so none of
this needs the ``tui`` extra — which is the point of keeping it separate from
the widget that colours it in.
"""

import pytest

from tinydiffusion.tui.chart import (
    BLANK,
    braille_cells,
    format_tick,
    nice_bounds,
    tick_values,
)

# --- the axis --------------------------------------------------------------


def test_the_axis_covers_the_data_with_a_little_room():
    low, high = nice_bounds([0.2, 1.0])
    assert low < 0.2
    assert high > 1.0


def test_the_axis_does_not_go_below_zero_for_a_loss():
    # A loss cannot be negative, and an axis that says it can is a worse lie
    # than a curve sitting slightly high in its panel.
    low, _ = nice_bounds([0.0, 1.0])
    assert low == 0.0


def test_a_negative_series_still_gets_its_margin():
    low, high = nice_bounds([-1.0, 1.0])
    assert low < -1.0
    assert high > 1.0


def test_a_flat_series_is_opened_out_rather_than_dividing_by_nothing():
    low, high = nice_bounds([0.5, 0.5, 0.5])
    assert low < 0.5 < high


def test_a_series_of_zeros_still_has_a_range():
    low, high = nice_bounds([0.0])
    assert high > low


def test_nothing_finite_falls_back_to_a_unit_axis():
    assert nice_bounds([]) == (0.0, 1.0)
    assert nice_bounds([float("nan"), float("inf")]) == (0.0, 1.0)


def test_the_ticks_run_from_the_top_down():
    ticks = tick_values(0.0, 1.0, 5)
    assert ticks[0] == 1.0
    assert ticks[-1] == 0.0
    assert ticks == sorted(ticks, reverse=True)


def test_a_one_row_chart_is_labelled_at_its_middle():
    assert tick_values(0.0, 1.0, 1) == [0.5]


def test_no_rows_means_no_ticks():
    assert tick_values(0.0, 1.0, 0) == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.1234, "0.1234"), (1.5, "1.50"), (250.0, "250"), (0.0, "0.0000")],
)
def test_ticks_are_formatted_to_be_read_not_to_be_precise(value, expected):
    assert format_tick(value) == expected


def test_the_very_small_and_the_very_large_go_to_exponents():
    # Which is where a loss curve spends its later epochs.
    assert "e" in format_tick(1e-5)
    assert "e" in format_tick(1e9)


def test_a_non_finite_tick_is_a_dash_not_a_crash():
    assert format_tick(float("nan")) == "-"


# --- the plot --------------------------------------------------------------


def rendered(cells):
    """The cells as lines of text, for asserting on shape rather than colour."""
    return ["".join(character for character, _ in row) for row in cells]


def test_the_grid_is_exactly_the_size_asked_for():
    cells = braille_cells([[0.0, 1.0]], width=10, height=4)
    assert len(cells) == 4
    assert all(len(row) == 10 for row in cells)


def test_a_box_with_no_room_draws_nothing():
    assert braille_cells([[0.0, 1.0]], width=0, height=4) == []
    assert braille_cells([[0.0, 1.0]], width=10, height=0) == []


def test_a_rising_series_ends_higher_than_it_starts():
    cells = braille_cells([[0.0, 1.0]], width=12, height=4, low=0.0, high=1.0)
    lines = rendered(cells)
    # The top row is drawn on at the right-hand end and not at the left.
    assert lines[0][0] == BLANK
    assert lines[0][-1] != BLANK
    assert lines[-1][0] != BLANK


def test_the_line_is_continuous_across_a_steep_step():
    # Two epochs whose losses halve should read as a fall, not as two dots with
    # a gap between them.
    cells = braille_cells([[1.0, 0.0]], width=8, height=4, low=0.0, high=1.0)
    assert all(any(character != BLANK for character in row) for row in rendered(cells))


def test_every_column_is_drawn_when_there_are_fewer_points_than_columns():
    cells = braille_cells([[0.0, 1.0, 0.0]], width=20, height=4, low=0.0, high=1.0)
    columns = list(zip(*rendered(cells), strict=True))
    assert all(any(character != BLANK for character in column) for column in columns)


def test_a_spike_survives_being_scaled_down():
    # More points than dot columns: the spike must not be sampled away.
    values = [0.0] * 200
    values[100] = 1.0
    cells = braille_cells([values], width=8, height=4, low=0.0, high=1.0)
    assert rendered(cells)[0].strip(BLANK) != ""


def test_a_single_point_draws_a_flat_line():
    # One measurement is a flat line, not a lone dot lost in a blank chart.
    cells = braille_cells([[0.5]], width=10, height=3, low=0.0, high=1.0)
    drawn = [index for index, row in enumerate(rendered(cells)) if row.strip(BLANK)]
    assert drawn == [1]


def test_an_empty_series_draws_nothing_and_does_not_raise():
    cells = braille_cells([[]], width=6, height=2)
    assert rendered(cells) == [BLANK * 6, BLANK * 6]


def test_each_cell_says_which_series_drew_it():
    cells = braille_cells([[0.0, 0.0], [1.0, 1.0]], width=6, height=4, low=0.0, high=1.0)
    owners = {owner for row in cells for _, owner in row if owner is not None}
    assert owners == {0, 1}


def test_the_later_series_is_the_one_drawn_on_top():
    # Identical series share every cell; validation sitting on top of training
    # is what makes it legible where the two coincide.
    cells = braille_cells([[0.5, 0.5], [0.5, 0.5]], width=6, height=3, low=0.0, high=1.0)
    owners = {owner for row in cells for _, owner in row if owner is not None}
    assert owners == {1}


def test_a_value_outside_an_explicit_axis_is_pinned_to_its_edge():
    # Rather than drawn off the chart, or off the end of the dot grid.
    cells = braille_cells([[5.0, -5.0]], width=6, height=3, low=0.0, high=1.0)
    lines = rendered(cells)
    assert lines[0].strip(BLANK)
    assert lines[-1].strip(BLANK)


def test_an_axis_with_no_range_draws_a_blank_grid():
    cells = braille_cells([[1.0]], width=4, height=2, low=1.0, high=1.0)
    assert rendered(cells) == [BLANK * 4, BLANK * 4]


def test_non_finite_points_are_skipped_rather_than_poisoning_the_axis():
    cells = braille_cells([[0.0, float("nan"), 1.0]], width=8, height=3, low=0.0, high=1.0)
    assert any(row.strip(BLANK) for row in rendered(cells))
