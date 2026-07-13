"""Unit tests for cycle detection + cycle-position helpers.

These exercise the pure-numpy primitives that the new HGBR predictor is built
on. They do not need Home Assistant, sklearn, or any fitted model.
"""

from __future__ import annotations

from custom_components.fuel_predictor_wa.predictor import (
    cycle_pos_at,
    detect_hikes,
    median_cycle_len,
)


def test_detect_hikes_flags_large_positive_diffs() -> None:
    """A clear single-day spike must be flagged at the spike index."""
    # Flat at 160 for 10 days, then a hard jump to 180 held for the rest.
    prices = [160.0] * 10 + [180.0] * 10
    hikes = detect_hikes(prices)
    # diff[10] = +20.0; every other diff is 0. The hike "lands" at index 10
    # (the first post-jump price).
    assert hikes == [10]


def test_detect_hikes_flags_multiple_weekly_jumps() -> None:
    """A clean 7-day cycle (WA-like) must flag each weekly jump."""
    # Each week: jump +15 then decay -2/day for 6 days. 3 cycles -> 21 days.
    week = [180.0, 178.0, 176.0, 174.0, 172.0, 170.0, 168.0]
    prices: list[float] = []
    for _ in range(3):
        prices.extend(week)
    hikes = detect_hikes(prices)
    # First hike at index 7 (start of cycle 2), then 14 (cycle 3).
    # Index 0 cannot be a hike (no diff[-1]).
    assert hikes == [7, 14]


def test_detect_hikes_ignores_small_noise() -> None:
    """+-0.5c day-to-day wiggle must NOT register as a hike."""
    prices = [160.0 + (0.5 if i % 2 else 0.0) for i in range(30)]
    assert detect_hikes(prices) == []


def test_detect_hikes_ignores_small_declines() -> None:
    """Declines (negative diffs) are never hikes."""
    prices = [180.0 - 0.5 * i for i in range(20)]
    assert detect_hikes(prices) == []


def test_cycle_pos_zero_on_hike_grows_after() -> None:
    """cycle_pos is 0 on a hike-day and increments each subsequent day."""
    hike_days = [10]
    assert cycle_pos_at(10, hike_days) == 0
    assert cycle_pos_at(11, hike_days) == 1
    assert cycle_pos_at(12, hike_days) == 2
    assert cycle_pos_at(13, hike_days) == 3


def test_cycle_pos_resets_at_next_hike() -> None:
    """A second hike resets cycle_pos to 0."""
    hike_days = [10, 17]
    assert cycle_pos_at(10, hike_days) == 0
    assert cycle_pos_at(16, hike_days) == 6
    assert cycle_pos_at(17, hike_days) == 0
    assert cycle_pos_at(18, hike_days) == 1


def test_cycle_pos_before_first_hike() -> None:
    """Days before the first hike still produce a sane (positive) position."""
    hike_days = [10]
    # Most-recent hike <= 5 is none -> falls back to 0 (or the model's
    # empty-past convention). Either way the value must be deterministic and
    # non-negative.
    pos = cycle_pos_at(5, hike_days)
    assert pos >= 0


def test_cycle_pos_empty_hike_list() -> None:
    """No hikes at all -> cycle_pos stays non-negative and bounded."""
    pos = cycle_pos_at(20, [])
    assert pos >= 0


def test_median_cycle_len_default_when_too_few_hikes() -> None:
    """<2 hikes -> default cycle length of 7."""
    assert median_cycle_len([]) == 7
    assert median_cycle_len([10]) == 7


def test_median_cycle_len_from_intervals() -> None:
    """Three weekly hikes -> median interval 7."""
    # Hikes one week apart.
    assert median_cycle_len([7, 14, 21]) == 7
    # Irregular: intervals 5 and 9 -> median 7.0 -> banded to 7.
    assert median_cycle_len([10, 15, 24]) == 7


def test_median_cycle_len_banded_to_range() -> None:
    """Cycle length is clamped to [4, 14]."""
    # Very short intervals (3) -> banded up to 4.
    assert median_cycle_len([5, 8, 11]) == 4
    # Very long intervals (20) -> banded down to 14.
    assert median_cycle_len([0, 20, 40]) == 14
