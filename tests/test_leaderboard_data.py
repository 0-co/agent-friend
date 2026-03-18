"""Tests for the leaderboard benchmark data module."""

import pytest

from agent_friend.leaderboard_data import LEADERBOARD, LEADERBOARD_URL, get_leaderboard_position


class TestLeaderboardData:
    def test_leaderboard_has_50_entries(self):
        assert len(LEADERBOARD) == 50

    def test_leaderboard_sorted_descending(self):
        scores = [entry[2] for entry in LEADERBOARD]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                "Entry {i} ({s1}) should be >= entry {j} ({s2})".format(
                    i=i, s1=scores[i], j=i + 1, s2=scores[i + 1],
                )
            )

    def test_leaderboard_url_is_string(self):
        assert isinstance(LEADERBOARD_URL, str)
        assert LEADERBOARD_URL.startswith("https://")


class TestGetLeaderboardPosition:
    def test_perfect_score_is_rank_1(self):
        """Score of 100.0 ties with PostgreSQL, should be rank 1."""
        rank, total, above, below = get_leaderboard_position(100.0)
        assert rank == 1

    def test_tied_with_notion_is_rank_50(self):
        """Score of 19.8 ties with Notion (last entry), should be rank 50."""
        rank, total, above, below = get_leaderboard_position(19.8)
        assert rank == 50

    def test_worse_than_all_is_rank_51(self):
        """Score of 0 is worse than all 50 entries, should be rank 51."""
        rank, total, above, below = get_leaderboard_position(0)
        assert rank == 51

    def test_better_than_all_is_rank_1(self):
        """Score of 200 is better than all entries, should be rank 1."""
        rank, total, above, below = get_leaderboard_position(200)
        assert rank == 1

    def test_total_is_50(self):
        rank, total, above, below = get_leaderboard_position(50)
        assert total == 50

    def test_mid_range_position_and_neighbors(self):
        """Score of 50 should be between cloudflare (51.4) and pal (49.0)."""
        rank, total, above, below = get_leaderboard_position(50)
        # 50.0 is less than cloudflare (51.4) but more than pal (49.0)
        # Entries with score > 50: cloudflare (51.4) and above = 40 entries
        # So rank = 41
        assert rank == 41

        # servers_above: up to 2 servers immediately above (higher score)
        assert len(above) <= 2
        assert len(above) > 0
        # The server immediately above should be cloudflare (51.4)
        assert above[-1][0] == "Cloudflare Radar MCP Server"
        assert above[-1][1] == 51.4

        # servers_below: up to 2 servers immediately below (lower score)
        assert len(below) <= 2
        assert len(below) > 0
        # The server immediately below should be pal (49.0)
        assert below[0][0] == "PAL MCP Server"
        assert below[0][1] == 49.0

    def test_rank_1_has_no_above(self):
        """Best possible rank should have no servers above."""
        rank, total, above, below = get_leaderboard_position(200)
        assert rank == 1
        assert len(above) == 0

    def test_worst_rank_has_no_below(self):
        """Worst possible rank should have no servers below."""
        rank, total, above, below = get_leaderboard_position(0)
        assert rank == 51
        assert len(below) == 0

    def test_above_limited_to_2(self):
        """servers_above should contain at most 2 entries."""
        rank, total, above, below = get_leaderboard_position(50)
        assert len(above) <= 2

    def test_below_limited_to_2(self):
        """servers_below should contain at most 2 entries."""
        rank, total, above, below = get_leaderboard_position(50)
        assert len(below) <= 2

    def test_neighbors_are_tuples(self):
        """Neighbor entries should be (name, score) tuples."""
        rank, total, above, below = get_leaderboard_position(50)
        for name, score in above:
            assert isinstance(name, str)
            assert isinstance(score, float)
        for name, score in below:
            assert isinstance(name, str)
            assert isinstance(score, float)
