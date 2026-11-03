"""Index invariants and timing semantics; CPU-only, no vLLM required."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from semantic_offload.profiling import ComponentProfiler
from semantic_offload.scoring_search import ClusterSearch, SearchConfig


class SearchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(310)
        self.means = torch.randn(160, 2, 8)
        self.queries = torch.randn(4, 2, 8)

    def test_small_pool_is_exhaustive(self):
        index = ClusterSearch(SearchConfig("ivf", 256))
        index.build(self.means)
        self.assertIsNone(index.centroids)
        self.assertEqual(index.candidates(self.queries), [list(range(160))] * 4)

    def test_all_probes_recover_all_rows_and_exact_dot_product_ranking(self):
        index = ClusterSearch(SearchConfig("ivf", 32, 8, 8, 1))
        index.build(self.means)
        for query, rows in zip(self.queries, index.candidates(self.queries)):
            self.assertEqual(rows, list(range(160)))
            flat = (self.means * query).sum(-1).mean(-1)
            selected = (self.means[rows] * query).sum(-1).mean(-1)
            torch.testing.assert_close(flat, selected, rtol=0, atol=0)

    def test_shortlists_are_bounded_unique_and_exploration_covers_every_row(self):
        index = ClusterSearch(SearchConfig("ivf", 32, 16, 1, 4))
        index.build(self.means)
        visited = set()
        for _ in range(40):
            rows = index.candidates(self.queries[:1])[0]
            self.assertEqual(rows, sorted(set(rows)))
            self.assertTrue(all(0 <= row < 160 for row in rows))
            visited.update(rows)
        self.assertEqual(visited, set(range(160)))

    def test_rebuild_removes_old_rows_and_handles_empty_and_nonfinite(self):
        index = ClusterSearch(SearchConfig("ivf", 32))
        index.build(self.means)
        index.build(self.means[:12])
        self.assertEqual(index.candidates(self.queries[:1]), [list(range(12))])
        index.build(self.means[:0])
        self.assertEqual(index.candidates(self.queries[:1]), [[]])
        broken = self.means.clone()
        broken[0, 0, 0] = float("nan")
        index.build(broken)
        self.assertIsNone(index.centroids)

    def test_invalid_config_rejected(self):
        for kwargs in ({"mode": "tree"}, {"probes": 0}, {"min_pool": True}):
            with self.assertRaises(ValueError):
                SearchConfig(**kwargs)


class ProfileTests(unittest.TestCase):
    def test_host_distribution_and_deferred_device_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = ComponentProfiler(directory, cuda_stride=1)
            events = []
            class Event:
                def __init__(self, **kwargs):
                    events.append(self)
                def record(self):
                    pass
                def query(self):
                    return len(events) > 2
                def elapsed_time(self, end):
                    return 0.25
                def synchronize(self):
                    raise AssertionError("profiling must not synchronize CUDA")
            with patch("torch.cuda.is_initialized", return_value=True), patch("torch.cuda.Event", Event):
                with profile.scope("math", cuda=True):
                    profile.count("rows", 7)
                profile.drain()
                self.assertEqual(len(profile.pending), 1)
                with profile.scope("math", cuda=True):
                    pass
                profile.flush()
            report = json.loads(next(Path(directory).glob("*.json")).read_text())
            self.assertEqual(report["metrics"]["math.host_wall"]["count"], 2)
            self.assertEqual(report["metrics"]["math.cuda_stream"]["total_ns"], 500_000)
            self.assertEqual(report["counters"]["rows"], 7)
            self.assertEqual(report["pending_cuda_samples"], 0)


if __name__ == "__main__":
    unittest.main()
