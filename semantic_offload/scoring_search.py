"""Small IVF shortlist for mean-key / mean-head inner-product scoring.

The index is an optimization, not a replacement for exact identity checks.
Unvisited rows have no observation; they must not be published as zero scores.
"""

from dataclasses import dataclass
import torch

from semantic_offload.profiling import PROFILE


@dataclass(frozen=True)
class SearchConfig:
    mode: str = "flat"
    min_pool: int = 128
    clusters: int = 16
    probes: int = 4
    exploration: int = 8

    def __post_init__(self):
        if self.mode not in ("flat", "ivf"):
            raise ValueError("scoring_search must be flat or ivf")
        for name in ("min_pool", "clusters", "probes", "exploration"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"search {name} must be a positive integer")


class ClusterSearch:
    def __init__(self, config):
        self.config = config
        self.centroids = None
        self.groups = []
        self.size = 0
        self.cursor = 0

    def build(self, means):
        """Rebuild on pool mutation, never on an unchanged query step.

        Four deterministic spherical assignment rounds cluster directions;
        search uses raw inner products, preserving the actual scoring metric.
        """
        self.size = len(means)
        self.cursor = 0
        if self.size < self.config.min_pool or self.config.mode == "flat":
            self.centroids, self.groups = None, []
            return
        with PROFILE.scope("search.index_build", cuda=means.is_cuda):
            vectors = means.float().flatten(1)
            if not bool(torch.isfinite(vectors).all()):
                self.centroids, self.groups = None, []
                PROFILE.count("search.nonfinite_flat_fallbacks")
                return
            directions = torch.nn.functional.normalize(vectors, dim=1)
            count = min(self.config.clusters, self.size)
            seeds = torch.linspace(0, self.size - 1, count, device=means.device).long()
            centers = directions.index_select(0, seeds).clone()
            for _ in range(4):
                labels = (directions @ centers.T).argmax(dim=1)
                sums = torch.zeros_like(centers).index_add_(0, labels, directions)
                counts = torch.bincount(labels, minlength=count).clamp_min(1)
                centers = torch.nn.functional.normalize(sums / counts[:, None], dim=1)
            # Lists are materialized once per rebuild, not once per request.
            labels_cpu = labels.tolist()
            self.groups = [[] for _ in range(count)]
            for row, label in enumerate(labels_cpu):
                self.groups[label].append(row)
            # Raw centroids retain the magnitude information used by the
            # actual dot-product score. Routing is approximate; rerank exact.
            self.centroids = torch.zeros((count, vectors.shape[1]), device=means.device)
            self.centroids.index_add_(0, labels, vectors)
            self.centroids /= counts[:, None]
            PROFILE.count("search.index_builds")
            PROFILE.count("search.index_build_vector_cluster_comparisons", 4 * self.size * count)

    def candidates(self, queries):
        """Return per-query row lists, plus rotating coverage of other rows."""
        count = len(queries)
        if self.centroids is None:
            PROFILE.count("search.flat_queries", count)
            return [list(range(self.size)) for _ in range(count)]
        with PROFILE.scope("search.route", cuda=queries.is_cuda):
            route_scores = queries.float().flatten(1) @ self.centroids.T
            probes = min(self.config.probes, len(self.groups))
            routes = route_scores.topk(probes, dim=1).indices.tolist()
            result = []
            for route in routes:
                selected = {row for cluster in route for row in self.groups[cluster]}
                for _ in range(min(self.config.exploration, self.size)):
                    selected.add(self.cursor % self.size)
                    self.cursor += 1
                result.append(sorted(selected))
            PROFILE.count("search.indexed_queries", count)
            PROFILE.count("search.route_comparisons", count * len(self.groups))
            PROFILE.count("search.shortlist_rows", sum(map(len, result)))
            return result
