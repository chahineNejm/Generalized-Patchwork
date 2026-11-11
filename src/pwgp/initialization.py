"""Support, patch graph, stitch, and lengthscale initialization."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.distance import pdist
from sklearn.cluster import MiniBatchKMeans

from pwgp.models.patchwork import GeneralizedPatchworkGP


def _numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def observed_support_indices(
    x: torch.Tensor,
    count: int,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Sample row indices of distinct interior observations without replacement."""
    values = _numpy(x)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Observed support inputs must be a finite [N,D] array")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Observed support inputs must be normalized to [0,1]")
    _, indices = np.unique(values, axis=0, return_index=True)
    indices = indices[((values[indices] > 0) & (values[indices] < 1)).all(axis=1)]
    if not 1 <= count <= len(indices):
        raise ValueError(f"Requested {count} supports from {len(indices)} distinct observed locations inside (0,1)")
    chosen = np.random.default_rng(seed).choice(indices, count, replace=False)
    return torch.as_tensor(chosen, dtype=torch.long, device=x.device)


def observed_supports(x: torch.Tensor, count: int, *, seed: int = 0) -> torch.Tensor:
    """Sample distinct interior observations for the open-unit-cube parameters."""
    return x[observed_support_indices(x, count, seed=seed)].clone()


def hierarchical_initialization(
    x: torch.Tensor,
    patches: int,
    supports_per_patch: int,
    *,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Return GPW's initial supports, original row indices and clustering centers."""
    values = _numpy(x)
    if patches < 1 or supports_per_patch < 1:
        raise ValueError("patch and support counts must be positive")
    if patches > len(values) or supports_per_patch > len(values):
        raise ValueError("cluster counts cannot exceed the observation count")
    outer = MiniBatchKMeans(
        n_clusters=patches,
        random_state=seed,
        batch_size=min(2048, len(values)),
        n_init=5,
    )
    labels = outer.fit_predict(values)
    groups: list[torch.Tensor] = []
    selected: list[torch.Tensor] = []
    for patch in range(patches):
        local = values[labels == patch]
        if len(local) < supports_per_patch:
            raise ValueError(
                f"Patch {patch} has {len(local)} points for {supports_per_patch} supports"
            )
        local_x = torch.as_tensor(local, dtype=x.dtype, device=x.device)
        local_indices = observed_support_indices(local_x, supports_per_patch, seed=seed + 100 + patch)
        groups.append(local_x[local_indices].clone())
        original_indices = torch.as_tensor(np.flatnonzero(labels == patch), dtype=torch.long, device=x.device)
        selected.append(original_indices[local_indices])
    return {"supports": torch.stack(groups), "indices": torch.stack(selected),
            "centers": torch.as_tensor(outer.cluster_centers_, dtype=x.dtype, device=x.device)}


def hierarchical_supports(x: torch.Tensor, patches: int, supports_per_patch: int, *, seed: int = 0) -> torch.Tensor:
    """Cluster observations into patches, then initialize local supports."""
    return hierarchical_initialization(x, patches, supports_per_patch, seed=seed)["supports"]


def patch_neighbor_graph(
    supports: torch.Tensor, *, neighbors: int = 2
) -> list[tuple[int, int]]:
    """Build an undirected k-nearest-neighbor graph over patch centers."""
    patches = supports.shape[0]
    if not isinstance(neighbors, int) or isinstance(neighbors, bool):
        raise TypeError("neighbors must be an integer")
    minimum = 0 if patches == 1 else 1
    if not minimum <= neighbors < patches:
        raise ValueError(f"neighbors must be in [{minimum}, {patches - 1}] for {patches} patches")
    if patches == 1:
        return []
    centers = supports.mean(1)
    distances = torch.cdist(centers, centers)
    distances.fill_diagonal_(torch.inf)
    edges: set[tuple[int, int]] = set()
    for left in range(patches):
        for right in torch.topk(distances[left], k=neighbors, largest=False).indices.tolist():
            edges.add(tuple(sorted((left, right))))
    return sorted(edges)


def initialize_stitches(
    supports: torch.Tensor,
    edges: list[tuple[int, int]],
    stitches_per_edge: int,
) -> torch.Tensor:
    """Place stitches at midpoints of nearby supports across adjacent patches."""
    if stitches_per_edge < 0:
        raise ValueError("stitches_per_edge cannot be negative")
    if not edges:
        return supports.new_empty((0, stitches_per_edge, supports.shape[-1]))
    maximum_stitches = supports.shape[1] ** 2
    if stitches_per_edge > maximum_stitches:
        raise ValueError(
            f"stitches_per_edge={stitches_per_edge} exceeds {maximum_stitches} support pairs"
        )
    initialized: list[torch.Tensor] = []
    for left, right in edges:
        distances = torch.cdist(supports[left], supports[right])
        order = torch.argsort(distances.reshape(-1))
        points: list[torch.Tensor] = []
        for flat in order[:stitches_per_edge].tolist():
            a = flat // supports.shape[1]
            b = flat % supports.shape[1]
            points.append(0.5 * (supports[left, a] + supports[right, b]))
        initialized.append(
            torch.stack(points) if points else supports.new_empty((0, supports.shape[-1]))
        )
    return torch.stack(initialized)


def median_ard_lengthscale(
    x: torch.Tensor, *, sample_limit: int = 4000, seed: int = 0
) -> np.ndarray:
    values = _numpy(x)
    if len(values) > sample_limit:
        generator = np.random.default_rng(seed)
        values = values[generator.choice(len(values), sample_limit, replace=False)]
    scales: list[float] = []
    for dimension in range(values.shape[1]):
        distances = pdist(values[:, dimension : dimension + 1])
        distances = distances[distances > 1e-12]
        if not len(distances):
            raise ValueError(f"Input dimension {dimension} has no nonzero pairwise distances")
        scales.append(float(np.median(distances)))
    return np.asarray(scales)


def local_ard_lengthscales(
    x: torch.Tensor, supports: torch.Tensor, *, seed: int = 0
) -> torch.Tensor:
    values = _numpy(x)
    support_values = _numpy(supports)
    centers = support_values.mean(1)
    labels = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
    local_scales: list[np.ndarray] = []
    for patch in range(len(centers)):
        local = values[labels == patch]
        if len(local) < 8:
            raise ValueError(
                f"Patch {patch} has {len(local)} observations; at least 8 are required"
            )
        scales: list[float] = []
        for dimension in range(values.shape[1]):
            distances = pdist(local[:, dimension : dimension + 1])
            nonzero = distances[distances > 1e-12]
            if not len(nonzero):
                raise ValueError(
                    f"Patch {patch}, dimension {dimension} has no nonzero distances"
                )
            scales.append(float(np.median(nonzero)))
        local_scales.append(np.asarray(scales))
    return torch.as_tensor(
        np.stack(local_scales), dtype=x.dtype, device=x.device
    )


def build_patchwork(
    x: torch.Tensor,
    *,
    patches: int,
    supports_per_patch: int,
    stitches_per_edge: int,
    q_rank: int,
    neighbors: int = 2,
    seed: int = 0,
    **model_options: object,
) -> GeneralizedPatchworkGP:
    """Construct a Patchwork model with reproducible shared initialization."""
    supports = hierarchical_supports(
        x, patches, supports_per_patch, seed=seed
    )
    edges = patch_neighbor_graph(supports, neighbors=neighbors)
    if stitches_per_edge == 0:
        edges = []
    stitches = initialize_stitches(supports, edges, stitches_per_edge)
    lengthscales = local_ard_lengthscales(x, supports, seed=seed)
    return GeneralizedPatchworkGP(
        supports,
        stitches,
        edges,
        q_rank=q_rank,
        lengthscale=lengthscales,
        **model_options,
    )

