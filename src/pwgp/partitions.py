"""Fixed checkerboards and clipped Voronoi cells with straight boundary stitches."""

import numpy as np
from scipy.spatial import HalfspaceIntersection


def validate_inputs(x, stitches_per_edge):
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("Partition inputs must be finite x[N,2]")
    if (x <= 0).any() or (x >= 1).any():
        raise ValueError("Partition training inputs must be strictly inside the unit square")
    if not isinstance(stitches_per_edge, int) or isinstance(stitches_per_edge, bool) or stitches_per_edge < 1:
        raise ValueError("stitches_per_edge must be a positive integer")


def checkerboard_partition(x, *, grid_shape, stitches_per_edge):
    validate_inputs(x, stitches_per_edge)
    if (grid_shape is None or len(grid_shape) != 2 or
            any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in grid_shape)):
        raise ValueError("grid_shape must contain positive integer counts (longitude, latitude)")
    nx, ny = grid_shape
    bounds, polygons, points, edges = [], [], [], []
    for ix in range(nx):
        for iy in range(ny):
            lo, hi = np.array([ix / nx, iy / ny]), np.array([(ix + 1) / nx, (iy + 1) / ny])
            bounds.append([lo, hi])
            polygons.append(np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]]))
            for axis, neighbor in ((0, (ix + 1) * ny + iy), (1, ix * ny + iy + 1)):
                if (axis == 0 and ix + 1 == nx) or (axis == 1 and iy + 1 == ny):
                    continue
                for value in np.linspace(lo[1 - axis], hi[1 - axis], stitches_per_edge + 2)[1:-1]:
                    point = hi.copy()
                    point[1 - axis] = value
                    points.append(point)
                    edges.append((ix * ny + iy, neighbor))
    labels = np.floor(x * [nx, ny]).astype(np.int64)
    labels = labels[:, 0] * ny + labels[:, 1]
    return labels, np.asarray(bounds), np.asarray(points).reshape(-1, 2), edges, polygons


def voronoi_partition(x, *, centers, stitches_per_edge, tolerance=1e-12):
    validate_inputs(x, stitches_per_edge)
    centers = np.asarray(centers, dtype=np.float64)
    if (centers.ndim != 2 or centers.shape[1] != 2 or len(centers) < 1 or
            not np.isfinite(centers).all() or (centers <= 0).any() or (centers >= 1).any()):
        raise ValueError("Voronoi centers must be finite [patches,2] coordinates inside (0,1)")
    if len(np.unique(centers, axis=0)) != len(centers):
        raise ValueError("Voronoi centers must be distinct")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Geometry tolerance must be finite and positive")
    box = np.array([[-1., 0., 0.], [1., 0., -1.], [0., -1., 0.], [0., 1., -1.]])
    polygons, bounds, points, edges = [], [], [], []
    for i, center in enumerate(centers):
        others = np.delete(centers, i, axis=0)
        halfspaces = np.column_stack((2 * (others - center), (center**2).sum() - (others**2).sum(1)))
        vertices = HalfspaceIntersection(np.vstack((box, halfspaces)), center).intersections
        middle = vertices.mean(0)
        vertices = vertices[np.arctan2(vertices[:, 1] - middle[1], vertices[:, 0] - middle[0]).argsort()]
        polygons.append(vertices)
        bounds.append([vertices.min(0), vertices.max(0)])
    for i, center in enumerate(centers):
        for j in range(i + 1, len(centers)):
            normal = centers[j] - center
            normal = normal / np.linalg.norm(normal)
            middle = 0.5 * (center + centers[j])
            on_line = np.abs((polygons[i] - middle) @ normal) <= tolerance
            if on_line.sum() < 2:
                continue
            tangent = np.array([-normal[1], normal[0]])
            coordinates = (polygons[i][on_line] - middle) @ tangent
            lo, hi = coordinates.min(), coordinates.max()
            if hi - lo <= tolerance:
                continue
            for value in np.linspace(lo, hi, stitches_per_edge + 2)[1:-1]:
                points.append(middle + value * tangent)
                edges.append((i, j))
    labels = ((x[:, None] - centers[None])**2).sum(-1).argmin(1)
    return labels, np.asarray(bounds), np.asarray(points).reshape(-1, 2), edges, polygons
