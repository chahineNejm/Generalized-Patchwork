"""Fixed geometric partitions and balanced PC trees."""

import numpy as np
from scipy.optimize import linprog
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


def pc_tree_partition(x, *, patches, stitches_per_edge, seed, tolerance=1e-12):
    if x.ndim == 2 and x.shape[1] > 2:
        return nd_pc_tree_partition(x, patches=patches, stitches_per_edge=stitches_per_edge,
                                    seed=seed, tolerance=tolerance)
    return _pc_tree_partition_2d(x, patches=patches, stitches_per_edge=stitches_per_edge,
                                  seed=seed, tolerance=tolerance)


def _pc_tree_partition_2d(x, *, patches, stitches_per_edge, seed, tolerance):
    """Balanced principal-component splits with uniform random boundary stitches."""
    validate_inputs(x, stitches_per_edge)
    if not isinstance(patches, int) or isinstance(patches, bool) or patches < 1 or patches & (patches - 1):
        raise ValueError("PC-tree patches must be a positive power of two; counts are never rounded")
    if len(x) < patches or len(x) % patches:
        raise ValueError("PC-tree observation count must divide evenly across the requested patches")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Geometry tolerance must be finite and positive")
    box = np.array([[-1., 0., 0.], [1., 0., -1.], [0., -1., 0.], [0., 1., -1.]])
    nodes, leaves = [], []
    labels = np.empty(len(x), dtype=np.int64)

    def split(indices, halfspaces, depth):
        index = len(nodes)
        nodes.append([0., 0., 0., -1, -1, -1])
        values = x[indices]
        if depth == 0:
            patch = len(leaves)
            labels[indices] = patch
            nodes[index][-1] = patch
            leaves.append((halfspaces, values.mean(0)))
            return index
        centered = values - values.mean(0)
        eigenvalues, vectors = np.linalg.eigh(centered.T @ centered)
        if eigenvalues[-1] <= 0:
            raise ValueError("PC-tree cannot split a region with zero input variance")
        normal = vectors[:, -1]
        normal = normal if normal[np.abs(normal).argmax()] > 0 else -normal
        projection = values @ normal
        ordered = np.sort(projection)
        lo, hi = ordered[len(indices) // 2 - 1:len(indices) // 2 + 1]
        cut = lo + (hi - lo) / 2
        if not lo < cut < hi:
            raise ValueError("PC-tree projection ties prevent an equal-count geometric split")
        left = projection <= cut
        plane = np.r_[normal, -cut]
        a = split(indices[left], np.vstack((halfspaces, plane)), depth - 1)
        b = split(indices[~left], np.vstack((halfspaces, -plane)), depth - 1)
        nodes[index] = [*normal, cut, a, b, -1]
        return index

    split(np.arange(len(x)), box, patches.bit_length() - 1)
    polygons = []
    for halfspaces, interior in leaves:
        vertices = HalfspaceIntersection(halfspaces, interior).intersections
        middle = vertices.mean(0)
        polygons.append(vertices[np.arctan2(vertices[:, 1] - middle[1], vertices[:, 0] - middle[0]).argsort()])
    rng = np.random.default_rng(seed)
    points, edges = [], []
    for i in range(patches):
        for j in range(i + 1, patches):
            for plane in leaves[i][0][4:]:
                if not np.isclose(leaves[j][0], -plane, atol=tolerance, rtol=0).all(1).any():
                    continue
                vertices = [p[np.abs(p @ plane[:2] + plane[2]) <= tolerance] for p in (polygons[i], polygons[j])]
                if any(len(v) < 2 for v in vertices):
                    continue
                tangent = np.array([-plane[1], plane[0]])
                intervals = [v @ tangent for v in vertices]
                lo = max(v.min() for v in intervals)
                hi = min(v.max() for v in intervals)
                if hi - lo <= tolerance:
                    continue
                samples = rng.uniform(lo, hi, stitches_per_edge)
                if not ((samples > lo) & (samples < hi)).all():
                    raise ValueError("PC-tree sampled a boundary endpoint, not a boundary-interior stitch")
                points.extend(-plane[2] * plane[:2] + samples[:, None] * tangent)
                edges.extend([(i, j)] * stitches_per_edge)
                break
    connected = {0}
    for _ in range(patches):
        connected.update(b for a, b in edges if a in connected)
        connected.update(a for a, b in edges if b in connected)
    if len(connected) != patches:
        raise ValueError("PC-tree shared-boundary graph is disconnected")
    bounds = np.asarray([[v.min(0), v.max(0)] for v in polygons])
    return labels, bounds, np.asarray(points).reshape(-1, 2), edges, polygons, np.asarray(nodes)


def nd_pc_tree_partition(x, *, patches, stitches_per_edge, seed, tolerance):
    """Paper-style balanced PC splits and exact uniform hyperplane rejection draws."""
    if (x.ndim != 2 or x.shape[1] < 3 or not np.isfinite(x).all()
            or (x <= 0).any() or (x >= 1).any()):
        raise ValueError("PC-tree requires finite x[N,D], D>=3, strictly inside the unit cube")
    if not isinstance(patches, int) or isinstance(patches, bool) or patches < 1 or patches & (patches - 1):
        raise ValueError("PC-tree patches must be a positive power of two")
    if len(x) < patches or len(x) % patches:
        raise ValueError("PC-tree observation count must divide evenly across requested patches")
    if not isinstance(stitches_per_edge, int) or isinstance(stitches_per_edge, bool) or stitches_per_edge < 1:
        raise ValueError("stitches_per_edge must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Geometry tolerance must be finite and positive")
    dimension = x.shape[1]
    box = np.column_stack((np.vstack((-np.eye(dimension), np.eye(dimension))),
                           np.r_[np.zeros(dimension), -np.ones(dimension)]))
    nodes, leaves = [], []
    labels = np.empty(len(x), dtype=np.int64)

    def split(indices, planes, depth):
        index = len(nodes)
        nodes.append([*np.zeros(dimension), 0., -1., -1., -1.])
        if depth == 0:
            labels[indices] = len(leaves)
            nodes[index][-1] = len(leaves)
            leaves.append(planes)
            return index
        values = x[indices]
        centered = values - values.mean(0)
        eigenvalues, vectors = np.linalg.eigh(centered.T @ centered)
        if eigenvalues[-1] <= 0:
            raise ValueError("PC-tree cannot split a region with zero input variance")
        normal = vectors[:, -1]
        normal = normal if normal[np.abs(normal).argmax()] > 0 else -normal
        projection = values @ normal
        ordered = np.sort(projection)
        lo, hi = ordered[len(indices) // 2 - 1:len(indices) // 2 + 1]
        cut = lo + (hi - lo) / 2
        if not lo < cut < hi:
            raise ValueError("PC-tree projection ties prevent an equal-count geometric split")
        left = projection <= cut
        plane = np.r_[normal, -cut]
        a = split(indices[left], np.vstack((planes, plane)), depth - 1)
        b = split(indices[~left], np.vstack((planes, -plane)), depth - 1)
        nodes[index] = [*normal, cut, a, b, -1]
        return index

    split(np.arange(len(x)), box, patches.bit_length() - 1)
    rng = np.random.default_rng(seed)
    points, edges, bounds = [], [], []
    for planes in leaves:
        limits = []
        for axis in np.eye(dimension):
            pair = []
            for objective in (axis, -axis):
                result = linprog(objective, A_ub=planes[:, :-1], b_ub=-planes[:, -1],
                                 bounds=[(None, None)] * dimension, method="highs")
                if not result.success:
                    raise ValueError(f"PC-tree region bounds failed: {result.message}")
                pair.append(result.x @ axis)
            limits.append(pair)
        bounds.append(np.asarray(limits).T)
    for i in range(patches):
        for j in range(i + 1, patches):
            for plane in leaves[i][2 * dimension:]:
                if not np.isclose(leaves[j], -plane, atol=tolerance, rtol=0).all(1).any():
                    continue
                planes = np.vstack((leaves[i], leaves[j]))
                on_plane = np.isclose(planes, plane, atol=tolerance, rtol=0).all(1)
                on_plane |= np.isclose(planes, -plane, atol=tolerance, rtol=0).all(1)
                constraints = planes[~on_plane]
                normal = plane[:-1]
                result = linprog(np.r_[np.zeros(dimension), -1.],
                    A_ub=np.column_stack((constraints[:, :-1], np.linalg.norm(constraints[:, :-1], axis=1))),
                    b_ub=-constraints[:, -1], A_eq=np.array([np.r_[normal, 0.]]), b_eq=[-plane[-1]],
                    bounds=[(None, None)] * dimension + [(0, None)], method="highs")
                if result.status == 2:
                    continue
                if not result.success:
                    raise ValueError(f"PC-tree boundary feasibility failed: {result.message}")
                if result.x[-1] <= tolerance:
                    continue
                # Eliminating one coordinate has a constant surface Jacobian.
                # Uniform cube draws accepted inside the face are thus uniform on it.
                pivot = np.abs(normal).argmax()
                free = np.arange(dimension) != pivot
                drawn = []
                while len(drawn) < stitches_per_edge:
                    candidate = np.empty(dimension)
                    candidate[free] = rng.random(dimension - 1)
                    candidate[pivot] = (-plane[-1] - candidate[free] @ normal[free]) / normal[pivot]
                    if ((candidate > 0).all() and (candidate < 1).all()
                            and (constraints[:, :-1] @ candidate + constraints[:, -1] < 0).all()):
                        drawn.append(candidate)
                points.extend(drawn)
                edges.extend([(i, j)] * stitches_per_edge)
                break
    connected = {0}
    for _ in range(patches):
        connected.update(b for a, b in edges if a in connected)
        connected.update(a for a, b in edges if b in connected)
    if len(connected) != patches:
        raise ValueError("PC-tree shared-boundary graph is disconnected")
    vertices = [np.empty((0, dimension)) for _ in leaves]
    return labels, np.asarray(bounds), np.asarray(points).reshape(-1, dimension), edges, vertices, np.asarray(nodes)
