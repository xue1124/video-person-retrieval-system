"""OSNet 单图特征匹配：均匀分桶抽帧 + 一对一两两余弦。"""

from __future__ import annotations

import numpy as np

DEFAULT_MAX_SAMPLES = 15
DEFAULT_TOP_K = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_MIN_MATCH_PAIRS = 2


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    features = np.asarray(matrix, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.size == 0:
        return features.reshape(0, 0)
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    return features / norms


def select_sample_indices(
    n_total: int,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    areas: np.ndarray | None = None,
) -> np.ndarray:
    """时间均匀分桶；每桶优先选框面积更大的帧。"""
    if n_total <= 0:
        return np.zeros(0, dtype=np.int64)
    if n_total <= max_samples:
        return np.arange(n_total, dtype=np.int64)

    edges = np.linspace(0, n_total, max_samples + 1)
    indices: list[int] = []
    for bucket in range(max_samples):
        lo = int(edges[bucket])
        hi = int(edges[bucket + 1])
        if hi <= lo:
            hi = min(n_total, lo + 1)
        if areas is None:
            indices.append((lo + hi - 1) // 2)
            continue
        bucket_areas = areas[lo:hi]
        indices.append(lo + int(np.argmax(bucket_areas)))
    # 去重并保持时间顺序
    return np.array(sorted(set(indices)), dtype=np.int64)


def select_feature_matrix(
    features: list[np.ndarray] | np.ndarray,
    *,
    areas: list[float] | np.ndarray | None = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> np.ndarray:
    if isinstance(features, np.ndarray):
        matrix = normalize_rows(features)
    else:
        if not features:
            return np.zeros((0, 0), dtype=np.float32)
        matrix = normalize_rows(np.stack(features, axis=0))
    area_arr = None if areas is None else np.asarray(areas, dtype=np.float32)
    indices = select_sample_indices(matrix.shape[0], max_samples, area_arr)
    return matrix[indices]


def topk_mean_similarity(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> float:
    """单图两两余弦相似度，取最高 top_k 个的平均（旧逻辑，检索工具仍可用）。"""
    query = normalize_rows(query_features)
    gallery = normalize_rows(gallery_features)
    if query.size == 0 or gallery.size == 0:
        return -1.0
    similarities = (query @ gallery.T).reshape(-1)
    if similarities.size == 0:
        return -1.0
    k = max(1, min(int(top_k), int(similarities.size)))
    if k == similarities.size:
        return float(np.mean(similarities))
    top = np.partition(similarities, -k)[-k:]
    return float(np.mean(top))


def greedy_one_to_one_pairs(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """按相似度从高到低做一对一匹配，同一帧只能用一次。"""
    query = normalize_rows(query_features)
    gallery = normalize_rows(gallery_features)
    if query.size == 0 or gallery.size == 0:
        return []
    sim = query @ gallery.T
    n_q, n_g = int(sim.shape[0]), int(sim.shape[1])
    order = np.argsort(-sim, axis=None)
    used_q: set[int] = set()
    used_g: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for flat in order:
        i = int(flat) // n_g
        j = int(flat) % n_g
        if i in used_q or j in used_g:
            continue
        score = float(sim[i, j])
        if score < threshold:
            break
        used_q.add(i)
        used_g.add(j)
        pairs.append((i, j, score))
        if len(used_q) == n_q or len(used_g) == n_g:
            break
    return pairs


def identity_match(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_pairs: int = DEFAULT_MIN_MATCH_PAIRS,
) -> tuple[bool, float]:
    """轨迹合并判定：两两余弦、一对一。

    - 两边都至少 min_pairs 张有效样本：至少 min_pairs 对独立匹配 ≥ threshold
    - 短轨迹不足 min_pairs 张：较短一侧的全部样本都必须匹配上且 ≥ threshold
    返回 (是否合并, 用于排序的分数=达标对中最弱的那一对)
    """
    query = normalize_rows(query_features)
    gallery = normalize_rows(gallery_features)
    n_q = int(query.shape[0]) if query.size else 0
    n_g = int(gallery.shape[0]) if gallery.size else 0
    if n_q == 0 or n_g == 0:
        return False, -1.0
    required = min(int(min_pairs), n_q, n_g)
    required = max(1, required)
    pairs = greedy_one_to_one_pairs(query, gallery, threshold=threshold)
    if len(pairs) < required:
        return False, float(pairs[-1][2]) if pairs else -1.0
    score = min(pair[2] for pair in pairs[:required])
    return True, float(score)


def merge_feature_galleries(
    *galleries: np.ndarray,
    areas_list: list[np.ndarray | None] | None = None,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    areas: list[float] = []
    for index, gallery in enumerate(galleries):
        matrix = normalize_rows(gallery)
        if matrix.size == 0:
            continue
        pieces.append(matrix)
        if areas_list and areas_list[index] is not None:
            area_arr = np.asarray(areas_list[index], dtype=np.float32).reshape(-1)
            if area_arr.size == matrix.shape[0]:
                areas.extend(float(value) for value in area_arr)
                continue
        areas.extend([1.0] * matrix.shape[0])
    if not pieces:
        return np.zeros((0, 0), dtype=np.float32)
    return select_feature_matrix(np.concatenate(pieces, axis=0), areas=areas, max_samples=max_samples)


def pack_embedding_matrix(matrix: np.ndarray) -> tuple[bytes | None, int | None, int]:
    features = normalize_rows(matrix)
    if features.size == 0:
        return None, None, 0
    return features.astype(np.float32).tobytes(), int(features.shape[1]), int(features.shape[0])


def unpack_embedding_matrix(
    blob: bytes | memoryview | None,
    *,
    embedding_dim: int | None = None,
) -> np.ndarray | None:
    if blob is None:
        return None
    vector = np.frombuffer(blob, dtype=np.float32).astype(np.float32)
    if vector.size == 0:
        return None
    dim = int(embedding_dim or 0)
    if dim > 0 and vector.size % dim == 0:
        return normalize_rows(vector.reshape(-1, dim))
    # 兼容旧数据：单向量
    return normalize_rows(vector.reshape(1, -1))
