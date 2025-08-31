from pathlib import Path
from gptcache import cache as GLOBAL_CACHE, Config
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.embedding import Onnx
from gptcache.processor.pre import get_prompt
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.similarity_evaluation.exact_match import ExactMatchEvaluation

from ..ext.cost_aware import install_cost_provider


def init_cache(
    artifacts_dir: Path,
    cache_type: str = "semantic",      # "semantic" or "exact"
    max_size: int = 5,
    clean_size: int = 2,
    eviction: str = "LRU",
    similarity: str = "distance",      
    similarity_threshold: float = 0.05,
    onnx_model: str | None = None,
    cost_metric: str = "latency_ms",
    cost_decay: float = 0.0
):

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = artifacts_dir / "sqlite.db"
    faiss_path = artifacts_dir / "faiss.index"

    emb = Onnx(model=onnx_model) if onnx_model else Onnx()
    dim = emb.dimension

    dm = get_data_manager(
        cache_base=CacheBase("sqlite", sql_url=f"sqlite:///{sqlite_path}"),
        vector_base=VectorBase("faiss", index_path=str(faiss_path), dimension=dim),
        max_size=max_size,
        clean_size=clean_size,
        eviction=eviction,
    )


    sim = (similarity or "distance").lower()
    if cache_type == "exact" or sim == "exact":
        evaluator = ExactMatchEvaluation()
        thr = 1.0  # require exact string match
    elif sim in ("distance_pos", "distance-positive"):
        # Map distance -> similarity in [0..1]; higher = better
        evaluator = SearchDistanceEvaluation(max_distance=20.0, positive=True)
        # Use a HIGH threshold (0.85–0.98). Identical -> score≈1.0 -> HIT
        thr = float(similarity_threshold)
    else:
        # Raw distance; smaller = better (older GPTCache default)
        evaluator = SearchDistanceEvaluation(max_distance=20.0, positive=False)
        # Use a SMALL threshold (e.g., 0.001–0.05). Identical -> distance≈0 -> HIT
        thr = float(similarity_threshold)

    GLOBAL_CACHE.init(
        pre_embedding_func=get_prompt,
        embedding_func=emb.to_embeddings,              # keep embeddings so FAISS always gets numeric vectors
        data_manager=dm,
        similarity_evaluation=evaluator,
        config=Config(similarity_threshold=thr),
    )
    if eviction.upper() == "COST_AWARE":
        install_cost_provider(metric=cost_metric, decay=cost_decay)

    print(f"[init] similarity={sim} thr={thr}")

