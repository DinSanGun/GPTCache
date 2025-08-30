from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional

from gptcache import cache as GLOBAL_CACHE, Config
from gptcache.adapter.api import init_similar_cache
from gptcache.embedding import Onnx
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

def init_cache(
    artifacts_dir: Path,
    cache_type: Literal["exact","semantic"] = "semantic",
    max_size: int = 5,
    clean_size: int = 2,
    eviction: Literal["LRU","FIFO"] = "LRU",
    similarity: Literal["distance","cosine","exact"] = "distance",
    onnx_model: Optional[str] = None,
    similarity_threshold: float = 0.3,
):
    """Initialize GPTCache.
    For semantic mode, this sets up the adapter API so cache_get/cache_put work.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = artifacts_dir / "sqlite.db"
    faiss_path = artifacts_dir / "faiss.index"

    if cache_type == "exact":
        dm = get_data_manager(
            CacheBase("sqlite", sql_url=f"sqlite:///{sqlite_path}"),
            max_size=max_size, clean_size=clean_size, eviction=eviction,
        )
        GLOBAL_CACHE.init(data_manager=dm)
        return sqlite_path, None

    # semantic mode
    onnx = Onnx(model=onnx_model) if onnx_model else Onnx()
    dim = onnx.dimension

    dm = get_data_manager(
        CacheBase("sqlite", sql_url=f"sqlite:///{sqlite_path}"),
        VectorBase("faiss", index_path=str(faiss_path), dimension=dim),
        max_size=max_size, clean_size=clean_size, eviction=eviction,
    )

    evaluation = SearchDistanceEvaluation()
    init_similar_cache(
        cache_obj=GLOBAL_CACHE,
        data_manager=dm,
        embedding=onnx,
        evaluation=evaluation,
        config=Config(similarity_threshold=float(similarity_threshold)),
    )

    print(f"[init] sqlite: {sqlite_path.resolve()}")
    print(f"[init] faiss : {faiss_path.resolve()}")

    return sqlite_path, faiss_path
