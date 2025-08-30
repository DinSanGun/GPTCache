from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional

from gptcache import cache as GLOBAL_CACHE
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

# Optional: import Onnx embedding only when semantic mode is chosen
def _lazy_onnx():
    from gptcache.embedding import Onnx
    return Onnx()

def init_cache(
    artifacts_dir: Path,
    cache_type: Literal["exact","semantic"] = "exact",
    max_size: int = 5,
    clean_size: int = 2,
    eviction: Literal["LRU","FIFO"] = "LRU",
    similarity: Literal["exact","distance","cosine"] = "exact",
    onnx_model: Optional[str] = None,
):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = artifacts_dir / "sqlite.db"
    faiss_path = artifacts_dir / "faiss.index"

    if cache_type == "exact":
        # Use scalar-only storage;
        dm = get_data_manager(
            CacheBase("sqlite", sql_url=f"sqlite:///{sqlite_path}"),
            max_size=max_size, clean_size=clean_size, eviction=eviction,
        )
        GLOBAL_CACHE.init(data_manager=dm)  # exact match (default)
    else:
        onnx = _lazy_onnx()
        dim = onnx.dimension
        dm = get_data_manager(
            CacheBase("sqlite", sql_url=f"sqlite:///{sqlite_path}"),
            VectorBase("faiss", index_file=str(faiss_path), dimension=dim),
            max_size=max_size, clean_size=clean_size, eviction=eviction,
        )
        GLOBAL_CACHE.init(
            embedding_func=onnx.to_embeddings,
            data_manager=dm,
            similarity_evaluation=SearchDistanceEvaluation(),
        )

    return sqlite_path, faiss_path
