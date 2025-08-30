# Ensures the GPTCache parent repo is importable when running from BGU-LLM-Cache-Project.
import sys, os
from pathlib import Path

_here = Path(__file__).resolve()
# BGU-LLM-Cache-Project/src -> BGU-LLM-Cache-Project -> GPTCache root
gptcache_root = _here.parents[2].parent
if str(gptcache_root) not in sys.path:
    sys.path.append(str(gptcache_root))
