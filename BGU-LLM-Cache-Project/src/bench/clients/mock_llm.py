import time
from dataclasses import dataclass

@dataclass
class MockConfig:
    latency_ms: int = 50
    jitter_ms: int = 0

class MockLLM:
    def __init__(self, cfg: MockConfig | None = None):
        self.cfg = cfg or MockConfig()

    def generate(self, prompt: str) -> str:
        # Optional latency to simulate real model
        delay = self.cfg.latency_ms / 1000.0
        if self.cfg.jitter_ms:
            import random
            delay += random.uniform(0, self.cfg.jitter_ms) / 1000.0
        time.sleep(delay)
        # Deterministic text for stable caching
        return f"[MOCK::{abs(hash(prompt)) % 1_000_000}]"
