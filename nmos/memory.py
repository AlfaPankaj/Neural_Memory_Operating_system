import os
import time
from collections import OrderedDict

import torch


class NMOSMemoryController:
    """
    Manages the 4GB VRAM action zone and a paged-KV metadata pool.
    """

    def __init__(self, total_vram_mb=4096):
        self.total_vram_mb = int(total_vram_mb)
        self.reserved_for_system = 300
        self.draft_model_reserved = 1100
        self.kv_cache_reserved = 800
        self.page_size_mb = 16
        self.max_pages = max(1, self.kv_cache_reserved // self.page_size_mb)
        self.action_window_mb = (
            self.total_vram_mb
            - self.reserved_for_system
            - self.draft_model_reserved
            - self.kv_cache_reserved
        )
        self.real_kv_alloc = os.getenv("NMOS_REAL_KV_ALLOC", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
        self.draft_active = False

        # page_id -> metadata (LRU order by insertion/touch)
        self.pages = OrderedDict()

    def set_draft_active(self, enabled):
        self.draft_active = bool(enabled)

    def get_vram_status(self):
        """Returns estimated real-time VRAM status."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            if allocated < 10 and self.draft_active:
                allocated = float(self.draft_model_reserved)
            used = max(allocated, reserved)
        else:
            used = float(self.draft_model_reserved if self.draft_active else 0)

        free_mb = max(0.0, float(self.total_vram_mb) - used)
        return {
            "allocated_mb": used,
            "free_mb": free_mb,
            "cuda_available": torch.cuda.is_available(),
        }

    def allocate_kv_page(self, page_id):
        """Allocates or refreshes a KV page metadata entry."""
        if page_id in self.pages:
            page = self.pages.pop(page_id)
            page["last_used"] = time.time()
            self.pages[page_id] = page
            return page

        if len(self.pages) >= self.max_pages:
            self.apply_h2o_folding()

        page = {
            "id": page_id,
            "allocated_at": time.time(),
            "last_used": time.time(),
            "size_mb": self.page_size_mb,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        if self.real_kv_alloc and torch.cuda.is_available():
            page["tensor"] = torch.zeros(8 * 1024 * 1024, dtype=torch.float16, device="cuda")

        self.pages[page_id] = page
        return page

    def apply_h2o_folding(self):
        """
        Heavy Hitter Oracle (H2O) folding:
        evict least recently used page under pressure.
        """
        if not self.pages:
            return None
        oldest_page_id, oldest_page = self.pages.popitem(last=False)
        print(f"[Memory] H2O Folding: Evicted Page {oldest_page_id} to save VRAM.")
        return oldest_page

    def estimate_required_pages(self, max_tokens):
        # Empirical prototype ratio: ~1 KV page per 64 generated tokens.
        pages = max(1, int((max(1, int(max_tokens)) + 63) // 64))
        return min(self.max_pages, pages)

    def reserve_for_generation(self, max_tokens):
        required_pages = self.estimate_required_pages(max_tokens)
        shortfall = required_pages - len(self.pages)
        for idx in range(max(0, shortfall)):
            page_id = f"kv-{int(time.time() * 1000)}-{idx}"
            self.allocate_kv_page(page_id)
        return {
            "required_pages": required_pages,
            "resident_pages": len(self.pages),
            "page_size_mb": self.page_size_mb,
        }

    def can_stream_layer(self, layer_size_mb):
        """Checks if River can safely stream the next shard."""
        status = self.get_vram_status()
        reclaimable_mb = len(self.pages) * self.page_size_mb
        effective_free_mb = status["free_mb"] + (0.5 * reclaimable_mb)
        return (effective_free_mb - float(layer_size_mb)) > 100.0

    def print_memory_map(self):
        print("\n--- NMOS VRAM MEMORY MAP ---")
        print(f"Total GPU VRAM:   {self.total_vram_mb} MB")
        print(f"Draft Model:      {self.draft_model_reserved if self.draft_active else 0} MB")
        print(f"KV-Cache Pool:    {self.kv_cache_reserved} MB ({len(self.pages)}/{self.max_pages} pages)")
        print(f"Streaming Window: {self.action_window_mb} MB")
        print("----------------------------\n")
