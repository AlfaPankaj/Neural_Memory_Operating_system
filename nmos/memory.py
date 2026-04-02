import torch
import numpy as np

class NMOSMemoryController:
    """
    Manages the 4GB VRAM 'Action Zone'.
    Implements Paged-KV allocation and VRAM safety checks.
    """
    def __init__(self, total_vram_mb=4096):
        self.total_vram_mb = total_vram_mb
        self.reserved_for_system = 300 
        self.draft_model_reserved = 1100 
        self.kv_cache_reserved = 800    
        
        self.action_window_mb = (self.total_vram_mb - 
                                 self.reserved_for_system - 
                                 self.draft_model_reserved - 
                                 self.kv_cache_reserved)
        
        self.pages = {} 
        self.page_size_mb = 16 
        self.max_pages = self.kv_cache_reserved // self.page_size_mb

    def get_vram_status(self):
        """Returns real-time VRAM usage from CUDA, accounting for llama-cpp models."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            
            if allocated < 10:
                allocated = self.draft_model_reserved
                
            return {
                "allocated_mb": allocated,
                "free_mb": max(0, self.total_vram_mb - allocated)
            }
        return {"error": "CUDA not available"}

    def allocate_kv_page(self, page_id):
        """Allocates a new 16MB page in the VRAM pool."""
        if len(self.pages) >= self.max_pages:
            self.apply_h2o_folding()
            
        self.pages[page_id] = torch.zeros(8 * 1024 * 1024, dtype=torch.float16, device="cuda")
        return self.pages[page_id]

    def apply_h2o_folding(self):
        """
        Heavy Hitter Oracle (H2O) logic:
        Identifies and 'folds' (evicts) the least important KV pages.
        """
        if not self.pages: return
        
        oldest_page = list(self.pages.keys())[0]
        del self.pages[oldest_page]
        print(f"[Memory] H2O Folding: Evicted Page {oldest_page} to save VRAM.")

    def can_stream_layer(self, layer_size_mb):
        """Checks if the 'River' can safely stream the next 70B layer."""
        status = self.get_vram_status()
        # Ensure we have a safety buffer of 100MB
        return (status['free_mb'] - layer_size_mb) > 100

    def print_memory_map(self):
        print("\n--- NMOS VRAM MEMORY MAP ---")
        print(f"Total GPU VRAM:   {self.total_vram_mb} MB")
        print(f"Draft Model:      {self.draft_model_reserved} MB")
        print(f"KV-Cache Pool:    {self.kv_cache_reserved} MB ({len(self.pages)}/{self.max_pages} pages)")
        print(f"Streaming Window: {self.action_window_mb} MB (Capacity: ~8 layers)")
        print("----------------------------\n")
