import time
import threading
import torch
import os
import numpy as np

class NMOSRiver(threading.Thread):
    def __init__(self, vram_limit_mb=2500, shard_size_mb=325):
        super().__init__()
        self.vram_limit_mb = vram_limit_mb
        self.shard_size_mb = shard_size_mb
        self.shard_queue = []
        self.active_vram_buffer = {}
        self.staging_ram_buffer = {}
        self.is_running = True
        self.daemon = True 

    def queue_shard(self, shard_id, ssd_path):
        """Add a shard to the prefetch queue."""
        if shard_id not in self.active_vram_buffer and shard_id not in self.staging_ram_buffer:
            self.shard_queue.append((shard_id, ssd_path))

    def run(self):
        """Main prefetcher loop: SSD -> RAM -> VRAM (Double Buffering)"""
        while self.is_running:
            if self.shard_queue:
                shard_id, ssd_path = self.shard_queue.pop(0)
                if not os.path.exists(ssd_path):
                    print(f"[River] Missing shard source: {ssd_path}")
                    continue
                
                max_bytes = self.shard_size_mb * 1024 * 1024
                with open(ssd_path, "rb") as f:
                    shard_data = f.read(max_bytes)
                if not shard_data:
                    print(f"[River] Empty shard read from: {ssd_path}")
                    continue
                self.staging_ram_buffer[shard_id] = shard_data
                
                numpy_shard = np.frombuffer(shard_data, dtype=np.uint8).copy()
                gpu_tensor = torch.from_numpy(numpy_shard).to("cuda")
                self.active_vram_buffer[shard_id] = gpu_tensor

                
                del self.staging_ram_buffer[shard_id]
            else:
                time.sleep(0.1) 

    def stop(self):
        self.is_running = False
        self.join()
