import os
import time
import threading
from collections import deque


class NMOSRiver(threading.Thread):
    def __init__(self, shard_size_mb=325, warm_chunk_mb=8):
        super().__init__()
        self.shard_size_mb = shard_size_mb
        self.warm_chunk_mb = max(1, int(warm_chunk_mb))
        self.shard_queue = deque()  # Queue of prewarm jobs
        self.enqueued_ids = set()
        self.queue_lock = threading.Lock()
        self.last_warm_stats = {}
        self.is_running = True
        self.daemon = True  # Keep running in background

    def queue_shard(self, shard_id, ssd_path, max_bytes=None, offset_bytes=0):
        """
        Queue a file prewarm request.
        max_bytes controls how much of the file to touch in page cache.
        offset_bytes controls where prewarm starts in the file.
        """
        with self.queue_lock:
            if shard_id in self.enqueued_ids:
                return
            self.shard_queue.append((shard_id, ssd_path, max_bytes, offset_bytes))
            self.enqueued_ids.add(shard_id)

    def _warm_file_pages(self, file_path, max_bytes, offset_bytes):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing shard source: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size <= 0:
            raise ValueError(f"Empty shard source: {file_path}")

        start = max(0, int(offset_bytes))
        if start >= file_size:
            start = start % file_size

        remaining = file_size - start
        target_bytes = min(remaining, max_bytes)
        chunk_size = self.warm_chunk_mb * 1024 * 1024
        warmed = 0

        # Touch file pages to encourage OS page cache residency.
        with open(file_path, "rb", buffering=0) as f:
            f.seek(start)
            while warmed < target_bytes:
                to_read = min(chunk_size, target_bytes - warmed)
                data = f.read(to_read)
                if not data:
                    break
                warmed += len(data)

        return warmed

    def run(self):
        """Main prefetcher loop: warm model file ranges in OS page cache."""
        while self.is_running:
            job = None
            with self.queue_lock:
                if self.shard_queue:
                    job = self.shard_queue.popleft()

            if job:
                shard_id, ssd_path, max_bytes, offset_bytes = job
                default_bytes = self.shard_size_mb * 1024 * 1024
                target_bytes = default_bytes if max_bytes is None else max(1, int(max_bytes))
                try:
                    warmed = self._warm_file_pages(ssd_path, target_bytes, offset_bytes)
                    self.last_warm_stats[shard_id] = {
                        "path": ssd_path,
                        "offset_bytes": int(offset_bytes),
                        "bytes": warmed,
                        "target_bytes": target_bytes,
                        "timestamp": time.time(),
                    }
                except Exception as exc:
                    print(f"[River] Prefetch failed for {shard_id}: {exc}")
                finally:
                    with self.queue_lock:
                        self.enqueued_ids.discard(shard_id)
            else:
                time.sleep(0.05)  # Idle wait

    def stop(self):
        self.is_running = False
        self.join()
