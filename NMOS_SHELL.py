import time
import msvcrt
import sys
import threading
import os
from nmos.engine import NMOSEngine

def clear_line():
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush()

class NMOSShell:
    @staticmethod
    def _get_system_ram_gb():
        try:
            if os.name == "nt":
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                memory_status = MEMORYSTATUSEX()
                memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
                    return memory_status.ullTotalPhys / (1024**3)
        except Exception:
            return None
        return None

    @staticmethod
    def _get_file_size_gb(path):
        if not os.path.exists(path):
            return None
        try:
            return os.path.getsize(path) / (1024**3)
        except OSError:
            return None

    def __init__(self, oracle_path, draft_path):
        oracle_mode = os.getenv("NMOS_ORACLE_MODE", "local").strip().lower()
        remote_url = os.getenv("NMOS_REMOTE_URL", "").strip()
        remote_model = os.getenv("NMOS_REMOTE_MODEL", "").strip()
        remote_api_key = os.getenv("NMOS_REMOTE_API_KEY", "").strip()
        local_gpu_layers = int(os.getenv("NMOS_LOCAL_GPU_LAYERS", "10"))
        local_n_ctx = int(os.getenv("NMOS_LOCAL_N_CTX", "2048"))
        local_n_batch = int(os.getenv("NMOS_LOCAL_N_BATCH", "512"))
        enable_speculative = os.getenv("NMOS_ENABLE_SPECULATIVE", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
        enable_scout = os.getenv("NMOS_ENABLE_SCOUT", "1").strip() in {"1", "true", "TRUE", "yes", "YES"}
        enable_river = os.getenv("NMOS_ENABLE_RIVER", "1").strip() in {"1", "true", "TRUE", "yes", "YES"}
        enable_failure_memory = os.getenv("NMOS_ENABLE_FAILURE_MEMORY", "1").strip() in {"1", "true", "TRUE", "yes", "YES"}
        adaptive_local_profile = os.getenv("NMOS_ADAPTIVE_LOCAL_PROFILE", "1").strip() in {"1", "true", "TRUE", "yes", "YES"}
        warmup_before_prompt = os.getenv("NMOS_WARMUP_BEFORE_PROMPT", "1").strip() in {"1", "true", "TRUE", "yes", "YES"}
        failure_memory_path = os.getenv("NMOS_FAILURE_MEMORY_PATH", "nmos_failure_memory.json").strip()
        failure_memory_similarity = float(os.getenv("NMOS_FAILURE_MEMORY_SIMILARITY", "0.78"))
        failure_memory_max_events = int(os.getenv("NMOS_FAILURE_MEMORY_MAX_EVENTS", "2000"))
        local_reply_mode = os.getenv("NMOS_LOCAL_REPLY_MODE", "oracle_only").strip().lower()
        fast_reply_tokens = int(os.getenv("NMOS_FAST_REPLY_TOKENS", "24"))

        self.auto_profile_note = None
        has_explicit_reply_mode = "NMOS_LOCAL_REPLY_MODE" in os.environ
        if oracle_mode == "local" and adaptive_local_profile and not has_explicit_reply_mode:
            oracle_size_gb = self._get_file_size_gb(oracle_path)
            ram_gb = self._get_system_ram_gb()
            if oracle_size_gb is not None and ram_gb is not None and oracle_size_gb > (ram_gb * 0.9):
                local_reply_mode = "draft_then_oracle"
                if "NMOS_FAST_REPLY_TOKENS" not in os.environ:
                    fast_reply_tokens = 32
                self.auto_profile_note = (
                    f"Adaptive profile enabled: switched to '{local_reply_mode}' "
                    f"(Oracle {oracle_size_gb:.1f}GB > RAM {ram_gb:.1f}GB)."
                )

        self.engine = NMOSEngine(
            oracle_path=oracle_path,
            draft_path=draft_path,
            oracle_mode=oracle_mode,
            remote_url=remote_url,
            remote_model=remote_model,
            remote_api_key=remote_api_key,
            local_gpu_layers=local_gpu_layers,
            local_n_ctx=local_n_ctx,
            local_n_batch=local_n_batch,
            enable_speculative=enable_speculative,
            enable_scout=enable_scout,
            enable_river=enable_river,
            local_reply_mode=local_reply_mode,
            fast_reply_tokens=fast_reply_tokens,
            enable_failure_memory=enable_failure_memory,
            failure_memory_path=failure_memory_path,
            failure_memory_similarity=failure_memory_similarity,
            failure_memory_max_events=failure_memory_max_events,
        )
        self.runtime_label = self.engine.get_runtime_label()
        self.typing_pipeline_enabled = self.engine.typing_pipeline_enabled()
        self.current_input = ""
        self.is_typing = False
        self.last_type_time = time.time()
        self.exit_flag = False
        self.pending_refinement_threads = []
        self.refine_lock = threading.Lock()
        self.warmup_before_prompt = warmup_before_prompt

    def typing_monitor(self):
        """Background thread to trigger the Scout every 500ms during typing."""
        while not self.exit_flag:
            if not self.typing_pipeline_enabled:
                time.sleep(0.2)
                continue

            # Only scout if we are actively typing and NOT generating (handled by is_typing flag)
            if self.is_typing and len(self.current_input) > 2:
                # Every 500ms of active typing, update the Scout's prediction
                intent = self.engine.process_typing(self.current_input)
                
                # NMOS DASHBOARD (Manual Verification)
                # Only print if we are still in typing mode
                if self.is_typing:
                    if self.engine.enable_river and self.engine.river is not None and self.engine.river.last_warm_stats:
                        last_key = next(reversed(self.engine.river.last_warm_stats))
                        warmed_mb = self.engine.river.last_warm_stats[last_key]["bytes"] / (1024 * 1024)
                        river_status = f"Warmed {warmed_mb:.0f}MB"
                    elif self.engine.enable_river:
                        river_status = "Prefetching..."
                    else:
                        river_status = "Disabled"
                    sys.stdout.write(f"\n\033[F\033[K[BRAIN] Scout: {intent} | River: {river_status} | VRAM: {self.engine.memory.get_vram_status()['free_mb']:.0f}MB Free\n")
                    sys.stdout.write(f"NMOS> {self.current_input}")
                    sys.stdout.flush()
                
                self.is_typing = False # Reset for next character
                time.sleep(0.5)
            else:
                time.sleep(0.1)

    def run(self):
        print("\n" + "="*50)
        print("   NEURAL MEMORY OPERATING SYSTEM (NMOS) v1.0")
        print("="*50)
        print(f"Hardware: RTX 2050 (4GB) | Engine: {self.runtime_label}")
        print("Status: Predictive Pre-fetching ACTIVE")
        if self.auto_profile_note:
            print(f"[NMOS] {self.auto_profile_note}")
        if self.engine.failure_memory is not None:
            fm_stats = self.engine.failure_memory.stats()
            print(f"Failure Memory: {fm_stats['events']} events loaded")
        if self.warmup_before_prompt and self.engine.oracle_mode == "local":
            print("[NMOS] Preparing Oracle before accepting queries...")
            self.engine.prepare_before_queries()
            print("[NMOS] Oracle ready. You can start querying.")
        print("--------------------------------------------------")
        print("(Type your prompt and hit ENTER. Type 'exit' to quit.)\n")

        # Start the background typing monitor
        monitor_thread = threading.Thread(target=self.typing_monitor, daemon=True)
        monitor_thread.start()

        while not self.exit_flag:
            sys.stdout.write("NMOS> ")
            sys.stdout.flush()
            self.current_input = ""
            
            while True:
                if msvcrt.kbhit():
                    char = msvcrt.getch()
                    if char == b'\r': # Enter
                        print()
                        self.is_typing = False # STOP BACKGROUND WORK
                         
                        # FINAL SCOUT SYNC: Update the status one last time for the full prompt
                        if self.typing_pipeline_enabled and len(self.current_input.strip()) > 2:
                            intent = self.engine.scout.predict_intent(self.current_input.strip())
                            vram = self.engine.memory.get_vram_status()
                            sys.stdout.write(f"\033[F\033[K[BRAIN] Scout: {intent} | River: Prefetching Shards... | VRAM: {vram['free_mb']:.0f}MB Free\n")
                            sys.stdout.write(f"NMOS> {self.current_input}\n")
                            sys.stdout.flush()
                        break
                    elif char == b'\x08': # Backspace
                        if len(self.current_input) > 0:
                            self.current_input = self.current_input[:-1]
                            sys.stdout.write('\b \b')
                            sys.stdout.flush()
                    else:
                        try:
                            c = char.decode('utf-8')
                            self.current_input += c
                            sys.stdout.write(c)
                            sys.stdout.flush()
                            self.is_typing = True
                            self.last_type_time = time.time()
                        except UnicodeDecodeError:
                            continue
            
            prompt = self.current_input.strip()
            if self.typing_pipeline_enabled and len(prompt) <= 2:
                self.engine.clear_typing_state()
            if prompt.lower() in ["exit", "quit"]:
                if self.typing_pipeline_enabled:
                    self.engine.clear_typing_state()
                self.exit_flag = True
                break
             
            if not prompt:
                continue

            if self.typing_pipeline_enabled and len(prompt) > 2:
                self.engine.finalize_prompt_intent(prompt)

            print(f"[NMOS] Processing ({self.runtime_label})...")
            if self.engine.oracle_mode == "local" and self.engine.local_reply_mode == "draft_then_oracle":
                draft_tokens = []
                for token in self.engine.generate_draft_reply(prompt):
                    draft_tokens.append(token)
                    sys.stdout.write(token)
                    sys.stdout.flush()

                draft_text = "".join(draft_tokens).strip()
                print("\n[NMOS] Draft reply done. Oracle refinement started in background.\n")

                t = threading.Thread(
                    target=self._run_background_refinement,
                    args=(prompt, draft_text),
                    daemon=True,
                )
                t.start()
                self.pending_refinement_threads.append(t)
                print("\n" + "-"*30 + "\n")
            else:
                for token in self.engine.generate(prompt):
                    sys.stdout.write(token)
                    sys.stdout.flush()
                print("\n" + "-"*30 + "\n")

        self._join_refinements()
        self.engine.shutdown()
        print("[NMOS] System Shutdown Complete.")

    def _run_background_refinement(self, prompt, draft_text):
        with self.refine_lock:
            try:
                sys.stdout.write("\n[NMOS][Background] Oracle refining previous answer...\n")
                sys.stdout.flush()
                for token in self.engine.refine_draft_response(prompt, draft_text, max_tokens=100):
                    sys.stdout.write(token)
                    sys.stdout.flush()
                sys.stdout.write("\n[NMOS][Background] Oracle refinement completed.\n")
                sys.stdout.write("-"*30 + "\n")
                sys.stdout.write("NMOS> ")
                sys.stdout.flush()
            except Exception as exc:
                sys.stdout.write(f"\n[NMOS][Background] Oracle refinement failed: {exc}\n")
                sys.stdout.write("NMOS> ")
                sys.stdout.flush()

    def _join_refinements(self):
        for t in self.pending_refinement_threads:
            if t.is_alive():
                t.join()

if __name__ == "__main__":
    ORACLE = "Qwen2.5-72B-Instruct-Q3_K_M.gguf"
    DRAFT = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    
    print("\n[BOOT] Starting NMOS Core Sequence...")
    try:
        shell = NMOSShell(ORACLE, DRAFT)
        shell.run()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Boot Sequence Aborted: {e}")
