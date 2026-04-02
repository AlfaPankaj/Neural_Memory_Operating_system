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
    def __init__(self, oracle_path, draft_path):
        oracle_mode = os.getenv("NMOS_ORACLE_MODE", "local").strip().lower()
        remote_url = os.getenv("NMOS_REMOTE_URL", "").strip()
        remote_model = os.getenv("NMOS_REMOTE_MODEL", "").strip()
        remote_api_key = os.getenv("NMOS_REMOTE_API_KEY", "").strip()
        local_gpu_layers = int(os.getenv("NMOS_LOCAL_GPU_LAYERS", "10"))
        enable_speculative = os.getenv("NMOS_ENABLE_SPECULATIVE", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}

        self.engine = NMOSEngine(
            oracle_path=oracle_path,
            draft_path=draft_path,
            oracle_mode=oracle_mode,
            remote_url=remote_url,
            remote_model=remote_model,
            remote_api_key=remote_api_key,
            local_gpu_layers=local_gpu_layers,
            enable_speculative=enable_speculative,
        )
        self.runtime_label = self.engine.get_runtime_label()
        self.current_input = ""
        self.is_typing = False
        self.last_type_time = time.time()
        self.exit_flag = False

    def typing_monitor(self):
        """Background thread to trigger the Scout every 500ms during typing."""
        while not self.exit_flag:
            if self.is_typing and len(self.current_input) > 2:
                intent = self.engine.process_typing(self.current_input)
                
                if self.is_typing:
                    sys.stdout.write(f"\n\033[F\033[K[BRAIN] Scout: {intent} | River: Prefetching Shards... | VRAM: {self.engine.memory.get_vram_status()['free_mb']:.0f}MB Free\n")
                    sys.stdout.write(f"NMOS> {self.current_input}")
                    sys.stdout.flush()
                
                self.is_typing = False 
                time.sleep(0.5)
            else:
                time.sleep(0.1)

    def run(self):
        print("\n" + "="*50)
        print("   NEURAL MEMORY OPERATING SYSTEM (NMOS) v1.0")
        print("="*50)
        print(f"Hardware: RTX 2050 (4GB) | Engine: {self.runtime_label}")
        print("Status: Predictive Pre-fetching ACTIVE")
        print("--------------------------------------------------")
        print("(Type your prompt and hit ENTER. Type 'exit' to quit.)\n")

        monitor_thread = threading.Thread(target=self.typing_monitor, daemon=True)
        monitor_thread.start()

        while not self.exit_flag:
            sys.stdout.write("NMOS> ")
            sys.stdout.flush()
            self.current_input = ""
            
            while True:
                if msvcrt.kbhit():
                    char = msvcrt.getch()
                    if char == b'\r': 
                        print()
                        self.is_typing = False
                        
                        if len(self.current_input) > 2:
                            intent = self.engine.process_typing(self.current_input)
                            sys.stdout.write(f"\033[F\033[K[BRAIN] Scout: {intent} | River: Prefetching Shards... | VRAM: {self.engine.memory.get_vram_status()['free_mb']:.0f}MB Free\n")
                            sys.stdout.write(f"NMOS> {self.current_input}\n")
                            sys.stdout.flush()
                        break
                    elif char == b'\x08': 
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
                        except:
                            pass
            
            prompt = self.current_input.strip()
            if prompt.lower() in ["exit", "quit"]:
                self.exit_flag = True
                break
            
            if not prompt:
                continue

            print(f"[NMOS] Processing ({self.runtime_label})...")
            for token in self.engine.generate(prompt):
                sys.stdout.write(token)
                sys.stdout.flush()
            print("\n" + "-"*30 + "\n")

        self.engine.shutdown()
        print("[NMOS] System Shutdown Complete.")

if __name__ == "__main__":
    ORACLE = "Qwen2.5-72B-Instruct-Q3_K_M.gguf"
    DRAFT = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    
    print("\n[BOOT] Starting NMOS Core Sequence...")
    try:
        shell = NMOSShell(ORACLE, DRAFT)
        shell.run()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Boot Sequence Aborted: {e}")
