import json
import time
import urllib.error
import urllib.request
from llama_cpp import Llama
from .scout import NMOSScout
from .river import NMOSRiver
from .memory import NMOSMemoryController

class NMOSEngine:
    """
    Orchestrates Scout, River, and Oracle execution.
    Supports:
      - local oracle mode (stable non-speculative by default)
      - remote oracle mode for mandatory 72B inference
    """
    def __init__(
        self,
        oracle_path,
        draft_path,
        oracle_mode="local",
        remote_url=None,
        remote_model=None,
        remote_api_key=None,
        local_gpu_layers=10,
        enable_speculative=False,
    ):
        self.oracle_mode = (oracle_mode or "local").strip().lower()
        if self.oracle_mode not in {"local", "remote"}:
            raise ValueError("oracle_mode must be 'local' or 'remote'.")

        self.remote_url = (remote_url or "").strip()
        self.remote_model = (remote_model or "").strip()
        self.remote_api_key = (remote_api_key or "").strip()
        self.local_gpu_layers = max(0, int(local_gpu_layers))
        self.enable_speculative = bool(enable_speculative) and self.oracle_mode == "local"

        if self.oracle_mode == "remote":
            if not self.remote_url:
                raise ValueError("Remote mode requires NMOS_REMOTE_URL.")
            if not self.remote_model:
                raise ValueError("Remote mode requires NMOS_REMOTE_MODEL.")

        print("[DEBUG] Step 1: Initializing Memory Controller...")
        self.memory = NMOSMemoryController()
        
        print("[DEBUG] Step 2: Initializing River Prefetcher...")
        self.river = NMOSRiver()
        self.river_started = False
        
        print("[DEBUG] Step 3: Initializing Scout Classifier...")
        self.scout = NMOSScout()
        self.oracle_path = oracle_path
        self.draft_path = draft_path
        self.oracle = None
        self.draft = None

        if self.enable_speculative:
            print(f"[DEBUG] Step 4: Loading Qwen Draft Model ({draft_path})...")
            self.draft = Llama(
                model_path=draft_path,
                n_gpu_layers=-1,
                n_ctx=2048,
                verbose=False,
            )
            print("[DEBUG] Draft Model Loaded Successfully.")
        else:
            print("[DEBUG] Step 4: Speculative Draft DISABLED (stable mode).")

        if self.oracle_mode == "local":
            print(f"[DEBUG] Step 5: Oracle deferred ({oracle_path})")
            print("[DEBUG] Step 6: Starting River Thread...")
            self.river.start()
            self.river_started = True
        else:
            print("[DEBUG] Step 5: Remote Oracle mode selected.")
            print("[DEBUG] Step 6: River Thread not required in remote mode.")

        print(f"[NMOS] Engine is Online. Mode: {self.get_runtime_label()}")

    def get_runtime_label(self):
        if self.oracle_mode == "remote":
            if self.remote_model:
                return f"Remote 72B ({self.remote_model})"
            return "Remote 72B (endpoint-managed model)"
        if self.enable_speculative:
            return "Local 72B + Draft (Speculative)"
        return "Local 72B (Stable Non-Speculative)"

    def _ensure_oracle_loaded(self):
        if self.oracle_mode != "local" or self.oracle is not None:
            return
        if self.enable_speculative:
            print(f"[DEBUG] Initializing 72B Oracle (Speculative): {self.oracle_path}")
        else:
            print(f"[DEBUG] Initializing 72B Oracle (Stable): {self.oracle_path}")
        print("[NOTE] This step may take 60-120 seconds.")

        llama_kwargs = {
            "model_path": self.oracle_path,
            "n_gpu_layers": self.local_gpu_layers,
            "n_ctx": 2048,
            "use_mmap": True,
            "verbose": False,
        }
        if self.enable_speculative:
            llama_kwargs["draft_model"] = self.draft

        self.oracle = Llama(**llama_kwargs)
        print("[DEBUG] Oracle ONLINE.")

    def process_typing(self, partial_text):
        """Called every 500ms while user is typing."""
        intent = self.scout.predict_intent(partial_text)

        if self.oracle_mode == "local":
            if intent == "CODE":
                self.river.queue_shard("logic_shard_1", self.oracle_path)
            elif intent == "CHAT":
                self.river.queue_shard("style_shard_1", self.oracle_path)
             
        return intent

    def _remote_chat_endpoint(self):
        base = self.remote_url.rstrip("/")
        if base.endswith("/v1/chat/completions"):
            return base
        return f"{base}/v1/chat/completions"

    @staticmethod
    def _extract_token_from_openai_payload(payload):
        choices = payload.get("choices") or []
        if not choices:
            return ""

        choice = choices[0]

        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content

        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

        text = choice.get("text")
        if isinstance(text, str):
            return text

        return ""

    def _stream_remote_oracle(self, prompt, max_tokens):
        endpoint = self._remote_chat_endpoint()
        payload = {
            "model": self.remote_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}
        if self.remote_api_key:
            headers["Authorization"] = f"Bearer {self.remote_api_key}"

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                content_type = (response.headers.get("Content-Type") or "").lower()

                if "text/event-stream" in content_type:
                    for raw_line in response:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        token = self._extract_token_from_openai_payload(chunk)
                        if token:
                            yield token
                else:
                    body = response.read().decode("utf-8", errors="ignore")
                    response_payload = json.loads(body) if body else {}
                    token = self._extract_token_from_openai_payload(response_payload)
                    if token:
                        yield token
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Remote oracle HTTP {exc.code}: {error_body or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Remote oracle connection failed: {exc.reason}") from exc

    def _stream_local_oracle(self, prompt, max_tokens):
        self._ensure_oracle_loaded()
        output = self.oracle.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in output:
            yield chunk["choices"][0]["text"]

    def generate(self, prompt, max_tokens=100):
        if self.oracle_mode == "remote":
            print("\n[NMOS] Executing Remote 72B Oracle...")
            token_stream = self._stream_remote_oracle(prompt, max_tokens)
            pipeline = "REMOTE-72B"
        else:
            if self.enable_speculative:
                print("\n[NMOS] Executing Speculative Tree (K=15)...")
                pipeline = "NMOS Speculative"
            else:
                print("\n[NMOS] Executing Stable Local Oracle...")
                pipeline = "NMOS Stable"
            token_stream = self._stream_local_oracle(prompt, max_tokens)

        start_time = time.perf_counter()
        token_count = 0
        for token in token_stream:
            token_count += 1
            yield token

        elapsed = time.perf_counter() - start_time
        tps = token_count / elapsed if elapsed > 0 else 0
        print(f"\n[Stats] Throughput: {tps:.2f} TPS | Pipeline: {pipeline}")

    def shutdown(self):
        self.oracle = None
        self.draft = None
        if self.river_started:
            self.river.stop()
        print("[NMOS] Engine Offline.")

if __name__ == "__main__":
    pass
