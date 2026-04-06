import json
import os
import threading
import time
import urllib.error
import urllib.request
from llama_cpp import Llama
from .scout import NMOSScout
from .river import NMOSRiver
from .memory import NMOSMemoryController
from .failure_memory import NMOSFailureMemory

class NMOSEngine:
    """
    Orchestrates optional Scout/River and Oracle execution.
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
        local_n_ctx=2048,
        local_n_batch=512,
        enable_speculative=False,
        enable_scout=True,
        enable_river=True,
        local_reply_mode="oracle_only",
        fast_reply_tokens=24,
        enable_failure_memory=True,
        failure_memory_path="nmos_failure_memory.json",
        failure_memory_similarity=0.78,
        failure_memory_max_events=2000,
    ):
        self.oracle_mode = (oracle_mode or "local").strip().lower()
        if self.oracle_mode not in {"local", "remote"}:
            raise ValueError("oracle_mode must be 'local' or 'remote'.")

        self.remote_url = (remote_url or "").strip()
        self.remote_model = (remote_model or "").strip()
        self.remote_api_key = (remote_api_key or "").strip()
        self.local_gpu_layers = max(0, int(local_gpu_layers))
        self.local_n_ctx = max(256, int(local_n_ctx))
        self.local_n_batch = max(32, int(local_n_batch))
        self.enable_speculative = bool(enable_speculative) and self.oracle_mode == "local"
        self.enable_scout = bool(enable_scout) and self.oracle_mode == "local"
        self.enable_river = bool(enable_river) and self.oracle_mode == "local"
        self.local_reply_mode = (local_reply_mode or "oracle_only").strip().lower()
        self.fast_reply_tokens = max(1, int(fast_reply_tokens))
        self.enable_failure_memory = bool(enable_failure_memory)
        self.failure_memory_path = (failure_memory_path or "nmos_failure_memory.json").strip()
        self.failure_memory_similarity = float(failure_memory_similarity)
        self.failure_memory_max_events = max(100, int(failure_memory_max_events))
        self.oracle_warm_started = False
        self.oracle_warm_lock = threading.Lock()
        self._oracle_load_thread = None
        self.last_partial_text = ""
        self.last_partial_intent = None
        self._memory_pressure_warned = False
        self._intent_profiles = {
            "CODE": {"offset_mb": 0, "window_mb": 448},
            "DOCS": {"offset_mb": 896, "window_mb": 384},
            "CHAT": {"offset_mb": 1728, "window_mb": 320},
        }

        if self.local_reply_mode not in {"oracle_only", "draft_then_oracle"}:
            raise ValueError("local_reply_mode must be 'oracle_only' or 'draft_then_oracle'.")

        if self.oracle_mode == "remote":
            if not self.remote_url:
                raise ValueError("Remote mode requires NMOS_REMOTE_URL.")
            if not self.remote_model:
                raise ValueError("Remote mode requires NMOS_REMOTE_MODEL.")

        print("[DEBUG] Step 1: Initializing Memory Controller...")
        self.memory = NMOSMemoryController()
        
        print("[DEBUG] Step 2: Initializing River Prefetcher...")
        self.river = NMOSRiver() if self.enable_river else None
        self.river_started = False
        
        print("[DEBUG] Step 3: Initializing Scout Classifier...")
        self.scout = NMOSScout() if self.enable_scout else None
        self.failure_memory = None
        self.oracle_path = oracle_path
        self.draft_path = draft_path
        self.oracle = None
        self.draft = None

        if self.oracle_mode == "local" and os.path.exists(self.oracle_path):
            oracle_size_gb = os.path.getsize(self.oracle_path) / (1024**3)
            if oracle_size_gb >= 30 and self.local_gpu_layers <= 12:
                print(
                    "[WARN] Local 72B monolithic GGUF is larger than typical system RAM on this class of machine. "
                    "Expect very high TTFT/token latency. Use remote mode for production-speed 70B behavior."
                )

        draft_needed = self.enable_speculative or (
            self.oracle_mode == "local" and self.local_reply_mode == "draft_then_oracle"
        )
        if draft_needed:
            print(f"[DEBUG] Step 4: Loading Draft Model ({draft_path})...")
            self.draft = Llama(
                model_path=draft_path,
                n_gpu_layers=-1,
                n_ctx=2048,
                verbose=False,
            )
            print("[DEBUG] Draft Model Loaded Successfully.")
        else:
            print("[DEBUG] Step 4: Speculative Draft DISABLED (stable mode).")

        self.memory.set_draft_active(self.draft is not None)

        if self.enable_scout and self.enable_failure_memory:
            self.failure_memory = NMOSFailureMemory(
                db_path=self.failure_memory_path,
                similarity_threshold=self.failure_memory_similarity,
                max_events=self.failure_memory_max_events,
            )
            stats = self.failure_memory.stats()
            print(f"[DEBUG] Step 4b: Failure Memory Loaded ({stats['events']} events).")
        else:
            print("[DEBUG] Step 4b: Failure Memory DISABLED.")

        if self.oracle_mode == "local" and self.enable_river and self.river is not None:
            print(f"[DEBUG] Step 5: Oracle deferred ({oracle_path})")
            print("[DEBUG] Step 6: Starting River Thread...")
            self.river.start()
            self.river_started = True
        elif self.oracle_mode == "local":
            print(f"[DEBUG] Step 5: Oracle deferred ({oracle_path})")
            print("[DEBUG] Step 6: River Thread DISABLED (proof mode).")
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
            base = "Local 72B + Draft (Speculative)"
        elif self.local_reply_mode == "draft_then_oracle":
            base = "Local 72B (Draft-Then-Oracle)"
        elif not self.enable_scout and not self.enable_river:
            base = "Local 72B (Proof Mode)"
        else:
            base = "Local 72B (Stable Non-Speculative)"
        return f"{base} | GPU={self.local_gpu_layers} | ctx={self.local_n_ctx}"

    def typing_pipeline_enabled(self):
        return self.enable_scout and self.scout is not None

    def prepare_before_queries(self):
        """
        Blocking startup warmup so Oracle is fully ready before first prompt.
        """
        if self.oracle_mode != "local":
            return
        self._ensure_oracle_loaded()

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
            "n_ctx": self.local_n_ctx,
            "n_batch": self.local_n_batch,
            "use_mmap": True,
            "verbose": False,
        }
        if self.enable_speculative:
            llama_kwargs["draft_model"] = self.draft

        self.oracle = Llama(**llama_kwargs)
        print("[DEBUG] Oracle ONLINE.")

    def maybe_start_oracle_prewarm(self, partial_text):
        """
        Start low-risk prewarm work during typing:
          1) page-cache warm on oracle GGUF through River
          2) optional background oracle model load trigger
        """
        if self.oracle_mode != "local":
            return

        if self.enable_river and self.river is not None:
            if self.memory.can_stream_layer(self.river.shard_size_mb):
                # Warm first slice while typing; bounded to reduce contention.
                self.river.queue_shard(
                    shard_id="oracle-prewarm-window",
                    ssd_path=self.oracle_path,
                    max_bytes=512 * 1024 * 1024,
                    offset_bytes=0,
                )
                self._memory_pressure_warned = False
            elif not self._memory_pressure_warned:
                self._memory_pressure_warned = True
                print("[Memory] River prewarm paused due to VRAM pressure.")

        # Trigger background oracle load only after meaningful typing starts.
        if self.oracle is None and len(partial_text.strip()) >= 12:
            with self.oracle_warm_lock:
                if not self.oracle_warm_started:
                    self.oracle_warm_started = True
                    self._oracle_load_thread = threading.Thread(
                        target=self._ensure_oracle_loaded,
                        daemon=True,
                    )
                    self._oracle_load_thread.start()
                    print("[NMOS] Oracle prewarm started in background.")

    @staticmethod
    def _override_ranking(ranking, override_intent):
        ordered = []
        added = set()
        ordered.append((override_intent, 1.0))
        added.add(override_intent)
        for intent, score in ranking:
            if intent in added:
                continue
            ordered.append((intent, score))
            added.add(intent)
        return ordered

    def _route_prefetch_for_ranked_intents(self, ranked_intents):
        if self.oracle_mode != "local" or not self.enable_river or self.river is None:
            return
        if not self.memory.can_stream_layer(self.river.shard_size_mb):
            return

        for rank, (intent, confidence) in enumerate(ranked_intents[:3]):
            profile = self._intent_profiles.get(intent)
            if not profile:
                continue
            window_mb = max(64, int(profile["window_mb"] * max(0.2, confidence)))
            offset_mb = profile["offset_mb"] + (rank * 128)
            self.river.queue_shard(
                shard_id=f"expert-{intent.lower()}-r{rank}",
                ssd_path=self.oracle_path,
                max_bytes=window_mb * 1024 * 1024,
                offset_bytes=offset_mb * 1024 * 1024,
            )

    def process_typing(self, partial_text):
        """Called every 500ms while user is typing."""
        if self.oracle_mode == "local":
            self.maybe_start_oracle_prewarm(partial_text)

        if not self.typing_pipeline_enabled():
            return "SCOUT_OFF"

        ranked = self.scout.predict_topk(partial_text, k=3)
        intent = ranked[0][0]

        if self.failure_memory is not None:
            override = self.failure_memory.suggest_override(partial_text, intent)
            if override is not None:
                intent = override["intent"]
                ranked = self._override_ranking(ranked, intent)

        self.last_partial_text = partial_text
        self.last_partial_intent = intent
        self._route_prefetch_for_ranked_intents(ranked)
        return intent

    def finalize_prompt_intent(self, full_prompt):
        """
        Called on prompt submission to compare final intent vs typing-stage intent.
        Mismatches are persisted to failure memory for future override.
        """
        if not self.typing_pipeline_enabled():
            return "SCOUT_OFF"

        final_intent = self.scout.predict_intent(full_prompt)
        if (
            self.failure_memory is not None
            and self.last_partial_text
            and self.last_partial_intent in {"CODE", "CHAT", "DOCS"}
            and final_intent in {"CODE", "CHAT", "DOCS"}
            and self.last_partial_intent != final_intent
        ):
            recorded = self.failure_memory.record_misprediction(
                partial_text=self.last_partial_text,
                predicted_intent=self.last_partial_intent,
                corrected_intent=final_intent,
                full_prompt=full_prompt,
            )
            if recorded:
                print(
                    f"[FailureMemory] Learned correction: "
                    f"{self.last_partial_intent} -> {final_intent}"
                )

        self.clear_typing_state()
        return final_intent

    def clear_typing_state(self):
        self.last_partial_text = ""
        self.last_partial_intent = None

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
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "NMOS/1.0 (Windows; Python urllib)",
        }
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
        formatted_prompt = self._build_local_chat_prompt(prompt)
        output = self.oracle.create_completion(
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
            stream=True,
        )
        for chunk in output:
            yield chunk["choices"][0]["text"]

    def _stream_local_draft(self, prompt, max_tokens):
        if self.draft is None:
            raise RuntimeError("Draft model is not loaded.")
        formatted_prompt = self._build_local_chat_prompt(prompt)
        output = self.draft.create_completion(
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
            stream=True,
        )
        for chunk in output:
            yield chunk["choices"][0]["text"]

    @staticmethod
    def _build_local_chat_prompt(user_prompt):
        return (
            "<|im_start|>system\n"
            "You are NMOS Oracle. Provide accurate, concise, high-quality answers.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @staticmethod
    def _build_refine_prompt(prompt, draft_text):
        return (
            "You are the 72B oracle model. Refine the draft answer for factual accuracy and clarity.\n"
            "Keep it concise and do not mention this instruction.\n\n"
            f"User prompt:\n{prompt}\n\n"
            f"Draft answer:\n{draft_text}\n\n"
            "Final refined answer:\n"
        )

    def generate_draft_reply(self, prompt, max_tokens=None):
        if self.local_reply_mode != "draft_then_oracle":
            raise RuntimeError("Draft reply generation is only available in draft_then_oracle mode.")
        draft_budget = self.fast_reply_tokens if max_tokens is None else max(1, int(max_tokens))
        for tok in self._stream_local_draft(prompt, draft_budget):
            yield tok

    def refine_draft_response(self, prompt, draft_text, max_tokens=100):
        refine_prompt = self._build_refine_prompt(prompt, draft_text)
        for tok in self._stream_local_oracle(refine_prompt, max_tokens):
            yield tok

    def generate(self, prompt, max_tokens=100):
        memory_plan = self.memory.reserve_for_generation(max_tokens)

        if self.oracle_mode == "remote":
            print("\n[NMOS] Executing Remote 72B Oracle...")
            token_stream = self._stream_remote_oracle(prompt, max_tokens)
            pipeline = "REMOTE-72B"
        else:
            if self.enable_speculative:
                print("\n[NMOS] Executing Speculative Tree (K=15)...")
                pipeline = "NMOS Speculative"
                token_stream = self._stream_local_oracle(prompt, max_tokens)
            elif self.local_reply_mode == "draft_then_oracle":
                print("\n[NMOS] Executing Draft-Then-Oracle pipeline...")
                pipeline = "NMOS Draft->Oracle"
                token_stream = self._stream_local_oracle_with_fast_reply(prompt, max_tokens)
            elif not self.enable_scout and not self.enable_river:
                print("\n[NMOS] Executing Local 72B Proof Mode...")
                pipeline = "NMOS Proof"
                token_stream = self._stream_local_oracle(prompt, max_tokens)
            else:
                print("\n[NMOS] Executing Stable Local Oracle...")
                pipeline = "NMOS Stable"
                token_stream = self._stream_local_oracle(prompt, max_tokens)

        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        for token in token_stream:
            if token:
                token_count += 1
                if first_token_time is None:
                    first_token_time = time.perf_counter()
            yield token

        elapsed = time.perf_counter() - start_time
        if first_token_time is not None and elapsed > 0:
            ttft_ms = (first_token_time - start_time) * 1000.0
            generation_elapsed = max(1e-9, time.perf_counter() - first_token_time)
            tps = token_count / generation_elapsed
            print(
                f"\n[Stats] TTFT: {ttft_ms:.2f} ms | Throughput: {tps:.2f} TPS | "
                f"Pipeline: {pipeline} | KV Pages: {memory_plan['resident_pages']}"
            )
        else:
            print(
                f"\n[Stats] No token output | Pipeline: {pipeline} | "
                f"KV Pages: {memory_plan['resident_pages']}"
            )

    def _stream_local_oracle_with_fast_reply(self, prompt, max_tokens):
        draft_budget = min(max_tokens, self.fast_reply_tokens)
        draft_tokens = []

        print(f"[NMOS] Fast draft reply ({draft_budget} tokens budget)...")
        for tok in self.generate_draft_reply(prompt, draft_budget):
            draft_tokens.append(tok)
            yield tok

        draft_text = "".join(draft_tokens).strip()
        yield "\n\n[NMOS] Oracle refining draft...\n"

        for tok in self.refine_draft_response(prompt, draft_text, max_tokens):
            yield tok

    def shutdown(self):
        self.oracle = None
        self.draft = None
        if self.river_started and self.river is not None:
            self.river.stop()
        print("[NMOS] Engine Offline.")

if __name__ == "__main__":
    pass
