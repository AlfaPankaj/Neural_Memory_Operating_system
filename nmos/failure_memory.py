import hashlib
import json
import math
import os
import time


class NMOSFailureMemory:
    """
    Persistent failure-memory store for Scout mispredictions.
    Uses lightweight hashed vectors with cosine similarity to retrieve
    prior correction patterns and optionally override Scout intent.
    """

    INTENTS = ("CODE", "CHAT", "DOCS")

    def __init__(
        self,
        db_path="nmos_failure_memory.json",
        vector_dim=256,
        similarity_threshold=0.78,
        max_events=2000,
        min_override_votes=1,
    ):
        self.db_path = db_path
        self.vector_dim = max(64, int(vector_dim))
        self.similarity_threshold = float(similarity_threshold)
        self.max_events = max(100, int(max_events))
        self.min_override_votes = max(1, int(min_override_votes))
        self.events = []
        self._load()

    def _load(self):
        if not os.path.exists(self.db_path):
            self.events = []
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError as exc:
            backup_path = f"{self.db_path}.corrupt-{int(time.time())}"
            os.replace(self.db_path, backup_path)
            print(
                f"[FailureMemory] Corrupt DB moved to '{backup_path}'. "
                f"Starting with an empty memory. Error: {exc}"
            )
            self.events = []
            return

        if not isinstance(raw, list):
            print("[FailureMemory] Invalid DB format, expected a JSON list. Starting empty.")
            self.events = []
            return

        normalized = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            predicted = str(item.get("predicted_intent", "")).upper()
            corrected = str(item.get("corrected_intent", "")).upper()
            vector = item.get("vector", [])
            if predicted not in self.INTENTS or corrected not in self.INTENTS:
                continue
            if not isinstance(vector, list) or len(vector) != self.vector_dim:
                continue
            normalized.append(
                {
                    "timestamp": float(item.get("timestamp", time.time())),
                    "partial_text": str(item.get("partial_text", "")),
                    "full_prompt": str(item.get("full_prompt", "")),
                    "predicted_intent": predicted,
                    "corrected_intent": corrected,
                    "vector": [float(v) for v in vector],
                }
            )
        self.events = normalized[-self.max_events :]

    def _persist(self):
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        tmp_path = f"{self.db_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, ensure_ascii=True, indent=2)
        os.replace(tmp_path, self.db_path)

    @staticmethod
    def _tokenize(text):
        tokens = []
        current = []
        for ch in (text or "").lower():
            if ch.isalnum() or ch in {"_", "-", "."}:
                current.append(ch)
            elif current:
                tokens.append("".join(current))
                current = []
        if current:
            tokens.append("".join(current))

        # Include bigrams for stronger phrase matching.
        bigrams = [f"{tokens[i]}::{tokens[i + 1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def _vectorize(self, text):
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.vector_dim

        vector = [0.0] * self.vector_dim
        for tok in tokens:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
            idx = int.from_bytes(digest, byteorder="big", signed=False) % self.vector_dim
            vector[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vector))
        if norm <= 0:
            return [0.0] * self.vector_dim
        return [v / norm for v in vector]

    @staticmethod
    def _cosine_similarity(vec_a, vec_b):
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def record_misprediction(self, partial_text, predicted_intent, corrected_intent, full_prompt=""):
        predicted = str(predicted_intent or "").upper()
        corrected = str(corrected_intent or "").upper()
        if predicted not in self.INTENTS or corrected not in self.INTENTS:
            raise ValueError("Intent must be one of CODE, CHAT, DOCS.")
        if predicted == corrected:
            return False

        source_text = partial_text.strip() if partial_text else full_prompt.strip()
        if not source_text:
            return False

        event = {
            "timestamp": time.time(),
            "partial_text": partial_text or "",
            "full_prompt": full_prompt or "",
            "predicted_intent": predicted,
            "corrected_intent": corrected,
            "vector": self._vectorize(source_text),
        }
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]
        self._persist()
        return True

    def suggest_override(self, partial_text, predicted_intent, top_n=8):
        predicted = str(predicted_intent or "").upper()
        if predicted not in self.INTENTS:
            return None

        query_vector = self._vectorize(partial_text)
        if not any(query_vector):
            return None

        candidates = []
        for event in self.events:
            if event["predicted_intent"] != predicted:
                continue
            similarity = self._cosine_similarity(query_vector, event["vector"])
            if similarity >= self.similarity_threshold:
                candidates.append((similarity, event))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_matches = candidates[: max(1, int(top_n))]
        aggregate = {}
        for similarity, event in top_matches:
            corrected = event["corrected_intent"]
            aggregate.setdefault(corrected, {"score": 0.0, "hits": 0})
            aggregate[corrected]["score"] += similarity
            aggregate[corrected]["hits"] += 1

        best_intent = None
        best_score = -1.0
        best_hits = 0
        for intent, payload in aggregate.items():
            if payload["score"] > best_score:
                best_intent = intent
                best_score = payload["score"]
                best_hits = payload["hits"]

        if best_intent is None or best_hits < self.min_override_votes:
            return None

        mean_similarity = best_score / max(1, best_hits)
        return {
            "intent": best_intent,
            "mean_similarity": mean_similarity,
            "hits": best_hits,
        }

    def stats(self):
        corrections = {}
        for event in self.events:
            key = f"{event['predicted_intent']}->{event['corrected_intent']}"
            corrections[key] = corrections.get(key, 0) + 1
        return {
            "events": len(self.events),
            "similarity_threshold": self.similarity_threshold,
            "corrections": corrections,
        }
