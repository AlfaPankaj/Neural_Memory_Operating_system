import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class NMOSScout:
    INTENTS = ("CODE", "CHAT", "DOCS")

    def __init__(self, model_name="HuggingFaceTB/SmolLM2-135M-Instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
        self.current_intent = "CHAT"
        self._slm_error_logged = False

        self.code_keywords = (
            "python",
            "script",
            "code",
            "function",
            "implement",
            "binary",
            "c++",
            "sql",
            "calculate",
            "math",
            "algorithm",
            "debug",
            "stack",
            "compiler",
            "query",
        )
        self.docs_keywords = (
            "summarize",
            "extract",
            "pdf",
            "csv",
            "data",
            "text",
            "analysis",
            "table",
            "convert",
            "document",
            "report",
            "spreadsheet",
            "json",
            "metrics",
        )
        self.chat_keywords = (
            "story",
            "joke",
            "chat",
            "talk",
            "hello",
            "hi",
            "poem",
            "creative",
        )

    @staticmethod
    def _normalize(scores):
        stable = {intent: max(0.01, float(value)) for intent, value in scores.items()}
        total = sum(stable.values())
        return {intent: value / total for intent, value in stable.items()}

    def _heuristic_scores(self, text_lower):
        scores = {intent: 0.1 for intent in self.INTENTS}
        scores[self.current_intent] += 0.2

        for kw in self.code_keywords:
            if kw in text_lower:
                scores["CODE"] += 0.9
        for kw in self.docs_keywords:
            if kw in text_lower:
                scores["DOCS"] += 0.9
        for kw in self.chat_keywords:
            if kw in text_lower:
                scores["CHAT"] += 0.7

        return scores

    def _predict_with_slm(self, text):
        if len(text.strip()) < 10:
            return None

        prompt = (
            "<|im_start|>system\n"
            "You are the NMOS Shard Router. Output ONLY CODE, CHAT, or DOCS."
            "<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        except (RuntimeError, ValueError, OSError) as exc:
            if not self._slm_error_logged:
                self._slm_error_logged = True
                print(f"[Scout] SLM inference degraded, using heuristics only: {exc}")
            return None

        input_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[0][input_len:]
        if len(gen_tokens) == 0:
            return None

        prediction = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).upper()
        if "CODE" in prediction:
            return "CODE"
        if "DOCS" in prediction or "SUMMAR" in prediction:
            return "DOCS"
        return "CHAT"

    def predict_topk(self, text, k=3):
        text_lower = (text or "").lower()
        scores = self._heuristic_scores(text_lower)
        slm_vote = self._predict_with_slm(text)
        if slm_vote in self.INTENTS:
            scores[slm_vote] += 1.2

        normalized = self._normalize(scores)
        ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
        self.current_intent = ranked[0][0]
        return ranked[: max(1, min(int(k), len(ranked)))]

    def predict_intent(self, text):
        return self.predict_topk(text, k=1)[0][0]
