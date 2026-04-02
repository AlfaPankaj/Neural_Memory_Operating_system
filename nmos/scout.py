import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading

class NMOSScout:
    def __init__(self, model_name="HuggingFaceTB/SmolLM2-135M-Instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
        self.current_intent = "CHAT" 

    def predict_intent(self, text):
        """Hybrid Classifier: Combines Keyword heuristics with SLM reasoning."""
        text_lower = text.lower()
        
        code_keywords = ["python", "script", "code", "function", "implement", "binary", "c++", "sql", "calculate", "math"]
        docs_keywords = ["summarize", "extract", "pdf", "csv", "data", "text", "analysis", "table", "convert"]
        
        if any(kw in text_lower for kw in code_keywords):
            self.current_intent = "CODE"
            return "CODE"
        if any(kw in text_lower for kw in docs_keywords):
            self.current_intent = "DOCS"
            return "DOCS"

        if len(text.strip()) < 10: return self.current_intent
        
        try:
            prompt = f"<|im_start|>system\nYou are the NMOS Shard Router. Output ONLY CODE, CHAT, or DOCS.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            input_len = inputs['input_ids'].shape[1]
            gen_tokens = outputs[0][input_len:]
            
            if len(gen_tokens) == 0:
                return self.current_intent
                
            prediction = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).upper()
            
            if "CODE" in prediction: self.current_intent = "CODE"
            elif "DOCS" in prediction or "SUMMAR" in prediction: self.current_intent = "DOCS"
            else: self.current_intent = "CHAT"
        except Exception as e:
            pass
        
        return self.current_intent
