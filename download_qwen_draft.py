from huggingface_hub import hf_hub_download
import os

def download_qwen_draft():
    repo_id = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
    quant_file = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    
    print(f"\n[NMOS] Downloading Matching Qwen Draft (Q4_K_M)...")
    print(f"[NMOS] Size: ~1.1 GB. Target: {quant_file}")
    
    try:
        file_path = hf_hub_download(
            repo_id=repo_id, 
            filename=quant_file, 
            local_dir=".", 
            local_dir_use_symlinks=False
        )
        print(f"\n[SUCCESS] Qwen Draft is ready at: {file_path}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")

if __name__ == "__main__":
    download_qwen_draft()
