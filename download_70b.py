from huggingface_hub import hf_hub_download
import os

def download_nmos_oracle():
    repo_id = "bartowski/Qwen2.5-72B-Instruct-GGUF"
    
    quant_file = "Qwen2.5-72B-Instruct-Q3_K_M.gguf"
    
    print(f"\n[NMOS] Downloading 72B Oracle (Qwen2.5-72B Q3_K_M)...")
    print(f"[NMOS] This is an UNGATED model (no login required).")
    print(f"[NMOS] Size: ~33 GB. Target: {quant_file}")
    
    try:
        file_path = hf_hub_download(
            repo_id=repo_id, 
            filename=quant_file, 
            local_dir=".", 
            local_dir_use_symlinks=False
        )
        print(f"\n[SUCCESS] 72B Oracle is ready at: {file_path}")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("Tip: Check your internet connection and SSD space (35GB+).")

if __name__ == "__main__":
    download_nmos_oracle()
