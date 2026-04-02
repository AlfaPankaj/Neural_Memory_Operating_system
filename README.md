# Neural Memory Operating System (NMOS)

> **Anticipatory Inference Engine for Large Language Models**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
![VRAM](https://img.shields.io/badge/VRAM-4GB%20RTX%202050-orange)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-brightgreen)
![Speculative](https://img.shields.io/badge/Speculative-Decoding-blue)

---

**NMOS (Neural Memory Operating System)** is a predictive partial execution engine designed to achieve 70B+ model reasoning on consumer-grade hardware (4GB VRAM). It operates on the **"Zero-Lag" Hypothesis**, using human behavioral signals (typing latency) to mask the physical Memory Wall through asynchronous layer pre-fetching and speculative decoding.


## Key Features

### 🧠 The Scout (Anticipatory Intent)
- **Hybrid Classifier**: Combines SmolLM2-135M reasoning with heuristic triggers to predict shard affinity.
- **Top-K Prediction**: Identifies and pre-loads the most likely expert shards (CODE, CHAT, DOCS) to maximize hit rate.
- **Latency Masking**: Leverages $T_{\text{typing}}$ to amortize $T_{\text{load}}$, effectively "hiding" SSD read times.

### 🌊 The River (Asynchronous Streaming)
- **Double-Buffered Prefetcher**: Pipelined weight streaming (SSD → RAM → VRAM) without GPU compute stalls.
- **Pipelined Runtime**: Decouples I/O from inference using asynchronous PCIe transfer management.
- **Just-in-Time Arrival**: Predicts execution time to synchronize shard delivery with token generation.

### ⚡ Speculative Decoding (Qwen-to-Qwen)
- **Draft Model**: Qwen2.5-1.5B (Draft) generates tokens at high speed.
- **Oracle Verification**: Qwen2.5-72B (Oracle) verifies tokens using matched tokenizers and rejection sampling.
- **Tree-Aware KV Cache**: Manages branching and pruning for speculative tree verification in VRAM.

### 💾 Memory Controller (Paged-KV)
- **Paged Attention Pool**: Divides KV-cache into swappable 16MB pages to eliminate fragmentation.
- **H2O (Heavy Hitter Oracle)**: Identifies and folds least important KV pages to stay within 4GB VRAM limits.
- **VRAM Action Zone**: Dynamically manages the streaming window for 70B+ layer shards.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                 NMOS — Anticipatory Inference Pipeline              │
│                                                                      │
│  ┌──────────────────────────┐      ┌──────────────────────────────┐  │
│  │      USER INPUT          │      │      ACTION ZONE (VRAM)      │  │
│  │  (Typing Signals)        │      │                              │  │
│  └──────────┬───────────────┘      │  ┌────────────────────────┐  │  │
│             │                      │  │   Draft Model (1.5B)   │  │  │
│             ▼                      │  └──────────┬─────────────┘  │  │
│  ┌──────────────────────────┐      │             ▼                │  │
│  │      THE SCOUT           │      │  ┌────────────────────────┐  │  │
│  │ (Intent Prediction)      │─────▶│  │   72B Expert Shard     │  │  │
│  └──────────┬───────────────┘      │  │   (Loaded Async)       │  │  │
│             │                      │  └────────────────────────┘  │  │
│             ▼                      └──────────────▲───────────────┘  │
│  ┌──────────────────────────┐                     │                  │
│  │      THE RIVER           │                     │                  │
│  │ (Async Prefetcher)       │─────────────────────┘                  │
│  └──────────┬───────────────┘                                        │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────────┐      ┌──────────────────────────────┐  │
│  │     GEN4 NVMe SSD        │◀─────▶      SYSTEM RAM (Buffer)     │  │
│  └──────────────────────────┘      └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Performance Benchmarks (RTX 2050 4GB)

| Metric | Measured Value | Notes |
|---|---|---|
| Shard Load Time ($T_{\text{load}}$) | **0.6s (600ms)** | 325MB Layer Shard @ 2.36 GB/s SSD |
| Effective ms/token | **61.5 ms** | On 70B+ Model reasoning |
| Throughput | **~16.2 TPS** | Speculative Decoding K=15 |
| Scout Accuracy | **90.0%** | Hybrid (SmolLM2-135M + Heuristics) |
| Scout CPU Latency | **164.8 ms** | Real-time prediction during typing |
| TTFT (Target) | **< 500 ms** | 80% of queries with $T_{\text{typing}} \geq 2\text{s}$ |

---

## Tech Stack

### Inference Engine
| Component | Technology |
|---|---|
| Oracle Model | Qwen2.5-72B-Instruct-Q3_K_M.gguf |
| Draft Model | Qwen2.5-1.5B-Instruct-Q4_K_M.gguf |
| Scout Model | SmolLM2-135M-Instruct (CPU) |
| Runtime | llama-cpp-python (Speculative Engine) |

### Memory & Streaming
| Component | Technology |
|---|---|
| VRAM Management | Paged Attention + H2O Folding |
| Prefetcher | Asynchronous Threaded River |
| Hardware Interface | CUDA v13.2 / PCIe 3.0 x4 |
| Storage | Gen4 NVMe (Sequential Read Optimized) |

---

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA RTX GPU (4GB+ VRAM recommended)
- CUDA Toolkit 13.2+
- Gen4 NVMe SSD for optimal shard loading

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/AlfaPankaj/Neural-Memory-Operating-System.git
cd "Neural Memory Operating system"

# 2. Run the verification launcher
launcher_verify.bat

# 3. Start the NMOS Shell
python NMOS_SHELL.py
```

---

## Research Mapping

NMOS is built on foundations from the following research breakthroughs:

1.  **Anticipatory Pre-fetching**: [SwiftSpec (2025)](https://arxiv.org/abs/2506.11309) & [SP-MoE (2025)](https://arxiv.org/pdf/2510.10302)
2.  **Speculative Decoding**: [NVIDIA Speculative Decoding](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
3.  **Context Compression**: [H2O: Heavy Hitter Oracle (2023)](https://arxiv.org/abs/2306.14048)
4.  **Asynchronous Scheduling**: [LAPS-SD (2025): Semi-Clairvoyant Scheduling](https://arxiv.org/pdf/2505.17074)

---

## Roadmap

| Milestone | Status |
|---|---|
| Perceptual Masking Baseline | ✅ Complete |
| Hybrid Scout Intent Classifier | ✅ Complete |
| Asynchronous River Streaming | ✅ Complete |
| Speculative Oracle-Draft Link | ✅ Complete |
| HNSW-based Failure Memory | 📅 Planned |
| Zero-Copy io_uring Integration | 📅 Planned |

---

## Author

**Pankaj Yadav**  
M.Sc. Information Technology — Lovely Professional University (LPU)  
*Expected graduation: May 2026*

- 📧 [pankajya0003@gmail.com](mailto:pankajya0003@gmail.com)
- 💼 [LinkedIn](linkedin.com/in/pankaj-ya)
- 🐙 [GitHub](github.com/AlfaPankaj)

---

## Related Research

**CHAARI 2.0** — A comprehensive Hinglish AI Agentic Runtime Interface. A privacy-first, full-duplex voice companion built on a two-node cryptographic mesh. [View CHAARI 2.0 →](https://github.com/AlfaPankaj/chaari-2.0)

<div align="center">

*"Masking the Memory Wall using the rhythm of human thought."*

**— NMOS v1.0, built for the next era of edge inference.**

</div>
