"""
RoPE base. Production: one base, 10^6, in both stages (Qwen2.5-Coder's repository-stage base, §3.2.2; Llama 3 trains
at a large base from the start), adopted without a study. Alternative: 10^4 in the file-level stage, raised to 10^6
for the repository stage, as Code Llama's long-context stage does (arXiv 2308.12950 §2.4; ablations/model.py).

    python -m ablations.rope    (ablations/harness.py; ablations.md)
"""
from ablations.harness import ARCHITECTURE, main

if __name__ == "__main__":
    main(alternatives={"staged RoPE base": {"rope": "staged"}}, setting=ARCHITECTURE)
