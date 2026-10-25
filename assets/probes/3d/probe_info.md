3D PROBE — Elephant Rehabilitation Habitat
--------------------------------------------
Prompt (system + user): from ccore/Qwen3.5-4B-3D-Coder README.
System: "Use only Three.js core ES module from cdnjs... implement controls directly."
User: "Please create one self-contained HTML file... deliberate 3D voxel model built from visible cubic blocks..."

Sampling (same family protocol): temp 1.0, top_k 20, top_p 0.95, min_p 0.05,
max_tokens 8192, ctx 16384, --reasoning-effort high, --cache-ram 2048.

Files:
  NETPURE.html  — Ling-3.0-tiny-MERNIK-4500-NETPURE.gguf (22688 B) → full scene working
  Q4_K_M.html   — Ling-3.0-tiny-Q4_K_M.gguf (17518 B) → broken geometry / "каша"

Quick reuse (same server port 28082, kill after):
  HUMANEVAL_SERVE_MODEL=... HUMANEVAL_SERVER_ARGS="--cache-ram 2048 --reasoning-effort high"
  HE_MAX_TOKENS=8192 HE_CTX=16384 HE_TEMP=1.0 HE_TOP_K=20 HE_TOP_P=0.95 HE_MIN_P=0.05
  python -c "...generate HTML..."

Results (2026-10-25):
  NETPURE: Scene, WebGLRenderer, OrbitControls, legend, coherent voxel layout (153 blocks reported in output).
  Q4_K_M: HTML header present, Three.js import present, but geometry broken — scattered/random blocks, missing structures, no readable spatial layout (author reran, same failure).
  Verdict: NETPURE delivers a qualitatively different (functional) result; Q4_K_M fails the 3D construction task entirely.
