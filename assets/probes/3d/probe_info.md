3D PROBE — Elephant Rehabilitation Habitat
--------------------------------------------
Prompt (from ccore/Qwen3.5-4B-3D-Coder, same family sampling):

System: Use only the Three.js core ES module from https://cdnjs.cloudflare.com/ajax/libs/three.js/0.160.0/three.module.min.js.

User: Please create one self-contained HTML file with Three.js from cdnjs showing an elephant rehabilitation habitat. A broad sheltered yard connects water, shade, and feeding areas through accessible paths. Use the elephant's scale and movement clearance to guide the layout around the structures. Render it as a deliberate 3D voxel model built from visible cubic blocks, with clear depth, readable silhouettes, and solid spatial relationships.

Sampling: temp 1.0, top_k 20, top_p 0.95, min_p 0.05, max_tokens 8192, ctx 16384, --reasoning-effort high, --cache-ram 2048.

Files (assets/probes/3d/):
  NETPURE.html (22688 B): working scene — Scene, WebGLRenderer, OrbitControls, legend, coherent voxel geometry (see screenshot_netpure_20-16-02.png).
  Q4_K_M.html (17518 B): broken geometry — scattered blocks, missing elements, no coherent layout (see screenshot_q4_20-15-51.png for first broken scene; second rerun shows same failure — scattered/random blocks, missing structures, no readable spatial layout).
  probe_info.md: this file.

Results (2026-10-25):
  NETPURE: coherent voxel scene, full scene working.
  Q4_K_M: HTML header and Three.js import present, but geometry broken — scattered/random blocks, missing structures, no readable spatial layout (author reran, same failure; second attempt confirms failure, not transient).
  Verdict: NETPURE delivers a qualitatively different (functional) result; Q4_K_M fails the 3D construction task entirely.
