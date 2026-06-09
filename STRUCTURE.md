# MLX-RFSN Repository Structure

Current structure:
- `rfsn_v10/`  = stable alpha baseline
- `rfsn_v11/`  = fusion prototype
- `turboquant-mlx-main/`  = reference repo (moving → external/)
- `mlx-turboquant-main/`  = reference repo (moving → external/)
- `vmlx-main/`            = reference repo (moving → external/)

Do not modify anything in rfsn_v10 beyond bug fixes.
New research goes into rfsn_v11 or benchmarks/candidates.

Branches:
- `mlx-rfsn-current-snapshot` = frozen state before cleanup
- `mlx-rfsn-fusion-cleanup`   = active cleanup branch (this branch)
- `main`                      = merge target after cleanup passes gates
