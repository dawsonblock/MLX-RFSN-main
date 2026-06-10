# Integrations Zone

This zone documents how a promoted candidate connects to the rest of the local AI stack.

## Target Architecture

```
Open WebUI
    ↓
LiteLLM
    ↓
MLX-LM or vMLX local model server
    ↓
RFSN cache policy winner
    ↓
External memory (Qdrant / TurboVec)
    ↓
Tools (file search, code analysis)
```

## Cache Policy Abstraction

Even if MLX-LM does not support custom cache policies directly yet, the internal abstraction is:

```python
from rfsn_v11.integrations.cache_policy import CachePolicy, create_cache_policy

policy = create_cache_policy("turboquant_v2_b4_gs64_rot")
# Future: model.generate(prompt, cache_policy=policy)
```

See `rfsn_v11/integrations/cache_policy.py` for the current scaffold.

## External Memory

- **Qdrant**: Stable vector memory. Use first.
- **TurboVec**: Compressed local memory experiment. Optional.

See `memory/` for the memory layer scaffold.

## Future Integration Paths

- `integrations/mlx_lm/`: Direct MLX-LM cache policy hook
- `integrations/vmlx/`: vMLX serving integration
- `integrations/openwebui/`: Open WebUI pipeline function
- `integrations/litellm/`: LiteLLM router config
