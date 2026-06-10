# Integration Notes

The promoted candidate `rfsn_v10_k8_v5_gs64` can be integrated via:

```python
from rfsn_v11.integrations.cache_policy import create_cache_policy
policy = create_cache_policy("rfsn_v10_k8_v5_gs64")
# model.generate(prompt, cache_policy=policy)
```
