# RFSN v10 API Documentation

## Core API

### RFSNRuntime

Main runtime class for RFSN v10.

```python
from rfsn_v10.runtime import RFSNRuntime

runtime = RFSNRuntime(
    cache_dir="~/.cache/rfsn",
    sparse_decode=True,
    top_k_ratio=0.3,
    enable_adaptive_sparsity=True,
)
```

#### Methods

##### `compress_kv(keys, values, bits=8, group_size=64)`

Compress KV cache using quantization.

**Parameters:**
- `keys` (mx.array): Keys tensor
- `values` (mx.array): Values tensor
- `bits` (int): Quantization bit width (2-8)
- `group_size` (int): Group size for quantization

**Returns:**
- `tuple`: (packed_keys, packed_values, scales_k, scales_v)

##### `reconstruct_kv(packed, scales, n_values, shape, bits, seed, use_wht=True, use_incoherent_signs=True)`

Reconstruct KV cache from compressed format.

**Parameters:**
- `packed` (mx.array): Packed quantized data
- `scales` (mx.array): Quantization scales
- `n_values` (int): Number of values
- `shape` (tuple): Target shape
- `bits` (int): Quantization bit width
- `seed` (int): Random seed for signs
- `use_wht` (bool): Apply WHT transform
- `use_incoherent_signs` (bool): Apply incoherent signs

**Returns:**
- `mx.array`: Reconstructed tensor

##### `execute_sparse_attention(queries, keys, values, top_k_ratio, block_size=64)`

Execute sparse attention with block selection.

**Parameters:**
- `queries` (mx.array): Query tensor
- `keys` (mx.array): Key tensor
- `values` (mx.array): Value tensor
- `top_k_ratio` (float): Fraction of blocks to select (0-1)
- `block_size` (int): Block size for sparse attention

**Returns:**
- `tuple`: (output, k_active, execution_mode, metadata)

### AdaptiveBlockSparseAttention

Sparse attention with adaptive block selection.

```python
from rfsn_v10.attention import AdaptiveBlockSparseAttention

attention = AdaptiveBlockSparseAttention()
output, k_active, mode, metadata = attention.execute(
    queries=queries,
    keys=keys,
    values=values,
    top_k_ratio=0.3,
    block_size=64,
)
```

### BitPackedQuantizer

Bit-packed quantization for KV cache.

```python
from rfsn_v10.bitpack import BitPackedQuantizer

quantizer = BitPackedQuantizer(bits=8, group_size=64)
packed, scales = quantizer.quantize(values)
dequant = quantizer.dequantize(packed, scales, n_values)
```

#### Methods

##### `quantize(values)`

Quantize values to bit-packed format.

**Parameters:**
- `values` (mx.array): Input values

**Returns:**
- `tuple`: (packed, scales)

##### `dequantize(packed, scales, n_values)`

Dequantize from bit-packed format.

**Parameters:**
- `packed` (mx.array): Packed data
- `scales` (mx.array): Quantization scales
- `n_values` (int): Number of values

**Returns:**
- `mx.array`: Dequantized values

## Configuration API

### RFSNConfig

Configuration management.

```python
from rfsn_v10.config import RFSNConfig, load_config

# Load from environment
config = RFSNConfig.from_env()

# Load from YAML
config = load_config("rfsn_config.yaml")

# Access configuration
log_level = config.logging.level
max_memory = config.memory.max_gb
```

## Health Check API

### HealthChecker

System health monitoring.

```python
from rfsn_v10.health import get_health_checker

checker = get_health_checker()
report = checker.get_health_report()
```

#### Methods

##### `run_all_checks()`

Run all registered health checks.

**Returns:**
- `dict`: Health check results

##### `get_health_report()`

Get comprehensive health report.

**Returns:**
- `dict`: Health report with overall status

## Metrics API

### MetricsRegistry

Metrics collection and export.

```python
from rfsn_v10.metrics import MetricsRegistry

registry = MetricsRegistry()

# Record metrics
registry.gauge("cache_size_bytes", 1073741824)
registry.counter("requests_total", 1)
registry.histogram("latency_ms", 5.2)

# Export
prometheus_format = registry.export_prometheus()
json_format = registry.export_json()
```

## Error Handling API

### ErrorHandler

Centralized error handling.

```python
from rfsn_v10.errors import handle_error, ErrorCode

try:
    # RFSN operation
    pass
except Exception as e:
    error = handle_error(e, ErrorCode.KV_QUANTIZATION_FAILED, context={"layer": 0})
    print(error.to_dict())
```

### Exception Classes

- `KVCacheException`: KV cache errors
- `AttentionException`: Attention errors
- `MemoryException`: Memory errors
- `KernelException`: Kernel errors
- `PersistenceException`: Persistence errors
- `ValidationException`: Validation errors

## Logging API

### RFSNLogger

Structured logging.

```python
from rfsn_v10.logging import get_logger

logger = get_logger("rfsn", level="INFO", log_file="rfsn.log")

logger.info("Operation started", context={"operation": "compress"})
logger.error("Operation failed", context={"error": str(e)})
```

## Memory Management API

### MultiTenantMemoryManager

Multi-tenant memory management.

```python
from rfsn_v10.memory_manager import MultiTenantMemoryManager, TenantId

manager = MultiTenantMemoryManager()

# Allocate memory
alloc_id = manager.allocate(
    tenant_id=TenantId("tenant1"),
    region=MemoryRegion.KV_CACHE,
    size_bytes=1024 * 1024,
)

# Record access
manager.access(alloc_id)

# Release
manager.release(alloc_id)

# Get stats
stats = manager.get_stats()
```

## Disk Persistence API

### CachePersistenceManager

Disk persistence with WAL.

```python
from rfsn_v10.disk_persistence import CachePersistenceManager, CacheMetadata

manager = CachePersistenceManager(cache_dir="~/.cache/rfsn")

# Persist
metadata = CacheMetadata(
    model_id="gpt2",
    layer_id="0",
    seq_len=1024,
    timestamp=time.time(),
    checksum="abc123",
    size_bytes=1024,
)
manager.persist(data, metadata)

# Load
data = manager.load(metadata)

# Recover
recovered = manager.recover()
```

## Performance Profiling API

### RFSNProfiler

Performance profiling.

```python
from rfsn_v10.profiler import RFSNProfiler

profiler = RFSNProfiler()

with profiler.profile("compress_kv"):
    result = compress_kv(keys, values)

report = profiler.get_report()
```

## Adaptive Batch Sizing API

### AdaptiveBatchSizer

Dynamic batch size adjustment.

```python
from rfsn_v10.adaptive_batch import AdaptiveBatchSizer

sizer = AdaptiveBatchSizer()
batch_size = sizer.get_batch_size()

# After processing
new_batch_size = sizer.update(latency_ms)
```

## Type Definitions

### MemoryRegion

Memory region types:
- `KV_CACHE`: KV cache storage
- `ATTENTION`: Attention computation
- `TEMPORARY`: Temporary buffers

### HealthStatus

Health status levels:
- `HEALTHY`: System operating normally
- `DEGRADED`: System degraded but functional
- `UNHEALTHY`: System not functional

### ErrorCode

Structured error codes:
- KV cache errors (1xxx)
- Attention errors (2xxx)
- Memory errors (3xxx)
- Kernel errors (4xxx)
- Persistence errors (5xxx)
- Validation errors (6xxx)
- General errors (9xxx)
