# RFSN v10 Deployment Guide

This guide covers deploying RFSN v10 in production environments.

## Prerequisites

- Python 3.10 or higher
- Apple Silicon (M1/M2/M3) for Metal acceleration
- MLX 0.20 or higher

## Installation

### From PyPI

```bash
pip install rfsn-v10[mlx,production]
```

### From Source

```bash
git clone https://github.com/yourusername/rfsn_v10.git
cd rfsn_v10
pip install -e ".[mlx,production]"
```

## Docker Deployment

### Using Docker Compose

```bash
docker-compose up -d
```

### Using Docker directly

```bash
docker build -t rfsn-v10:latest .
docker run -d \
  --name rfsn-v10 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v rfsn_cache:/app/.cache \
  -e RFSN_LOG_LEVEL=INFO \
  -p 8080:8080 \
  rfsn-v10:latest
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RFSN_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `RFSN_CACHE_DIR` | Cache directory path | `~/.cache/rfsn` |
| `RFSN_METRICS_ENABLED` | Enable metrics collection | false |
| `RFSN_MAX_MEMORY_GB` | Maximum memory usage in GB | 8 |
| `RFSN_QUOTA_GB` | Disk quota for cache in GB | 10 |

### Configuration File

Create a `rfsn_config.yaml`:

```yaml
logging:
  level: INFO
  format: json
  file: /var/log/rfsn/rfsn.log

memory:
  max_gb: 8
  quota_gb: 10
  enable_leak_detection: true

cache:
  directory: /var/cache/rfsn
  enable_persistence: true
  enable_wal: true

sparse_attention:
  default_top_k_ratio: 0.3
  block_size: 64
  enable_adaptive: true

quantization:
  default_bits: 8
  group_size: 64
  enable_wht: true
  enable_incoherent_signs: true
```

Load configuration:

```python
from rfsn_v10.config import load_config

config = load_config("rfsn_config.yaml")
```

## Health Checks

Check system health:

```bash
curl http://localhost:8080/health
```

Response:

```json
{
  "overall_status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "checks": [
    {
      "name": "metal_availability",
      "status": "healthy",
      "message": "Metal kernels available"
    },
    {
      "name": "memory_usage",
      "status": "healthy",
      "message": "Memory usage normal: 2.5GB"
    }
  ]
}
```

## Metrics

Enable metrics export:

```bash
curl http://localhost:8080/metrics
```

Prometheus format output:

```
# TYPE rfsn_cache_size_bytes gauge
rfsn_cache_size_bytes 1073741824

# TYPE rfsn_sparse_ratio gauge
rfsn_sparse_ratio 0.3

# TYPE rfsn_avg_latency_ms gauge
rfsn_avg_latency_ms 5.2
```

## Production Checklist

- [ ] Set appropriate log level (INFO for production)
- [ ] Configure cache directory with sufficient disk space
- [ ] Enable metrics collection for monitoring
- [ ] Set memory limits appropriate for your hardware
- [ ] Configure disk quota to prevent cache overflow
- [ ] Enable write-ahead logging for durability
- [ ] Set up health check monitoring
- [ ] Configure log rotation
- [ ] Test recovery from crashes
- [ ] Validate sparse quality thresholds

## Troubleshooting

### Metal not available

Ensure you're running on Apple Silicon with MLX installed:

```bash
python -c "import mlx; print(mlx.__version__)"
```

### Memory issues

Reduce cache size or increase memory limits:

```bash
export RFSN_MAX_MEMORY_GB=4
```

### Cache corruption

Clear cache directory:

```bash
rm -rf ~/.cache/rfsn/*
```

## Monitoring

Key metrics to monitor:

- `rfsn_cache_size_bytes` - Cache memory usage
- `rfsn_sparse_ratio` - Sparse attention ratio
- `rfsn_avg_latency_ms` - Average inference latency
- `rfsn_fallback_count` - Number of fallbacks to dense attention
- `rfsn_quality_cosine` - Sparse quality cosine similarity

Alert on:

- `rfsn_fallback_count` > 10 in 5 minutes
- `rfsn_quality_cosine` < 0.90
- `rfsn_avg_latency_ms` > 50ms
- Memory usage > 90% of limit
