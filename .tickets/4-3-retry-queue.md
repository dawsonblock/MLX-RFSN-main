---
id: "4-3"
title: "Exponential Back-off Queue with SIGTERM Flush"
owner: Backend
estimate: 4h
priority: P0
epic: Week 4 - Secure Telemetry
status: complete
created: 2026-06-07
labels: [reliability, telemetry, queue, kubernetes, p0]
---

# Ticket 4-3: Exponential Back-off Queue with SIGTERM Flush

## Description
Implement retry queue with exponential backoff and graceful shutdown handling. Ensure zero log loss during pod termination in Kubernetes.

## Exit Criteria
- [ ] Exponential backoff: 1s, 2s, 4s, 8s, max 60s
- [ ] Max 5 retries per event
- [ ] Dead letter queue for permanently failed events
- [ ] SIGTERM handler flushes queue before exit
- [ ] Kubernetes pod kill test shows zero log loss

## Technical Details

### Retry Strategy
| Attempt | Delay | Total Elapsed |
|---------|-------|---------------|
| 1 | 1s | 1s |
| 2 | 2s | 3s |
| 3 | 4s | 7s |
| 4 | 8s | 15s |
| 5 | 16s | 31s |

After 5 failures → Dead Letter Queue (DLQ).

### Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Event     │────▶│  Retry Queue   │────▶│ ClickHouse  │
│   Source    │     │  (in-memory)   │     │   Server    │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼ (after 5 failures)
                     ┌──────────────┐
                     │  Dead Letter │
                     │    Queue     │
                     └──────────────┘
```

### SIGTERM Handling
```python
import signal
import sys

class TelemetryQueue:
    def __init__(self):
        self._queue = []
        self._shutdown = False
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)

    def _handle_sigterm(self, signum, frame):
        print("SIGTERM received, flushing queue...")
        self._shutdown = True
        self.flush()
        sys.exit(0)

    def flush(self):
        """Block until queue is empty."""
        while self._queue and not self._shutdown:
            self._process_one()
```

### Kubernetes Graceful Shutdown
```yaml
# deployment.yaml
spec:
  template:
    spec:
      containers:
        - name: rfsn
          lifecycle:
            preStop:
              exec:
                command: ["/bin/sh", "-c", "sleep 30"]
      terminationGracePeriodSeconds: 60
```

## Implementation Requirements

**RetryQueue Class**:
```python
class RetryQueue:
    """Async queue with exponential backoff and graceful shutdown."""

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._queue: deque = deque()
        self._dlq: list = []

    def put(self, event: dict):
        """Add event to queue."""
        self._queue.append({"event": event, "retries": 0})

    async def run(self):
        """Process queue until shutdown signal."""
        while not self._shutdown:
            if self._queue:
                item = self._queue.popleft()
                success = await self._send(item["event"])
                if not success:
                    item["retries"] += 1
                    if item["retries"] >= self.max_retries:
                        self._dlq.append(item)
                    else:
                        delay = min(
                            self.base_delay * (2 ** item["retries"]),
                            self.max_delay
                        )
                        await asyncio.sleep(delay)
                        self._queue.append(item)
            else:
                await asyncio.sleep(0.1)

    def flush(self):
        """Synchronous flush for signal handlers."""
        while self._queue:
            item = self._queue.popleft()
            self._send_sync(item["event"])
```

## Verification Steps

```bash
# 1. Deploy with queue
kubectl apply -f k8s/

# 2. Generate events
kubectl exec -it deploy/rfsn -- python -c "
from rfsn_v10.telemetry import queue
for i in range(100):
    queue.put({'test': i})
"

# 3. Kill pod gracefully
kubectl delete pod -l app=rfsn --grace-period=60

# 4. Verify all events in ClickHouse
docker exec rfsn-clickhouse clickhouse-client -q "SELECT count() FROM telemetry WHERE test IS NOT NULL"
# Should return 100
```

## Related Files
- `rfsn_v10/async_writer.py` — Current async implementation
- `rfsn_v10/telemetry/queue.py` — New retry queue (to create)
- `k8s/deployment.yaml` — Add preStop hook

## Risks
- In-memory queue = data loss on OOM/crash (acceptable for telemetry)
- For critical events, consider persistent queue (Redis/SQS)
