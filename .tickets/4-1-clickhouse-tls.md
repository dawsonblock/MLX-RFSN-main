---
id: "4-1"
title: "Wrap ClickHouse Client in TLS + RFSN-Auth Header"
owner: Backend
estimate: 3h
priority: P0
epic: Week 4 - Secure Telemetry
status: complete
created: 2026-06-07
labels: [security, telemetry, clickhouse, tls, p0]
---

# Ticket 4-1: Wrap ClickHouse Client in TLS + RFSN-Auth Header

## Description
Secure ClickHouse telemetry with HTTPS and custom authentication header. Currently, the docker-compose.yml shows ClickHouse on HTTP (port 8123) without TLS. All telemetry must be encrypted in transit.

## Exit Criteria
- [ ] ClickHouse client uses `https://` protocol
- [ ] `RFSN-Auth` header sent with all requests
- [ ] MITM test: plain-text prompt no longer visible in Wireshark
- [ ] Certificate validation enabled (no `verify=False`)

## Technical Details

### Current State
File: `docker-compose.yml`
```yaml
clickhouse:
  ports:
    - "8123:8123"  # HTTP, not HTTPS
```

### Required Changes
1. Enable HTTPS on ClickHouse server (port 8443 or custom)
2. Generate or mount TLS certificates
3. Update client to use `https://` and verify certs
4. Add `RFSN-Auth: <token>` header to all requests

### Implementation Hints

**ClickHouse Docker TLS Setup**:
```yaml
clickhouse:
  volumes:
    - ./certs:/etc/clickhouse-server/certs:ro
  environment:
    CLICKHOUSE_SSL_CERTIFICATE: /etc/clickhouse-server/certs/server.crt
    CLICKHOUSE_SSL_PRIVATE_KEY: /etc/clickhouse-server/certs/server.key
```

**Client Code Changes**:
```python
# Before
requests.post("http://clickhouse:8123", data=query)

# After
requests.post(
    "https://clickhouse:8443",
    data=query,
    headers={"RFSN-Auth": auth_token},
    verify="/path/to/ca.crt"  # Don't use verify=False
)
```

## Verification Steps

```bash
# 1. Capture network traffic
tshark -i any -f "port 8123 or port 8443" -w /tmp/capture.pcap

# 2. Run telemetry insert
python -c "from rfsn_v10.telemetry import insert; insert('test')"

# 3. Verify no plaintext in capture
tshark -r /tmp/capture.pcap -Y http -T fields -e http.request.uri
# Should show NO output (no HTTP traffic)

tshark -r /tmp/capture.pcap -Y tls -T fields -e tls.handshake.ciphersuite
# Should show TLS handshake
```

## Related Files
- `docker-compose.yml` — Add TLS volumes
- `rfsn_v10/telemetry/clickhouse_client.py` — Add HTTPS + auth header
- `rfsn_v10/config.py` — Add TLS cert paths

## Blockers
None — this is a foundational security task.

## Risks
- Certificate management in Docker/K8s
- Performance impact of TLS (minimal for telemetry)

## Notes
- Use mkcert for local dev certs
- In production, mount certs from Kubernetes secrets
- Consider mutual TLS (mTLS) for future hardening
