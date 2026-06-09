# LAN / iPhone Mode

Run the RFSN v10 server so that iPhones and other devices on the same Wi-Fi
network can reach it.

> **Security requirement:** binding to `0.0.0.0` (all network interfaces)
> requires an API key. The server refuses to start without one.

---

## Quick start

```bash
# 1. Pick a strong API key
export RFSN_API_KEY="your-secret-token-here"
export RFSN_REQUIRE_API_KEY=true

# 2. Set model and bind to all interfaces
export RFSN_MODEL_ID="mlx-community/Qwen2.5-0.5B-Instruct-4bit"
export RFSN_HOST=0.0.0.0
export RFSN_PORT=8000

# 3. Start
rfsn-server
```

Or with CLI flags:

```bash
rfsn-server \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --host 0.0.0.0 \
  --require-api-key \
  --api-key your-secret-token-here
```

---

## Find your Mac's local IP

```bash
ipconfig getifaddr en0
# e.g. 192.168.1.42
```

The server will be reachable at `http://192.168.1.42:8000`.

---

## Call from an iPhone (Shortcut / curl)

```bash
curl http://192.168.1.42:8000/v1/chat/completions \
  -H "Authorization: Bearer your-secret-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rfsn-v10",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128
  }'
```

The server is OpenAI-compatible, so any OpenAI iOS SDK or Shortcuts HTTP action
works with the same base URL.

---

## Dashboard

Open `http://192.168.1.42:8000/dashboard` in Mobile Safari to see server
status, model info, and live performance metrics.

---

## Firewall note

macOS will prompt "Do you want to allow incoming network connections?" when
the server first binds to `0.0.0.0`. Click **Allow**.

If the prompt does not appear and the device cannot connect, run:

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add $(which python3)
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp $(which python3)
```

---

## Security checklist

- [ ] API key is set and `RFSN_REQUIRE_API_KEY=true`
- [ ] Key is not committed to git (`echo $RFSN_API_KEY` in a `.env` file only)
- [ ] Server is on a trusted home/office network
- [ ] `RFSN_HOST=0.0.0.0` is not used in production or on public Wi-Fi
