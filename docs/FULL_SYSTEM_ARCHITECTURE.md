# Full Local AI System Architecture

## Stack

```
User / iPhone / Browser
        ↓
Open WebUI
        ↓
LiteLLM router
        ↓
MLX-LM or vMLX local model server
        ↓
RFSN cache policy winner
        ↓
External memory:
    - Qdrant (stable)
    - TurboVec (compressed experimental)
        ↓
Tools:
    - file search
    - code analysis
    - news/crime map pipeline (future)
```

## Layer Responsibilities

| Layer | Responsibility | Technology |
|-------|---------------|------------|
| UI | Chat interface, settings, history | Open WebUI |
| Router | Model selection, load balancing, API compatibility | LiteLLM |
| Model server | Load model, run inference | MLX-LM / vMLX |
| KV cache | Compress and serve KV cache | RFSN winner |
| Memory | Retrieve relevant context from documents | Qdrant / TurboVec |
| Tools | Extend capabilities beyond chat | Custom pipelines |

## Data Flow

1. User sends a message via Open WebUI.
2. LiteLLM routes it to the local MLX-LM/vMLX server.
3. Before generation, the system queries external memory for relevant context.
4. Retrieved context is prepended to the prompt.
5. MLX-LM generates tokens using the RFSN cache policy winner.
6. Response flows back up to the user.

## Key Boundary

RFSN is **only** the cache policy winner layer. It is not the UI, router, model server, or memory store. It integrates with those layers but does not replace them.
