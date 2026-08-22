# Custom LLM Provider Integration

TradingAgents features a modular multi-provider LLM client factory (`tradingagents/llm_clients/`).

---

## 1. Supported Providers Out of the Box

| Provider | Supported Models | Thinking / Reasoning Parameter |
| :--- | :--- | :--- |
| **OpenAI** | `gpt-5.4`, `gpt-4.1`, `o3`, `o4-mini` | `openai_reasoning_effort` (`low`, `medium`, `high`) |
| **Anthropic** | `claude-sonnet-4-6`, `claude-opus-4-8`, `claude-haiku-4-5` | `anthropic_effort` (`low`, `medium`, `high`) |
| **Google** | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-3.0` | `google_thinking_level` (`minimal`, `high`) |
| **DeepSeek** | `deepseek-chat` (V3), `deepseek-reasoner` (R1) | Native reasoning content extraction & roundtrip |
| **Ollama** | `llama3.3`, `qwen2.5`, `mistral` (Local self-hosted) | `OLLAMA_BASE_URL` (default: `http://localhost:11434/v1`) |
| **Azure OpenAI** | Enterprise private Azure deployments | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME` |
| **BluesMind / Moonshot** | `moonshotai/kimi-*` models | OpenAI-compatible base URL & key routing |

---

## 2. Integrating a Custom OpenAI-Compatible Provider

To connect any private LLM endpoint (vLLM, SGLang, LiteLLM, OpenClaw):

```bash
# In your .env file:
TRADINGAGENTS_LLM_PROVIDER=openai
TRADINGAGENTS_LLM_BACKEND_URL=https://my-custom-proxy.internal/v1
OPENAI_API_KEY=my-proxy-token
TRADINGAGENTS_DEEP_THINK_LLM=my-custom-model-name
```
TradingAgents will automatically route requests, normalize reasoning blocks, and bind Pydantic output schemas cleanly.
