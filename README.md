# Copilot API Proxy

A reverse proxy for the GitHub Copilot API that exposes it as an OpenAI and Anthropic compatible service. Use GitHub Copilot with any tool that supports the OpenAI Chat Completions API or Anthropic Messages API, including [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview).

> Inspired by [copilot-api](https://github.com/ericc-ch/copilot-api) and similar to [gemini-api-proxy](https://github.com/IT-BAER/gemini-api-proxy)

> [!WARNING]
> This project uses GitHub's internal Copilot API endpoints that are not publicly documented. Using this proxy may violate GitHub's Terms of Service.
>
> **Potential risks:** Rate limits, API suspension, account review, or account termination.
>
> **Use at your own risk.** This project is for educational purposes only.

---

## ⚡ Quick Install (Debian/Ubuntu)

**One-liner installation** - installs and runs as a systemd service:

```bash
curl -fsSL https://raw.githubusercontent.com/IT-BAER/copilot-api-proxy/main/install.sh | sudo bash
```

After installation:
- **Setup:** `http://<your-ip>:4141/setup`
- **Dashboard:** `http://<your-ip>:4141/`
- **Service:** `sudo systemctl status copilot-api-proxy`

---

## Features

| Feature | Description |
|---------|-------------|
| 🔌 **Dual API Compatibility** | OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) endpoints |
| 🌐 **Web Authentication** | Secure GitHub OAuth Device Flow via `/setup` |
| 📊 **Usage Dashboard** | Real-time stats, quota monitoring, request history |
| 🚦 **Rate Limiting** | Configurable throttling with smart queuing |
| 🔄 **Auto Token Refresh** | Seamless token renewal before expiration |
| 🔒 **Secure Storage** | Tokens stored with restricted permissions |
| 🤖 **All Copilot Models** | GPT-5, Claude 4.5, Gemini 3, and 35+ more |

---

## Installation

### Option 1: One-Liner (Debian/Ubuntu) — Recommended

```bash
curl -fsSL https://raw.githubusercontent.com/IT-BAER/copilot-api-proxy/main/install.sh | sudo bash
```

This will:
- Install Python 3 and dependencies
- Download the latest release to `/opt/copilot-api-proxy`
- Create a Python virtual environment
- Install and enable `copilot-api-proxy.service`
- Start the service automatically

### Option 2: Docker

```bash
git clone https://github.com/IT-BAER/copilot-api-proxy.git
cd copilot-api-proxy
docker-compose up -d
docker-compose logs -f  # View authentication URL
```

### Option 3: Manual Installation

```bash
git clone https://github.com/IT-BAER/copilot-api-proxy.git
cd copilot-api-proxy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

---

## Service Management (systemd)

```bash
# Status
sudo systemctl status copilot-api-proxy

# Logs (follow)
sudo journalctl -u copilot-api-proxy -f

# Restart
sudo systemctl restart copilot-api-proxy

# Stop
sudo systemctl stop copilot-api-proxy

# Uninstall
sudo systemctl stop copilot-api-proxy
sudo systemctl disable copilot-api-proxy
sudo rm /etc/systemd/system/copilot-api-proxy.service
sudo rm -rf /opt/copilot-api-proxy
sudo systemctl daemon-reload
```

---

## Configuration

Environment variables (set in `/etc/systemd/system/copilot-api-proxy.service`):

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Port to listen on | `4141` |
| `HOST` | Host to bind to | `0.0.0.0` |
| `TOKEN_FILE` | Path to token storage | `data/github_token.json` |
| `RATE_LIMIT` | Seconds between requests (0 = disabled) | `0` |
| `WAIT_MODE` | Queue requests when rate limited | `false` |
| `ACCOUNT_TYPE` | `individual`, `business`, or `enterprise` | `individual` |

To modify, edit the service file and reload:

```bash
sudo systemctl edit copilot-api-proxy --force
sudo systemctl daemon-reload
sudo systemctl restart copilot-api-proxy
```

---

## Authentication

### Web-Based (Recommended)

1. Open `http://localhost:4141/setup` in your browser
2. Click **"Connect GitHub Account"**
3. Enter the code shown on GitHub's device activation page
4. Authorize the application
5. Redirected to dashboard automatically

### Console Mode (Headless)

```bash
CONSOLE_AUTH=true python app.py
```

Tokens are stored in `data/github_token.json`.

---

## Web UI

### Dashboard (`/` or `/dashboard`)

Real-time monitoring with:
- Request statistics and success rates
- Copilot quota (premium interactions, chat, completions)
- Available models list
- Request history with token counts

### Setup Page (`/setup`)

GitHub OAuth authentication flow with device code entry.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI chat completion |
| `/v1/models` | GET | List available models |
| `/v1/embeddings` | POST | Create embeddings |
| `/v1/messages` | POST | Anthropic messages API |
| `/dashboard` | GET | Web dashboard |
| `/usage` | GET | Quota information (JSON) |
| `/stats` | GET | Usage statistics (JSON) |
| `/health` | GET | Health check |

---

## Usage Examples

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4141/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Anthropic SDK (Python)

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:4141",
    api_key="dummy"
)

message = client.messages.create(
    model="claude-3.5-sonnet",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(message.content[0].text)
```

### cURL

```bash
# OpenAI format
curl http://localhost:4141/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello!"}]}'

# Anthropic format
curl http://localhost:4141/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3.5-sonnet", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## Claude Code Integration

Add to your Claude Code configuration:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:4141",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_MODEL": "gpt-4.1",
    "ANTHROPIC_SMALL_FAST_MODEL": "gpt-4.1-mini"
  }
}
```

---

## Available Models

Use `/v1/models` for the full list. Current models include:

| Category | Models |
|----------|--------|
| **GPT** | `gpt-5`, `gpt-5-mini`, `gpt-5.1`, `gpt-4o`, `gpt-4.1`, `gpt-4` |
| **Claude** | `claude-sonnet-4.5`, `claude-opus-4.5`, `claude-haiku-4.5` |
| **Gemini** | `gemini-3-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-pro` |
| **Other** | `grok-code-fast-1`, `text-embedding-3-small` |

---

## Prerequisites

- Python 3.11+
- GitHub account with Copilot subscription (individual, business, or enterprise)

---

## License

MIT License - See [LICENSE](LICENSE) for details.
