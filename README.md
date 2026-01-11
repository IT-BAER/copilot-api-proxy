# Copilot API Proxy

A reverse proxy for the GitHub Copilot API that exposes it as an OpenAI and Anthropic compatible service. This allows you to use GitHub Copilot with any tool that supports the OpenAI Chat Completions API or the Anthropic Messages API, including to power [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview).

> Inspired by [copilot-api](https://github.com/ericc-ch/copilot-api) and similar to [gemini-api-proxy](https://github.com/IT-BAER/)

> [!WARNING]
>
> This project uses GitHub's internal Copilot API endpoints that are not publicly documented. Using this proxy may violate GitHub's Terms of Service.
> **Potential risks to your GitHub account:**
>
> - **Rate limit enforcement** - Exceeding usage limits may trigger restrictions
> - **API access suspension** - GitHub may temporarily or permanently suspend API access
> - **Account review** - Unusual usage patterns may flag your account for manual review
> - **Account termination** - Serious or repeated violations could result in account closure
>
> **Use at your own risk.** This project is for educational purposes. The authors are not responsible for any consequences to your GitHub account.

## Features

- **OpenAI & Anthropic Compatibility**: Exposes GitHub Copilot as OpenAI-compatible (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`) and Anthropic-compatible (`/v1/messages`) endpoints.
- **Web-Based Authentication**: Secure GitHub OAuth Device Flow via browser-based setup page (`/setup`).
- **Detailed Usage Dashboard**: Beautiful web-based dashboard displaying real-time request status, usage statistics, Copilot quota information, and request history.
- **Live Quota Monitoring**: View your Copilot premium interactions, chat, and completions quotas with visual progress bars.
- **Rate Limit Control**: Configurable request throttling (`RATE_LIMIT` env var) and smart queuing (`WAIT_MODE`).
- **Token Auto-Refresh**: Automatically refreshes Copilot tokens before they expire.
- **Secure Token Storage**: Tokens stored with 0o600 permissions for security.
- **All Copilot Models**: Access all available Copilot models including GPT-5, Claude 4.5, Gemini 3, and more.

## Prerequisites

- Python 3.11+
- GitHub account with Copilot subscription (individual, business, or enterprise)

## Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd copilot-api-proxy

# Start the container
docker-compose up -d

# View logs for authentication URL
docker-compose logs -f
```

### Manual Installation

```bash
git clone <your-repo-url>
cd copilot-api-proxy
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Port to listen on | `4141` |
| `HOST` | Host to bind to | `0.0.0.0` |
| `TOKEN_FILE` | Path to token storage file | `data/github_token.json` |
| `RATE_LIMIT` | Minimum seconds between requests (0 = disabled) | `0` |
| `WAIT_MODE` | Wait instead of error when rate limited | `false` |
| `SHOW_TOKEN` | Display tokens in logs | `false` |
| `ACCOUNT_TYPE` | GitHub Copilot account type: `individual`, `business`, `enterprise` | `individual` |

## First Run - Authentication

On first run, the proxy will guide you through GitHub Device OAuth:

### Web-Based Authentication (Recommended)

1. Start the proxy: `python app.py`
2. Open `http://localhost:4141/setup` in your browser
3. Click "Connect GitHub Account"
4. Visit the GitHub URL shown and enter the code
5. Authorize the application
6. You'll be automatically redirected to the dashboard

### Console Authentication (Development)

For headless/development environments, set `CONSOLE_AUTH=true`:

```bash
CONSOLE_AUTH=true python app.py
```

The GitHub token is saved to `data/github_token.json` for future runs.

## Web UI

### Setup Page (`/setup`)

The setup page provides a secure way to authenticate with GitHub:

- Start/cancel authentication flow
- View device code to enter on GitHub
- Auto-polling for authorization completion
- Sign out and reconnect options

### Dashboard (`/` or `/dashboard`)

The dashboard displays:

- **Real-time statistics**: Total requests, success/failure rates, token usage
- **Copilot quota**: Premium interactions, chat, completions with progress bars
- **Available models**: All 39+ models (GPT-5, Claude 4.5, Gemini 3, etc.)
- **Request history**: Recent API calls with timestamps and token counts
- **User info**: Shows connected GitHub account

## API Endpoints

### OpenAI Compatible

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Creates a chat completion |
| `/v1/models` | GET | Lists available models |
| `/v1/embeddings` | POST | Creates embeddings |

### Anthropic Compatible

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Creates a message (Claude API compatible) |
| `/v1/messages/count_tokens` | POST | Counts tokens in a message |

### Utility

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dashboard` | GET | Web-based usage dashboard |
| `/usage` | GET | Copilot usage and quota information |
| `/stats` | GET | Proxy usage statistics |
| `/health` | GET | Health check |

## Usage Examples

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4141/v1",
    api_key="dummy"  # Not used, but required by the SDK
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
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
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(message.content[0].text)
```

### curl (OpenAI)

```bash
curl http://localhost:4141/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### curl (Anthropic)

```bash
curl http://localhost:4141/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3.5-sonnet",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Using with Claude Code

Configure Claude Code to use this proxy:

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

## Available Models

The proxy exposes all models available through GitHub Copilot. As of January 2026, available models include:

**GPT Models:**
- `gpt-5`, `gpt-5-mini`, `gpt-5.1`, `gpt-5.2`
- `gpt-5-codex`, `gpt-5.1-codex`, `gpt-5.1-codex-mini`, `gpt-5.1-codex-max`
- `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4`

**Claude Models:**
- `claude-sonnet-4`, `claude-sonnet-4.5`
- `claude-opus-4.5`, `claude-opus-41`
- `claude-haiku-4.5`

**Gemini Models:**
- `gemini-3-pro-preview`, `gemini-3-flash-preview`
- `gemini-2.5-pro`

**Other:**
- `grok-code-fast-1`
- `text-embedding-3-small`, `text-embedding-ada-002`

Use the `/v1/models` endpoint to get the full list of available models.

## License

MIT License - See [LICENSE](LICENSE) for details.
