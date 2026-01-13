"""
Copilot API Proxy - OpenAI & Anthropic Compatible API for GitHub Copilot

A reverse-proxy that exposes GitHub Copilot API as OpenAI and Anthropic
compatible endpoints. Supports all Copilot models with rate limiting and usage tracking.
"""

import os
import json
import uuid
import time
import asyncio
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from collections import deque

import requests
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

# =============================================================================
# Configuration
# =============================================================================

GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"
GITHUB_APP_SCOPES = "read:user"
GITHUB_BASE_URL = "https://github.com"
GITHUB_API_BASE_URL = "https://api.github.com"

TOKEN_FILE = os.environ.get("TOKEN_FILE", "data/github_token.json")
PORT = int(os.environ.get("PORT", 4141))
HOST = os.environ.get("HOST", "0.0.0.0")
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", 0))  # Seconds between requests (0 = disabled)
WAIT_MODE = os.environ.get("WAIT_MODE", "false").lower() == "true"
SHOW_TOKEN = os.environ.get("SHOW_TOKEN", "false").lower() == "true"
ACCOUNT_TYPE = os.environ.get("ACCOUNT_TYPE", "individual")  # individual, business, enterprise

VERSION = "0.2.0"
COPILOT_VERSION = "0.26.7"
VSCODE_VERSION = "1.104.3"
EDITOR_PLUGIN_VERSION = f"copilot-chat/{COPILOT_VERSION}"
USER_AGENT = f"GitHubCopilotChat/{COPILOT_VERSION}"
API_VERSION = "2025-04-01"

# =============================================================================
# Internal State & Usage Tracking
# =============================================================================

state = {
    "github_token": None,
    "copilot_token": None,
    "copilot_token_expires_at": 0,
    "copilot_token_refresh_in": 0,
    "models": None,
    "vscode_version": VSCODE_VERSION,
    "account_type": ACCOUNT_TYPE,
    "last_request_time": 0,
    "rate_limit_lock": threading.Lock(),
    # Authentication state for web UI
    "github_user": None,
    "device_code": None,
    "device_code_expires_at": 0,
    "auth_in_progress": False,
    "poll_interval": 5,  # Default polling interval in seconds
    "http_client": None, # Global HTTP client for performance
}

usage_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "requests_by_model": {},
    "requests_today": 0,
    "last_reset": datetime.now().date().isoformat(),
    "request_history": deque(maxlen=100),
}

def track_request(model: str, success: bool, input_tokens: int = 0, output_tokens: int = 0):
    """Track usage statistics for a request."""
    usage_stats["total_requests"] += 1
    if success:
        usage_stats["successful_requests"] += 1
    else:
        usage_stats["failed_requests"] += 1
    
    usage_stats["total_input_tokens"] += input_tokens
    usage_stats["total_output_tokens"] += output_tokens
    
    if model not in usage_stats["requests_by_model"]:
        usage_stats["requests_by_model"][model] = 0
    usage_stats["requests_by_model"][model] += 1
    
    # Reset daily counter if needed
    today = datetime.now().date().isoformat()
    if usage_stats["last_reset"] != today:
        usage_stats["requests_today"] = 0
        usage_stats["last_reset"] = today
    usage_stats["requests_today"] += 1
    
    # Add to history
    usage_stats["request_history"].append({
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "success": success,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    })

# =============================================================================
# Token Management
# =============================================================================

def ensure_data_dir():
    """Ensure data directory exists."""
    data_dir = os.path.dirname(TOKEN_FILE)
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

def load_tokens():
    """Load tokens from file."""
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                data = json.load(f)
                state["github_token"] = data.get("github_token")
                state["copilot_token"] = data.get("copilot_token")
                state["copilot_token_expires_at"] = data.get("copilot_token_expires_at", 0)
                return True
        except Exception as e:
            print(f"Error loading tokens: {e}")
    return False

def save_tokens():
    """Save tokens to file."""
    ensure_data_dir()
    try:
        with open(TOKEN_FILE, "w") as f:
            json.dump({
                "github_token": state["github_token"],
                "copilot_token": state["copilot_token"],
                "copilot_token_expires_at": state["copilot_token_expires_at"],
            }, f, indent=2)
        # Secure the file
        try:
            os.chmod(TOKEN_FILE, 0o600)
        except Exception:
            pass
    except Exception as e:
        print(f"Error saving tokens: {e}")

def standard_headers():
    """Return standard headers for API requests."""
    return {
        "content-type": "application/json",
        "accept": "application/json",
    }

def github_headers():
    """Return headers for GitHub API requests."""
    return {
        **standard_headers(),
        "authorization": f"token {state['github_token']}",
        "editor-version": f"vscode/{state['vscode_version']}",
        "editor-plugin-version": EDITOR_PLUGIN_VERSION,
        "user-agent": USER_AGENT,
        "x-github-api-version": API_VERSION,
        "x-vscode-user-agent-library-version": "electron-fetch",
    }

def copilot_base_url():
    """Return the base URL for Copilot API based on account type."""
    if state["account_type"] == "individual":
        return "https://api.githubcopilot.com"
    return f"https://api.{state['account_type']}.githubcopilot.com"

def copilot_headers(vision: bool = False):
    """Return headers for Copilot API requests."""
    headers = {
        "Authorization": f"Bearer {state['copilot_token']}",
        "content-type": "application/json",
        "copilot-integration-id": "vscode-chat",
        "editor-version": f"vscode/{state['vscode_version']}",
        "editor-plugin-version": EDITOR_PLUGIN_VERSION,
        "user-agent": USER_AGENT,
        "openai-intent": "conversation-panel",
        "x-github-api-version": API_VERSION,
        "x-request-id": str(uuid.uuid4()),
        "x-vscode-user-agent-library-version": "electron-fetch",
    }
    if vision:
        headers["copilot-vision-request"] = "true"
    return headers

def get_device_code():
    """Get device code for GitHub OAuth."""
    response = requests.post(
        f"{GITHUB_BASE_URL}/login/device/code",
        headers=standard_headers(),
        json={
            "client_id": GITHUB_CLIENT_ID,
            "scope": GITHUB_APP_SCOPES,
        }
    )
    if not response.ok:
        raise Exception(f"Failed to get device code: {response.text}")
    data = response.json()
    print(f"Device code response: {data}")  # Debug logging
    return data

def poll_access_token(device_code_response: dict) -> str:
    """Poll for access token after user authorizes."""
    interval = device_code_response.get("interval", 5) + 1
    device_code = device_code_response["device_code"]
    
    while True:
        time.sleep(interval)
        response = requests.post(
            f"{GITHUB_BASE_URL}/login/oauth/access_token",
            headers=standard_headers(),
            json={
                "client_id": GITHUB_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }
        )
        
        if not response.ok:
            print(f"Polling failed: {response.text}")
            continue
        
        data = response.json()
        if "access_token" in data:
            return data["access_token"]
        
        error = data.get("error")
        if error == "authorization_pending":
            continue
        elif error == "slow_down":
            interval += 5
        elif error == "expired_token":
            raise Exception("Device code expired. Please try again.")
        elif error == "access_denied":
            raise Exception("Access was denied by user.")
        else:
            print(f"Unknown error: {data}")

def get_copilot_token():
    """Get Copilot token using GitHub token."""
    response = requests.get(
        f"{GITHUB_API_BASE_URL}/copilot_internal/v2/token",
        headers=github_headers()
    )
    if not response.ok:
        raise Exception(f"Failed to get Copilot token: {response.text}")
    return response.json()

def refresh_copilot_token():
    """Refresh the Copilot token if expired or expiring soon."""
    current_time = time.time()
    # Refresh if expired or expiring in the next 60 seconds
    if state["copilot_token"] and current_time < (state["copilot_token_expires_at"] - 60):
        return True
    
    try:
        token_data = get_copilot_token()
        state["copilot_token"] = token_data["token"]
        state["copilot_token_expires_at"] = token_data["expires_at"]
        state["copilot_token_refresh_in"] = token_data["refresh_in"]
        save_tokens()
        print("Copilot token refreshed successfully")
        if SHOW_TOKEN:
            print(f"Copilot token: {state['copilot_token'][:50]}...")
        return True
    except Exception as e:
        print(f"Failed to refresh Copilot token: {e}")
        return False

def get_github_user():
    """Get the authenticated GitHub user."""
    response = requests.get(
        f"{GITHUB_API_BASE_URL}/user",
        headers={
            "authorization": f"token {state['github_token']}",
            **standard_headers(),
        }
    )
    if not response.ok:
        raise Exception(f"Failed to get GitHub user: {response.text}")
    user_data = response.json()
    state["github_user"] = user_data
    return user_data

def setup_github_token(force: bool = False):
    """Setup GitHub token via device flow or from storage."""
    if state["github_token"] and not force:
        try:
            user = get_github_user()
            print(f"Logged in as {user['login']}")
            return True
        except Exception:
            print("Stored token invalid, requesting new token...")
    
    # For web UI, we'll use start_device_flow() and poll_device_flow()
    # Console mode for development/debugging only
    if os.environ.get("CONSOLE_AUTH", "false").lower() == "true":
        print("Starting GitHub device authentication flow...")
        device_code = get_device_code()
        print(f"\n{'='*60}")
        print(f"Please visit: {device_code['verification_uri']}")
        print(f"And enter code: {device_code['user_code']}")
        print(f"{'='*60}\n")
        
        token = poll_access_token(device_code)
        state["github_token"] = token
        save_tokens()
        
        if SHOW_TOKEN:
            print(f"GitHub token: {token}")
        
        user = get_github_user()
        print(f"Successfully logged in as {user['login']}")
        return True
    
    # Web UI mode - don't block
    return False

def start_device_flow():
    """Start GitHub device flow and return device code info for web UI."""
    device_code_response = get_device_code()
    state["device_code"] = device_code_response
    state["device_code_expires_at"] = time.time() + device_code_response.get("expires_in", 900)
    state["auth_in_progress"] = True
    # Store the interval (add 1 second buffer to be safe)
    state["poll_interval"] = device_code_response.get("interval", 5) + 1
    return device_code_response

def poll_device_flow():
    """Check if user has authorized the device. Non-blocking."""
    if not state["device_code"]:
        return {"status": "no_pending_auth"}
    
    if time.time() > state["device_code_expires_at"]:
        state["device_code"] = None
        state["auth_in_progress"] = False
        return {"status": "expired"}
    
    device_code = state["device_code"]["device_code"]
    
    response = requests.post(
        f"{GITHUB_BASE_URL}/login/oauth/access_token",
        headers=standard_headers(),
        json={
            "client_id": GITHUB_CLIENT_ID,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        }
    )
    
    if not response.ok:
        print(f"Poll error (HTTP {response.status_code}): {response.text}")
        return {"status": "error", "error": response.text}
    
    data = response.json()
    print(f"Poll response: {data}")  # Debug logging
    
    if "access_token" in data:
        # Success!
        state["github_token"] = data["access_token"]
        state["device_code"] = None
        state["auth_in_progress"] = False
        save_tokens()
        
        # Get user info and copilot token
        try:
            user = get_github_user()
            setup_copilot_token()
            
            # Get available models (use sync version since we're in sync context)
            models = get_models_sync()
            if models:
                state["models"] = models
            
            return {
                "status": "success",
                "user": {
                    "login": user.get("login"),
                    "name": user.get("name"),
                    "avatar_url": user.get("avatar_url"),
                }
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    error = data.get("error")
    if error == "authorization_pending":
        return {"status": "pending", "interval": state["poll_interval"]}
    elif error == "slow_down":
        # Increase polling interval as requested by GitHub
        new_interval = data.get("interval", state["poll_interval"] + 5) + 1
        state["poll_interval"] = new_interval
        print(f"GitHub requested slow down, increasing interval to {new_interval}s")
        return {"status": "slow_down", "interval": new_interval}
    elif error == "expired_token":
        state["device_code"] = None
        state["auth_in_progress"] = False
        return {"status": "expired"}
    elif error == "access_denied":
        state["device_code"] = None
        state["auth_in_progress"] = False
        return {"status": "denied"}
    else:
        return {"status": "error", "error": str(data)}

def setup_copilot_token():
    """Setup Copilot token."""
    return refresh_copilot_token()

async def get_models():
    """Get available models from Copilot API."""
    if not refresh_copilot_token():
        return None
    
    headers = copilot_headers()
    client = state["http_client"] or httpx.AsyncClient(http2=True)
    try:
        response = await client.get(
            f"{copilot_base_url()}/models",
            headers=headers,
            timeout=10.0
        )
        if response.status_code != 200:
            print(f"Failed to get models: {response.text}")
            return None
        return response.json()
    except Exception as e:
        print(f"Error fetching models: {e}")
        return None

def get_models_sync():
    """Get available models from Copilot API (sync version)."""
    if not refresh_copilot_token():
        return None
    
    headers = copilot_headers()
    try:
        response = requests.get(
            f"{copilot_base_url()}/models",
            headers=headers,
            timeout=10.0
        )
        if response.status_code != 200:
            print(f"Failed to get models: {response.text}")
            return None
        return response.json()
    except Exception as e:
        print(f"Error fetching models: {e}")
        return None

def get_copilot_usage():
    """Get Copilot usage information."""
    response = requests.get(
        f"{GITHUB_API_BASE_URL}/copilot_internal/user",
        headers=github_headers()
    )
    if not response.ok:
        return None
    return response.json()

# =============================================================================
# Rate Limiting
# =============================================================================

def check_rate_limit():
    """Check and enforce rate limiting."""
    if RATE_LIMIT <= 0:
        return True
    
    with state["rate_limit_lock"]:
        current_time = time.time()
        elapsed = current_time - state["last_request_time"]
        
        if elapsed < RATE_LIMIT:
            if WAIT_MODE:
                wait_time = RATE_LIMIT - elapsed
                time.sleep(wait_time)
            else:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Please wait {RATE_LIMIT - elapsed:.1f} seconds."
                )
        
        state["last_request_time"] = time.time()
    return True

# =============================================================================
# API Request Handling
# =============================================================================

async def create_chat_completions(payload: dict, stream: bool = False):
    """Create chat completions using Copilot API."""
    if not refresh_copilot_token():
        raise HTTPException(status_code=401, detail="Failed to refresh Copilot token")
    
    check_rate_limit()
    
    # Check if any message contains images (vision request)
    enable_vision = False
    for msg in payload.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    enable_vision = True
                    break
    
    # Determine if this is an agent call (for X-Initiator header)
    is_agent_call = any(
        msg.get("role") in ["assistant", "tool"]
        for msg in payload.get("messages", [])
    )
    
    headers = copilot_headers(vision=enable_vision)
    headers["X-Initiator"] = "agent" if is_agent_call else "user"
    
    if stream:
        # Use httpx for async streaming
        client = state["http_client"] or httpx.AsyncClient(http2=True)
        request = client.build_request(
            "POST",
            f"{copilot_base_url()}/chat/completions",
            headers=headers,
            json=payload
        )
        return await client.send(request, stream=True)
    else:
        # Regular request
        client = state["http_client"] or httpx.AsyncClient(http2=True)
        response = await client.post(
            f"{copilot_base_url()}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120.0
        )
        
        if response.status_code != 200:
            error_text = response.text
            print(f"Chat completion failed: {error_text}")
            raise HTTPException(status_code=response.status_code, detail=error_text)
        
        return response

async def create_embeddings(payload: dict):
    """Create embeddings using Copilot API."""
    if not refresh_copilot_token():
        raise HTTPException(status_code=401, detail="Failed to refresh Copilot token")
    
    check_rate_limit()
    
    headers = copilot_headers()
    client = state["http_client"] or httpx.AsyncClient(http2=True)
    response = await client.post(
        f"{copilot_base_url()}/embeddings",
        headers=headers,
        json=payload,
        timeout=60.0
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    return response.json()

# =============================================================================
# Pydantic Models
# =============================================================================

class ChatMessage(BaseModel):
    # Preserve tool-call fields (e.g. tool_calls, tool_call_id) so follow-up
    # requests after tool execution remain valid.
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Any] = None
    n: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Any] = None

class EmbeddingRequest(BaseModel):
    model: str
    input: Any

class AnthropicMessage(BaseModel):
    role: str
    content: Any

class AnthropicRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: Optional[int] = 4096
    system: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Any] = None

# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print(f"Copilot API Proxy v{VERSION} starting...")
    print(f"Account type: {ACCOUNT_TYPE}")
    
    # Initialize global HTTP client with connection pooling and HTTP/2
    state["http_client"] = httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(120.0, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )
    
    # Load tokens
    load_tokens()
    
    # Try to validate existing GitHub token
    if state["github_token"]:
        try:
            user = get_github_user()
            print(f"Logged in as {user['login']}")
            
            # Setup Copilot token
            setup_copilot_token()
            
            # Get available models
            models = await get_models()
            if models:
                state["models"] = models
                print(f"\nAvailable models: {len(models.get('data', []))}")
        except Exception as e:
            print(f"Stored GitHub token invalid, please authenticate via /setup: {e}")
            state["github_token"] = None
    else:
        print("No authentication found, please visit /setup to authenticate")
    
    print(f"\nServer ready at http://{HOST}:{PORT}")
    if state["github_token"]:
        print(f"Dashboard: http://localhost:{PORT}/dashboard")
    else:
        print(f"Setup: http://localhost:{PORT}/setup")
    print(f"OpenAI-compatible endpoint: http://localhost:{PORT}/v1/chat/completions")
    print(f"Anthropic-compatible endpoint: http://localhost:{PORT}/v1/messages")
    
    # Start token refresh background task
    async def refresh_token_loop():
        while True:
            await asyncio.sleep(state.get("copilot_token_refresh_in", 1800) - 60)
            if state["github_token"]:
                refresh_copilot_token()
    
    refresh_task = asyncio.create_task(refresh_token_loop())
    
    yield
    
    # Clean up
    refresh_task.cancel()
    if state["http_client"]:
        await state["http_client"].aclose()

app = FastAPI(
    title="Copilot API Proxy",
    description="OpenAI & Anthropic Compatible API for GitHub Copilot",
    version=VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Routes - Health & Info
# =============================================================================

@app.get("/")
async def root():
    return {"status": "ok", "version": VERSION, "service": "copilot-api-proxy"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/usage")
async def get_usage():
    """Get Copilot usage information."""
    usage = get_copilot_usage()
    if usage:
        return usage
    return {"error": "Failed to get usage information"}

@app.get("/token")
async def get_token():
    """Get current Copilot token (for debugging)."""
    if SHOW_TOKEN:
        return {"token": state["copilot_token"]}
    return {"error": "Token display is disabled. Set SHOW_TOKEN=true to enable."}

@app.get("/stats")
async def get_stats():
    """Get proxy usage statistics."""
    return {
        "total_requests": usage_stats["total_requests"],
        "successful_requests": usage_stats["successful_requests"],
        "failed_requests": usage_stats["failed_requests"],
        "total_input_tokens": usage_stats["total_input_tokens"],
        "total_output_tokens": usage_stats["total_output_tokens"],
        "requests_by_model": usage_stats["requests_by_model"],
        "requests_today": usage_stats["requests_today"],
        "last_reset": usage_stats["last_reset"],
        "recent_requests": list(usage_stats["request_history"]),
    }

# =============================================================================
# Routes - OpenAI Compatible
# =============================================================================

@app.get("/models")
@app.get("/v1/models")
async def list_models():
    """List available models."""
    if state["models"]:
        return state["models"]
    
    models = await get_models()
    if models:
        state["models"] = models
        return models
    
    # Fallback model list
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4o", "object": "model", "owned_by": "github"},
            {"id": "gpt-4o-mini", "object": "model", "owned_by": "github"},
            {"id": "gpt-4.1", "object": "model", "owned_by": "github"},
            {"id": "claude-3.5-sonnet", "object": "model", "owned_by": "github"},
            {"id": "claude-3.7-sonnet", "object": "model", "owned_by": "github"},
            {"id": "o1", "object": "model", "owned_by": "github"},
            {"id": "o3-mini", "object": "model", "owned_by": "github"},
        ]
    }

@app.get("/models/{model_id}")
@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get a specific model by ID (OpenAI compatible)."""
    models_response = await list_models()
    
    # Search for the model in the list
    for model in models_response.get("data", []):
        if model.get("id") == model_id:
            return model
    
    # Model not found
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": f"The model '{model_id}' does not exist",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found"
            }
        }
    )

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Create chat completion (OpenAI compatible)."""
    payload = request.model_dump(exclude_none=True)
    
    # Preserve any message-level fields (e.g. tool_calls, tool_call_id).
    payload["messages"] = [
        {k: v for k, v in msg.items() if v is not None}
        for msg in payload["messages"]
    ]
    
    if request.stream:
        response = await create_chat_completions(payload, stream=True)

        if response.status_code != 200:
            try:
                error_text = (await response.aread()).decode("utf-8", errors="replace")
            finally:
                await response.aclose()
            raise HTTPException(status_code=response.status_code, detail=error_text)
        
        async def generate():
            try:
                # Important: forward the upstream SSE bytes without re-chunking
                # by lines; dropping blank separator lines breaks SSE parsing.
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
            finally:
                await response.aclose()
        
        track_request(request.model, True)
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    else:
        response = await create_chat_completions(payload, stream=False)
        result = response.json()
        
        # Track usage
        usage = result.get("usage", {})
        track_request(
            request.model,
            True,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0)
        )
        
        return result

@app.post("/embeddings")
@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Create embeddings (OpenAI compatible)."""
    payload = request.model_dump(exclude_none=True)
    result = await create_embeddings(payload)
    track_request(request.model, True)
    return result

# =============================================================================
# Routes - Anthropic Compatible
# =============================================================================

def translate_anthropic_to_openai(anthropic_request: dict) -> dict:
    """Translate Anthropic request to OpenAI format."""
    messages = []
    
    # Add system message if present
    if anthropic_request.get("system"):
        messages.append({
            "role": "system",
            "content": anthropic_request["system"]
        })
    
    # Convert messages
    for msg in anthropic_request.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        
        # Handle Anthropic content blocks
        if isinstance(content, list):
            # Convert to OpenAI format
            openai_content = []
            for block in content:
                if block.get("type") == "text":
                    openai_content.append({
                        "type": "text",
                        "text": block.get("text", "")
                    })
                elif block.get("type") == "image":
                    # Handle base64 images
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        openai_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                            }
                        })
                elif block.get("type") == "tool_use":
                    # This should be handled as a tool call in assistant messages
                    pass
                elif block.get("type") == "tool_result":
                    # Add tool result as a separate message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": block.get("content", "")
                    })
                    continue
            
            if openai_content:
                messages.append({
                    "role": role,
                    "content": openai_content if len(openai_content) > 1 else openai_content[0].get("text", openai_content)
                })
        else:
            messages.append({
                "role": role,
                "content": content
            })
    
    # Build OpenAI request
    openai_request = {
        "model": anthropic_request.get("model", "gpt-4o"),
        "messages": messages,
        "max_tokens": anthropic_request.get("max_tokens", 4096),
        "stream": anthropic_request.get("stream", False),
    }
    
    if anthropic_request.get("temperature") is not None:
        openai_request["temperature"] = anthropic_request["temperature"]
    if anthropic_request.get("top_p") is not None:
        openai_request["top_p"] = anthropic_request["top_p"]
    if anthropic_request.get("stop_sequences"):
        openai_request["stop"] = anthropic_request["stop_sequences"]
    
    # Convert tools
    if anthropic_request.get("tools"):
        openai_tools = []
        for tool in anthropic_request["tools"]:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            })
        openai_request["tools"] = openai_tools
    
    return openai_request

def translate_openai_to_anthropic(openai_response: dict, model: str) -> dict:
    """Translate OpenAI response to Anthropic format."""
    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})
    
    content = []
    
    # Handle text content
    if message.get("content"):
        content.append({
            "type": "text",
            "text": message["content"]
        })
    
    # Handle tool calls
    if message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            content.append({
                "type": "tool_use",
                "id": tool_call.get("id"),
                "name": tool_call.get("function", {}).get("name"),
                "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            })
    
    # Map stop reason
    finish_reason = choice.get("finish_reason", "end_turn")
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")
    
    usage = openai_response.get("usage", {})
    
    return {
        "id": openai_response.get("id", f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
    }

@app.post("/messages")
@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Create message (Anthropic compatible)."""
    body = await request.json()
    
    # Translate to OpenAI format
    openai_request = translate_anthropic_to_openai(body)
    
    model = body.get("model", "gpt-4o")
    stream = body.get("stream", False)
    
    if stream:
        response = await create_chat_completions(openai_request, stream=True)
        
        async def generate():
            try:
                # Send message_start event
                yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': f'msg_{uuid.uuid4().hex}', 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
                
                # Send content_block_start
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                
                async for line in response.aiter_lines():
                    if line:
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"
                            except json.JSONDecodeError:
                                pass
                
                # Send content_block_stop
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                
                # Send message_delta and message_stop
                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            finally:
                await response.aclose()
        
        track_request(model, True)
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    else:
        response = await create_chat_completions(openai_request, stream=False)
        openai_result = response.json()
        
        # Translate to Anthropic format
        result = translate_openai_to_anthropic(openai_result, model)
        
        # Track usage
        usage = openai_result.get("usage", {})
        track_request(
            model,
            True,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0)
        )
        
        return result

@app.post("/messages/count_tokens")
@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    """Count tokens (Anthropic compatible)."""
    body = await request.json()
    
    # Simple estimation: ~4 characters per token
    total_chars = 0
    
    if body.get("system"):
        total_chars += len(body["system"])
    
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    total_chars += len(block.get("text", ""))
    
    estimated_tokens = max(1, total_chars // 4)
    
    return {"input_tokens": estimated_tokens}

# =============================================================================
# Routes - OpenAI Responses API Compatible
# =============================================================================

# Storage for response objects (in-memory, limited)
response_storage: Dict[str, dict] = {}
RESPONSE_STORAGE_MAX = 100

def translate_responses_input_to_messages(input_data: Any, instructions: Optional[str] = None) -> List[dict]:
    """Translate Responses API input to chat messages format."""
    messages = []
    
    # Add instructions as system message
    if instructions:
        messages.append({"role": "system", "content": instructions})
    
    # Handle simple string input
    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
        return messages
    
    # Handle array of input items
    if isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                item_type = item.get("type", "message")
                
                if item_type == "message":
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    
                    # Handle content blocks
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") in ("text", "input_text", "output_text"):
                                    text_parts.append(block.get("text", ""))
                                elif block.get("type") == "image_url":
                                    # Pass through image_url blocks
                                    text_parts.append(block)
                            elif isinstance(block, str):
                                text_parts.append(block)
                        
                        # If we have mixed content (text + images), keep as array
                        if any(isinstance(p, dict) for p in text_parts):
                            content = text_parts
                        else:
                            content = " ".join(text_parts)
                    
                    messages.append({"role": role, "content": content})
                
                elif item_type == "function_call_output":
                    # Tool response
                    messages.append({
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", "")
                    })
    
    return messages

def translate_chat_response_to_responses_format(
    chat_response: dict,
    response_id: str,
    model: str,
    created_at: int
) -> dict:
    """Translate chat completion response to Responses API format."""
    output = []
    
    choices = chat_response.get("choices", [])
    for choice in choices:
        message = choice.get("message", {})
        role = message.get("role", "assistant")
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        
        # Create message output item
        if content:
            output.append({
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:40]}",
                "status": "completed",
                "role": role,
                "content": [{
                    "type": "output_text",
                    "text": content,
                    "annotations": []
                }]
            })
        
        # Handle tool calls
        for tc in tool_calls:
            output.append({
                "type": "function_call",
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                "call_id": tc.get("id", ""),
                "name": tc.get("function", {}).get("name", ""),
                "arguments": tc.get("function", {}).get("arguments", "{}"),
                "status": "completed"
            })
    
    usage = chat_response.get("usage", {})
    
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "completed_at": int(time.time()),
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": chat_response.get("model", model),
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.get("total_tokens", 0)
        },
        "user": None,
        "metadata": {}
    }

async def stream_responses_format(chat_stream, response_id: str, model: str, created_at: int):
    """Stream chat completion as Responses API SSE format."""
    message_id = f"msg_{uuid.uuid4().hex[:40]}"
    full_content = ""
    
    # Initial response.created event
    yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': {'id': response_id, 'object': 'response', 'status': 'in_progress', 'created_at': created_at, 'model': model, 'output': []}})}\n\n"
    
    # Output item added event
    yield f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': {'type': 'message', 'id': message_id, 'status': 'in_progress', 'role': 'assistant', 'content': []}})}\n\n"
    
    # Content part added
    yield f"event: response.content_part.added\ndata: {json.dumps({'type': 'response.content_part.added', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': []}})}\n\n"
    
    try:
        buffer = ""
        async for chunk in chat_stream.aiter_bytes():
            if not chunk:
                continue
            
            buffer += chunk.decode("utf-8", errors="replace")
            
            # Process complete SSE lines
            while "\n\n" in buffer or "\r\n\r\n" in buffer:
                if "\r\n\r\n" in buffer:
                    line, buffer = buffer.split("\r\n\r\n", 1)
                else:
                    line, buffer = buffer.split("\n\n", 1)
                
                if not line.strip():
                    continue
                
                # Parse SSE data line
                for sub_line in line.split("\n"):
                    if sub_line.startswith("data: "):
                        data_str = sub_line[6:]
                        if data_str.strip() == "[DONE]":
                            continue
                        
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            for choice in choices:
                                delta = choice.get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_content += content
                                    # Emit text delta
                                    yield f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'delta': content})}\n\n"
                        except json.JSONDecodeError:
                            pass
    finally:
        await chat_stream.aclose()
    
    # Content part done
    yield f"event: response.content_part.done\ndata: {json.dumps({'type': 'response.content_part.done', 'item_id': message_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': full_content, 'annotations': []}})}\n\n"
    
    # Output item done
    yield f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': {'type': 'message', 'id': message_id, 'status': 'completed', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': full_content, 'annotations': []}]}})}\n\n"
    
    # Response done
    final_response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "completed_at": int(time.time()),
        "model": model,
        "output": [{
            "type": "message",
            "id": message_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": full_content, "annotations": []}]
        }],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    }
    yield f"event: response.done\ndata: {json.dumps({'type': 'response.done', 'response': final_response})}\n\n"

@app.post("/responses")
@app.post("/v1/responses")
async def create_response(request: Request):
    """Create a model response (OpenAI Responses API compatible)."""
    body = await request.json()
    
    model = body.get("model", "gpt-4o")
    input_data = body.get("input", "")
    instructions = body.get("instructions")
    stream = body.get("stream", False)
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    max_output_tokens = body.get("max_output_tokens")
    tools = body.get("tools", [])
    tool_choice = body.get("tool_choice")
    previous_response_id = body.get("previous_response_id")
    
    # Build messages from previous response if provided
    messages = []
    if previous_response_id and previous_response_id in response_storage:
        prev_resp = response_storage[previous_response_id]
        # Add output from previous response as context
        for out_item in prev_resp.get("output", []):
            if out_item.get("type") == "message":
                role = out_item.get("role", "assistant")
                content_parts = out_item.get("content", [])
                text_content = " ".join(
                    c.get("text", "") for c in content_parts 
                    if c.get("type") in ("output_text", "input_text", "text")
                )
                if text_content:
                    messages.append({"role": role, "content": text_content})
    
    # Add current input
    messages.extend(translate_responses_input_to_messages(input_data, instructions))
    
    # Build chat completion payload
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens
    
    # Convert tools from Responses format to chat format
    if tools:
        chat_tools = []
        for tool in tools:
            tool_type = tool.get("type", "function")
            if tool_type == "function":
                chat_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {})
                    }
                })
        if chat_tools:
            payload["tools"] = chat_tools
    
    if tool_choice:
        payload["tool_choice"] = tool_choice
    
    response_id = f"resp_{uuid.uuid4().hex[:48]}"
    created_at = int(time.time())
    
    if stream:
        chat_stream = await create_chat_completions(payload, stream=True)
        
        if chat_stream.status_code != 200:
            try:
                error_text = (await chat_stream.aread()).decode("utf-8", errors="replace")
            finally:
                await chat_stream.aclose()
            raise HTTPException(status_code=chat_stream.status_code, detail=error_text)
        
        track_request(model, True)
        
        return StreamingResponse(
            stream_responses_format(chat_stream, response_id, model, created_at),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    else:
        chat_response = await create_chat_completions(payload, stream=False)
        result = chat_response.json()
        
        # Track usage
        usage = result.get("usage", {})
        track_request(
            model,
            True,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0)
        )
        
        # Build Responses API format
        response_obj = translate_chat_response_to_responses_format(
            result, response_id, model, created_at
        )
        
        # Store for retrieval and multi-turn
        if len(response_storage) >= RESPONSE_STORAGE_MAX:
            # Remove oldest
            oldest_key = next(iter(response_storage))
            del response_storage[oldest_key]
        response_storage[response_id] = response_obj
        
        return response_obj

@app.get("/responses/{response_id}")
@app.get("/v1/responses/{response_id}")
async def get_response(response_id: str):
    """Get a model response by ID."""
    if response_id not in response_storage:
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": f"Response '{response_id}' not found", "type": "invalid_request_error"}}
        )
    return response_storage[response_id]

@app.delete("/responses/{response_id}")
@app.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str):
    """Delete a model response."""
    if response_id in response_storage:
        del response_storage[response_id]
    return {"id": response_id, "object": "response", "deleted": True}

@app.post("/responses/input_tokens")
@app.post("/v1/responses/input_tokens")
async def responses_input_tokens(request: Request):
    """Count input tokens for a Responses API request."""
    body = await request.json()
    
    input_data = body.get("input", "")
    instructions = body.get("instructions", "")
    
    total_chars = len(instructions) if instructions else 0
    
    if isinstance(input_data, str):
        total_chars += len(input_data)
    elif isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, str):
                total_chars += len(item)
            elif isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            total_chars += len(block.get("text", ""))
    
    estimated_tokens = max(1, total_chars // 4)
    return {"object": "response.input_tokens", "input_tokens": estimated_tokens}

# =============================================================================
# Authentication Page
# =============================================================================

SETUP_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Copilot API Proxy - Setup</title>
    <style>
        :root {
            --bg-primary: #0f0f0f;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #242424;
            --border: #2e2e2e;
            --text-primary: #fafafa;
            --text-secondary: #a0a0a0;
            --text-muted: #6b6b6b;
            --accent: #8b5cf6;
            --success: #10b981;
            --error: #ef4444;
            --warning: #f59e0b;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            line-height: 1.5;
        }
        .container { max-width: 480px; width: 100%; }
        .card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 2.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }
        .header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
        .logo { font-size: 1.75rem; }
        h1 { font-size: 1.5rem; font-weight: 600; }
        .version { font-size: 0.625rem; opacity: 0.5; vertical-align: middle; margin-left: 0.25rem; }
        .subtitle { color: var(--text-secondary); margin-bottom: 2rem; font-size: 0.9375rem; }
        .step {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }
        .step-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.5rem;
            height: 1.5rem;
            background: var(--accent);
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.75rem;
            margin-right: 0.75rem;
        }
        .step h3 { display: flex; align-items: center; margin-bottom: 0.75rem; font-size: 1rem; font-weight: 500; }
        .step p { color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.875rem; }
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--accent);
            color: white;
            text-decoration: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            font-weight: 500;
            border: none;
            cursor: pointer;
            font-size: 0.875rem;
            transition: all 0.15s;
            width: 100%;
        }
        .btn:hover { filter: brightness(1.1); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .btn-outline {
            background: none;
            border: 1px solid var(--border);
            color: var(--text-secondary);
        }
        .btn-outline:hover { background: var(--bg-tertiary); color: var(--text-primary); }
        .code-box {
            background: var(--bg-primary);
            border: 2px dashed var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        .code {
            font-family: ui-monospace, monospace;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: 0.25em;
            color: var(--accent);
        }
        .code-label { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.5rem; }
        .status {
            padding: 0.875rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
            font-size: 0.875rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .status.success {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            color: var(--success);
        }
        .status.pending {
            background: rgba(251, 191, 36, 0.1);
            border: 1px solid rgba(251, 191, 36, 0.2);
            color: var(--warning);
        }
        .status.error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: var(--error);
        }
        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .user-card {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--success);
            border-radius: 8px;
            margin-bottom: 1.5rem;
        }
        .user-avatar {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            border: 2px solid var(--success);
        }
        .user-info h4 { font-size: 1rem; font-weight: 500; }
        .user-info p { font-size: 0.8125rem; color: var(--text-secondary); }
        .footer { margin-top: 1.5rem; text-align: center; color: var(--text-muted); font-size: 0.75rem; }
        .hidden { display: none; }
        .link { color: var(--accent); text-decoration: none; }
        .link:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="header">
                <span class="logo">🤖</span>
                <h1>Copilot API Proxy<span class="version">v{{VERSION}}</span></h1>
            </div>
            <p class="subtitle">Connect your GitHub account to enable the API</p>
            
            <!-- Authenticated State -->
            <div id="authenticated" class="hidden">
                <div class="status success">✓ Successfully connected to GitHub</div>
                <div class="user-card">
                    <img id="user-avatar" src="" alt="Avatar" class="user-avatar">
                    <div class="user-info">
                        <h4 id="user-name">-</h4>
                        <p id="user-login">@-</p>
                    </div>
                </div>
                <a href="/dashboard" class="btn">Open Dashboard →</a>
                <button onclick="logout()" class="btn btn-outline" style="margin-top: 0.75rem;">Sign out & Reconnect</button>
            </div>
            
            <!-- Not Authenticated State -->
            <div id="unauthenticated">
                <div class="step">
                    <h3><span class="step-number">1</span> Start Authentication</h3>
                    <p>Click below to generate a device code. You'll then sign in with your GitHub account.</p>
                    <button id="start-btn" onclick="startAuth()" class="btn">🔐 Connect GitHub Account</button>
                </div>
            </div>
            
            <!-- Pending Auth State -->
            <div id="pending" class="hidden">
                <div class="status pending">
                    <span class="spinner"></span>
                    <span>Waiting for authorization...</span>
                </div>
                
                <div class="step">
                    <h3><span class="step-number">1</span> Visit GitHub</h3>
                    <p>Open this URL in your browser:</p>
                    <a id="verify-url" href="" target="_blank" class="btn" style="margin-bottom: 0.5rem;">Open github.com/login/device →</a>
                </div>
                
                <div class="step">
                    <h3><span class="step-number">2</span> Enter Code</h3>
                    <p>Enter this code on the GitHub page:</p>
                    <div class="code-box">
                        <div class="code-label">YOUR CODE</div>
                        <div class="code" id="device-code">----</div>
                    </div>
                    <button onclick="copyCode()" class="btn btn-outline">📋 Copy Code</button>
                </div>
                
                <div class="step">
                    <h3><span class="step-number">3</span> Authorize</h3>
                    <p>After entering the code, click "Authorize" on GitHub. This page will update automatically.</p>
                </div>
                
                <button onclick="cancelAuth()" class="btn btn-outline">Cancel</button>
            </div>
            
            <!-- Error State -->
            <div id="error" class="hidden">
                <div class="status error" id="error-message">An error occurred</div>
                <button onclick="resetAuth()" class="btn">Try Again</button>
            </div>
            
            <div class="footer">
                Secure authentication via GitHub Device Flow<br>
                <a href="https://github.com" target="_blank" class="link">github.com</a>
            </div>
        </div>
    </div>
    
    <script>
        let pollInterval = null;
        let pollDelay = 6000;  // Default 6 seconds (GitHub requires 5+ seconds)
        let deviceCode = null;
        
        async function checkAuthStatus() {
            try {
                const resp = await fetch('/auth/status');
                const data = await resp.json();
                
                if (data.authenticated) {
                    showAuthenticated(data.user);
                } else if (data.pending && data.device) {
                    // Resume pending auth flow
                    deviceCode = data.device.user_code;
                    document.getElementById('device-code').textContent = data.device.user_code || '----';
                    document.getElementById('verify-url').href = data.device.verification_uri || 'https://github.com/login/device';
                    showPending();
                    startPolling();
                } else {
                    showUnauthenticated();
                }
            } catch (e) {
                console.error('Failed to check auth status:', e);
            }
        }
        
        async function startAuth() {
            const btn = document.getElementById('start-btn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner" style="margin-right: 0.5rem;"></span> Starting...';
            
            try {
                const resp = await fetch('/auth/start', { method: 'POST' });
                const data = await resp.json();
                
                console.log('Auth start response:', data);
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                if (!data.user_code || !data.verification_uri) {
                    showError('Invalid response from server. Please try again.');
                    return;
                }
                
                deviceCode = data.user_code;
                document.getElementById('device-code').textContent = data.user_code;
                document.getElementById('verify-url').href = data.verification_uri;
                
                // Use interval from server (convert seconds to milliseconds)
                if (data.interval) {
                    pollDelay = data.interval * 1000;
                }
                
                showPending();
                startPolling();
            } catch (e) {
                showError('Failed to start authentication: ' + e.message);
            }
        }
        
        function startPolling() {
            if (pollInterval) clearInterval(pollInterval);
            // First poll immediately, then respect the interval
            pollAuth();
            pollInterval = setInterval(pollAuth, pollDelay);
        }
        
        function stopPolling() {
            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }
        }
        
        async function pollAuth() {
            try {
                const resp = await fetch('/auth/poll', { method: 'POST' });
                const data = await resp.json();
                
                console.log('Poll response:', data);
                
                if (data.status === 'success') {
                    stopPolling();
                    showAuthenticated(data.user);
                } else if (data.status === 'expired') {
                    stopPolling();
                    showError('Device code expired. Please try again.');
                } else if (data.status === 'denied') {
                    stopPolling();
                    showError('Access was denied. Please try again.');
                } else if (data.status === 'error') {
                    stopPolling();
                    showError(data.error || 'An error occurred');
                } else if (data.status === 'slow_down' && data.interval) {
                    // GitHub is telling us to slow down - restart polling with new interval
                    stopPolling();
                    pollDelay = data.interval * 1000;
                    console.log('Slowing down, new interval:', pollDelay);
                    pollInterval = setInterval(pollAuth, pollDelay);
                }
                // pending - keep polling at current rate
            } catch (e) {
                console.error('Poll error:', e);
            }
        }
        
        function copyCode() {
            if (deviceCode) {
                const btn = event.target;
                // Try modern clipboard API first, fallback to execCommand for HTTP
                if (navigator.clipboard && window.isSecureContext) {
                    navigator.clipboard.writeText(deviceCode).then(() => {
                        btn.textContent = '✓ Copied!';
                        setTimeout(() => btn.textContent = '📋 Copy Code', 2000);
                    }).catch(err => fallbackCopy(deviceCode, btn));
                } else {
                    fallbackCopy(deviceCode, btn);
                }
            }
        }
        
        function fallbackCopy(text, btn) {
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            try {
                document.execCommand('copy');
                btn.textContent = '✓ Copied!';
                setTimeout(() => btn.textContent = '📋 Copy Code', 2000);
            } catch (err) {
                console.error('Copy failed:', err);
                alert('Copy failed. Please manually copy: ' + text);
            }
            document.body.removeChild(textArea);
        }
        
        async function cancelAuth() {
            stopPolling();
            try {
                await fetch('/auth/cancel', { method: 'POST' });
            } catch (e) {}
            showUnauthenticated();
        }
        
        async function logout() {
            try {
                await fetch('/auth/logout', { method: 'POST' });
            } catch (e) {}
            showUnauthenticated();
        }
        
        function resetAuth() {
            showUnauthenticated();
        }
        
        function showAuthenticated(user) {
            document.getElementById('authenticated').classList.remove('hidden');
            document.getElementById('unauthenticated').classList.add('hidden');
            document.getElementById('pending').classList.add('hidden');
            document.getElementById('error').classList.add('hidden');
            
            if (user) {
                document.getElementById('user-avatar').src = user.avatar_url || 'https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png';
                document.getElementById('user-name').textContent = user.name || user.login;
                document.getElementById('user-login').textContent = '@' + user.login;
            }
        }
        
        function showUnauthenticated() {
            document.getElementById('authenticated').classList.add('hidden');
            document.getElementById('unauthenticated').classList.remove('hidden');
            document.getElementById('pending').classList.add('hidden');
            document.getElementById('error').classList.add('hidden');
            
            const btn = document.getElementById('start-btn');
            btn.disabled = false;
            btn.innerHTML = '🔐 Connect GitHub Account';
        }
        
        function showPending() {
            document.getElementById('authenticated').classList.add('hidden');
            document.getElementById('unauthenticated').classList.add('hidden');
            document.getElementById('pending').classList.remove('hidden');
            document.getElementById('error').classList.add('hidden');
        }
        
        function showError(message) {
            document.getElementById('authenticated').classList.add('hidden');
            document.getElementById('unauthenticated').classList.add('hidden');
            document.getElementById('pending').classList.add('hidden');
            document.getElementById('error').classList.remove('hidden');
            document.getElementById('error-message').textContent = '❌ ' + message;
        }
        
        // Check initial auth status
        checkAuthStatus();
    </script>
</body>
</html>
"""

@app.get("/setup", response_class=HTMLResponse)
async def setup_page():
    """Show the authentication setup page."""
    return SETUP_HTML.replace("{{VERSION}}", VERSION)

@app.get("/auth/status")
async def auth_status():
    """Get current authentication status."""
    if state["github_token"] and state["github_user"]:
        return {
            "authenticated": True,
            "user": {
                "login": state["github_user"].get("login"),
                "name": state["github_user"].get("name"),
                "avatar_url": state["github_user"].get("avatar_url"),
            }
        }
    elif state["auth_in_progress"] and state["device_code"]:
        return {
            "authenticated": False,
            "pending": True,
            "device": {
                "user_code": state["device_code"].get("user_code"),
                "verification_uri": state["device_code"].get("verification_uri"),
            }
        }
    else:
        return {"authenticated": False, "pending": False}

@app.post("/auth/start")
async def auth_start():
    """Start the GitHub device authentication flow."""
    try:
        device = start_device_flow()
        return {
            "user_code": device.get("user_code"),
            "verification_uri": device.get("verification_uri"),
            "expires_in": device.get("expires_in"),
            "interval": state["poll_interval"],
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/auth/poll")
async def auth_poll():
    """Poll for authentication completion."""
    return poll_device_flow()

@app.post("/auth/cancel")
async def auth_cancel():
    """Cancel pending authentication."""
    state["device_code"] = None
    state["auth_in_progress"] = False
    return {"status": "cancelled"}

@app.post("/auth/logout")
async def auth_logout():
    """Logout and clear tokens."""
    state["github_token"] = None
    state["copilot_token"] = None
    state["copilot_token_expires_at"] = 0
    state["github_user"] = None
    state["models"] = None
    
    # Clear token file
    if os.path.exists(TOKEN_FILE):
        try:
            os.remove(TOKEN_FILE)
        except Exception:
            pass
    
    return {"status": "logged_out"}

# =============================================================================
# Dashboard
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Copilot API Proxy</title>
    <style>
        :root {
            --bg-primary: #0f0f0f;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #242424;
            --border: #2e2e2e;
            --text-primary: #fafafa;
            --text-secondary: #a0a0a0;
            --text-muted: #6b6b6b;
            --accent: #3b82f6;
            --success: #10b981;
            --error: #ef4444;
            --warning: #f59e0b;
            --copilot: #8b5cf6;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; }
        .logo { display: flex; align-items: center; gap: 0.75rem; }
        .logo-icon { font-size: 1.5rem; }
        h1 { font-size: 1.5rem; font-weight: 600; }
        .version { font-size: 0.625rem; opacity: 0.5; vertical-align: middle; margin-left: 0.5rem; }
        .status { display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem; color: var(--text-secondary); }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--success); animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .stats { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin-bottom: 2rem; }
        @media (max-width: 1024px) { .stats { grid-template-columns: repeat(3, 1fr); } }
        @media (max-width: 640px) { .stats { grid-template-columns: repeat(2, 1fr); } }
        .stat-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.25rem;
        }
        .stat-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
        .stat-value { font-size: 1.75rem; font-weight: 600; font-variant-numeric: tabular-nums; }
        .stat-value.success { color: var(--success); }
        .stat-value.error { color: var(--error); }
        .stat-value.copilot { color: var(--copilot); }
        .stat-sub { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
        .section { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 8px; }
        .section-header { padding: 1rem 1.25rem; border-bottom: 1px solid var(--border); font-size: 0.875rem; font-weight: 500; color: var(--text-secondary); display: flex; justify-content: space-between; align-items: center; }
        .section-content { padding: 1rem 1.25rem; }
        .models { display: flex; flex-wrap: wrap; gap: 0.5rem; max-height: 200px; overflow-y: auto; }
        .model-chip {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            padding: 0.375rem 0.75rem;
            border-radius: 4px;
            font-size: 0.8125rem;
            font-family: ui-monospace, monospace;
            color: var(--text-secondary);
            transition: all 0.15s;
        }
        .model-chip:hover { border-color: var(--copilot); color: var(--text-primary); }
        .model-chip.premium { border-color: var(--warning); color: var(--warning); }
        table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
        th { text-align: left; padding: 0.75rem 0; color: var(--text-muted); font-weight: 500; border-bottom: 1px solid var(--border); }
        td { padding: 0.75rem 0; border-bottom: 1px solid var(--border); color: var(--text-secondary); }
        tr:last-child td { border-bottom: none; }
        .mono { font-family: ui-monospace, monospace; }
        .success-text { color: var(--success); }
        .error-text { color: var(--error); }
        .empty { color: var(--text-muted); font-style: italic; padding: 2rem; text-align: center; }
        .refresh { background: none; border: 1px solid var(--border); color: var(--text-secondary); padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-size: 0.875rem; transition: all 0.15s; }
        .refresh:hover { background: var(--bg-tertiary); color: var(--text-primary); }
        .progress-bar { background: var(--bg-tertiary); height: 8px; border-radius: 4px; overflow: hidden; margin-top: 0.5rem; }
        .progress-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
        .progress-fill.success { background: var(--success); }
        .progress-fill.warning { background: var(--warning); }
        .progress-fill.error { background: var(--error); }
        .quota-item { padding: 0.75rem 0; border-bottom: 1px solid var(--border); }
        .quota-item:last-child { border-bottom: none; }
        .quota-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem; }
        .quota-label { font-size: 0.875rem; color: var(--text-secondary); }
        .quota-value { font-size: 0.875rem; font-weight: 500; color: var(--text-primary); }
        .endpoints { font-family: ui-monospace, monospace; font-size: 0.8125rem; }
        .endpoint { display: flex; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid var(--border); }
        .endpoint:last-child { border-bottom: none; }
        .method { padding: 0.125rem 0.5rem; border-radius: 3px; font-size: 0.6875rem; font-weight: 600; margin-right: 0.75rem; min-width: 50px; text-align: center; }
        .method.get { background: rgba(16, 185, 129, 0.2); color: var(--success); }
        .method.post { background: rgba(59, 130, 246, 0.2); color: var(--accent); }
        .tabs { display: flex; gap: 0.5rem; }
        .tab { padding: 0.375rem 0.75rem; border-radius: 4px; font-size: 0.75rem; cursor: pointer; color: var(--text-muted); transition: all 0.15s; }
        .tab:hover { color: var(--text-secondary); }
        .tab.active { background: var(--bg-tertiary); color: var(--text-primary); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <span class="logo-icon">🤖</span>
                <h1>Copilot API Proxy<span class="version">v{{VERSION}}</span></h1>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div class="status"><span class="status-dot"></span><span id="status-text">Connected</span></div>
                <div id="user-info" style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem; color: var(--text-secondary);">
                    <img id="user-avatar" src="" alt="" style="width: 24px; height: 24px; border-radius: 50%; display: none;">
                    <span id="user-login"></span>
                </div>
                <button class="refresh" onclick="fetchData()">↻ Refresh</button>
                <a href="/setup" class="refresh" style="text-decoration: none;">⚙ Settings</a>
            </div>
        </header>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Requests</div>
                <div class="stat-value" id="total">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Successful</div>
                <div class="stat-value success" id="success">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Failed</div>
                <div class="stat-value error" id="failed">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Input ↑</div>
                <div class="stat-value" id="input-tokens">-</div>
                <div class="stat-sub">tokens</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Output ↓</div>
                <div class="stat-value" id="output-tokens">-</div>
                <div class="stat-sub">tokens</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="section">
                <div class="section-header">
                    <span>🎯 Copilot Quota</span>
                    <span id="plan-badge" style="font-size: 0.75rem; padding: 0.25rem 0.5rem; background: var(--copilot); color: white; border-radius: 4px;">-</span>
                </div>
                <div class="section-content" id="quota-info">
                    <div class="empty">Loading quota information...</div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">
                    <span>🔗 API Endpoints</span>
                </div>
                <div class="section-content endpoints">
                    <div class="endpoint"><span class="method get">GET</span>/v1/models</div>
                    <div class="endpoint"><span class="method post">POST</span>/v1/chat/completions</div>
                    <div class="endpoint"><span class="method post">POST</span>/v1/embeddings</div>
                    <div class="endpoint"><span class="method post">POST</span>/v1/messages</div>
                    <div class="endpoint"><span class="method get">GET</span>/usage</div>
                    <div class="endpoint"><span class="method get">GET</span>/stats</div>
                </div>
            </div>
        </div>
        
        <div class="section" style="margin-bottom: 1.5rem;">
            <div class="section-header">
                <span>📦 Available Models</span>
                <span id="model-count" style="font-size: 0.75rem; color: var(--text-muted);">-</span>
            </div>
            <div class="section-content">
                <div class="models" id="models">
                    <div class="empty">Loading models...</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <span>📜 Recent Requests</span>
                <div class="tabs">
                    <span class="tab active" onclick="filterRequests('all')">All</span>
                    <span class="tab" onclick="filterRequests('success')">Success</span>
                    <span class="tab" onclick="filterRequests('error')">Errors</span>
                </div>
            </div>
            <div class="section-content" style="padding: 0;">
                <table style="padding: 0 1.25rem;">
                    <thead>
                        <tr>
                            <th style="padding-left: 1.25rem;">Time</th>
                            <th>Model</th>
                            <th>Status</th>
                            <th style="padding-right: 1.25rem; text-align: right;">Tokens (↑/↓)</th>
                        </tr>
                    </thead>
                    <tbody id="history"></tbody>
                </table>
                <div class="empty" id="empty-state">No requests yet</div>
            </div>
        </div>
    </div>
    
    <script>
        let allRequests = [];
        let currentFilter = 'all';
        
        function formatNumber(num) {
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
            return num.toLocaleString();
        }
        
        function getProgressClass(percent) {
            if (percent >= 90) return 'error';
            if (percent >= 70) return 'warning';
            return 'success';
        }
        
        function renderQuota(usage) {
            if (!usage || !usage.quota_snapshots) {
                return '<div class="empty">Unable to load quota information</div>';
            }
            
            const quotas = [
                { name: 'Premium Interactions', data: usage.quota_snapshots.premium_interactions },
                { name: 'Chat', data: usage.quota_snapshots.chat },
                { name: 'Completions', data: usage.quota_snapshots.completions }
            ];
            
            return quotas.filter(q => q.data).map(q => {
                const used = q.data.entitlement - q.data.remaining;
                const total = q.data.entitlement;
                const percent = total > 0 ? (used / total) * 100 : 0;
                const remaining = q.data.percent_remaining || 0;
                
                return `
                    <div class="quota-item">
                        <div class="quota-header">
                            <span class="quota-label">${q.name}</span>
                            <span class="quota-value">${formatNumber(used)} / ${formatNumber(total)}</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill ${getProgressClass(percent)}" style="width: ${percent}%"></div>
                        </div>
                    </div>
                `;
            }).join('') + `
                <div class="quota-item" style="border-bottom: none; padding-bottom: 0;">
                    <div class="quota-header">
                        <span class="quota-label">Reset Date</span>
                        <span class="quota-value">${usage.quota_reset_date || 'N/A'}</span>
                    </div>
                </div>
            `;
        }
        
        function renderHistory() {
            const filtered = currentFilter === 'all' 
                ? allRequests 
                : allRequests.filter(r => currentFilter === 'success' ? r.success : !r.success);
            
            const emptyState = document.getElementById('empty-state');
            const historyTable = document.getElementById('history');
            
            if (filtered.length === 0) {
                emptyState.style.display = 'block';
                historyTable.innerHTML = '';
            } else {
                emptyState.style.display = 'none';
                historyTable.innerHTML = filtered.map(r => {
                    const time = new Date(r.timestamp);
                    const timeStr = time.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'});
                    
                    let tokenDisplay = '—';
                    if (r.input_tokens > 0 || r.output_tokens > 0) {
                        tokenDisplay = `<span style="color: #94a3b8">${r.input_tokens}</span> <span style="color: #4ade80">↑</span> <span style="color: #475569">/</span> <span style="color: #94a3b8">${r.output_tokens}</span> <span style="color: #60a5fa">↓</span>`;
                    }
                    
                    return `
                        <tr>
                            <td style="padding-left: 1.25rem;" class="mono">${timeStr}</td>
                            <td class="mono">${r.model}</td>
                            <td class="${r.success ? 'success-text' : 'error-text'}">${r.success ? '✓ OK' : '✗ Error'}</td>
                            <td style="padding-right: 1.25rem; text-align: right;" class="mono">${tokenDisplay}</td>
                        </tr>
                    `;
                }).join('');
            }
        }
        
        function filterRequests(filter) {
            currentFilter = filter;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab:nth-child(${filter === 'all' ? 1 : filter === 'success' ? 2 : 3})`).classList.add('active');
            renderHistory();
        }
        
        async function fetchData() {
            try {
                const [stats, models, usage, auth] = await Promise.all([
                    fetch('/stats').then(r => r.json()),
                    fetch('/v1/models').then(r => r.json()),
                    fetch('/usage').then(r => r.json()),
                    fetch('/auth/status').then(r => r.json())
                ]);
                
                // Update user info
                if (auth.authenticated && auth.user) {
                    document.getElementById('user-avatar').src = auth.user.avatar_url;
                    document.getElementById('user-avatar').style.display = 'block';
                    document.getElementById('user-login').textContent = '@' + auth.user.login;
                }
                
                // Update stats
                document.getElementById('status-text').textContent = 'Connected';
                document.getElementById('total').textContent = formatNumber(stats.total_requests);
                document.getElementById('success').textContent = formatNumber(stats.successful_requests);
                document.getElementById('failed').textContent = formatNumber(stats.failed_requests);
                document.getElementById('input-tokens').textContent = formatNumber(stats.total_input_tokens);
                document.getElementById('output-tokens').textContent = formatNumber(stats.total_output_tokens);
                
                // Update models
                const modelList = models.data || [];
                document.getElementById('model-count').textContent = modelList.length + ' models';
                
                const premiumModels = ['gpt-5', 'gpt-5.1', 'gpt-5.2', 'claude-opus', 'claude-sonnet-4', 'o1', 'o3'];
                document.getElementById('models').innerHTML = modelList
                    .map(m => {
                        const isPremium = premiumModels.some(p => m.id.includes(p));
                        return `<span class="model-chip ${isPremium ? 'premium' : ''}">${m.id}</span>`;
                    })
                    .join('');
                
                // Update quota
                if (usage.copilot_plan) {
                    document.getElementById('plan-badge').textContent = usage.copilot_plan;
                    document.getElementById('quota-info').innerHTML = renderQuota(usage);
                } else {
                    document.getElementById('quota-info').innerHTML = '<div class="empty">Unable to load quota</div>';
                }
                
                // Update request history
                allRequests = (stats.recent_requests || []).slice(-20).reverse();
                renderHistory();
                
            } catch (e) {
                document.getElementById('status-text').textContent = 'Error';
                console.error('Failed to fetch data:', e);
            }
        }
        
        // Polling with visibility API
        let pollInterval = null;
        
        function startPolling() {
            if (!pollInterval) {
                fetchData();
                pollInterval = setInterval(fetchData, 10000);
            }
        }
        
        function stopPolling() {
            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }
        }
        
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                stopPolling();
            } else {
                startPolling();
            }
        });
        
        if (!document.hidden) {
            startPolling();
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the usage dashboard or redirect to setup."""
    # Redirect to setup if not authenticated
    if not state["github_token"]:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/setup", status_code=302)
    
    return DASHBOARD_HTML.replace("{{VERSION}}", VERSION)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
