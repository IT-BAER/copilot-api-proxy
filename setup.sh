#!/bin/bash
set -e

# Configuration
APP_NAME="copilot-api-proxy"
APP_DIR="/opt/$APP_NAME"
SERVICE_NAME="$APP_NAME.service"
PORT=4141

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Copilot API Proxy Setup ===${NC}"

# Check for root
if [ "$EUID" -ne 0 ]; then 
  echo -e "${RED}Please run as root (sudo ./setup.sh)${NC}"
  exit 1
fi

# Install dependencies
echo -e "${GREEN}Installing system dependencies...${NC}"
apt-get update
apt-get install -y python3 python3-venv python3-pip git

# Create directory
if [ ! -d "$APP_DIR" ]; then
    echo -e "${GREEN}Creating directory $APP_DIR...${NC}"
    mkdir -p "$APP_DIR"
fi

# Copy files
echo -e "${GREEN}Copying application files...${NC}"
# Use rsync if available, or just check paths to avoid "same file" error
if [ "$(realpath .)" != "$(realpath "$APP_DIR")" ]; then
    cp -r app.py requirements.txt "$APP_NAME.service" "$APP_DIR/"
else
    echo -e "${YELLOW}Already in $APP_DIR, skipping redundant copy.${NC}"
fi

# Create data directory with proper permissions
mkdir -p "$APP_DIR/data"
chmod 700 "$APP_DIR/data"

# Setup Virtual Environment
echo -e "${GREEN}Setting up Python environment...${NC}"
cd "$APP_DIR"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Install requirements
echo -e "${GREEN}Installing Python dependencies...${NC}"
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# Setup Service
echo -e "${GREEN}Installing systemd service...${NC}"
cp "$APP_NAME.service" "/etc/systemd/system/$SERVICE_NAME"
systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

# Stop existing service if running
systemctl stop "$SERVICE_NAME" 2>/dev/null || true

# Start the service
systemctl start "$SERVICE_NAME"

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
systemctl status "$SERVICE_NAME" --no-pager || true
echo ""
echo -e "${YELLOW}Manage service with:${NC}"
echo "  sudo systemctl start/stop/restart $APP_NAME"
echo "  sudo journalctl -u $APP_NAME -f"
echo ""
echo -e "${GREEN}Complete authentication at:${NC}"
echo "  http://$(hostname -I | awk '{print $1}'):$PORT/setup"
echo "  http://localhost:$PORT/setup"
echo ""
echo -e "${GREEN}API Endpoints:${NC}"
echo "  OpenAI:    http://localhost:$PORT/v1/chat/completions"
echo "  Anthropic: http://localhost:$PORT/v1/messages"
echo ""
