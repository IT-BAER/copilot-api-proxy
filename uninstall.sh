#!/bin/bash
#
# Copilot API Proxy - Uninstaller
#
# Usage: sudo ./uninstall.sh
#
set -e

APP_NAME="copilot-api-proxy"
APP_DIR="/opt/$APP_NAME"
SERVICE_FILE="/etc/systemd/system/$APP_NAME.service"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Copilot API Proxy Uninstaller ===${NC}"
echo ""

if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (sudo ./uninstall.sh)${NC}"
    exit 1
fi

# Confirm
read -p "This will remove Copilot API Proxy. Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo -e "${GREEN}Stopping service...${NC}"
systemctl stop "$APP_NAME" 2>/dev/null || true

echo -e "${GREEN}Disabling service...${NC}"
systemctl disable "$APP_NAME" 2>/dev/null || true

echo -e "${GREEN}Removing service file...${NC}"
rm -f "$SERVICE_FILE"

echo -e "${GREEN}Reloading systemd...${NC}"
systemctl daemon-reload

# Ask about data
if [ -f "$APP_DIR/data/github_token.json" ]; then
    read -p "Remove saved GitHub token? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}Removing application directory...${NC}"
        rm -rf "$APP_DIR"
    else
        echo -e "${GREEN}Removing application (keeping data)...${NC}"
        find "$APP_DIR" -mindepth 1 ! -path "$APP_DIR/data*" -delete 2>/dev/null || true
        echo -e "${YELLOW}Token saved at: $APP_DIR/data/github_token.json${NC}"
    fi
else
    echo -e "${GREEN}Removing application directory...${NC}"
    rm -rf "$APP_DIR"
fi

echo ""
echo -e "${GREEN}✓ Uninstallation complete!${NC}"
