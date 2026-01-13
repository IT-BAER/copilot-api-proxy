#!/bin/bash
#
# Copilot API Proxy - One-liner Installer for Debian/Ubuntu
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/IT-BAER/copilot-api-proxy/main/install.sh | sudo bash
#
set -e

# =============================================================================
# Configuration
# =============================================================================
APP_NAME="copilot-api-proxy"
APP_DIR="/opt/$APP_NAME"
SERVICE_FILE="/etc/systemd/system/$APP_NAME.service"
REPO_URL="https://github.com/IT-BAER/copilot-api-proxy"
BRANCH="main"
PORT=4141

# =============================================================================
# Colors
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# =============================================================================
# Functions
# =============================================================================
print_banner() {
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║           ${BOLD}${GREEN}Copilot API Proxy Installer${NC}${CYAN}                        ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║   OpenAI & Anthropic Compatible API for GitHub Copilot       ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} ${BOLD}$1${NC}"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root (use sudo)"
        echo ""
        echo "Usage: curl -fsSL https://raw.githubusercontent.com/IT-BAER/copilot-api-proxy/main/install.sh | sudo bash"
        exit 1
    fi
}

check_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
    else
        log_error "Cannot detect OS. This script supports Debian/Ubuntu."
        exit 1
    fi

    case $OS in
        debian|ubuntu|linuxmint|pop)
            log_info "Detected: $PRETTY_NAME"
            ;;
        *)
            log_warn "Untested OS: $OS. Proceeding anyway (Debian-based assumed)..."
            ;;
    esac
}

install_dependencies() {
    log_step "Installing system dependencies..."
    
    apt-get update -qq
    apt-get install -y -qq python3 python3-venv python3-pip curl git > /dev/null
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log_info "Python version: $PYTHON_VERSION"
    
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
        log_info "Python version OK (>= 3.11)"
    else
        log_warn "Python version is $PYTHON_VERSION. Recommended: 3.11+"
    fi
}

download_application() {
    log_step "Downloading $APP_NAME..."
    
    # Remove existing installation
    if [ -d "$APP_DIR" ]; then
        log_warn "Existing installation found. Backing up data..."
        if [ -f "$APP_DIR/data/github_token.json" ]; then
            cp "$APP_DIR/data/github_token.json" /tmp/github_token.json.bak
            log_info "Token backup saved to /tmp/github_token.json.bak"
        fi
        rm -rf "$APP_DIR"
    fi
    
    # Clone repository
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$APP_DIR" > /dev/null 2>&1
    
    # Restore token if backed up
    if [ -f /tmp/github_token.json.bak ]; then
        mkdir -p "$APP_DIR/data"
        mv /tmp/github_token.json.bak "$APP_DIR/data/github_token.json"
        chmod 600 "$APP_DIR/data/github_token.json"
        log_info "Token restored from backup"
    fi
    
    log_info "Downloaded to $APP_DIR"
}

setup_python_env() {
    log_step "Setting up Python virtual environment..."
    
    cd "$APP_DIR"
    
    # Create virtual environment
    python3 -m venv venv
    
    # Upgrade pip and install requirements
    ./venv/bin/pip install --upgrade pip -q
    ./venv/bin/pip install -r requirements.txt -q
    
    log_info "Python dependencies installed"
}

create_data_directory() {
    log_step "Creating data directory..."
    
    mkdir -p "$APP_DIR/data"
    chmod 700 "$APP_DIR/data"
    
    log_info "Data directory created with secure permissions"
}

install_systemd_service() {
    log_step "Installing systemd service..."
    
    # Create service file
    cat > "$SERVICE_FILE" << 'EOF'
[Unit]
Description=GitHub Copilot API Proxy (OpenAI & Anthropic Compatible)
Documentation=https://github.com/IT-BAER/copilot-api-proxy
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/copilot-api-proxy
Environment="PORT=4141"
Environment="HOST=0.0.0.0"
Environment="TOKEN_FILE=/opt/copilot-api-proxy/data/github_token.json"
Environment="ACCOUNT_TYPE=individual"
ExecStart=/opt/copilot-api-proxy/venv/bin/python -u app.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/copilot-api-proxy/data
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    systemctl daemon-reload
    
    # Enable service
    systemctl enable "$APP_NAME" > /dev/null 2>&1
    
    log_info "Systemd service installed and enabled"
}

start_service() {
    log_step "Starting service..."
    
    # Stop if running
    systemctl stop "$APP_NAME" 2>/dev/null || true
    
    # Start service
    systemctl start "$APP_NAME"
    
    # Wait a moment for startup
    sleep 2
    
    # Check status
    if systemctl is-active --quiet "$APP_NAME"; then
        log_info "Service started successfully"
    else
        log_error "Service failed to start. Check: journalctl -u $APP_NAME -n 50"
        exit 1
    fi
}

get_local_ip() {
    hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost"
}

print_success() {
    LOCAL_IP=$(get_local_ip)
    
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}║             ${BOLD}✓ Installation Complete!${NC}${GREEN}                          ║${NC}"
    echo -e "${GREEN}║                                                               ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BOLD}Next Steps:${NC}"
    echo ""
    echo -e "  1. ${CYAN}Complete GitHub authentication:${NC}"
    echo -e "     ${YELLOW}http://$LOCAL_IP:$PORT/setup${NC}"
    echo -e "     ${YELLOW}http://localhost:$PORT/setup${NC}"
    echo ""
    echo -e "  2. ${CYAN}View dashboard:${NC}"
    echo -e "     ${YELLOW}http://$LOCAL_IP:$PORT/${NC}"
    echo ""
    echo -e "${BOLD}API Endpoints:${NC}"
    echo ""
    echo -e "  OpenAI:    ${YELLOW}http://$LOCAL_IP:$PORT/v1/chat/completions${NC}"
    echo -e "  Anthropic: ${YELLOW}http://$LOCAL_IP:$PORT/v1/messages${NC}"
    echo -e "  Models:    ${YELLOW}http://$LOCAL_IP:$PORT/v1/models${NC}"
    echo ""
    echo -e "${BOLD}Service Management:${NC}"
    echo ""
    echo -e "  Status:   ${CYAN}sudo systemctl status $APP_NAME${NC}"
    echo -e "  Logs:     ${CYAN}sudo journalctl -u $APP_NAME -f${NC}"
    echo -e "  Restart:  ${CYAN}sudo systemctl restart $APP_NAME${NC}"
    echo -e "  Stop:     ${CYAN}sudo systemctl stop $APP_NAME${NC}"
    echo ""
    echo -e "${BOLD}Uninstall:${NC}"
    echo ""
    echo -e "  ${CYAN}sudo systemctl stop $APP_NAME && sudo systemctl disable $APP_NAME${NC}"
    echo -e "  ${CYAN}sudo rm $SERVICE_FILE && sudo rm -rf $APP_DIR${NC}"
    echo -e "  ${CYAN}sudo systemctl daemon-reload${NC}"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
main() {
    print_banner
    
    check_root
    check_os
    install_dependencies
    download_application
    create_data_directory
    setup_python_env
    install_systemd_service
    start_service
    print_success
}

main "$@"
