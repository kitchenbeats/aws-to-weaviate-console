#!/bin/bash
#
# AWS to Weaviate Console - Installation Script
# ---------------------------------------

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print styled header
echo -e "${BOLD}${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              AWS to Weaviate Console                       ║"
echo "║                INSTALLATION SCRIPT                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Ensure script is run from the correct directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    exit 1
fi

# Check Python version is 3.8 or higher
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}Error: Python 3.8 or higher is required. Current version: $PYTHON_VERSION${NC}"
    exit 1
fi

# Create virtual environment
VENV_PATH="./venv"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo -e "${BLUE}Installing project dependencies...${NC}"
pip install -r requirements.txt

# Create startup script
echo -e "${BLUE}Creating startup script...${NC}"
cat > startup.sh << 'EOL'
#!/bin/bash
source venv/bin/activate
streamlit run main.py
EOL
chmod +x startup.sh

# Create VSCode settings
echo -e "${BLUE}Creating VSCode settings...${NC}"
mkdir -p .vscode
cat > .vscode/settings.json << EOL
{
    "python.pythonPath": "${SCRIPT_DIR}/venv/bin/python",
    "python.venvPath": "${SCRIPT_DIR}/venv",
    "python.defaultInterpreterPath": "${SCRIPT_DIR}/venv/bin/python",
    "python.terminal.activateEnvironment": true
}
EOL

# Final message
echo -e "${GREEN}Installation complete!${NC}"
echo -e "\nTo start the application:"
echo -e "1. Activate the virtual environment: ${BOLD}source venv/bin/activate${NC}"
echo -e "2. Run the startup script: ${BOLD}./startup.sh${NC}"

# Option to launch immediately
read -p "Would you like to launch the application now? (y/n): " launch_app
if [[ $launch_app =~ ^[Yy]$ ]]; then
    ./startup.sh
fi