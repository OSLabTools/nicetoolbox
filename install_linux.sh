#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root so relative paths work wherever the script is called from
cd "$(dirname "$0")"

echo ""
echo "************************************************************"
echo "  NICE Toolbox Installer"
echo "************************************************************"
echo ""

# ---------------------------------------------------------------------------
# Step 1 - Install uv if not already on PATH
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "[1/3] uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the rest of this session
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo "ERROR: uv was installed but is not on PATH."
        echo "Please open a new terminal and run \"$PWD/$(basename "$0")\" again with the same options."
        exit 1
    fi
    echo "uv installed successfully."
else
    echo "[1/3] uv found: OK"
fi

# ---------------------------------------------------------------------------
# Step 2 - Use uv to install Python 3.10 (no separate Python installer needed)
# ---------------------------------------------------------------------------
echo "[2/3] Ensuring Python 3.10 is available via uv..."
uv python install 3.10
echo "Python 3.10 ready."

# ---------------------------------------------------------------------------
# Step 3 - Run the main installer script
# ---------------------------------------------------------------------------
echo "[3/3] Running the installer..."
echo ""
rc=0
uv run --no-project --python 3.10 python -m nicetoolbox.installation.install "$@" || rc=$?
if [ "$rc" -ne 0 ]; then
    echo ""
    echo "Installation finished with errors. Check the output above for details."
fi
exit "$rc"
