#!/usr/bin/env bash
# Setup script-doctor dependencies for GEPA PuzzleScript evaluation.
#
# This script:
#   1. Clones script-doctor if not present
#   2. Clones the PuzzleScript submodule
#   3. Creates a venv with uv and installs deps
#   4. Builds the C++ extension
#   5. Downloads Node.js if not available
#
# Usage:
#   bash scripts/setup_script_doctor.sh [--script-doctor-path PATH]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_SD_PATH="$(dirname "$PROJECT_ROOT")/script-doctor"
SD_PATH="${1:-$DEFAULT_SD_PATH}"
NODE_VERSION="v20.18.0"
NODE_DIR="$HOME/node-${NODE_VERSION}-linux-x64"

echo "=== Setup script-doctor for GEPA ==="
echo "  script-doctor path: $SD_PATH"
echo "  project root: $PROJECT_ROOT"
echo ""

# --- 1. Clone script-doctor ---
if [ ! -d "$SD_PATH" ]; then
    echo "[1/5] Cloning script-doctor..."
    git clone https://github.com/smearle/script-doctor.git "$SD_PATH"
else
    echo "[1/5] script-doctor already cloned at $SD_PATH"
fi

# --- 2. Clone PuzzleScript submodule ---
if [ ! -d "$SD_PATH/PuzzleScript" ]; then
    echo "[2/5] Cloning PuzzleScript engine source..."
    git clone https://github.com/increpare/PuzzleScript.git "$SD_PATH/PuzzleScript"
else
    echo "[2/5] PuzzleScript already present"
fi

# --- 3. Create venv and install deps ---
if [ ! -d "$SD_PATH/.venv" ]; then
    echo "[3/5] Creating venv and installing deps..."
    if command -v uv &>/dev/null; then
        cd "$SD_PATH"
        uv venv --python 3.12
        uv pip install jax lark numpy py-cpuinfo pybind11 imageio setuptools wheel \
            python-dotenv chex openai tiktoken einops flax hydra-core Pillow javascript
    else
        echo "ERROR: uv not found. Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
else
    echo "[3/5] venv already exists"
fi

# --- 4. Build C++ extension ---
SO_FILE=$(find "$SD_PATH/puzzlescript_cpp" -name "_puzzlescript_cpp*.so" 2>/dev/null | head -1)
if [ -z "$SO_FILE" ]; then
    echo "[4/5] Building C++ extension..."
    cd "$SD_PATH"
    .venv/bin/python setup_cpp.py build_ext --inplace
else
    echo "[4/5] C++ extension already built: $SO_FILE"
fi

# --- 5. Ensure Node.js is available ---
if ! command -v node &>/dev/null && [ ! -f "$NODE_DIR/bin/node" ]; then
    echo "[5/5] Downloading Node.js $NODE_VERSION..."
    curl -sL "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz" \
        | tar -xJ -C "$(dirname "$NODE_DIR")"
    echo "  Add to PATH: export PATH=\"$NODE_DIR/bin:\$PATH\""
elif [ -f "$NODE_DIR/bin/node" ]; then
    echo "[5/5] Node.js available at $NODE_DIR/bin/node"
else
    echo "[5/5] Node.js already on PATH: $(which node)"
fi

# Create data dirs used by the parser
mkdir -p "$SD_PATH/data/game_trees"
mkdir -p "$SD_PATH/data/pretty_trees"
mkdir -p "$SD_PATH/data/simplified_games"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To use GEPA evaluator, ensure Node.js is on PATH:"
echo "  export PATH=\"$NODE_DIR/bin:\$PATH\""
echo ""
echo "Then run:"
echo "  python scripts/gepa_evaluate.py --mode=benchmark"
