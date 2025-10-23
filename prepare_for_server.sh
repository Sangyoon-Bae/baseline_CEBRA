#!/bin/bash
# Prepare files for server transfer
# Usage: bash prepare_for_server.sh

echo "============================================"
echo "Preparing Allen Dataset files for server"
echo "============================================"

# Create a directory for server files
SERVER_DIR="allen_server_files"
mkdir -p "$SERVER_DIR"

echo ""
echo "Copying essential files to $SERVER_DIR/..."

# Copy configuration and scripts
cp allen_config.yaml "$SERVER_DIR/"
cp run_allen_cebra.py "$SERVER_DIR/"
cp simple_allen_test.py "$SERVER_DIR/"
cp ALLEN_DATASET_README.md "$SERVER_DIR/"

# Copy fixed kirby taxonomy file
mkdir -p "$SERVER_DIR/kirby/taxonomy"
cp kirby/taxonomy/core.py "$SERVER_DIR/kirby/taxonomy/"
cp kirby/taxonomy/__init__.py "$SERVER_DIR/kirby/taxonomy/" 2>/dev/null || true
cp kirby/taxonomy/task.py "$SERVER_DIR/kirby/taxonomy/" 2>/dev/null || true

echo "✅ Files copied to $SERVER_DIR/"
echo ""
echo "Files in $SERVER_DIR:"
ls -lh "$SERVER_DIR/"

echo ""
echo "============================================"
echo "Next steps:"
echo "============================================"
echo ""
echo "1. Transfer to server:"
echo "   scp -r $SERVER_DIR/ server:/path/to/CEBRA/"
echo ""
echo "2. On server, also copy your data directory:"
echo "   scp -r data/ server:/path/to/CEBRA/"
echo ""
echo "3. On server, run:"
echo "   cd /path/to/CEBRA/$SERVER_DIR"
echo "   python3 simple_allen_test.py  # Test data loading"
echo "   python3 run_allen_cebra.py --config allen_config.yaml --max_sessions 4"
echo ""
echo "4. If kirby errors occur on server:"
echo "   cp kirby/taxonomy/core.py /path/to/CEBRA/kirby/taxonomy/"
echo ""
echo "============================================"
echo "✅ Preparation complete!"
echo "============================================"
