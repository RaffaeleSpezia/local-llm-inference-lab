#!/bin/bash
# Switch tra versioni old/new dei file

if [ "$1" != "old" ] && [ "$1" != "new" ]; then
  echo "Usage: $0 {old|new}"
  echo "  old = usa versione .old"
  echo "  new = usa versione .new"
  exit 1
fi

VERSION="$1"
BASE_DIR="/home/lele/mi50_stack/mi50_chat_ui"

FILES=(
  "app/main.py"
  "app/prompt_formatter.py"
  "app/storage.py"
  "static/index.html"
)

echo "=== Switching to version: $VERSION ==="

for file in "${FILES[@]}"; do
  src="${BASE_DIR}/${file}.${VERSION}"
  dest="${BASE_DIR}/${file}"
  
  if [ -f "$src" ]; then
    cp "$src" "$dest"
    echo "✓ ${file} <- ${file}.${VERSION}"
  else
    echo "✗ ${src} not found"
  fi
done

echo ""
echo "✓ Switch completato! Riavvia servizio con: sudo systemctl restart mi50_chat_ui"
