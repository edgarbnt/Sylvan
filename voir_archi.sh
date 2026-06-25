#!/bin/zsh
# Carte vivante de l'architecture Sylvan. Valide la donnée puis sert la page + ouvre le navigateur.
# Usage: bash voir_archi.sh [port=8765]
set -e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
PORT=${1:-8765}
# Refuse de servir une carte invalide (anti-mensonge).
./env_pytorch_3.12/bin/python tools/archi_hud/validate_architecture.py
URL="http://127.0.0.1:${PORT}/tools/archi_hud/index.html"
echo "Archi-HUD → $URL  (Ctrl-C pour arrêter)"
( sleep 1; (xdg-open "$URL" >/dev/null 2>&1 || echo "Ouvre $URL dans ton navigateur") ) &
exec "$ROOT/env_pytorch_3.12/bin/python" -m http.server "$PORT" --bind 127.0.0.1
