#!/bin/bash
# VOIR LA FORÊT — la configuration EXACTE qui a passé le gate d'arrangement (2026-07-24).
#
# Ce qui est à l'écran = ce que l'entité PERÇOIT, et rien d'autre :
#   * 45 massifs SOLIDES (bit 2) et PERCEPTIBLES (bit 7 + retina_color), vert foncé — teinte choisie
#     par la MESURE (fuite 0,0000 sur la requête rouge ; un tronc brun fuirait à 0,2271, donc un
#     tronc brun EST perceptuellement de la nourriture pour le slot) ;
#   * arrangés en 6 PEUPLEMENTS (processus de Neyman-Scott/Thomas) avec 3 CLAIRIÈRES.
#     Mesuré sur 24 tirages : Clark-Evans 0,794 ± 0,044 contre 1,054 ± 0,031 en tirage uniforme.
#     C'est ce groupement qui crée une occlusion NON uniforme — couloirs, écrans, ouvertures — donc
#     des positions qui ne se valent pas. Un semis uniforme n'est qu'un brouillard homogène.
#
# 🚨 DÉCOR VISUAL-ONLY ÉTEINT (SYLVAN_FOREST_DECOR=0), et c'est le point le plus important de ce
# script. forest_manager.gd dessine des arbres SANS collision et SANS couleur-rétine : l'entité ne
# les voit pas et les traverse. Les laisser à côté de la forêt solide rendrait l'image INDÉCHIFFRABLE
# — impossible de distinguer un arbre réel d'un arbre qui n'existe que pour l'observateur. Or ce
# visuel sert à JUGER (§2.1 : le visuel ne doit pas mentir). Mettre SYLVAN_FOREST_DECOR=1 pour
# comparer les deux, en sachant ce qu'on regarde.
#
# LE REGARD EST MAINTENANT SERVI (2026-07-26) : le preset foret_v1 pose SYLVAN_GAZE=1 et le WM par
# défaut est en proprioception 133. La note précédente (« le montrer exigerait le retrain ») est
# donc PÉRIMÉE — le retrain a eu lieu.
#
# ⚠️ CE QUE CE VISUEL NE PROUVE PAS : le WM servi par défaut a été entraîné sur l'ANCIENNE dynamique
# (kin_speed 2,83, coût de locomotion sur la seule énergie). Il navigue, mais ses chiffres ne valent
# rien — c'est un visuel pour VOIR LE MONDE avant de le collecter, pas pour juger l'entité.
#
# Usage : bash scripts/voir_foret.sh            (fermer la fenêtre pour arrêter)
#         TREES=28 STANDS=4 bash scripts/voir_foret.sh
set +e
# ROOT DÉRIVÉ DU SCRIPT (2026-07-26) : il pointait en dur vers un AUTRE dossier
# (SylvanV1-foret, un worktree voisin), donc ce viewer montrait un dépôt figé — un
# visuel qui sert à JUGER ne peut pas afficher une autre version que celle qu'on teste.
cd "$(dirname "$0")/.." || exit 1; ROOT="$(pwd)"

# PRESET = le monde GELÉ à servir (python/sylvan/world.py). Défaut bosquets_v7_types : la nourriture
# SE DÉPLACE (proies, 0,0099 m/tick ≈ 0,9x l'agent) et porte 4 TYPES de teintes différentes dont la
# valeur nutritive est ARBITRAIRE. Ces deux briques existaient et n'étaient pas servies ici.
# Mettre PRESET=bosquets_v2 pour revoir le monde plat (billes rouges immobiles + eau).
# Défaut = le monde qu'on collecte VRAIMENT (foret_v1), pas son ancêtre.
PRESET=${PRESET:-foret_v1}

PORT=${PORT:-6291}
WM=${WM_CKPT:-data/checkpoints/wm_foret_attn_slot/wm_best.pt}

# On ne tue QUE nos propres processus : une autre session peut travailler dans un worktree voisin.
pkill -9 -f "serve_planner_command.*$PORT" 2>/dev/null
pkill -9 -f "$ROOT/tools/godot" 2>/dev/null
sleep 1
export GODOT_BIN="$ROOT/tools/godot/godot"

# Le monde vient du PRESET GELÉ, pas de variables recopiées à la main dans ce script : c'est la
# raison d'être de sylvan/world.py (une seule source de vérité, sinon un harnais dérive en silence).
# `eval` et PAS `sed | tr` : la palette de teintes contient des « ; » et doit rester citée, ce que
# le découpage par espaces massacrerait. Les variables sont ainsi EXPORTÉES, donc héritées par Godot
# sans être recopiées dans la ligne `env` — et le preset reste la seule source de vérité du monde.
eval "$(PYTHONPATH=python ./env_pytorch_3.12/bin/python -m sylvan.world --preset "$PRESET" --env)" \
  || { echo "[voir-foret] preset '$PRESET' inconnu"; exit 1; }
echo "[voir-foret] monde = $PRESET (kin_speed=$SYLVAN_KIN_SPEED, fov=$SYLVAN_RETINA_FOV_DEG)"

# DENSITÉ D'ARBRES : le preset décide, ce script n'écrase QUE si on le lui demande explicitement.
# Auparavant TREES=45 était forcé en dur, hérité du plafond de navigabilité mesuré dans l'ARÈNE DE
# 11 m (54 arbres → immobile 85 % du temps). L'arène fait maintenant 35 m, dix fois l'aire : ce
# plafond n'y veut plus rien dire, et le défaut faisait regarder l'owner une forêt QUATRE FOIS plus
# claire (45 arbres) que celle qui sera collectée (191) — exactement ce que le commit « montrer le
# monde qu'on collecte vraiment » devait supprimer. Un knob qui survit à son monde devient un
# mensonge silencieux. TREES/STANDS/CLEARINGS restent disponibles pour comparer à l'œil.
export SYLVAN_FOREST_COUNT=${TREES:-$SYLVAN_FOREST_COUNT}
export SYLVAN_FOREST_STANDS=${STANDS:-$SYLVAN_FOREST_STANDS}
export SYLVAN_FOREST_CLEARINGS=${CLEARINGS:-$SYLVAN_FOREST_CLEARINGS}
env SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
    SYLVAN_PLANNER_HEADING_W=0.0 \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 \
    --egomotion-head data/checkpoints/egomotion_head/best.pt --slot-memory \
    > /tmp/voir_foret_srv.log 2>&1 &
SRV=$!
trap "kill -9 $SRV 2>/dev/null; pkill -9 -f \"$ROOT/tools/godot\" 2>/dev/null" EXIT INT TERM
for _ in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
if ! ss -ltn 2>/dev/null | grep -q ":$PORT"; then
  echo "[voir-foret] le serveur planner n'a pas démarré — voir /tmp/voir_foret_srv.log"; tail -20 /tmp/voir_foret_srv.log; exit 1
fi
echo "[voir-foret] serveur prêt (WM=$WM). Fenêtre Godot : $SYLVAN_FOREST_COUNT massifs, \
$SYLVAN_FOREST_STANDS peuplements, $SYLVAN_FOREST_CLEARINGS clairières \
(bosquets $SYLVAN_FOOD_PATCHES x $SYLVAN_FOOD_COUNT items, kin_speed $SYLVAN_KIN_SPEED)."
echo "[voir-foret] décor visual-only ÉTEINT — tout ce qui est dessiné est perçu par l'entité."
# BUISSON ÉTEINT PAR DÉFAUT (2026-07-26) : le preset foret_v1 n'en sert pas, donc le laisser à 1
# afficherait un objet PERCEPTIBLE de plus que le monde réellement collecté. Un visuel qui sert à
# juger doit montrer le monde qu'on collecte, pas un cousin.

# Fenêtré (PAS de --headless) => temps réel, regardable.
env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 \
  SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 \
  SYLVAN_KINEMATIC=1 \
  SYLVAN_WOLF=1 SYLVAN_WOLF_SCALE=${WOLF_SCALE:-0.4} SYLVAN_WOLF_Y=${WOLF_Y:--0.30} \
  SYLVAN_WOLF_LIGHTEN=${WOLF_LIGHTEN:-0.35} \
  SYLVAN_FOREST_DECOR=${SYLVAN_FOREST_DECOR:-0} \
  SYLVAN_FOREST_MESH=${SYLVAN_FOREST_MESH:-1} \
  SYLVAN_FOREST_CLEARING_R=${CLEARING_R:-4.0} SYLVAN_FOREST_STAND_SIGMA=${STAND_SIGMA:-3.0} \
  SYLVAN_FOOD_BUSH=${SYLVAN_FOOD_BUSH:-0} \
  SYLVAN_INIT_ENERGY=${INIT_ENERGY:-70} \
  SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=10 SYLVAN_SEED=${SEED:-1} \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=/tmp/voir_foret_run \
  ./tools/godot/godot --path godot
