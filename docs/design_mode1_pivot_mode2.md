# Bascule Mode-1 → Mode-2 : le mur d'arbitrage est un problème de DÉLIBÉRATION

_2026-07-02. Le *pourquoi* de la bascule ; l'*état* vit dans `architecture.json` (nœud `mode_1`) et
`memory/sylvan-mode1-build.md`._

## 1. Mission
Une entité qui **arbitre ses pulsions pour survivre** (faim + soif) *mieux* que le coût codé-main du
planner (plafonné ~2300). Le BUT = survie multi-drive, mesurée en pas-vécus (pas le return).

## 2. À lire d'abord
- `memory/sylvan-mode1-build.md` — l'arc Gate-2 complet (conditionnement → arbitrage).
- `diag_mode1_death_cause.py` — la sonde qui a localisé le mur (cause de mort via rétine).
- `docs/design_mode1.md` + `memory/sylvan-second-drive-arbitration.md` — la voie-critique et ses 3 verrous.

## 3. Limite MESURÉE (chiffrée, sonde à l'appui)
**Une politique RÉACTIVE (Mode-1, model-free PPO) plafonne au niveau BC (~1900), et le mur est le
LOOK-AHEAD, pas le réflexe :**
- Gate-2b/2c : survie oscille ~1824-1890, jamais de percée vers ~2130, malgré critique qui apprend
  (value_loss 24k→0.14 après fix conditionnement) + exploration ω déjà large (std 0.447) + shaping.
- `diag_mode1_death_cause.py` (97 morts) : **96% des morts = DÉCISION** (66% « voit la ressource qui tue
  mais ne l'approche pas », 30% campe sur l'autre) ; **4% moteur, 0% perception** ; marge trajet/drain
  **3.9×** → **corps réfuté** comme cause.
- **Preuve directe du besoin de look-ahead** : le planner (look-ahead MPC) fait ~2300 ; son clone
  RÉACTIF (BC) tombe à **1930** → le look-ahead vaut **~370** sur cette tâche exacte.

→ **Le mur d'arbitrage est un problème de Mode 2 (délibération/look-ahead), qu'on a tenté avec un
Mode 1 (réflexe). Le réflexe ne peut pas le porter.** (Cadre Kahneman, cf le nom même « Mode-1 ».)

## 4. Essayé → résultat (négatifs INFORMATIFS — ne pas répéter)
- **Fix conditionnement critique** (`--reward-scale 0.004`, `diag_mode1_critic_fit`) → ✅ le critique
  apprend (R² -4→0.74 ; value_loss 24k→0.09). A résolu le mur #1, PAS l'arbitrage.
- **Pain-shaping annealé** (`--pain-shaping-w-*`) → ❌ *game* la récompense : `mean_reward` monte
  (drives gardés tièdes) mais `mean_steps` ne suit pas ; death-cause vire « erre 66%→51% / campe
  30%→42% » (optimise un objectif ~orthogonal à la survie, §2). Pas le fix.
- **Exploration** → réfutée AVANT run : ω explore déjà large (std 0.447) ; vx borné PAR DESIGN
  (`map_action` [0.55,0.75]) et non pertinent pour l'arbitrage. Levier faible.
- **Test greedy-oracle standalone** → cul-de-sac : le CPG seul ne propulse pas (fwd_v=0), le résidu
  est indispensable et vit dans un serveur → abandonné au profit des preuves convergentes.
- Rappel des 3 verrous historiques de la voie-critique (`sylvan-second-drive-arbitration`) :
  (1) MC naïf off-policy **[LEVÉ par la collecte on-policy Mode-1]** ; (2) WM aveugle au repas
  **[risque restant]** ; (3) eat-dynamics = tête rapide sur le slot, pas un head WM.

## 5. Le pivot (direction) + prochain pas cheaper-first
**PIVOT = garder le planner (look-ahead dans le WM) mais remplacer son coût codé-main (`min_dist`) par
une VALEUR DE SURVIE APPRISE sur le latent.** Look-ahead conservé (bat ~1900) + coût appris (bat ~2300) ;
c'est le Mode 2 JEPA-pur (archi LeCun). Réutilise la technique **value-sur-latents-RÊVÉS déjà résolue**
(saga rétine : `train_value_head.py SYLVAN_VALUE_DREAM=1` → close 1.02 m + mange).

**Mode 1 n'est pas jeté** : il reste la couche RÉFLEXE (fonctionne ~BC) ; Mode 2 tranche l'arbitrage dur ;
la nuit distille Mode 2 → Mode 1 (vision jour/nuit). Acquis Mode-1 réutilisés : collecte on-policy,
critique drive-symétrique, diagnostics.

**PROCHAIN = GATE GRATUIT de faisabilité (avant tout retrain WM/valeur, §1) :** sur des rollouts RÊVÉS
depuis des états multi-drive, une valeur-survie apprise sépare-t-elle « bon arbitrage » de « myope » ?
(rang/sonde, comme le gate close de la rétine.) Ne payer le retrain QUE si la sonde sépare.

## 6. Critère de succès = le BUT (falsifiable)
- **Gate gratuit** : la valeur-survie sur latents rêvés classe correctement (rang) des états
  « soigne-les-deux-drives » > « myope » (séparation nette, pas au hasard). Si non → le WM/latent ne
  porte pas l'info de survie multi-drive → creuser le latent (object-centric) avant planning.
- **Succès final** (si le gate passe) : survie médiane multi-drive **> planner ~2300**, mesurée en
  pas-vécus, multi-seed. PAS le return ; PAS en élargissant la bouche/le seuil (§2).
