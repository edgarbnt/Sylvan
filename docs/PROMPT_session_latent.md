# Prompt — nouvelle session : ÉTAPE B (🅑) = PLANIFIER EN LATENT

Copie-colle le bloc ci-dessous au début d'une nouvelle session Claude Code.

---

Projet Sylvan (ALife émergente dans un world-model JEPA, Godot + PyTorch CPU). Avant tout : lis
`ETAT_DES_LIEUX.md` (surtout §8, MAJ « nuit-2 ») + `memory/MEMORY.md` puis `memory/sylvan-retina-decision.md`,
`docs/scope_retina.md` et le schéma `docs/schema_jepa_sylvan.md`. Respecte CLAUDE.md — surtout les PRINCIPES
§1 (« diagnostiquer GRATUITEMENT avant d'entraîner / gater le cher derrière le pas-cher ») et §2 (« ne pas
masquer le vrai problème, pas de fausse solution, pas de conclusion arrangeante »), et le protocole de kill.

**Acquis (session précédente).** La RÉTINE est LIVRÉE : la perception est 100 % apprise, l'oracle radar est mort.
- Base motrice : `data/checkpoints/hexapod_v2/policy_best.pt` (inchangée).
- Tête de perception apprise 🅐 (rayons couleur bruts → position food) : `data/checkpoints/retina_head/head_best.pt`.
- **WM-rétine à latent RICHE** : `data/checkpoints/wm_command_hex_retina_jepa_v2/wm_best.pt` (obs **277** = proprio132
  + rétine144 + énergie1 ; flag `SYLVAN_WM_USE_RETINA=1` ; **eff_rank 14**, open-loop **0.22 m@100**, displacement 0.015).
- Foraging closed-loop avec ce WM + la tête = **médiane 980 ≥ oracle 965** (`run_forage_retina.sh`, `WM_CKPT=…retina_jepa_v2/wm_best.pt`).
- Serveur `serve_planner_command.py` : gère `--retina-head` + détecte obs_dim 277 (`self.wm_uses_retina`) + un mode
  `override_pos` dans `command_planner.plan(...)`. Godot envoie la rétine live (`SYLVAN_RETINA_PLANNER=1`).

**TÂCHE = ÉTAPE B (🅑) : planifier dans le LATENT, sans coordonnées explicites.** Aujourd'hui le coût du planner
(`command_planner.py`) est géométrique : `-min_dist` vers la position (fx,fz) fournie par la tête + alignement de
cap + énergie. Le but de 🅑 = un coût **abstrait** : maximiser l'**énergie/le confort futur PRÉDIT par le WM** sur
le rollout (le WM imagine qu'aller vers la bouffe → manger → énergie qui remonte), **sans `-min_dist` ni position
de bouffe**. La tête 🅐 devient optionnelle (ancre/fallback). C'est le JEPA le plus pur : la « bouffe » n'existe
plus que comme conséquence prédite dans le latent.

**⚠️ NUANCE HONNÊTE À GARDER EN TÊTE (mesurée, `diag_latent_foodaware.py`).** Le feu vert est RÉEL mais NON
TRIVIAL : l'énergie prédite EST food-aware (corrélation dist_finale↔ΔE = **−0.46**, forte et consistante → les
trajectoires qui finissent plus près de la bouffe prédisent bien plus d'énergie). MAIS : **l'avantage ABSOLU
d'énergie d'aller vers la bouffe est FAIBLE** (écart proche−loin ≈ **+0.023** seulement) et **mesuré seulement à
horizon 80 pas** — sur cet horizon l'énergie DRAINE plus vite qu'elle ne remonte (ΔE négatif pour tous les
candidats, juste « moins négatif » vers la bouffe). Conséquence : un coût **100 % énergie-latente risque d'être
fragile** (signal de gradient ténu, noyé dans le bruit du WM). Donc **NE PARS PAS direct sur le pur-énergie.**

**Plan recommandé (gater le cher derrière le pas-cher) :**
1. **D'ABORD un diagnostic GRATUIT** (pas de retrain) : ré-utilise/étends `diag_latent_foodaware.py` pour mesurer
   comment le signal énergie-latente se comporte selon l'**horizon** (50/80/120/150) — l'écart proche−loin
   grandit-il avec H (atteindre+manger pèse plus) ? Et selon un **niveau d'énergie initial bas** (l'urgence
   rend-elle le signal plus net ?). Écris des critères SUCCÈS/KILL AVANT.
2. **Implémente un coût latent HYBRIDE** dans `command_planner.py` (nouveau mode, ex. env `SYLVAN_PLANNER_LATENT=1`) :
   score = `w_energy · énergie_future_prédite  − done_penalty · risque_chute  (+ petite ancre de cap optionnelle)`,
   en lisant `out["predicted_next_obs"][..., -1]` sur le rollout (déjà dispo). Garde l'option de monter l'horizon.
   Commence hybride (énergie dominante + ancre faible), puis **réduis l'ancre vers 0** = transition vers le pur-🅑.
3. **JALON FALSIFIABLE** : « l'agent forage avec le coût LATENT, **coordonnées de bouffe DÉBRANCHÉES** (assert),
   survie ≥ baseline ». SUCCÈS = survie médiane ≥ ~900 (proche du 980 actuel). KILL = il erre/cale sans s'approcher
   (le signal énergie est trop faible) → ne PAS gonfler artificiellement ; conclure que le WM doit d'abord mieux
   prédire l'eat-dynamics (améliorer le WM / horizon / données) AVANT le pur-latent.

**Discipline.** Aucun gros run sans hypothèse falsifiable. Si tenté de ré-injecter `-min_dist`/les coordonnées
pour « faire marcher », STOP et dis-le (ce serait revenir en arrière, pas faire 🅑) — sauf comme *ancre hybride
explicite et décroissante*, pas comme solution déguisée (CLAUDE.md §2). Le pipeline d'ordre global restant après
🅑 : **critique foresighted** (fix robustesse multi-pulsions myope) → **Mode-1** (politique amortie). Ne touche
au MPC brute-force que dans le cadre du coût (pas de Mode-1 maintenant).

Outils prêts : `run_forage_retina.sh` (`WM_CKPT=`, arg head, `SYLVAN_RETINA_PLANNER=1`), `serve_planner_command.py`
(`--retina-head`, détection obs 277, `SYLVAN_RETINA_POS_ALPHA`), `diag_latent_foodaware.py`, `collect_forage_retina.sh`,
`train_retina_head.py`, recette WM JEPA dans `train_wm_jepa.sh` (`SYLVAN_WM_USE_RETINA=1 --latent-loss cosine
--vicreg-var/cov/gamma 1.0 --lr 1e-4`). Données conservées : `data/replay_buffer/retina_wm_a|b` (400 ép WM rétine),
`godot/data/replay_buffer/retina_forage` (foraging + labels). Commence par le SCOPE/diagnostic gratuit, pas par un
entraînement.

---
