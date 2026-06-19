# Critères pré-enregistrés — JEPA-ification ÉTAPE 2 (shift reconstruction→latent), 2026-06-18

> Run : `train_wm_jepa2.sh` → `data/checkpoints/wm_command_hex_v3_jepa2`. Mesure : `diag_jepa.py` +
> `eval_wm_command` + **boucle fermée** (A→B sur le nouveau WM). Le hex_v2 ET le v3_jepa restent intacts.
> Baseline = v3_jepa (étape 1) : eff_rank 25.8, displacement 0.126 cm/pas, énergie 1.15 %, pos 0.49 m@100.

## Hypothèse falsifiable
Dropper la reconstruction d'entrée (proprio+radar = 0) et monter la voie latente, **avec l'anti-collapse
maintenu (cosine+VICReg)**, donne un WM **fonctionnellement JEPA** (prédiction portée par le latent, plus
par la reconstruction) qui **garde le signal du planner** et **fonctionne en boucle fermée**.

## SUCCÈS (→ on est fonctionnellement JEPA ; banker v3_jepa2)
Les TROIS blocs :
- **A. Pas de ré-effondrement** : eff_rank latent **≥ 20 /128** (dropper la reconstruction ne doit PAS
  re-collapser la rep — c'est tout l'enjeu : la rep tient sur latent+VICReg+ancres, sans reconstruction).
- **B. Signal planner préservé (PORTE)** : `eval_wm_command` pos **< 0.5 m@50** et **< 1.2 m@100** ;
  displacement RMSE ≤ **0.18 cm/pas** (depuis 0.126, +~40 % toléré), énergie RMSE ≤ **2 %**.
- **C. Boucle fermée** : A→B sur v3_jepa2 (`diag_nav_ab.sh` 4m, heading_w=2) reste **≥ baseline opérationnelle**
  (devant+côté ~11/12) — le planner doit piloter le nouveau WM sans casser la nav.

## KILL / ESCALADE
- **eff_rank < 12** (ré-effondrement) → la rep avait besoin d'une ancre de reconstruction. Escalade :
  remettre un PETIT poids proprio (0.2–0.5) comme ancre, garder radar=0.
- **Porte B cassée** (pos@50 > 1.0 m, OU displacement RMSE > 0.25, OU énergie > 3 %) → le latent a perdu
  l'info corps/énergie. Escalade : remettre proprio anchor 0.3, OU baisser w-latent (5→2).
- **Boucle fermée cassée** (A→B s'effondre alors qu'open-loop est OK) → divergence dream↔réel sous le
  nouveau régime → investiguer (pas un simple tweak de poids).

## Kill précoce (gratuit, `grep -E "eff_rank|displacement" /tmp/wm_jepa2.log`)
- eff_rank s'effondre sous ~10 dès les 1ers epochs → drop trop agressif → tuer, remettre proprio anchor.
- val displacement/energy explosent (×2) → tuer, le latent perd les ancres → baisser w-latent.

## Note d'interprétation
- Avec w-proprio=w-radar=0, le `jepa_ratio` du log (latent / (proprio+radar)) devient ∞/instable → NE PAS
  s'y fier ; juger sur eff_rank + cosine latent (diag_jepa) + la PORTE planner.
- Le `diag_jepa` eff_rank (pooled sur tout le val) est la VÉRITÉ ; le eff_rank du log par-epoch sous-estime
  (calcul par-batch) — leçon de l'étape 1 (on avait cru à un échec à l'epoch 19, c'était un succès).
