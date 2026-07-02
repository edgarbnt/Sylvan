# PROMPT — reprise « clé de voûte » (WM qui imagine la perception sous rotation), JEPA-pur

## Mission
Continuer à construire la **CLÉ DE VOÛTE** de l'archi JEPA de Sylvan : un World Model dont le **rêve open-loop
transporte fidèlement la perception (le bearing des objets) à travers une rotation imaginée**. C'est le verrou
mesuré dont dépendent recherche dirigée, curiosité, décisions réfléchies, mémoire. **Décision owner : rester
JEPA-PUR** (pas de béquille active-rétine, pas de raccourci). Discipline CLAUDE.md : diagnostiquer gratuitement,
gater le cher, ne pas enchaîner les runs en croisant les doigts.

## À lire d'abord
`memory/sylvan-retina-decision.md` (le thread complet) + `docs/design_cle_de_voute_wm.md` (la carte de design,
options + gates) + `docs/diagnostic_perception_rotation_wm.md` (la mesure du manque) + `docs/archi_jepa_etat_des_lieux.md`
(le recul JEPA-LeCun, le chemin pur).

## Le bug (mesuré, pas supposé)
En latent-pur, l'entité atteint une cible **devant** mais ne sait pas poursuivre une cible **non perçue / derrière**.
Cause-racine remontée jusqu'à : **le rêve du WM ne transporte pas le bearing perçu à travers une rotation imaginée.**
Mesure (`diag_test1_readout.py`, held-out par épisode, sur segments de virage) : le rêve suit le bearing à
**corr ≈ +0.08–0.15** vs cible **≥ +0.35**. Localisation : la **dynamique/rollout** est le verrou (la représentation
teacher-forced, elle, est *liftable*).

## Ce qu'on a essayé (et ce que ça a fait)
Perte auxiliaire **bearing-through-rollout** (cible color-agnostic = bearing du plus proche objet vu dans la rétine,
§3-pur ; tête `bearing_head` non sauvée). Flags dans `train_wm_command.py` : `--w-bearing` (sur le RÊVE),
`--w-bearing-tf` (sur les latents TEACHER-FORCED = presse la représentation). Re-gate PUISSANT (`BUF=retina_eat_a`, 60 ép) :

| run | REPR (encodeur) | TEST1 (rêve) |
|---|---|---|
| `wm_rich_fidele_sym` (base) | +0.18 | +0.09 |
| `wm_keystone_bearing_v1` (3a : bearing sur le rêve) | +0.22 | +0.14 |
| `wm_keystone_bearing_v2` (3a′ : rêve + représentation) | +0.25 | **+0.15** |

→ **Presser la représentation LÈVE REPR (+0.18→+0.25) mais le RÊVE ne suit pas (+0.15 plateau).** Le verrou est le
**rollout** : le rêve colle globalement au teacher-forced (perte rollout ~0.12) mais perd la **sous-composante fine**
du bearing. eff_rank ~13 / displacement ~0.009 préservés (rien cassé). **Le levier « perte de poids » est épuisé
(2 négatifs informatifs, §1 → escalade vers un levier DIFFÉRENT).**

## Prochain pas : 3b (cheaper-first), puis 3c si plateau

**3b — DONNÉES (à faire d'abord, le moins cher) :** la dynamique manque peut-être de signal pour APPRENDRE le
transport (TEST 2 : seulement ~425 événements d'acquisition derrière→devant dans toute la data).
1. **Construire un collecteur SCRIPTÉ d'acquisitions** (Godot headless) : cibles placées à 360° (souvent DERRIÈRE) +
   **commandes de rotation imposées** pour balayer la cible derrière→devant, en loggant `retina0` + `food_rel0` + `cmd`.
   ⚠️ PAS de babbling aléatoire (mesuré : fait PIRE). S'inspirer des collecteurs existants (`collect_wm_*`, `run_forage_*`).
2. **Gate GRATUIT avant retrain** : `diag_test2_data_audit.py` (ajouter le nouveau buffer) → vérifier que la densité
   d'acquisitions (derrière→devant) est nettement > l'existant (~425). Sinon, le collecteur est à corriger.
3. **Retrain** (warm-start, pertes bearing déjà en place) :
   ```
   SYLVAN_WM_USE_RETINA=1 PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.train_wm_command \
     --runs godot/data/replay_buffer/retina_eat_a godot/data/replay_buffer/retina_eat_b \
            godot/data/replay_buffer/retina_forage <NOUVEAU_BUFFER_ACQUISITIONS> \
     --out data/checkpoints/wm_keystone_bearing_v3 \
     --init-from data/checkpoints/wm_rich_fidele_sym/wm_best.pt \
     --latent-loss cosine --vicreg-var 1 --vicreg-cov 1 --vicreg-gamma 1 \
     --w-rollout 3 --w-bearing 1.0 --w-bearing-tf 1.0 --mirror-augment --lr 1e-4 --epochs 8 --stride 8
   ```
4. **Re-gate** : `BUF=retina_eat_a WM_CKPT=data/checkpoints/wm_keystone_bearing_v3/wm_latest.pt PYTHONPATH=python
   ./env_pytorch_3.12/bin/python diag_test1_readout.py` → **SUCCÈS si TEST1 ≥ +0.35** (rêve) **sans casser** eff_rank/displacement.

**3c — ARCHITECTURE (si 3b plateaue ; ce que la localisation désigne) :** le rollout déterministe *smear* le détail fin.
Options pures : (a) **rollout rotation-ÉQUIVARIANT** — structurer la part perceptive du latent pour que ω applique une
transformation CONNUE (la rotation devient « gratuite ») ; (b) **latent STOCHASTIQUE** (variables latentes, RSSM
stochastique) — capture l'incertitude des « reveals » (sonde uncertainty à re-mesurer proprement d'abord). Vraie
refonte → diagnostiquer/concevoir avant de lancer (§1).

## Outils & ops (prêts)
- Trainer : `python -m scripts.train_wm_command` avec `--w-bearing`, `--w-bearing-tf`, `--w-rollout`, `--mirror-augment`,
  `--init-from`, `--latent-loss cosine`, `--vicreg-var/cov/gamma`. **TOUJOURS `SYLVAN_WM_USE_RETINA=1`** (sinon obs=145
  radar au lieu de 277 rétine → mismatch warm-start). lr 1e-4.
- Gate : `diag_test1_readout.py` (env `WM_CKPT`, `BUF`) ; localisation : `diag_wm_rotation.py` ; audit data :
  `diag_test2_data_audit.py`.
- Données rétine (avec `food_rel0`+`cmd`) sous `godot/data/replay_buffer/` : retina_eat_a (60), retina_eat_b (60),
  retina_forage (12), retina_wm_a (babbling, 200, sous `data/replay_buffer/`).
- Checkpoints : base WM `wm_rich_fidele_sym` ; assets `wm_keystone_bearing_v1/v2` (NON promus). **Live de secours
  inchangé = planner-coordonnées + `wm_command_hex_v2` (marche, mange).** L'entraînement WM tourne bien en background
  agent (la vieille note « ça plante » est obsolète).
- Tout le travail de cette session est **non commité** (35+ fichiers) — proposer un commit sur branche au owner.

## Critère de fin (le BUT, pas le proxy)
TEST1 (rêve) ≥ +0.35 held-out, PUIS valider en closed-loop que la planif latent-pure engage une cible derrière
(le rêve choisit de tourner parce qu'il IMAGINE l'acquisition). C'est ça, la voûte qui tient.
