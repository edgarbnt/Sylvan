# PROMPT — session COLLECTE + RETRAIN du monde-forêt RE-DIMENSIONNÉ

> À coller tel quel pour démarrer la session. Écrit le 2026-07-28, à la fin de la session qui a
> re-dimensionné le monde. Projet : `/home/edgarbrunet/Documents/PERSO/SylvanV1`.

---

## MISSION

Collecter le monde-forêt **re-dimensionné**, ré-entraîner le WM, et faire passer les gates jusqu'au
**foraging closed-loop** — le seul chiffre qui manque pour promouvoir ce WM comme substrat vivant.

## ⛔ RÈGLES DURES

- **NE RIEN METTRE SUR `main`, ne rien pousser.** `main` = dépôt public (vitrine). Si tu crois qu'il
  faut y toucher : arrête-toi et demande.
- Travailler sur `feat/perception-consequence` (@ `9cb636e`).
- venv `env_pytorch_3.12/bin/python` (CPU), `PYTHONPATH=python` depuis la racine.
- `tools/archi_hud/architecture.json` à jour **dans le même commit** que tout changement de conception.
- Commits **anglais, Conventional Commits, sans co-author**.
- **Diagnostiquer GRATUITEMENT avant tout run coûteux** (PRINCIPE N°1). Pré-enregistrer succès ET kill.

## À LIRE D'ABORD

1. `docs/etat_foret_collecte_retrain.md` — l'état mesuré (A1 levé, ce qui reste ouvert).
2. `python/sylvan/world.py` → preset `FORET_V1` — **la seule source de vérité du monde**, avec le
   pourquoi de chaque constante en commentaire.
3. `CLAUDE.md` §1-§4 — les principes qui gouvernent les décisions.

---

## ÉTAT — CE QUI EST FAIT, MESURÉ, ET NE DOIT PAS ÊTRE REFAIT

**Le verrou A1 est LEVÉ** : le type de proie est lisible dans le latent à **99,7 %** (linéaire), contre
27,3 % de majorité. Il a fallu **DEUX leviers ensemble**, et c'est une interaction, pas une addition :

| encodeur | sans pression | `--w-hue 50` |
|---|---|---|
| dense (MLP) | 33,3 % | 37,3 % |
| **attention par rayon** | 30,4 % | **99,7 %** |

La dynamique ne paie rien, elle s'améliore (open-loop 0,132 m à h=50, jalon @50 atteint).

**Le monde a été re-dimensionné et audité** (`diag_world_scale.py`, 9/9 cohérent). Cinq mécaniques
étaient **silencieusement inertes** parce que le corps avait été multiplié par 10 sans revisiter ce
qui l'entourait : proie à 0,19× la croisière (gain d'interception nul sous 0,6×), repousse qui ne se
déclenchait jamais dans une vie, cycle de flaque invisible, arène de 10 longueurs de corps, rayon de
braquage 4× plus grand que l'écart entre arbres. Toutes corrigées.

**Le coût de calcul a été mesuré** (gate §6quater D, jamais passé avant) : +3,6 % pour ×10 d'aire et
×6 d'objets. L'agrandissement ne coûte rien.

---

## ⚠️ CE QUI EST PÉRIMÉ — À SUPPRIMER AVANT DE COMMENCER

Le corpus et les checkpoints existants ont été produits **avant** trois changements de dynamique
(coût de locomotion réparti sur les deux jauges, `kin_speed` 0,8→8,0, re-dimensionnement). Ils ne
décrivent plus le monde servi.

```bash
rm -rf data/replay_buffer/foret_v1_* data/replay_buffer/foret_v1b_* data/replay_buffer/foret_v1c_*
```

Les checkpoints `wm_foret_*` restent sur disque comme **référence historique** — ne pas les servir.

---

## ORDRE, CHAQUE ÉTAPE GATÉE

### 0. Vérifier que le monde est cohérent (gratuit, 5 s)
```bash
PYTHONPATH=python env_pytorch_3.12/bin/python -m sylvan.world --selfcheck
PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_world_scale.py
```
Attendu : `SELFCHECK PASSED` et **9/9 échelles cohérentes**. Si une échelle échoue, **ne pas
collecter** — une mécanique inerte produit un corpus qui ment.

### 1. Collecte mixte (~45-60 min)
```bash
bash scripts/collect_foret_all.sh 150 foret_v1
```
Trois lots × 150 vies, chacun réparti planner 50 % / babillage 30 % / exploration 20 %.
**La part planner est obligatoire** : le babillage seul produit un corpus DÉGÉNÉRÉ (mesuré : 0 repas,
le garde-fou de la matrice refuse le corpus).

### 2. Vérifier le corpus AVANT d'entraîner (gratuit)
```bash
PYTHONPATH=python env_pytorch_3.12/bin/python -m sylvan.world --preset foret_v1 --env \
  | sed 's/^export //; s/"//g' > /tmp/foret_v1.env
PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_world_contract.py \
  data/replay_buffer/foret_v1_planner --preset-file /tmp/foret_v1.env
```
Attendu : **toutes les clauses vertes**, zéro part dégénérée, les 3 conséquences vécues
(consommations **et** dégâts). Sinon, ne pas payer le retrain.

### 3. Retrain (~40 min) — **les DEUX leviers, sinon A1 retombe au hasard**
```bash
PYTHONPATH=python SYLVAN_WM_USE_RETINA=1 env_pytorch_3.12/bin/python -m scripts.train_wm_command \
  --runs data/replay_buffer/foret_v1_planner data/replay_buffer/foret_v1_babble data/replay_buffer/foret_v1_explore \
         data/replay_buffer/foret_v1b_planner data/replay_buffer/foret_v1b_babble data/replay_buffer/foret_v1b_explore \
         data/replay_buffer/foret_v1c_planner data/replay_buffer/foret_v1c_babble data/replay_buffer/foret_v1c_explore \
  --out data/checkpoints/wm_foret_v2 --proprio-dim 133 \
  --retina-attention --w-hue 50 --epochs 20 --seq-len 64 --lr 1e-4 \
  --w-latent 1.0 --w-proprio 0.0 --w-radar 0.0 --w-energy 20.0 --w-displacement 10.0 --w-done 1.0 \
  --latent-loss cosine --vicreg-var 1.0 --vicreg-cov 1.0 --vicreg-gamma 1.0 \
  --w-rollout 3.0 --predictor-arch shallow --mirror-augment
```
`SYLVAN_WM_USE_RETINA=1` est **obligatoire** : sans lui le dataset assemble un WM-radar (obs 146) au
lieu du WM-rétine (obs 278) — une régression majeure qui ne lève aucune erreur.

### 4. Gates post-retrain
```bash
# canal-slot (greffe : géométrie pure, lit la rétine — voir la réserve plus bas)
PYTHONPATH=python env_pytorch_3.12/bin/python scripts/graft_slot_channel.py \
  --dst data/checkpoints/wm_foret_v2/wm_best.pt \
  --src data/checkpoints/wm_objcentric_kin_typed/wm_best.pt \
  --out data/checkpoints/wm_foret_v2_slot/wm_best.pt

# A1 — le type survit-il à l'encodeur ?
PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_latent_carries_type.py \
  --corpus data/replay_buffer/foret_v1_planner data/replay_buffer/foret_v1b_planner \
  --wm data/checkpoints/wm_foret_v2_slot/wm_best.pt --depth 0 --stride 4

# non-régression open-loop
PYTHONPATH=python SYLVAN_WM_USE_RETINA=1 env_pytorch_3.12/bin/python -m scripts.eval_wm_command \
  --checkpoint data/checkpoints/wm_foret_v2/wm_best.pt --horizons 10 50 80

# foraging closed-loop — LE gate qui manque
WM_CKPT=data/checkpoints/wm_foret_v2_slot/wm_best.pt bash scripts/gate_foret_closedloop.sh 12 3000
```

---

## CRITÈRES DE SUCCÈS — PRÉ-ENREGISTRÉS, MESURÉS AU BUT

| gate | seuil | référence |
|---|---|---|
| **A1** type à l'encodeur | **> 70 %** | 99,7 % au cycle précédent ; majorité 27,3 % |
| **open-loop** position @50 | **≤ 0,20 m** | 0,132 m au cycle précédent |
| **closed-loop** survie | **médiane > 1000 ticks** | 350 avant les correctifs |
| **closed-loop** repas | **médiane ≥ 3** | 0,5 avant |
| **closed-loop** cause de mort | **pas 11/12 de faim** | l'asymétrie des jauges est corrigée |

Un échec closed-loop est un **négatif informatif** : STOP, diagnostiquer gratuitement, ne pas
enchaîner un tweak. Le trajet par repas re-mesuré à la densité servie (**12,5 m**, 25 bosquets)
contre le budget toléré (12,70 m) ne laisse que **1,02× de marge** — mesuré sur 6 vies / 18 repas
seulement, avec un WM hors-distribution. C'est mince, c'est assumé, et le selfcheck l'assère
désormais au lieu de simplement l'afficher.

## 🚨 ÉTAT AU 2026-07-28 — DEUX DÉFAUTS DE MONDE RÉPARÉS, UN BLOCAGE OUVERT

Le visuel (pas un gate) a révélé deux défauts que la collecte aurait gravés dans le corpus.

**1. L'eau était quasi absente.** `to_env()` ne servait la géométrie des bosquets qu'à la
NOURRITURE ; l'eau retombait sur les défauts du GDScript, dont un anneau de 7 m — 26× plus petit.
Mesuré : `WATER : 2/180 bosquets placés` contre `FOOD : 176/180`, et 3 vies sur 3 mortes de soif
sans une gorgée. Un corpus collecté ainsi n'aurait contenu **aucun événement de boisson**.
Réparé (178/180) et gardé par `--selfcheck` (symétrie des 8 clés, comparée en valeur).

**2. Un repas était indélivrable.** 84 points servis sur une jauge plafonnée à 100 : encaissable
seulement à 16 d'énergie, un niveau qu'un planner qui survit ne s'autorise jamais. Mesuré sur trois
densités : elle mange à 73/51/56 et n'encaisse que 21/32/28 pts (25-38 %). Tout le calibrage
« 10 événements/vie » reposait sur cette valeur. Corrigé en deux temps, sur mesure :

| | avant | après | preuve |
|---|---|---|---|
| bosquets/ressource | 180 | **25** | survie médiane 412 → 856 (`probe_foret_densite.sh`) |
| plafond de jauge | 100 | **200** | énergie observée jusqu'à 197 — impossible avant |
| départ (init) | 60 | **60** | inchangé : le monter aurait doublé la réserve (plancher 25 % → 50 %) |

**Le blocage restant est la BOISSON, et il est localisé.** Gate closed-loop après correctifs
(8 vies) : survie médiane **415** (exigé > 1000), **6 morts de soif sur 8**, énergie à 178-197 au
moment de mourir de soif. Une vie a tenu les **3000 ticks pleins** (12 repas, 6 boissons) — le monde
EST survivable. Ce n'est **pas** un problème de perception : la matrice requête×objet est propre,
chaque objet n'est vu que par sa propre requête et l'eau est détectée à **cos 0,999**. Elle VOIT
l'eau et n'y va pas. C'est le mur d'arbitrage déjà documenté (`sylvan-second-drive-arbitration`),
ici inversé : elle se gave et meurt de soif.

**Deux réserves à ne pas oublier** : le WM servi vient de l'ANCIENNE dynamique (hors-distribution
partout), et le canal-slot est GREFFÉ. Le prochain pas honnête est donc de **diagnostiquer
gratuitement l'arbitrage** (pourquoi le planner ne cible pas l'eau alors qu'il la voit) AVANT de
payer collecte + retrain — pas de collecter en espérant que le retrain le règle.

---

## PIÈGES DÉJÀ PAYÉS — NE PAS LES REPAYER

- **Chemin relatif `SYLVAN_RUN_DIR`** : résolu depuis `godot/`, pas la racine → le corpus atterrit
  ailleurs sans erreur. Utiliser un chemin absolu.
- **Un préambule `pkill` dans une commande backgroundée** la fait sortir en erreur avant le vrai
  lancement (CLAUDE.md règle 3). Tuer dans une commande séparée.
- **`pgrep -f <motif>`** matche son propre shell → il compte un processus fantôme. Utiliser
  `ps -C godot -o pid=`.
- **Trois consommateurs ont leur PROPRE largeur de proprioception** : le WM (133), la politique
  résiduelle (132), la tête d'égomotion (132). Chacun doit tronquer. Une exception à chaque tick
  faisait répondre le serveur par un `safe fallback` (vx=0,5, ω=0) — l'entité fonçait tout droit et
  rien ne le signalait.
- **`CommandWorldModel.from_checkpoint(payload)`** existe : construire le modèle à la main en
  recopiant les clés du meta le rend faux au premier champ d'architecture ajouté.

## CE QUI RESTE OUVERT

- **A2** (requêtes-slot apprises) : `build_typed_slots` exige K=3 groupes de couleur alors que le
  monde en sert 4 pour la nourriture (clustering → K=5), et le danger noie la contingence. Deux voies
  possibles : autoriser plusieurs clusters par drive, ou compter les **événements** de dégât au lieu
  des ticks. **Réserve du canal greffé** : ses requêtes viennent de l'ancien monde — vérifié qu'elles
  voient les 4 teintes servies (cos 0,946-0,991), donc utilisable, mais ce n'est pas A2.
- **La pression `--w-hue` reste une grandeur choisie à la main.** La perception est APPRISE (aucun
  seuil codé, aucun oracle, la cible sort de la rétine) mais le gradient ne vient pas encore de la
  **conséquence vécue** — le north-star §6ter. L'encodeur à attention est en place pour l'accueillir.
- **`kin_body_extent`** (encombrement réel du corps) est mesuré et committé mais **non servi** :
  l'activer coince le planner 43,9 % du temps. Le visuel a été corrigé en reculant le maillage.
