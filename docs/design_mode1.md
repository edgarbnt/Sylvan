# design_mode1.md — Mode-1 : politique apprise multi-pulsions (RL + exploration)

> Spec issu du brainstorm owner du 2026-06-26 (fin de session « tester la vie »). À lire avec
> `memory/sylvan-second-drive-arbitration.md` (les 5 gates qui ont cartographié la voie apprise) et
> `CLAUDE.md` (§1 anti-boucle, §2 ne-pas-masquer, §3 substrat/pulsions, §4 étape-par-étape).
> **Rien n'est entraîné avant que les gates GRATUITS passent (§1).**

---

## 0. Le verrou (pourquoi Mode-1)

Le coût **designed** du planner plafonne (~2300 de survie multi-pulsions) parce qu'il est **myope**, et la
voie apprise **incrémentale** (critique `V(état)`) est bloquée par **2 murs** (5 gates gratuits, session
2026-06-26, cf nœud `critique_appris`) :
1. **OFF-POLICY** : la politique myope ne mange jamais quand elle a faim → le vécu ne contient pas l'action
   utile → un critique appris dessus n'apprend que « énergie basse = mort », pas « va manger ».
2. **PERCEPTION COURTE-PORTÉE imprécise du WM** : au contact, le slot capte **14 %** du signal-repas vs **46 %**
   avec la distance brute (perception parfaite). Le signal existe mais le WM le perd.

**Mode-1 = une politique APPRISE (RL + exploration) qui contourne LES DEUX murs :**
- l'**exploration** génère elle-même les transitions « faim → atteint la bouffe → énergie remonte » → le
  signal off-policy apparaît dans le vécu (lève mur #1) ;
- le **bout-en-bout sur la rétine brute** lui fait apprendre **sa propre perception** → le mur #2 vit dans
  l'**encodeur-slot**, PAS dans les rayons rétine (qui portent la profondeur vraie au contact) (lève mur #2).

C'est le vrai « elle se gère elle-même », explicitement différé « en dernier » et démarré ici **délibérément**.

---

## 1. Décisions verrouillées (brainstorm owner 2026-06-26)

| Choix | Décision | Raison |
|---|---|---|
| **Action** | 2-D `(vx,ω)` borné au régime propre (`vx∈[0.55,0.75]`, `ω∈[-0.6,0.6]`) | réutilise **CPG + résidu `hexapod_v2` GELÉS** ; même contrat TCP `{action,command}` → **0 changement Godot** |
| **Perception** | **rétine apprise** (36 rayons × [depth,R,G,B] ; rouge=bouffe, bleu=eau) | perception 100 % apprise, **zéro oracle** ; évite le slot (14 %) — le mur #2 est dans l'encodeur-slot, pas les rayons |
| **Paradigme** | **model-free PPO**, env réel | l'infra PPO est rétargetable (`action_dim`→2 trivial, update/buffer réutilisés) ; le model-based (Dreamer) est **bloqué par l'aveuglement eat-dynamics du WM** (1 % de la bosse) |
| **Warm-start** | **BC du planner** → RL-finetune | expert gratuit déterministe ; donne le **gate gratuit décisif** ; le RL n'a plus qu'à battre la myopie |
| **Récompense** | **survie pure** (vivre = +1/pas, mort = terminal) | le SEUL axiome = *persister* ; la **forme de la douleur n'est PAS posée — elle ÉMERGE** comme la tête de valeur APPRISE `V(état)` (critique contextuel) ; `(1−niveau)²` = simple béquille de démarrage annealable, jamais la cible |
| **Scalabilité** | **politique drive-symétrique** | ajouter une pulsion = brancher un token, **zéro retrain** (prouvé par gate) — §3 appliqué au RL |

---

## 2. Architecture

### 2.1 Substrat moteur (inchangé, GELÉ)
CPG hexapode trépied + résidu `hexapod_v2` **gelés**. Mode-1 n'émet que `(vx,ω)` ; la chaîne
commande→`cpg_reference()`→résidu→18-DOF est **identique au mode PLANNER** (`main.gd:461-469`,
`sylvan_agent.gd:662-675`). Même protocole TCP que `serve_planner_command.py` → **aucun changement Godot**.
⚠️ `turn-fade` zéro le résidu à grand `|ω|` → la commande de Mode-1 doit être **juste** (le résidu ne corrige
pas le virage).

### 2.2 Cadence
Décision au **command-cadence** (`replan_every`≈10, ~3.3 Hz). ~300 décisions / épisode de 3000 pas (bon pour
PPO). La récompense d'une transition = **somme des récompenses par-pas de la fenêtre** tenue.

### 2.3 Observation — perception apprise & DRIVE-SYMÉTRIQUE (le cœur scalable)
**Aucun slot `énergie`/`soif` en dur.** La politique voit :
- `proprio` (≈132, état-corps partagé) ;
- **un TOKEN par pulsion `d` active** = `(niveau_d, perception_d, valence_d)` où
  - `niveau_d` ∈ [0,1] (la politique apprend l'urgence elle-même),
  - `perception_d` = les **rayons rétine filtrés sur la couleur de `d`** (rouge→faim, bleu→soif, …),
  - `valence_d` = +1 « approcher-consommer » / −1 « fuir » (le danger entre dans le même moule).
- **encodeur PARTAGÉ** appliqué à chaque token → **pooling invariant par permutation** (mean / attention) →
  concaténé à `proprio` → tronc de la politique.

→ La politique apprend la compétence GÉNÉRALE *« satisfaire la ressource urgente et atteignable, quelle
qu'elle soit »*, pas « gérer énergie+soif ». **Ajouter une pulsion = un token de plus, mêmes poids.**
Entraînement avec **randomisation des pulsions/ressources** (nombre, positions, drains, lesquelles actives)
pour forcer LA compétence générale (sinon l'archi symétrique sur-apprend les 2 vues).

> **Hypothèse à vérifier (Gate-0)** : l'eau (bleu) est-elle rendue dans la rétine ? (la rétine raycast une
> couche de collision dédiée ; le `food_manager` généralisé sert l'eau avec albedo/emission bleus). Si non →
> petit build (mettre l'eau sur la couche rétine). Le lien couleur↔pulsion est une **définition-corps** (« ce
> drive est soulagé par le rouge »), définie une fois — l'idéal §3 (lien APPRIS auto-supervisé) est un upgrade
> de pureté ultérieur.

### 2.4 Récompense — SURVIE PURE (la forme de la douleur est APPRISE, pas posée)
**Le seul axiome = persister** (le plancher le plus minimal/neutre, quasi-tautologique pour de l'ALife : ce qui
ne « tient » pas à durer ne dure pas). `reward_t = +1 par pas vivant`, **mort = terminal** (l'épisode s'arrête).
Maximiser le retour = **maximiser la durée de vie** = LE but, directement. Avec plusieurs pulsions qui se vident,
rester en vie **FORCE l'arbitrage** (laisser un drive tomber à 0 = mort = fin du +1) → **l'arbitrage émerge de la
survie**, sans qu'on encode aucune urgence ni importance (décision owner 2026-06-26, cf `[[sylvan-second-drive-arbitration]]`).

**La FORME de la douleur n'est PAS dans la récompense — elle ÉMERGE comme la tête de valeur `V(état)` du PPO**
(elle aussi **drive-symétrique**, donc scalable) : le critic apprend « depuis cet état de manque, combien de
survie future ? ». C'est *contextuel* (0.5-d'énergie près de bouffe ≠ 0.5 dans un désert) là où une forme figée
`(1−niveau)²` est **aveugle au contexte** (et donc fausse). **C'est le nœud `critique_appris` réalisé** : la
douleur apprise = la valeur apprise (LeCun). On ne dit JAMAIS « manger=bien » ni « la faim prime » : le
« comment » et l'« importance relative » sont **100 % découverts**.

**Béquille de démarrage (optionnelle, annealable)** : la survie pure est *sparse* (signal = « mort au pas N »).
Pour amorcer l'apprentissage on PEUT ajouter un shaping `−λ·Σ_d (1−niveau_d)²` — idéalement **potential-based**
(`Φ(s)=−Σ(1−niveau)²` → `shaping = γΦ(s')−Φ(s)`, **ne change PAS la politique optimale**) et/ou **annealé λ→0**.
La cible finale reste donc la **survie pure**, et la forme de douleur finale est **celle qu'elle a apprise**, pas
la nôtre. `λ` est un **cadran** (env `SYLVAN_PAIN_SHAPING_W`), pas la définition de la douleur.

### 2.5 Politique = model-free PPO
Réutilise `ppo/update.py` + `ppo/rollout.py` **tels quels** (action-dim-agnostiques). `action_dim=2` (config),
**symétrie DÉSACTIVÉE** (`sym_coef=0` ; `symmetry.py` est hardcodée 18-D, sans objet pour 2-D). Actor/critic MLP,
politique gaussienne tanh-squash → mappée sur `(vx∈[0.55,0.75], ω∈[-0.6,0.6])`. **`--lr 1e-4`** (3e-4 diverge).

### 2.6 Warm-start BC
Collecter `(obs → commande planner)` en régime multi-pulsions (le planner LIT `out['slot']` + `water_radar`),
puis **régression supervisée** `obs_Mode1 → (vx,ω)` (la politique symétrique apprend rétine→commande). Le RL
part de cette politique BC (pas de zéro).

### 2.7 Exploration (lève l'off-policy)
**Garantir** qu'elle essaie d'aller manger/boire = combinaison :
1. **warm-start BC** (démarre compétente en nav) ;
2. **resets à drives randomisés** (`SYLVAN_INIT_ENERGY/THIRST` existent déjà — randomisés/épisode) → vit
   souvent « faim près de bouffe » / « soif près d'eau » → produit le signal que le planner myope ne produit jamais ;
3. **entropie PPO** ;
4. la **douleur graduée** densifie le crédit (soulagement immédiat à chaque gorgée/bouchée).

### 2.8 Planner = fallback + teacher (conservé)
Le planner-MPC reste : teacher BC, fallback de sécurité, collecte WM. Mode-1 ne devient **live** que s'il bat
la baseline **closed-loop** (CLAUDE.md).

---

## 3. Critères falsifiables (écrits AVANT tout entraînement — §1)

- **BUT mesuré** : médiane de survie multi-pulsions via `baseline_multidrive_slot.sh` (drain 0.05, eau active).
  **À BATTRE : ~2300** (le coût designed `survival_weight=300`). Mono = 3000 (cible secondaire).
- **SUCCÈS GLOBAL** : médiane **> 2300**, **multi-seed ≥ 3**, closed-loop ; **+ no-retrain prouvé** (Gate-S) ;
  **idéalement sous récompense de survie PURE** (béquille de shaping annealée → 0) — sinon on ne revendique pas
  la « douleur émergente », la forme reste partiellement posée.
- **KILL GLOBAL** : si, après gates, le RL **ne dépasse pas** la baseline malgré convergence (KL<0.03, std
  stable) → **négatif informatif → STOP + escalade**, pas d'enchaînement de tweaks à l'aveugle.

---

## 4. Gate étagé (gater le cher derrière le pas-cher — §1)

### Gate-0 — sonde rétine courte-portée (le plus cheap, ~minutes)
Réutilise `diag_drive_head_*.py` avec `FEAT=retina-raw` : un head supervisé sur les **rayons rétine bruts**
prédit-il le signal-repas / la distance courte-portée nettement mieux que le slot ?
**SUCCÈS** : ≥ ~35 % de la bosse (slot=14 %, brut=46 %). **KILL** : ≈14 % → la rétine n'aide pas plus que le
slot → revoir la perception (ou repli radar-oracle, variante B) **avant tout RL**. Inclut la **vérif
eau-dans-rétine** (§2.3).

### Gate-1 — le BC atteint la baseline (gratuit, ~30–60 min)
Politique BC déployée dans `baseline_multidrive_slot` (serveur Mode-1 déterministe).
**SUCCÈS** : médiane survie **≥ ~2000** sur ≥ 12 ép (dans le bruit du planner ~2300) **ET** elle close
(fraction `food_d<1m` ≈ planner, via `diag_death_multidrive.py`). Valide obs/réseau/déploiement **+ que la
rétine suffit à reproduire la nav**. **KILL** : médiane ≪ planner (<1500) ou ne close jamais → obs/perception
insuffisante → STOP, corriger avant RL.

### [GATE] — construire le plumbing RL **seulement si Gate-0 + Gate-1 passent**
`serve_mode1_collect.py` + `train_mode1_ppo.py` (cf §5).

### Gate-2 — un RL court montre un gradient de survie (depuis le BC)
Budget court (quelques centaines d'itérations / quelques heures), resets randomisés.
**SUCCÈS** : médiane **> BC (+≥200)** OU le taux « manger-quand-faim » monte (l'off-policy prend), **sans
divergence** (KL<0.03, std stable). **KILL** : survie s'effondre sous BC et y reste, ou std/KL divergent →
STOP + escalade (négatif informatif).

### Gate-3 — run long : bat la baseline (le BUT)
Médiane multi-pulsions **> 2300**, **multi-seed ≥ 3**, closed-loop. → promotion live.

### Gate-S — scalabilité no-retrain (falsifie §3/§4 pour Mode-1)
Politique **GELÉE** + une **3ᵉ pulsion jamais vue** au test → survit-elle **sans retrain** ?
**SUCCÈS** : survie maintenue avec 3 pulsions. (Le gate EST le test : « pas de retrain » est la propriété.)

---

## 5. À construire (UNIQUEMENT après les gates correspondants)

- **`survival_multi`** dans `reward_manager.gd` : `−Σ_d (1−level_d)²` + terminal mort ; vérifier
  `max_episode_steps` élevé (~3000+) pour la survie (défaut 400 est pour la locomotion).
- **resets à drives randomisés** : randomiser `SYLVAN_INIT_ENERGY/THIRST` par épisode (env déjà présents).
- **politique drive-symétrique** : `python/sylvan/control/mode1/policy.py` (encodeur token partagé + pooling
  invariant + tronc), action_dim=2.
- **BC** : `collect_mode1_bc.sh` (lance le planner, log `obs→cmd`) + `train_mode1_bc.py` (régression) +
  `serve_mode1.py` (déploiement déterministe, = `serve_planner_command` avec `plan()`→`policy(obs)`).
- **RL** : `serve_mode1_collect.py` (= `serve_planner_command` + politique stochastique + **log des transitions
  au command-cadence** : obs, commande, reward-fenêtre, done) ; `train_mode1_ppo.py` (réutilise
  `ppo/update.py` + `ppo/rollout.py` ; `action_dim=2` ; `sym_coef=0` ; `--lr 1e-4`).
- **harness** : réutiliser `baseline_multidrive_slot.sh` en pointant le serveur Mode-1.

---

## 6. Risques / caveats honnêtes (§2 ne-pas-masquer)

- **No-retrain** : vrai dans la **famille « approcher-consommer »** ; la valence ± étend à l'évitement
  (danger) ; une pulsion de nature radicalement autre = **extension** (non survendu).
- **Rétine courte-portée** = HYPOTHÈSE (Gate-0 la teste, ne pas présumer le succès). **Eau-dans-rétine** à vérifier.
- **BC hérite la myopie** → il **matche** la baseline (≈2300), ne la **bat** pas. BC = feasibility du chemin,
  PAS le succès. C'est le **RL** (Gate-2/3) qui doit battre — sinon KILL global.
- **Survie pure = sparse** : le signal « mort au pas N » est dur pour le crédit → s'appuie fort sur le warm-start
  BC + l'exploration (resets randomisés) + la béquille de shaping (annealée). HONNÊTE (§2) : on ne revendique
  « douleur émergente » (la forme apprise = la tête de valeur) que si la politique **TIENT quand le shaping → 0**
  (testé Gate-3) ; sinon la forme reste partiellement posée — on le dira.
- **Pureté** : model-free met de côté la **planification-dans-le-WM** pour la décision (le WM/slot ne servent
  plus à décider). Réconciliable plus tard (latent en input auxiliaire, ou Dreamer quand le WM saura
  l'eat-dynamics). Le WM reste **vivant** (perception slot, foraging mono).
- **Coût collecte BC sur CPU** (planner ≈117×120 forwards WM/appel) — gérable (peu d'épisodes suffisent).

---

## 7. Ops (rappels qui ont coûté cher)
venv `env_pytorch_3.12/bin/python`, **CPU OBLIGATOIRE**, racine + `PYTHONPATH=python` + `GODOT_BIN`. PPO
**`--lr 1e-4`**. Tuer un train : `pkill -9 -f serve_mode1_collect` + `pkill -9 -f train_mode1_ppo` +
`pkill -9 -f 'godot --path godot'` PUIS **vérifier** 0 restant. Lancer un train en background = la commande
python **seule**. Régime propre : `SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0
SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5`.
