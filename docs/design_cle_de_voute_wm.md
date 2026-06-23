# Carte de design — CLÉ DE VOÛTE : un World Model qui imagine la perception sous auto-mouvement (2026-06-21)

> Objectif JEPA : un rollout open-loop qui **transporte fidèlement la perception** (le bearing des objets) à travers
> une rotation **imaginée**. C'est le manque dominant mesuré (`diagnostic_perception_rotation_wm.md`), et la brique
> dont dépendent curiosité, recherche dirigée, décisions réfléchies (`archi_jepa_etat_des_lieux.md`).
>
> **Principe (owner « archi × gain, pas gain ») : chaque option a un TEST GRATUIT qui la tranche AVANT de payer un
> retrain. Rien de cher n'est lancé sans son gate gratuit passé.** (CLAUDE.md §1.)

## Le manque, en chiffres (point de départ)

Sur de vrais virages : représentation teacher-forced (encodeur) suit le bearing qui tourne **~+0.5** ; rêve
open-loop (dynamique) **~+0.15**. → manque **dominant = la DYNAMIQUE** (le rollout jette la perception) ; encodeur
**secondaire** (+0.5, marge) ; **uncertainty non mesurée proprement** (sonde reveal non concluante).

## La question-pivot (elle départage TOUTE la suite)

Le bearing est-il **(1) DANS le latent rêvé mais mal lu** [readout — *cheap*], ou **(2) JETÉ par la dynamique
pendant le rollout** [retrain — *cher*] ? → **TEST 1**.

## Tests / gates gratuits (ordonnés : le moins cher décisif d'abord)

### TEST 1 — readout vs retrain (gratuit, ~minutes) ⟵ À FAIRE EN PREMIER
Entraîner une sonde de bearing **directement sur les latents OPEN-LOOP** (cross-val par épisode) sur les segments
de virage, et comparer à REPR(+0.5) / RÊVE-actuel(+0.15).
- corr ≈ **+0.5** → l'info **EST** dans le rêve, juste mal exposée → **fix CHEAP** (tête lue sur latents rêvés /
  léger alignement ; le planner peut l'utiliser direct, **pas de retrain**).
- corr ≈ **+0.15** → l'info est **JETÉE** par la dynamique → retrain nécessaire (→ TEST 2/GATE 3).
- **Seuil pré-enregistré : +0.35.** Outil : extension de `diag_wm_rotation.py`.

### TEST 2 — audit data rotation (gratuit, si TEST 1 dit « retrain »)
Compter dans le replay les frames « **virage + objet visible + bearing qui change** » (l'événement
acquisition-par-virage) ; mesurer leur densité. Comparer à ce qu'un **collecteur SCRIPTÉ de virages** produirait.
⚠️ un babbling **aléatoire** fait PIRE (mémoire) → collecteur **scripté** (cibles à 360°, rotations commandées).
- densité suffisante → pas de recollecte ; sinon → collecteur scripté (cheap, Godot, 0 entraînement).

### GATE 3 — décisif, cheap-mais-payant (le vrai gate du retrain)
**Fine-tune court** (warm-start `wm_rich_fidele_sym`) avec une **perte AUXILIAIRE bearing-à-travers-le-rollout** :
le long de `dream_latents`, prédire le bearing du **plus proche objet perçu** (cible **color-agnostic**, dérivée du
**rayon de rétine le plus proche** — PAS « food » → reste général/§3-pur), supervisé sur les vrais virages [+ data
rotation si TEST 2 le dit]. Puis re-passer `diag_wm_rotation.py`.
- **SUCCÈS pré-enregistré : RÊVE +0.15 → ≥ +0.35, SANS casser** la pose (displacement) ni la richesse (eff_rank).
- **KILL : pas de remontée** → la dynamique ne **peut** pas transporter (capacité/arch) → escalade vers C.

## Les options de design (ce que c'est × pureté × levier × coût/risque)

| # | Option | Pureté JEPA | Levier | Coût / risque |
|---|---|---|---|---|
| **A** | **Données riches en rotation** (collecteur scripté) | pure (perception générale) | nourrit B | cheap (collecte) ; data seule peut ne pas suffire |
| **B** | **Perte aux-bearing-through-rollout** (force le rêve à garder la perception) | pure si **color-agnostic** (plus proche objet, pas food) | **DOMINANT** (cible la dynamique) | fine-tune, gaté par TEST1/3 ; trade-off fidélité à mesurer |
| **C** | **Représentation rotation-équivariante** (ω = transform connue du latent) | pure, élégante | secondaire (encodeur +0.5) | **HAUT** (refonte, from-scratch probable) → seulement si GATE 3 KILL |
| **D** | **Latent stochastique** (variables latentes = uncertainty/reveals) | pure, JEPA-natif | nécessaire pour la **CURIOSITÉ** (pilier suivant), pas pour le transport immédiat | medium-haut → **mesurer le gap reveal d'abord**, pilier ultérieur |

## RÉSULTATS (2026-06-21) — TEST 1 & TEST 2 faits

- **TEST 1 = RETRAIN** (`diag_test1_readout.py`) : sonde entraînée *sur* les latents open-loop = **+0.08** (held-out,
  split par épisode) → l'info est **jetée par la dynamique**, pas juste mal lue. (Bonus : a révélé une **fuite** dans
  `diag_wm_rotation.py` → le +0.5 « représentation » était gonflé ; held-out propre = REPR +0.14 / rêve +0.08, tout bas.)
- **TEST 2 = donnée utilisable mais acquisition rare** (`diag_test2_data_audit.py`) : rotation abondante (42–76% des
  frames) mais ~**425** événements d'acquisition (derrière→devant) au total. La rotation seule n'a pas suffi.

## Séquence recommandée (gated) — MISE À JOUR

1. ~~TEST 1~~ ✅ → RETRAIN. ~~TEST 2~~ ✅ → données ok mais acquisition rare.
2. **GATE 3a (le moins cher payant, PROCHAIN)** : fine-tune court warm-start `wm_rich_fidele_sym` + **perte
   aux-bearing-through-rollout** (color-agnostic, plus proche objet) sur la donnée **existante** (retina_forage +
   retina_eat_a) → **re-gate PROPRE held-out** (`diag_test1_readout.py`). Succès = rêve held-out remonte nettement
   (viser ≥ +0.3) SANS casser pose/eff_rank. ~30–60 min. **Premier pas payant → signalé à l'owner avant lancement.**
3. **GATE 3b** (si 3a insuffisant) : **collecteur SCRIPTÉ d'acquisitions** (cibles 360° + rotations commandées →
   densifie les ~425 ; améliore aussi la mesure held-out en donnant plus d'épisodes) + re-train + re-gate.
4. **GATE 3c** (dernier recours) : représentation **rotation-équivariante** (refonte, cher) si 3a+3b plafonnent.
5. **D (stochastique/uncertainty)** = mesurer proprement le gap reveal, puis pilier **curiosité** (étape 2 du chemin).

## Anti-patterns (ce qu'on NE fait pas)

- Pas de `--w-food` / forçage ressource dans le WM (le bearing-aux est **color-agnostic** = général, §3).
- Pas de retrain à l'aveugle (chaque retrain derrière son gate gratuit).
- Pas de babbling aléatoire pour la data (fait pire — scripté seulement).
- Pas de conclure « plafond » sans que le gate falsifiable l'ait montré (CLAUDE.md §2).

## RÉSULTATS GATE 3a / 3a′ + DÉCISION (2026-06-23)

Re-gate PUISSANT (`diag_test1_readout.py`, `BUF=retina_eat_a`, 60 ép, 261 segments test) :

| run | REPR (encodeur, TF) | TEST1 (rêve, OL) |
|---|---|---|
| ancien `wm_rich_fidele_sym` | +0.18 | +0.09 |
| v1 `wm_keystone_bearing_v1` (3a : perte bearing sur le RÊVE) | +0.22 | +0.14 |
| v2 `wm_keystone_bearing_v2` (3a′ : bearing sur rêve **+** TF) | +0.25 | +0.15 |

(eff_rank ~13 et displacement ~0.009 préservés → rien cassé.)

**Localisation nette :** 3a′ **lève la REPRÉSENTATION** (+0.18→+0.25 — presser les latents teacher-forced marche,
l'encodeur PEUT porter le bearing) **mais le RÊVE ne suit pas** (+0.14→+0.15, plateau). Donc **le verrou résiduel =
la DYNAMIQUE / le rollout open-loop** : le rêve colle globalement au TF (perte rollout ~0.12) mais **perd la
sous-composante fine du bearing** en route. Le levier « perte auxiliaire » plafonne à **+0.15** (seuil +0.35).

**Règle budget §1 : STOP les tweaks de poids** (2 négatifs informatifs) → escalade vers un levier DIFFÉRENT.

**DÉCISION OWNER (2026-06-23) : on reste JEPA-PUR** (PAS l'alternative active-rétine). Suite :
- **3b — données (cheaper-first)** : collecteur SCRIPTÉ d'acquisitions denses (cibles 360° + rotations commandées →
  densifier les ~425 événements) + retrain avec les pertes bearing déjà en place. Hypothèse : la dynamique n'apprend
  pas le transport faute de signal (acquisitions rares).
- **3c — architecture (ce que la localisation désigne)** : rollout rotation-ÉQUIVARIANT (ω = transform connue du
  latent) OU latent STOCHASTIQUE — attaque le smearing du rollout. Vraie refonte.
- Ordre : 3b → si plateau → 3c. Chacun gaté par `diag_test1_readout.py` (`BUF=retina_eat_a`, viser TEST1 ≥ +0.35).

**Outils prêts :** `train_wm_command.py` a `--w-bearing` (rêve) + `--w-bearing-tf` (représentation) + `bearing_head`
(color-agnostic, NON sauvée) ; `diag_test1_readout.py` (env `WM_CKPT` + `BUF`) = le gate held-out. Checkpoints
`wm_keystone_bearing_v1/v2` = assets (NON promus live ; live de secours = planner-coords + `wm_command_hex_v2`).
