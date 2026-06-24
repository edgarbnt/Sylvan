# Recul archi — JEPA (LeCun) × état du projet × chemin pur (2026-06-21)

> Prise de recul demandée par l'owner. Principe directeur **non négociable** : raisonner **archi × gain**, jamais
> « gain » seul. Ne pas brûler d'étapes pour atteindre le but plus vite — ça remplit l'archi de hacks et tue la
> pureté. But réel = un **vrai JEPA** appliqué à une entité qui survit dans une forêt (multi-pulsions, curiosité,
> décisions réfléchies, mémoire). Chemin long, étape par étape.

## 1. L'architecture JEPA de LeCun (référence) — *A Path Towards Autonomous Machine Intelligence*, 2022

**6 modules :**
- **Perception** — encode l'état du monde en représentation.
- **World Model** — le cœur : prédit les états futurs **en représentation** (pas en pixels), multi-pas, et **gère
  l'incertitude** via des **variables latentes** (plusieurs futurs plausibles).
- **Cost** — scalaire « inconfort » = **intrinsèque** (câblé, immuable = les pulsions/la nature) + **critique**
  (appris, prédit le coût intrinsèque *futur* = une valeur).
- **Actor** — propose des séquences d'actions, les **optimise à travers le World Model** pour minimiser le coût
  (= la planification).
- **Short-term memory** — stocke états/prédictions/coûts.
- **Configurator** — exécutif : configure les autres modules selon le but du moment.

**3 principes qui DÉFINISSENT le JEPA :**
1. **Prédire en représentation, pas reconstruire l'entrée** (l'encodeur jette l'imprévisible/non pertinent).
2. **Anti-collapse non-contrastif** (VICReg…).
3. **Mode-2 (planifier, lent) → Mode-1 (politique amortie, rapide)** ; **H-JEPA** = plusieurs World Models à des
   échelles de temps/abstraction (planification hiérarchique).

## 2. Où en est le projet (honnête)

| Module LeCun | État |
|---|---|
| Perception | ✅ rétine apprise (raycast couleur, « rouge=bouffe » émergé) |
| World Model | 🟡 **partiel** — RSSM **déterministe**, encore des termes de **reconstruction** (résidu Dreamer), **pas d'incertitude/variables latentes**, **fidélité multi-pas fragile** (surtout la perception sous rotation) |
| Cost intrinsèque | ✅ faim/soif (analytique) — pile le « immuable » de LeCun |
| Critique (valeur) | ✅ `value_head` sur le latent (V = « repas imminent ») — bon move archi |
| Actor | 🟡 **Mode-2** (planner MPC) ✅ ; **Mode-1 absent** (prévu en dernier) |
| Short-term memory | 🟡 replay (entraînement) seulement |
| **Mémoire épisodique/spatiale** | ❌ (« qu'il se souvienne ») |
| **Configurator** | ❌ (juste arbitrage homéostatique) |
| **Curiosité / exploration** | ❌ (le scan CHERCHER est codé-main, pas une pulsion apprise) |
| **Hiérarchie (H-JEPA)** | ❌ |

→ On a **l'échafaudage** (perception + WM partiel + coût + critique + planificateur Mode-2). Il manque les pièces
qui font l'**intelligence** visée : incertitude, curiosité, mémoire, hiérarchie, configurator.

Acquis archi réels (à créditer, ≠ hacks) : critique-sur-latent ; `--w-rollout` (corrige l'exposure-bias du rêve) ;
**découplage substrat/pulsions** (WM riche+fidèle entraîné une fois, pulsions = têtes) ; WM symétrisé.

## 3. Le fil rouge de TOUS nos murs = le World Model (la clé de voûte)

Reprendre nos blocages : eat-dynamics inapprenable par régression (→ critique) ; close/asymétrie/dérive open-loop
(→ fidélité) ; engage-derrière (→ encodeur jette les percepts rares + rêve ne transporte pas la rotation, cf
`diagnostic_engagement_perception.md`) ; recherche forcément non-dirigée (→ le WM ne sait pas imaginer ce qu'un
virage révélerait).

**Tout remonte à UNE chose : la qualité prédictive du World Model — et précisément sa capacité à prédire la
PERCEPTION future sous l'effet de ses PROPRES mouvements (surtout la rotation).** Aujourd'hui il prédit bien comment
le *corps* bouge (tête déplacement), mais mal comment la *perception* change quand le corps tourne. Or **« prédire
la prochaine représentation perçue » EST le cœur du JEPA.** + tension récurrente **richesse↔fidélité** (VICReg riche
mais rêve qui dérive) = exactement le défi nommé par LeCun.

## 4. CHERCHER relu en « archi × gain »

Le scan codé-main est un move **gain**, pas **archi**. La version *pure* (LeCun) = une **pulsion de CURIOSITÉ dans
le coût intrinsèque** (réduire l'incertitude / chercher la nouveauté) que l'**Actor optimise à travers le World
Model** → l'exploration **émerge** de la planification. MAIS ça **bute sur le même mur** : planifier « tourner
réduira mon incertitude » exige un WM qui sait **imaginer la perception après rotation**. Donc **CHERCHER-réflexe =
échafaudage temporaire**, pas la brique pure. (Détail + gate négatif : `diagnostic_engagement_perception.md`.)

## 5. Le chemin pur, étape par étape (ordre non négociable)

1. **CLÉ DE VOÛTE — World Model qui prédit la prochaine représentation perçue, fidèlement, sous l'auto-mouvement,
   avec INCERTITUDE** (variables latentes = « ce que je verrais en tournant », plusieurs possibles). Le vrai JEPA.
   Sans lui, rien d'émergent ne tient. Pistes pures générales : représentation rotation-aware/équivariante ; perte
   de fidélité perceptuelle du rollout (déjà amorcée `--w-rollout`) ; latent stochastique ; données riches en
   rotation. **Payoff per-tâche incertain MAIS débloque tout l'arbre → c'est l'investissement archi×gain n°1.**
2. **Curiosité = coût intrinsèque** (incertitude/nouveauté) → l'Actor planifie pour la réduire → **exploration
   ÉMERGE** et remplace le scan codé. (Possible seulement après 1.)
3. **Mémoire épisodique/spatiale** → « l'eau est par là » → navigation vers un but **non perçu mais connu** émerge
   (≠ chercher au hasard). Pilier distinct, parallèle.
4. **H-JEPA (hiérarchie)** → planifier long et abstrait.
5. **Mode-1** (politique amortie) → habitudes rapides. En dernier.
6. **Multi-pulsions** tout du long (termes de coût), arbitrées par le critique → **décisions réfléchies** =
   minimiser l'inconfort futur prédit.

## 6. À quoi ressemble l'intelligence à l'arrivée

Une entité qui **explore parce qu'elle est incertaine**, **se souvient** où sont les ressources, **arbitre**
faim/soif/curiosité/danger en **imaginant** le futur, et **planifie hiérarchiquement** — le tout **émergent** du
couple *World Model × coûts*, **rien câblé par ressource**. C'est le « intelligent dans la forêt ».

## 7. Décision en cours

**MESURÉ** (sondes gratuites, `diagnostic_perception_rotation_wm.md`) : sur de vrais virages, la **représentation**
(teacher-forced) suit le bearing qui tourne **modérément (~+0.5)**, mais le **rêve** (open-loop) le **perd (~+0.15)**.
→ Le manque **dominant** de la clé de voûte = la **DYNAMIQUE / fidélité du rollout sous rotation** (l'encodeur est
secondaire, il a une marge). Convergence : ce même rêve faible explique le foraging modeste ET l'échec de la
recherche latent-pure (`diagnostic_engagement_perception.md`). **Prochaine étape archi×gain : concevoir cette clé
de voûte sur ces mesures** (pistes : données riches en rotation, perte de fidélité rollout ciblée perception,
représentation équivariante, latent stochastique pour l'uncertainty) — départager par tests gratuits AVANT tout retrain.

## 8. MAJ POST-CLÉ-DE-VOÛTE (2026-06-23) — résolue PRATIQUEMENT via un SLOT object-centric (≠ fix pur)

Voir `docs/design_wm_factorise.md` (autoritaire). Parcours gratuit : 3b (mur géométrique) → 3c-linéaire (transport
latent +0.30, entortillé) → re-validation closed-loop (verrou = WM, pas moteur) → F1/F2/F3 → S1.
**Découverte :** le readout **pur-latent-valeur** (`plan_latent`) est intrinsèquement LOSSY (le rêve monolithique perd
l'objet, +0.30). Le fix = un **SLOT** = coordonnée ego de l'objet, lue depuis la perception apprise (retina_head) et
**transportée par la displacement-head** du WM (intégration = `plan()` override_pos). Un slot qui PERSISTE transporte
le bearing à **+0.90/+0.65** vs +0.30. **Résultat closed-loop : engagement arrière 0/4→2/4 (=oracle), global 14/16 >
oracle 10/16 ; foraging survie méd 1045 > oracle 610.** Promu forager vivant (commit 1ee1f85). Base déplacement
**validée complète en reach** (tous azimuts, `diag_rear_gap.sh` : le « trou ±179° » = temps, pas moteur) ; seul résidu
= orbite terminale (forward-only, masquée par eat_radius).

**Ce que le slot CHANGE à l'audit (honnête) :**
- Il **route AUTOUR** du mur de la §7 (rêve latent qui perd la perception) au lieu de le résoudre purement : la
  position d'objet vit dans une **coordonnée explicite** (object-centric), pas dans une représentation latente
  rotation-équivariante. C'est JEPA-légitime (perception + dynamique APPRISES, zéro oracle) mais c'est un
  **échafaudage object-centric codé-main** (1 seul objet le + proche, transport analytique), PAS des slots appris
  ni l'incertitude. Le WM reste **déterministe + reconstructif** (🟡 inchangé).
- Il **débloque la suite** : on a maintenant une **position d'objet qui persiste dans un rêve** = la brique de base
  d'une **mémoire spatiale** (persister le slot ENTRE replans = se souvenir d'une ressource non perçue → recherche
  DIRIGÉE, l'étape « chercher » du north-star, jusqu'ici impossible).

**Table MAJ :**

| Module LeCun | État post-23/06 |
|---|---|
| Perception | ✅ rétine apprise |
| World Model | 🟡 partiel (déterministe, reconstructif, pas d'incertitude) — MAIS contourné en pratique par le slot object-centric |
| Cost intrinsèque | ✅ faim/soif |
| Critique (valeur) | 🟡 `value_head` existe mais le chemin VIVANT = slot + coût analytique (le pur-latent-valeur était lossy) ; critique foresighted multi-pulsions = myope |
| Actor | 🟡 Mode-2 (MPC) ✅ ; Mode-1 absent |
| Mémoire spatiale/épisodique | ❌ — mais le SLOT en est la brique de base (persister entre replans = chantier naturel) |
| Configurator | ❌ |
| Curiosité / incertitude | ❌ (WM déterministe) |
| Hiérarchie (H-JEPA) | ❌ |

**Candidats prochain chantier (à décider) :** (A) **mémoire spatiale** (persister les slots → recherche dirigée,
north-star « chercher », build direct sur le slot) ; (B) **multi-pulsions + critique foresighted** (arbitrage =
décisions réfléchies, couche ALife) ; (C) **curiosité/latent stochastique** (exploration émergente, mais exige
l'incertitude du WM = plus dur) ; (D) **clé de voûte PURE** (WM équivariant/stochastique — payoff immédiat baissé car
le slot route autour). Reco = A (momentum + débloque le north-star).
