# PLAN — World Model OBJECT-CENTRIC PUR (internaliser le slot, JEPA strict) — 2026-06-23

> Owner : « maximum JEPA pur ; si on rogne sur la pureté, des briques plus tard tombent/deviennent impossibles. »
> Ce plan pose le design cible PUR, les décisions qui conditionnent les modules futurs (mémoire, curiosité,
> hiérarchie, multi-pulsions), les pièges anticipés, et les TESTS GRATUITS qui gatent chaque pas cher. **Rien
> n'est entraîné tant que les gates gratuits ne sont pas passés** (discipline CLAUDE.md §1).

## 0. Pourquoi ce chantier (recadrage owner, validé)

La clé de voûte a été résolue PRATIQUEMENT par un **slot codé-main** (coordonnée objet explicite, 1 seul objet,
transport analytique, vivant dans le *planner* via `plan()` override_pos). Ça MARCHE (engagement arrière 2/4,
foraging 1045) et c'est JEPA-légitime (perception + dynamique apprises, zéro oracle) — MAIS c'est un **échafaudage
codé-main**, pas une représentation **apprise et émergente dans l'état du World Model**. Empiler la mémoire spatiale
dessus = construire sur une dette → impureté qui se propage. **Le chantier pur = INTERNALISER le slot dans le WM
comme un composant APPRIS de l'état-monde**, ce qui en une refonte débloque trois modules : pureté récupérée +
mémoire spatiale émergente (object permanence = propriété de l'état) + substrat d'incertitude (curiosité).

## 1. Le design CIBLE pur (north-star de la brique)

**Encodeur object-centric :** `Enc(observation) → {slot_k}` un ENSEMBLE de K slots (permutation-invariant, type
slot-attention). Chaque slot = **[position égocentrique (géométrie) ⊕ identité (couleur/feature) ⊕ (option)
incertitude]**. La géométrie et l'identité sont SÉPARÉES (drive-agnostique, §3).

**Dynamique (le rêve) :** chaque slot perceptuel est transporté par l'**ego-motion** que le WM prédit (déplacement
d_fwd,d_lat,d_yaw — F2 a montré que c'est apprenable +0.98). Le transport d'une coordonnée par une transformation
rigide est **équivariant PAR CONSTRUCTION** → le rêve ne PEUT plus smear la perception sous rotation (résout la clé
de voûte *purement*, là où le latent monolithique plafonnait à +0.30). Slot non perçu → **persiste et se transforme**
= object permanence = mémoire.

**Objectif d'apprentissage = JEPA strict :** prédire la **prochaine REPRÉSENTATION** (slot encodé) — PAS reconstruire
l'entrée. Loss = énergie de prédiction dans l'espace des slots + anti-collapse (VICReg) + **consistance de transport
auto-supervisée** (le slot transporté à t doit matcher le slot encodé à t+1). **Pas de reconstruction perceptuelle.**

**Incertitude (extension) :** variable latente par slot (« ce que je verrais en tournant », plusieurs futurs) → la
curiosité = coût intrinsèque qui réduit cette incertitude, optimisé par l'Actor → exploration ÉMERGE.

## 2. Les 3 principes JEPA appliqués (la jauge de pureté)

1. **Prédire en représentation, PAS reconstruire.** → on DROP la reconstruction perceptuelle (retina/radar obs_head).
   On GARDE la prédiction du proprio/déplacement (c'est le « self-model » du corps, pas de la perception externe ;
   et la displacement-head alimente le transport des slots). Le canal perceptuel passe 100% object-centric en latent.
2. **Anti-collapse non-contrastif (VICReg).** Déjà en place ; étendu aux slots (sinon tous les slots → même point).
3. **Mode-2→Mode-1, H-JEPA.** L'état-monde = ensemble de slots = naturellement compatible planification (le planner
   roule les slots), abstraction hiérarchique (objets→groupes→régions), et amortissation Mode-1. On ne ferme aucune
   de ces portes.

## 3. Décisions de design qui CONDITIONNENT les briques futures (ne PAS rogner)

| Décision | Pourquoi (quelle brique tombe sinon) |
|---|---|
| **Multi-slot dès le départ (K slots)** | Mémoire spatiale = se souvenir de PLUSIEURS ressources. Single-slot → mémoire impossible sans tout refaire. |
| **Géométrie ⊕ identité SÉPARÉES** | Multi-pulsions : eau/prédateur = même machinerie, identité différente. Slot food-spécifique → multi-drive impossible. |
| **Slot prédit en latent, reconstruction perceptuelle DROP** | Pureté principe 1 ; et la pression de reconstruction COMBAT la fidélité du rêve (mesuré, compromis richesse↔fidélité). |
| **Slot uncertainty-ready (autoriser variance/latent)** | Curiosité = réduire l'incertitude des slots. Slot point-estimate → curiosité impossible. |
| **Auto-supervision (consistance de transport), label de position SEULEMENT en sonde/éval** | Émergence/pureté : un slot supervisé par la position vraie = le hack codé-main re-déguisé. La consistance rigide FORCE le slot à DEVENIR une coordonnée (seul un coordonnée transforme ainsi) → spatial SANS label. |
| **Égocentrique + re-grounding sur la perception** | Pragmatique : dérive bornée comme le MPC. Allocentrique (carte monde) = upgrade mémoire-long-terme PLUS TARD ; éviter d'introduire des coordonnées-monde maintenant. |
| **Additif/parallèle, garder le secours** | Ne pas casser le forager vivant ; le nouveau WM se valide CONTRE le slot codé-main comme baseline. |

## 4. Pièges anticipés + mitigations

- **Instabilité du slot-attention** (binding objets) → notre monde est SIMPLE (quelques pastilles colorées, rétine
  1D) ; l'extraction « objet le + proche » est déjà à 0.08 m (retina_head). Risque d'extraction FAIBLE. Le dur = le
  rendre appris+émergent+transporté+persistant, pas l'extraction.
- **Dérive de l'intégration ego-motion** sur long horizon (mémoire) → re-grounding à chaque re-perception (motif MPC) ;
  borne la dérive. Mesurer la dérive AVANT de prétendre « mémoire ».
- **Collapse des slots** (tous identiques, loss trivialement nulle) → VICReg sur slots + diversité ; surveiller eff_rank.
- **Drop reconstruction casse le proprio/déplacement** → on GARDE la prédiction proprio+déplacement ; on ne drop QUE
  la reconstruction perceptuelle externe.
- **Association de données multi-slot entre frames** (quel slot = quel objet à t+1) → la partie ML dure ; monde simple
  aide ; la consistance de transport fournit le signal d'association. À tester (F-pure-2).
- **Auto-supervision insuffisante** (le slot ne devient pas spatial sans label) → F-pure-1 le tranche GRATUITEMENT ;
  si échec, un anchor de position LÉGER = compromis de pureté à FLAGGER explicitement à l'owner (pas en douce).

## 5. Plan PHASÉ et GATÉ (chaque pas cher derrière un test gratuit)

### Phase 0 — DESIGN GELÉ + GATES GRATUITS (0 entraînement du WM) ⟵ ON COMMENCE ICI
Trois tests offline qui dé-risquent les paris centraux AVANT toute refonte (comme F1/F2/F3 l'ont fait pour le slot) :

- **F-pure-1 — le slot spatial ÉMERGE-t-il SANS label ? ✅ SUCCÈS (2026-06-23).** `diag_fpure1b_attn.py` : encodeur à
  ATTENTION GÉOMÉTRIQUE (soft-argmax sur rayons → coordonnée par construction, style retina_head) + transport par
  l'ego-motion VRAIE (consistance auto-supervisée `transport(slot_t,egomotion)≈stopgrad(slot_{t+k})`, gap k=8) + VICReg,
  ZÉRO label. **Bearing corr |0.69-0.70| = ÉGAL à la borne supérieure supervisée (+0.70)** ; la consistance baisse pendant
  que la magnitude monte → le slot DEVIENT spatial tout seul. Le signe inversé = JAUGE (la consistance fixe le repère à
  une réflexion près) = non-problème. **⚠️ leçon : un MLP plat ÉCHOUE (+0.16, F-pure-1 v1) — il FAUT l'attention
  géométrique** (le projet le savait déjà pour le décodage rétine). → pari de pureté VALIDÉ : la représentation d'objet
  s'apprend émergente + label-free. (NB ego-motion vraie ici ; dans le WM = displacement-head, F2 +0.98.)
- **F-pure-2 — séparation MULTI-slot.** Sur données multi-pastilles, K slots se lient-ils à des objets DISTINCTS
  (chaque slot suit un objet) ? Métrique : assignation stable + bearing par slot. SUCCÈS = K≥2 objets séparés proprement.
- **F-pure-3 — PERMANENCE.** Occlure un objet (sort du champ / derrière), le slot transporté prédit-il sa position au
  retour ? Métrique : erreur de bearing à travers le trou d'occlusion. SUCCÈS = erreur bornée (≪ « slot perdu »). =
  la précondition de la mémoire.

→ Si les 3 passent : le design pur est viable, on gate la Phase 1. Si l'un casse : on a appris GRATUITEMENT et on
re-conçoit (pas de retrain à l'aveugle).

### Phase 1 — INTERNALISER les slots dans le WM (la refonte, gatée par Phase 0)
Ajouter la voie object-centric au WM (encodeur slots + transport ego-motion + losses JEPA/transport/VICReg, DROP
reconstruction perceptuelle), warm-start si possible. **Gate = (a) le slot RÊVÉ appris bat le monolithique +0.30 vers
+0.65 ; (b) émergent (zéro coordonnée codée-main dans la boucle) ; (c) re-gate closed-loop `diag_nav_ab_slot.sh` :
engagement ≥ le slot codé-main ; (d) eff_rank/displacement non cassés.** Succès → le slot codé-main est DISSOUS.

### Phase 2+ — débloqués par Phase 1 (chantiers suivants, séparés)
- **Mémoire spatiale** : persister les slots entre replans (ego-motion + re-grounding) → recherche DIRIGÉE vers une
  ressource mémorisée hors-champ (north-star « chercher »). Émergent, pas un SLAM ajouté.
- **Curiosité** : slots stochastiques → coût intrinsèque « réduire l'incertitude » → exploration émergente.
- **Multi-pulsions / critique foresighted**, **H-JEPA**, **Mode-1** : couches hautes, plus tard.

## 6. Ce qu'on NE fait PAS maintenant (discipline de scope)
- Pas de stochasticité/curiosité en Phase 1 (mais design uncertainty-ready). Pas d'allocentrique (égocentrique+regrounding
  d'abord). Pas de Mode-1/hiérarchie. Pas de casse du forager vivant (slot codé-main = baseline+secours).
- Pas de retrain avant que Phase 0 (gratuit) passe. Pas d'anchor de position caché : si nécessaire, FLAGGÉ.

## 7. Critère de fin du chantier (le BUT, pas le proxy)
Un WM dont l'**état-monde porte des slots d'objets APPRIS** (géométrie⊕identité), **transportés par l'ego-motion**
(équivariant par construction), **persistants** (permanence) et **uncertainty-ready** ; le forager latent lit ces
slots (plus aucune coordonnée codée-main) ; engagement+foraging ≥ le slot codé-main. = la clé de voûte rendue PURE,
et le substrat sur lequel mémoire et curiosité émergeront SANS nouveau hack.
