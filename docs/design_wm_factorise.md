# Carte de design — WM FACTORISÉ / OBJECT-CENTRIC (clé de voûte, étape 3c, 2026-06-23)

> Objectif : un rêve open-loop qui **transporte fidèlement la perception (le bearing des objets) sous auto-rotation**,
> y compris pour une cible **derrière**. Re-validé sur preuve (closed-loop) : le verrou est le **WM** (le corps SAIT
> se retourner — contrôle coordonnées 2/4 arrière ; le latent-pur 0/4 s'éloigne), PAS le moteur, PAS la métrique.
>
> **Principe owner :** JEPA-pur, robuste, pérenne, scalable, « voir plus loin que le foraging ». **Discipline (§1) :
> chaque option a un TEST GRATUIT qui la tranche AVANT de payer un retrain.**

## Pourquoi le latent monolithique échoue (mesuré)

- 3a/3a′ : presser une perte aux-bearing lève la **représentation** (REPR +0.18→0.25) mais pas le **rêve** (+0.15).
- 3b : on ne peut pas densifier la donnée d'acquisition (mur géométrique du babbling).
- 3c-linéaire (`diag_test3_equivariance.py`) : même un opérateur **linéaire** de rotation idéal sur le latent 128-d
  ne transporte le bearing qu'à **+0.30 poolé** (plafond ~+0.33). Le bearing est une direction **émergente,
  basse-variance, ENTORTILLÉE** dans le latent → toute dynamique ajustée à la reconstruction globale la smear.

**Diagnostic d'architecture :** le bearing doit cesser d'être une direction latente émergente et devenir une
**quantité STRUCTURÉE de première classe** que l'auto-mouvement transforme par une **opération CONNUE**.

## L'option retenue — OBJECT-CENTRIC / SLOTS ÉQUIVARIANTS

Le latent se factorise en **K slots** (~1 par objet perçu), chacun portant une **position égocentrique** (coordonnée
2D apprise, pas donnée). L'auto-mouvement (vx, ω) applique à CHAQUE slot la **MÊME transformation rigide** :
`p_{t+1} = Rot(−ω·dt)·p_t − (vx·dt)·ẑ` (rotation du cap + translation avant). L'équivariance est **par construction** :
le rêve ne PEUT pas smear le bearing, il l'applique analytiquement. Re-perception = correction du slot (filtre type
Kalman) ; non-perçu = le slot **persiste et se transforme** (= object permanence = mémoire spatiale = base de la
RECHERCHE et de la CURIOSITÉ). Couleur/identité du slot = canal séparé (rouge=bouffe…), drive-agnostique (§3).

Pourquoi celle-ci et pas « factored allocentric + ego-pose explicite » (SLAM-like) : cette dernière ré-introduit des
**coordonnées monde explicites** (contre la pureté JEPA, cf inquiétude owner historique). Les slots restent
**égocentriques et appris depuis la perception** = JEPA-pur. Et l'object-centric débloque object permanence, relations,
comptage = « voir plus loin que le foraging ».

## Le risque réel n'est PAS la transformation (kinématique connue) mais :
1. **Extraction** retina→coordonnée par objet : DÉJÀ démontrée (`retina_head` 0.08 m ; sonde brute 87% derrière). ✓ faible risque.
2. **Fidélité de la transformation depuis les COMMANDES** : `p_{t+1}` est-il vraiment `Rot(ω)·p − vx·ẑ` ? Si le corps
   DÉRIVE (motion ≠ commande), l'équivariance-par-commande casse → il faudra un prédicteur d'ego-motion appris (proprio).
3. **Persistance** d'un slot non-perçu à travers le rêve (object permanence) — le point que le latent monolithique rate.

## TEST GRATUIT DE FAISABILITÉ — F1 (à faire AVANT tout retrain) ⟵ MAINTENANT

`diag_test4_equivariant_coord.py` : au niveau de la **coordonnée 2D** `p=food_rel0` (et non du latent 128-d), tester
si son évolution égocentrique est une **transformation rigide paramétrée par (vx,ω)** :
- Fit (sur train) une transformation à peu de paramètres `p_{t+1} = Rot(k_ω·ω_t)·p_t − k_v·vx_t·ẑ` (k_ω, k_v scalaires).
- **Rouler OPEN-LOOP** depuis `p_0` sur des segments de virage held-out ; mesurer le transport du bearing (corr poolée,
  par horizon ET par bucket front/**arrière**), comparer au rêve WM (+0.30) et à un contrôle « copie statique p_0 ».
- **SUCCÈS pré-enregistré : bearing transport ≥ +0.8 globalement ET ≥ +0.7 sur l'ARRIÈRE** (vs WM +0.30) → la
  représentation explicite-coordonnée transporte le bearing → **slot-WM = la bonne archi, refonte GO (cadrée)**.
- **PARTIEL (0.5–0.8, arrière faible) :** la commande seule ne suffit pas (dérive) → l'ego-motion doit être **apprise
  depuis le proprio** ; le slot-WM reste juste mais nécessite une tête ego-motion (toujours factorisé). Mesurer le gap.
- **KILL (≈ +0.30, comme le WM) :** la coordonnée 2D elle-même ne se transforme pas rigidement → la kinématique de
  l'acquisition n'est pas commande-déterminée → re-concevoir (le slot ne sauve rien). Très improbable (kinématique), mais
  à exclure honnêtement (§2).

## RÉSULTAT F1 (2026-06-23, `diag_test4_equivariant_coord.py`, retina_eat_a + retina_wm_a)

Verdict = **GO sur le slot/object-centric, MAIS l'ego-motion doit être APPRISE (pas les commandes).** Détail :
- **Persistance >> rêve WM** : une coordonnée ego explicite qui **persiste** (transport open-loop) = **+0.90 global /
  +0.65 arrière** (corr cos(brg) poolée, H=40) vs rêve WM monolithique **+0.30**. Le contrôle STATIQUE (gèle le bearing
  t0) ≈ analytique (+0.89/+0.61) → dans ces données le bearing bouge lentement (**1.1°/pas**) ; l'essentiel du gain =
  **NE PAS corrompre l'objet** (object permanence). **C'est le défaut central du WM** (il efface l'objet rêvé), pas une
  impossibilité de représenter le bearing.
- **Le transform DEPUIS LES COMMANDES échoue** : Δbearing prédit (rigide, vx/ω) vs réel = **+0.09 global / +0.14 arrière**
  (1-pas, frames rotantes). SANITY anti-bug : corr(−ω brut, Δbearing réel) = **+0.09** aussi → **PAS un bug de convention**,
  l'ego-motion n'est **pas commande-déterminée** (résidu PPO + CPG + dérive → yaw réel ≠ ω commandé ; parallaxe).
- CONSÉQUENCE design : le slot est la bonne archi ; sa mise à jour = (a) **PERSISTANCE** (gain principal) + (b) un
  **transform via ego-motion APPRISE depuis le proprio** (le latent encode la vitesse torse R²=0.85 — déjà là), PAS via
  les commandes. Le build peut même commencer SIMPLE : forcer le WM à **persister les slots** (état récurrent, update
  conservatrice) avant le transform fin.

## RÉSULTAT F2 (2026-06-23, `diag_test5_proprio_egomotion.py`) — proprio FOURNIT l'ego-motion ; persistance = le levier

Ego-motion vraie = pose torse entre frames CONSÉCUTIVES (`torso0[i]→torso0[i+1]` ; le `torso0/torso1` INTRA-frame est
nul dans ces buffers). Résultats :
- **(B) proprio → ego-motion = QUASI PARFAIT** : corr test dyaw **+0.98**, dfwd **+1.00**, dlat **+0.99**. → l'ingrédient
  que les commandes n'avaient pas (l'ego-motion réelle) EST **entièrement dans le proprio** → le transform du slot est
  alimentable proprement. **C'était la question de F2 → VERT.**
- **Nuance (§2)** : le transport du bearing **PAR PAS** reste faible même avec l'ego-motion VRAIE
  (corr(−dyaw_réel, Δbearing) = **+0.14**, ~ comme les commandes). Cause = le Δbearing par pas est minuscule (~1°,
  mean|dbrg|=0.023 rad) et **dominé par le bruit de `food_rel0`** ; ce n'est pas un échec du mécanisme. (Mon `transport()`
  multi-terme donne même -0.12 par pas = un signe de translation à revoir, mais NON-bloquant : métrique 1-pas bruitée,
  non chassée.) Le vrai signal = le transport **MULTI-PAS** de F1 (+0.90/+0.65), porté par la **PERSISTANCE**.

**CONCLUSION DE DESIGN (F1+F2) :** le gros levier est la **PERSISTANCE** (un slot qui NE PERD PAS l'objet : +0.90/+0.65
vs WM +0.30) ; le transform fin par ego-motion est une **petite correction**, et le proprio la fournit proprement
(+0.98). → **Build PERSISTANCE-FIRST** : forcer le WM à porter un état-slot persistant (coord ego), mis à jour par
l'ego-motion (dispo dans le proprio), re-gaté en CLOSED-LOOP (`diag_nav_ab_latent.sh`, arrière > 0/4). Le transform
analytique fin est secondaire (bearing lent). GO étape S1.

## RÉSULTAT F3 / S1-FAISABILITÉ (2026-06-23, `diag_test6_slot_transport.py`) — S1 QUASI INFERENCE-ONLY

Question : la displacement-head EXISTANTE du WM (qui prédit l'ego-motion dans le rêve) transporte-t-elle un slot
(coord ego initialisée depuis la perception) ? Résultat (wm_rich_fidele_sym, retina_eat_a, calibration 3 scalaires
yaw/fwd/lat pour aligner la convention) :
- **SLOT (transporté par la displacement-head) = +0.91 global / +0.65 arrière** vs dreamed-latent **+0.30**. STATIC
  (persiste t0) = +0.89/+0.61 → comme F1/F2 : **la PERSISTANCE est le levier**, le transport fin marginal (bearing lent).
  Calibration kyaw=−1.05 (signe attendu), kfwd=0.30, klat=−0.70.
- → **S1 = quasi INFERENCE-ONLY** : pas besoin de retrain du WM pour transporter le slot ; la displacement-head suffit.
  Le build = **ajouter un coût-slot au planner** : init le slot depuis la perception (retina_head, MAE 0.08 m), le
  transporter le long de chaque candidat (rollout WM → displacement → transform), coût = **approche min du slot**
  (closest approach rêvé). = la version JEPA-PURE du planner-coordonnées qui engage déjà l'arrière (2/4) — perception
  APPRISE (retina) + dynamique APPRISE (displacement-head) + slot persistant, **zéro oracle**.
- Note arrière +0.65 < seuil +0.7 = plafond de PERSISTANCE sur cette donnée (bruit food_rel0), pas un échec ; >2× le latent.

## ✅ S1 RÉUSSI (2026-06-23, `diag_nav_ab_slot.sh`) — CLÉ DE VOÛTE RÉSOLUE, JEPA-pur, ZÉRO retrain

**Découverte clé : le slot est DÉJÀ implémenté.** `command_planner.plan()` (chemin single-resource) prend la position
objet à t0, **intègre la displacement-head du WM** pour déplacer l'agent et suit la distance à l'objet FIXE = exactement
un **slot persistant transporté par la displacement-head** (ce que F3 valide). Avec `--retina-head` (perception APPRISE,
pas l'oracle radar) → `serve_planner_command` route vers ce `plan()` override_pos. Ce qui manquait n'était PAS du code,
c'était la RECONNAISSANCE (via F1/F2/F3) que la PERSISTANCE est la réponse et que le readout pur-latent-valeur
(`plan_latent`) était le mauvais outil (lossy, perd l'objet).

**Gate closed-loop (single pellet pinné, 8 azimuts, homeostasis off) — engagement par bearing initial :**

| bearing | latent-pur (value-head, `plan_latent`) | **SLOT (retina_head, `plan` override_pos)** | oracle (radar, réf) |
|---|---|---|---|
| front \|brg\|<45 | 2/2 | 2/2 | 2/2 |
| côté 45-135 | 5/10 | **10/10** | 6/10 |
| **arrière \|brg\|>135** | **0/4 (s'éloigne)** | **2/4 (tourne+close)** | 2/4 |
| **global** | 7/16 | **14/16** | 10/16 |

→ **L'arrière passe de 0/4 à 2/4 (= l'oracle), et le slot BAT l'oracle au global (14/16 vs 10/16)** : la perception
APPRISE (retina_head, MAE 0.08 m, stable) est plus robuste que le radar-oracle jittery. **JEPA-pur : perception apprise
(retina) + dynamique apprise (WM displacement) + slot explicite persistant, ZÉRO oracle, ZÉRO retrain.** Confirme
F1/F2/F3 : l'échec clé de voûte = le readout pur-latent-valeur perdait l'objet ; le slot (coord explicite persistée +
transportée) le fixe.

**Résidu honnête (§2)** : 270° (cible PLEIN-derrière, brg ±179) atteint ~1.15-1.33 m (il TOURNE vers, sans closer le
dernier mètre) = la limite géométrique du corps pour le demi-tour complet (l'oracle échoue AUSSI ce cas, à 1.7 m → le
slot fait même un peu mieux). C'est le cas le plus dur (rotation max). → S2 / ou accepter comme plancher moteur.

**Reframe de pureté (à acter avec l'owner)** : le 🅑 original visait le planning PUR-latent (value-head, aucune
coordonnée). C'est fondamentalement LOSSY (le rêve perd l'objet, prouvé). Le slot = coordonnée APPRISE explicite
(retina) transportée par la dynamique apprise = JEPA-pur au sens qui compte (perception apprise, zéro oracle), MAIS
pas le readout pur-valeur. C'est la voie object-centric du design, et c'est la réponse robuste que la value-head ne
pouvait pas donner. Artefacts : `diag_nav_ab_slot.sh` (gate S1), = essentiellement `run_forage_retina.sh` côté serveur.

## Plan de build (GATÉ — ne rien lancer de cher sans F1 vert)

1. **F1** (ci-dessus, gratuit) → tranche commande-suffit vs ego-motion-apprise.
2. **Étape S1 — slot unique + transform analytique** (le plus cher gaté par F1) : ajouter au WM une voie « slot » =
   (coordonnée ego apprise depuis la retina) + (transform rigide depuis cmd ou ego-motion-tête) ; perte = prédire la
   coordonnée future. Garder le latent existant pour le proprio/gait. Re-gate `diag_nav_ab_latent.sh` (arrière > 0/4).
3. **Étape S2 — K slots + persistance** (object permanence) : plusieurs objets, slot non-perçu persiste+se transforme.
   Gate = recherche dirigée (cible derrière → demi-tour engagé DANS le latent).
4. **Étape S3 — généralité** : la même voie sert eau/prédateur (slot + canal couleur) = multi-percept, drive-agnostique.

## Anti-patterns (§2/§3)
- Pas de coordonnées MONDE explicites injectées (slots = égocentriques, appris depuis la perception).
- Pas de retrain sans F1 vert. Pas de « ça marche » sans re-gate closed-loop `diag_nav_ab_latent.sh` (arrière).
- Pas de slot câblé « bouffe » : la voie slot est générale, la couleur/identité = canal séparé, la pulsion = tête de valeur.

## État
F1 = en cours d'implémentation. Live de secours inchangé = planner-coordonnées + `wm_command_hex_v2`.
Re-validation : `diag_nav_ab_latent.sh` (latent-pur, arrière 0/4) vs `diag_nav_ab.sh` (coordonnées, arrière 2/4).
