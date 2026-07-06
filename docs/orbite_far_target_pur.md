# Orbite cible-lointaine : diagnostic complet et options pures

_2026-07-06. Pourquoi le forager n'atteint pas une ressource au-delà de ~5 m (il orbite), ce qu'on a
corrigé, et les voies PURES pour finir le travail. Le *pourquoi* vit ici ; l'*état* vit dans
`architecture.json` et `memory/sylvan-mode1-build.md`._

## 1. Mission
Que l'entité rejoigne une ressource lointaine (5-9 m), pas seulement les proches. C'est le vrai
plafond du monde ÉPARS (1+1) et la racine derrière l'échec du critique (fatalisme appris sur un
corpus sans succès de poursuite lointaine).

## 2. À lire d'abord
- `python/sylvan/control/planning/command_planner.py` : branche multi-ressource + `_survival_extension`.
- `python/sylvan/models/slot_head.py` : le readout du slot (le masque couleur dur, §3).
- Sondes : `diagnostics/diag_orbit_scoring.py`, `scripts/collect_curriculum_farfood.sh`.

## 3. Ce qui est MESURÉ (chaîne de sondes gratuites)
1. **L'agent orbite** une bouffe à 5-8 m : `food_d` reste ~5,6-6,1 m tout l'épisode, il tourne
   (`om=-0.6` constant), meurt de faim à côté. Budget énergie LARGE (atteindre 8 m ≈ 150 pas, il en
   a ~700) → **pas la physique**. Les 2 coûts (survie ET designed) orbitent → **pas le coût**.
2. **Le corps OBÉIT** (sign dyaw == sign om 78-90 %) → **pas le moteur**. C'est le PLANNER.
3. **Deux couches empilées** :
   - **(a) BUG SLOT [CORRIGÉ, PUR]** : le slot-bouffe lisait la position de l'EAU (1,1 m au lieu de
     6,0 m) quand bouffe-loin + eau-proche. Cause = masque couleur MOU (`log(...+1e-8)` = −18, pas
     −∞) : un rayon BLEU proche battait un rayon ROUGE loin via le prior −4/m·dist. Fix = masque DUR
     (logit −1e9 si mauvaise couleur). Non-rég dense = **améliore** (2465→2835). Toggle
     `SYLVAN_SLOT_HARD_MASK`.
   - **(b) SCORING PLANNER far-target [NON RÉSOLU]** : slot corrigé, l'agent orbite ENCORE. La cible
     >5 m est au-delà de l'horizon 80≈5 m → le score survie est PLAT (les candidats tournants ne
     translatent jamais dans le rêve, virage tue l'avance) → argmax pique ω MAX → pivote sans avancer
     → orbite. **Le mur « tourner-en-avançant » connu** (cf `sylvan-keystone-3b-geometric-wall`).

## 4. Essayé → résultat (négatifs INFORMATIFS)
- **Échafaudage steering-proportionnel** (`SYLVAN_PLANNER_FAR_ALIGN`) → RETIRÉ : se gâtait sur la
  distance CRUE >4 m, qui ne se déclenchait pas tant que le slot croyait 2 m. À re-tenter maintenant
  que le slot lit juste.
- **Option 1 = horizon plus long** (80/140/200) → RÉFUTÉ : tous orbitent (0/3 atteint). Allonger
  l'horizon ne change pas le CHOIX (ω max), car le score reste plat quel que soit l'horizon → le
  problème n'est pas « voir plus loin » mais « le score ne valorise pas l'alignement/progrès ».

## 5. Le point-clé conceptuel (receding-horizon)
On n'a PAS besoin d'un horizon qui atteint la cible. La replanification glissante (tous les 10 pas)
**accumule le progrès** : aligner puis avancer, replan après replan, franchit une distance > horizon.
L'orbite CASSE cette accumulation (score plat → pas de cap consistant). Le fix = restaurer un cap
consistant vers la cible hors-horizon.

## 6. Les options PURES (survey + faisabilité)
| Option | Idée | Pureté | Faisabilité |
|---|---|---|---|
| Horizon plus long | rêve assez long pour voir la cible | pure | **RÉFUTÉ** (§4) |
| **Critique = valeur terminale** | V(état) appris valorise « finir aligné+en approche vers la cible loin » | **pure (LeCun)** | échoué à froid (corpus sans succès loin) → besoin d'amorçage |
| H-JEPA hiérarchie | niveau haut planifie « aller à la zone » en 1 action abstraite | pure (blueprint) | énorme (recherche ouverte) ; overkill pour 6 m |
| Curiosité + WM incertain | explorer → générer les succès loin par hasard | pure | prérequis = WM stochastique (déterministe aujourd'hui) |
| Candidats pivot | ajouter vx≈0/ω fort (tourner sur place) | pure (élargit la recherche) | nécessaire, PAS suffisant seul (score reste plat) |

## 7. La voie recommandée (pure en bout de course)
Toutes les routes pures convergent vers **une valeur terminale apprise (le critique)** : seul moyen
d'obtenir le signal au-delà de l'horizon sans le coder en dur. Son blocage = démarrage-à-froid
(l'orbite empêche les données). Réconciliation avec « pas de mauvaise règle codée » :
**l'échafaudage codé-main est un AMORCEUR DE DONNÉES retirable, pas une règle de runtime.**
1. Hint de cap TEMPORAIRE (align-puis-fonce) + candidats pivot → l'agent complète des poursuites
   lointaines → génère les succès manquants (gate : far-food atteint ≥ 60 %).
2. Entraîner le critique sur ce corpus enrichi (curriculum + vécu) → il valorise « aligné vers loin ».
3. Le planner utilise le critique comme valeur terminale du rêve court → préfère « aligner+avancer ».
4. **RETIRER le hint** → la boucle finale est 100 % pure (critique appris fournit le signal).

## 8. Critère de succès = le BUT
- Gate intermédiaire : far-food (`collect_curriculum_farfood.sh TARGET=food`) atteint ≥ 60 %.
- Gate final : monde ÉPARS 1+1 survie médiane > 1800 (au-dessus du plancher ~1600), le hint RETIRÉ.
- Non-régression dense (5+5) ≥ bande record (~2735-2835) à chaque étape.

## 9. Échafaudage de cap IMPLÉMENTÉ + MESURÉ (2026-07-06) — ⚠️ CONCLUSION « mur substrat » SUPERSEDED PAR §10
> ⚠️ La conclusion « mur de substrat / recette bloquée » ci-dessous était un **ARTEFACT D'UN TEST RIGGÉ**
> (énergie curriculum 35 → portée physique ~3 m → bouffe 5-8 m inatteignable même parfaite). Voir **§10** :
> à budget ÉQUITABLE l'échafaudage MARCHE (mange la bouffe lointaine). Gardé tel quel pour la traçabilité.
Étape 1 exécutée. Scaffold `SYLVAN_PLANNER_FAR_ALIGN` (RÉ-implémenté propre : récompense la trajectoire
RÊVÉE qui pointe la ressource urgente = `align_f/align_w`, l'OUTCOME, PAS l'ω brut ≠ le hack retiré) +
candidats `SYLVAN_PLANNER_PIVOT` (virage serré in-band ; vrai vx≈0 = HORS bande WM 0.55-0.75). Flags OFF
par défaut → zéro régression. Instrumentation : `diag_orbit_scoring.py` (vx choisi, `mindf_best`,
`dfend_best`, `SYLVAN_DIAG_HORIZON`).

**Chaîne de sondes (gratuites sauf 1 A/B closed-loop) :**
- **Free A/B (4 combos)** : FAR_ALIGN déplace le classement vers l'alignement (corr(ω,food) 0.51-0.75 →
  0.81-0.95 ; corr(score,-min_df) 0.02-0.50 → 0.43-0.67). **PIVOT seul INERTE** (confirme §6 : pas suffisant
  seul). Le vrai levier = le terme de cap, pas les candidats.
- **Closed-loop far-food (OFF vs FA, 16 ep, gain=60)** : **0/16 les DEUX**, survie méd 560. MAIS **causal** :
  OFF food_d plat 5.7→5.4 (orbite) ; FA food_d **descend** 5.7→4.8. Le scaffold marche, mais **trop lent**
  (progrès net réel ~0.003 m/pas, spirale) → meurt avant d'arriver (budget ~560 pas). Fuite : ~40 % des pas
  en branche designed (`plan_multi`, bouffe hors-vue) où le scaffold ne s'applique pas.
- **Free horizon-sweep sous FA (80/120/160/200)** : `mindf_best` (approche du candidat GAGNANT) — à 90°
  (cible latérale) : 4.97→4.98→4.98→4.92 sur 200 pas = **la cible latérale ne s'approche PAS dans le rêve**,
  quel que soit l'horizon. À 30° : ~0.85 m puis plateau. **Horizon plus long RE-RÉFUTÉ** avec le mécanisme
  épinglé (le rêve near-sighted sature).
- **Free dream-vs-vérité** : rêve straight ~0.0046 m/pas vs vérité ~1e-2 m/pas (`command_wm.py:29`) =
  sous-prédit ~2× (modeste, échelle ~bonne). PAS un bug d'échelle → limite de FIDÉLITÉ, pas d'unité.

**RACINE (quantifiée, remplace « scoring plat » vague §3b)** : (a) rêve WM **near-sighted** (~0.4 m
translatés sur horizon 80) + **acquisition LATÉRALE qui sature** (mur géométrique 3b mesuré à neuf) → aucun
candidat ne discrimine l'approche d'une cible à 5 m ; (b) le cap récompense le cos-bearing **MOYEN** → spirale
de tracking, pas « aligne-puis-COMMIT-droit » ; (c) MPC mono-plan ne peut pas récompenser une stratégie qui
ne paie que sur PLUSIEURS replans. Le corps réel POURRAIT couvrir 5 m en ~500 pas droits (0.01 m/pas × budget),
mais il **ne commit jamais un cruise droit soutenu** (fwd médian 0.09, spin 33 %).

**Verdict** : gate step-1 (far-food ≥ 60 %) **inatteignable avec ce WM+planner**. La recette
échafaudage→critique est bloquée à l'étape 1 par une **limite de SUBSTRAT** (fidélité du rêve pour
l'acquisition latérale/lointaine + incapacité du MPC mono-plan à committer un beeline), PAS par le critique
ni par un cap manquant. Négatif INFORMATIF (règle §1 : STOP + escalade ; §2 : gate gardé honnête, pas relâché).

**Escalade — options PURES de repli (owner tranche, gros chantier)** :
1. **Recollecte WM ciblée** (dé-near-sight le rêge + représenter l'acquisition-par-virage : babbling avec des
   manœuvres tourne-vers-cible, horizon effectif plus grand). Attaque la racine mesurée. LÉGITIME (§3 :
   dynamique/portée, PAS forcer une ressource dans le latent). Medium-gros.
2. **H-JEPA** : niveau haut planifie « aller à la zone » en 1 action abstraite (le beeline devient 1 primitive,
   plus de mono-plan near-sighted). Gros / recherche ouverte.
3. **Curiosité + WM stochastique** : explorer pour générer les succès loin. Prérequis = WM stochastique
   (déterministe aujourd'hui). Gros.
Le scaffold `FAR_ALIGN`/`PIVOT` reste EN PLACE (flaggé OFF, déclaré, retirable) — réutilisable dès qu'un WM
moins near-sighted rend l'amorçage viable.

## 10. CORRECTION (2026-07-06, Gate 1 gratuit d'Option A) — le §9 était un test RIGGÉ ; l'échafaudage MARCHE
Avant tout retrain WM (Option A), un diag GRATUIT sur les données WM existantes (vérité-terrain `torso`,
`wm_dataset.py`) a **renversé le §9** :
- **Le rêve WM est FIDÈLE** : corps réel **0.0043 m/pas** en ligne droite (buffer `wm_hex_v2`), rêve 0.0046 —
  ils COÏNCIDENT. Le « rêve myope-par-erreur » du §9 était faux. Les virages soutenus existent (runs jusqu'à
  293 pas). Ce n'est PAS un manque de couverture ni une infidélité.
- **Le corps est juste LENT (~0.0043 m/pas) et le test far-food était RIGGÉ** : `INIT_ENERGY=35` → ~700 pas →
  portée ~3 m → bouffe à 5-8 m **physiquement inatteignable même par un agent parfait**. Le 0/16 du §9
  n'accusait PAS le WM. (Le monde ÉPARS réel utilise énergie ~80 → ~1600 pas → portée ~7 m.)
- **Re-test à budget ÉQUITABLE** (`INIT_ENERGY` paramétrable, défaut 35 inchangé) : à énergie 80 / bouffe
  5-6 m, FA transforme l'orbite en **approche quasi-complète** (5.5→1.24 m, meurt d'énergie EN approche, pas
  en orbite). À énergie 90 / bouffe 5 m : **FA MANGE 3/10** (min food_d 1.02-1.06 m ; OFF orbite toujours).
- **Périmètre honnête** : l'échafaudage étend la portée de ~4 m (dense) à ~5 m (mange), bord mou ~6 m. Il ne
  résout PAS 7-8 m dans le budget énergie (corps lent). 30 % < 60 % = **efficacité/consistance** (approches qui
  spiralent → meurent courtes), PAS un mur.

**Verdict corrigé** : la recette échafaudage→critique est **DÉBLOQUÉE** (on génère de vrais positifs far-food).
**Option A (retrain WM) N'EST PAS nécessaire** — le WM est fidèle. Prochaines pistes (cheap, pas de retrain WM) :
(a) améliorer l'efficacité d'approche (moins de spirale : cap END-align vs MEAN, ou tuning `align_gain`/
`heading_far_gate`) pour monter 30→60 % ; (b) A/B monde ÉPARS 1+1 réel (OFF vs FA) = le vrai but (survie méd
>1800) ; (c) collecter le corpus (curriculum food+water à budget équitable + vécu) → entraîner le critique →
brancher → RETIRER l'échafaudage. Outils : `collect_curriculum_farfood.sh` (params `INIT_ENERGY/FAR_MIN/FAR_SPAWN`).
