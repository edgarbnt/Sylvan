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
