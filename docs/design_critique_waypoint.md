# Design — le critique note les WAYPOINTS (chantier post-G1, 2026-07-16)

## Mission
Remplacer progressivement le scoreur codé-main de l'étage waypoint par une valeur APPRISE du vécu
(forme LeCun `note = inné + λ·correction`). C'est la réhabilitation du critique : ~10 options très
dissemblables (écarts de centaines de pas) au lieu de 33 arcs à 1e-5 — l'étage où un réseau PEUT
discriminer (HIQL fig.8, `docs/recherche_hjepa_waypoint.md` §3).

## À lire d'abord
- `docs/recherche_hjepa_waypoint.md` §6 (« après G1 ») — la licence du chantier.
- `python/scripts/train_survival_critic.py:219` (`residual_labels`) + `:257` (gate CV-4 +0.10) — réutilisés.
- `python/sylvan/control/waypoint_layer.py` — l'étage v2 gaté (G1 PASS 14 repas / 1 mort).

## Fait mesuré qui fonde le design
Corpus déterministe = boucle auto-confirmante (mesuré 2026-07-08 : 18.6 % d'approches alignées →
le critique ne peut pas apprendre ce qu'il ne voit jamais). → L'exploration vit À L'ÉTAGE WAYPOINT
(varier les candidats commités ; leçon Director : le bonus au manager, jamais au worker), collecte
seulement, déploiement déterministe.

## Design
- **Objet noté** : la paire (état, candidat) au moment de la DÉCISION — un Q, pas un V d'état.
- **Label** : pas RÉELLEMENT vécus après la décision (mêmes conventions que `load_lived` : coupe aux
  respawns, vies censurées exclues). Monnaie = pas.
- **Forme IC+TC** : label résiduel = (vécu − `innate_steps(état)`)/H — l'inné d'état est aveugle au
  candidat, donc TOUT l'effet du choix de route (danger traversé, détour payé) vit dans la correction,
  qui seule voit le candidat. Au déploiement : `score(c) = score_analytique_route(c)_en_pas + λ·H·Q(s,c)`
  (le scoreur main reste l'IC de l'étage tant que l'appris n'a pas gagné le droit de le remplacer).
- **Symétrie miroir PAR CONSTRUCTION** (leçon token |sin| : une symétrie connue s'impose, ne se fitte
  pas) : canonicalisation — si wp_x < 0, miroir de TOUTES les x (wp, cible, verts). Géométrie relative
  wp↔cible↔verts préservée, côté aboli.
- **Features candidat (post-canon)** : dist/sin/cos du wp, dist/sin(signé relatif)/cos de la cible,
  longueur totale, **d_vert_leg1, d_vert_leg2 = distance brute du vert perçu le plus proche à chaque
  segment, SANS marge** (les constantes main 1.0/1.4 SORTENT des features — le critique apprend la
  distance létale de ses morts), is_direct. + les 2 tokens drives existants (`token(e,food)`,
  `token(t,water)`).
- **Exploration** : `SYLVAN_WP_EXPLORE_EPS` (défaut 0 = OFF) — par décision, avec prob ε≈0.15, commettre
  un candidat UNIFORME (y compris les mauvais : le critique doit voir « à travers le vert = mort »).
  Un leg dure 100+ ticks → ~1 leg exploratoire sur 5, vies viables, contraste réel.
- **Log décision** : `SYLVAN_WP_LOG=dir` → jsonl par décision {tick, cible, drives, features par
  candidat, coûts analytiques, choisi, explore}. Issue jointe par tick au flux BC.

## Gates PRÉ-ENREGISTRÉS (écrits avant toute collecte/entraînement — on ne déplace pas les poteaux)
1. **G-gap (licence, gratuit post-collecte)** : les issues divergent-elles selon le choix ? Critère :
   écart médian de survie-après-décision entre legs exploratoires et legs argmin, à états comparables,
   **> 100 pas** ; ET morts-danger surreprésentées dans les choix à-travers-vert. Écarts ≈ 0 → **KILL
   du chantier** (rien à apprendre à cet étage non plus), à commiter comme négatif.
2. **G-res** : gate résiduel INCHANGÉ — `inné + correction` > `inné seul` de **+0.10 R²** (CV 4 plis
   par VIE, jamais par instant).
3. **G-rank** : sur paires exploratoires tenues (même état, choix différents), Q ordonne les issues
   avec **AUC > 0.65** (c'est le gate décisionnel : prédire ne suffit pas, cf 2026-07-08).
4. **A/B closed-loop** (SEULEMENT si 2 ET 3) : analytique vs analytique+λQ, 12 vies seed 1, monde
   danger : **repas > 10 ET morts ≤ 2 ET ≥ bras analytique (14/1)**. Échec = on garde l'analytique,
   négatif commité.

## Étapes (le cher gaté derrière le gratuit)
Explo+log (code, smoke offline) → collecte ~24 vies (2 seeds, séquentiel) → G-gap → entraînement
(minutes, CPU) + G-res/G-rank → A/B. Chaque étape s'arrête si son gate échoue.
