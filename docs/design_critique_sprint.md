# Design — Critique appris du SPRINT (IC+TC, modulation d'intrusion) — chantier 2026-07-16

## Mission
Remplacer la règle-oracle du sprint (`SYLVAN_WP_ORACLE_SPRINT`, bouche-trou déclaré) par une
correction APPRISE du vécu qui module la pénalité verte de l'étage waypoint selon l'état
(drives, santé, douleur prédite). Forme LeCun `note = inné + correction` : l'analytique complet
reste le socle, la correction n'apprend QUE l'arbitrage du sprint (quand outrepasser la géométrie).

## À lire d'abord
- `docs/design_monde_v2_risque.md` (fin) — G-place licencié + critère juge pré-enregistré.
- `docs/design_critique_waypoint.md` — les 3 négatifs (v2 fenêtre-200, v3 poursuite+κ mesuré) :
  leurs leçons sont des CONTRAINTES, pas des options.
- `python/sylvan/control/waypoint_layer.py:397` (`decide()`), `:439` (règle oracle à remplacer),
  `:172` (`route_cost`), `:184` (`candidate_features`, canonicalisation miroir).

## Fait mesuré qui fonde le chantier
Monde v2 (P=0.5, regen=0.05), 24 vies ×2 seeds : géométrie **34 repas/11 morts** poolés ;
oracle-sprint (bloqué + santé>60 + énergie<50 + cible<3 m) **47/9** = Pareto-dominance sur les
2 seeds. La géométrie ne lit ni les drives ni la santé → la place de l'appris existe et est
chiffrée. Bruit d'instrument : ±5 repas par 24-total à seed identique → gates poolés uniquement.

## Essayé → résultat (ne pas répéter)
- Remplacement du scoreur par douleur-MC (v2, κ=100 deviné) : 7/1 vs 14/1 — troc évitement↔repas.
- Idem label-poursuite + κ=8 mesuré (v3) : 8/5 — aucun κ ne rejoint l'analytique. **Le déficit est
  la FORME** : notes MC par état = choix flottants (36 commits/21 aborts vs 57/14) ; l'analytique
  gagne par CONSISTANCE inter-replans. → jamais de remplacement, correction par-dessus.
- Acquis banké : la douleur EST apprenable (`waypoint_pain_v3`, AUC 0.894, gelé) — elle devient
  une ENTRÉE de la correction, plus jamais le scoreur.
- Corpus g24 (4×24 vies, monde v2, bras a=géométrie b=oracle) : 5 869 décisions, **0% ε** —
  les sprints vécus viennent du bras oracle, les refus du bras géométrie ; les contrefactuels
  hors-oracle (sprint blessé…) n'existent PAS → collecte ε obligatoire (leçon auto-confirmante).

## Design (D1 tranchée owner 2026-07-16 : modulation d'intrusion)
- **Déploiement** : `score(c) = leg1 + leg2 + (W − g(s,c))·intrusion(c)`, `g = W·σ(MLP)` ∈ [0, W].
  - g=0 ⇒ bit-identique à l'analytique (plancher de perf = bras géométrie).
  - La correction ne touche QUE les candidats à intrusion>0 et ne peut qu'ADOUCIR la pénalité
    (jamais l'aggraver, jamais la rendre négative) — le scope EST la licence de sprint.
  - Hystérésis pro-direct inchangée, APRÈS correction. Exclusif avec `SYLVAN_WP_PAIN_CRITIC` et
    `SYLVAN_WP_ORACLE_SPRINT` (erreur au démarrage si combinés).
  - ⚠ l'intrusion géométrique est recalculée proprement — ne pas réutiliser `intr_direct` du mode
    pain (sémantique piégée, wl:419).
- **Entrées (14-d)** : les 10 features canoniques du candidat (miroir déjà aboli par
  `candidate_features`) + énergie/100 + soif/100 + santé/100 + `pain_pred_v3(c)` (checkpoint
  `data/checkpoints/waypoint_pain_v3/pain_best.pt` GELÉ).
- **Cible d'apprentissage** : `p(s,c) = P(la traversée PAIE | s, c)` par BCE sur les décisions
  vécues dont le candidat choisi CROISE le vert (sprints oracle + traversées ε + directs
  d'hystérésis). Déploiement `g = W·p`. Aucune nouvelle constante (W est déjà l'IC).
- **Label « la traversée paie »** : `y = 1[U > 0]` avec `U = gain_repas − κ_data·dégâts_poursuite`
  (en pas de vie). `gain_repas` = remontée d'énergie OBSERVÉE / drain (donc l'état repu réduit le
  gain par plafonnement — la condition « affamé » de l'oracle émerge du label, pas d'un seuil).
  `κ_data` = médiane(pas-restants aux décisions)/100, re-mesuré sur corpus MONDE V2 en Phase 0.
  Fenêtre = POURSUITE (conventions v3 : changement de cible / consommation / mort / cap 600).
  - Variante pré-enregistrée (UNE seule) : plancher-mort `U = −κ_data·100` si mort pendant la
    poursuite (le linéaire sous-compte la mort à santé basse). Choix PINNÉ en fin de Phase 0 par
    ce critère : si >10 % des traversées labellisées meurent avec santé<50 à la décision → variante.
- **CV** : 4 plis PAR VIE (jamais par instant). Symétrie : héritée de la canonicalisation.
- **Corpus** : g24as1/as2/bs1/bs2 (jointure tick decisions↔BC pour drives/santé) + 2 collectes ε
  monde-v2 **seeds 3+4** (tranché owner — les seeds 1+2 restent la propriété du juge),
  `SYLVAN_WP_EXPLORE_EPS=0.15`, oracle OFF, 24 vies chacune, séquentiel. Le log de décision est
  ENRICHI (additif) avant collecte : `drives:[e,t,h]` + `intr:[...]` par candidat.

## Gates PRÉ-ENREGISTRÉS (écrits AVANT tout run/train — ordre cheaper-first)
0. **G0 (corpus, gratuit)** : ≥100 décisions-traversée labellisées ET ≥100 refus-bloqués tenus
   pour l'éval ; ET contraste directionnel : U̅(traversée) > U̅(refus) sur les buckets
   sains-affamés ET l'inverse sur les buckets blessés. Échec après collecte ε → STOP chantier
   (rien à apprendre dans ces données), négatif commité.
1. **G-rank** : sur paires tenues à états comparables (bloqué, buckets santé×énergie×dist),
   AUC(le scoreur corrigé ordonne la décision empiriquement meilleure) > **0.70**.
2. **G-res** : précision du choix (traverser/refuser) vs l'action empiriquement meilleure du
   bucket, sur décisions tenues : corrigé ≥ analytique seul + **10 pts**.
3. **G-consist (le gate que v2/v3 n'avaient pas)** : replay offline des séquences de décisions
   d'une même poursuite → taux de bascule du choix corrigé ≤ **1.2×** celui de l'analytique.
4. **Juge closed-loop (cher, gaté par 1-3)** : 2×24 vies seeds 1+2, monde v2, bras apprenant
   (`SYLVAN_WP_SPRINT_CRITIC`, oracle OFF) vs réfs MESURÉES (pas de re-run) :
   **PASS = repas poolés ≥ 42 (34+8) ET morts poolées ≤ 13 (11+2)**. Plafond connu : 47/9.
   **KILL précoce** : premier seed < géométrie−5 repas à 24 vies. Entre les deux → owner.
Budget dur : 1 entraînement + 1 seul re-train sur hypothèse NOUVELLE diagnostiquée sur trace ;
tout échec au-delà = négatif commité + STOP (CLAUDE.md §1).

## ⭐ PHASE 0 FAITE (2026-07-16) — G0 provisoire ✅, label PINNÉ
`diagnostics/diag_sprint_corpus.py` (jointure tick, poursuites v3 partagées via `pursuit_end`,
intrusion exacte reconstruite de costs−longueur) sur g24×4 :
- Volumes : **1530 traversées / 216 refus-bloqués** (≥100/≥100 ✓). Même les bras géométrie
  traversent via l'hystérésse (intrusions peu profondes) → contraste géométrique réel
  (profondeur q1/méd/q3 = 0.25/0.59/0.77 m) — la forme modulation a de quoi apprendre.
- Mesures : drain 0.0500/pas → **valeur repas = 799 pas** ; **κ_data v2 = 9.5 pas/dégât** ;
  traversées mortes à santé<50 = 39/1530 = **3 % < 10 %** → **label LINÉAIRE pinné**
  (variante plancher-mort NON retenue, per critère pré-enregistré).
- Direction sains-affamés : U̅cross **523** > U̅refuse **335** ✓ (le signal oracle est dans les données).
- Volet blessés : CONTREDIT (303 > 220) mais sur n_refuse=20 et traversées AUTO-SÉLECTIONNÉES
  (l'oracle ne sprinte jamais sous 60 ; les directs blessés observés sont peu profonds) → NA
  jusqu'à la collecte ε, comme pré-enregistré. Observation consignée (pas un bouton) : sous
  plancher-mort, sain-repu et blessé changent de signe — hypothèse de secours DIAGNOSTIQUÉE si
  G-rank échoue, à ne rouvrir que sur trace.

## ⭐ PHASE B FAITE + CORRECTION DU G0 À DÉCOUVERT (2026-07-16, décision owner)
Collecte ε livrée : spx3 792 déc (ε 15 %) + spx4 617 (ε 14 %), 24 vies chacune, monde v2, seeds
3+4. G0 complet : volumes ✓ (1939 traversées / 298 refus), direction sains-affamés ✓ (516>346),
**volet blessés CONTREDIT (317 > 267)** — diagnostic gratuit : le critère supposait une INVERSION
de signe au seuil h=60 de l'oracle (une constante-triche, pas une vérité du monde) ; les données
montrent un GRADIENT CONTINU (morts en traversée : 21 % à h<30 → 7 % au-dessus ; U̅ 271→338→400
par bande de santé) et le sous-bucket refus-blessés est structurellement vide (n=27 même avec ε).
Vérifié aussi : le label LINÉAIRE ne ment pas (149 traversées mortes → 0 U positif ; le
plancher-mort tariferait à −919 les morts MÉTABOLIQUES pendant poursuite → bruit) → pin conservé.
**Correction owner (précédent G-place, angle mort corrigé ouvertement)** : le volet
inversion-blessés est REMPLACÉ par un gate falsifiable aligné sur le signal réel —
**G-mono (pré-enregistré avant le train)** : sur les traversées, p̂ moyen STRICTEMENT croissant
par bande de santé [0,30)/[30,60)/[60,100] ET strictement décroissant par tercile de profondeur
d'intrusion. Le JUGE closed-loop reste INCHANGÉ (repas poolés ≥42 ET morts ≤13 — jamais déplacé).

## ⭐ NÉGATIF n°1 (2026-07-16) — BCE sur le SIGNE de U : gates échoués, cause DIAGNOSTIQUÉE
Train 1 (BCE, y=1[U>0], 1939 traversées) : G-rank 0.684 ✗ (plis 0.63/0.63/0.74/0.74) ;
G-res +3 pts ✗ ; **G-consist ✅ (6.5 % = 6.5 % — la forme modulation ne flotte PAS, le tueur de
v2/v3 est absent)** ; G-mono ✗ (profondeur inversée). Diagnostic sur trace (gratuit) :
1. **y==got à 97.6 %** — le gain (799 pas) écrase κ·dégâts (~250) → le SIGNE de U ≡ « repas
   obtenu » ; le signal risque/santé est JETÉ par le seuillage (AUC(santé)=0.510 sur y).
   La MAGNITUDE, elle, le porte : U̅|repas = 557/591/716 par bande de santé.
2. **Profondeur ⊥confondue avec proximité** (intr>0.7 → cible 2.1 m, 62 % payants ; <0.35 →
   3.3 m, 52 %) → le volet profondeur de G-mono (non conditionné) est invérifiable dans ce monde.
3. Plafond du label : l'énergie SEULE fait 0.66-0.79 selon pli — le gate 0.70 bute sur le label,
   pas sur la capacité (le modèle 14-d fait 0.684 ≈ niveau 1-feature).
Le checkpoint (gates_pass=False) reste sur disque, non branché.

## Critère de succès = le BUT
Le juge du §4 (repas ET morts, poolés, seeds du juge) — jamais un proxy offline. Offline-PASS ne
préjuge de rien (leçon v2/v3) ; les gates 0-3 ne servent qu'à ne pas payer un A/B perdu d'avance.
Si PASS : la règle-oracle MEURT (retirée du chemin vivant), la carte et la mémoire sont mises à
jour dans le même commit.

## Prochain pas — cheaper-first
Phase 0 (0 run, 0 train) : loaders .gz + liaison logs g24 + `diag_sprint_corpus.py` (jointure,
issues de poursuite, κ_data v2, valeur-repas mesurée, contrastes G0 sur le corpus EXISTANT) →
Phase A (hook + log enrichi + smoke bit-identité) → Phase B (collecte ε, smoke 3 vies avant
chaque 24) → G0 complet → Phase C (train + gates 1-3) → Phase D (juge) → Phase E (verdict).
