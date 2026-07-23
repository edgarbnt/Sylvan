GATE DU CRITIQUE SUR LE VECU — resultat 2026-07-23

CE QUI A CHANGE depuis le gate RETRACTE le matin meme :
 1. le VECU reel (40 vies collectees sur le monde gele bosquets_v2, replan=60) au lieu d un
    simulateur maison dont la fidelite n avait jamais ete verifiee ;
 2. la cible compte des EVENEMENTS (repas observes) au lieu d etre derivee d un etat final —
    le selfcheck VERIFIE que manger tot et manger tard comptent pareil (l artefact retracte) ;
 3. un CONTROLE DE COHERENCE BLOQUANT ouvre le gate : « le cout qui fait VIVRE l entite est-il
    correle a ma cible ? ». C est le test qui manquait et qui avait laisse passer un faux positif.

NOTE D INFRASTRUCTURE : en mono-pulsion la branche plan_wm_slot ne loggue QUE target+reason
(verifie) — les coordonnees ne sont PAS dans le corpus. La geometrie est donc lue a la SOURCE,
dans la retine (144 floats/tick), avec les angles du FOV reellement servi (cone 120).

--- RESULTAT (40 vies, 1440 instants de replan, delta=600, split PAR VIE) ---
CONTROLE DE COHERENCE : corr(inne, cible) = +0,502 -> GATE OUVERT (la cible mesure du reel).

  predicteur              R2 (vies jamais vues)
  inne (recalibre)                     +0,234
  tete apprise (ridge)                 +0,327
  gain                                 +0,093   (barre +0,05)

Direction POSITIVE sur 4/4 plis : +0,196 / +0,056 / +0,108 / +0,013.
VERDICT : PASS.

⚠️ TROIS RESERVES, a lire avec le PASS :
 1. **Le gain est FRAGILE** : +0,013 sur le pli 3, soit un quart de la barre. Positif partout,
    mais un pli de plus aurait pu le faire basculer.
 2. **L ABLATION ne trouve AUCUNE feature porteuse** : retirer la faim coute 0,022, le cap 0,016,
    la visibilite 0,029 — et retirer la DISTANCE AMELIORE le score (-0,021). Le gain ne vient donc
    pas d une information identifiable ; c est le signe d un modele qui exploite des combinaisons
    faibles, pas d une comprehension. A surveiller : c est typiquement ce qui ne transfere pas.
 3. **Ce gate ne juge PAS un classement de candidats.** Le vecu n observe qu UN candidat par etat.
    Il dit que la cible VAUT LA PEINE, pas qu une tete saura ranger 117 options — la question qui
    decide vraiment de l utilite d un critique dans un planner MPC.

⇒ Le chantier critique est ROUVERT, sur base saine cette fois. Prochain pas : entrainer une vraie
tete (capacite > ridge) sur cette cible et ce corpus, et surtout construire le juge de CLASSEMENT
qui manque — probablement en instrumentant le serveur pour logger le score par candidat, plutot
qu en fabriquant des contrefactuels (ce qui a deja coute un faux positif).

--- LE JUGE DE CLASSEMENT NE PEUT PAS ETRE HORS-LIGNE (2026-07-23) ---

Avant de batir un juge de classement (« la tete met-elle le meilleur candidat en tete ? »), test
de fidelite du simulateur hors-ligne, cheap et decisif : le piloter par le VRAI cout analytique
(argmax -min_dist+heading sur 117 candidats, replan=60) et comparer sa performance a la REALITE.

  simulateur + vrai cout : survie mediane 3000, 100 % pleins, 8,00 repas/vie
  REALITE (vecu)         : survie mediane 2900,  50 % pleins, 1,40 repas/vie

Le simulateur est **6x trop facile**. Il connait toutes les positions et tous les stocks, et
navigue parfaitement ; l entite reelle ne voit qu a travers son cone 120, perd les bosquets
hors-champ, et son slot localise avec erreur. Aucun classement calcule dans ce monde ne vaut pour
le monde reel.

⇒ **NEGATIF STRUCTUREL** : le juge de classement (la seule mesure qui reponde a la vraie question
d un critique dans un planner MPC — le RANG des candidats) exige des contrefactuels DANS GODOT
(rejouer un etat, forcer un candidat, observer la suite). Il ne peut pas se construire hors-ligne.
C est aussi l explication profonde de l echec du gate de ce matin : pas seulement le relais glouton,
mais un monde hors-ligne entierement trop clement.

⇒ CE QUI RESTE VALIDE : le gate sur le VECU (corpus reel, PASS +0,093) n est PAS touche — il
n utilise aucun simulateur, seulement des outcomes vecus. Il dit que la cible vaut la peine ; il
ne dit pas qu une tete saura ranger. Ces deux questions sont desormais clairement separees.

DECISION OWNER REQUISE : le juge de classement demande une infra Godot (save/restore d etat +
commande forcee) — un vrai build, non chiffre ici, peut-etre pas faisable sans toucher au moteur
(a scoper). A trancher avant de s y engager.

--- JUGE DE CONTREFACTUELS : HOOK OK, mais REJEU NON DETERMINISTE (2026-07-23) ---

Le hook SYLVAN_CF_TICK/CF_CMD fonctionne (commit fbf9904, override d une decision au tick k verifie).
Sonde de validation a un etat (scripts/cf_rank_probe.sh, tick 600, 21 candidats + 2 controles) :

CONTROLE 1 (determinisme) ECHOUE : deux runs SANS contrefactuel, config identique, donnent
2 repas (survie 3000) contre 0 repas (survie 2000). Le rejeu n est PAS reproductible.
CONTROLE 2 : variation presente (repas 0 a 2 entre candidats) MAIS non interpretable, car une
partie est du bruit de rejeu, pas l effet du candidat.

⇒ VERDICT : le juge de contrefactuels est BLOQUE tant que le rejeu n est pas deterministe. C est
exactement ce que le scoping avait signale : main.gd:64 `randomize()` reseed le RNG GLOBAL depuis
l entropie a chaque lancement. En cinematique il ne touche QUE gait_phase (cosmetique) pour la
MOTRICITE — mais une autre piece (memoire spatiale ? planner ?) introduit une source non-seedee
entre deux runs identiques, assez pour changer 0->2 repas.

PROCHAIN PAS avant tout juge de rang : rendre le rejeu deterministe. Pistes a verifier :
(1) retirer/seeder `randomize()` (main.gd:64) ; (2) seeder le RNG cote serveur planner + memoire
spatiale. Puis re-passer le CONTROLE 1 (deux runs identiques -> repas identiques) AVANT de mesurer
un seul candidat. NEGATIF BANKE : ne pas lire le controle 2 tant que le controle 1 ne passe pas.

--- DETERMINISME PAYE (2026-07-23) : trois causes, trois corrections ---
Deux runs meme seed etaient NON reproductibles (0 vs 2 repas, commande divergente des le step 0).
Traque : la COMMANDE diverge au step 0 (pas l energie) -> source cote planner, pas Godot physique.
Trois causes empilees, corrigees dans l ordre :
1. RNG GLOBAL de Godot : `gait_phase = randf()` (main.gd:536) alimente la proprio (sin/cos, dims
   1137-1138) -> WM voit une entree differente. FIX : main.gd:64 seede le RNG global depuis
   SYLVAN_SEED au lieu de randomize() ; sans SYLVAN_SEED, randomize() conserve (non-regression).
2. TORCH MULTI-THREAD : reductions flottantes non-associatives -> argmax bascule sur candidats
   quasi ex-aequo. FIX : SYLVAN_PLANNER_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1.
3. SERVEUR PARTAGE STATEFUL : la memoire spatiale (MultiSlotMemory) garde son belief d un run a
   l autre. FIX : un serveur FRAIS par run.
Les TROIS sont necessaires. Verifie : serveur frais + seed + mono-thread => 201/201 pas
bit-identiques entre deux runs. Le juge de contrefactuels est DEBLOQUE.

--- JUGE EN FENÊTRE AUX FORKS (2026-07-23) : NÉGATIF INATTAQUABLE ---
Déterminisme PAYÉ (serveur frais + seed + mono-thread) : deux runs sans CF -> 1=1 repas dans la
fenêtre. Le juge mesure enfin la vraie chose.

Seed 3, fork k=1560 (repas connu à 1680), fenêtre [1560,2160]. Les 21 candidats donnent TOUS
**1 repas** — y compris le demi-tour brutal (vx=0.75, om=-0.6). AUCUNE variation.

Contrôle de qualité du fork (run déterministe) : à k=1560 la bouffe est à **2,1 m**, l'agent
APPROCHE, affamé (énergie 23). C'est un VRAI fork, pas une bouchée déjà acquise. Forcer le pire
choix pendant 60 ticks ne coûte pourtant pas le repas : le planner replanifie à 1620 et rattrape,
toujours dans la fenêtre.

⇒ VERDICT TRIPLEMENT CONFIRMÉ (diag_consequence 1,9 % ; juge vie-entière 0 variation ; juge
fenêtre-fork 0 variation à 2,1 m) : **AUCUNE DÉCISION UNIQUE NE COMPTE dans ce monde.** Le planner
(replan 60) récupère de n'importe quel choix. Un critique qui améliore le classement d'UNE décision
ne peut donc rien apporter. Sa seule valeur possible = le compoundage de micro-améliorations sur
TOUTES les décisions, que le gate sur vécu a estimé à +0,093 R² FRAGILE, sans feature porteuse.

⇒ **CHANTIER CRITIQUE FERMÉ POUR CE MONDE.** Pas parce que « l'appris ne marche pas » (la mémoire
apprise a payé +2,17 sous cône) mais parce que le MONDE ne rend aucune décision conséquente. Même
cause-racine que le WM-prédicteur inutile et la perception active affamée : le corps/monde est trop
RÉCUPÉRABLE. Le seul levier qui débloquerait les trois ensemble = l'IRRÉVERSIBILITÉ (momentum,
effet différé, ou perte permanente d'une ressource sur mauvais choix — occlusion+mémoire en est une
forme). C'est un changement de CORPS/MONDE, décision owner.

NÉGATIF BANKÉ : ne pas rouvrir le critique appris tant que les décisions ne sont pas rendues
conséquentes. L'instrument déterministe (hook CF + juge fenêtre) est PRÊT pour re-juger dès qu'elles
le seront.
