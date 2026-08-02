# PROMPT — Grand nettoyage + outil de visualisation (session neuve)

*Copier le bloc ci-dessous tel quel dans une session neuve, à la racine du dépôt.*

---

Salut. Ce soir est un **tournant** : on professionnalise le projet. Deux livrables, dans cet ordre :
**(A) un grand nettoyage**, **(B) un outil visuel complet**. Utilise plusieurs agents en parallèle.

## 0. Lis d'abord (dans cet ordre, rien d'autre)
1. `CLAUDE.md` — les 4 principes de travail. Ils prévalent sur tout ce prompt.
2. `ETAT_DES_LIEUX.md` — l'état courant (96 lignes, réécrit le 2026-08-02).
3. `tools/archi_hud/architecture.json` — la carte des modules. **Attention : 318 Ko, illisible.**
   La nettoyer fait partie du travail (voir A-3).

## 1. Où en est le projet (faits MESURÉS, à ne pas re-découvrir)
- **Le monde était INVIVABLE** jusqu'au 2026-08-02 : survivre exigeait 50 m de trajet par 1000 pas
  quand le corps en parcourt 47. Cause : la valeur nutritive n'était jamais exportée à l'eau
  (`restore_per_item` → seulement `SYLVAN_FOOD_ENERGY_PER`). **3ᵉ panne silencieuse de cette
  famille.** ⇒ **toute mesure comportementale antérieure est VIDE**, pas fausse : vide.
- Ce qui MARCHE, mesuré : le modèle du monde rêve juste (0,13 m à 50 pas) ; la perception pose sa
  lecture sur une **vraie proie 85 % du temps** ; le corps cinématique obéit.
- Facteurs limitants du forage, ablation dans le budget métabolique réel :
  **ralenti terminal −64,5 pts** · rayon de braquage −23,3 · **biais de visée −14,8** ·
  intermittence de la vue −0,0.
- **6 négatifs propres** bankés le 2026-08-02 (arbitrage, critique-de-rang, mouvement des objets,
  perception honnête, persistance de cible, sélection par valeur). **Ne pas les rouvrir sans une
  hypothèse NOUVELLE.** Ils sont dans `docs/design_*.md` et dans la carte.
- **Le vrai blocage aujourd'hui n'est pas scientifique, il est INSTRUMENTAL** : la survie est
  BIMODALE (29 vies sur 36 meurent vers 350 pas, 6 vont à 3000), donc **tous** les A/B sortent entre
  p = 0,13 et p = 0,88. Un gain réel de 20 % est invisible.

## 2. Règles NON NÉGOCIABLES (elles ont toutes coûté cher le 2026-08-02)
1. **Ne jamais supprimer un négatif banké ni une leçon.** Les docs de design contiennent les
   RÉFUTATIONS — c'est le plus précieux du dépôt. On les CONSOLIDE, on ne les jette pas.
2. **Une table plate = un test qui ne mesure rien.** Toujours balayer un paramètre.
3. **Vérifier qu'un bras expérimental a AGI** avant de lire son verdict — mesurer son MÉCANISME,
   pas sa bannière de chargement. ⚠️ Le serveur est tué par `kill -9`, ce qui **détruit stdout non
   vidé** : tout `print` de diagnostic doit être en `flush=True`.
4. **Corriger pour comparaisons multiples** (permutation) dès qu'on teste plusieurs signaux.
5. **Juger sur une métrique PAR TEMPS VÉCU**, jamais sur la survie brute (bimodale).
6. **Le simulateur n'est PAS déterministe** : deux runs identiques divergent. Jamais de comparaison
   de trajectoires appariées ; toujours graines + vies.
7. **Un KILL porte une MAGNITUDE, pas une direction.**
8. `git rm` jamais en masse : **lister d'abord**, faire valider, supprimer ensuite. Un
   `rm -rf '*cr*'` a déjà détruit tout le code de ce dépôt une fois.

---

# PARTIE A — LE GRAND NETTOYAGE

État mesuré au 2026-08-02 : **81 docs / 13 863 lignes** · **106 scripts** · **131 diagnostics** ·
**49 scripts python** · **79 modules python / 10 909 lignes** · **20 fichiers Godot / 7 400 lignes** ·
**105 checkpoints (211 Mo)** · **128 corpus (7 Go)** · **315 flags `SYLVAN_*` distincts**.

### A-1. Audit d'abord, suppression ensuite (agents en parallèle)
Lance un agent par domaine. Chacun produit un **inventaire classé** avant toute suppression :
`VIVANT` (servi par la config promue) · `OUTIL` (diagnostic réutilisable) · `ARCHIVE` (négatif banké,
à consolider dans un doc) · `MORT` (rien ne l'appelle, aucune valeur historique).

- **Agent DOCS** : les 81 `docs/*.md`. Beaucoup sont des chantiers clos. Objectif : **≤ 15 docs**,
  organisés en `docs/design/` (chantiers actifs), `docs/negatifs/` (UN doc consolidé par famille de
  réfutation, avec les chiffres), `docs/archive/`. Zéro perte de chiffre mesuré.
- **Agent HARNAIS** : `scripts/` (106) + `diagnostics/` (131). **C'est là qu'étaient TOUS les bugs
  du 2026-08-02** — un script qui `rm -rf` le corpus de référence, un contrôle affichant
  « DÉTERMINISTE » sur deux runs morts, une sonde câblée sur l'ancien monde, 5 gates cassés depuis
  un commit de nettoyage. Objectif : un harnais unique paramétré plutôt que 106 variantes.
- **Agent FLAGS** : les **315 flags**. Pour chacun : lu où, écrit où, valeur servie, DÉFAUT.
  Produire `docs/flags.md` (référence unique) et **supprimer les flags morts**. Signaler tout flag
  dont le défaut ne correspond plus à la config promue.
- **Agent CODE MORT** : `python/sylvan` + `godot/scripts`. Cibles connues : `command_planner.py`
  (1 388 lignes, branches superposées), `reward_manager.gd` (1 769 lignes, l'ère PPO est révolue),
  `sylvan_agent.gd` (1 280). Chercher aussi les **chemins morts** (une condition jamais vraie —
  il y en avait un le 2026-08-02 : un bloc `_slots0` qu'on croyait actif).
- **Agent DONNÉES** : 105 checkpoints / 128 corpus / 7 Go. Garder les checkpoints VIVANTS + ceux
  cités comme preuve ; proposer la suppression du reste **avec la liste**, ne rien supprimer sans
  validation owner.

### A-2. Chasse aux bugs (agent dédié)
Cherche spécifiquement la famille qui a coûté le plus : **un réglage donné à une ressource et pas à
l'autre** (l'eau n'héritait pas de la nourriture — 3 occurrences déjà trouvées). Puis : contrôles
qui valident l'échec, valeurs par défaut périmées, chemins de code inatteignables, `print` sans
`flush` dans un processus tué par `kill -9`.

### A-3. `architecture.json` (318 Ko — critique)
Elle est devenue illisible : les champs `etat_detail` font des milliers de mots empilés. La
restructurer : un état COURT par module + les preuves déplacées dans `docs/negatifs/`, avec un
schéma versionné et le validateur existant (`tools/archi_hud/validate_architecture.py`) étendu pour
**refuser un champ trop long**. La carte doit redevenir lisible d'un coup d'œil.

### A-4. Ce qu'on NE touche PAS
Le substrat qui marche : `command_wm.py`, `slot_head.py`, la rétine, le corps cinématique, les
checkpoints `wm_foret_v2_slot` / `drive_saliency_food` / `danger_saliency`, et `sylvan/world.py`
(source de vérité du monde). Refactoriser oui, changer le comportement NON — tout changement doit
être **bit-identique par défaut** et vérifié.

---

# PARTIE B — L'OUTIL VISUEL

Un outil web local (serveur Python + front léger ; pas de dépendance lourde) qui couvre TOUT le
spectre. Il doit servir **deux usages** : que l'owner VOIE, et que l'agent DEBUGGE.

### B-1. Vue « ŒIL » — ce que l'entité perçoit vraiment
- les **36 rayons** de la rétine en éventail, à leur vraie couleur et profondeur ;
- lesquels sont **retenus** par le slot, et le **barycentre** qui en sort ;
- la **vérité-terrain** superposée (proie réelle) + l'erreur de gisement en direct ;
- ⚠️ marquer visuellement les **bascules de cible** (saut > 2 m) : c'est le défaut mesuré qui coûte
  22,7 points de réussite et personne ne l'avait vu avant de le tracer.

### B-2. Vue « VIE » — la timeline d'un épisode
Jauges (énergie/soif/santé), repas, boissons, cause de mort, distance aux ressources, vitesse
demandée **contre** vitesse réelle (c'est ce qui a révélé le ralenti terminal à −64,5 pts).
Curseur temporel synchronisé avec la vue ŒIL.

### B-3. Vue « DÉCISION » — pourquoi elle a choisi ça
Les candidats évalués au replan, leur score, celui qui gagne, la cible retenue, les bascules
d'arbitrage. Et le **contrefactuel** quand il existe (`scripts/cf_fork_probe.sh` produit de vrais
contrefactuels : 21 commandes × conséquence réelle).

### B-4. Vue « A/B » — le comparateur, avec la statistique intégrée
Charger deux bras, afficher les métriques **par temps vécu** (jamais la survie brute seule),
lancer le **test de permutation**, appliquer la **correction pour comparaisons multiples**, et
afficher un **encart de contrôles** obligatoire : *le bras a-t-il agi ? la table est-elle plate ?
la métrique est-elle bimodale ? l'étiquette est-elle partielle ?*
⇒ Ces 4 contrôles ont attrapé, en une seule journée, un faux positif, un verdict vide et un faux
négatif. L'outil doit les rendre impossibles à oublier.

### B-5. Vue « ARCHI » — la carte vivante
Remplace/absorbe `voir_archi.sh`. Modules, état (pur/partiel/échafaudage/manquant), **dettes
déclarées**, dépendances, et surtout : **quels échafaudages sont ACTIFS dans le run affiché**
(bannière). Cliquer sur un module → ses preuves, ses négatifs, son ancre de code.

### B-6. Vue « CHANTIERS » — les gates et leurs verdicts
Chaque chantier avec ses gates **pré-inscrits**, leurs barres, leur verdict (passé/stop/kill), et la
date. Rendre visible d'un coup d'œil ce qui est **réfuté** (pour ne jamais le rouvrir sans
hypothèse nouvelle) et ce qui est **ouvert**.

### B-7. Vue « SANTÉ » — l'hygiène du dépôt
Flags servis contre défauts, checkpoints vivants contre orphelins, corpus et leur poids, processus
orphelins (`serve_planner_command`, `godot`), et le **contrôle de viabilité du monde**
(`diagnostics/diag_viabilite_monde.py`) affiché en permanence — un monde invivable rend toute
mesure vide, et ça a coûté des semaines.

### B-8. Exigences techniques
Lecture seule sur les données (aucune mutation), démarrage en une commande, fonctionne hors-ligne,
capable de rejouer un corpus enregistré **et** de suivre un run en direct. Prévoir un mode
« exporter une capture » pour illustrer un résultat dans un doc.

---

# LIVRABLES ET CRITÈRES D'ACCEPTATION
1. **Inventaires** des 5 agents, validés par l'owner **avant** toute suppression.
2. Nettoyage exécuté : docs ≤ 15, harnais unifié, flags documentés et élagués, code mort retiré,
   `architecture.json` lisible. **Chaque suppression dans un commit séparé et réversible.**
3. **Non-régression prouvée** : le gate closed-loop de référence donne les mêmes chiffres qu'avant
   nettoyage (consommation par 1000 pas vécus **3,60** ; survie moyenne **1358**, avec
   `SYLVAN_PLANNER_SPRINT=1`, 3 graines × 12 vies). Si ça diverge, le nettoyage a cassé quelque
   chose — trouver quoi avant de continuer.
4. L'outil lancé en une commande, couvrant les 7 vues.
5. `ETAT_DES_LIEUX.md` réécrit (remplacé, pas empilé) et `CLAUDE.md` mis à jour si des chemins
   changent.

**Priorité si le temps manque** : A-2 (bugs) > A-3 (carte) > B-4 (comparateur A/B avec ses
contrôles) > B-1 (œil) > le reste. Ce sont les quatre qui débloquent le plus de travail futur.
