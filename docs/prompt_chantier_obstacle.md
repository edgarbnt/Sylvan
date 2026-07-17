# Prompt, chantier « Canal OBSTACLE / affordance » (premiere nature de consequence NON-homeostatique)

> Copier-coller comme premier message d'une session fraiche. Ecrit le 2026-07-17, a la cloture du
> chantier attribution-de-credit (baie-buisson, G0-G3 PASS, perception 100 % apprise au sens
> connaissance-du-monde). La pre-inscription et le diag gratuit ne sont PAS encore faits : cette
> session PRE-INSCRIT puis fait le DIAG GRATUIT DECISIF avant tout build.

---

Sylvan, chantier : ajouter le PREMIER canal de consequence NON-homeostatique, les affordances
PHYSIQUES (un obstacle BLOQUE le mouvement / l'herbe haute est TRAVERSABLE), apprises de l'ERREUR DE
PREDICTION du deplacement. C'est la generalisation de « apparence -> consequence apprise » (chantier
baie-buisson clos) a un NOUVEAU canal (le mouvement, pas les drives). Et ca cree la TOPOLOGIE qui
rendra chercher + memoire enfin decisifs (aujourd'hui triviaux en arene ouverte). Venv
`env_pytorch_3.12` CPU, `PYTHONPATH=python`, depuis la racine. Re-checker : orphelins
(`pgrep -xc godot`), git log, disque.

LIRE D'ABORD : 1) `docs/research_appearance_consequence.md` AXE 3 (la synthese de recherche : « absorber
l'obstacle DANS le WM » est LE PLUS MINCE en sources, la litterature preferant un PREDICTEUR
D'AFFORDANCE SEPARE, a trancher par le diag) ; 2) `docs/design_attribution_credit.md` (le patron du
chantier precedent : pre-inscription, gates cheaper-first, negatif commite) ; 3) `memory/MEMORY.md` +
fin de `memory/sylvan-mode1-build.md`.

## Etat a la reprise (rien a re-decouvrir)
- Perception = 100 % apprise au sens connaissance-du-monde. WM vivant = `wm_objcentric_kin_typed`
  (requetes mesurees, voit 100 % de l'eau). Le WM credit-type (classe neutre) = artefact-PREUVE, NON
  promu (regression eau mesuree : marge water 0.976 ne voit que 53 % de l'eau reelle, espace couleur
  encombre). Ne PAS re-litiger.
- Le corps CINEMATIQUE glisse par TELEPORTATION (`sylvan_agent.gd` `_kinematic_step`,
  `PhysicsServer3D.body_set_state`) : il NE respecte PAS les solides aujourd'hui. Un obstacle bloquant
  exige une addition PHYSIQUE (le corps s'arrete contre un solide) = couche CORPS/MONDE, jamais le
  cerveau (a verifier/coder dans le corps, pas dans le planner).
- Le buisson (chantier precedent) a etabli le patron d'ingredient-monde opt-in defaut OFF
  bit-identique (`food_manager.gd`) : le reprendre pour l'obstacle.

## Le principe (a garder honnete)
- MONDE/physique/sens = donnes legitimes (Principe n3). La REACTION (contourner) = APPRISE ou
  EMERGENTE, JAMAIS un `if gris then evite` ni un cout-obstacle code-main dans la boucle de decision.
- Cout par CANAL (un sens, fixe au corps), pas par objet : le canal-mouvement paye UNE fois, toute la
  famille des affordances physiques (bloque / traverse / ralentit) arrive DECOUVERTE, zero code par objet.
- Critere de purete : ca survit a un changement de monde (l'apparence de l'obstacle change -> l'entite
  re-apprend qu'il bloque, elle ne presume rien).

## Ordre cheaper-first (Principe n1 : diag GRATUIT decisif AVANT tout train)
1. **PRE-INSCRIRE** : `docs/design_obstacle_affordance.md` (mission, mecanisme, ingredient-monde, gates
   falsifiables ECRITS AVANT, critere de succes = le BUT en vies). Rien ne tourne avant ce doc.
2. **DIAG GRATUIT (gate le cher)** : le WM gele conditionne-t-il DEJA son deplacement predit sur la
   PERCEPTION (la retine), ou seulement sur la commande (vx, omega) ? Sonde offline : injecter un
   obstacle synthetique devant dans la retine, mesurer si le deplacement predit par le WM CHUTE.
   - Si OUI (le WM PEUT representer « bloque ») -> voie « absorber dans le WM » viable (re-collecte +
     fine-tune WM avec obstacles ; le planner evite par rollout, zero cout code). COUTEUX (cycle WM).
   - Si NON (deplacement = commande seule) -> l'absorption est impossible sans changer l'archi ->
     PREDICTEUR D'AFFORDANCE SEPARE (la voie preferee par la deep-research), integre au cout/rollout.
   Ce diag TRANCHE la voie et evite un cycle WM devine (la lecon anti-boucle du projet).
3. **BUILD** (selon le diag) : Godot obstacle + soit re-collecte/retrain WM, soit predicteur separe.
4. **VERIF EN VIES** : l'entite CONTOURNE l'obstacle (vs fonce dedans), sans cout code. Gate poole
   pre-enregistre, contre une baseline (OFF = pas d'obstacle, ou obstacle-aveugle).

## Pieces a construire (apres le diag qui tranche la voie)
- **Godot** : region/mur COLORE bloquant, opt-in `SYLVAN_OBSTACLE_*` defaut OFF bit-identique (patron
  du buisson) ; + le corps cinematique RESPECTE les solides (shapecast/move_and_collide avant le glide
  dans `_kinematic_step`) = physique du corps, pas du cerveau. `main.gd` jamais stage.
- **Canal-mouvement** : signal = deplacement PREDIT (WM) vs REALISE (proprio/odometrie), deja
  disponible, aucun nouveau capteur.
- **La voie** : soit WM re-collecte + fine-tune avec obstacles (dynamique de blocage apprise), soit
  predicteur d'affordance separe (appris des erreurs de deplacement) + son integration planner.

## Contraintes imposees par les lecons (pas des options)
- Une variable a la fois ; gates chiffres PRE-ENREGISTRES ; negatif = commite + STOP + escalade owner ;
  ne JAMAIS stager `godot/scripts/main.gd` ni `ui/` ; carte `tools/archi_hud/architecture.json` a jour
  DANS LE MEME COMMIT ; commits Conventional anglais SANS attribution IA ; README = constat au present.
- Lancer un run en background = la commande SEULE, AUCUN preambule (bash -n / pgrep) : un grep sans
  match exit1 detache le run de sa notification ET laisse un orphelin qui respawn (lecon G3, coute cher).
- Tuer = `/sylvan-kill` + verifier 0 orphelin (pgrep par NOM, pas par pattern qui s'auto-matche).
- Purger `godot/data/replay_buffer/critic_tmp_*` apres chaque collecte. PPO `--lr 1e-4` si retrain.
- Viabilite du MONDE mesuree AVANT de juger l'agent (lecon plafond metabolique) : l'obstacle doit
  creer une VRAIE place (contourner coute un detour, mais reste possible et survivable) ; sinon on juge
  des vies condamnees par l'arithmetique.
- 100 % artifact interdit (pas de page hebergee claude.ai, hook + reglage) : tout point = texte ou docs/*.md.

## Fichiers probablement touches
`godot/scripts/world/` (obstacle, patron food_manager) ; `godot/scripts/agent/sylvan_agent.gd`
(`_kinematic_step` respecte les solides) ; `diagnostics/diag_obstacle_*.py` (le diag gratuit) ;
selon la voie : `scripts.collect_wm_*` + `scripts.train_wm_command` (retrain WM) OU un nouveau
`python/sylvan/models/affordance_head.py` + son integration `serve_planner_command.py` ;
`docs/design_obstacle_affordance.md` (pre-inscription + verdicts) ; carte.

## Si PASS
Premier canal de consequence NON-homeostatique appris ; la TOPOLOGIE existe. Ca debloque le chantier
suivant, CHERCHER + memoire spatiale (le vrai mur de capacite, enfin decisif car le monde a des
detours et des occlusions). Mettre a jour carte + README + memoire dans le meme commit. Reviser
`docs/roadmap_vers_monde_v3.md` (qui mettait chercher/memoire AVANT les obstacles = inverse).
