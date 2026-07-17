# Les « formes d'intelligence » de Sylvan — audit honnête (snapshot 2026-07-18)

> Auto-audit clinique, dans l'esprit §2 (ne pas survendre) : pour CHAQUE forme d'intelligence du
> projet, on dit si elle **PORTE vraiment** le comportement (load-bearing), ou si elle est
> démo/échafaudage/échec/pas-encore. C'est un instantané ; à réviser quand l'archi bouge (source de
> vérité des états = `tools/archi_hud/architecture.json`).
>
> North-star (CLAUDE.md) : une entité ALife qui décide elle-même (faim → chercher → approcher →
> survivre) via planification dans un WM appris. La question de cet audit : **de cette intelligence,
> qu'est-ce qui sert concrètement aujourd'hui ?**

On classe en trois familles : **Percevoir**, **Décider/planifier**, **Intelligence plus profonde**.

---

## A. Percevoir (associer un visuel à un sens) — LA vraie réussite, load-bearing

| Forme | Utilisée ? | Détail |
|---|---|---|
| Perception object-centric (slots requête-couleur, readout géométrique) | ✅ décisif | l'agent NAVIGUE par elle (localise bouffe/eau). |
| Perception typée (couleur→drive DÉCOUVERT par contingence) | ✅ oui | la perception vivante, 100 % apprise du vécu (`wm_objcentric_kin_typed`). |
| Lunette saillance-danger (visuel → « fait mal », appris des dégâts) | ✅ promue | juge PASS 41/9 ; la clé-apparence « danger=vert » dissoute. |
| Attribution de crédit (baie-vs-buisson, classe NEUTRE, bon indice) | ✅ juge PASS | crédit au bon indice parmi co-occurrents + reconnaît un neutre. |
| Prédicteur d'affordance (visuel → « bloque », voie B, canal mouvement) | 🟡 en cours | appris (AUC 1.0, sélectivité cyan-vs-bouffe) ; pas encore jugé en vies. |

**→ C'est ici que le projet a une intelligence RÉELLE et utile.** L'entité *sait ce qu'est le monde*
(bouffe/eau/danger/obstacle et leurs conséquences) **sans qu'on le lui code** — chaque « sens » est une
petite TÊTE apprise du vécu, posée sur un WM gelé. C'est la frontière et l'acquis solide.

---

## B. Décider / planifier — utilisé, mais surtout DESIGNÉ (peu « appris »)

| Forme | Utilisée ? | Détail |
|---|---|---|
| WM (imagination / rêve) | ✅ décisif | le planner rêve dedans pour noter les commandes. « partiel » : Dreamer-like, pas JEPA pur (latent dé-collapsé partiellement). |
| Planner MPC (rollouts (vx,ω) notés) | ✅ oui | LE décideur de locomotion. Mais l'« intelligence » est mince : *va vers la ressource la plus proche* (glouton). |
| Coût survie / arbitrage multi-drive | ✅ promu | MAIS c'est un **coût DESIGNÉ** (drain/restore + look-ahead à la consommation), pas une valeur apprise. |
| Étage waypoint (hiérarchie petit-H-JEPA) | ✅ oui (monde-danger) | casse le troc éviter↔manger que le planner plat ne pouvait pas ; MAIS routage **codé-main** (échafaudage déclaré). |
| Sprint-critique (arbitrage du sprint-danger, appris) | ✅ promu | **la SEULE décision vraiment APPRISE validée en vies** (juge PASS, tue l'oracle codé). Étroite : *quand sprinter dans le danger pour manger*. |

**→ Le décideur est surtout DESIGNÉ.** Une seule décision *apprise* tient en vies (le sprint-critique),
et elle est étroite. Le reste des choix vient de coûts codés-main (agnostiques-à-l'objet, mais designés).

---

## C. Intelligence « plus profonde » — échec honnête ou pas construite

| Forme | Utilisée ? | Détail |
|---|---|---|
| Critique-résidu appris (remplacer le coût designé) | ❌ non | *ne décidait que 3 % des replans en épars* ; forcé (`SYLVAN_CRITIC_ALWAYS`), il **divise le forage par 2**. Cause diagnostiquée : horizon du rêve ~0.8 m vs ressources 2-8 m → on lui demande de RANGER alors qu'il a appris à PRÉDIRE. Bâti mais **ne sert pas** aujourd'hui. |
| Mode-1 (politique apprise, RL model-free) | ❌ non | plafonne au niveau BC (~1900), jamais promu. Mur = arbitrage. Parqué. |
| Mémoire spatiale | ⚪ pas encore | gardée pour un substrat propre ; le monde ouvert (topologie/occlusion) la rendra décisive — c'est justement ce que le chantier obstacle prépare. |
| Curiosité / motivation intrinsèque | ⚪ pas construite | recherche AXE 5 (RND / Plan2Explore, plafonnée par l'homéostasie). |
| Configurator / cycle jour-nuit (ré-apprendre EN VIVANT) | ⚪ manquant | l'orchestrateur qui déciderait *quelle tête (ré)entraîner* n'existe pas (embryon `remeasure.py` seulement). |

---

## Le verdict honnête : « ça sert à quoi ? »

**Oui, mais à un niveau modeste, et pas là où on croit.**

1. **Ce qui sert vraiment = la PERCEPTION apprise.** Mise bout à bout avec le planner, elle produit un
   comportement **autonome réel** : l'entité a faim/soif, perçoit, va vers la bonne ressource, évite le
   danger, arbitre, survit (~2735 en multi-drive). Ce n'est pas de la démo — l'entité *fait* quelque
   chose sans monde codé-main. C'est le north-star atteint **partiellement**.
2. **Ce qui est plus faible = la DÉCISION.** Aujourd'hui l'agent est surtout « perception apprise +
   planner glouton designé ». Les vraies décisions *apprises* (critique, Mode-1) ont **échoué ou
   stagné** — et le projet le **documente** au lieu de le maquiller (c'est tout le point du §2). La
   seule décision apprise qui tient est le sprint-critique, étroite.
3. **Ce qui n'existe pas encore = l'intelligence « haute »** (mémoire, curiosité, découverte ouverte,
   ré-apprentissage en vivant). C'est l'horizon, pas le présent.

**Formule** : Sylvan a une **intelligence de PERCEPTION** réelle et load-bearing (le corps sait ce
qu'est le monde par expérience), branchée sur une **intelligence de DÉCISION encore largement designée**
(un seul arbitrage appris validé), l'**intelligence d'APPRENTISSAGE CONTINU / mémoire / curiosité**
restant à bâtir.

**Pourquoi le chantier obstacle est cohérent avec ça** : il **étend la perception apprise** à une
nouvelle nature de conséquence (le mouvement, non-homéostatique) — donc il renforce précisément la
partie qui *marche* — et il crée la **topologie** (détours, occlusions) sans laquelle la mémoire
(l'intelligence d'après) resterait triviale et inutile. On consolide la fondation qui porte avant de
poser l'étage qui ne tient pas encore.
