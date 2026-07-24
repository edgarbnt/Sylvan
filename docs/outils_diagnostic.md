# OUTILS DE DIAGNOSTIC — mode d'emploi (matrice · contrat de monde · tableau des gates)

**Mission** : savoir OÙ une information du monde meurt dans le pipeline, savoir si le monde SERVI est
celui qu'on a DEMANDÉ, et savoir où on en est des gates — sans re-payer une heure de sondes ad-hoc à
chaque fois. Les trois sont **gratuits** : ils lisent un corpus déjà collecté et un WM **gelé**.

**À lire d'abord** : `docs/design_outil_matrice_information.md` (le pourquoi, la validation chiffrée,
et les pièges de mesure trouvés en construisant) · `python/sylvan/info_matrix.py` (toute la mécanique
de mesure) · `diagnostics/guards.py` (les garde-fous auxquels ces outils se branchent).

Tout se lance depuis la racine, avec `PYTHONPATH=python` et le venv CPU :

```bash
PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/<outil>.py ...
```

---

## 1. `diag_info_matrix.py` — la matrice de survie de l'information

**La question** : « la représentation garde-t-elle ce qui compte ? » — la question centrale d'une
archi JEPA. Un logger dit ce qu'un module a FAIT ; la matrice dit où l'information MEURT.

```bash
PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_info_matrix.py \
    --corpus data/replay_buffer/critic_bosq_ripe11 --depths 0 20 79 --json m.json
```

**Lignes** = propriétés du monde · **colonnes** = étages du pipeline · **case** = la part récupérable
de l'information, sonde LINÉAIRE puis sonde MLP, en held-out par épisode.

| ligne | ce qu'elle porte |
|---|---|
| `position` | le slot (x, z) — ce que le coût du planner consomme ; rapporte aussi le R² de la DISTANCE |
| `distance` | la distance de la proie visée — le terme porteur de `-min_dist` |
| `type` | l'apparence de la proie (catégorie) — sans elle, aucun critique n'apprend une table de valeurs |
| `vue` | « y a-t-il de la bouffe en vue » — le fait le plus élémentaire de la scène |
| `maturite` | la luminosité du buisson — indice perceptible NON géométrique |
| `retine` | la rétine entière — la borne haute de ce que l'observation contient |
| `repas` | « un repas dans les K ticks » — la seule ligne qui porte une CONSÉQUENCE ; rapporte l'AUC |

**Comment la lire.** Une case ≈ baseline → l'information n'est plus là. Une CHUTE entre deux étages
désigne le module qui la détruit : c'est toute la valeur de l'outil. Les chutes sont lues le long des
**arêtes réelles**, pas de gauche à droite — le slot est une branche séparée qui part de la rétine,
jamais la suite du latent.

**Deux options qui changent le sens de ce qu'on mesure** :
- `--target predit` (défaut) : le latent rêvé à la profondeur *d* est jugé sur ce qui est VRAI à
  *t+d*. C'est la question JEPA : le rêve prédit-il encore ? `--target percu` juge tout sur *t* : la
  question de la mémoire. À la profondeur 0 les deux coïncident.
- `--split episode` (défaut, honnête) vs `--split positional` (⚠️ **fuite assumée**) : la seconde
  rejoue la convention des mesures historiques, uniquement pour comparer des chiffres à des chiffres.

**Usage NON-RÉGRESSION** — le vrai gain à long terme : `--json m.json` après chaque retrain, puis
comparer les colonnes. Une case qui baisse est une régression du substrat, visible AVANT de dépenser
une heure d'A/B. Deux runs sont bit-identiques (seed fixé, mono-thread, sonde linéaire = ridge).

**Coût** : ~7 min pour 3 lignes × 7 colonnes sur 5 000 états. Réduire avec `--rows`, `--depths`,
`--stride`, `--mlp-steps`.

### Ajouter une ligne (une mécanique de monde = une ligne de plus)

Dans `python/sylvan/info_matrix.py`, écrire une fonction qui rend `(valeurs, validité)` depuis un
`Sample`, puis l'ajouter à `PROPERTIES`. La vérité-terrain se calcule **depuis la rétine** (ce que
l'agent VOIT), jamais depuis un état caché du monde — sinon on mesure un oracle, pas une perception.

```python
def _p_ma_mecanique(s: Sample) -> tuple[torch.Tensor, torch.Tensor]:
    ...  # -> (valeurs [N, D] ou classes [N], validité [N] booléenne)

PROPERTIES.append(Property("ma_meca", "libellé affiché", "cont", _p_ma_mecanique,
                           why="pourquoi cette propriété compte"))
```

C'est ce qui matérialise la règle « une sonde écrite EN MÊME TEMPS que la mécanique, jamais après ».

---

## 2. `diag_world_contract.py` — le monde servi est-il celui qu'on a demandé ?

**La question** : un réglage silencieusement inactif ne produit pas d'erreur, il produit un
RÉSULTAT — qu'on interprète alors comme une propriété du monde. C'est le mode de panne qui a déjà
coûté trois fois du temps au projet, et il est entièrement automatisable.

```bash
PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_world_contract.py \
    data/replay_buffer/critic_bosq_ripe11 --set SYLVAN_ENERGY_DRAIN=0.05 --set SYLVAN_FOOD_RIPE_CUE=1
```

Dix clauses, chacune = un réglage DEMANDÉ opposé à une mesure qui l'atteste sur le corpus SERVI.
Quand la variable n'est pas posée, c'est le **défaut du code** qui est opposé à la mesure, cité avec
son fichier. **Sortie ≠ 0 en cas de divergence** → à mettre en garde-fou avant une collecte longue.

Trois verdicts, et le troisième est celui qui justifie l'outil :

| verdict | sens |
|---|---|
| ✅ | demandé et servi (ou inactif des deux côtés) |
| 🚨 DEMANDÉ mais NON SERVI | le réglage n'a pas pris |
| 🚨 **SERVI SANS ÊTRE DEMANDÉ** | le réglage fantôme — celui qu'on ne voit jamais |
| ⚠️ | non mesurable sur ce corpus (l'événement ne s'y produit pas) — jamais un ✅ déguisé |

**Ajouter une clause** : une `Clause(env, label, défaut, source_du_défaut, mesure, mode)` dans
`CONTRACT`. Le défaut DOIT être lu dans le code et cité — l'inventer ferait de l'outil un menteur de
plus. Modes : `approx` (tolérance relative), `plafond` (le servi doit rester sous le demandé),
`presence` (actif/inactif), `compte` (égalité entière).

---

## 3. `diag_gates_board.py` — où en est-on des gates ?

**La question** : le projet décide par gates pré-enregistrés et il en a des dizaines ; les verdicts
vivent dispersés dans les messages de commit. Sans vue d'ensemble, on re-teste ce qui est tranché et
on empile sur ce qui ne l'est pas.

```bash
PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_gates_board.py [--chantier obstacle] [--full]
PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_gates_board.py --run <gate> [--force]
```

**Aucun registre écrit à la main** — il dériverait du code en une semaine. Tout est dérivé : la liste
des gates vient des `diagnostics/diag_*.py` dont le docstring pré-enregistre un critère, la question
de leur première ligne, le critère de leur bloc « CRITÈRES PRÉ-ENREGISTRÉS », et **le verdict est
CITÉ depuis git** (le commit le plus récent portant un mot de verdict explicite ET mentionnant ce
gate). `--run` rejoue et inscrit dans `tools/gates/ledger.jsonl`.

**Lire les symboles** : `✅` passé · `❌` échoué/réfuté · `⏸️` gelé · `?` **aucun verdict retrouvable
dans l'historique — ce n'est PAS « non testé »** · `$` gate coûteux (son CODE lance le monde ou un
entraînement ; `--run` le refuse sans `--force`) · `*` module au focus de la carte d'archi.

**Faire apparaître un nouveau gate** : rien à déclarer — écrire le diagnostic avec un bloc
« CRITÈRES PRÉ-ENREGISTRÉS » dans son docstring (ce que la discipline du projet impose déjà), et le
committer avec un sujet qui porte le verdict et nomme le gate.

---

## Ce à quoi les trois se branchent (et ne dupliquent pas)

- **`diagnostics/guards.py`** — bannière d'échafaudages, `sanity()` (la matrice REFUSE de rendre un
  tableau sur un corpus dégénéré), `measured_constants()` (le contrat s'en sert, il ne le re-code
  pas). Les palettes du monde sont MESURÉES sur le corpus et comparées aux palettes déclarées : même
  esprit que `check_constants`.
- **`tools/archi_hud/architecture.json`** — la matrice compare le checkpoint WM que la carte déclare
  à celui qu'elle sonde et crie si ça diverge ; le tableau des gates y lit le focus et l'état des
  modules. Ces outils ne sont PAS des modules d'architecture : ils ne figurent pas dans la carte, au
  même titre que `guards.py` et l'archi-HUD eux-mêmes.
- **`python/sylvan/critic_corpus.py`** — chargement (`.jsonl` et `.jsonl.gz`), frontières d'épisode,
  étiquetage « repas dans K ticks », token du planner, AUC. Rien de tout ça n'est ré-implémenté.

## Limites connues

- La ligne `type` est **structurellement bruitée** : les types sont re-tirés à la repousse, donc les
  classes sont décalées entre train et held-out (la matrice le signale). La conclusion qualitative
  tient, le centième non.
- La matrice mesure ce que le WM CONTIENT, pas ce que le planner UTILISE. Une information présente
  dans une colonne peut rester inexploitée en aval.
- Le contrat ne couvre que dix réglages, ceux qui sont mesurables sur un corpus BC. Un réglage
  invisible dans les logs (le champ de vision de la rétine, par exemple) reste non vérifié.
- Le tableau des gates ne rejoue rien tout seul : `?` veut dire « verdict non retrouvable dans
  l'historique », pas « non testé ».
