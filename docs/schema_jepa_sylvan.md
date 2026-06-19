# 🧠 Schéma complet — JEPA / AMI (LeCun) × projet Sylvan

> Rendu gratuit : colle un bloc dans **mermaid.live**, ou ouvre ce fichier dans **VS Code** avec
> l'extension *Markdown Preview Mermaid Support*, ou pousse-le sur **GitHub** (rendu natif).
> Pour rendre en PNG/SVG sans compte : `npx -y @mermaid-js/mermaid-cli -i docs/schema_jepa_sylvan.md -o schema.png`.
>
> Le doc va du **général (théorie JEPA)** au **particulier (notre implémentation)**, puis les **mappings**,
> le **WM en détail**, la **rétine**, le **planner**, les **pulsions**, et l'**état + roadmap**.

---

## 0. Vue d'ensemble — la thèse en une image

```mermaid
flowchart TB
    subgraph THEORY["🌍 THÉORIE — JEPA / AMI (Yann LeCun)"]
        direction TB
        T1["Apprendre une REPRÉSENTATION du monde<br/>à partir de la perception BRUTE"]
        T2["Prédire dans l'ESPACE LATENT<br/>(pas reconstruire les pixels/entrées)"]
        T3["Planifier des ACTIONS en imaginant<br/>le futur dans ce latent (Mode-2)"]
        T1 --> T2 --> T3
    end

    subgraph SYLVAN["🪲 SYLVAN — ALife émergente"]
        direction TB
        S1["Hexapode qui doit DÉCIDER seul :<br/>faim → chercher → aller → manger → survivre"]
        S2["3 couches : CPG codé + résidu PPO + planner JEPA"]
        S3["Perception APPRISE (rétine) + World-Model + MPC"]
        S1 --> S2 --> S3
    end

    THEORY -.cap visé.-> SYLVAN
    classDef th fill:#1e293b,stroke:#7dd3fc,color:#e0f2fe
    classDef sy fill:#14532d,stroke:#86efac,color:#dcfce7
    class T1,T2,T3 th
    class S1,S2,S3 sy
```

---

## 1. Concepts JEPA (le cœur théorique)

```mermaid
flowchart TB
    subgraph CORE["JEPA = Joint Embedding Predictive Architecture"]
        direction LR
        X["Observation x<br/>(état/percept à t)"] --> EncX["Encodeur f(x)<br/>→ s_x (latent)"]
        Y["Observation y<br/>(état/percept à t+1)"] --> EncY["Encodeur cible f̄(y)<br/>→ s_y (latent cible)"]
        A["Action / commande a"] --> Pred
        EncX --> Pred["Prédicteur g(s_x, a)<br/>→ ŝ_y (latent prédit)"]
        Pred --> Loss["Perte dans le LATENT<br/>d(ŝ_y, s_y)  (cosine/L2)"]
        EncY --> Loss
    end

    subgraph WHYNOT["Pourquoi PAS génératif (Dreamer/VAE) ?"]
        direction TB
        G1["Reconstruire pixels/entrées =<br/>gaspille la capacité sur des détails"]
        G2["JEPA prédit l'ABSTRAIT<br/>(ce qui est prévisible), ignore le bruit"]
        G1 --> G2
    end

    subgraph COLLAPSE["⚠️ Problème : EFFONDREMENT (collapse)"]
        direction TB
        C1["Si la perte latente seule :<br/>l'encodeur peut sortir une CONSTANTE<br/>→ d(ŝ_y,s_y)=0 trivialement"]
        C2["Anti-collapse VICReg :<br/>• Variance (chaque dim varie)<br/>• Covariance (dims décorrélées)<br/>• Invariance (prédiction juste)"]
        C3["Métrique : eff_rank du latent<br/>(rang effectif ; haut = riche)"]
        C1 --> C2 --> C3
    end

    CORE --> COLLAPSE
    classDef c fill:#0f172a,stroke:#7dd3fc,color:#e0f2fe
    class X,Y,A,EncX,EncY,Pred,Loss,G1,G2,C1,C2,C3 c
```

---

## 2. AMI — l'architecture cognitive cible de LeCun (et où on en est)

```mermaid
flowchart TB
    World([" Monde / Simulateur Godot "])

    subgraph AMI["AMI — Autonomous Machine Intelligence (LeCun)"]
        direction TB
        Perc["👁️ PERCEPTION<br/>perçoit l'état du monde → latent"]
        WM["🌐 WORLD MODEL<br/>prédit l'évolution du latent | action"]
        Cost["💲 COST / CRITIC<br/>mesure l'inconfort (intrinsèque+homéo)"]
        Actor["🎮 ACTOR<br/>propose des actions"]
        Mem["🧮 MÉMOIRE court-terme"]
        Config["🎚️ CONFIGURATOR<br/>oriente selon la tâche"]

        Perc --> WM
        WM --> Cost
        Cost --> Actor
        Actor --> WM
        Config -.-> Perc & WM & Cost & Actor
        Mem -.-> WM
    end

    World --> Perc
    Actor --> World

    subgraph MODES["System 1 / System 2"]
        M1["Mode-1 : politique réactive<br/>(1 passe, rapide) — À FAIRE (dernier)"]
        M2["Mode-2 : planification/recherche<br/>(MPC, lent, délibéré) — ACTUEL"]
    end
    Actor --- MODES

    classDef ami fill:#1e1b4b,stroke:#a5b4fc,color:#e0e7ff
    classDef done fill:#14532d,stroke:#86efac,color:#dcfce7
    classDef todo fill:#7c2d12,stroke:#fdba74,color:#ffedd5
    class Perc,WM ami
    class Cost,Actor,Mem,Config ami
    class M2 done
    class M1 todo
```

---

## 3. Sylvan — le PIPELINE 3 couches (l'architecture réelle)

```mermaid
flowchart TB
    subgraph L3["☁️ COUCHE 3 — CERVEAU JEPA (espace COMMANDE)"]
        direction TB
        Retina["👁️ RÉTINE (perception apprise)<br/>36 rayons × depth+RGB → tête → position food/eau"]
        WMc["🌐 WORLD-MODEL commande<br/>(vx,ω) → déplacement+énergie imaginés"]
        Planner["🧭 PLANNER MPC<br/>cherche la meilleure (vx,ω)"]
        Retina --> Planner
        WMc --> Planner
    end

    subgraph L2["⚙️ COUCHE 2 — RÉSIDU PPO (borné ±0.4)"]
        Res["Politique résiduelle apprise<br/>équilibre + propulsion en tournant"]
    end

    subgraph L1["🦿 COUCHE 1 — CPG codé main"]
        CPG["Générateur de patrons<br/>marche TRÉPIED + virage PAR CONSTRUCTION"]
    end

    Body([" 🪲 Hexapode (Godot)<br/>proprio=132, action=18 "])

    Planner -->|"commande (vx, ω)"| CPG
    Planner -->|"obs"| Res
    CPG -->|"angles cibles"| Add(("➕"))
    Res -->|"résidu borné"| Add
    Add -->|"action 18-D"| Body
    Body -->|"proprio + rétine"| Retina
    Body -->|"proprio + radar(dyn)"| WMc

    classDef l3 fill:#1e3a8a,stroke:#93c5fd,color:#dbeafe
    classDef l2 fill:#3730a3,stroke:#c7d2fe,color:#e0e7ff
    classDef l1 fill:#374151,stroke:#d1d5db,color:#f3f4f6
    class Retina,WMc,Planner l3
    class Res l2
    class CPG l1
```

> **Principe clé (BLUEPRINT §14)** : le WM/planner raisonnent en **(vx, ω)** ; la « bouffe » ne vit
> QUE dans le **coût du planner** (agnosticité de la tâche). La locomotion est un *prérequis*, pas le but.

---

## 4. World-Model en détail (le moteur JEPA de Sylvan)

```mermaid
flowchart LR
    subgraph IN["Entrée obs"]
        P["proprio (132)"]
        R["rétine (144)<br/>— remplace radar(12)"]
        E["énergie (1)"]
    end
    P & R & E --> Enc["🧩 ENCODEUR<br/>obs(277) → latent"]

    Enc --> Lat[("latent z_t")]
    Cmd["commande (vx,ω)"] --> Predr["🔮 PRÉDICTEUR<br/>g(z_t, cmd) → z_t+1"]
    Lat --> Predr --> Lat2[("z_t+1")]

    Lat2 --> H1["📍 tête DÉPLACEMENT<br/>(d_fwd, d_lat, d_yaw)"]
    Lat2 --> H2["🔋 tête ÉNERGIE"]
    Lat2 --> H3["💀 tête DONE (chute)"]
    Lat2 -.recon génératif, poids→0.-> H4["proprio/rétine reconstruits"]

    subgraph TRAIN["🏋️ Pertes d'entraînement"]
        direction TB
        TL1["latent (cosine) — JEPA"]
        TL2["VICReg var+cov+gamma — anti-collapse"]
        TL3["displacement / energy / done — ancres abstraites"]
        TL4["recon proprio/rétine — génératif (poids → 0)"]
    end
    H1 & H2 & H3 --> TRAIN
    H4 --> TL4

    classDef io fill:#0f172a,stroke:#7dd3fc,color:#e0f2fe
    classDef core fill:#1e3a8a,stroke:#93c5fd,color:#dbeafe
    class P,R,E,Cmd io
    class Enc,Predr,Lat,Lat2,H1,H2,H3,H4 core
```

### Statut JEPA du WM (honnête)

```mermaid
flowchart LR
    D1["DREAMER-like<br/>(reconstruction, eff_rank 6 = effondré)"]
    -->|"étape 1 : VICReg+cosine"| D2["dé-collapse<br/>eff_rank 6→26 (sur grosses data)"]
    -->|"étape 2 : drop recon, latent×5"| D3["FONCTIONNELLEMENT JEPA<br/>prédit SUR le latent"]
    -->|"RÉTINE : le WM perçoit les rayons bruts"| D4["JEPA + complet<br/>(obs 145→277)"]
    classDef s fill:#14532d,stroke:#86efac,color:#dcfce7
    class D1,D2,D3,D4 s
```

---

## 5. LA RÉTINE — perception apprise (le grand acquis récent)

```mermaid
flowchart TB
    Scene([" Scène : food=rouge(0.9,0.3,0.2), eau=bleu(0.2,0.5,0.95) "])
    Scene --> Ray["📡 RAYCAST PHYSIQUE<br/>36 rayons sur 360° depuis la tête<br/>couche collision dédiée (gait non perturbé)"]
    Ray --> Vec["Vecteur RÉTINE (144)<br/>par rayon : [depth, R, G, B]<br/>(miss → depth=1, RGB=0)"]

    Vec --> Head["🧠 TÊTE DE PERCEPTION APPRISE 🅐<br/>attention géométrique (soft-argmax)<br/>scoreur/rayon : [depth,R,G,B,sinθ,cosθ]"]
    Head --> Pos["position estimée (x_right, z_fwd)<br/>+ présence — food ET eau"]
    Vec --> WMin["→ entrée du WORLD-MODEL (étage 2)"]

    Pos --> PlannerCost["💲 entre dans le COÛT du planner<br/>(remplace l'oracle food_xz_from_radar)"]

    subgraph EMERGE["Ce qui ÉMERGE (pas codé)"]
        Em["« rouge = nourriture »<br/>« bleu = eau »<br/>localisation par triangulation"]
    end
    Head -.fait émerger.-> EMERGE

    subgraph OLD["❌ AVANT = ORACLE"]
        Or["radar 12-secteurs analytique<br/>souffle direction+distance de la bouffe"]
    end
    OLD -.remplacé par.-> Head

    classDef r fill:#7f1d1d,stroke:#fca5a5,color:#fee2e2
    classDef n fill:#14532d,stroke:#86efac,color:#dcfce7
    class Scene,Ray,Vec r
    class Head,Pos,WMin,PlannerCost,Em n
    class Or r
```

---

## 6. LE PLANNER MPC (Mode-2, recherche dans l'imaginaire)

```mermaid
flowchart TB
    Obs["obs courante + position food/eau (rétine)"] --> Gen["1) GÉNÈRE ~102 séquences candidates<br/>vx∈{.55,.65,.75} × ω∈{−.6..+.6}<br/>+ candidats 2-segments"]
    Gen --> Roll["2) ROLLOUT OPEN-LOOP dans le WM (latent)<br/>imagine déplacement+énergie sur l'horizon (~80-120)"]
    Roll --> Score["3) COÛT par candidat"]

    subgraph COST["💲 Fonction de coût"]
        direction TB
        K1["− min_dist (se rapprocher de la cible)"]
        K2["+ heading_weight · alignement (cos bearing)<br/>gate de distance (anti-orbite)"]
        K3["+ energy_weight · énergie prédite"]
        K4["− done_penalty · risque de chute"]
        K5["MULTI-PULSION : urgence(1−niveau)^p<br/>arbitrage faim/soif ÉMERGENT"]
    end
    Score --> COST
    COST --> Best["4) argmax → commande (vx, ω)"]
    Best -->|"envoyée au CPG + résidu"| Exec([" exécution in-engine "])

    Note["🔒 NE PAS toucher au MPC brute-force maintenant<br/>→ remplacé par Mode-1 EN DERNIER"]
    Gen -.- Note
    classDef p fill:#1e3a8a,stroke:#93c5fd,color:#dbeafe
    class Obs,Gen,Roll,Score,Best,K1,K2,K3,K4,K5 p
```

---

## 7. Pulsions / homéostasie (la boucle de survie ALife)

```mermaid
flowchart LR
    subgraph DRIVES["🩸 Homéostasie"]
        En["Énergie ↓ (drain)"] -->|"faim"| NeedF["besoin FOOD 🔴"]
        Th["Soif ↓ (drain)"] -->|"soif"| NeedW["besoin WATER 🔵"]
    end
    NeedF & NeedW --> Urg["URGENCE = (1 − niveau)^p<br/>(convexe → le critique domine)"]
    Urg --> Arb["⚖️ ARBITRAGE dans le coût planner<br/>affamé→food, assoiffé→eau,<br/>urgence bat proximité"]
    Arb --> Act["va manger / boire → niveau remonte"]
    Act -.boucle.-> DRIVES

    Note2["Arbitrage ÉMERGENT (pas codé)<br/>mais MYOPE (horizon court) →<br/>fix = critique foresighted"]
    Arb -.- Note2
    classDef d fill:#3730a3,stroke:#c7d2fe,color:#e0e7ff
    class En,Th,NeedF,NeedW,Urg,Arb,Act d
```

---

## 8. MAPPING — concept JEPA/AMI → composant Sylvan → statut

```mermaid
flowchart LR
    subgraph J["Concept JEPA/AMI"]
        JP["Perception (x→latent)"]
        JW["World Model"]
        JC["Cost / Critic"]
        JA["Actor (Mode-1)"]
        JS["Recherche (Mode-2)"]
        JR["Représentation non-effondrée"]
    end
    subgraph SY["Composant Sylvan"]
        SP["Rétine raycast + tête apprise + encodeur WM"]
        SW["CommandWorldModel (latent, displacement/énergie)"]
        SC["Coût planner codé main (dist+heading+énergie+urgence)"]
        SA["CPG + résidu PPO (réactif)"]
        SS["MPC brute-force ~102 candidats"]
        SR["VICReg + cosine ; métrique eff_rank"]
    end
    JP --> SP
    JW --> SW
    JC --> SC
    JA --> SA
    JS --> SS
    JR --> SR

    SP --- stP["✅ FAIT (oracle mort)"]
    SW --- stW["✅ perçoit la rétine (étage 2)"]
    SC --- stC["⚠️ codé main → 🎯 critique APPRIS (à venir)"]
    SA --- stA["🟡 réactif présent → Mode-1 distillé (dernier)"]
    SS --- stS["✅ actuel (à garder jusqu'à Mode-1)"]
    SR --- stR["⚠️ eff_rank data-limité (en cours d'enrichissement)"]

    classDef j fill:#1e1b4b,stroke:#a5b4fc,color:#e0e7ff
    classDef s fill:#14532d,stroke:#86efac,color:#dcfce7
    classDef st fill:#0f172a,stroke:#64748b,color:#e2e8f0
    class JP,JW,JC,JA,JS,JR j
    class SP,SW,SC,SA,SS,SR s
    class stP,stW,stC,stA,stS,stR st
```

---

## 9. ÉTAT & ROADMAP (où on en est, où on va)

```mermaid
flowchart TB
    subgraph DONE["✅ FAIT"]
        B["Base motrice hexapode v2<br/>(0.49 m/s, cap droit, fall 0%)"]
        WMv2["WM commande (wm_command_hex_v2)"]
        AB["Fix navigation A→B (heading_weight)"]
        Drv["2ᵉ pulsion soif+eau, arbitrage émergent"]
        Ret["RÉTINE étages 0-1-2<br/>foraging WM-rétine 1050 > oracle 965"]
    end

    subgraph NOW["🔄 EN COURS"]
        Enr["Enrichir le latent du WM-rétine<br/>(collecte babbling 2×200, eff_rank 3.6→~12)"]
    end

    subgraph NEXT["🎯 À VENIR (ordre)"]
        Lat["🅑 PLANIFIER EN LATENT<br/>coût = énergie/inconfort futur prédit<br/>(plus de coordonnées x,z) = JEPA pur"]
        Crit["🧮 CRITIQUE FORESIGHTED<br/>horizon long, fix robustesse multi-pulsions myope"]
        M1["⚡ MODE-1<br/>politique apprise 1-passe (+ recherche en fallback)"]
        Water["💧 rétiniser l'eau (tête n_res=2)"]
    end

    DONE --> NOW --> Lat
    Lat --> Crit --> M1
    Ret -.optionnel.-> Water

    classDef d fill:#14532d,stroke:#86efac,color:#dcfce7
    classDef n fill:#78350f,stroke:#fcd34d,color:#fef3c7
    classDef x fill:#7c2d12,stroke:#fdba74,color:#ffedd5
    class B,WMv2,AB,Drv,Ret d
    class Enr n
    class Lat,Crit,M1,Water x
```

---

## 10. Dimensions & checkpoints (référence rapide)

| Élément | Valeur |
|---|---|
| proprio | **132** |
| action | **18** (6 pattes × 3 DOF) |
| obs policy | **144** (132 + vision-commande 12) |
| obs WM (radar) | **145** (132 + radar 12 + énergie 1) |
| obs WM (RÉTINE) | **277** (132 + rétine 144 + énergie 1) |
| rétine | **36 rayons × 4** (depth,R,G,B) = 144 |
| Base motrice | `data/checkpoints/hexapod_v2/policy_best.pt` |
| WM radar | `data/checkpoints/wm_command_hex_v2/wm_best.pt` |
| WM rétine (étage 2) | `data/checkpoints/wm_command_hex_retina_v1/wm_best.pt` |
| Tête perception 🅐 | `data/checkpoints/retina_head/head_best.pt` |

---

### Légende couleurs
🟦 théorie / cerveau JEPA · 🟩 fait / acquis · 🟧 à venir / en cours · 🟥 perception (rétine) ou ancien oracle.
