# Sylvan : Emergent survival in a learned world-model

> An embodied agent that learns to **decide for itself** : *get hungry → look → go to food → survive*, by **planning inside a world-model it learned from its own experience**, in the spirit of Yann LeCun's blueprint for autonomous machine intelligence.

<p align="center">
  <img src="assets/forage_sim.gif" width="640" alt="The 3D agent foraging in the simulation: perceiving food/water and juggling hunger & thirst to survive."/>
</p>

<p align="center"><i>The real simulation: a six-legged agent perceives resources (red = food, blue = water) and arbitrates two competing drives to stay alive.</i></p>

---

## The goal

Most agents are told what to do. The goal here is the opposite: an agent whose behavior **emerges** from a few built-in needs and a model of the world it learns on its own.

Concretely, the target is an agent that:
- has **several competing drives** (hunger, thirst, …) and must **arbitrate between them over time**, *"drink now, or eat first?"*, with genuine look-ahead;
- **perceives** the world through learned senses (no hand-labeled oracle);
- learns its model of the world **self-supervised**, from raw experience, with no external reward telling it what a "good" state is;
- **plans** by imagining consequences inside that model, rather than reacting reflexively or being scripted.

This is a small, embodied step toward the research program Yann LeCun laid out for **autonomous machine intelligence**, so it's worth starting with that blueprint before the specifics.

---

## Background: JEPA and LeCun's blueprint

In *[A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)* (LeCun, 2022), the path to agents that can perceive, reason, and plan runs through a **learned world-model**, trained **self-supervised**, and used for **planning**, not through ever-larger reactive/generative models. Two ideas matter here.

### 1. JEPA, predict in representation space, not pixel space

A **Joint Embedding Predictive Architecture (JEPA)** learns by predicting the **embedding** of a missing/future part of the world from the embedding of the observed part, in an *abstract representation space*, **not** by reconstructing raw pixels or tokens. Two related inputs are encoded; a predictor maps one representation to the other.

```mermaid
flowchart LR
    x["observation x<br/>(now)"] --> Ex["encoder"] --> sx["repr. s_x"]
    y["observation y<br/>(future / masked)"] --> Ey["encoder"] --> sy["repr. s_y"]
    sx --> P["predictor"] --> psy["predicted ŝ_y"]
    psy -. "match in<br/>REPRESENTATION space" .-> sy
    classDef g fill:#0d3b2e,stroke:#10b981,color:#e6edf3;
    class P,psy g
```

Why it matters: predicting in representation space lets the encoder **throw away unpredictable, irrelevant detail** (every leaf, every ripple) and keep only what's useful, avoiding the trap of generative models that must render every pixel. It is **self-supervised** (the signal is the world's own structure, no labels, no reward) and is formulated as an **[energy-based model](https://arxiv.org/abs/2306.02572)**. Meta's [I-JEPA / V-JEPA line](https://arxiv.org/pdf/2506.09985) shows the same principle scaling to images and video, including **planning** from learned video world-models.

**Hierarchical JEPA (H-JEPA)** stacks this: lower levels keep fine detail for short-horizon prediction; higher levels form coarser, abstract representations for **longer-horizon planning**.

### 2. A cognitive architecture built around that world-model

LeCun frames the whole agent as a set of differentiable modules organized around the world-model:

```mermaid
flowchart TB
    percept["Perception<br/>(estimate current world state)"] --> wm
    mem["Short-term memory"] --> wm
    wm["World Model (JEPA)<br/>imagine future states<br/>under candidate actions"] --> cost
    cost["Cost module<br/>Intrinsic Cost (built-in drives)<br/>+ trainable Critic"] --> actor
    actor["Actor<br/>Mode 1: reactive policy<br/>Mode 2: plan through the world-model"]
    actor -- "candidate actions" --> wm
    cfg["Configurator (controller)"] -.-> percept
    cfg -.-> wm
    cfg -.-> cost
    cfg -.-> actor
    classDef wmc fill:#0d3b2e,stroke:#10b981,color:#e6edf3;
    class wm wmc
```

- **Perception** turns observations into a world-state estimate.
- The **World Model** predicts how that state evolves under *imagined* actions.
- The **Cost module** = an immutable **Intrinsic Cost** (the agent's built-in drives / "motivations") plus a **trainable Critic** that predicts future cost. *Behavior is driven by intrinsic motivation, not an external reward.*
- The **Actor** produces actions in two modes: **Mode 1** : a fast **reactive** policy; **Mode 2** : **deliberation**, by optimizing an action sequence *through the world-model* to minimize predicted future cost.
- The **Configurator** orchestrates the rest for the task at hand.

A key detail: the world-model stays **self-supervised** and reward-free — it's the general substrate. Reward/cost only ever touches the *fast* modules (critic, actor). And a good agent keeps **both** Mode 1 and Mode 2: the reflex for speed, deliberation for the hard cases, and — over a lifetime — **the reflex is trained to imitate the deliberation** (amortized inference).

---

## Sylvan: a concrete, embodied instantiation

Sylvan takes those ideas and makes them **run** on a physical body, on a laptop CPU. It is a six-legged creature in a physics simulation (Godot + PyTorch) that must keep **hunger and thirst** above zero by finding food (red) and water (blue).

The design separates **what to do** (learned, body-agnostic) from **how to move the limbs** (body-specific), with a compact 2-D **command** `(v, ω)` = *(forward speed, turn rate)* as the interface between them.

```mermaid
flowchart TB
    subgraph BRAIN["③ DECISION — command space (v, ω)  ·  LeCun's Actor"]
        M1["<b>Mode 1 — reflex</b><br/>fast learned reactive policy<br/>(drive-symmetric)"]
        M2["<b>Mode 2 — deliberation</b><br/>plans (v,ω) rollouts<br/>through the world-model"]
    end
    subgraph WM["② WORLD-MODEL (JEPA) : frozen, self-supervised  ·  LeCun's World Model"]
        P["learned retina<br/>(red=food, blue=water)"]
        I["imagines futures in<br/>latent space + object slot"]
    end
    subgraph BODY["① BODY : hexapod (the 'how to move')"]
        CPG["hand-coded CPG<br/>(tripod gait: walks + turns by construction)"]
        RES["bounded PPO residual<br/>(balance + propulsion while turning)"]
    end
    BRAIN -- "(v, ω)" --> WM
    WM --> BODY
    BODY -. "proprioception + retina" .-> WM
```

| LeCun's blueprint | Sylvan's instantiation |
|---|---|
| World Model (JEPA), self-supervised | a frozen world-model trained by latent-space prediction; perceives via a **learned retina**; imagines futures + an object-centric **slot** |
| Intrinsic Cost / built-in drives | **hunger & thirst** as body properties; the survival value is *learned*, not hand-designed |
| Actor : Mode 1 & Mode 2 | **Mode 2** = a receding-horizon planner that dreams `(v,ω)` rollouts through the world-model; **Mode 1** = a fast **drive-symmetric** reactive policy |
| Mode 2 → Mode 1 consolidation | the reflex is trained to **imitate** the planner (behavioral cloning) |

**① Body.** A hexapod whose gait is a **hand-coded CPG** (walks and steers *by construction*), corrected by a **small bounded PPO residual** for balance and speed. Turning is kinematics, not a reward gradient, which sidesteps the classic "the turn freezes" failure of end-to-end legged RL. *(≈ 3× the speed of the earlier quadruped, 0 % falls.)*

**② World-model.** Trained **self-supervised by prediction in latent space** : the slow, general substrate. It perceives through a **learned retina** (color-gated depth rays; no oracle) and can **imagine** how the world evolves under a command. It stays **frozen and task-agnostic**: the reward never touches it.

**③ Decision** happens in command space, as a **dual process**: **Mode 2** dreams candidate `(v, ω)` sequences through the world-model and picks the one that best keeps the agent alive (this is the "thinking" agent, it arbitrates food vs. water *with foresight*); **Mode 1** is a fast reflex that approximates it. **Drive-symmetric**: adding a drive = plugging in a token, no retrain.

---

## Method : diagnose before you train

A hard rule of this project: **never launch an hours-long run on a hunch.** Every expensive step is gated behind a **free, falsifiable diagnostic** that localizes the bottleneck first, with success/kill criteria written *before* running. Negative results are first-class findings, documented, not buried.

A representative slice of the current frontier (multi-drive arbitration):

| Gate (free) | Question | Verdict |
|---|---|---|
| death-cause | Why does the reflex die? | **96 % decisions** (not the motor, refuted on data) |
| G1 | Does the world-model's *dreamed* latent carry water? | ✅ yes, and it transports through the dream |
| G2 | Can a survival-value on that latent *arbitrate*? | ❌ predicts survival, but the open-loop dream is direction-blind → points to an object-slot |
| bridge | Does a "panic and defer to deliberation" trigger rescue the reflex? | ❌ too late, motivates a *principled* (uncertainty/surprise) trigger |

Each honest negative *shaped* the architecture, and each is a commit with the probe that produced it.

## Status

- ✅ **Body**: hexapod walks/turns, banked.
- ✅ **World-model + planner**: the agent perceives → imagines → navigates → **eats/drinks → survives** (single- and multi-drive).
- ✅ **Mode 2 (deliberation)** is a coherent, foresightful arbiter today.
- 🔬 **Current frontier**: making the arbitration **learned and pure** (a survival-value / object-slot replacing the hand-designed planner cost), and wiring the **Mode 1 ↔ Mode 2 bridge** with a principled trigger.

A **research prototype**, meant to *investigate* emergent embodied cognition — it wears its open problems on its sleeve.

## Honesty notes (kept deliberately visible)

- **"JEPA" here is *functional*, not doctrinaire**: the world-model was de-collapsed and shifted toward latent-space prediction (VICReg + a latent loss), but it is not a strict JEPA. The property that matters — a self-supervised, frozen substrate — is preserved.
- The current multi-drive planner still tracks the **drive levels analytically** (a hand-coded piece). The pure version, a **learned drive-dynamics head** on the latent, is a *named* debt on the roadmap, not swept under the rug.

## Repository layout

```
godot/        # physics + environment (Godot 4, Jolt): hexapod body, drives, resources
python/sylvan/  # the brain — models/ (world-model, retina, value/slot heads) + control/ (CPG+residual, PPO, planner, Mode-1 policy)
diagnostics/  # the free, falsifiable probes (diag_*.py) that gate every run
scripts/      # run / train / evaluate scripts
docs/         # design docs (the "why" behind each decision) + the blueprint
tools/        # the live architecture map + the visualization scripts
```

The full pipeline runs on **CPU**. Heavy artifacts (checkpoints, replay buffers) are regenerable and git-ignored, so a fresh clone is a **readable showcase**, not a one-command reproduction.

---

<p align="center"><i>A solo research project exploring emergent, embodied cognition in a learned world-model.</i></p>
