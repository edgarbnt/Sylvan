#!/bin/zsh
# Gate CHERCHER paramétrable : compare survie / repas / distance-bouffe, search OFF vs ON, conditions identiques.
# Usage: bash scripts/gate_search.sh [FC=6] [DRAIN=0.12] [NEP=8] [ER=1.0] [HZ=300]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
FC=${1:-6}; DRAIN=${2:-0.12}; NEP=${3:-8}; ER=${4:-1.0}; HZ=${5:-300}
echo "### GATE CHERCHER : FC=$FC drain=$DRAIN nep=$NEP eat_radius=$ER horizon=$HZ ###"

runit() {  # $1 label  $2 search_enable
  pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
  FC=$FC SYLVAN_ENERGY_DRAIN=$DRAIN SYLVAN_SEARCH_ENABLE=$2 SYLVAN_SEARCH_LOG=1 \
    bash scripts/run_forage_latent.sh $ER $HZ $NEP > /tmp/gate_$1.out 2>&1
  cp /tmp/forage_latent.log  /tmp/gate_$1_godot.log 2>/dev/null
  cp /tmp/planner_latent.log /tmp/gate_$1_plan.log  2>/dev/null
}

analyze() {  # $1 label — survie médiane/moy/range, repas (saut énergie>5), distance bouffe moyenne, %<1.2m
  awk '/\[Godot\] Episode/{
    e="";s="";en="";fd="";
    for(i=1;i<=NF;i++){ if($i=="Episode")e=$(i+1); if($i=="Step")s=$(i+1); if($i=="Energy:")en=$(i+1); if($i=="food_d:")fd=$(i+1); }
    last[e]=s+0;
    if(e==pe && en>pen+5) meals[e]++;
    pe=e; pen=en;
    if(fd!=""){ nfd++; if(fd+0<1.2)c12++; sfd+=fd+0; }
  }
  END{
    ns=0; tm=0;
    for(k=0;k<=30;k++) if(k in last){ surv[ns++]=last[k]; tm+=(meals[k]+0); }
    for(a=0;a<ns;a++) for(b=a+1;b<ns;b++) if(surv[b]<surv[a]){t=surv[a];surv[a]=surv[b];surv[b]=t;}
    med=(ns%2)?surv[int(ns/2)]:(surv[ns/2-1]+surv[ns/2])/2; ssum=0; for(a=0;a<ns;a++)ssum+=surv[a];
    printf "  survie med=%.0f moy=%.0f [%.0f..%.0f] | repas=%d | moyFD=%.2f %%a-portee(<1.2)=%.0f\n",
      med, ssum/ns, surv[0], surv[ns-1], tm, (nfd?sfd/nfd:0), (nfd?100*c12/nfd:0);
  }' /tmp/gate_$1_godot.log
}

runit base 0
runit search 1
echo "==== RESULTATS (FC=$FC drain=$DRAIN, $NEP ep) ===="
echo "BASE   (search OFF):"; analyze base
echo "SEARCH (search ON) :"; analyze search
echo "CHERCHER transitions: base=$(grep -ac CHERCHER /tmp/gate_base_plan.log) search=$(grep -ac CHERCHER /tmp/gate_search_plan.log)"
echo "### GATE done ###"
