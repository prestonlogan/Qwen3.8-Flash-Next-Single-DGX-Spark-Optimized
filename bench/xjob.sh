#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of Qwen3.8-Flash-Next-Single-DGX-Spark-Optimized (builds on MiaAI-Lab's recipe).
# Minimal tracked-job helper (replaces `pgrep -f` waits). State lives in jobs/<name>/:
#   cmd, pid (= process-group id), stime (/proc starttime for identity), t0, log, rc (written by the job itself on exit)
# usage: xjob.sh start <name> <command...>   | wait <name> <timeout_s> | status <name> | kill <name>
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); J=${Q38_JOBS:-$ROOT/.state/jobs}
op=${1:?op}; name=${2:?name}; D=$J/$name
ident() { [ -r /proc/$1/stat ] && awk '{print $22}' /proc/$1/stat; }
alive() { local p; p=$(cat $D/pid 2>/dev/null) || return 1; [ "$(ident $p)" = "$(cat $D/stime)" ]; }
report() {
  local p; p=$(cat $D/pid 2>/dev/null); echo "job=$name pid=$p cmd=$(cat $D/cmd)"
  echo "elapsed=$(( $(date +%s) - $(cat $D/t0) ))s alive=$(alive && echo yes || echo no) rc=$(cat $D/rc 2>/dev/null || echo none)"
  alive && ps -o pid,stat,etime,time,args --ppid $p -p $p 2>/dev/null | cut -c1-160
  echo "--- log tail"; tail -n ${TAILN:-8} $D/log | cut -c1-300
}
case $op in
start)
  shift 2; [ -e $D/pid ] && alive && { echo "job $name already running"; exit 2; }
  rm -rf $D; mkdir -p $D; printf '%q ' "$@" > $D/cmd; date +%s > $D/t0
  cd $ROOT
  setsid nohup bash -c 'bash -c "$1" > "$2/log" 2>&1; echo $? > "$2/rc.tmp"; mv "$2/rc.tmp" "$2/rc"' _ "$(cat $D/cmd)" $D > /dev/null 2>&1 &
  p=$!; echo $p > $D/pid; sleep 0.2; ident $p > $D/stime
  echo "started job=$name pid=$p" ;;
wait)
  to=${3:?timeout_s}; end=$(( $(date +%s) + to ))
  while [ ! -e $D/rc ]; do
    if ! alive; then sleep 1; [ -e $D/rc ] && break; echo "DIED (no rc)"; report; exit 3; fi
    [ $(date +%s) -ge $end ] && { echo "TIMEOUT after ${to}s (job left running)"; report; exit 124; }
    sleep ${POLL:-15}
  done
  echo "DONE rc=$(cat $D/rc)"; report; exit $(cat $D/rc) ;;
status) report ;;
kill)
  alive && { p=$(cat $D/pid); kill -TERM -- -$p && echo "sent TERM to process group $p"; } || echo "not running" ;;
*) echo "bad op"; exit 1 ;;
esac
