#!/usr/bin/env bash
# Limit the outgoing bandwidth and optionally add latency on a network interface,
# for the PP runs of run_pp.sh. Run it on both hosts with the same settings, so
# that both directions are shaped:
#
#   ./throttle.sh IFACE RATE              e.g. ./throttle.sh enp1s0 500mbit
#   ./throttle.sh IFACE RATE DELAY_MS     e.g. ./throttle.sh enp1s0 500mbit 30
#   ./throttle.sh IFACE off [DELAY_MS]    remove the limit (or keep only delay)
#   ./throttle.sh IFACE show              current setting and counters
#
# DELAY_MS is one-way egress delay. Applied on both hosts it gives about
# 2*DELAY_MS of RTT on top of the LAN. 30 ms on each side is about 60 ms RTT.
#
# The token bucket has a small burst (256kb): a large burst lets the activation
# messages of PP (several MB each) through at line rate and hides the limit.
# Check with iperf3 and ping between the hosts.

set -euo pipefail
IFACE=${1:?interface, see ip -br addr}
RATE=${2:?rate like 500mbit, or off, or show}
DELAY=${3:-}

case "$RATE" in
    off|show) ;;
    *)
        if [[ ! "$RATE" =~ ^[0-9]+(k|m|g)bit$ ]]; then
            echo "rate like 500mbit or 1gbit expected, got '$RATE'" >&2
            exit 1
        fi
        ;;
esac
if [[ -n "$DELAY" && ! "$DELAY" =~ ^[0-9]+$ ]]; then
    echo "delay must be integer milliseconds, got '$DELAY'" >&2
    exit 1
fi

sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true

if [[ "$RATE" == "show" ]]; then
    tc -s qdisc show dev "$IFACE"
    exit 0
fi

if [[ -n "$DELAY" ]]; then
    # netem as root for delay; tbf as child when a rate is set
    sudo tc qdisc add dev "$IFACE" root handle 1: netem delay "${DELAY}ms"
    if [[ "$RATE" != "off" ]]; then
        sudo tc qdisc add dev "$IFACE" parent 1:1 handle 10: \
            tbf rate "$RATE" burst 256kb latency 50ms
    fi
else
    if [[ "$RATE" != "off" ]]; then
        sudo tc qdisc add dev "$IFACE" root \
            tbf rate "$RATE" burst 256kb latency 50ms
    fi
fi

tc -s qdisc show dev "$IFACE"
