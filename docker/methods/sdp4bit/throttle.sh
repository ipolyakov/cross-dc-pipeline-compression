#!/usr/bin/env bash
# Limit the outgoing bandwidth of a network interface of this host, for the PP runs of run_pp.sh.
# Run it on both hosts with the same rate, so that both directions are limited:
#
#   ./throttle.sh IFACE RATE    e.g. ./throttle.sh enp1s0 500mbit   (1gbit, 200mbit, 100mbit, ...)
#   ./throttle.sh IFACE off     remove the limit
#   ./throttle.sh IFACE show    current setting and counters
#
# The token bucket has a small burst (256kb): a large burst lets the activation messages of PP
# (several MB each) through at line rate and hides the limit. Check with iperf3 between the hosts.

set -euo pipefail
IFACE=${1:?interface, see ip -br addr}
RATE=${2:?rate like 500mbit, or off, or show}

case "$RATE" in
    off)
        sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true
        ;;
    show)
        ;;
    *)
        if [[ ! "$RATE" =~ ^[0-9]+(k|m|g)bit$ ]]; then
            echo "rate like 500mbit or 1gbit expected, got '$RATE'" >&2
            exit 1
        fi
        sudo tc qdisc replace dev "$IFACE" root tbf rate "$RATE" burst 256kb latency 50ms
        ;;
esac
tc -s qdisc show dev "$IFACE"
