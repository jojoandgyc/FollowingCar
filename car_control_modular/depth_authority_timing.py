"""Supported physical-age ceiling for an existing forward Depth grant.

This is not a default lease, a fresh-measurement window or motion permission.
Each configured grant retains its own original deadline. New PI observations
and reverse grants still require their separate 180 ms qualification; ROI,
identity and encoder clocks are independent of this ceiling.
"""

MAX_FORWARD_DEPTH_TTL_SEC = .30
