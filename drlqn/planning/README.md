# Spatial proposal design

`generate_click_candidates` keeps the 24-slot click budget but fills it from four sources: connected objects, inferred logical tile centres, recent changed regions, and a coarse lattice. Each slot retains the existing five centre/cardinal-offset lambda operators.

This preserves the 128-way Q-router and replay schema while covering actions that the original connected-component representation could not express. The proposal score is only a weak exploration prior; learned Q-values remain authoritative.
