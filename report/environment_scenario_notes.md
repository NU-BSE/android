# Environment scenario implications

Representative packages in the ARC environment collection expose several game families that stress the original object-only DRLQN interface:

- **Grid paint / pattern toggles (`gp01`, `lo01`, `sy01`)**: adjacent logical cells can render as one connected component. Correct play needs cell-level click anchors rather than one component centroid.
- **Bridge, mirror, path and construction games (`tb01`, `ml01`, `lw01`, `wl01`)**: actions frequently target empty floor/water cells. Object-local click offsets are insufficient.
- **Memory and hidden-state games (`mm01`, `ms01`)**: a four-scalar trace summary cannot retain revealed identities, deductions or action-conditioned hidden state.
- **Navigation variants (`tp*`, `ic*`, `va*`, `nw01`, `gr01`, `wk01`, `rf01`, `mo01`)**: a reusable symbolic transition model and specialist search are more action-efficient than first-contact graph expansion.
- **Simulation and exact-horizon games (`ll01`, `zq01`, `ex01`)**: the policy needs explicit time/history state and lookahead, not only the current frame.

The spatial proposal branch directly addresses the first two groups. The remaining groups motivate trace distillation, transition memory, and mechanic-specific planning before a serious leaderboard submission.
