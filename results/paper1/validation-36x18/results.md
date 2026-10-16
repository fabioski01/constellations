# Connected maneuver validation on a 36 by 18 grid

This unchanged E2 +50 km experiment requests 648 targets, of which 648 have nominal solutions and 648 complete the numerical convergence checks. There are 0 numerical failures.

The corresponding latitude crossing is inside the original longitude band for 385 requests (59.4136%). Including the requirement that insertion is complete, 15 (2.3148%) are accepted at the matched event. The longer post-insertion window contains acceptable access for 439 requests (67.7469%); that separate quantity does not validate the planned response time.

Longitude prediction error is propagated minus nominal longitude at corresponding crossings. Its mean is -0.00547985 deg, MAE 0.03012580 deg, RMS 0.04778561 deg, 95th absolute percentile 0.13119150 deg, and maximum absolute error 0.13817000 deg. This differs from the propagated longitude offset from the target.

The matched latitude-crossing timing MAE is 0.495022 s. Among extended-window successes, the median delay beyond prediction is 23.612661 h. The mean 15-node closure is 0.555512 km.

Every target retains a record, including rejected or unresolved cases. Primary rates count target coordinates equally; spherical cell-weighted rates are separate. No empirical margin, correction or additional force has been introduced. This single configuration and request state do not establish an all-configuration success rate.
