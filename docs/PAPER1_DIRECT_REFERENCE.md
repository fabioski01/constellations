# Direct transfer reference and controlled execution diagnostics

These utilities extend the existing Paper 1 branch without modifying the
archived nominal planner, Cartesian force model, impulse law, or datasets.

## Direct strategy

`reconfig_direct_reference.py` independently searches an initial wait, one
Hohmann transfer, and a regional coast. It shares the declared secular rates,
radial WGS84 surface geometry, and continuous GMST with the primary study.
Finite wait/coast caps bound the phase wraps. Phase eliminates the regional
coast, and bracketed longitude-band boundaries define admissible wait times.
Candidates receive strict forward acceptance. This is a reference physical
strategy, not a reproduction of the complete method in a cited paper.

`paper1_compare_direct.py` requires the privately retained completed
`validation-sensitivity-plus100` campaign. It pairs the fixed +100 km setting
with exact regional coincidence, checks initial-state and constraint matching,
and recomputes every direct-reference response independently. It exports the
paired table, input/code hashes, descriptive results, and one comparison plot.
The direct strategy retains the original initial-wait cap; the other coast
cap applies to regional coasting. Comparisons distinguish nominal response,
commanded outbound delta-V, full matched-event acceptance, and later access.

```bash
python paper1_compare_direct.py \
  --data-dir results/paper1/validation-sensitivity-plus100 \
  --output-dir results/paper1/direct-reference-comparison
```

## Controlled diagnostics

`paper1_diagnose_execution.py` chooses latitude rows 0, 8 and 17 and longitude
columns 3 and 12 from the common grid before inspecting outcomes. It compares
target-independent initial numerical averages over one and three nominal
nodal periods. The maneuver commands remain fixed; no target-error correction
or state reset is introduced. Each execution receives the existing tighter
integration comparison and retains the original event and insertion criteria.

Free-coast controls omit all impulses. Local free-orbit numerical means at
circular interfaces identify where orbital discrepancies appear. The exact
scheduled burn epochs and explicit before/after sides must be used: a rounded
CSV epoch can select the opposite side of a velocity discontinuity.

```bash
python paper1_diagnose_execution.py \
  --data-dir results/paper1/validation-sensitivity-plus100 \
  --output-dir results/paper1/mean-convention-diagnostics
python -m unittest discover -s tests -v
```

An averaging-window comparison does not establish a theory-specific
mean-to-osculating transformation. Interface diagnostics are not an additive
causal error budget. These prescribed coordinates and one request state do
not establish operational probabilities or performance over all request states.

The result datasets and manuscript files are retained separately from the
source-code branch. Reproduction requires the original private campaign.
