"""Controls for experimental comparability and safe evidence reuse."""
from copy import deepcopy
import json
from pathlib import Path
import gzip
import tempfile
import unittest

from paper1_validate_grid import target_grid
from paper1_validate_sensitivity import (PHYSICAL_SOURCES, build_experiments,
                                        physical_signature, regional_radius, reusable_records)

ROOT = Path(__file__).resolve().parents[1]


class SensitivityControls(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'configs/paper1_validation_sensitivity.json').read_text())
        self.base = json.loads((ROOT/self.config['scenario_template']).read_text())

    def test_reference_shared_by_all_three_sweeps_is_not_duplicated(self):
        experiments = build_experiments(self.config, self.base)
        references = [e for e in experiments if 'reference' in e['labels']]
        self.assertEqual(len(references), 1)
        self.assertEqual(set(references[0]['sweeps']), {'tolerance','drift_offset','orbit'})
        self.assertEqual(references[0]['scenario']['drift_offset_km'], 100)
        self.assertEqual(len(experiments), 21)

    def test_tolerance_sweep_changes_planner_input_and_keeps_orbit_and_offset(self):
        experiments = build_experiments(self.config, self.base)
        tolerances = []
        for e in experiments:
            if 'tolerance' in e['sweeps']:
                s = e['scenario']; tolerances.append(s['longitude_tolerance_deg'])
                self.assertEqual((s['global_altitude_km'],s['inclination_deg'],s['drift_offset_km']), (700,50,100))
        self.assertEqual(sorted(tolerances), self.config['tolerances_deg'])

    def test_screening_grid_is_an_exact_subset_of_the_refinement(self):
        coarse = target_grid(self.config)
        finer = target_grid(dict(self.config,n_longitude=36))
        fine_coordinates = {(r['target_latitude_deg'],r['target_longitude_deg']) for r in finer}
        self.assertEqual(len(coarse),324)
        self.assertTrue(all((r['target_latitude_deg'],r['target_longitude_deg']) in fine_coordinates for r in coarse))
        self.assertAlmostEqual(sum(r['area_weight'] for r in coarse),1.)

    def test_reuse_signature_ignores_metadata_but_detects_every_physical_change(self):
        modified = deepcopy(self.base)
        modified.update(case_id='different',scope='different',target_latitude_deg=-49)
        self.assertEqual(physical_signature(self.base),physical_signature(modified))
        for key, value in [('drift_offset_km',100),('longitude_tolerance_deg',1),('inclination_deg',60)]:
            modified = dict(self.base,**{key:value})
            self.assertNotEqual(physical_signature(self.base),physical_signature(modified))

    def test_regional_coincidence_is_computed_from_the_same_repeat_root(self):
        experiments = build_experiments(self.config,self.base)
        special = next(e for e in experiments if 'exact_regional_coincidence' in e['labels'])
        s = special['scenario']
        self.assertEqual(6371+s['global_altitude_km']+s['drift_offset_km'],regional_radius(s))
        self.assertAlmostEqual(regional_radius(s),6874.316367539429,places=9)

    def test_nested_reuse_changes_only_grid_metadata_and_retains_origin(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temporary:
            folder=Path(temporary)
            source_context=dict(scenario=self.base,initial_state=[1.,2.,3.,4.,5.,6.])
            source_hashes={name:'same-version' for name in PHYSICAL_SOURCES}
            old_manifest=dict(context=source_context,source_sha256=source_hashes)
            (folder/'manifest.json').write_text(json.dumps(old_manifest))
            old_target=dict(case_index=8,latitude_index=0,longitude_index=8,
                            target_latitude_deg=-49.,target_longitude_deg=-100.,area_weight=.001)
            original=dict(row=dict(old_target,status='completed',matched_time_error_s=-.25),
                          evidence=dict(physical_event='unchanged'))
            with gzip.open(folder/'case_evidence.jsonl.gz','wt') as f:f.write(json.dumps(original)+'\n')
            target=dict(old_target,case_index=4,longitude_index=4,area_weight=.002)
            current=dict(source_sha256=source_hashes,targets=[target],
                         experiments=[dict(experiment_id='new-grid',context=source_context)])
            reused=reusable_records([folder],current)[('new-grid',4)]
            self.assertEqual(reused['evidence'],original['evidence'])
            self.assertEqual(reused['row']['matched_time_error_s'],-.25)
            self.assertEqual(reused['row']['area_weight'],.002)
            self.assertEqual(reused['reused_from']['case_index'],8)
            self.assertEqual(len(reused['reused_from']['sha256']),64)
            # Initial-state or numerical-source mismatch must invalidate reuse.
            changed=deepcopy(current)
            changed['experiments'][0]['context']['initial_state'][0]=10.
            with self.assertRaises(ValueError):reusable_records([folder],changed)
            changed=deepcopy(current)
            changed['source_sha256'][PHYSICAL_SOURCES[0]]='changed-version'
            with self.assertRaises(ValueError):reusable_records([folder],changed)


if __name__ == '__main__':unittest.main()
