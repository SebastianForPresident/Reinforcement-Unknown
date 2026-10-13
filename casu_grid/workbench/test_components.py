from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from .components import (discover_components, resolve_reward_reference,
                         snapshot_reward_reference, validate_component)


class RewardComponentTests(unittest.TestCase):
    def _component(self, root: Path, *, body: str | None = None):
        source = root / "reward.py"
        source.write_text(body or '''
REWARD_COMPONENT = {
 "id":"test.progress", "name":"Progress", "description":"test",
 "entry_point":"reward", "contract_version":1, "required_fields":[],
 "parameter_schema":[{"name":"weight","type":"float","default":2.0,"minimum":0.0,"maximum":10.0}],
 "output":{"type":"scalar_and_terms","shape":[]}, "stateful":False,
 "deterministic":True, "synthetic_context":{"previous":{"progress":0.0},"current":{"progress":1.0}},
}
def reward(previous, current, terminated, truncated, *, weight):
 return weight, {"progress": weight}
''')
        metadata = {"id": "test.progress", "name": "Progress", "entry_point": "reward",
                    "contract_version": 1, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),"required_fields":[],"reward_terms":None,
                    "parameter_schema": [{"name": "weight", "type": "float", "default": 2.0,
                                          "minimum": 0.0, "maximum": 10.0}],
                    "output": {"type": "scalar_and_terms", "shape": []},
                    "required_fields": [], "deterministic": True}
        sidecar = Path(str(source) + ".component.json")
        sidecar.write_text(json.dumps(metadata))
        return source, metadata, sidecar

    def test_discovery_is_metadata_only_and_validation_executes_selected_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source, metadata, sidecar = self._component(root)
            found = discover_components(root)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["metadata"]["id"], "test.progress")
            result = validate_component(source, metadata, {})
            self.assertEqual(result["synthetic_result"], [2.0, {"progress": 2.0}])
            self.assertEqual(result["source_sha256"], metadata["source_sha256"])

    def test_parameter_constraints_and_nonfinite_results_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source,metadata,_=self._component(root)
            with self.assertRaisesRegex(ValueError,"<= 10.0"):
                validate_component(source,metadata,{"weight":11.0})
            source,metadata,_=self._component(root,body='''
REWARD_COMPONENT={"id":"x","name":"x","description":"x","entry_point":"reward","contract_version":1,"required_fields":[],"parameter_schema":[],"output":{"type":"scalar_and_terms","shape":[]},"stateful":False,"deterministic":True}
def reward(previous,current,terminated,truncated): return float("nan"), {}
''')
            metadata.update({"id":"x","required_fields":[],"parameter_schema":[],"reward_terms":None})
            with self.assertRaisesRegex(ValueError,"finite"):
                validate_component(source,metadata,{})

    def test_source_hash_change_invalidates_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,metadata,_=self._component(Path(tmp));source.write_text(source.read_text()+"\n# changed\n")
            with self.assertRaisesRegex(ValueError,"changed"):
                validate_component(source,metadata,{})

    def test_bad_signature_and_worker_timeout_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,metadata,_=self._component(root,body='''
REWARD_COMPONENT={"id":"x","name":"x","description":"x","entry_point":"reward","contract_version":1,"required_fields":[],"parameter_schema":[],"output":{"type":"scalar_and_terms","shape":[]},"stateful":False,"deterministic":True}
def reward(unexpected): return 1.0, {}
''')
            metadata.update({"id":"x","required_fields":[],"parameter_schema":[],"reward_terms":None})
            with self.assertRaisesRegex(ValueError,"entry point must accept"):
                validate_component(source,metadata,{})
            source,metadata,_=self._component(root,body='''
import time
REWARD_COMPONENT={"id":"x","name":"x","description":"x","entry_point":"reward","contract_version":1,"required_fields":[],"parameter_schema":[],"output":{"type":"scalar_and_terms","shape":[]},"stateful":False,"deterministic":True}
def reward(previous,current,terminated,truncated): time.sleep(2); return 0.0, {}
''')
            metadata.update({"id":"x","required_fields":[],"parameter_schema":[],"reward_terms":None})
            with self.assertRaisesRegex(ValueError,"timed out"):
                validate_component(source,metadata,{},timeout=0.1)

    def test_run_snapshot_resolves_reward_after_original_source_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,metadata,_=self._component(root)
            reference={"id":"test.progress","source":source.name,"source_sha256":metadata["source_sha256"],
                       "entry_point":"reward","contract_version":1,"parameters":{"weight":2.0},
                       "reward_terms":["progress"]}
            run=root/"run";run.mkdir()
            snapshot=snapshot_reward_reference(reference,root/"experiment.py",run)
            source.rename(root/"reward.py.hidden")
            class Config: REWARD_COMPONENT_REF=snapshot
            callbacks=resolve_reward_reference(Config,run/"research_config.py",(None,None,None))
            self.assertEqual(callbacks[2]({}, {}, False, False),(2.0,{"progress":2.0}))


if __name__ == "__main__":
    unittest.main()
