"""Configuration consistency: every plan names known candidates, profiles, SLOs and pricing; candidate hashes are
stable; the bundled schema validator agrees with the reference jsonschema package where it is installed."""
import json
import unittest

from helpers import HARNESS, make_run, simple_finite

from lqh import schema_lite
from lqh.util import config_hash, read_json


class Config(unittest.TestCase):
    def setUp(self):
        self.campaign = read_json(HARNESS / "config" / "campaign.json")
        self.candidates = {c["candidate_id"]: c for c in read_json(HARNESS / "config" / "candidates.json")["candidates"]}
        self.profiles = {p["profile_id"]: p for p in read_json(HARNESS / "config" / "profiles.json")["profiles"]}
        self.slos = {s["slo_id"] for s in read_json(HARNESS / "config" / "slo.json")["slos"]}

    def test_plans_reference_known_things(self):
        for plan_id, plan in self.campaign["plans"].items():
            for cand in plan["candidates"]:
                self.assertIn(cand, self.candidates, f"{plan_id}: unknown candidate {cand}")
            for point in plan["points"]:
                self.assertIn(point["profile"], self.profiles, f"{plan_id}: unknown profile")
                self.assertIn(point["lane"], ("finite", "sustained", "open_loop", "infx_upstream"))
            self.assertIn(plan.get("slo_id", self.campaign["default_slo_id"]), self.slos)

    def test_profile_versions_are_pinned_in_the_campaign(self):
        for profile_id, version in self.campaign["profile_versions"].items():
            self.assertEqual(self.profiles[profile_id]["version"], version)

    def test_candidate_ids_unique_and_hashes_stable(self):
        ids = [c["candidate_id"] for c in read_json(HARNESS / "config" / "candidates.json")["candidates"]]
        self.assertEqual(len(ids), len(set(ids)))
        hashes = {cid: config_hash(c["config"]) for cid, c in self.candidates.items()}
        self.assertEqual(len(set(hashes.values())), len(hashes), "two candidates share one configuration")
        self.assertEqual(hashes, {cid: config_hash(json.loads(json.dumps(c["config"]))) for cid, c in self.candidates.items()})

    def test_same_capacity_candidates_differ_only_in_state_settings(self):
        bf16 = self.candidates["bf16_window"]["config"]
        lq = self.candidates["lq_combined"]["config"]
        self.assertEqual(bf16["configured_slots"], lq["configured_slots"])
        self.assertEqual(bf16["max_running_requests"], lq["max_running_requests"])

    def test_decision_rule_and_margin_are_declared(self):
        rule = self.campaign["quality_qualification"]
        self.assertEqual(rule["margin_points"], 1.0)
        self.assertIn("EVERY task", rule["decision_rule_frozen"])
        self.assertEqual(rule["sampling"]["presence_penalty"], 1.5)


class Schemas(unittest.TestCase):
    def test_records_validate_and_agree_with_jsonschema(self):
        reqs, phases = simple_finite()
        run = make_run(reqs, phases)
        manifest = json.loads((run / "run_manifest.json").read_text())
        records = [json.loads(l) for l in (run / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(schema_lite.check(manifest, "run_manifest"), [])
        for record in records:
            self.assertEqual(schema_lite.check(record, "request"), [])
        bad = dict(records[0], status="maybe")
        self.assertTrue(schema_lite.check(bad, "request"))
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed (the bundled validator is used)")
        jsonschema.validate(manifest, schema_lite.load_schema("run_manifest"))
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema_lite.load_schema("request"))


if __name__ == "__main__":
    unittest.main()
