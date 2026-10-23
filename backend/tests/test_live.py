"""
Live integration tests for TruthLens.

These exercise the real stack end to end: FastAPI on :8000, Groq, Neo4j and
ChromaDB. They are skipped automatically, with the reason, if the API or the
database cannot be reached, so they never produce a false failure on a machine
that simply is not set up.

Run from backend/ with the API running (python -m uvicorn main:app --port 8000):

    python -m unittest tests.test_live -v

Every claim these tests create is deleted again in tearDownClass.
"""

import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
import urllib.request
import uuid
from unittest import mock

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)
os.environ.setdefault("CHROMA_PATH", os.path.join(BACKEND, "chroma_db"))

API = os.getenv("TRUTHLENS_API", "http://127.0.0.1:8000")

HOAX = ("Drinking bleach cures COVID-19 and doctors are hiding it from the public, "
        "share this before it gets deleted")


def call(path, method="GET", timeout=240):
    req = urllib.request.Request(f"{API}{path}", method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def q(text):
    return urllib.parse.quote(text, safe="")


class LiveBase(unittest.TestCase):
    created_claims: list = []

    @classmethod
    def setUpClass(cls):
        try:
            call("/health", timeout=10)
        except Exception as e:
            raise unittest.SkipTest(f"API not reachable at {API} ({e}). "
                                    "Start it: python -m uvicorn main:app --port 8000")
        try:
            from graph.neo4j_client import driver
            driver.verify_connectivity()
        except Exception as e:
            raise unittest.SkipTest(f"Neo4j unreachable ({type(e).__name__}: {str(e)[:120]})")
        cls.created_claims = []

    @classmethod
    def tearDownClass(cls):
        try:
            from graph.neo4j_client import driver
            from storage.claim_index import claim_collection, _key
            with driver.session() as s:
                for text in cls.created_claims:
                    s.run("MATCH (c:Claim {text:$t}) DETACH DELETE c", t=text)
                    s.run("MATCH (a:AuditEvent {claim_id:$t}) DELETE a", t=text)
                    s.run("MATCH (r:Review {claim_id:$t}) DELETE r", t=text)
                    try:
                        claim_collection.delete(ids=[_key(text)])
                    except Exception:
                        pass
        except Exception as e:  # pragma: no cover
            print(f"[cleanup] {e}")


class TestLiveSystem(LiveBase):
    def test_L01_health(self):
        self.assertEqual(call("/health")["status"], "ok")

    def test_L02_sources_report_configuration(self):
        by_id = {s["id"]: s for s in call("/api/sources")["sources"]}
        for sid in ("newsapi", "youtube", "rss"):
            self.assertIn(sid, by_id)
            self.assertTrue(by_id[sid]["live"])
        self.assertTrue(by_id["rss"]["configured"])          # RSS needs no key

    def test_L03_rag_evidence_store_is_populated(self):
        self.assertGreater(call("/api/evidence").get("count", 0), 0,
                           "empty RAG store: the drafter can only ever say no_evidence")

    def test_L04_neutral_fact_is_not_flagged(self):
        r = call(f"/api/claims/detect?text={q('Paris is the capital of France.')}", "POST")
        self.assertFalse(r["is_misinformation"])
        self.assertEqual(r["status"], "dropped")


class TestLivePipeline(LiveBase):
    """One real claim through all four agents, then the audit trail and review gate."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        res = call(f"/api/ingest?text={q(HOAX)}&source=manual", "POST")
        cls.res = res
        cls.pr = res["pipeline_result"]
        cls.claim = cls.pr.get("claim") or HOAX
        cls.created_claims.append(cls.claim)

    def test_L05_detector_flags_an_obvious_hoax(self):
        pr = self.pr
        self.assertTrue(pr["is_misinformation"])
        self.assertTrue(0.0 <= pr["detection_confidence"] <= 1.0)
        self.assertTrue(1 <= pr["severity"] <= 10)
        self.assertIn(pr["detection_status"], ("review_required", "pending"))

    def test_L06_all_downstream_agents_ran(self):
        pr = self.pr
        self.assertIsNotNone(pr.get("origin_status"))
        self.assertEqual(pr.get("spread_status"), "predicted")
        self.assertIn(pr.get("narrative_status"), ("drafted", "no_evidence"))

    def test_L07_spread_numbers_are_internally_consistent(self):
        pr = self.pr
        self.assertTrue(0.0 <= pr["risk_score"] <= 1.0)
        self.assertLessEqual(pr["spread_proj_contained"], pr["spread_proj_uncontained"])
        self.assertTrue(json.loads(pr["spread_drivers"]))

    def test_L08_narrative_is_grounded_or_absent(self):
        pr = self.pr
        if pr["narrative_status"] == "drafted":
            self.assertTrue(json.loads(pr["rag_sources"]), "a drafted correction must carry its sources")
        else:
            self.assertEqual(pr["narrative_status"], "no_evidence")

    def test_L09_audit_chain_is_intact_after_a_real_run(self):
        v = call(f"/api/claims/{q(self.claim)}/verify-chain")
        self.assertEqual(v["status"], "intact")
        self.assertGreaterEqual(v["events_chained"], 4)
        self.assertEqual(v["algorithm"], "SHA-256 per-claim hash chain")

    def test_L10_human_review_decision_is_recorded_and_chained(self):
        before = call(f"/api/claims/{q(self.claim)}/verify-chain")["events_chained"]
        call(f"/api/claims/{q(self.claim)}/review?approved=true", "POST")
        got = call(f"/api/claims/{q(self.claim)}/review")
        self.assertEqual(got["status"], "approved")
        after = call(f"/api/claims/{q(self.claim)}/verify-chain")
        self.assertEqual(after["status"], "intact")
        self.assertGreater(after["events_chained"], before)


class TestLiveAuditTamperDetection(LiveBase):
    """Edits a REAL Neo4j audit trail and checks the verifier notices."""

    def setUp(self):
        from graph import neo4j_client as nc
        self.nc = nc
        self.claim = f"TEST-AUDIT-{uuid.uuid4().hex[:10]}"
        self.created_claims.append(self.claim)
        tmp = tempfile.NamedTemporaryFile(suffix=".log", delete=False); tmp.close()
        self.addCleanup(os.remove, tmp.name)
        p = mock.patch.object(nc, "_CHAIN_TIP_LOG", tmp.name); p.start(); self.addCleanup(p.stop)
        for action, details in [("claim_detected", "sev 9"), ("origin_traced", "BBC"),
                                ("spread_predicted", "risk .6"), ("narrative_drafted", "2 docs")]:
            nc.create_audit_event(self.claim, action, details)

    def _run(self, cypher, **kw):
        with self.nc.driver.session() as s:
            s.run(cypher, claim=self.claim, **kw)

    def test_L11_fresh_chain_verifies(self):
        v = self.nc.verify_claim_chain(self.claim)
        self.assertEqual((v["status"], v["events_chained"], v["tip_log_checked"]), ("intact", 4, True))

    def test_L12_editing_a_stored_event_in_neo4j_is_detected(self):
        self._run("MATCH (a:AuditEvent {claim_id:$claim, seq:1}) SET a.details = 'rewritten'")
        v = self.nc.verify_claim_chain(self.claim)
        self.assertEqual((v["status"], v["broken_at_seq"]), ("broken", 1))

    def test_L13_deleting_the_newest_events_is_detected_via_the_external_log(self):
        self._run("MATCH (a:AuditEvent {claim_id:$claim}) WHERE a.seq >= 2 DELETE a")
        v = self.nc.verify_claim_chain(self.claim)
        self.assertEqual(v["status"], "broken")
        self.assertIn("tail truncated", v["reason"])


class TestLiveSemanticDedup(LiveBase):
    def test_L14_paraphrases_collapse_onto_one_canonical_claim(self):
        a = call(f"/api/ingest?text={q('5G towers cause cancer in people living nearby')}&source=manual", "POST")
        b = call(f"/api/ingest?text={q('Radiation from 5G masts is giving residents cancer')}&source=manual", "POST")
        ca, cb = a["pipeline_result"].get("claim"), b["pipeline_result"].get("claim")
        self.created_claims.extend([c for c in (ca, cb) if c])
        self.assertEqual(ca, cb, "paraphrases should resolve to the same canonical claim")


class TestLiveBulkIngest(LiveBase):
    def test_L15_bulk_ingest_respects_the_top_n_cap(self):
        out = call("/api/ingest/bulk?max_total=3&include_news=false&include_youtube=false&include_rss=true",
                   "POST", timeout=600)
        self.assertLessEqual(out["total_ingested"], 3)
        for r in out["results"]:
            self.created_claims.append((r.get("pipeline_result") or {}).get("claim") or "")
            self.assertEqual(r["post"]["platform"], "rss")
            self.assertIn("suspicion_score", r["post"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
