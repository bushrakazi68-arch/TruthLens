"""
Offline unit tests for TruthLens.

These need no network, no Neo4j and no LLM: the LLM, the graph database and the
vector store are replaced with stand-ins, so every result here is deterministic
and re-runnable by anyone. They test the decisions the agents make, not the
availability of third-party services.

Run from backend/:

    python -m unittest tests.test_unit -v
"""

import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
import unittest
from types import SimpleNamespace
from unittest import mock

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)
os.environ.setdefault("CHROMA_PATH", os.path.join(BACKEND, "chroma_db"))

from agents import claim_detector, narrative_drafter                    # noqa: E402
from agents.spread_predictor import predict_spread                      # noqa: E402
from graph import neo4j_client as nc                                    # noqa: E402
from ingestion import ingestion_manager as im                           # noqa: E402
from storage import claim_index, vector_store                           # noqa: E402
from text_utils import normalize_text                                   # noqa: E402
from workflow import graph as wf                                        # noqa: E402


# --------------------------------------------------------------------------
# 1. Ingestion: suspicion ranking, top-N cap, dedup, fault isolation
# --------------------------------------------------------------------------
HOAX = ("BREAKING!!! Scientists EXPOSED: the vaccine contains a microchip "
        "they don't want you to know about, share before deleted")
MUNDANE = "City council approves new budget for road maintenance next quarter"


class TestSuspicionScore(unittest.TestCase):
    def test_TC01_conspiratorial_text_scores_high(self):
        self.assertGreaterEqual(im._suspicion_score(HOAX), 0.8)

    def test_TC02_mundane_text_scores_zero(self):
        self.assertEqual(im._suspicion_score(MUNDANE), 0.0)

    def test_TC03_ranking_order_and_bounds(self):
        scores = [im._suspicion_score(t) for t in (MUNDANE, "Leaked memo: miracle cure", HOAX)]
        self.assertEqual(scores, sorted(scores))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in scores))

    def test_TC04_empty_input_is_safe(self):
        self.assertEqual(im._suspicion_score(""), 0.0)
        self.assertEqual(im._suspicion_score(None), 0.0)


def _post(text, platform="news"):
    return {"text": text, "platform": platform, "source": platform, "url": "http://x/" + text[:5]}


class TestFetchAllPosts(unittest.TestCase):
    def setUp(self):
        im._seen_hashes.clear()
        self.news = [_post(HOAX), _post(MUNDANE), _post("Local team wins the match"),
                     _post("Miracle cure the secret they don't want you to know")]
        self.rss = [_post("  " + MUNDANE.upper() + "  ", "rss"),     # duplicate of a news item
                    _post("Leaked: stolen election was rigged", "rss")]

    def _run(self, **kw):
        with mock.patch.object(im, "fetch_news_articles", return_value=self.news), \
             mock.patch.object(im, "fetch_youtube_videos", side_effect=RuntimeError("quota exceeded")), \
             mock.patch.object(im, "fetch_rss_articles", return_value=self.rss):
            return im.fetch_all_posts(**kw)

    def test_TC05_top_n_cap_keeps_most_suspicious(self):
        posts = self._run(max_total=3)
        self.assertEqual(len(posts), 3)
        scores = [p["suspicion_score"] for p in posts]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertIn(HOAX, [p["text"] for p in posts])
        self.assertNotIn("Local team wins the match", [p["text"] for p in posts])

    def test_TC06_duplicates_removed_case_and_whitespace_insensitive(self):
        posts = self._run(max_total=0)
        texts = [p["text"].lower() for p in posts]
        self.assertEqual(texts.count(MUNDANE.lower()), 1)

    def test_TC07_one_failing_source_does_not_stop_ingestion(self):
        posts = self._run(max_total=0)           # YouTube raises inside _run
        self.assertGreater(len(posts), 0)

    def test_TC08_every_post_is_stamped(self):
        for p in self._run(max_total=0):
            self.assertIn("suspicion_score", p)
            self.assertIn("ingested_at", p)


# --------------------------------------------------------------------------
# 2. Text normalisation (multilingual-safe)
# --------------------------------------------------------------------------
class TestNormalizeText(unittest.TestCase):
    def test_TC09_replacement_char_removed(self):
        self.assertNotIn("�", normalize_text("Bangladesh�s vaccination campaign"))

    def test_TC10_typographic_punctuation_flattened(self):
        self.assertEqual(normalize_text("“hello” — it’s"), '"hello" - it\'s')

    def test_TC11_devanagari_is_preserved(self):
        hindi = "टीका कैंसर का कारण बनता है"
        self.assertEqual(normalize_text(hindi), unicodedata.normalize("NFKC", hindi))

    def test_TC12_whitespace_collapsed(self):
        self.assertEqual(normalize_text("a   b \n\t c"), "a b c")

    def test_TC13_empty_is_safe(self):
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")


# --------------------------------------------------------------------------
# 3. Claim Detector routing (LLM stubbed with canned responses)
# --------------------------------------------------------------------------
def _llm(misinfo="true", conf="0.95", sev="9", claim="Bleach cures COVID",
         category="public_health", techniques="fabricated authority, false urgency"):
    return (f"MISINFORMATION: {misinfo}\nCONFIDENCE: {conf}\nSEVERITY: {sev}\n"
            f"CLAIM: {claim}\nRATIONALE: because\nHARM: poisoning\n"
            f"TECHNIQUES: {techniques}\nKEYWORDS: bleach, cures\nCATEGORY: {category}")


def detect(canned, text="some post"):
    with mock.patch.object(claim_detector, "complete", return_value=canned):
        return claim_detector.detect_claim(text)


class TestDetectorRouting(unittest.TestCase):
    THR = claim_detector.REVIEW_SEVERITY_THRESHOLD

    def test_TC14_severe_and_confident_goes_to_human_review(self):
        r = detect(_llm(sev=str(self.THR), conf="0.95"))
        self.assertEqual(r.status, "review_required")

    def test_TC15_just_below_threshold_is_not_escalated(self):
        r = detect(_llm(sev=str(self.THR - 1), conf="0.95"))
        self.assertEqual(r.status, "pending")

    def test_TC16_severe_but_unsure_is_analysed_not_binned(self):
        r = detect(_llm(sev="9", conf="0.5"))
        self.assertEqual(r.status, "pending")

    def test_TC17_not_misinformation_is_capped_and_dropped(self):
        r = detect(_llm(misinfo="false", sev="8", conf="0.9"))
        self.assertFalse(r.is_misinformation)
        self.assertLessEqual(r.severity, 3)
        self.assertEqual(r.status, "dropped")

    def test_TC18_low_stakes_low_certainty_is_dropped(self):
        self.assertEqual(detect(_llm(conf="0.3", sev="3")).status, "dropped")

    def test_TC19_harmful_but_unsure_is_kept(self):
        self.assertEqual(detect(_llm(conf="0.3", sev="7")).status, "pending")

    def test_TC20_out_of_range_numbers_are_clamped(self):
        r = detect(_llm(sev="99", conf="7"))
        self.assertEqual(r.severity, 10)
        self.assertEqual(r.confidence, 1.0)

    def test_TC21_garbage_llm_output_does_not_crash(self):
        r = detect("lorem ipsum, no structure at all", text="original post text")
        self.assertFalse(r.is_misinformation)
        self.assertEqual(r.severity, 1)
        self.assertEqual(r.claim, "original post text")     # falls back to the input
        self.assertEqual(r.status, "dropped")

    def test_TC22_empty_llm_output_after_rate_limit_does_not_crash(self):
        r = detect("", text="post")
        self.assertEqual(r.status, "dropped")

    def test_TC23_unknown_category_falls_back_to_other(self):
        self.assertEqual(detect(_llm(category="aliens")).category, "other")

    def test_TC24_none_techniques_become_empty_list(self):
        self.assertEqual(detect(_llm(techniques="none")).techniques, [])

    def test_TC25_techniques_are_parsed_and_deduplicated(self):
        r = detect(_llm(techniques="false urgency, false urgency, emotional appeal"))
        self.assertEqual(r.techniques, ["false urgency", "emotional appeal"])


# --------------------------------------------------------------------------
# 4. Spread Predictor: bounds, monotonicity, explainability
# --------------------------------------------------------------------------
TL3 = [{"platform": "youtube", "timestamp": "2026-09-01T00:00:00Z"},
       {"platform": "rss", "timestamp": "2026-09-02T00:00:00Z"},
       {"platform": "news", "timestamp": "2026-09-03T00:00:00Z"}]


def spread(**kw):
    base = dict(severity=6, suspicion_score=0.3, is_misinformation=True,
                detection_confidence=0.8, outlet_count=2, timeline=TL3[:2])
    base.update(kw)
    return predict_spread("claim", **base)


class TestSpreadPredictor(unittest.TestCase):
    def test_TC26_outputs_within_bounds(self):
        for sev in (1, 5, 10):
            for n in (0, 3, 9):
                r = spread(severity=sev, outlet_count=n)
                self.assertTrue(0.0 <= r.risk_score <= 1.0)
                self.assertTrue(0.8 <= r.r0 <= 5.0)
                self.assertTrue(0.0 <= r.reduction_pct <= 100.0)
                self.assertLessEqual(r.projected_6h_contained, r.projected_6h_uncontained)

    def test_TC27_higher_severity_means_higher_risk(self):
        self.assertGreater(spread(severity=9).risk_score, spread(severity=3).risk_score)

    def test_TC28_more_outlets_means_higher_risk(self):
        self.assertGreater(spread(outlet_count=8).risk_score, spread(outlet_count=1).risk_score)

    def test_TC29_higher_suspicion_means_higher_risk(self):
        self.assertGreater(spread(suspicion_score=1.0).risk_score,
                           spread(suspicion_score=0.0).risk_score)

    def test_TC30_not_misinformation_scales_risk_down_by_60_percent(self):
        yes, no = spread(is_misinformation=True), spread(is_misinformation=False)
        self.assertAlmostEqual(no.risk_score, yes.risk_score * 0.4, delta=0.002)

    def test_TC31_deterministic(self):
        a, b = spread(), spread()
        self.assertEqual((a.risk_score, a.r0, a.current_reach_est, a.drivers),
                         (b.risk_score, b.r0, b.current_reach_est, b.drivers))

    def test_TC32_velocity_from_real_timestamps(self):
        # 3 sightings over 2 days -> 1.5 outlets/day
        self.assertAlmostEqual(spread(timeline=TL3, outlet_count=3).velocity_outlets_per_day, 1.5)

    def test_TC33_burst_within_hours_is_boosted(self):
        burst = [{"platform": "x", "timestamp": f"2026-09-01T00:0{i}:00Z"} for i in range(3)]
        self.assertEqual(spread(timeline=burst, outlet_count=3).velocity_outlets_per_day, 12.0)

    def test_TC34_rss_style_timestamps_parse(self):
        tl = [{"platform": "rss", "timestamp": "Tue, 01 Sep 2026 00:00:00 +0000"},
              {"platform": "rss", "timestamp": "Thu, 03 Sep 2026 00:00:00 +0000"}]
        self.assertAlmostEqual(spread(timeline=tl, outlet_count=2).velocity_outlets_per_day, 1.0)

    def test_TC35_unparseable_timestamps_are_ignored_not_fatal(self):
        r = spread(timeline=[{"platform": "rss", "timestamp": "not a date"}], outlet_count=1)
        self.assertEqual(r.velocity_outlets_per_day, 0.0)

    def test_TC36_empty_timeline_still_works(self):
        r = spread(timeline=[], outlet_count=0)
        self.assertEqual(r.status, "predicted")
        self.assertGreater(r.current_reach_est, 0)

    def test_TC37_reach_uses_per_platform_audience(self):
        r = spread(timeline=[{"platform": "news", "timestamp": "2026-09-01T00:00:00Z"}], outlet_count=1)
        self.assertEqual(r.current_reach_est, int(60000 * (0.5 + r.risk_score)))

    def test_TC38_graph_node_count_matches_structure(self):
        r = spread(timeline=TL3, outlet_count=3)
        self.assertEqual(r.predicted_reach, 3 + 1 + int(round(r.risk_score * 4)))

    def test_TC39_drivers_explain_the_score(self):
        r = spread(timeline=TL3, outlet_count=3)
        contributions = [float(x) for x in re.findall(r"\+([0-9.]+) of", " ".join(r.drivers))]
        self.assertAlmostEqual(sum(contributions), r.risk_score, delta=0.006)
        self.assertEqual(contributions, sorted(contributions, reverse=True))   # ranked

    def test_TC40_zero_contribution_is_called_out(self):
        r = spread(suspicion_score=0.0)
        self.assertTrue(any("Ingest suspicion" in d and "contributed nothing" in d for d in r.drivers))


# --------------------------------------------------------------------------
# 5. Narrative Drafter: grounded or silent
# --------------------------------------------------------------------------
class TestNarrativeDrafter(unittest.TestCase):
    def test_TC41_no_evidence_means_no_generation(self):
        with mock.patch.object(narrative_drafter, "search_documents",
                               return_value={"documents": [[]], "ids": [[]]}), \
             mock.patch.object(narrative_drafter, "complete",
                               side_effect=AssertionError("LLM must NOT be called without evidence")) as llm:
            r = narrative_drafter.draft_narrative("an unfounded claim")
        self.assertEqual(r.status, "no_evidence")
        self.assertEqual(r.sources, [])
        llm.assert_not_called()

    def test_TC42_prompt_is_constrained_to_retrieved_evidence(self):
        docs = {"documents": [["WHO says X is safe."]], "ids": [["who-1"]]}
        with mock.patch.object(narrative_drafter, "search_documents", return_value=docs), \
             mock.patch.object(narrative_drafter, "complete", return_value="X is safe, per WHO.") as llm:
            r = narrative_drafter.draft_narrative("X is dangerous")
        prompt = llm.call_args[0][0]
        self.assertIn("Use ONLY the following retrieved evidence", prompt)
        self.assertIn("WHO says X is safe.", prompt)
        self.assertIn("Does not invent facts", prompt)
        self.assertEqual(r.status, "drafted")

    def test_TC43_citations_are_returned_with_the_output(self):
        docs = {"documents": [["d1", "d2"]], "ids": [["id-1", "id-2"]]}
        with mock.patch.object(narrative_drafter, "search_documents", return_value=docs), \
             mock.patch.object(narrative_drafter, "complete", return_value="text"):
            r = narrative_drafter.draft_narrative("claim")
        self.assertEqual(r.source_ids, ["id-1", "id-2"])
        self.assertEqual(r.sources, ["d1", "d2"])

    def test_TC44_relevance_cutoff_discards_unrelated_documents(self):
        raw = {"documents": [["near", "mid", "far"]],
               "distances": [[0.5, 1.1, 1.9]], "ids": [["i1", "i2", "i3"]]}
        with mock.patch.object(vector_store.collection, "query", return_value=raw):
            out = vector_store.search_documents("q")
        limit = vector_store.RELEVANCE_MAX_DISTANCE
        expected = [i for i, d in zip(("i1", "i2", "i3"), (0.5, 1.1, 1.9)) if d <= limit]
        self.assertEqual(out["ids"][0], expected)
        self.assertNotIn("i3", out["ids"][0])


# --------------------------------------------------------------------------
# 6. Semantic claim resolution
# --------------------------------------------------------------------------
def _collection(count, doc=None, dist=None):
    c = mock.MagicMock()
    c.count.return_value = count
    c.query.return_value = {"documents": [[doc] if doc else []],
                            "distances": [[dist] if dist is not None else []]}
    return c


class TestClaimResolution(unittest.TestCase):
    def test_TC45_paraphrase_merges_into_canonical(self):
        col = _collection(1, "5G towers cause cancer", 0.38)
        with mock.patch.object(claim_index, "claim_collection", col):
            text, is_new, dist = claim_index.resolve_claim("5G radiation causes cancer in humans")
        self.assertEqual((text, is_new), ("5G towers cause cancer", False))
        col.add.assert_not_called()

    def test_TC46_distinct_claim_stays_separate_and_is_registered(self):
        col = _collection(1, "vaccines cause autism", 0.80)
        with mock.patch.object(claim_index, "claim_collection", col):
            text, is_new, _ = claim_index.resolve_claim("vaccines cause infertility")
        self.assertEqual((text, is_new), ("vaccines cause infertility", True))
        col.add.assert_called_once()

    def test_TC47_threshold_boundary(self):
        limit = claim_index.MAX_DISTANCE
        with mock.patch.object(claim_index, "claim_collection", _collection(1, "canon", limit)):
            self.assertFalse(claim_index.resolve_claim("x")[1])          # at the line: merge
        with mock.patch.object(claim_index, "claim_collection", _collection(1, "canon", limit + 0.01)):
            self.assertTrue(claim_index.resolve_claim("x")[1])           # past it: separate

    def test_TC48_empty_claim_is_safe(self):
        self.assertEqual(claim_index.resolve_claim("  "), ("", True, 0.0))

    def test_TC49_index_failure_never_blocks_the_pipeline(self):
        col = mock.MagicMock()
        col.count.side_effect = RuntimeError("chroma down")
        with mock.patch.object(claim_index, "claim_collection", col):
            text, is_new, _ = claim_index.resolve_claim("a claim")
        self.assertEqual((text, is_new), ("a claim", True))


# --------------------------------------------------------------------------
# 7. LangGraph routing: which agents actually run for which verdict
# --------------------------------------------------------------------------
def _detection(status, sev=9, conf=0.95, misinfo=True, claim="canonical claim"):
    return claim_detector.DetectionResult(
        claim=claim, confidence=conf, is_misinformation=misinfo, severity=sev, status=status,
        rationale="r", techniques=["t"], keywords=["k"], harm="h", category="public_health")


ORIGIN = SimpleNamespace(
    source="BBC", confidence=0.7, status="traced", first_seen="2026-09-01T00:00:00Z",
    outlet_count=2, summary="s",
    timeline=[{"outlet": "BBC", "platform": "rss", "timestamp": "2026-09-01T00:00:00Z"},
              {"outlet": "CNN", "platform": "news", "timestamp": "2026-09-02T00:00:00Z"}])
DRAFT = SimpleNamespace(narrative="n", confidence=0.8, status="drafted",
                        sources=["doc"], source_ids=["id1"])


def run_graph(detection, resolved=None):
    """Invoke the real compiled graph with I/O stubbed; return (result, audit actions)."""
    actions = []
    resolved = resolved or (detection.claim, True, 0.0)
    patches = dict(
        detect_claim=mock.Mock(return_value=detection),
        resolve_claim=mock.Mock(return_value=resolved),
        create_claim_node=mock.Mock(),
        upsert_claim=mock.Mock(),
        add_claim_alias=mock.Mock(),
        create_audit_event=mock.Mock(side_effect=lambda cid, action, d: actions.append(action)),
        trace_origin=mock.Mock(return_value=ORIGIN),
        draft_narrative=mock.Mock(return_value=DRAFT),
    )
    with mock.patch.multiple(wf, **patches):
        result = wf.misinformation_graph.invoke({"text": "post", "platform": "rss"})
    return result, actions, patches


class TestWorkflowRouting(unittest.TestCase):
    def test_TC50_dropped_claim_stops_after_detection(self):
        result, actions, p = run_graph(_detection("dropped", sev=1, conf=0.2, misinfo=False))
        self.assertEqual(actions, ["claim_dropped"])
        p["trace_origin"].assert_not_called()
        p["draft_narrative"].assert_not_called()
        self.assertNotIn("origin_status", result)

    def test_TC51_pending_claim_runs_all_four_agents_without_review(self):
        _, actions, _ = run_graph(_detection("pending", sev=6))
        self.assertEqual(actions, ["claim_detected", "origin_traced",
                                   "spread_predicted", "narrative_drafted"])

    def test_TC52_review_required_runs_full_analysis_THEN_the_human_gate(self):
        result, actions, _ = run_graph(_detection("review_required"))
        self.assertEqual(actions, ["claim_detected", "origin_traced", "spread_predicted",
                                   "narrative_drafted", "review_required"])
        self.assertEqual(result["detection_status"], "review_required")
        self.assertIn("origin_status", result)             # analysis was not skipped

    def test_TC53_state_is_enriched_stage_by_stage(self):
        result, _, _ = run_graph(_detection("pending", sev=6))
        for key in ("detection_confidence", "origin_status", "risk_score",
                    "spread_r0", "narrative_status", "origin_outlet_count"):
            self.assertIn(key, result)
        self.assertEqual(result["origin_outlet_count"], 2)

    def test_TC54_spread_agent_consumes_the_origin_agents_timeline(self):
        result, _, _ = run_graph(_detection("pending", sev=6))
        self.assertAlmostEqual(result["spread_velocity"], 2.0, places=1)   # 2 outlets / 1 day

    def test_TC55_semantic_variant_is_merged_and_audited(self):
        det = _detection("pending", sev=6, claim="5G radiation causes cancer")
        _, actions, p = run_graph(det, resolved=("5G towers cause cancer", False, 0.38))
        self.assertEqual(actions[0], "claim_merged")
        p["add_claim_alias"].assert_called_once_with("5G towers cause cancer",
                                                     "5G radiation causes cancer")

    def test_TC56_route_functions_directly(self):
        self.assertEqual(wf.route_after_detection({"detection_status": "dropped"}), "end")
        self.assertEqual(wf.route_after_detection({"detection_status": "pending"}), "trace")
        self.assertEqual(wf.route_after_detection({"detection_status": "review_required"}), "trace")
        self.assertEqual(wf.route_after_narrative({"detection_status": "review_required"}), "review")
        self.assertEqual(wf.route_after_narrative({"detection_status": "pending"}), "end")


# --------------------------------------------------------------------------
# 8. SHA-256 tamper-evident audit chain
# --------------------------------------------------------------------------
class FakeSession:
    def __init__(self, rows): self.rows = rows
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def run(self, *a, **k): return list(self.rows)


class FakeDriver:
    def __init__(self, rows): self.rows = rows
    def session(self): return FakeSession(self.rows)


CLAIM = "bleach cures covid"
STEPS = [("claim_detected", "severity 10/10"), ("origin_traced", "BBC first"),
         ("spread_predicted", "risk 0.62"), ("narrative_drafted", "from 2 sources")]


def build_chain(claim=CLAIM, steps=STEPS):
    """Rows exactly as the real Cypher query would return them."""
    prev, rows = nc.GENESIS_HASH, []
    for seq, (action, details) in enumerate(steps):
        ts = f"2026-09-10T10:00:0{seq}+00:00"
        digest = nc._event_hash(prev, claim, action, details, ts, seq)
        rows.append(dict(action=action, details=details, timestamp=ts, ts_iso=ts,
                         seq=seq, prev_hash=prev, hash=digest))
        prev = digest
    return rows


class TestAuditChainHash(unittest.TestCase):
    ARGS = (nc.GENESIS_HASH, CLAIM, "claim_detected", "details", "2026-09-10T10:00:00+00:00", 0)

    def test_TC57_digest_is_a_64_char_sha256_hex(self):
        self.assertRegex(nc._event_hash(*self.ARGS), r"^[0-9a-f]{64}$")

    def test_TC58_matches_an_independent_sha256_of_canonical_json(self):
        prev, cid, action, details, ts, seq = self.ARGS
        payload = json.dumps({"seq": seq, "claim_id": cid, "action": action, "details": details,
                              "ts_iso": ts, "prev_hash": prev},
                             sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.assertEqual(nc._event_hash(*self.ARGS),
                         hashlib.sha256(payload.encode("utf-8")).hexdigest())

    def test_TC59_deterministic(self):
        self.assertEqual(nc._event_hash(*self.ARGS), nc._event_hash(*self.ARGS))

    def test_TC60_changing_any_single_field_changes_the_digest(self):
        base = nc._event_hash(*self.ARGS)
        variants = [("0" * 63 + "1",), (None, "other claim"), (None, None, "claim_dropped"),
                    (None, None, None, "edited"), (None, None, None, None, "2026-09-10T10:00:01+00:00"),
                    (None, None, None, None, None, 1)]
        for v in variants:
            args = list(self.ARGS)
            for i, val in enumerate(v):
                if val is not None:
                    args[i] = val
            self.assertNotEqual(nc._event_hash(*args), base, f"field change not detected: {v}")

    def test_TC61_one_character_edit_flips_about_half_the_bits(self):
        a = nc._event_hash(*self.ARGS)
        args = list(self.ARGS); args[3] = "detailt"
        b = nc._event_hash(*args)
        differing = sum(x != y for x, y in zip(a, b))
        self.assertGreater(differing, 40)          # of 64 hex digits (avalanche effect)

    def test_TC62_genesis_sentinel(self):
        self.assertEqual(nc.GENESIS_HASH, "0" * 64)


class TestAuditChainVerification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        self.tmp.close()
        self.addCleanup(lambda: os.path.exists(self.tmp.name) and os.remove(self.tmp.name))
        p = mock.patch.object(nc, "_CHAIN_TIP_LOG", self.tmp.name); p.start(); self.addCleanup(p.stop)

    def verify(self, rows):
        with mock.patch.object(nc, "driver", FakeDriver(rows)):
            return nc.verify_claim_chain(CLAIM)

    def log_tip(self, rows):
        last = rows[-1]
        with open(self.tmp.name, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts_iso": last["ts_iso"], "claim_id": CLAIM,
                                 "seq": last["seq"], "hash": last["hash"]}) + "\n")

    def test_TC63_untouched_chain_verifies_intact(self):
        rows = build_chain()
        r = self.verify(rows)
        self.assertEqual((r["status"], r["verified"], r["events_chained"]), ("intact", True, 4))
        self.assertEqual(r["tip_hash"], rows[-1]["hash"])
        self.assertEqual(r["algorithm"], "SHA-256 per-claim hash chain")

    def test_TC64_edited_details_detected_at_the_exact_event(self):
        rows = build_chain(); rows[1]["details"] = "EDITED AFTER THE FACT"
        r = self.verify(rows)
        self.assertEqual((r["status"], r["broken_at_seq"]), ("broken", 1))
        self.assertIn("contents", r["reason"])

    def test_TC65_edited_action_detected(self):
        rows = build_chain(); rows[2]["action"] = "claim_dropped"
        self.assertEqual(self.verify(rows)["broken_at_seq"], 2)

    def test_TC66_edited_hashed_timestamp_detected(self):
        rows = build_chain(); rows[3]["ts_iso"] = "2030-01-01T00:00:00+00:00"
        self.assertEqual(self.verify(rows)["broken_at_seq"], 3)

    def test_TC67_timestamp_altered_in_database_detected(self):
        rows = build_chain(); rows[2]["timestamp"] = "2031-05-05T05:05:05+00:00"
        r = self.verify(rows)
        self.assertEqual(r["broken_at_seq"], 2)
        self.assertIn("timestamp", r["reason"])

    def test_TC68_deleted_middle_event_detected(self):
        rows = build_chain(); del rows[1]
        r = self.verify(rows)
        self.assertEqual(r["status"], "broken")
        self.assertIn("previous-hash link", r["reason"])

    def test_TC69_edit_invalidates_everything_after_it_not_just_itself(self):
        rows = build_chain(); rows[0]["details"] = "edited first event"
        self.assertEqual(self.verify(rows)["broken_at_seq"], 0)   # first break reported

    def test_TC70_tail_truncation_is_missed_by_hash_links_alone(self):
        # Documents WHY the external tip log exists: with no log, deleting the
        # newest events leaves a perfectly valid-looking shorter chain.
        rows = build_chain()[:2]
        r = self.verify(rows)
        self.assertEqual(r["status"], "intact")
        self.assertFalse(r["tip_log_checked"])

    def test_TC71_tail_truncation_caught_by_external_tip_log(self):
        full = build_chain(); self.log_tip(full)
        r = self.verify(full[:2])                       # attacker deletes the last 2 events
        self.assertEqual(r["status"], "broken")
        self.assertIn("tail truncated", r["reason"])
        self.assertTrue(r["tip_log_checked"])

    def test_TC72_consistently_rewritten_tail_caught_by_external_tip_log(self):
        original = build_chain(); self.log_tip(original)
        forged = build_chain(steps=STEPS[:3] + [("narrative_drafted", "FORGED but re-hashed")])
        r = self.verify(forged)                         # every link + hash is internally valid
        self.assertEqual(r["status"], "broken")
        self.assertIn("tail rewritten", r["reason"])

    def test_TC73_intact_chain_passes_the_tip_log_cross_check(self):
        rows = build_chain(); self.log_tip(rows)
        r = self.verify(rows)
        self.assertEqual(r["status"], "intact")
        self.assertTrue(r["tip_log_checked"])

    def test_TC74_stale_tip_from_a_wiped_claim_does_not_cause_a_false_alarm(self):
        old = build_chain(steps=STEPS + [("review_approved", "old chain, seq 4")])
        self.log_tip(old)                                # old, longer history in the log
        fresh = build_chain(steps=[("claim_detected", "re-ingested after reset")])
        self.log_tip(fresh)                              # newest log line = the new chain
        self.assertEqual(self.verify(fresh)["status"], "intact")

    def test_TC75_legacy_unhashed_events_are_reported_not_failed(self):
        legacy = [dict(action="old", details="pre-hashing", timestamp="2026-01-01T00:00:00",
                       ts_iso=None, seq=None, prev_hash=None, hash=None)] * 2
        r = self.verify(legacy + build_chain())
        self.assertEqual((r["status"], r["events_unchained_legacy"]), ("intact", 2))

    def test_TC76_all_legacy_is_not_chained_never_mistaken_for_verified(self):
        legacy = [dict(action="old", details="x", timestamp="2026-01-01T00:00:00",
                       ts_iso=None, seq=None, prev_hash=None, hash=None)]
        r = self.verify(legacy)
        self.assertEqual((r["status"], r["verified"]), ("not_chained", False))

    def test_TC77_no_events_at_all(self):
        self.assertEqual(self.verify([])["status"], "not_chained")


class RecordingDriver:
    """Minimal in-memory stand-in for the two queries create_audit_event runs."""
    def __init__(self): self.events = []

    class _Result:
        def __init__(self, row): self.row = row
        def single(self): return self.row

    def session(self): return self

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def run(self, query, **kw):
        if "CREATE INDEX" in query:
            return self._Result(None)
        if "RETURN a.hash AS hash" in query:
            mine = [e for e in self.events if e["claim_id"] == kw["claim_id"]]
            tip = max(mine, key=lambda e: e["seq"]) if mine else None
            return self._Result({"hash": tip["hash"], "seq": tip["seq"]} if tip else None)
        if "CREATE (a:AuditEvent" in query:
            self.events.append(dict(claim_id=kw["claim_id"], action=kw["action"],
                                    details=kw["details"], ts_iso=kw["ts_iso"], seq=kw["seq"],
                                    prev_hash=kw["prev_hash"], hash=kw["hash"],
                                    timestamp=kw["timestamp"].isoformat()))
        return self._Result(None)


class TestAuditChainWriting(unittest.TestCase):
    def test_TC78_create_audit_event_links_each_event_to_the_previous_one(self):
        drv = RecordingDriver()
        tmp = tempfile.NamedTemporaryFile(suffix=".log", delete=False); tmp.close()
        self.addCleanup(os.remove, tmp.name)
        with mock.patch.object(nc, "driver", drv), mock.patch.object(nc, "_CHAIN_TIP_LOG", tmp.name), \
             mock.patch.object(nc, "_audit_index_ready", True):
            digests = [nc.create_audit_event(CLAIM, a, d) for a, d in STEPS]
            ev = drv.events
            self.assertEqual([e["seq"] for e in ev], [0, 1, 2, 3])
            self.assertEqual(ev[0]["prev_hash"], nc.GENESIS_HASH)
            for earlier, later in zip(ev, ev[1:]):
                self.assertEqual(later["prev_hash"], earlier["hash"])
            self.assertEqual(digests, [e["hash"] for e in ev])

            logged = [json.loads(l) for l in open(tmp.name, encoding="utf-8")]
            self.assertEqual([l["hash"] for l in logged], digests)       # tip mirrored outside the DB

            with mock.patch.object(nc, "driver", FakeDriver(ev)):         # written chain re-verifies
                self.assertEqual(nc.verify_claim_chain(CLAIM)["status"], "intact")


# --------------------------------------------------------------------------
# 9. PyTorch is optional: the lightweight deploy image runs without it
# --------------------------------------------------------------------------
import subprocess  # noqa: E402

from agents import spread_predictor as sp  # noqa: E402

_GRID = [
    dict(severity=s, suspicion_score=sus, is_misinformation=mis, detection_confidence=c,
         outlet_count=n, timeline=tl)
    for s in (1, 5, 10) for sus in (0.0, 0.7) for mis in (True, False)
    for c in (0.3, 0.95) for n, tl in ((0, []), (2, TL3[:2]), (3, TL3))
]

# Runs in a fresh interpreter in which torch / torch_geometric cannot be imported
# (setting a sys.modules entry to None makes `import` raise ImportError).
_NO_TORCH_PROBE = """
import sys, json
sys.modules['torch'] = None
sys.modules['torch_geometric'] = None
sys.modules['torch_geometric.data'] = None
from agents import spread_predictor as sp
tl = [{"platform": "youtube", "timestamp": "2026-09-01T00:00:00Z"},
      {"platform": "rss", "timestamp": "2026-09-02T00:00:00Z"}]
r = sp.predict_spread("c", severity=9, suspicion_score=0.5, is_misinformation=True,
                      detection_confidence=0.9, outlet_count=2, timeline=tl)
print(json.dumps({"backend": sp.GRAPH_BACKEND, "risk": r.risk_score,
                  "reach": r.predicted_reach, "r0": r.r0}))
"""

_NO_TORCH_BOOT = """
import sys
sys.modules['torch'] = None
sys.modules['torch_geometric'] = None
sys.modules['torch_geometric.data'] = None
import main                                   # the whole FastAPI app
from agents import spread_predictor as sp
print("BOOTED", sp.GRAPH_BACKEND, "torch_loaded=" + str("torch" in sys.modules and sys.modules["torch"] is not None))
"""


def _run_without_torch(code):
    return subprocess.run([sys.executable, "-c", code], cwd=BACKEND,
                          capture_output=True, text=True, timeout=240)


class TestTorchIsOptional(unittest.TestCase):
    @unittest.skipUnless(sp.GRAPH_BACKEND == "pytorch_geometric", "PyTorch Geometric not installed here")
    def test_TC79_plain_and_pytorch_backends_return_identical_results(self):
        for kw in _GRID:
            with_pyg = predict_spread("claim", **kw)
            with mock.patch.object(sp, "GRAPH_BACKEND", "plain"):
                plain = predict_spread("claim", **kw)
            self.assertEqual(with_pyg, plain, f"backends disagree for {kw}")

    def test_TC80_spread_predictor_imports_and_runs_with_torch_blocked(self):
        r = _run_without_torch(_NO_TORCH_PROBE)
        self.assertEqual(r.returncode, 0, r.stderr[-600:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(out["backend"], "plain")
        tl = [{"platform": "youtube", "timestamp": "2026-09-01T00:00:00Z"},
              {"platform": "rss", "timestamp": "2026-09-02T00:00:00Z"}]
        expected = predict_spread("c", severity=9, suspicion_score=0.5, is_misinformation=True,
                                  detection_confidence=0.9, outlet_count=2, timeline=tl)
        self.assertEqual((out["risk"], out["reach"], out["r0"]),
                         (expected.risk_score, expected.predicted_reach, expected.r0))

    def test_TC81_whole_backend_boots_with_torch_blocked(self):
        r = _run_without_torch(_NO_TORCH_BOOT)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn("BOOTED plain", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
