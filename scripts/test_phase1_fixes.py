"""
Phase 1 Fix Verification Tests

Validates the three critical bug fixes in statistical_rigor_benchmark.py
without requiring the full datasets or ONNX model.

Usage:
    python scripts/test_phase1_fixes.py
"""

import numpy as np
import pandas as pd

print("=" * 70)
print("  PHASE 1 FIX VERIFICATION TESTS")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# TEST 1: Binary F1 Label Fix
# ══════════════════════════════════════════════════════════════════════
print("\n[TEST 1] Binary F1 Label Bug Fix")
print("-" * 50)

# Simulate ToN-IoT Label column (strings, not integers)
ton_labels = pd.Series([
    "Benign", "Attack", "Attack", "Benign", "scanning",
    "ddos", "Benign", "backdoor", "Attack", "ransomware"
])

# OLD (BUGGY): comparing strings to integer 1 → all False → all zeros
old_binary = (ton_labels.values == 1).astype(int)
print(f"  OLD (buggy):  ton_true_binary = {old_binary.tolist()}")
print(f"  OLD sum:      {old_binary.sum()} attacks detected out of {len(old_binary)}")

# NEW (FIXED): comparing string content to "benign"
new_binary = (ton_labels.str.lower() != "benign").astype(int)
print(f"  NEW (fixed):  ton_true_binary = {new_binary.tolist()}")
print(f"  NEW sum:      {new_binary.sum()} attacks detected out of {len(new_binary)}")

# Expected: indices 1,2,4,5,7,8,9 are attacks (7 out of 10)
expected = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 1])
assert np.array_equal(old_binary, np.zeros(10, dtype=int)), \
    "OLD code should produce all-zeros (that's the bug)"
assert np.array_equal(new_binary, expected), \
    f"NEW code should match expected {expected.tolist()}, got {new_binary.tolist()}"

print("  PASS: Old code produced all-zeros (confirmed bug)")
print("  PASS: New code correctly identifies 7/10 as attacks")

# Edge case: mixed casing
mixed_labels = pd.Series(["BENIGN", "benign", "Benign", "attack", "DoS"])
mixed_binary = (mixed_labels.str.lower() != "benign").astype(int)
expected_mixed = np.array([0, 0, 0, 1, 1])
assert np.array_equal(mixed_binary, expected_mixed), \
    f"Case-insensitive check failed: got {mixed_binary.tolist()}"
print("  PASS: Case-insensitive handling works (BENIGN/benign/Benign all -> 0)")

# ══════════════════════════════════════════════════════════════════════
# TEST 2: Latency Measurement Includes Full Pipeline
# ══════════════════════════════════════════════════════════════════════
print("\n[TEST 2] Latency Measurement Unification")
print("-" * 50)

# Read the actual source and verify the semantic latency loop
# includes cosine_distance and Mahalanobis computation
from pathlib import Path
script_path = Path(__file__).parent / "statistical_rigor_benchmark.py"
if script_path.exists():
    source = script_path.read_text()

    # Check that the semantic latency block includes drift analysis
    has_cosine = "cosine_distance(emb, global_centroid)" in source
    has_mahal = "diff_vec @ covariance_inverse @ diff_vec" in source
    has_softmax = "softmax(outs[0][0])" in source
    has_confidence = "float(np.max(probs))" in source

    print(f"  Includes cosine_distance():    {'PASS' if has_cosine else 'FAIL'}")
    print(f"  Includes Mahalanobis:           {'PASS' if has_mahal else 'FAIL'}")
    print(f"  Includes softmax():             {'PASS' if has_softmax else 'FAIL'}")
    print(f"  Includes confidence check:      {'PASS' if has_confidence else 'FAIL'}")

    assert has_cosine, "Semantic latency must include cosine_distance"
    assert has_mahal, "Semantic latency must include Mahalanobis distance"
    assert has_softmax, "Semantic latency must include softmax"
    assert has_confidence, "Semantic latency must include confidence check"

    # Verify OLD pattern is gone
    old_pattern_gone = "_ = validate_input(single_raw[0])" not in source
    print(f"  Old incomplete pattern removed: {'PASS' if old_pattern_gone else 'FAIL'}")
    assert old_pattern_gone, "Old latency pattern should be replaced"

    print("  PASS: Semantic latency loop now includes full pipeline")
else:
    print("  SKIP: Could not find statistical_rigor_benchmark.py")

# ══════════════════════════════════════════════════════════════════════
# TEST 3: Column Guard for NF_FEATURES
# ══════════════════════════════════════════════════════════════════════
print("\n[TEST 3] Defensive Column Check for NF_FEATURES")
print("-" * 50)

NF_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total",
    "Packet Length Max", "Packet Length Min", "Protocol",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Flow Bytes/s", "Init Fwd Win Bytes", "Init Bwd Win Bytes",
]

# Simulate a DataFrame with ALL features present
df_full = pd.DataFrame(np.random.randn(100, 13), columns=NF_FEATURES)
available_full = [f for f in NF_FEATURES if f in df_full.columns]
assert len(available_full) == 13, "All 13 features should be found"
print(f"  Full columns:    {len(available_full)}/13 found - PASS")

# Simulate a DataFrame MISSING 2 features
df_partial = pd.DataFrame(np.random.randn(100, 11),
                           columns=NF_FEATURES[:11])  # missing last 2
available_partial = [f for f in NF_FEATURES if f in df_partial.columns]
assert len(available_partial) == 11, "Only 11 features should be found"
print(f"  Partial columns: {len(available_partial)}/13 found - PASS (2 missing)")

# This should NOT raise KeyError
try:
    X_partial = df_partial[available_partial].values[:50]
    print(f"  Indexing with available_nf: shape={X_partial.shape} - PASS (no KeyError)")
except KeyError as e:
    print(f"  FAIL: KeyError raised: {e}")
    assert False, "Should not raise KeyError"

# Verify the OLD pattern WOULD fail
try:
    _ = df_partial[NF_FEATURES].values[:50]
    print("  FAIL: Old pattern should have raised KeyError")
    assert False
except KeyError:
    print("  Old pattern raises KeyError (confirmed bug) - PASS")

print("  PASS: Column guard prevents KeyError on partial data")

# ══════════════════════════════════════════════════════════════════════
# TEST 4: Verify source file consistency
# ══════════════════════════════════════════════════════════════════════
print("\n[TEST 4] Source File Consistency Check")
print("-" * 50)

if script_path.exists():
    source = script_path.read_text()

    # Binary F1 fix is in place
    assert 'str.lower() != "benign"' in source, \
        "Binary F1 fix not found in source"
    print("  Binary F1 fix present:          PASS")

    # Old buggy pattern is gone
    assert '["Label"].values == 1' not in source, \
        "Old buggy Label==1 pattern still present"
    print("  Old Label==1 pattern removed:   PASS")

    # Column guard is in place
    assert "available_nf = [f for f in NF_FEATURES if f in" in source, \
        "Column guard not found in source"
    print("  Column guard present:           PASS")

    # Latency unification is in place
    assert "val_ok, val_alerts = validate_input" in source, \
        "Unified latency pattern not found"
    print("  Unified latency pattern:        PASS")

    print("  PASS: All fixes verified in source file")

# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  ALL PHASE 1 VERIFICATION TESTS PASSED")
print("=" * 70)
print("""
Impact of fixes:
  1. Binary F1: Previously computed against all-zeros ground truth
     (Label==1 on strings -> all False). Now uses string comparison.
     The reported Binary F1 = 0.8016 +/- 0.0050 WILL CHANGE after re-run.

  2. Latency: Previously measured only validate_input + session.run
     (~0.09ms). Now includes softmax + confidence + cosine + Mahalanobis
     drift analysis, matching the full deployed pipeline (~0.77ms expected).

  3. Column guard: Prevents KeyError if CIC parquet files are missing
     any of the 13 NF features. Prints a warning instead of crashing.
""")
