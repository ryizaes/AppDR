# OPTIMEYE Final Defense Bug Check: All 0% / Non-Referable Results

Generated: 2026-06-21

## Executive Conclusion

The trained models and current dual-model inference pipeline are **not collapsed to a single non-referable prediction**. Direct Python inference, synchronous FastAPI inference, and the asynchronous `/analyze` route used by React Native produced matching, varied outputs on ten real held-out images.

The reproducible software defect was an unsafe unavailable/low-quality display path:

1. When image quality blocked inference, or when the dual-model bundle was unavailable, the nested result contained `dr_probability=0.0`.
2. The backend previously exposed that unavailable value as a real referable probability and could map an unavailable bundle to non-referable.
3. The frontend fell back to the nested zero and displayed `0%` instead of “Unavailable.”
4. The frontend accepted any backend returning `/health` status `ok`; it did not require `MODEL_MODE=dual_model_screening_hybrid_severity` and `dual_model_ready=true`. A stale/unready backend could therefore be treated as valid.

This combination explains how an old, stale, or artifact-unready phone/backend pairing could show repeated 0%/non-referable outputs even though the current artifacts work. The exact physical-phone session could not be replayed because no Android device was connected during this audit.

## Fix Applied

- Low-quality images now return `null` screening probability/confidence, not a fabricated zero.
- Missing dual-model artifacts now return an explicit uncertain/unavailable response, never non-referable.
- React Native now renders `Unavailable` when referable probability is absent.
- Screening labels prioritize quality, consistency, and top-level backend fields over legacy nested fallbacks.
- Backend health is accepted only when the expected defense mode is active and all dual-model artifacts are ready.
- Analysis responses are rejected if they are not from the expected dual-model backend.
- Models, thresholds, feature order, artifacts, and task-specific routing were not changed.

## Active Backend and Artifact Evidence

| Check | Result |
| --- | --- |
| Active mode | `dual_model_screening_hybrid_severity` |
| Binary artifact | Loaded |
| Binary features | 203 |
| Binary threshold | 0.20 |
| Severity artifact | Loaded |
| Severity features | 220 (203 AppDR + 17 CNN prediction features) |
| EfficientNet-B3 checkpoint | Loaded |
| `/health` dual_model_ready | `true` |

No artifact was replaced and no model was trained.

## The Ten Images Originally Supplied

The images under `C:\Users\User\Downloads\OPTIMEYE_LABELED_TRAINING_IMAGES` are genuine APTOS fundus photographs. They are not synthetic or generated. However, they came from `train_manifest.csv`, not the held-out test partition. They are valid app smoke-test images but should not be used as unbiased performance evidence.

Current direct inference on those same ten images produced:

- Two class-0 images: non-referable at 5.02% and 14.64% referable probability.
- One usable class-1 image: needs review at 65.50%; the other failed quality gating.
- One usable class-2 image: referable at 98.16%; the other failed quality gating.
- One usable class-3 image: referable at 84.84%; the other failed quality gating.
- One usable class-4 image: referable at 80.42%; the other failed quality gating.

Four of ten were rejected by the quality gate. After the fix those cases show **Uncertain screening result**, **Image quality insufficient for screening**, and **Unavailable** probability instead of a misleading 0%.

## Real Held-Out Ten-Image Results

Two genuine images per internal grade were selected from `backend/results/full_hybrid_cnn_appdr_full_training/test_manifest.csv`. Selection and paths are recorded in `selected_real_test_manifest.csv`.

| Image | True medical label | Referable probability | Final screening | Supporting severity | Consistency |
| --- | --- | ---: | --- | --- | --- |
| `942f544c4e15.png` | No apparent DR | 0.79% | Non-referable DR | No apparent DR | Aligned |
| `9c6512166557.png` | No apparent DR | 11.00% | Non-referable DR | No apparent DR | Aligned |
| `12683_right.jpeg` | Mild NPDR | 58.75% | Referable / Needs review | Mild NPDR | Screening/severity disagreement |
| `Mild_DR_156.png` | Mild NPDR | 58.75% | Referable / Needs review | Mild NPDR | Screening/severity disagreement |
| `007-4819-300.jpg` | Moderate NPDR | 85.71% | Referable DR | Moderate NPDR | Aligned |
| `007-5363-300.jpg` | Moderate NPDR | 90.45% | Referable DR | Moderate NPDR | Aligned |
| `31616ff6b53b.png` | Severe NPDR | 88.95% | Referable DR | Severe NPDR | Aligned |
| `29346_right.jpeg` | Severe NPDR | 58.76% | Uncertain screening result | Severe NPDR | Binary confidence below display threshold |
| `007-7122-400.jpg` | Proliferative DR | Unavailable | Uncertain screening result | Image quality insufficient | Quality gate |
| `Proliferate DR_182.png` | Proliferative DR | 58.75% | Referable / Needs review | Mild NPDR | Screening/severity disagreement |

The last row is a real severity under-call and remains an honest model limitation; the binary screening signal still escalates it for review.

## Feature Extraction Evidence

- Usable held-out images produced 142 to 185 nonzero values out of 203.
- NaN count was zero for all ten images.
- Feature order came from `config.FEATURE_NAMES` and matched binary metadata.
- The severity frame contained exactly 220 columns.
- CNN probabilities, logits, entropy, margin, and referable probability were populated for usable images.
- The sole all-zero feature vector belonged to the image rejected before inference by the quality gate; it did not receive a non-referable result after the fix.
- First-20 feature values and complete CNN prediction features are preserved in `direct_diagnostics.json`.

## Direct, Sync API, and Async API Comparison

| Path | Screening matches | Severity matches |
| --- | ---: | ---: |
| Direct Python vs `/analyze-sync` | 10/10 | 10/10 |
| `/analyze-sync` vs `/analyze` + `/status` | 10/10 | 10/10 |

The upload bytes, FastAPI decoding, local background queue, feature extraction, model inference, and response field serialization are consistent.

## UI and Upload Static Verification

- Native Android picker copies each selected image to a unique timestamped cache filename.
- Multipart upload supplies URI, filename, and MIME type.
- Selecting a new image clears the previous analysis.
- The current result screen reads top-level screening and medical severity fields.
- Missing probabilities now display `Unavailable`; `0%` is shown only when the backend returns a genuine numeric zero for a completed inference.
- The app refuses stale/unready backend modes instead of silently showing their results.
- Numeric class names and technical model names remain hidden from the normal result screen.

Physical-device UI output was not captured because `adb devices` showed no connected phone. Reinstall/rebuild the current app before the next phone test; an older installed APK will not contain this fix.

## Consistency Rule Verification

Unit tests confirm:

- Binary non-referable + moderate-or-worse severity escalates to `Referable / Needs ophthalmologist review`.
- Binary referable + early severity also requires review.
- Aligned referable and aligned non-referable cases remain unchanged.
- Low-quality and unavailable-model results are uncertain and expose no probability.

## Files Changed

- `App.tsx`
- `backend/app/pipeline.py`
- `backend/tests/test_dual_model_consistency.py`
- `backend/results/final_defense_bugcheck/all_nonreferable_bug_report.md`
- `backend/results/final_defense_bugcheck/real_test_10_images_results.csv`

Diagnostic support files in this folder record image audit, selected manifest rows, direct feature statistics, and API comparison details.

## Verification Results

| Check | Result |
| --- | --- |
| Python compile | Passed |
| Artifact load/count check | Passed: 203 / 220 / threshold 0.20 |
| Direct ten held-out images | Passed; varied outputs |
| `/health` | Passed; expected mode and artifacts ready |
| `/analyze-sync` ten images | Passed; 10/10 direct matches |
| `/analyze` asynchronous ten images | Passed; 10/10 sync matches |
| TypeScript `npx tsc --noEmit` | Passed |
| ESLint | Passed |
| Jest | Passed: 3 tests |
| Backend unit tests | Passed: 6 tests after fix |
| Metro startup | Passed on port 8091 |
| Physical Android UI | Not run; no device connected |

## Remaining Limitations

- The supplied ten images were training examples, not independent evaluation samples.
- Four supplied images failed the current quality gate; this is now represented honestly rather than as 0% risk.
- A held-out PDR image was under-graded as mild by severity; screening disagreement still escalated review.
- The physical phone must be rebuilt/reinstalled and tested against the current `0.0.0.0:8000` backend.
- These checks verify software behavior, not clinical validation.
