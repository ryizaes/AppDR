from experiments.config import FusionParams


def weighted_feature_score(features: dict[str, float | int | str], params: FusionParams) -> float:
    return float(
        (float(features["microaneurysm_count"]) * params.ma_weight)
        + (float(features["exudate_count"]) * params.exudate_weight)
        + (float(features["exudate_quadrant_count"]) * params.quadrant_weight)
        + (float(features["pathology_area_index"]) * params.pai_weight)
        + (float(features["vessel_density"]) * 100.0 * params.vessel_weight)
        + (float(features["glcm_contrast"]) * params.glcm_weight)
    )


def predict_stage(features: dict[str, float | int | str], params: FusionParams) -> int:
    score = weighted_feature_score(features, params)
    lesion_count = int(features["microaneurysm_count"]) + int(features["exudate_count"])

    if (
        score >= params.stage4_threshold
        and float(features["vessel_density"]) >= params.stage4_vessel_density
        and float(features["glcm_contrast"]) >= params.stage4_glcm_contrast
    ):
        return 4
    if lesion_count == 0 and score < params.stage1_threshold:
        return 0
    if score >= params.stage3_threshold:
        return 3
    if score >= params.stage2_threshold:
        return 2
    if score >= params.stage1_threshold or lesion_count > 0:
        return 1

    return 0


def predict_stages(
    feature_rows: list[dict[str, float | int | str]],
    params: FusionParams,
) -> list[int]:
    return [predict_stage(features, params) for features in feature_rows]
