# services/generative_agent/application/analyzers/pattern.py
from .pattern_helpers import _identify_patterns, _find_correlations


async def process_pattern_analysis(
    responses: list[str], metadata: Dict[str, Any]
) -> AnalysisResponse:
    patterns = await _identify_patterns(responses)
    correlations = await _find_correlations(responses)
    anomalies = await _detect_anomalies(patterns)
    confidence = await _assess_pattern_confidence(patterns, anomalies)

    return AnalysisResponse(
        request_id=metadata.get("request_id"),
        analysis_type=AnalysisType.PATTERN,
        summary=_summarize_patterns(patterns, anomalies),
        detailed_analysis=_format_pattern_details(patterns, correlations),
        recommendations=_generate_pattern_recommendations(patterns, anomalies),
        confidence_score=confidence,
        metadata=metadata,
        processing_time=metadata.get("processing_time", 0),
        timestamp=datetime.now(),
    )
