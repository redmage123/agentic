 services/generative_agent/application/analyzers/financial.py
async def process_financial_analysis(responses: list[str], metadata: Dict[str, Any]) -> AnalysisResponse:
   summary = responses[0]  # Base summary from first response
   detailed = await _extract_financial_details(responses)
   recommendations = await _extract_recommendations(responses)
   confidence = await _calculate_confidence(responses)
   
   return AnalysisResponse(
       request_id=metadata.get('request_id'),
       analysis_type=AnalysisType.FINANCIAL,
       summary=summary,
       detailed_analysis=detailed,
       recommendations=recommendations,
       confidence_score=confidence,
       metadata=metadata,
       processing_time=metadata.get('processing_time', 0),
       timestamp=datetime.now()
   )
