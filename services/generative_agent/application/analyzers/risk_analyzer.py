# services/generative_agent/application/analyzers/risk_analyzer.py

from typing import List, Dict, Any, Tuple
from datetime import datetime
import asyncio
from enum import Enum

from ...domain.models import AnalysisResponse, AnalysisType
from ...domain.interfaces import NNClient

class RiskLevel(Enum):
   LOW = "low"
   MEDIUM = "medium"
   HIGH = "high"
   CRITICAL = "critical"

class RiskCategory(Enum):
   MARKET = "market"
   VOLATILITY = "volatility"
   REGIME_CHANGE = "regime_change"
   CYCLICAL = "cyclical"
   SYSTEMIC = "systemic"

class RiskAnalyzer:
   """
   Analyzes financial risks using multiple neural networks and LLM insights
   """
   
   def __init__(
       self,
       hamiltonian_client: NNClient,
       fourier_client: NNClient,
       perturbation_client: NNClient,
       risk_threshold: float = 0.7
   ):
       self.hamiltonian_client = hamiltonian_client
       self.fourier_client = fourier_client
       self.perturbation_client = perturbation_client
       self.risk_threshold = risk_threshold

   async def analyze(
       self,
       market_data: Dict[str, Any],
       llm_responses: List[str],
       metadata: Dict[str, Any]
   ) -> AnalysisResponse:
       """
       Process risk analysis using neural networks and LLM responses
       """
       # Get risk assessments from neural networks
       nn_results = await asyncio.gather(
           self._assess_conservation_risks(market_data),  # Hamiltonian
           self._assess_cyclical_risks(market_data),      # Fourier
           self._assess_regime_risks(market_data)         # Perturbation
       )
       
       conservation_risks, cyclical_risks, regime_risks = nn_results
       
       # Extract LLM risk insights
       llm_risks = self._extract_llm_risks(llm_responses)
       
       # Combine all risk assessments
       combined_risks = await self._combine_risk_assessments(
           conservation_risks,
           cyclical_risks,
           regime_risks,
           llm_risks
       )
       
       # Calculate overall risk metrics
       risk_metrics = self._calculate_risk_metrics(combined_risks)
       
       # Generate risk analysis and recommendations
       detailed_analysis = self._generate_risk_analysis(
           combined_risks,
           risk_metrics
       )
       
       recommendations = self._generate_risk_recommendations(
           combined_risks,
           risk_metrics
       )
       
       return AnalysisResponse(
           request_id=metadata.get('request_id', ''),
           analysis_type=AnalysisType.RISK,
           summary=self._generate_risk_summary(combined_risks, risk_metrics),
           detailed_analysis=detailed_analysis,
           recommendations=recommendations,
           confidence_score=risk_metrics['confidence'],
           metadata={
               **metadata,
               'risk_counts': risk_metrics['risk_counts'],
               'highest_risk_level': risk_metrics['highest_risk'].value,
               'risk_scores': risk_metrics['risk_scores']
           },
           processing_time=metadata.get('processing_time', 0),
           timestamp=datetime.now()
       )

   async def _assess_conservation_risks(
       self,
       market_data: Dict[str, Any]
   ) -> List[Dict[str, Any]]:
       """
       Use Hamiltonian NN to assess risks based on conservation law violations
       """
       response = await self.hamiltonian_client.analyze(market_data)
       risks = []
       
       # Check energy conservation violations
       energy_violations = response.get('energy_violations', [])
       for violation in energy_violations:
           risks.append({
               'category': RiskCategory.MARKET,
               'level': self._determine_risk_level(violation['severity']),
               'description': f"Energy conservation violation: {violation['description']}",
               'severity': violation['severity'],
               'confidence': violation['confidence'],
               'source': 'hamiltonian'
           })
       
       # Check momentum anomalies
       momentum_anomalies = response.get('momentum_anomalies', [])
       for anomaly in momentum_anomalies:
           risks.append({
               'category': RiskCategory.MARKET,
               'level': self._determine_risk_level(anomaly['severity']),
               'description': f"Momentum anomaly: {anomaly['description']}",
               'severity': anomaly['severity'],
               'confidence': anomaly['confidence'],
               'source': 'hamiltonian'
           })
       
       return risks

   async def _assess_cyclical_risks(
       self,
       market_data: Dict[str, Any]
   ) -> List[Dict[str, Any]]:
       """
       Use Fourier NN to assess risks based on cyclical patterns
       """
       response = await self.fourier_client.analyze(market_data)
       risks = []
       
       # Check cycle disruptions
       cycle_disruptions = response.get('cycle_disruptions', [])
       for disruption in cycle_disruptions:
           risks.append({
               'category': RiskCategory.CYCLICAL,
               'level': self._determine_risk_level(disruption['severity']),
               'description': f"Cycle disruption: {disruption['description']}",
               'severity': disruption['severity'],
               'confidence': disruption['confidence'],
               'source': 'fourier'
           })
       
       # Check frequency anomalies
       freq_anomalies = response.get('frequency_anomalies', [])
       for anomaly in freq_anomalies:
           risks.append({
               'category': RiskCategory.CYCLICAL,
               'level': self._determine_risk_level(anomaly['severity']),
               'description': f"Frequency anomaly: {anomaly['description']}",
               'severity': anomaly['severity'],
               'confidence': anomaly['confidence'],
               'source': 'fourier'
           })
       
       return risks

   async def _assess_regime_risks(
       self,
       market_data: Dict[str, Any]
   ) -> List[Dict[str, Any]]:
       """
       Use Perturbation NN to assess risks based on regime changes
       """
       response = await self.perturbation_client.analyze(market_data)
       risks = []
       
       # Check regime transitions
       regime_transitions = response.get('regime_transitions', [])
       for transition in regime_transitions:
           risks.append({
               'category': RiskCategory.REGIME_CHANGE,
               'level': self._determine_risk_level(transition['severity']),
               'description': f"Regime transition: {transition['description']}",
               'severity': transition['severity'],
               'confidence': transition['confidence'],
               'source': 'perturbation'
           })
       
       # Check stability risks
       stability_risks = response.get('stability_risks', [])
       for risk in stability_risks:
           risks.append({
               'category': RiskCategory.SYSTEMIC,
               'level': self._determine_risk_level(risk['severity']),
               'description': f"Stability risk: {risk['description']}",
               'severity': risk['severity'],
               'confidence': risk['confidence'],
               'source': 'perturbation'
           })
       
       return risks

   def _extract_llm_risks(self, llm_responses: List[str]) -> List[Dict[str, Any]]:
       """Extract risk assessments from LLM responses"""
       risks = []
       for response in llm_responses:
           if 'Risk Assessment:' in response:
               risk_section = response.split('Risk Assessment:')[1].split('\n\n')[0]
               risk_lines = risk_section.strip().split('\n')
               for line in risk_lines:
                   if line.strip():
                       risk = self._parse_llm_risk(line.strip())
                       if risk:
                           risks.append(risk)
       return risks

   def _parse_llm_risk(self, text: str) -> Dict[str, Any]:
       """Parse risk information from LLM text"""
       try:
           # Expected format: "Risk: description (severity: X.XX, confidence: X.XX)"
           parts = text.split('(')
           if len(parts) == 2:
               description = parts[0].replace('Risk:', '').strip()
               metrics = parts[1].replace(')', '').strip().split(',')
               severity = float(metrics[0].split(':')[1].strip())
               confidence = float(metrics[1].split(':')[1].strip())
               
               return {
                   'category': self._determine_risk_category(description),
                   'level': self._determine_risk_level(severity),
                   'description': description,
                   'severity': severity,
                   'confidence': confidence,
                   'source': 'llm'
               }
       except:
           pass
       return None

   def _determine_risk_level(self, severity: float) -> RiskLevel:
       """Determine risk level based on severity score"""
       if severity >= 0.8:
           return RiskLevel.CRITICAL
       elif severity >= 0.6:
           return RiskLevel.HIGH
       elif severity >= 0.4:
           return RiskLevel.MEDIUM
       else:
           return RiskLevel.LOW

   def _determine_risk_category(self, description: str) -> RiskCategory:
       """Determine risk category based on description"""
       description = description.lower()
       if 'regime' in description or 'transition' in description:
           return RiskCategory.REGIME_CHANGE
       elif 'cycle' in description or 'pattern' in description:
           return RiskCategory.CYCLICAL
       elif 'volatility' in description:
           return RiskCategory.VOLATILITY
       elif 'systemic' in description:
           return RiskCategory.SYSTEMIC
       else:
           return RiskCategory.MARKET

   def _calculate_risk_metrics(
       self,
       risks: List[Dict[str, Any]]
   ) -> Dict[str, Any]:
       """Calculate overall risk metrics"""
       if not risks:
           return {
               'risk_counts': {},
               'risk_scores': {},
               'highest_risk': RiskLevel.LOW,
               'confidence': 0.0
           }
       
       # Count risks by level and category
       risk_counts = {
           'by_level': {level: 0 for level in RiskLevel},
           'by_category': {category: 0 for category in RiskCategory}
       }
       
       # Calculate risk scores
       risk_scores = {
           'total': 0.0,
           'by_category': {category: 0.0 for category in RiskCategory}
       }
       
       # Track highest risk and confidence
       highest_risk = RiskLevel.LOW
       weighted_confidence = 0.0
       total_weight = 0.0
       
       for risk in risks:
           # Update counts
           risk_counts['by_level'][risk['level']] += 1
           risk_counts['by_category'][risk['category']] += 1
           
           # Update scores
           weight = risk['confidence']
           score = risk['severity'] * weight
           risk_scores['total'] += score
           risk_scores['by_category'][risk['category']] += score
           
           # Update highest risk
           if risk['severity'] >= self.risk_threshold:
               highest_risk = max(highest_risk, risk['level'])
           
           # Update confidence
           weighted_confidence += risk['confidence'] * weight
           total_weight += weight
       
       # Normalize scores and confidence
       risk_scores['total'] /= len(risks)
       for category in RiskCategory:
           if risk_counts['by_category'][category] > 0:
               risk_scores['by_category'][category] /= risk_counts['by_category'][category]
       
       confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
       
       return {
           'risk_counts': risk_counts,
           'risk_scores': risk_scores,
           'highest_risk': highest_risk,
           'confidence': confidence
       }

   def _generate_risk_analysis(
       self,
       risks: List[Dict[str, Any]],
       metrics: Dict[str, Any]
   ) -> str:
       """Generate detailed risk analysis"""
       analysis_parts = []
       
       # Overall risk assessment
       analysis_parts.append("## Overall Risk Assessment\n")
       analysis_parts.append(f"Highest Risk Level: {metrics['highest_risk'].value.upper()}")
       analysis_parts.append(f"Overall Risk Score: {metrics['risk_scores']['total']:.2f}")
       analysis_parts.append(f"Analysis Confidence: {metrics['confidence']:.2f}\n")
       
       # Risks by category
       for category in RiskCategory:
           if metrics['risk_counts']['by_category'][category] > 0:
               analysis_parts.append(f"\n## {category.value.title()} Risks\n")
               category_risks = [r for r in risks if r['category'] == category]
               category_risks.sort(key=lambda x: x['severity'], reverse=True)
               
               for risk in category_risks:
                   analysis_parts.append(
                       f"- {risk['description']}\n"
                       f"  Level: {risk['level'].value.upper()}\n"
                       f"  Severity: {risk['severity']:.2f}\n"
                       f"  Confidence: {risk['confidence']:.2f}\n"
                       f"  Source: {risk['source']}"
                   )
       
       return "\n".join(analysis_parts)

   def _generate_risk_recommendations(
       self,
       risks: List[Dict[str, Any]],
       metrics: Dict[str, Any]
   ) -> List[str]:
       """Generate risk-based recommendations"""
       recommendations = []
       
       # Handle critical risks first
       critical_risks = [
           r for r in risks 
           if r['level'] == RiskLevel.CRITICAL
       ]
       if critical_risks:
           recommendations.append(
               "IMMEDIATE ACTION REQUIRED: Critical risks detected"
           )
           for risk in critical_risks:
               recommendations.append(
                   f"Address {risk['description']} immediately "
                   f"(Severity: {risk['severity']:.2f})"
               )
       
       # Handle high risks
       high_risks = [
           r for r in risks 
           if r['level'] == RiskLevel.HIGH
       ]
       if high_risks:
           for risk in high_risks:
               recommendations.append(
                   f"Develop mitigation strategy for {risk['description']} "
                   f"(Severity: {risk['severity']:.2f})"
               )
       
       # Category-specific recommendations
       for category in RiskCategory:
           if metrics['risk_scores']['by_category'][category] > self.risk_threshold:
               recommendations.append(
                   f"Review {category.value} risk management strategy "
                   f"(Score: {metrics['risk_scores']['by_category'][category]:.2f})"
               )
       
       # Confidence-based recommendations
       if metrics['confidence'] < 0.6:
           recommendations.append(
               "Gather additional data to improve risk assessment confidence"
           )
       
       return recommendations

    def _generate_risk_summary(
       self,
       risks: List[Dict[str, Any]], 
       metrics: Dict[str, Any]
   ) -> str:
       """Generate concise risk summary"""
       summary_parts = []
       
       # Overall risk status
       summary_parts.append(
           f"Overall Risk Level: {metrics['highest_risk'].value.upper()} "
           f"(Confidence: {metrics['confidence']:.2f})"
       )
       
       # Critical and high risks
       critical_count = metrics['risk_counts']['by_level'][RiskLevel.CRITICAL]
       high_count = metrics['risk_counts']['by_level'][RiskLevel.HIGH]
       
       if critical_count > 0:
           summary_parts.append(
               f"\nCritical Risks Identified: {critical_count}"
           )
           # Add brief description of critical risks
           critical_risks = [r for r in risks if r['level'] == RiskLevel.CRITICAL]
           for risk in critical_risks[:3]:  # Show top 3
               summary_parts.append(f"- {risk['description']}")
           if len(critical_risks) > 3:
               summary_parts.append(f"  (and {len(critical_risks)-3} more...)")
               
       if high_count > 0:
           summary_parts.append(
               f"\nHigh Risks Identified: {high_count}"
           )
           # Add brief description of high risks
           high_risks = [r for r in risks if r['level'] == RiskLevel.HIGH]
           for risk in high_risks[:3]:  # Show top 3
               summary_parts.append(f"- {risk['description']}")
           if len(high_risks) > 3:
               summary_parts.append(f"  (and {len(high_risks)-3} more...)")

       # Category summaries
       summary_parts.append("\nRisk Categories:")
       for category in RiskCategory:
           category_score = metrics['risk_scores']['by_category'][category]
           if category_score > 0:
               category_level = self._determine_risk_level(category_score)
               summary_parts.append(
                   f"- {category.value}: {category_level.value.upper()} "
                   f"(Score: {category_score:.2f})"
               )

       # Source breakdown
       summary_parts.append("\nRisk Detection Sources:")
       sources = {
           'hamiltonian': 'Conservation/Momentum Analysis',
           'fourier': 'Cyclical Pattern Analysis',
           'perturbation': 'Regime Change Analysis',
           'llm': 'Contextual Analysis'
       }
       
       for source, description in sources.items():
           source_risks = [r for r in risks if r['source'] == source]
           if source_risks:
               avg_confidence = sum(r['confidence'] for r in source_risks) / len(source_risks)
               summary_parts.append(
                   f"- {description}: {len(source_risks)} risks identified "
                   f"(Avg. Confidence: {avg_confidence:.2f})"
               )

       # Key recommendations if critical or high risks exist
       if critical_count > 0 or high_count > 0:
           summary_parts.append("\nKey Actions Required:")
           if critical_count > 0:
               summary_parts.append("- Immediate attention needed for critical risks")
           if high_count > 0:
               summary_parts.append("- Develop mitigation strategies for high risks")
           
       return "\n".join(summary_parts)

   async def _combine_risk_assessments(
       self,
       conservation_risks: List[Dict[str, Any]],
       cyclical_risks: List[Dict[str, Any]],
       regime_risks: List[Dict[str, Any]],
       llm_risks: List[Dict[str, Any]]
   ) -> List[Dict[str, Any]]:
       """
       Combine and correlate risks from different sources
       """
       all_risks = []
       
       # Helper function to find similar risks
       def find_similar_risks(target_risk: Dict[str, Any], risk_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
           similar = []
           for risk in risk_list:
               # Check for similarity in description or category
               if (risk['description'].lower() in target_risk['description'].lower() or
                   target_risk['description'].lower() in risk['description'].lower() or
                   risk['category'] == target_risk['category']):
                   similar.append(risk)
           return similar

       # Process each risk source and look for correlations
       all_source_risks = {
           'conservation': conservation_risks,
           'cyclical': cyclical_risks,
           'regime': regime_risks,
           'llm': llm_risks
       }

       processed_risks = set()  # Track processed risks to avoid duplicates

       for source_name, source_risks in all_source_risks.items():
           for risk in source_risks:
               # Create risk signature for tracking
               risk_sig = f"{risk['description']}:{risk['category']}"
               
               if risk_sig in processed_risks:
                   continue
               
               # Find similar risks from other sources
               similar_risks = []
               for other_source, other_risks in all_source_risks.items():
                   if other_source != source_name:
                       similar_risks.extend(find_similar_risks(risk, other_risks))

               if similar_risks:
                   # Combine similar risks
                   combined_severity = (risk['severity'] + 
                                     sum(r['severity'] for r in similar_risks)) / (len(similar_risks) + 1)
                   combined_confidence = (risk['confidence'] + 
                                       sum(r['confidence'] for r in similar_risks)) / (len(similar_risks) + 1)
                   
                   # Mark all similar risks as processed
                   processed_risks.add(risk_sig)
                   for similar in similar_risks:
                       processed_risks.add(f"{similar['description']}:{similar['category']}")
                   
                   all_risks.append({
                       'category': risk['category'],
                       'level': self._determine_risk_level(combined_severity),
                       'description': risk['description'],
                       'severity': combined_severity,
                       'confidence': combined_confidence,
                       'source': 'multiple',
                       'correlated_sources': [source_name] + 
                                           [r['source'] for r in similar_risks]
                   })
               else:
                   # Add individual risk
                   processed_risks.add(risk_sig)
                   all_risks.append(risk)

       return sorted(all_risks, key=lambda x: x['severity'], reverse=True)
