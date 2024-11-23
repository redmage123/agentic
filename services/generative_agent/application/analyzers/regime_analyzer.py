async def _analyze_stability(
       self,
       current_regime: MarketRegime,
       energy_states: Dict[str, Any],
       frequency_states: Dict[str, Any],
       perturbation_states: Dict[str, Any]
   ) -> Dict[str, Any]:
       """
       Analyze market stability using physics-inspired metrics
       """

       # Analyze energy stability
       energy_stability = self._analyze_energy_stability(energy_states)
       
       # Analyze pattern stability
       pattern_stability = frequency_states['pattern_stability']
       
       # Analyze structural stability
       structural_stability = perturbation_states['stability_analysis']
       
       # Combine stability metrics
       stability_scores = {
           MarketStability.STABLE: 0,
           MarketStability.METASTABLE: 0,
           MarketStability.UNSTABLE: 0,
           MarketStability.CRITICAL: 0
       }
       
       # Energy state contribution
       stability_scores[energy_stability['state']] += energy_states['confidence']
       
       # Pattern stability contribution
       if pattern_stability.get('stability_index', 0) > 0.8:
           stability_scores[MarketStability.STABLE] += frequency_states['confidence']
       elif pattern_stability.get('stability_index', 0) > 0.5:
           stability_scores[MarketStability.METASTABLE] += frequency_states['confidence']
       else:
           stability_scores[MarketStability.UNSTABLE] += frequency_states['confidence']
           
       # Structural stability contribution
       if structural_stability.get('critical_proximity', 0) > 0.8:
           stability_scores[MarketStability.CRITICAL] += perturbation_states['confidence']
       elif structural_stability.get('stability_measure', 0) < 0.3:
           stability_scores[MarketStability.UNSTABLE] += perturbation_states['confidence']
       else:
           stability_scores[MarketStability.STABLE] += perturbation_states['confidence']
           
       # Determine overall stability
       overall_stability = max(stability_scores.items(), key=lambda x: x[1])[0]
       
       # Calculate confidence
       total_confidence = sum(stability_scores.values())
       confidence = stability_scores[overall_stability] / total_confidence if total_confidence > 0 else 0
       
       return {
           'stability': overall_stability,
           'confidence': confidence,
           'energy_stability': energy_stability,
           'pattern_stability': pattern_stability,
           'structural_stability': structural_stability,
           'stability_scores': stability_scores
       }

   def _analyze_energy_stability(
       self,
       energy_states: Dict[str, Any]
   ) -> Dict[str, Any]:
       """Analyze stability based on energy states"""
       energy_levels = energy_states['energy_levels']
       if not energy_levels:
           return {'state': MarketStability.UNKNOWN, 'confidence': 0}
           
       # Calculate energy stability metrics
       mean_energy = sum(energy_levels.values()) / len(energy_levels)
       energy_variance = sum((e - mean_energy) ** 2 for e in energy_levels.values()) / len(energy_levels)
       
       # Determine stability state
       if energy_variance > 0.8:
           state = MarketStability.CRITICAL
       elif energy_variance > 0.5:
           state = MarketStability.UNSTABLE
       elif energy_variance > 0.2:
           state = MarketStability.METASTABLE
       else:
           state = MarketStability.STABLE
           
       return {
           'state': state,
           'energy_variance': energy_variance,
           'mean_energy': mean_energy
       }

   async def _predict_transitions(
       self,
       current_regime: MarketRegime,
       energy_states: Dict[str, Any],
       frequency_states: Dict[str, Any],
       perturbation_states: Dict[str, Any],
       llm_insights: Dict[str, Any]
   ) -> Dict[str, Any]:
       """
       Predict potential regime transitions
       """
       # Get transition probabilities from each source
       energy_transitions = energy_states['state_transitions']
       frequency_transitions = frequency_states['frequency_transitions']
       perturbation_transitions = perturbation_states['transition_probabilities']
       
       # Combine transition signals
       transition_probability = 0.0
       confidence = 0.0
       potential_regimes = []
       
       # Energy transitions
       if energy_transitions:
           transition_probability += max(t.get('probability', 0) for t in energy_transitions)
           confidence += energy_states['confidence']
           potential_regimes.extend(t.get('target_regime') for t in energy_transitions)
           
       # Frequency transitions
       if frequency_transitions:
           transition_probability += max(t.get('probability', 0) for t in frequency_transitions)
           confidence += frequency_states['confidence']
           potential_regimes.extend(t.get('target_regime') for t in frequency_transitions)
           
       # Perturbation transitions
       if perturbation_transitions:
           transition_probability += perturbation_transitions.get('transition_imminent', 0)
           confidence += perturbation_states['confidence']
           potential_regimes.extend(perturbation_transitions.get('potential_regimes', []))
           
       # Normalize probability and confidence
       source_count = sum(1 for x in [energy_transitions, frequency_transitions, perturbation_transitions] if x)
       if source_count > 0:
           transition_probability /= source_count
           confidence /= source_count
           
       # Determine most likely next regime
       if potential_regimes:
           next_regime = max(set(potential_regimes), key=potential_regimes.count)
       else:
           next_regime = current_regime
           
       return {
           'probability': transition_probability,
           'confidence': confidence,
           'current_regime': current_regime,
           'predicted_regime': next_regime,
           'potential_regimes': list(set(potential_regimes)),
           'transition_indicators': {
               'energy': energy_transitions,
               'frequency': frequency_transitions,
               'perturbation': perturbation_transitions
           }
       }

   def _generate_regime_analysis(
       self,
       current_regime: MarketRegime,
       stability_analysis: Dict[str, Any],
       transition_analysis: Dict[str, Any],
       nn_results: Tuple[Dict[str, Any], ...],
       llm_insights: Dict[str, Any]
   ) -> str:
       """Generate detailed regime analysis"""
       analysis_parts = []
       
       # Current regime analysis
       analysis_parts.append(f"## Current Market Regime Analysis\n")
       analysis_parts.append(f"Current Regime: {current_regime.value}")
       analysis_parts.append(f"Stability State: {stability_analysis['stability'].value}")
       analysis_parts.append(f"Analysis Confidence: {stability_analysis['confidence']:.2f}\n")
       
       # Stability analysis
       analysis_parts.append("## Stability Analysis\n")
       analysis_parts.append("Energy State Stability:")
       analysis_parts.append(f"- Variance: {stability_analysis['energy_stability']['energy_variance']:.2f}")
       analysis_parts.append(f"- Mean Energy: {stability_analysis['energy_stability']['mean_energy']:.2f}\n")
       
       analysis_parts.append("Pattern Stability:")
       for metric, value in stability_analysis['pattern_stability'].items():
           analysis_parts.append(f"- {metric}: {value:.2f}")
       
       # Transition analysis
       analysis_parts.append("\n## Transition Analysis\n")
       analysis_parts.append(f"Transition Probability: {transition_analysis['probability']:.2f}")
       analysis_parts.append(f"Predicted Next Regime: {transition_analysis['predicted_regime'].value}")
       analysis_parts.append("\nPotential Future Regimes:")
       for regime in transition_analysis['potential_regimes']:
           analysis_parts.append(f"- {regime.value}")
           
       # LLM insights
       if llm_insights['contextual_factors']:
           analysis_parts.append("\n## Market Context\n")
           for factor in llm_insights['contextual_factors']:
               analysis_parts.append(f"- {factor}")
               
       return "\n".join(analysis_parts)

   def _generate_regime_recommendations(
       self,
       current_regime: MarketRegime,
       stability_analysis: Dict[str, Any],
       transition_analysis: Dict[str, Any]
   ) -> List[str]:
       """Generate regime-based recommendations"""
       recommendations = []
       
       # Stability-based recommendations
       if stability_analysis['stability'] == MarketStability.CRITICAL:
           recommendations.append("CRITICAL: Immediate risk management actions required")
           recommendations.append("Consider reducing exposure significantly")
       elif stability_analysis['stability'] == MarketStability.UNSTABLE:
           recommendations.append("Implement enhanced risk monitoring")
           recommendations.append("Review position sizing and leverage")
           
       # Regime-based recommendations
       if current_regime == MarketRegime.HIGH_VOLATILITY:
           recommendations.append("Adjust position sizes for high volatility")
           recommendations.append("Implement volatility-based risk controls")
       elif current_regime == MarketRegime.TRENDING_UP:
           recommendations.append("Consider trend-following strategies")
           recommendations.append("Monitor for trend exhaustion signals")
       elif current_regime == MarketRegime.TRENDING_DOWN:
           recommendations.append("Implement downside protection strategies")
           recommendations.append("Consider defensive positioning")
           
       # Transition-based recommendations
       if transition_analysis['probability'] > 0.7:
           recommendations.append("Prepare for potential regime transition")
           recommendations.append(
               f"Position for possible shift to {transition_analysis['predicted_regime'].value}"
           )
           
       return recommendations

   def _generate_regime_summary(
       self,
       current_regime: MarketRegime,
       stability_analysis: Dict[str, Any],
       transition_analysis: Dict[str, Any]
   ) -> str:
       """Generate concise regime summary"""
       summary_parts = []
       
       # Overall status
       summary_parts.append(
           f"Market Regime: {current_regime.value} "
           f"(Stability: {stability_analysis['stability'].value}, "
           f"Confidence: {stability_analysis['confidence']:.2f})"
       )
       
       # Transition warning if applicable
       if transition_analysis['probability'] > 0.5:
           summary_parts.append(
               f"\nRegime Transition Warning: "
               f"{transition_analysis['probability']:.1%} probability of transition to "
               f"{transition_analysis['predicted_regime'].value}"
           )
           
       # Key stability factors
       stability_scores = stability_analysis['stability_scores']
       critical_score = stability_scores[MarketStability.CRITICAL]
       if critical_score > 0.3:
           summary_parts.append(
               f"\nCritical Stability Warning: "
               f"{critical_score:.1%} critical stability score"
           )
           
       return "\n".join(summary_parts)
