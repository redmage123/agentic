# services/generative_agent/application/analyzers/pattern_helpers.py
from typing import List, Dict, Any, Tuple
from datetime import datetime
import json


async def _identify_patterns(responses: List[str]) -> List[Dict[str, Any]]:
    """Extract patterns from LLM responses"""
    patterns = []
    for response in responses:
        try:
            # Extract JSON pattern data if present
            pattern_data = _extract_json_patterns(response)
            if pattern_data:
                patterns.extend(pattern_data)
            else:
                # Parse text-based patterns
                patterns.extend(_parse_text_patterns(response))
        except Exception as e:
            continue
    return patterns


async def _find_correlations(responses: List[str]) -> List[Dict[str, float]]:
    """Find correlations between identified patterns"""
    correlations = []
    for response in responses:
        try:
            if '"correlations":' in response:
                corr_data = json.loads(response)["correlations"]
                correlations.extend(corr_data)
        except:
            continue
    return correlations


async def _detect_anomalies(patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect anomalies in identified patterns"""
    anomalies = []
    for pattern in patterns:
        if _is_anomalous(pattern):
            anomalies.append(
                {
                    "pattern_id": pattern.get("id"),
                    "anomaly_type": _determine_anomaly_type(pattern),
                    "severity": _calculate_anomaly_severity(pattern),
                    "timestamp": datetime.now().isoformat(),
                }
            )
    return anomalies


def _is_anomalous(pattern: Dict[str, Any]) -> bool:
    """Check if a pattern is anomalous based on defined criteria"""
    threshold = 2.0  # Standard deviations
    if "std_dev" in pattern:
        return abs(pattern["std_dev"]) > threshold
    return False


def _determine_anomaly_type(pattern: Dict[str, Any]) -> str:
    """Determine type of anomaly"""
    if pattern.get("std_dev", 0) > 3.0:
        return "extreme_outlier"
    elif pattern.get("std_dev", 0) > 2.0:
        return "moderate_outlier"
    return "mild_outlier"


def _calculate_anomaly_severity(pattern: Dict[str, Any]) -> float:
    """Calculate severity score for anomaly"""
    base_severity = abs(pattern.get("std_dev", 0)) / 2.0
    return min(1.0, base_severity)


def _extract_json_patterns(text: str) -> List[Dict[str, Any]]:
    """Extract JSON pattern data from text"""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            json_str = text[start : end + 1]
            data = json.loads(json_str)
            if "patterns" in data:
                return data["patterns"]
    except:
        pass
    return []


def _parse_text_patterns(text: str) -> List[Dict[str, Any]]:
    """Parse patterns from plain text response"""
    patterns = []
    lines = text.split("\n")
    current_pattern = {}

    for line in lines:
        if "Pattern:" in line:
            if current_pattern:
                patterns.append(current_pattern)
            current_pattern = {"description": line.split("Pattern:")[1].strip()}
        elif "Confidence:" in line and current_pattern:
            try:
                current_pattern["confidence"] = (
                    float(line.split("Confidence:")[1].strip().rstrip("%")) / 100
                )
            except:
                current_pattern["confidence"] = 0.5

    if current_pattern:
        patterns.append(current_pattern)

    return patterns
