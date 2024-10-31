# Configuration for agent service URLs
from typing import Dict

AGENT_URLS: Dict[str, str] = {
    "hnn_agent": "http://hnn-agent-service:5002",
    "fnn_agent": "http://fnn-agent-service:5003",
    "perturbation_agent": "http://perturbation-agent-service:5004"
}
