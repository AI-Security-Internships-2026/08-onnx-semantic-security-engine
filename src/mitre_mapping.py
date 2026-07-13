"""
MITRE ATT&CK mapping for CIC-IDS2018 traffic labels.
Maps predicted class labels to standardized MITRE technique IDs.
"""

MITRE_ATTACK_MAPPING = {
    "Benign": {
        "technique_id": None,
        "technique": "N/A (Benign Traffic)",
        "tactic": "N/A",
    },
    "FTP-BruteForce": {
        "technique_id": "T1110.001",
        "technique": "Brute Force: Password Guessing",
        "tactic": "Credential Access",
    },
    "SSH-Bruteforce": {
        "technique_id": "T1110.004",
        "technique": "Brute Force: SSH",
        "tactic": "Credential Access",
    },
    "DoS attacks-Hulk": {
        "technique_id": "T1499.002",
        "technique": "Endpoint DoS: Service Exhaustion Flood",
        "tactic": "Impact",
    },
    "DoS attacks-SlowHTTPTest": {
        "technique_id": "T1499.001",
        "technique": "Endpoint DoS: OS Exhaustion Flood",
        "tactic": "Impact",
    },
    "DoS attacks-Slowloris": {
        "technique_id": "T1499.001",
        "technique": "Endpoint DoS: OS Exhaustion Flood",
        "tactic": "Impact",
    },
    "DoS attacks-GoldenEye": {
        "technique_id": "T1499.002",
        "technique": "Endpoint DoS: Service Exhaustion Flood",
        "tactic": "Impact",
    },
    "DDOS attack-HOIC": {
        "technique_id": "T1498.001",
        "technique": "Network DoS: Direct Network Flood",
        "tactic": "Impact",
    },
    "DDOS attack-LOIC-UDP": {
        "technique_id": "T1498.001",
        "technique": "Network DoS: Direct Network Flood",
        "tactic": "Impact",
    },
    "DDoS attacks-LOIC-HTTP": {
        "technique_id": "T1498.001",
        "technique": "Network DoS: Direct Network Flood",
        "tactic": "Impact",
    },
    "Bot": {
        "technique_id": "T1583.005",
        "technique": "Acquire Infrastructure: Botnet",
        "tactic": "Resource Development",
    },
    "Infilteration": {
        "technique_id": "T1071.001",
        "technique": "Application Layer Protocol: Web",
        "tactic": "Command and Control",
    },
    "Brute Force -Web": {
        "technique_id": "T1110.001",
        "technique": "Brute Force: Password Guessing",
        "tactic": "Credential Access",
    },
    "Brute Force -XSS": {
        "technique_id": "T1059.007",
        "technique": "Command and Scripting: JavaScript",
        "tactic": "Execution",
    },
    "SQL Injection": {
        "technique_id": "T1190",
        "technique": "Exploit Public-Facing Application",
        "tactic": "Initial Access",
    },
}


def get_mitre_label(predicted_class: str) -> dict:
    """Map a predicted class label to MITRE ATT&CK technique."""
    return MITRE_ATTACK_MAPPING.get(predicted_class, {
        "technique_id": "UNKNOWN",
        "technique": f"Unmapped: {predicted_class}",
        "tactic": "Unknown",
    })


def get_all_mappings() -> dict:
    """Return the full mapping dictionary."""
    return MITRE_ATTACK_MAPPING
