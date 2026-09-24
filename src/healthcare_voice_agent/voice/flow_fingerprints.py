"""Fingerprints of loaded static flow instructions, never patient context or text logs."""
from __future__ import annotations

import hashlib
import json


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def flow_prompt_fingerprint(agent_mode: str) -> dict | None:
    """Hash the live Python module's role and native tool definitions, not disk bytes.

    Dynamic selection outcomes/user messages are deliberately not inspected.
    This identifies loaded static prompts/contracts, not an entire source build.
    Constructing schemas does not create a provider, database session, or model.
    """
    if agent_mode == "frontdesk_demo":
        from healthcare_voice_agent.demo import flow
        functions = flow.global_functions()
    elif agent_mode == "clinician":
        from healthcare_voice_agent.clinician import flow
        functions = flow.functions()
    else:
        return None
    contracts = [{"name": item.name, "description": item.description,
                  "properties": item.properties, "required": item.required,
                  "cancel_on_interruption": item.cancel_on_interruption,
                  "timeout_secs": item.timeout_secs} for item in functions]
    return {"agent_mode": agent_mode, "fingerprint_version": 1,
            "role_sha256": _digest(flow._ROLE),
            "tools_sha256": _digest(contracts),
            "sha256": _digest({"role": flow._ROLE, "tools": contracts})}
