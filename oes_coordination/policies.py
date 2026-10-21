"""The entire declared deterministic policy class; reporting is not an action."""

from .model import Action, PolicyInput


LOCAL_POLICIES = ("continue", "isolate", "local")
SHARED_POLICIES = LOCAL_POLICIES + ("any_report", "majority_report")


def act(policy: str, inputs: PolicyInput) -> Action:
    if policy not in SHARED_POLICIES:
        raise ValueError("unknown policy")
    if policy == "continue":
        isolate = False
    elif policy == "isolate":
        isolate = True
    elif policy == "local":
        isolate = bool(inputs.local)
    else:
        if inputs.reports is None:
            raise ValueError("report policy requires shared reports")
        threshold = 1 if policy == "any_report" else 2
        isolate = sum(inputs.reports) >= threshold
    return Action.ISOLATE if isolate else Action.CONTINUE
