"""Risk Alert Engine.

Generates structured, auditable alerts for high-risk and critical-risk shipments:
- CRITICAL risk: Priority Disruption Alert
- HIGH risk: Warning Disruption Alert
- MEDIUM risk: Monitoring Notice
- LOW risk: No urgent alert

Includes deduplication management to prevent duplicate alerts on repeated runs.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from src.risk_engine.config import RiskEngineConfig
from src.risk_engine.schemas import RiskAlert, RiskAssessmentResult


def generate_risk_alert(
    assessment: RiskAssessmentResult,
    config: RiskEngineConfig,
) -> Optional[RiskAlert]:
    """Generate structured RiskAlert for a shipment assessment if threshold met.

    Args:
        assessment: RiskAssessmentResult for a shipment.
        config: RiskEngineConfig containing alert settings.

    Returns:
        RiskAlert if severity warrants an alert, None otherwise.
    """
    level = assessment.risk_level.upper()
    score = assessment.composite_risk_score
    reasons = assessment.risk_reasons

    if level == "CRITICAL" or score >= config.alerts.critical_severity_threshold:
        alert_type = "PRIORITY_DISRUPTION_ALERT"
        severity = "CRITICAL"
        reasons_text = ", ".join(reasons) if reasons else "Composite critical risk thresholds exceeded"
        msg = (
            f"PRIORITY ALERT: Shipment '{assessment.event_id}' has reached CRITICAL risk "
            f"(composite score: {score:.2f}). Triggered reasons: {reasons_text}."
        )
    elif level == "HIGH" or score >= config.alerts.high_severity_threshold:
        alert_type = "WARNING_DISRUPTION_ALERT"
        severity = "HIGH"
        reasons_text = ", ".join(reasons) if reasons else "Elevated operational risk signals"
        msg = (
            f"WARNING ALERT: Shipment '{assessment.event_id}' exhibits HIGH disruption risk "
            f"(composite score: {score:.2f}). Triggered reasons: {reasons_text}."
        )
    elif level == "MEDIUM":
        alert_type = "MONITORING_NOTICE"
        severity = "MEDIUM"
        msg = (
            f"MONITORING NOTICE: Shipment '{assessment.event_id}' exhibits moderate risk "
            f"(composite score: {score:.2f}). Recommend routine operational tracking."
        )
    else:
        # LOW risk: no urgent alert by default
        return None

    return RiskAlert(
        event_id=assessment.event_id,
        alert_type=alert_type,
        severity=severity,
        risk_score=score,
        alert_message=msg,
        triggered_reasons=reasons,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


class RiskAlertEngine:
    """Manages batch generation and in-memory deduplication of supply alerts."""

    def __init__(self, config: RiskEngineConfig):
        self.config = config
        self._seen_alert_keys: Set[str] = set()
        self._alerts_history: List[RiskAlert] = []

    def process_assessments(
        self,
        assessments: List[RiskAssessmentResult],
        include_monitoring: bool = False,
    ) -> List[RiskAlert]:
        """Generate alerts for a collection of assessments with deduplication.

        Args:
            assessments: List of shipment risk assessment results.
            include_monitoring: If True, include MEDIUM severity monitoring notices.
                               If False (default), include only HIGH and CRITICAL alerts.

        Returns:
            List of unique newly generated RiskAlert objects.
        """
        new_alerts: List[RiskAlert] = []

        for assessment in assessments:
            alert = generate_risk_alert(assessment, self.config)
            if alert is None:
                continue

            if not include_monitoring and alert.severity == "MEDIUM":
                continue

            # Deduplication check
            key = alert.deduplication_key
            if key in self._seen_alert_keys:
                continue

            self._seen_alert_keys.add(key)
            self._alerts_history.append(alert)
            new_alerts.append(alert)

        return new_alerts

    def clear_seen_alerts(self) -> None:
        """Reset deduplication cache."""
        self._seen_alert_keys.clear()
        self._alerts_history.clear()

    @property
    def alert_counts(self) -> Dict[str, int]:
        """Return counts of accumulated alerts by severity."""
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for a in self._alerts_history:
            if a.severity in counts:
                counts[a.severity] += 1
        return counts
