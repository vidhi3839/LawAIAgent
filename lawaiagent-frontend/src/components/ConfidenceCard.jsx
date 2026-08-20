const DEFAULT_THRESHOLDS = {
  statute: 0.5,
  rule: 0.5,
  definition: 0.5,
  past_cases: 0.70,
  mock_court: 0.8,
  summarizer: 0.75,
  document_qa: 0.75,
  followup: 0.0,
};

function fmtPct(score) {
  return `${Math.round((score ?? 0) * 100)}%`;
}

function statusFor(score, intent, lawyerThreshold) {

  const threshold = lawyerThreshold ?? (DEFAULT_THRESHOLDS[intent] ?? 0.75);
  const above = score >= threshold;
  let cls, status;
  if (above) {
    cls = "pass";
    status = "Above threshold";
  } else if (score > threshold * 0.85) {
    cls = "warn";
    status = "Just below threshold — verify independently";
  } else {
    cls = "fail";
    status = "Below threshold — do not rely on this answer";
  }
  return { threshold, cls, status };
}


export default function ConfidenceCard({ confidence, intent, lawyerThreshold }) {
  if (confidence == null) return null;
  const { threshold, cls, status } = statusFor(confidence, intent, lawyerThreshold);

  return (
    <div className="conf-card">
      <div className="conf-title">
        Confidence: <span className={`badge ${cls}`}>{fmtPct(confidence)}</span> &mdash; {status}
      </div>
      <div className="conf-meta">
        Task: {intent || "unknown"} &middot; Threshold: {threshold}
      </div>
    </div>
  );
}