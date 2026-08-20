import { useEffect, useState } from "react";
import { uploadCase, fetchUploadedCases } from "../api";

const JURISDICTIONS = ["", "federal", "state", "ninth circuit", "second circuit", "other"];
const JURISDICTION_LABELS = {
  "": "Not sure — leave blank, we'll try to detect it",
  federal: "federal",
  state: "state",
  "ninth circuit": "ninth circuit",
  "second circuit": "second circuit",
  other: "other",
};

export default function CaseUploadPage({ lawyerName }) {
  const [file, setFile] = useState(null);
  const [caseName, setCaseName] = useState("");
  const [citation, setCitation] = useState("");
  const [year, setYear] = useState("");
  const [court, setCourt] = useState("");
  const [jurisdiction, setJurisdiction] = useState("");
  const [legalIssues, setLegalIssues] = useState("");
  const [status, setStatus] = useState(null);
  const [autoDetected, setAutoDetected] = useState(null);
  const [busy, setBusy] = useState(false);
  const [cases, setCases] = useState([]);
  const [casesError, setCasesError] = useState(false);

  function loadCases() {
    fetchUploadedCases()
      .then((c) => {
        setCases(c);
        setCasesError(false);
      })
      .catch(() => setCasesError(true));
  }

  useEffect(() => {
    loadCases();
  }, []);

  function reset() {
    setFile(null);
    setCaseName("");
    setCitation("");
    setYear("");
    setCourt("");
    setJurisdiction("");
    setLegalIssues("");
  }

  async function handleUpload() {
    if (!file || !caseName.trim() || !lawyerName) return;
    setBusy(true);
    setStatus(null);
    setAutoDetected(null);
    try {
      const data = await uploadCase({
        file, caseName, citation, year, court, jurisdiction, legalIssues, lawyerName,
      });
      setStatus({ type: "success", text: data.message });
      const detected = data.auto_detected || {};
      const detectedEntries = Object.entries(detected).filter(([, v]) => v);
      if (detectedEntries.length > 0) {
        setAutoDetected(detectedEntries);
      }
      reset();
      loadCases();
    } catch (e) {
      setStatus({ type: "error", text: e.message });
    }
    setBusy(false);
  }

  return (
    <main className="main">
      <h1>Case Law Database</h1>
      <p className="hint" style={{ marginBottom: 20 }}>
        Upload a case PDF here to add it to the firm&rsquo;s searchable past-cases database.
        Once ingested, it becomes available to the &ldquo;past cases&rdquo; search feature for
        every lawyer, immediately. <b>Only the file and case name are required</b> — citation,
        court, and jurisdiction are optional; the system tries to detect them automatically from
        the document itself, since you may not have that information on hand. Leave any of them
        blank if you&rsquo;re not sure.
      </p>

      {!lawyerName && (
        <p className="hint">Enter your name in the sidebar (on the Chat page) before uploading a case.</p>
      )}

      <div className="upload-body" style={{ maxWidth: 520 }}>
        <label className="field">
          Case PDF <span style={{ color: "var(--danger)" }}>*</span>
          <input type="file" accept=".pdf" onChange={(e) => setFile(e.target.files[0] || null)} />
        </label>

        <label className="field">
          Case name <span style={{ color: "var(--danger)" }}>*</span>
          <input
            type="text"
            placeholder="e.g. Van Buren v. United States"
            value={caseName}
            onChange={(e) => setCaseName(e.target.value)}
          />
        </label>

        <label className="field">
          Citation (optional — leave blank if unsure)
          <input
            type="text"
            placeholder="e.g. 593 U.S. 374 -- auto-detected if left blank"
            value={citation}
            onChange={(e) => setCitation(e.target.value)}
          />
        </label>

        <div style={{ display: "flex", gap: 10 }}>
          <label className="field" style={{ flex: 1 }}>
            Year (optional)
            <input type="text" placeholder="auto-detected" value={year} onChange={(e) => setYear(e.target.value)} />
          </label>
          <label className="field" style={{ flex: 2 }}>
            Jurisdiction (optional)
            <select value={jurisdiction} onChange={(e) => setJurisdiction(e.target.value)}>
              {JURISDICTIONS.map((j) => (
                <option key={j} value={j}>{JURISDICTION_LABELS[j]}</option>
              ))}
            </select>
          </label>
        </div>

        <label className="field">
          Court (optional — leave blank if unsure)
          <input
            type="text"
            placeholder="e.g. Supreme Court of the United States -- auto-detected if left blank"
            value={court}
            onChange={(e) => setCourt(e.target.value)}
          />
        </label>

        <label className="field">
          Legal issues (optional, comma-separated, helps future search relevance)
          <input
            type="text"
            placeholder="e.g. Computer Fraud and Abuse Act, exceeding authorized access"
            value={legalIssues}
            onChange={(e) => setLegalIssues(e.target.value)}
          />
        </label>

        <button
          className="btn-primary"
          disabled={!file || !caseName.trim() || !lawyerName || busy}
          onClick={handleUpload}
        >
          {busy ? "Ingesting..." : "Add to Case Database"}
        </button>

        {status && <p className={`status ${status.type}`}>{status.text}</p>}
        {autoDetected && (
          <div className="status info">
            Auto-detected from the document (not something you entered — verify these look right):
            <ul style={{ margin: "6px 0 0 18px" }}>
              {autoDetected.map(([field, value]) => (
                <li key={field}>{field}: {value}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <hr style={{ margin: "28px 0" }} />
      <h3>Cases Currently in the Database</h3>
      {casesError && <p className="hint">Could not load the case list.</p>}
      {!casesError && cases.length === 0 && <p className="hint">No lawyer-uploaded cases yet.</p>}
      <ul className="thread-list">
        {cases.map((c) => (
          <li key={c.case_name} style={{ marginBottom: 10 }}>
            <div style={{ fontWeight: 600 }}>{c.case_name}</div>
            <div className="hint">
              {c.citation || "no citation"} &middot; {c.court || "court unknown"} &middot; {c.jurisdiction}
              {c.uploaded_by ? ` \u00b7 uploaded by ${c.uploaded_by}` : ""}
            </div>
          </li>
        ))}
      </ul>
    </main>
  );
}