import { useState } from "react";
import { uploadDocument, sendMessage } from "../api";

export default function UploadPanel({ threadId, lawyerName, onNewMessages }) {
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);

  function reset() {
    setFile(null);
    setQuestion("");
    setStatus(null);
  }

  async function handleAnalyse() {
    if (!file || !lawyerName) return;
    setBusy(true);
    setStatus(null);

    let filePath;
    try {
      const uploadData = await uploadDocument({ file, threadId, lawyerName });
      filePath = uploadData.file_path;
      setStatus({ type: "success", text: `Uploaded: ${file.name}` });
    } catch (e) {
      setStatus({ type: "error", text: e.message });
      setBusy(false);
      return;
    }

    const q = question.trim() || "summarise";
    const displayText = `[Document] ${file.name} — ${q}`;
    try {
      const data = await sendMessage({
        threadId,
        lawyerName,
        query: `${filePath} ${q}`,
        displayQuery: displayText,
      });
      onNewMessages([
        { role: "user", content: displayText },
        { role: "assistant", content: data.response, confidence: data.confidence, intent: data.intent },
      ]);
      reset();
      setOpen(false);
    } catch (e) {
      setStatus({ type: "error", text: e.message });
    }

    setBusy(false);
  }

  return (
    <div className="upload-panel">
      <button className="upload-toggle" onClick={() => setOpen(!open)}>
        {open ? "▲" : "▼"} Upload Document for Analysis
      </button>
      {open && (
        <div className="upload-body">
          <div className="upload-row">
            <input type="file" accept=".pdf" onChange={(e) => setFile(e.target.files[0] || null)} />
          </div>
          <input
            type="text"
            className="upload-question"
            placeholder="summarise / What was the holding? / What clauses are missing?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button className="btn-primary" disabled={!file || !lawyerName || busy} onClick={handleAnalyse}>
            {busy ? "Working..." : "Analyse"}
          </button>

          {status && <p className={`status ${status.type}`}>{status.text}</p>}
        </div>
      )}
    </div>
  );
}