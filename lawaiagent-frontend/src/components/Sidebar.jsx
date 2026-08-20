import { useEffect, useState } from "react";
import { fetchThreads } from "../api";

export default function Sidebar({
  threadId,
  lawyerName,
  setLawyerName,
  lawyerThreshold,
  setLawyerThreshold,
  onNewConversation,
  onSelectThread,
  page,
  setPage,
}) {
  const [threads, setThreads] = useState([]);
  const [threadsError, setThreadsError] = useState(false);

  useEffect(() => {
    if (!lawyerName) {
      setThreads([]);
      return;
    }
    fetchThreads(lawyerName)
      .then((t) => {
        setThreads(t);
        setThreadsError(false);
      })
      .catch(() => setThreadsError(true));
  }, [threadId, lawyerName]);

  return (
    <aside className="sidebar">
      <h2>⚖ Law Firm Agent</h2>

      <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
        <button
          className="btn-primary"
          style={{ opacity: page === "chat" ? 1 : 0.5 }}
          onClick={() => setPage("chat")}
        >
          Chat
        </button>
        <button
          className="btn-primary"
          style={{ opacity: page === "cases" ? 1 : 0.5 }}
          onClick={() => setPage("cases")}
        >
          Case Database
        </button>
      </div>

      <p className="thread-id">Thread: {threadId ? threadId.slice(0, 8) : "None"}...</p>

      <hr />
      <h3>Lawyer</h3>
      <label className="field">
        Your name
        <input
          type="text"
          placeholder="e.g. Jane Smith"
          value={lawyerName}
          onChange={(e) => setLawyerName(e.target.value)}
        />
      </label>
      {!lawyerName && <p className="hint">Enter your name to send messages or view past conversations.</p>}

      <hr />
      <h3>Confidence Settings</h3>
      <p className="hint">Set how strict the system should be before flagging an answer as reliable.</p>
      <label className="field">
        Confidence threshold
        <input
          type="number"
          min="0"
          max="1"
          step="0.05"
          value={lawyerThreshold}
          onChange={(e) => setLawyerThreshold(parseFloat(e.target.value))}
        />
      </label>
      <p className="hint">Answers below {lawyerThreshold.toFixed(2)} will be flagged for review</p>

      <hr />
      <button className="btn-primary" onClick={onNewConversation}>
        New Conversation
      </button>

      <hr />
      <h3>Past Conversations</h3>
      {threadsError && <p className="hint">Could not load past conversations</p>}
      {!threadsError && threads.length === 0 && <p className="hint">No past conversations yet</p>}
      <ul className="thread-list">
        {threads.slice(0, 10).map((t) => (
          <li key={t.thread_id}>
            <button
              className={t.thread_id === threadId ? "thread-btn active" : "thread-btn"}
              onClick={() => onSelectThread(t.thread_id)}
            >
              {(t.label || "Conversation").slice(0, 35)}{" "}
              <span className="thread-date">({(t.created_at || "").slice(0, 10)})</span>
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
}