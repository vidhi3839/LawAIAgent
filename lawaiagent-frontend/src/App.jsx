import { useEffect, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import ChatMessage from "./components/ChatMessage";
import UploadPanel from "./components/UploadPanel";
import CaseUploadPage from "./components/CaseUploadPage";
import { createThreadId, sendMessage, fetchMessages } from "./api";
import "./App.css";

export default function App() {
  const [page, setPage] = useState("chat"); // "chat" | "cases"
  const [threadId, setThreadId] = useState(null);
  const [lawyerName, setLawyerName] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [lawyerThreshold, setLawyerThreshold] = useState(0.75);
  const [sending, setSending] = useState(false);
  const [connectionError, setConnectionError] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    setThreadId(createThreadId());
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  function handleNewConversation() {
    setMessages([]);
    setThreadId(createThreadId());
    setConnectionError(null);
  }

  async function handleSelectThread(id) {
    if (id === threadId) return;
    setThreadId(id);
    try {
      const msgs = await fetchMessages(id, lawyerName);
      setMessages(msgs);
    } catch {
      setMessages([]);
    }
  }

  async function handleSend() {
    const text = input.trim();
    if (!text || !lawyerName) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: text }]);
    setSending(true);
    try {
      const data = await sendMessage({ threadId, lawyerName, query: text, userThreshold: lawyerThreshold });
      setMessages((m) => [...m, { role: "assistant", content: data.response, confidence: data.confidence, intent: data.intent }]);
    } catch (e) {
      setConnectionError(`Cannot reach API: ${e.message}`);
      setMessages((m) => [...m, { role: "assistant", content: `Error: ${e.message}` }]);
    }
    setSending(false);
  }

  function handleUploadMessages(newMsgs) {
    setMessages((m) => [...m, ...newMsgs]);
  }

  if (connectionError) {
    return (
      <div className="fatal-error">
        {connectionError}
        <br />
        Make sure the API is running: <code>python -m uvicorn api:app --reload</code>
      </div>
    );
  }

  return (
    <div className="app">
      <Sidebar
        threadId={threadId}
        lawyerName={lawyerName}
        setLawyerName={setLawyerName}
        lawyerThreshold={lawyerThreshold}
        setLawyerThreshold={setLawyerThreshold}
        onNewConversation={handleNewConversation}
        onSelectThread={handleSelectThread}
        page={page}
        setPage={setPage}
      />
      {page === "cases" ? (
        <CaseUploadPage lawyerName={lawyerName} />
      ) : (
        <main className="main">
          <h1>Law Firm Internal Agent</h1>

          <div className="chat-window">
            {messages.map((m, i) => (
              <ChatMessage
                key={i}
                role={m.role}
                content={m.content}
                confidence={m.confidence}
                intent={m.intent}
                lawyerThreshold={lawyerThreshold}
              />
            ))}
            {sending && (
              <div className="message assistant">
                <div className="message-content">Researching legal sources...</div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="chat-input-row">
            <input
              type="text"
              placeholder={lawyerName ? "Ask the agent a legal question..." : "Enter your name in the sidebar first"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              disabled={!lawyerName}
            />
            <button onClick={handleSend} disabled={sending || !lawyerName}>
              Send
            </button>
          </div>

          <UploadPanel threadId={threadId} lawyerName={lawyerName} onNewMessages={handleUploadMessages} />
        </main>
      )}
    </div>
  );
}