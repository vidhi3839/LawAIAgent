import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ConfidenceCard from "./ConfidenceCard";

function stripThink(content) {
  if (content && content.includes("<think>") && content.includes("</think>")) {
    return content.split("</think>").pop().trim();
  }
  return content;
}

export default function ChatMessage({ role, content, confidence, intent, lawyerThreshold }) {
  const clean = stripThink(content);
  return (
    <div className={`message ${role}`}>
      <div className="message-role">{role === "user" ? "You" : "Agent"}</div>
      <div className="message-content">

        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ node, ...props }) => (
              <a {...props} target="_blank" rel="noopener noreferrer" />
            ),
          }}
        >
          {clean}
        </ReactMarkdown>
      </div>

      {confidence != null && (
        <ConfidenceCard confidence={confidence} intent={intent} lawyerThreshold={lawyerThreshold} />
      )}
    </div>
  );
}