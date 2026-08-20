export const API_BASE = "http://127.0.0.1:8000";

export function createThreadId() {
  return crypto.randomUUID();
}

export async function sendMessage({ threadId, lawyerName, query, userThreshold = null, displayQuery = null }) {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      thread_id: threadId,
      lawyer_name: lawyerName,
      user_threshold: userThreshold,
      display_query: displayQuery,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text.slice(0, 300)}`);
  }
  return res.json();
}

export async function uploadDocument({ file, threadId, lawyerName }) {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams({
    thread_id: threadId || "",
    lawyer_name: lawyerName || "",
  });
  const res = await fetch(`${API_BASE}/document?${params}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export async function fetchThreads(lawyerName) {
  if (!lawyerName) return [];
  const res = await fetch(`${API_BASE}/threads?lawyer_name=${encodeURIComponent(lawyerName)}`);
  if (!res.ok) throw new Error("Could not load past conversations");
  const data = await res.json();
  return data.threads || [];
}

export async function fetchMessages(threadId, lawyerName) {
  const res = await fetch(
    `${API_BASE}/threads/${threadId}/messages?lawyer_name=${encodeURIComponent(lawyerName)}`
  );
  if (!res.ok) throw new Error("Could not load conversation history (not found, or not yours)");
  const data = await res.json();
  return data.messages || [];
}

export async function uploadCase({ file, caseName, citation, year, court, jurisdiction, legalIssues, lawyerName }) {
  const form = new FormData();
  form.append("file", file);
  form.append("case_name", caseName);
  form.append("citation", citation || "");
  form.append("year", year || "");
  form.append("court", court || "");
  form.append("jurisdiction", jurisdiction || "federal");
  form.append("legal_issues", legalIssues || "");
  form.append("lawyer_name", lawyerName || "");

  const res = await fetch(`${API_BASE}/past-cases/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Case upload failed: ${text.slice(0, 300)}`);
  }
  return res.json();
}

export async function fetchUploadedCases() {
  const res = await fetch(`${API_BASE}/past-cases/list`);
  if (!res.ok) throw new Error("Could not load case list");
  const data = await res.json();
  return data.cases || [];
}