const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' ? 'http://localhost:8000' : '';
const chatMessages = document.getElementById('chat-messages');

function appendMessage(role, text) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.innerHTML = `<div class="content">${text}</div>`;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

function sendPrompt(text) {
  document.getElementById('user-input').value = text;
  document.getElementById('chat-form').dispatchEvent(new Event('submit'));
}

async function handleSubmit(e) {
  e.preventDefault();
  const input = document.getElementById('user-input');
  const message = input.value.trim();
  if (!message) return;

  input.value = '';
  appendMessage('user', message);

  const assistantMsg = appendMessage('assistant', '<span class="typing">Thinking...</span>');
  const contentDiv = assistantMsg.querySelector('.content');
  contentDiv.innerText = '';

  let executionInfo = { intent: null, latency: null, sql: null, citations: [] };

  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.substring(6));
            if (data.type === 'token') {
              contentDiv.innerText += data.content;
            } else if (data.type === 'intent') {
              executionInfo.intent = data.intent;
            } else if (data.type === 'sql') {
              executionInfo.sql = data.query;
            } else if (data.type === 'done') {
              executionInfo.latency = data.latency_ms;
              executionInfo.citations = data.citations;
            }
          } catch (e) {
            // Ignore partial JSON chunks
          }
        }
      }
    }

    if (executionInfo.citations && executionInfo.citations.length > 0) {
      executionInfo.citations.forEach(cit => {
        contentDiv.innerHTML += ` <span class="citation-badge" title="${cit.snippet}">[${cit.index}: ${cit.source}]</span>`;
      });
    }

    const drawer = document.createElement('div');
    drawer.className = 'inspector-drawer';
    drawer.innerHTML = `
      <details>
        <summary>🔍 Inspect AI Execution (${executionInfo.latency || '0'}ms)</summary>
        <p><strong>Intent Detected:</strong> ${executionInfo.intent || 'GENERAL_CHAT'}</p>
        ${executionInfo.sql ? `<p><strong>Executed SQL:</strong> <code>${executionInfo.sql}</code></p>` : ''}
        <p><strong>Citations Tracked:</strong> ${executionInfo.citations.length}</p>
      </details>
    `;
    assistantMsg.appendChild(drawer);

  } catch (err) {
    contentDiv.innerText = 'Error connecting to backend service. Please check API status.';
  }
}
