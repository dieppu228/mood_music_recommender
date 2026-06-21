import { useMemo, useRef, useState } from "react";
import {
  Bot,
  ExternalLink,
  Headphones,
  Music2,
  Send,
  Sparkles,
  Trash2,
  UserRound,
} from "lucide-react";
import moodWave from "../assets/mood-wave.svg";

const welcomeMessage = {
  id: "welcome",
  role: "assistant",
  text: "Tell me how you feel. I will turn your mood into a listening queue.",
  recommendations: [],
};

const integratedPorts = new Set(["5173", "8000", "8002"]);

function apiUrl(path) {
  const configuredBase = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "");
  if (configuredBase) return `${configuredBase}${path}`;
  if (integratedPorts.has(window.location.port)) return path;
  return `http://127.0.0.1:8000${path}`;
}

function spotifyUrl(track) {
  if (track.spotify_url) return track.spotify_url;
  const query = [track.title, track.artist].filter(Boolean).join(" ");
  return query ? `https://open.spotify.com/search/${encodeURIComponent(query)}` : "";
}

function cleanAnswer(text = "") {
  return text
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replaceAll("**", "")
    .replaceAll("__", "");
}

function normalizeText(value = "") {
  return value
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim();
}

function Answer({ text, recommendations = [] }) {
  const lines = cleanAnswer(text).split(/\r?\n/);
  const matched = new Set();

  const renderedLines = lines.map((line, lineIndex) => {
    const normalizedLine = normalizeText(line);
    let trackIndex = recommendations.findIndex(
      (track, index) =>
        !matched.has(index) &&
        track.title &&
        normalizedLine.includes(normalizeText(track.title)),
    );

    if (trackIndex < 0 && /^\s*(?:[-*]|\d+[.)])\s+/.test(line)) {
      trackIndex = recommendations.findIndex((_, index) => !matched.has(index));
    }

    const track = trackIndex >= 0 ? recommendations[trackIndex] : null;
    if (track) matched.add(trackIndex);

    return (
      <p key={`${lineIndex}-${line}`}>
        {line}
        {track && (
          <>
            {" "}
            <SpotifyLink track={track} />
          </>
        )}
      </p>
    );
  });

  recommendations.forEach((track, index) => {
    if (matched.has(index)) return;
    renderedLines.push(
      <p key={`track-${track.song_id || index}`}>
        {track.title || "Untitled"} - {track.artist || "Unknown artist"}{" "}
        <SpotifyLink track={track} />
      </p>,
    );
  });

  return renderedLines;
}

function SpotifyLink({ track }) {
  const url = spotifyUrl(track);
  if (!url) return null;
  return (
    <a href={url} target="_blank" rel="noreferrer">
      link <ExternalLink size={13} aria-hidden="true" />
    </a>
  );
}

function Avatar({ role }) {
  const isUser = role === "user";
  return (
    <div className={`avatar ${isUser ? "avatar-user" : "avatar-bot"}`} aria-hidden="true">
      {isUser ? <UserRound size={23} /> : <Bot size={24} />}
    </div>
  );
}

function Message({ message }) {
  return (
    <article className={`message ${message.role}`}>
      <Avatar role={message.role} />
      <div className="bubble">
        {message.pending ? (
          <div className="typing" aria-label="VibeCue is thinking">
            <span />
            <span />
            <span />
          </div>
        ) : message.role === "assistant" ? (
          <Answer text={message.text} recommendations={message.recommendations} />
        ) : (
          <p>{message.text}</p>
        )}
      </div>
    </article>
  );
}

function TrackCard({ track, index }) {
  const tags = [...(track.mood || []), ...(track.genres || []), ...(track.tags || [])].slice(
    0,
    4,
  );

  return (
    <article className="track-card">
      <div className="track-number">{String(index + 1).padStart(2, "0")}</div>
      <div className="track-content">
        <h3>{track.title || "Untitled"}</h3>
        <p className="artist">{track.artist || "Unknown artist"}</p>
        {track.reason && <p className="reason">{track.reason}</p>}
        {tags.length > 0 && (
          <div className="tag-row">
            {tags.map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
          </div>
        )}
      </div>
      <a
        className="track-action"
        href={spotifyUrl(track)}
        target="_blank"
        rel="noreferrer"
        aria-label={`Open ${track.title || "track"} on Spotify`}
        title="Open on Spotify"
      >
        <Headphones size={19} />
      </a>
    </article>
  );
}

export default function App() {
  const [messages, setMessages] = useState([welcomeMessage]);
  const [recommendations, setRecommendations] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const messagesRef = useRef(null);

  const resultLabel = useMemo(
    () => `${recommendations.length} ${recommendations.length === 1 ? "track" : "tracks"}`,
    [recommendations.length],
  );

  function scrollMessages() {
    window.requestAnimationFrame(() => {
      if (messagesRef.current) {
        messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
      }
    });
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const message = input.trim();
    if (!message || loading) return;

    const pendingId = crypto.randomUUID();
    setInput("");
    setError("");
    setLoading(true);
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", text: message },
      { id: pendingId, role: "assistant", pending: true },
    ]);
    scrollMessages();

    try {
      const response = await fetch(apiUrl("/v1/chat"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, max_results: 5 }),
      });
      if (!response.ok) throw new Error(`API returned HTTP ${response.status}`);

      const payload = await response.json();
      const tracks = payload.recommendations || [];
      setRecommendations(tracks);
      setMessages((current) =>
        current.map((item) =>
          item.id === pendingId
            ? {
                id: pendingId,
                role: "assistant",
                text: payload.answer || "No answer returned.",
                recommendations: tracks,
              }
            : item,
        ),
      );
    } catch {
      setError("VibeCue could not reach the music service. Please try again.");
      setMessages((current) =>
        current.map((item) =>
          item.id === pendingId
            ? {
                id: pendingId,
                role: "assistant",
                text: "I could not reach the music service just now.",
                recommendations: [],
              }
            : item,
        ),
      );
    } finally {
      setLoading(false);
      scrollMessages();
    }
  }

  function clearChat() {
    setMessages([welcomeMessage]);
    setRecommendations([]);
    setError("");
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <Music2 size={23} />
          </div>
          <div>
            <h1>VibeCue</h1>
            <p>Mood music assistant</p>
          </div>
        </div>
        <button className="icon-button" type="button" onClick={clearChat} title="Clear chat">
          <Trash2 size={19} />
          <span className="sr-only">Clear chat</span>
        </button>
      </header>

      <main>
        <section className="hero" aria-label="VibeCue mood visual">
          <img src={moodWave} alt="" />
          <div className="hero-overlay" />
          <div className="hero-copy">
            <span>
              <Sparkles size={15} /> Tune into your mood
            </span>
            <h2>What should today sound like?</h2>
            <p>Describe a feeling, moment, or atmosphere.</p>
          </div>
        </section>

        <div className="workspace-grid">
          <section className="chat-panel" aria-label="Chat with VibeCue">
            <div className="panel-heading">
              <div>
                <span className="online-dot" />
                VibeCue is ready
              </div>
            </div>

            <div className="messages" ref={messagesRef} aria-live="polite">
              {messages.map((message) => (
                <Message key={message.id} message={message} />
              ))}
            </div>

            {error && <div className="error-banner">{error}</div>}

            <form className="composer" onSubmit={handleSubmit}>
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    event.currentTarget.form.requestSubmit();
                  }
                }}
                rows={2}
                placeholder="Tell VibeCue how you feel..."
                aria-label="Your message"
              />
              <button type="submit" disabled={loading || !input.trim()} title="Send message">
                <Send size={19} />
                <span>Send</span>
              </button>
            </form>
          </section>

          <aside className="recommendations-panel" aria-label="Recommended tracks">
            <div className="section-heading">
              <div>
                <span className="section-icon">
                  <Headphones size={18} />
                </span>
                <div>
                  <h2>Your queue</h2>
                  <p>{resultLabel}</p>
                </div>
              </div>
            </div>

            {recommendations.length > 0 ? (
              <div className="recommendation-list">
                {recommendations.map((track, index) => (
                  <TrackCard key={track.song_id || `${track.title}-${index}`} track={track} index={index} />
                ))}
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-icon">
                  <Music2 size={25} />
                </div>
                <h3>Your next favorites live here</h3>
                <p>Start a conversation to build your queue.</p>
              </div>
            )}
          </aside>
        </div>
      </main>
    </div>
  );
}
