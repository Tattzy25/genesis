import { useVoiceAgent } from "@cloudflare/voice/react";
import { createRoot } from "react-dom/client";
import "./styles.css";

function App() {
  const {
    status,
    transcript,
    interimTranscript,
    metrics,
    isMuted,
    startCall,
    endCall,
    toggleMute
  } = useVoiceAgent({ agent: "ChatAgent" });

  return (
    <div className="p-8 font-sans">
      <h1 className="text-2xl font-bold mb-4">Voice Agent</h1>
      <p className="mb-4">
        Status: <span className="font-mono">{status}</span>
      </p>

      <div className="flex gap-4 mb-6">
        <button
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
          onClick={status === "idle" ? startCall : endCall}
        >
          {status === "idle" ? "Start Call" : "End Call"}
        </button>
        {status !== "idle" && (
          <button
            className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition"
            onClick={toggleMute}
          >
            {isMuted ? "Unmute" : "Mute"}
          </button>
        )}
      </div>

      {interimTranscript && (
        <p className="mb-4 text-gray-500">
          <em className="italic">{interimTranscript}</em>
        </p>
      )}

      <div className="space-y-2 mb-6">
        {transcript.map((msg, i) => (
          <p key={i} className="p-2 rounded bg-gray-100">
            <strong className="uppercase text-xs mr-2">{msg.role}:</strong>{" "}
            {msg.text}
          </p>
        ))}
      </div>

      {metrics && (
        <p className="text-xs text-gray-400 font-mono">
          LLM: {metrics.llm_ms}ms | TTS: {metrics.tts_ms}ms | First audio:{" "}
          {metrics.first_audio_ms}ms
        </p>
      )}
    </div>
  );
}

const root = createRoot(document.getElementById("root")!);
root.render(<App />);
