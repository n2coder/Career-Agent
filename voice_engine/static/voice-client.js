/**
 * voice-client.js — portable real-time voice pipeline client
 *
 * HOW TO USE
 * ----------
 * 1. Copy voice-client.js and vad-worklet.js to your static folder.
 * 2. Include in HTML: <script src="/static/voice-client.js"></script>
 * 3. Serve vad-worklet.js at /static/vad-worklet.js
 * 4. Instantiate:
 *
 *    const voice = new VoiceClient({
 *      wsUrl: "ws://localhost:8000/ws/voice",
 *      vadWorkletUrl: "/static/vad-worklet.js",
 *      onTranscript: (text) => console.log("User said:", text),
 *      onAudioChunk: (audioB64, sentence) => playAudio(audioB64),
 *      onTextDelta: (delta) => appendToUI(delta),
 *      onTurnEnd: (text) => console.log("Turn done:", text),
 *      onError: (msg) => console.error(msg),
 *      onStatusChange: (status) => updateUI(status),
 *    });
 *
 *    // Start listening
 *    await voice.start();
 *
 *    // Stop
 *    voice.stop();
 *
 *    // Mute/unmute
 *    voice.setMuted(true);
 *
 *    // Programmatic query (bypass STT)
 *    voice.sendQuery("suggest an action movie");
 */

const VOICE_CLIENT_VERSION = "1.0.0";

// ---------------------------------------------------------------------------
// Platform-adaptive VAD config
// ---------------------------------------------------------------------------
const _VAD_PROFILES = {
  desktop: {
    silenceMs: 700, vadIntervalMs: 100, bargeHoldMs: 80, bargeStrongHoldMs: 140,
    bargeMinAudioMs: 150, energyBase: 0.002, calibrationMs: 700,
    calibrationMultiplier: 1.18, echoSuppressMs: 300, bargeDynamicMultiplier: 1.06,
    bargeAssistantFloorMultiplier: 1.12, bargeStrongMultiplier: 1.05,
    voiceRatioThreshold: 0.40, backchannelMaxMs: 350, bargeInMinMs: 1500,
  },
  android: {
    silenceMs: 750, vadIntervalMs: 100, bargeHoldMs: 90, bargeStrongHoldMs: 150,
    bargeMinAudioMs: 160, energyBase: 0.0018, calibrationMs: 900,
    calibrationMultiplier: 1.16, echoSuppressMs: 340, bargeDynamicMultiplier: 1.07,
    bargeAssistantFloorMultiplier: 1.14, bargeStrongMultiplier: 1.06,
    voiceRatioThreshold: 0.38, backchannelMaxMs: 380, bargeInMinMs: 1600,
  },
  ios: {
    silenceMs: 800, vadIntervalMs: 100, bargeHoldMs: 95, bargeStrongHoldMs: 160,
    bargeMinAudioMs: 170, energyBase: 0.0016, calibrationMs: 1000,
    calibrationMultiplier: 1.18, echoSuppressMs: 360, bargeDynamicMultiplier: 1.08,
    bargeAssistantFloorMultiplier: 1.16, bargeStrongMultiplier: 1.07,
    voiceRatioThreshold: 0.38, backchannelMaxMs: 400, bargeInMinMs: 1700,
  },
};

function _detectVadProfile() {
  const ua = navigator.userAgent || "";
  if (/iPad|iPhone|iPod/i.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1)) return "ios";
  if (/Android/i.test(ua)) return "android";
  return "desktop";
}

// ---------------------------------------------------------------------------
// VoiceClient
// ---------------------------------------------------------------------------
class VoiceClient {
  constructor(opts = {}) {
    this._opts = opts;
    this._ws = null;
    this._sessionToken = "";
    this._stream = null;
    this._audioCtx = null;
    this._analyser = null;
    this._vadWorkletNode = null;
    this._mediaRecorder = null;
    this._chunks = [];
    this._isRecording = false;
    this._isMuted = false;
    this._awaitingTurn = false;
    this._sendingAudio = false;
    this._vadInterval = null;
    this._assistantAudioQueue = [];
    this._assistantPlaying = false;
    this._audioStartedAt = 0;
    this._suppressBargeUntil = 0;
    this._speakingSince = 0;
    this._captureSpeechSince = 0;
    this._speechCaptureStartedAt = 0;
    this._speechCaptureLastVoiceAt = 0;
    this._dynamicThreshold = 0;
    this._assistantEnergyFloor = 0;
    this._smoothedVoiceRatio = 1.0;
    this._lastTickSpeaking = false;
    this._backchannelBurstStart = 0;
    this._useWorkletVad = false;
    this._profile = _detectVadProfile();
    this._VAD = _VAD_PROFILES[this._profile];
    this._calibrating = false;
    this._calibrationStart = 0;
    this._calibrationSamples = [];
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  async start() {
    this._stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    await this._setupAudioPipeline();
    this._connectWs();
  }

  stop() {
    this._vadInterval && clearInterval(this._vadInterval);
    this._mediaRecorder && this._stopRecorder();
    this._audioCtx && this._audioCtx.state !== "closed" && this._audioCtx.close();
    this._stream && this._stream.getTracks().forEach(t => t.stop());
    this._ws && this._ws.close();
    this._ws = null;
    this._setStatus("stopped");
  }

  setMuted(muted) {
    this._isMuted = muted;
    if (muted && this._isRecording) this._stopRecorder();
    this._setStatus(muted ? "muted" : "listening");
  }

  sendQuery(text) {
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    this._ws.send(JSON.stringify({
      type: "user_query", query: text,
      lang_hint: "en", session_token: this._sessionToken,
    }));
  }

  stopAssistant() {
    this._assistantAudioQueue = [];
    this._assistantPlaying = false;
    this._audioStartedAt = 0;
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify({ type: "barge_in" }));
    }
  }

  // ------------------------------------------------------------------
  // WebSocket
  // ------------------------------------------------------------------

  _connectWs() {
    const ws = new WebSocket(this._opts.wsUrl);
    this._ws = ws;
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "start_session", session_token: this._sessionToken }));
      this._setStatus("connected");
    };
    ws.onmessage = (e) => this._handleWsMessage(JSON.parse(e.data));
    ws.onclose = () => { this._setStatus("disconnected"); setTimeout(() => this._connectWs(), 2000); };
    ws.onerror = () => ws.close();
  }

  _handleWsMessage(msg) {
    switch (msg.type) {
      case "session_started":
        this._sessionToken = msg.session_token || this._sessionToken;
        this._setStatus("listening");
        break;
      case "turn_started":
        this._awaitingTurn = true;
        this._opts.onTurnStarted?.(msg.query || "");
        this._setStatus("processing");
        break;
      case "text_delta":
        this._opts.onTextDelta?.(msg.delta || "");
        break;
      case "audio_chunk":
        this._assistantAudioQueue.push({ b64: msg.audio_b64, sentence: msg.sentence });
        if (!this._assistantPlaying) this._drainAudioQueue();
        break;
      case "turn_end":
        this._awaitingTurn = false;
        this._opts.onTurnEnd?.(msg.text || "");
        this._setStatus("listening");
        break;
      case "barge_in_ack":
        this._awaitingTurn = false;
        this._assistantAudioQueue = [];
        break;
      case "error":
        this._awaitingTurn = false;
        this._sendingAudio = false;
        this._opts.onError?.(msg.detail || "unknown error");
        this._setStatus("listening");
        break;
    }
  }

  // ------------------------------------------------------------------
  // Audio pipeline
  // ------------------------------------------------------------------

  async _setupAudioPipeline() {
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    this._audioCtx = ctx;
    const source = ctx.createMediaStreamSource(this._stream);

    // Try AudioWorklet VAD
    try {
      await ctx.audioWorklet.addModule(this._opts.vadWorkletUrl || "/static/vad-worklet.js");
      this._vadWorkletNode = new AudioWorkletNode(ctx, "LumaVadProcessor");
      this._vadWorkletNode.port.onmessage = (e) => {
        if (e.data?.type === "vad_frame") {
          this._smoothedVoiceRatio = this._smoothedVoiceRatio * 0.85 + (e.data.voice_ratio ?? 1) * 0.15;
        }
      };
      source.connect(this._vadWorkletNode);
      this._vadWorkletNode.connect(ctx.destination);
      this._useWorkletVad = true;
    } catch (_) {
      // Fallback: plain AnalyserNode
      this._analyser = ctx.createAnalyser();
      this._analyser.fftSize = 512;
      source.connect(this._analyser);
    }

    // Calibrate noise floor
    this._dynamicThreshold = this._VAD.energyBase;
    this._calibrating = true;
    this._calibrationStart = Date.now();
    this._calibrationSamples = [];

    this._vadInterval = setInterval(() => this._vadTick(), this._VAD.vadIntervalMs);
    this._setStatus("calibrating");
  }

  _getMicEnergy() {
    if (!this._analyser && !this._vadWorkletNode) return 0;
    if (this._analyser) {
      const buf = new Float32Array(this._analyser.fftSize);
      this._analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      return Math.sqrt(sum / buf.length);
    }
    return 0; // worklet path: energy is tracked via smoothedVoiceRatio
  }

  _assistantSpeaking() {
    return this._assistantPlaying || this._assistantAudioQueue.length > 0;
  }

  // ------------------------------------------------------------------
  // VAD tick
  // ------------------------------------------------------------------

  _vadTick() {
    if (this._isMuted) return;
    const now = Date.now();
    const energy = this._getMicEnergy();
    const VAD = this._VAD;

    // Calibration
    if (this._calibrating) {
      this._calibrationSamples.push(energy);
      if (now - this._calibrationStart >= VAD.calibrationMs) {
        this._calibrating = false;
        const avg = this._calibrationSamples.reduce((a, b) => a + b, 0) / this._calibrationSamples.length;
        this._dynamicThreshold = Math.max(VAD.energyBase, avg * VAD.calibrationMultiplier);
        this._setStatus("listening");
      }
      return;
    }

    const voiceRatioOk = !this._useWorkletVad || this._smoothedVoiceRatio >= (VAD.voiceRatioThreshold ?? 0.40);
    const speaking = energy > this._dynamicThreshold && voiceRatioOk;

    // Backchannel filler on falling edge during assistant speech
    if (this._lastTickSpeaking && this._assistantSpeaking() && this._backchannelBurstStart) {
      const burst = now - this._backchannelBurstStart;
      if (burst > 0 && burst < (VAD.backchannelMaxMs ?? 350)) {
        this._opts.onBackchannel?.();
      }
      this._backchannelBurstStart = 0;
    }

    if (speaking) {
      this._lastTickSpeaking = true;
      if (this._assistantSpeaking()) {
        if (energy < this._dynamicThreshold * VAD.bargeDynamicMultiplier) {
          this._assistantEnergyFloor = this._assistantEnergyFloor
            ? this._assistantEnergyFloor * 0.88 + energy * 0.12 : energy;
        }
        const bargeLevel = Math.max(
          this._dynamicThreshold * VAD.bargeDynamicMultiplier,
          this._assistantEnergyFloor * VAD.bargeAssistantFloorMultiplier
        );
        const strongVoice = energy > bargeLevel * 1.08;
        if (!this._speakingSince) this._speakingSince = now;
        if (!this._backchannelBurstStart) this._backchannelBurstStart = now;
        const burstDuration = now - this._backchannelBurstStart;
        const fullBargeIn = burstDuration >= (VAD.bargeInMinMs ?? 1500);
        if (
          this._audioStartedAt && now - this._audioStartedAt >= VAD.bargeMinAudioMs &&
          now >= this._suppressBargeUntil && strongVoice && fullBargeIn
        ) {
          this._backchannelBurstStart = 0;
          this.stopAssistant();
          this._beginCapture();
          this._speakingSince = 0;
          return;
        }
        if (!strongVoice) { this._speakingSince = 0; this._backchannelBurstStart = 0; }
      } else {
        this._speakingSince = 0;
        if (!this._captureSpeechSince) this._captureSpeechSince = now;
        if (!this._awaitingTurn && !this._sendingAudio && !this._isRecording && now - this._captureSpeechSince >= 400) {
          this._beginCapture();
        }
      }
      if (this._isRecording) this._speechCaptureLastVoiceAt = now;
    } else {
      this._lastTickSpeaking = false;
      this._speakingSince = 0;
      this._captureSpeechSince = 0;
      if (!this._assistantSpeaking() && this._isRecording && !this._awaitingTurn) {
        const silence = now - this._speechCaptureLastVoiceAt;
        if (silence >= VAD.silenceMs) this._flushCapture();
      }
    }
  }

  // ------------------------------------------------------------------
  // MediaRecorder capture
  // ------------------------------------------------------------------

  _beginCapture() {
    if (!window.MediaRecorder || this._isRecording || this._isMuted) return;
    const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"].find(m => MediaRecorder.isTypeSupported(m)) || "";
    this._chunks = [];
    const mr = new MediaRecorder(this._stream, mime ? { mimeType: mime } : {});
    this._mediaRecorder = mr;

    mr.ondataavailable = (e) => {
      if (!e.data?.size) return;
      this._chunks.push(e.data);
      // Send partial chunk for Deepgram streaming
      if (this._ws?.readyState === WebSocket.OPEN && this._sessionToken) {
        const reader = new FileReader();
        reader.onload = () => {
          const b64 = reader.result?.split(",")[1];
          if (b64) this._ws.send(JSON.stringify({ type: "audio_chunk_partial", audio_b64: b64, lang_hint: "en", session_token: this._sessionToken }));
        };
        reader.readAsDataURL(e.data);
      }
    };

    mr.onstop = async () => {
      const blob = new Blob(this._chunks, { type: mime || "audio/webm" });
      this._chunks = [];
      await this._submitAudio(blob);
    };

    this._isRecording = true;
    mr.start(500); // 500ms timeslice for partial chunks
    this._speechCaptureStartedAt = Date.now();
    this._speechCaptureLastVoiceAt = Date.now();
    this._setStatus("recording");
  }

  _flushCapture() {
    const duration = Date.now() - this._speechCaptureStartedAt;
    if (duration < 800) { this._stopRecorder(); return; } // too short
    this._stopRecorder();
  }

  _stopRecorder() {
    this._isRecording = false;
    if (this._mediaRecorder && this._mediaRecorder.state !== "inactive") {
      try { this._mediaRecorder.stop(); } catch (_) {}
    }
    this._mediaRecorder = null;
  }

  async _submitAudio(blob) {
    if (!blob.size || this._sendingAudio || !this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    this._sendingAudio = true;
    this._awaitingTurn = true;
    this._setStatus("processing");
    try {
      const b64 = await this._blobToB64(blob);
      this._ws.send(JSON.stringify({
        type: "user_audio",
        audio_b64: b64,
        mime_type: blob.type || "audio/webm",
        lang_hint: "en",
        session_token: this._sessionToken,
      }));
    } finally {
      this._sendingAudio = false;
    }
  }

  // ------------------------------------------------------------------
  // Audio playback
  // ------------------------------------------------------------------

  async _drainAudioQueue() {
    if (this._assistantPlaying || !this._assistantAudioQueue.length) return;
    this._assistantPlaying = true;
    while (this._assistantAudioQueue.length) {
      const { b64, sentence } = this._assistantAudioQueue.shift();
      this._setStatus("speaking");
      await this._playAudioB64(b64, sentence);
    }
    this._assistantPlaying = false;
    this._audioStartedAt = 0;
    this._suppressBargeUntil = Date.now() + (this._VAD.echoSuppressMs || 300);
    this._setStatus("listening");
  }

  async _playAudioB64(b64, sentence) {
    return new Promise((resolve) => {
      const audio = new Audio("data:audio/mpeg;base64," + b64);
      this._audioStartedAt = Date.now();
      this._opts.onAudioChunk?.(b64, sentence);
      audio.onended = resolve;
      audio.onerror = resolve;
      audio.play().catch(resolve);
    });
  }

  // ------------------------------------------------------------------
  // Utils
  // ------------------------------------------------------------------

  _blobToB64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  _setStatus(status) {
    this._opts.onStatusChange?.(status);
  }
}

// Export for module environments
if (typeof module !== "undefined") module.exports = { VoiceClient };
