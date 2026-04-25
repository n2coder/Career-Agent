import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

_log = logging.getLogger(__name__)

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
except Exception:  # pragma: no cover - optional dependency
    RTCPeerConnection = None  # type: ignore[assignment]
    RTCSessionDescription = None  # type: ignore[assignment]

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

# Maximum buffered audio per peer: 10 seconds of 48kHz mono int16 ≈ 960 KB
_MAX_PCM_FRAMES = 480  # at 100 frames/sec (10ms each), ~10 seconds


@dataclass
class WebRtcPeer:
    peer_id: str
    pc: Any
    audio_channel: Any | None = None
    last_seen: float = field(default_factory=lambda: time.time())
    # Uplink PCM buffer — populated when pcm_buffering is True
    pcm_frames: list = field(default_factory=list)
    pcm_sample_rate: int = 0
    pcm_channels: int = 1
    pcm_buffering: bool = False


class WebRtcBridge:
    def __init__(self) -> None:
        self._peers: dict[str, WebRtcPeer] = {}
        self._lock = asyncio.Lock()

    @property
    def is_available(self) -> bool:
        return RTCPeerConnection is not None and RTCSessionDescription is not None

    async def create_answer(self, sdp: str, type_: str) -> dict[str, str]:
        if not self.is_available:
            raise HTTPException(
                status_code=503,
                detail="WebRTC bridge not available. Install aiortc to enable WebRTC transport.",
            )

        pc = RTCPeerConnection()
        peer_id = uuid4().hex
        peer = WebRtcPeer(peer_id=peer_id, pc=pc)

        @pc.on("datachannel")
        def on_datachannel(channel):
            if channel.label not in {"audio_downlink", "luma-audio"}:
                return
            peer.audio_channel = channel
            peer.last_seen = time.time()

            @channel.on("close")
            def _on_close():
                peer.audio_channel = None

            @channel.on("message")
            def _on_message(_msg):
                peer.last_seen = time.time()

        @pc.on("track")
        def on_track(track):
            if track.kind != "audio":
                return

            async def _drain_audio():
                try:
                    while True:
                        frame = await track.recv()
                        peer.last_seen = time.time()
                        if peer.pcm_buffering and _NUMPY_AVAILABLE:
                            # frame.to_ndarray() → shape (channels, samples), dtype int16
                            peer.pcm_sample_rate = frame.sample_rate
                            peer.pcm_channels = len(frame.layout.channels)
                            peer.pcm_frames.append(frame.to_ndarray())
                            # Guard: stop buffering if buffer exceeds max length
                            if len(peer.pcm_frames) >= _MAX_PCM_FRAMES:
                                _log.warning(
                                    "WebRTC uplink buffer full for peer %s — auto-ending utterance",
                                    peer_id,
                                )
                                peer.pcm_buffering = False
                except Exception:
                    return

            asyncio.create_task(_drain_audio())

        @pc.on("connectionstatechange")
        async def on_state_change():
            state = pc.connectionState
            if state in {"failed", "closed", "disconnected"}:
                await self.close_peer(peer_id)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        async with self._lock:
            self._peers[peer_id] = peer

        return {
            "peer_id": peer_id,
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    async def close_peer(self, peer_id: str) -> None:
        async with self._lock:
            peer = self._peers.pop(peer_id, None)
        if not peer:
            return
        try:
            await peer.pc.close()
        except Exception:
            return

    async def _cleanup_stale_peers(self, stale_after_sec: float = 120.0) -> None:
        """Background task: evict WebRTC peers idle longer than stale_after_sec."""
        while True:
            await asyncio.sleep(60)
            cutoff = time.time() - stale_after_sec
            async with self._lock:
                stale_ids = [
                    pid for pid, peer in self._peers.items() if peer.last_seen < cutoff
                ]
            for pid in stale_ids:
                await self.close_peer(pid)

    def start_cleanup_task(self) -> "asyncio.Task[None]":
        return asyncio.create_task(self._cleanup_stale_peers())

    async def send_audio_chunk(self, peer_id: str, payload: dict) -> bool:
        if not peer_id:
            return False
        async with self._lock:
            peer = self._peers.get(peer_id)
        if not peer:
            return False
        channel = peer.audio_channel
        if not channel or getattr(channel, "readyState", "") != "open":
            return False
        try:
            channel.send(json.dumps(payload))
            peer.last_seen = time.time()
            return True
        except Exception:
            return False

    # --- WebRTC uplink (user mic → server STT) ---

    async def start_utterance(self, peer_id: str) -> bool:
        """Signal start of user utterance: begin buffering incoming PCM frames."""
        async with self._lock:
            peer = self._peers.get(peer_id)
        if not peer:
            return False
        peer.pcm_frames = []
        peer.pcm_sample_rate = 0
        peer.pcm_channels = 1
        peer.pcm_buffering = True
        return True

    async def end_utterance(self, peer_id: str) -> "tuple[bytes, int, int] | None":
        """Signal end of utterance: stop buffering and return WAV bytes.

        Returns (wav_bytes, sample_rate, channels) or None if no audio buffered.
        """
        async with self._lock:
            peer = self._peers.get(peer_id)
        if not peer:
            return None

        peer.pcm_buffering = False
        frames = peer.pcm_frames
        peer.pcm_frames = []

        if not frames or not _NUMPY_AVAILABLE:
            return None

        sample_rate = peer.pcm_sample_rate or 48000
        channels = peer.pcm_channels or 1

        try:
            wav_bytes = await asyncio.to_thread(
                _frames_to_wav, frames, sample_rate, channels
            )
            return (wav_bytes, sample_rate, channels)
        except Exception:
            _log.exception("Failed to convert WebRTC PCM frames to WAV for peer %s", peer_id)
            return None


def _frames_to_wav(frames: list, sample_rate: int, channels: int) -> bytes:
    """Convert a list of numpy int16 PCM arrays (shape: channels×samples) to WAV bytes."""
    # Concatenate along samples axis → shape (channels, total_samples)
    combined = np.concatenate(frames, axis=1)
    # Interleave channels: transpose to (total_samples, channels), then flatten
    interleaved = combined.T.flatten().astype(np.int16)
    pcm_bytes = interleaved.tobytes()

    num_samples = len(interleaved)
    byte_rate = sample_rate * channels * 2  # 16-bit = 2 bytes/sample
    block_align = channels * 2
    data_size = len(pcm_bytes)
    riff_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16,          # chunk size
        1,                    # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        16,                   # bits per sample
        b"data", data_size,
    )
    return header + pcm_bytes


_webrtc_bridge: WebRtcBridge | None = None


def get_webrtc_bridge() -> WebRtcBridge:
    global _webrtc_bridge
    if _webrtc_bridge is None:
        _webrtc_bridge = WebRtcBridge()
    return _webrtc_bridge
