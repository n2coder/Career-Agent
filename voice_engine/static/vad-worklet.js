class LumaVadProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._samplesSinceEmit = 0;
    this._sumsq = 0;
    this._peak = 0;
    this._targetSamples = Math.max(1, Math.floor(sampleRate * 0.02)); // ~20ms

    // Spectral analysis: 256-point real FFT for voice-frequency ratio
    this._fftSize = 256;
    this._fftBuf = new Float32Array(this._fftSize);
    this._fftFill = 0;
    this._re = new Float32Array(this._fftSize);
    this._im = new Float32Array(this._fftSize);
    this._hannWin = new Float32Array(this._fftSize);
    for (let i = 0; i < this._fftSize; i++) {
      this._hannWin[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (this._fftSize - 1)));
    }
    // Frequency resolution and voice band bins (dynamic — handles 16kHz and 48kHz contexts)
    const binHz = sampleRate / this._fftSize;
    this._loBin = Math.max(1, Math.ceil(85 / binHz));    // 85 Hz
    this._hiBin = Math.min(this._fftSize / 2 - 1, Math.floor(3000 / binHz)); // 3000 Hz
    // Latest computed voice ratio; default permissive so cold-start doesn't suppress speech
    this._voiceRatio = 1.0;
  }

  // Iterative in-place Cooley-Tukey radix-2 FFT (real input → complex output in _re/_im)
  _fft() {
    const N = this._fftSize;
    const re = this._re;
    const im = this._im;

    // Bit-reversal permutation
    let j = 0;
    for (let i = 1; i < N; i++) {
      let bit = N >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }

    // Butterfly stages
    for (let len = 2; len <= N; len <<= 1) {
      const halfLen = len >> 1;
      const ang = (-2 * Math.PI) / len;
      const wRe = Math.cos(ang);
      const wIm = Math.sin(ang);
      for (let i = 0; i < N; i += len) {
        let curRe = 1.0;
        let curIm = 0.0;
        for (let k = 0; k < halfLen; k++) {
          const uRe = re[i + k];
          const uIm = im[i + k];
          const vRe = re[i + k + halfLen] * curRe - im[i + k + halfLen] * curIm;
          const vIm = re[i + k + halfLen] * curIm + im[i + k + halfLen] * curRe;
          re[i + k] = uRe + vRe;
          im[i + k] = uIm + vIm;
          re[i + k + halfLen] = uRe - vRe;
          im[i + k + halfLen] = uIm - vIm;
          const nextRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = nextRe;
        }
      }
    }
  }

  _computeVoiceRatio() {
    const N = this._fftSize;
    const re = this._re;
    const im = this._im;

    // Apply Hann window then copy to FFT input
    for (let i = 0; i < N; i++) {
      re[i] = this._fftBuf[i] * this._hannWin[i];
      im[i] = 0;
    }
    this._fft();

    // Compute magnitude spectrum and voice-band / total-band energy
    let voiceEnergy = 0;
    let totalEnergy = 0;
    const half = N / 2;
    for (let k = 1; k < half; k++) {
      const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
      totalEnergy += mag;
      if (k >= this._loBin && k <= this._hiBin) voiceEnergy += mag;
    }
    return totalEnergy > 0 ? voiceEnergy / totalEnergy : 1.0;
  }

  process(inputs) {
    const input = inputs && inputs[0];
    if (!input || !input.length) return true;

    const channels = input.length;
    const frames = input[0].length || 0;
    if (!frames) return true;

    for (let i = 0; i < frames; i += 1) {
      let mixed = 0;
      for (let c = 0; c < channels; c += 1) {
        mixed += input[c][i] || 0;
      }
      mixed /= channels;
      const abs = Math.abs(mixed);
      this._sumsq += mixed * mixed;
      if (abs > this._peak) this._peak = abs;
      this._samplesSinceEmit += 1;

      // Accumulate FFT buffer
      this._fftBuf[this._fftFill++] = mixed;
      if (this._fftFill >= this._fftSize) {
        this._voiceRatio = this._computeVoiceRatio();
        this._fftFill = 0;
      }
    }

    if (this._samplesSinceEmit >= this._targetSamples) {
      const rms = Math.sqrt(this._sumsq / this._samplesSinceEmit);
      this.port.postMessage({
        type: "vad_frame",
        rms,
        peak: this._peak,
        voice_ratio: this._voiceRatio,
      });
      this._samplesSinceEmit = 0;
      this._sumsq = 0;
      this._peak = 0;
    }

    return true;
  }
}

registerProcessor("luma-vad-processor", LumaVadProcessor);
