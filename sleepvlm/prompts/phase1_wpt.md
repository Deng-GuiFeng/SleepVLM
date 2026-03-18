# Task

Analyze a **30‑second PSG waveform image** and estimate the following features at each second:
* **EEG/EOG channels**: frequency band power (delta, theta, alpha, beta) in dB, plus signal amplitude (MAV) in µV
* **EMG channel (Chin)**: muscle tone (MAV) in µV

# Image Rendering Parameters

You must interpret the image according to the following fixed amplitude scales:
* **EEG (F4‑M1, C4‑M1, O2‑M1)**: –50 µV to +50 µV
* **EOG (LOC, ROC)**: –50 µV to +50 µV
* **Chin EMG**: –40 µV to +40 µV

**Channel‑to‑color mapping**: The waveforms are rendered with fixed colors:
* **F4‑M1**: yellow
* **C4‑M1**: green
* **O2‑M1**: red
* **LOC**: cyan
* **ROC**: magenta
* **Chin EMG**: blue

**Channel order**: from top to bottom: F4‑M1, C4‑M1, O2‑M1, LOC, ROC, Chin EMG.

**Note**: Some channels may be missing in the image. Only output channels that are visible in the image.

# Output Format

```json
{
  "F4‑M1": [[d,t,a,b,mav], ...],
  "C4‑M1": [[d,t,a,b,mav], ...],
  "O2‑M1": [[d,t,a,b,mav], ...],
  "LOC": [[d,t,a,b,mav], ...],
  "ROC": [[d,t,a,b,mav], ...],
  "Chin": [[mav], ...]
}
```

* **EEG/EOG channels** (F4‑M1, C4‑M1, O2‑M1, LOC, ROC): Each contains 30 arrays (seconds 1–30), with 5 values per array: **[delta, theta, alpha, beta, mav]**. The first 4 values are band powers in dB; the 5th value (mav) is the Mean Absolute Value of signal amplitude in µV. All values use 1 decimal place.
* **Chin EMG channel**: Contains 30 arrays (seconds 1–30), with 1 value per array: **[mav]** representing muscle tone via Mean Absolute Value in µV (1 decimal place).