# Triton Neurotech — EMG BCI Workshop

Control a game with your muscles using a MindRove armband and Python.

---

## What You'll Build

A real-time EMG (electromyography) pipeline that:
1. Reads muscle signals from a MindRove 4-channel WiFi armband
2. Processes the signal to detect muscle contractions
3. Uses contractions as input to control a jump game and reaction-time challenge

---

## Hardware Setup

### Wearing the Armband

1. Put the MindRove armband on your **forearm**, about 2–3 finger-widths below the elbow
2. The electrodes should sit snugly against the skin — not too tight, not loose
3. Make a fist and feel where the muscles bulge — that's where you want the sensors
4. If signal is weak, try rotating the band slightly or moistening the skin

### Connecting to Wi-Fi

The MindRove armband creates its own Wi-Fi hotspot.

1. Power on the armband (hold the button until the LED blinks)
2. On your laptop, connect to the Wi-Fi network: **`MindRove_XXXX`** (the ID is on the band)
3. Password: `mindrove` (default)
4. Your laptop will now be connected to the armband's network — internet won't work while connected

---

## Software Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs: `mindrove`, `numpy`, `pygame`, `pylsl`, `scipy`

### 2. Install MindRove Connect (recommended — LSL stream)

MindRove Connect is a small app that connects to the armband and publishes an LSL stream your Python code can read.

**Download:** https://github.com/MindRove/MindRoveSDK/releases  
(look for `MindRoveConnect` under the latest release for your OS)

**Steps:**
1. Open MindRove Connect
2. Make sure you're connected to the armband's Wi-Fi
3. Click **"Start LSL Stream"**
4. You should see a green status indicator — the stream is now live
5. Leave MindRove Connect running in the background

> **Why LSL?** LSL (Lab Streaming Layer) is a standard protocol for real-time biosignal streaming. Using it means your Python code doesn't need to manage the hardware connection directly — MindRove Connect handles that.

### 3. Run the workshop

```bash
python main.py
```

Or, if you don't have the hardware yet, run in synthetic mode:

```bash
python main.py --synthetic
```

---

## Running the App

### Controls

| Key | Action |
|-----|--------|
| `C` | Calibrate (do this first!) |
| `G` | Jump game |
| `R` | Reaction time challenge |
| `M` | Main menu |
| `1`–`4` | Switch active EMG channel |
| `SPACE` | Manual trigger (test without hardware) |
| `ESC` | Quit |

### Calibration (important!)

Calibration teaches the app the difference between your resting muscle signal and a flex. Always calibrate before playing.

1. Press `C` from the main menu
2. **Rest phase (5 s):** relax your arm completely — don't move
3. **Flex phase (5 s):** repeatedly clench your fist firmly
4. The app computes a threshold and shows you a summary
5. If the threshold looks too high or too low, press `C` to re-calibrate

### Selecting the Right Channel

After connecting, the terminal prints a channel signal report like:

```
[EMGInput] Channel signal levels (higher = more signal):
           CH1 (idx  1):   12.3 µV  ██  ← try this
           CH2 (idx  2):    3.1 µV
           CH3 (idx  3):   45.6 µV  █████████  ← try this
           CH4 (idx  4):    8.2 µV  █
```

Press `1`–`4` to switch to the channel with the highest signal. Channels marked `← try this` are good candidates.

---

## Architecture Overview

```
MindRove armband
      │
      ▼  (Wi-Fi)
MindRove Connect app  ──►  LSL stream  ──►  emg_input.py
                                                  │
                                            signal_processing.py
                                            (rectify + smooth → µV scalar)
                                                  │
                                            controller.py
                                            (threshold + debounce → trigger)
                                                  │
                                    ┌─────────────┴──────────────┐
                                  game.py                  reaction.py
                               (jump game)            (reaction challenge)
```

| File | What it does |
|------|-------------|
| `emg_input.py` | Connects to LSL or SDK, buffers raw EMG samples |
| `signal_processing.py` | Rectifies + smooths signal into a single activation value (µV) |
| `calibration.py` | Records rest/flex windows, computes detection threshold |
| `controller.py` | Converts activation scalar → clean trigger with debounce & hysteresis |
| `game.py` | Endless runner jump game |
| `reaction.py` | Reaction time challenge with leaderboard |
| `main.py` | Main menu, wires everything together |

---

## Workshop Notebook

Open `workshop.ipynb` for a guided, hands-on walkthrough where you'll:
- Read and plot live EMG data
- Implement signal processing from scratch
- Build your own threshold detector
- Connect it to the full pipeline

```bash
jupyter notebook workshop.ipynb
```

---

## Troubleshooting

**No LSL stream found**
- Make sure MindRove Connect is running and shows a green status
- Check you're connected to the armband's Wi-Fi, not your normal network
- Try restarting MindRove Connect and clicking "Start LSL Stream" again

**Signal is flat / very noisy**
- The armband may be loose — tighten it slightly
- Try moistening the skin under the electrodes
- Switch to a different channel (keys `1`–`4`)

**Triggers firing on their own (too sensitive)**
- Re-calibrate: press `C` and relax more during the rest phase
- Or press `↑`/`↓` in the main menu to adjust threshold manually

**App crashes on import**
- Run `pip install -r requirements.txt` again
- Make sure you're using Python 3.10+: `python --version`
