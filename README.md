# Outfit Vibe Rater — Streamlit App

A fast, offline prototype that analyzes an outfit photo and produces **vibe scores**:

- **Maximalist Vibe**
- **Pattern Density**
- **Fit Tightness** (higher = tighter)
- **Colorfulness**
- **Visual Complexity**
- **Boldness**

No training required. Uses classical CV heuristics + optional MediaPipe segmentation.

---

## 1) Create & activate a virtual environment (macOS)

```bash
python3 --version
# If this prints 3.9+ you're good.

# Create project folder
mkdir -p ~/Projects/outfit-vibe-rater
cd ~/Projects/outfit-vibe-rater

# (optional) If you have Homebrew Python:  brew install python@3.11

# Create venv
python3 -m venv .venv
source .venv/bin/activate
```

> When you're done later: `deactivate`

## 2) Download this app’s code

1. Click the ZIP link I provided in chat and save it.
2. Unzip it and move the folder contents **into** your current directory (~/Projects/outfit-vibe-rater).  
   You should see `app.py`, `requirements.txt`, `assets/`, `utils/`, `styles.css` here.

## 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> If you face issues with `mediapipe`, the app still works without it.

## 4) Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## 5) Demo tips

- Keep **Show analysis maps** OFF for a cleaner demo.
- Click **Generate Shareable Report** to hand judges a one-pager.
- Prepare 3–4 photos (casual, formal, patterned, minimalist) to show range.

## 6) What’s under the hood?

- **Complexity** → Edge density via Canny.
- **Pattern density** → Local texture entropy.
- **Colorfulness** → Hasler & Süsstrunk metric.
- **Fit tightness** → Person silhouette solidity using MediaPipe segmentation (fallback heuristic if not available).
- **Vibe scores** → Weighted mix of the above into intuitive, explainable numbers.

## 7) Customize the look

- Edit `styles.css` (rounded chips, badges, etc.).
- Replace `assets/logo.svg` with your own logo.
- Change the title/subtitle in `app.py` to match your team branding.

---

### FAQ

**Q: Is this the same as the original notebooks?**  
A: The original GitHub repo needs training and datasets. For a hackathon, this app provides a **no-training prototype** with well-motivated CV heuristics. You can mention future work to swap in trained models.

**Q: Can I package it?**  
A: Yes. Use `pip freeze > requirements-lock.txt` and share the folder, or `pyinstaller` to make a one-file binary (optional).

**Q: Will it run offline?**  
A: Yes. Everything runs locally; `mediapipe` is optional.
