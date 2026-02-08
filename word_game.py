import streamlit as st
import random
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Word Guessing Game", layout="centered")

# -------------------------------
# LOAD WORDS
# -------------------------------
@st.cache_data
def load_words():
    with open("words.txt", "r", encoding="utf-8") as f:
        return [w.strip().lower() for w in f if w.strip()]

WORDS = load_words()

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------------
# SESSION INIT (NO AUTO RESET)
# -------------------------------
if "secret_word" not in st.session_state:
    st.session_state.secret_word = random.choice(WORDS)
    st.session_state.secret_embedding = model.encode(st.session_state.secret_word)
    st.session_state.guesses = []
    st.session_state.game_won = False
    st.session_state.hint_shown = False
    st.session_state.guess_input = ""

# -------------------------------
# MANUAL RESET ONLY
# -------------------------------
def reset_game():
    st.session_state.secret_word = random.choice(WORDS)
    st.session_state.secret_embedding = model.encode(st.session_state.secret_word)
    st.session_state.guesses = []
    st.session_state.game_won = False
    st.session_state.hint_shown = False
    st.session_state.guess_input = ""

# -------------------------------
# UTILS
# -------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def temperature(sim):
    if sim >= 0.85:
        return "🔥 HOTTEST"
    elif sim >= 0.70:
        return "🔥 HOT"
    elif sim >= 0.50:
        return "🌤 WARM"
    else:
        return "❄ COLD"

def rank(sim):
    return 0 if sim >= 0.98 else round((1 - sim) * 100, 2)

# -------------------------------
# COMPACT UI STYLE
# -------------------------------
st.markdown("""
<style>
.card {
    padding: 8px;
    border-radius: 10px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 6px;
    font-size: 14px;
}
.hint {
    color: #fbbf24;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.title("Word Guessing Game")

# -------------------------------
# INPUT (AUTO CLEAR FIX)
# -------------------------------
# ---------- INPUT STATE INIT ----------
if "guess_input" not in st.session_state:
    st.session_state.guess_input = ""

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# ---------- CLEAR INPUT BEFORE WIDGET ----------
if st.session_state.clear_input:
    st.session_state.guess_input = ""
    st.session_state.clear_input = False

# ---------- INPUT WIDGET ----------
guess = st.text_input(
    "Your guess",
    key="guess_input",
    disabled=st.session_state.game_won
)

# ---------- GUESS BUTTON ----------
if st.button("Guess", disabled=st.session_state.game_won) and guess:
    guess = guess.lower().strip()

    emb = model.encode(guess)
    sim = cosine_similarity(emb, st.session_state.secret_embedding)

    entry = {
        "word": guess,
        "rank": rank(sim),
        "temp": temperature(sim),
        "correct": sim >= 0.98
    }

    st.session_state.guesses.append(entry)

    # ✅ REQUEST INPUT CLEAR (SAFE WAY)
    st.session_state.clear_input = True

    if entry["correct"]:
        st.session_state.game_won = True



# -------------------------------
# HARD HINT (AFTER 5 GUESSES)
# -------------------------------
if len(st.session_state.guesses) >= 5 and not st.session_state.hint_shown:
    st.markdown(
        f"<div class='card hint'>Hint: "
        f"{st.session_state.secret_word[0]}••• "
        f"({len(st.session_state.secret_word)} letters)</div>",
        unsafe_allow_html=True
    )
    st.session_state.hint_shown = True

# -------------------------------
# DISPLAY GUESSES (COMPACT)
# -------------------------------
if st.session_state.guesses:
    sorted_guesses = sorted(
        st.session_state.guesses,
        key=lambda x: x["rank"]
    )

    last = st.session_state.guesses[-1]

    st.markdown("**Latest Guess**")
    if last["correct"]:
        st.markdown(
            f"<div class='card'>✅ {last['word']} | Rank 0 | 🔥 HOTTEST</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='card'>{last['word']} | Rank {last['rank']} | {last['temp']}</div>",
            unsafe_allow_html=True
        )

    st.markdown("**Closest Guesses**")
    for g in sorted_guesses[:9]:
        st.markdown(
            f"<div class='card'>{g['word']} → {g['rank']} → {g['temp']}</div>",
            unsafe_allow_html=True
        )

# -------------------------------
# RESET BUTTON (BOTTOM RIGHT)
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([8, 2])
with col2:
    if st.button("🔁 Reset Game"):
        reset_game()
