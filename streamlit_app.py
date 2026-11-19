from pathlib import Path

import pandas as pd
import streamlit as st
from transformers import pipeline


st.set_page_config(
    page_title="Lingua AI - analiza tekstu",
    page_icon="🌍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding: 1.2rem 0 0.4rem;
    }
    .index-footer {
        text-align: center;
        color: #7c8591;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #e6e8eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header"><h1>🌍 Lingua AI</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='text-align:center;'>Narzędzie do tłumaczenia i analizy emocji tekstu</h3>",
    unsafe_allow_html=True,
)

st.info(
    "Aplikacja wykorzystuje modele Hugging Face do tłumaczenia tekstu z języka "
    "angielskiego na niemiecki oraz do oceny wydźwięku emocjonalnego."
)

with st.expander("📖 Jak korzystać z aplikacji?", expanded=True):
    st.markdown(
        """
        1. Wybierz jedną z dwóch dostępnych funkcji.
        2. Wpisz lub wklej tekst w języku angielskim.
        3. Naciśnij przycisk analizy.
        4. Poczekaj na przygotowanie modelu i odczytaj wynik.

        Przy pierwszym uruchomieniu model musi zostać pobrany z Hugging Face,
        dlatego pierwsze działanie może potrwać dłużej.
        """
    )


@st.cache_resource(show_spinner=False)
def load_pipeline(task: str, model_name: str):
    return pipeline(task, model=model_name)


option = st.radio(
    "Wybierz funkcję:",
    [
        "🇬🇧 → 🇩🇪 Tłumaczenie angielski → niemiecki",
        "😊 Analiza wydźwięku emocjonalnego (angielski)",
    ],
    horizontal=True,
)

text = st.text_area(
    "Wpisz tekst:",
    height=150,
    max_chars=5000,
    placeholder="Enter text in English...",
)

if option.startswith("🇬🇧"):
    action_label = "Przetłumacz tekst"
else:
    action_label = "Analizuj tekst"

if st.button(action_label, type="primary", use_container_width=True):
    if not text or not text.strip():
        st.warning("Wpisz tekst przed uruchomieniem analizy.")
    else:
        try:
            if option.startswith("🇬🇧"):
                with st.spinner("Ładowanie modelu tłumaczenia i przetwarzanie tekstu..."):
                    translator = load_pipeline(
                        "translation",
                        "Helsinki-NLP/opus-mt-en-de",
                    )
                    result = translator(text.strip(), max_length=512)

                translated_text = result[0]["translation_text"]
                st.success("Tłumaczenie zostało zakończone.")
                st.subheader("Wynik tłumaczenia")
                st.markdown(
                    f"""
                    <div style="padding:1rem;border-radius:0.5rem;background:#f0f2f6;">
                    {translated_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Ładowanie modelu analizy i ocenianie tekstu..."):
                    classifier = load_pipeline(
                        "sentiment-analysis",
                        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                    )
                    result = classifier(text.strip())[0]

                label = result["label"].lower()
                score = result["score"]
                if label == "positive":
                    st.success(
                        f"Wynik: **POZYTYWNY** (pewność: {score:.1%})"
                    )
                else:
                    st.error(
                        f"Wynik: **NEGATYWNY** (pewność: {score:.1%})"
                    )
                st.progress(score, text=f"Pewność modelu: {score:.1%}")
        except Exception as exc:
            st.error(
                "Nie udało się przetworzyć tekstu. Sprawdź połączenie z internetem "
                "i spróbuj ponownie."
            )
            with st.expander("Szczegóły techniczne błędu"):
                st.code(str(exc))

st.divider()

st.subheader("📊 Dane przykładowe")
data_path = Path(__file__).resolve().parent / "DSP_4.csv"
try:
    df = pd.read_csv(data_path, sep=";")
    st.caption(f"Wczytano {len(df)} wierszy z pliku DSP_4.csv.")
    st.dataframe(df, use_container_width=True)
except FileNotFoundError:
    st.error("Nie znaleziono pliku DSP_4.csv obok pliku aplikacji.")
except pd.errors.ParserError:
    st.error("Nie udało się odczytać pliku DSP_4.csv.")

st.markdown(
    '<div class="index-footer"><strong>Numer indeksu: s30073</strong></div>',
    unsafe_allow_html=True,
)
