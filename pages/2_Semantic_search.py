import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from auth import render_user_sidebar, require_login
from db.search_video import search_similar_by_text
from db.supabase_storage import try_public_video_url

st.set_page_config(
    page_title="Semantic search",
    page_icon="🔎",
    layout="wide",
)

require_login()
render_user_sidebar()

st.title("🔎 Semantic search")
st.markdown(
    "Describe an incident in plain language. We match your text to saved summaries "
    "(and the frame captions stored with them) using the same embedding model as upload."
)

search_query = st.text_input(
    "Search",
    placeholder="e.g. red light violation with pedestrian in crosswalk",
    help="Uses cosine distance in pgvector; lower distance is a closer match.",
)

limit = st.slider("Max results", 3, 25, 10)

if search_query and search_query.strip():
    search_failed = False
    try:
        with st.spinner("Embedding query and searching saved videos..."):
            results = search_similar_by_text(search_query, limit=limit)
    except Exception as e:
        search_failed = True
        results = []
        st.error(f"Search failed (check SUPABASE_DB_URL and pgvector): {e}")

    if results:
        st.caption("Lower **cosine distance** = closer match. Videos play when a Storage path exists.")
        for i, r in enumerate(results, start=1):
            filename = r.get("filename") or "Unknown file"
            summary_text = r.get("summary_text") or ""
            distance = r.get("distance")
            storage_key = r.get("storage_object_path")

            st.subheader(f"{i}. {filename}")
            if distance is not None:
                st.caption(f"Cosine distance: {distance:.4f}")

            video_url = try_public_video_url(storage_key) if storage_key else None
            if video_url:
                st.video(video_url)
            elif storage_key:
                st.warning(
                    "This row has a storage path but **SUPABASE_URL** is not set, "
                    "so the player URL cannot be built."
                )
                st.code(storage_key)
            else:
                st.info("No video file on record for this row (upload was skipped or before Storage was enabled).")

            with st.expander("Summary text"):
                st.markdown(summary_text)
            st.divider()
    elif not search_failed:
        st.info("No saved rows matched. Process a video on the Upload page first.")
else:
    st.caption("Type a query above to search archived clips.")
