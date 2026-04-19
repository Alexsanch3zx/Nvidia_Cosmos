import os
import tempfile

import cv2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from auth import render_user_sidebar, require_login
from db.supabase_storage import (
    build_object_key,
    content_type_for_filename,
    is_storage_configured,
    upload_local_file_to_video_bucket,
)
from db.video_store import insert_summary
from embeddings.embedder import embed_text
from model_handler import CosmosModelHandler
from summarys.ollama_summarizer import summarize_frames_with_ollama
from summarys.summary_templates import (
    DEFAULT_VISION_MODEL_LABEL,
    parse_template_id_from_summary,
    style_key_from_label,
)
from video_processor import VideoProcessor
from vision_search import build_search_text

st.set_page_config(
    page_title="Upload & summarize",
    page_icon="🎥",
    layout="wide",
)

if "processed" not in st.session_state:
    st.session_state.processed = False
if "summary" not in st.session_state:
    st.session_state.summary = None
if "frames" not in st.session_state:
    st.session_state.frames = None

require_login()
render_user_sidebar()

st.title("🎥 Video Summarizer with Cosmos AI")
st.markdown(
    "Upload a video to get an AI-generated summary using Nvidia's Cosmos-reason2-8b model. "
    "Use **Semantic search** in the sidebar to find saved clips by meaning and watch them."
)
st.sidebar.caption("Other pages: open **Semantic search** below.")

st.sidebar.header("Configuration")
frame_interval = st.sidebar.slider(
    "Frame Sampling Interval (seconds)",
    min_value=1,
    max_value=10,
    value=2,
    help="Extract one frame every N seconds",
)

max_frames = st.sidebar.slider(
    "Maximum Frames to Analyze",
    min_value=5,
    max_value=50,
    value=20,
    help="Limit total frames to avoid overwhelming the model",
)

summary_style = st.sidebar.selectbox(
    "Summary Style",
    ["Detailed", "Concise", "Bullet Points"],
    help="Choose how you want the summary formatted",
)

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV",
    )

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("🚀 Generate Summary", type="primary"):
            with st.spinner("Processing video..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        video_path = tmp_file.name

                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration_sec = (total_frames / fps) if fps and fps > 0 else None
                    cap.release()

                    st.info("Step 1/3: Extracting frames from video...")
                    processor = VideoProcessor()
                    frames, timestamps = processor.extract_frames(
                        video_path,
                        interval_seconds=frame_interval,
                        max_frames=max_frames,
                    )
                    st.session_state.frames = frames
                    st.success(f"✓ Extracted {len(frames)} frames")

                    st.info("Step 2/3: Analyzing frames with Cosmos AI...")
                    model_handler = CosmosModelHandler()
                    frame_descriptions = model_handler.analyze_frames(frames)
                    st.success(f"✓ Analyzed {len(frame_descriptions)} frames")

                    st.info("Step 3/3: Generating video summary...")
                    style_key = style_key_from_label(summary_style)
                    summary = summarize_frames_with_ollama(
                        frame_descriptions,
                        timestamps,
                        style=style_key,
                    )
                    st.session_state.summary = summary

                    st.info("Uploading video file (if configured) and saving to database...")
                    storage_object_path: str | None = None
                    if is_storage_configured():
                        try:
                            key = build_object_key(
                                str(st.session_state.username or "user"),
                                getattr(uploaded_file, "name", None),
                            )
                            upload_local_file_to_video_bucket(
                                video_path,
                                key,
                                content_type=content_type_for_filename(getattr(uploaded_file, "name", None)),
                            )
                            storage_object_path = key
                            st.success("✓ Video stored in Supabase Storage")
                        except Exception as upload_exc:
                            st.warning(f"Could not upload video to Storage: {upload_exc}")
                    else:
                        st.info(
                            "Storage upload skipped: set **SUPABASE_URL** and "
                            "**SUPABASE_SERVICE_ROLE_KEY** to save the video file to your bucket."
                        )

                    try:
                        search_text = build_search_text(summary, frame_descriptions)
                        embedding = embed_text(search_text)
                        insert_summary(
                            filename=getattr(uploaded_file, "name", None),
                            duration_sec=duration_sec,
                            summary_style=style_key,
                            summary_text=summary,
                            embedding=embedding,
                            summary_engine="ollama",
                            vision_model=os.getenv("COSMOS_MODEL_LABEL", DEFAULT_VISION_MODEL_LABEL),
                            template_id=parse_template_id_from_summary(summary),
                            search_text=search_text,
                            storage_object_path=storage_object_path,
                        )
                        st.success("✓ Saved summary to database")
                    except Exception as e:
                        st.warning(f"Saved summary, but database insert failed: {e}")
                    st.session_state.processed = True
                    st.success("✓ Summary generated successfully!")

                    os.unlink(video_path)

                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    st.exception(e)

with col2:
    st.header("Summary Results")

    if st.session_state.processed and st.session_state.summary:
        st.markdown("### 📝 Video Summary")
        st.markdown(st.session_state.summary)

        st.download_button(
            label="📥 Download Summary",
            data=st.session_state.summary,
            file_name="video_summary.txt",
            mime="text/plain",
        )

        if st.session_state.frames and len(st.session_state.frames) > 0:
            with st.expander("🖼️ View Sample Frames"):
                num_to_show = min(6, len(st.session_state.frames))
                cols = st.columns(3)
                for idx in range(num_to_show):
                    with cols[idx % 3]:
                        st.image(
                            st.session_state.frames[idx],
                            caption=f"Frame {idx + 1}",
                            use_container_width=True,
                        )
    else:
        st.info("Upload a video and click 'Generate Summary' to see results here.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Powered by Nvidia Cosmos-reason2-8b | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
