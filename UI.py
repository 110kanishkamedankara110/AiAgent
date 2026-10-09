
import asyncio
import html
import re
import threading
import traceback
from concurrent.futures import Future
from io import BytesIO
from pathlib import Path

import streamlit as st
from mutagen import File as MutagenFile
from PIL import Image

from MusicAgentLlamaIndex import build_agent, MusicContext


# ============================================================
# CONFIGURATION
# ============================================================

MAX_DISPLAY_SONGS = 15

st.set_page_config(
    page_title="Music Assistant",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================
# ICON URLS
# ============================================================

ICON_URLS = {
    "app": "https://img.icons8.com/fluency/96/headphones.png",
    "user": "https://img.icons8.com/fluency/96/user-male-circle.png",
    "assistant": "https://img.icons8.com/fluency/96/bot.png",
    "library": "https://img.icons8.com/fluency/96/music-library.png",
    "play": "https://img.icons8.com/fluency/96/circled-play.png",
    "refresh": "https://img.icons8.com/fluency/96/refresh.png",
    "stop": "https://img.icons8.com/fluency/96/stop.png",
    "clear": "https://img.icons8.com/fluency/96/trash.png",
    "track": "https://img.icons8.com/fluency/96/musical-notes.png",
}


# ============================================================
# UI STYLES
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080B12;
    --sidebar: #0B101B;
    --surface: #101725;
    --surface-hover: #151F30;
    --border: #202B3D;
    --text: #EAF0FA;
    --muted: #8D9AB0;
    --accent: #00D6B2;
    --accent-hover: #35E9CD;
    --accent-soft: rgba(0, 214, 178, 0.12);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            ellipse at 50% -20%,
            rgba(0, 214, 178, 0.07),
            transparent 45%
        ),
        var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1120px;
    padding: 2rem 2rem 7rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stMainBlockContainer"] {
    gap: 1.25rem;
}

h1, h2, h3 {
    color: var(--text);
    letter-spacing: -0.8px;
    font-weight: 700;
}

p, label, .stMarkdown {
    color: var(--text);
}

small, .muted {
    color: var(--muted);
}

hr {
    border-color: var(--border);
    opacity: 0.8;
}

[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.4rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text);
    letter-spacing: -0.4px;
}

[data-testid="stSidebar"] hr {
    border-color: var(--border);
    margin: 1.2rem 0;
}

[data-testid="stSidebar"] [data-testid="stImage"] img {
    object-fit: contain;
    border-radius: 8px;
}

.brand-title {
    font-size: 17px;
    font-weight: 700;
    color: #F0F5FF;
    letter-spacing: -0.5px;
}

.brand-subtitle {
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
}

.section-label {
    color: #77869D;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.4px;
    margin: 8px 0 12px;
}

.app-description {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.7;
    margin-top: -10px;
}

.stButton > button {
    min-height: 42px;
    background: #111A29;
    color: #DDE6F4;
    border: 1px solid var(--border);
    border-radius: 11px;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    transition: background 160ms ease, border-color 160ms ease;
}

.stButton > button:hover {
    background: var(--surface-hover);
    border-color: var(--accent);
    color: #FFFFFF;
}

[data-testid="stChatMessage"] {
    background: rgba(16, 23, 37, 0.82);
    border: 1px solid rgba(40, 53, 74, 0.8);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    margin: 0.7rem 0;
    gap: 0.85rem;
}

[data-testid="stChatMessage"]:hover {
    border-color: #2B3A50;
}

[data-testid="stChatMessage"] [data-testid="stAvatar"] {
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stChatMessage"] [data-testid="stAvatar"] img {
    object-fit: contain;
    padding: 3px;
}

[data-testid="stChatInput"] {
    background: transparent;
    padding-top: 0.8rem;
}

[data-testid="stChatInput"] > div {
    background: #101725;
    border: 1px solid #29374B;
    border-radius: 20px;
    padding: 5px 7px;
}

[data-testid="stChatInput"]:focus-within > div {
    background: #121B2B;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(0, 214, 178, 0.09);
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #F0F5FF !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 12px 13px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    caret-color: var(--accent);
    resize: none;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #77869D !important;
    opacity: 1;
}

[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    color: #07120F !important;
    border: none !important;
    border-radius: 12px !important;
    min-width: 38px;
    min-height: 38px;
}

[data-testid="stChatInput"] button:hover {
    background: var(--accent-hover) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, #111A29, #0D1420);
    border: 1px solid var(--border);
    border-radius: 15px;
    transition: border-color 180ms ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #35465C;
}

[data-testid="stImage"] img {
    display: block;
    object-fit: cover;
    border-radius: 10px;
}

[data-testid="stColumn"] {
    min-width: 0;
}

.song-title {
    color: #F0F5FF;
    font-size: 14px;
    font-weight: 600;
    line-height: 1.5;
    overflow-wrap: anywhere;
}

.song-artist {
    color: #A7B5C9;
    font-size: 12px;
    margin-top: 3px;
    line-height: 1.5;
    overflow-wrap: anywhere;
}

.song-details {
    color: #77869D;
    font-size: 11px;
    margin-top: 5px;
    line-height: 1.5;
    overflow-wrap: anywhere;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: var(--surface);
    border-color: var(--border);
    border-radius: 10px;
    color: var(--text);
}

[data-testid="stCaptionContainer"] {
    color: var(--muted);
    font-size: 12px;
}

::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: #28364A;
    border-radius: 20px;
}

::-webkit-scrollbar-thumb:hover {
    background: #40536B;
}

footer {
    visibility: hidden;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1.1rem 0.8rem 6rem;
    }

    [data-testid="stChatMessage"] {
        padding: 0.8rem;
        border-radius: 13px;
    }

    [data-testid="stChatInput"] textarea {
        font-size: 14px !important;
        padding: 10px !important;
    }

    .song-title {
        font-size: 13px;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# ASYNC RUNNER
# ============================================================

class AsyncRunner:
    """Run agent calls on one persistent asynchronous event loop."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
        )
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coroutine):
        future: Future = asyncio.run_coroutine_threadsafe(
            coroutine,
            self.loop,
        )
        return future.result()


@st.cache_resource
def get_runner():
    return AsyncRunner()


@st.cache_resource
def get_agent():
    return build_agent()


runner = get_runner()
agent = get_agent()


# ============================================================
# SESSION STATE
# ============================================================

if "music_context" not in st.session_state:
    st.session_state.music_context = MusicContext()

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================
# GENERAL HELPERS
# ============================================================

def get_field(song, *names, default=None):
    """Read a value from a dictionary or object."""
    if song is None:
        return default

    for name in names:
        if isinstance(song, dict):
            value = song.get(name)
        else:
            value = getattr(song, name, None)

        if value is not None and value != "":
            return value

    return default


def get_song_path(song):
    path = get_field(
        song,
        "path",
        "file_path",
        "filepath",
        "file",
        "uri",
        "url",
        "location",
        "filename",
    )

    if isinstance(path, Path):
        return str(path)

    return str(path) if path else None


def get_song_title(song):
    title = get_field(
        song,
        "title",
        "name",
        "track_name",
        "track",
    )

    if title:
        return str(title)

    path = get_song_path(song)
    if path:
        return Path(path).stem

    return "Unknown title"


def get_song_artist(song):
    artist = get_field(
        song,
        "artist",
        "artists",
        "performer",
        "album_artist",
    )

    if isinstance(artist, (list, tuple)):
        artist = ", ".join(str(item) for item in artist)

    if artist:
        return str(artist)

    path = get_song_path(song)
    if path and Path(path).is_file():
        try:
            audio = MutagenFile(path, easy=True)
            if audio:
                value = audio.get("artist")
                if isinstance(value, list) and value:
                    return str(value[0])
        except Exception:
            pass

    return "Unknown artist"


def get_song_album(song):
    album = get_field(song, "album", "album_name")

    if album:
        return str(album)

    path = get_song_path(song)
    if path and Path(path).is_file():
        try:
            audio = MutagenFile(path, easy=True)
            if audio:
                value = audio.get("album")
                if isinstance(value, list) and value:
                    return str(value[0])
        except Exception:
            pass

    return None


def get_duration(song):
    duration = get_field(
        song,
        "duration",
        "length",
        "duration_seconds",
    )

    try:
        seconds = int(float(duration))
    except (TypeError, ValueError):
        seconds = None

    if seconds is None:
        path = get_song_path(song)
        if path and Path(path).is_file():
            try:
                audio = MutagenFile(path)
                if audio and audio.info:
                    seconds = int(audio.info.length)
            except Exception:
                pass

    if seconds is None:
        return None

    return f"{seconds // 60}:{seconds % 60:02d}"


# ============================================================
# ALBUM ARTWORK
# ============================================================

def get_cover_art(song):
    """Extract embedded artwork or use supplied image metadata."""
    direct_cover = get_field(
        song,
        "cover",
        "cover_art",
        "artwork",
        "album_art",
        "image",
        "image_url",
        "thumbnail",
    )

    if isinstance(direct_cover, Image.Image):
        return direct_cover.copy()

    if isinstance(direct_cover, bytes):
        try:
            return Image.open(
                BytesIO(direct_cover)
            ).convert("RGB")
        except Exception:
            pass

    if isinstance(direct_cover, str):
        if direct_cover.startswith(("https://", "http://")):
            return direct_cover

        if Path(direct_cover).is_file():
            try:
                return Image.open(direct_cover).convert("RGB")
            except Exception:
                pass

    path = get_song_path(song)
    if not path or not Path(path).is_file():
        return None

    try:
        audio = MutagenFile(path)
        if audio is None:
            return None

        tags = getattr(audio, "tags", None)

        # MP3 artwork.
        if tags:
            for tag in tags.values():
                if getattr(tag, "FrameID", None) == "APIC":
                    return Image.open(
                        BytesIO(tag.data)
                    ).convert("RGB")

        # FLAC artwork.
        pictures = getattr(audio, "pictures", None)
        if pictures:
            return Image.open(
                BytesIO(pictures[0].data)
            ).convert("RGB")

        # MP4 / M4A artwork.
        if tags:
            for key in ("covr", "coverart"):
                value = tags.get(key)
                if value:
                    image_data = value[0]
                    if hasattr(image_data, "data"):
                        image_data = image_data.data

                    return Image.open(
                        BytesIO(bytes(image_data))
                    ).convert("RGB")

    except Exception:
        pass

    return None


def show_icon(name, width=22):
    url = ICON_URLS.get(name)
    if url:
        st.image(url, width=width)


# ============================================================
# SONG RESULT NORMALIZATION
# ============================================================

def is_song_object(value):
    if value is None:
        return False

    keys = (
        "title",
        "name",
        "track_name",
        "path",
        "file_path",
        "filepath",
        "file",
        "uri",
        "url",
    )

    if isinstance(value, dict):
        return any(key in value for key in keys)

    return any(
        getattr(value, key, None) is not None
        for key in keys
    )


def snapshot_songs(value):
    """
    Extract song objects from common result formats without
    treating ordinary strings or LLM prose as song metadata.
    """
    if value is None or isinstance(value, (str, bytes)):
        return []

    if isinstance(value, dict):
        if is_song_object(value):
            return [value]

        for key in (
            "songs",
            "tracks",
            "results",
            "items",
            "data",
            "last_results",
            "nodes",
        ):
            if key in value:
                songs = snapshot_songs(value[key])
                if songs:
                    return songs

        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            return snapshot_songs(metadata)

        return []

    if isinstance(value, (list, tuple, set)):
        songs = []
        for item in value:
            songs.extend(snapshot_songs(item))
        return songs

    # LlamaIndex NodeWithScore or similar wrappers.
    node = getattr(value, "node", None)
    if node is not None:
        metadata = getattr(node, "metadata", None)
        if metadata:
            songs = snapshot_songs(metadata)
            if songs:
                return songs

    if is_song_object(value):
        return [value]

    for key in (
        "songs",
        "tracks",
        "results",
        "items",
        "data",
        "last_results",
        "nodes",
    ):
        nested = getattr(value, key, None)
        if nested is not None:
            songs = snapshot_songs(nested)
            if songs:
                return songs

    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, dict):
        return snapshot_songs(metadata)

    return []


def limit_songs(songs, limit=MAX_DISPLAY_SONGS):
    """Deduplicate songs and cap the number of displayed cards."""
    songs = snapshot_songs(songs)

    result = []
    seen = set()

    for song in songs:
        path = get_song_path(song)

        identity = (
            path.casefold()
            if path
            else f"{get_song_artist(song)}|{get_song_title(song)}".casefold()
        )

        if identity in seen:
            continue

        seen.add(identity)
        result.append(song)

        if len(result) >= limit:
            break

    return result


# ============================================================
# SONG METADATA CARD
# ============================================================

def render_song(song, index=None, artwork_size=64):
    title = get_song_title(song)
    artist = get_song_artist(song)
    album = get_song_album(song)
    duration = get_duration(song)
    cover = get_cover_art(song)

    with st.container(border=True):
        artwork_col, metadata_col = st.columns(
            [0.9, 5],
            vertical_alignment="center",
            gap="small",
        )

        with artwork_col:
            if isinstance(cover, Image.Image):
                st.image(cover, width=artwork_size)
            elif isinstance(cover, str):
                st.image(cover, width=artwork_size)
            else:
                show_icon("track", width=artwork_size)

        with metadata_col:
            heading = f"{index}. {title}" if index is not None else title

            st.markdown(
                f'<div class="song-title">{html.escape(heading)}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="song-artist">{html.escape(artist)}</div>',
                unsafe_allow_html=True,
            )

            details = []
            if album:
                details.append(album)
            if duration:
                details.append(duration)

            if details:
                st.markdown(
                    f'<div class="song-details">'
                    f'{html.escape(" · ".join(details))}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ============================================================
# DISPLAY INTENT CLASSIFICATION
# ============================================================

def classify_display(prompt):
    """Classify listing requests before checking playback intent."""
    text = prompt.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    replacements = {
        "avilable": "available",
        "availble": "available",
        "availabe": "available",
        "listdown": "list down",
        "songss": "songs",
        "wht": "what",
        "whats": "what is",
        "pls": "please",
        "patty guardy": "patty gurdy",
    }

    for wrong, correct in replacements.items():
        text = re.sub(
            rf"\b{re.escape(wrong)}\b",
            correct,
            text,
        )

    # Any request to find or list songs should produce metadata cards.
    listing_patterns = [
        r"\b(list|show|display|find|search|browse|view)\b",
        r"\b(give|gimme|fetch|get)\b",
        r"\bavailable songs?\b",
        r"\bavailable tracks?\b",
        r"\b(any|some|all|more)\s+(of\s+)?(the\s+)?(songs?|tracks?)\b",
        r"\b(songs?|tracks?)\s+(by|from|of)\b",
        r"\b(music|albums?)\s+(by|from|of)\b",
        r"\banything\s+(by|from)\b",
        r"\bwhat\s+(songs?|tracks?|music)\b",
        r"\bwhich\s+(songs?|tracks?)\b",
        r"\brecommend(ation)?s?\b",
        r"\bsuggest(ions?)?\b",
        r"\bsimilar songs?\b",
        r"\b(song|track|artist|album)\s+list\b",
        r"\bmy music library\b",
        r"\bin (the|my) library\b",
        r"\bwhat do i have\b",
        r"\bwhat is in (the|my) library\b",
        r"\bplaylist\b",
    ]

    if any(re.search(pattern, text) for pattern in listing_patterns):
        return "song_list"

    now_playing_patterns = [
        r"\bnow playing\b",
        r"\bcurrently playing\b",
        r"\bcurrent song\b",
        r"\bwhat is playing\b",
        r"\bwhat song is playing\b",
    ]

    if any(re.search(pattern, text) for pattern in now_playing_patterns):
        return "now_playing"

    playback_patterns = [
        r"\bplay\b",
        r"\bshuffle\b",
        r"\bskip song\b",
        r"\bplay next\b",
        r"\bplay previous\b",
        r"\bnext song\b",
        r"\bprevious song\b",
        r"\bstop music\b",
        r"\bpause music\b",
        r"\bresume music\b",
    ]

    if any(re.search(pattern, text) for pattern in playback_patterns):
        return "now_playing"

    return "text"


# ============================================================
# PROCESS A USER MESSAGE
# ============================================================

def process_message(prompt):
    display_type = classify_display(prompt)

    st.session_state.chat_messages.append({
        "role": "user",
        "content": prompt,
    })

    try:
        with st.spinner("Working on your request..."):
            result = runner.run(
                agent.run(
                    prompt,
                    deps=st.session_state.music_context,
                    message_history=st.session_state.message_history,
                )
            )

        response = str(getattr(result, "output", result))

        if hasattr(result, "all_messages"):
            st.session_state.message_history = result.all_messages()

        ctx = st.session_state.music_context
        current_song = getattr(ctx, "current_song", None)
        last_results = getattr(ctx, "last_results", None)

        if display_type == "now_playing":
            songs = limit_songs(
                [current_song] if current_song else [],
                limit=1,
            )

        elif display_type == "song_list":
            # Prefer song objects stored by the agent's search tool.
            songs = limit_songs(
                last_results,
                limit=MAX_DISPLAY_SONGS,
            )

            # If the result itself contains structured song objects,
            # use them when the context has no results.
            if not songs:
                output = getattr(result, "output", None)
                songs = limit_songs(
                    output,
                    limit=MAX_DISPLAY_SONGS,
                )

            # The cards replace the model's numbered text list.
            if songs:
                response = ""
            else:
                response = (
                    "I couldn't find structured song metadata for this "
                    "request. The agent may have returned song names as "
                    "text instead of populating MusicContext.last_results."
                )

        else:
            songs = []

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response,
            "display_type": display_type,
            "songs": songs,
        })

    except Exception:
        traceback.print_exc()

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": (
                "Something went wrong while processing your request. "
                "Check the terminal for the error details."
            ),
            "display_type": "text",
            "songs": [],
        })


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    brand_icon, brand_text = st.columns(
        [0.8, 3.5],
        vertical_alignment="center",
    )

    with brand_icon:
        show_icon("app", width=38)

    with brand_text:
        st.markdown(
            '<div class="brand-title">Music Assistant</div>'
            '<div class="brand-subtitle">'
            'Your personal music library'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown(
        '<div class="section-label">QUICK ACTIONS</div>',
        unsafe_allow_html=True,
    )

    actions = [
        ("library", "Show available songs", "Show available songs in my music library."),
        ("play", "Play a random song", "Play a random song from my library."),
        ("refresh", "Refresh library", "Refresh and rescan my music library."),
        ("stop", "Stop music", "Stop the music."),
    ]

    for icon_name, label, prompt_text in actions:
        icon_col, button_col = st.columns(
            [0.65, 4],
            vertical_alignment="center",
        )

        with icon_col:
            show_icon(icon_name, width=22)

        with button_col:
            if st.button(
                label,
                key=f"action_{icon_name}",
                use_container_width=True,
            ):
                if icon_name == "refresh":
                    st.session_state.music_context = MusicContext()
                    st.session_state.message_history = []
                    st.session_state.chat_messages = []
                    st.session_state.pending_prompt = None
                    st.rerun()

                elif icon_name == "stop":
                    ctx = st.session_state.music_context
                    stop_method = getattr(ctx, "stop_music", None)

                    if callable(stop_method):
                        try:
                            stop_method()
                            st.toast("Playback stopped.")
                        except Exception as exc:
                            st.error(f"Could not stop playback: {exc}")
                    else:
                        st.session_state.pending_prompt = prompt_text

                else:
                    st.session_state.pending_prompt = prompt_text

    st.divider()

    st.markdown(
        '<div class="section-label">CONVERSATION</div>',
        unsafe_allow_html=True,
    )

    icon_col, button_col = st.columns(
        [0.65, 4],
        vertical_alignment="center",
    )

    with icon_col:
        show_icon("clear", width=22)

    with button_col:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.message_history = []
            st.rerun()

    st.divider()
    st.caption(f"Song previews are limited to {MAX_DISPLAY_SONGS} cards.")


# ============================================================
# MAIN HEADER
# ============================================================

header_icon, header_text = st.columns(
    [0.75, 5],
    vertical_alignment="center",
)

with header_icon:
    show_icon("app", width=42)

with header_text:
    st.title("Music Assistant")

st.markdown(
    '<div class="app-description">'
    'Explore your collection, find your favourite tracks, '
    'and discover something new.'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.chat_messages:
    avatar = (
        ICON_URLS["user"]
        if message["role"] == "user"
        else ICON_URLS["assistant"]
    )

    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "user":
            st.markdown(message.get("content", ""))
            continue

        display_type = message.get("display_type", "text")

        if display_type == "song_list":
            songs = limit_songs(
                message.get("songs", []),
                limit=MAX_DISPLAY_SONGS,
            )

            if songs:
                st.caption(f"Showing {len(songs)} songs")

                for index, song in enumerate(songs, start=1):
                    render_song(
                        song,
                        index=index,
                        artwork_size=64,
                    )

            elif message.get("content"):
                st.info(message["content"])

        elif display_type == "now_playing":
            songs = message.get("songs", [])

            if songs:
                render_song(
                    songs[0],
                    artwork_size=84,
                )
            elif message.get("content"):
                st.markdown(message["content"])
            else:
                st.info("No currently playing track was returned.")

        else:
            st.markdown(message.get("content", ""))


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input("Ask me about your music library...")

pending = st.session_state.pending_prompt

if pending:
    st.session_state.pending_prompt = None
    process_message(pending)
    st.rerun()

elif prompt:
    process_message(prompt)
    st.rerun()