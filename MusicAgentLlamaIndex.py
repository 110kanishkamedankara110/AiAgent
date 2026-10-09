
import asyncio
import os
import random
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from mutagen import File as MutagenFile
from rapidfuzz import fuzz, process

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "music_agent.db"

MUSIC_FOLDER = Path(
    os.getenv("MUSIC_FOLDER", r"E:\MOVIES,MUSIC\Music\Music")
)

API_KEY = os.getenv("LLM_MODEL_KEY")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")

AUDIO_EXTENSIONS = {
    ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav",
    ".wma", ".opus", ".aiff", ".ape"
}

mpv_process = None


# ============================================================
# CONVERSATION STATE
# ============================================================

@dataclass
class MusicContext:
    # These are real database records, not just chat text.
    last_results: list[dict] = field(default_factory=list)
    current_song: dict | None = None
    last_artist: str | None = None
    last_query: str | None = None
    active_playlist: list[dict] = field(default_factory=list)
    playlist_index: int = -1
    played_song_ids: set[int] = field(default_factory=set)


# ============================================================
# DATABASE AND METADATA
# ============================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialise_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                artist TEXT NOT NULL DEFAULT 'Unknown Artist',
                album TEXT NOT NULL DEFAULT 'Unknown Album',
                duration REAL DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS playlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS playlist_tracks (
                playlist_id INTEGER NOT NULL,
                song_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                PRIMARY KEY (playlist_id, position),
                FOREIGN KEY (playlist_id) REFERENCES playlists(id)
                    ON DELETE CASCADE,
                FOREIGN KEY (song_id) REFERENCES songs(id)
                    ON DELETE CASCADE
            )
        """)


def get_tag(audio, key, default):
    if not audio or not audio.tags:
        return default

    try:
        value = audio.tags.get(key)
        if value:
            if hasattr(value, "text"):
                value = value.text[0]
            elif isinstance(value, list):
                value = value[0]

            value = str(value).strip()
            if value:
                return value
    except (AttributeError, IndexError, TypeError):
        pass

    return default


def read_metadata(path: Path):
    title = path.stem
    artist = "Unknown Artist"
    album = "Unknown Album"
    duration = 0.0

    try:
        audio = MutagenFile(path, easy=True)

        if audio:
            title = get_tag(audio, "title", title)
            artist = get_tag(
                audio, "artist",
                get_tag(audio, "albumartist", artist)
            )
            album = get_tag(audio, "album", album)

            if audio.info:
                duration = float(
                    getattr(audio.info, "length", 0) or 0
                )

    except Exception as exc:
        print(f"Metadata warning for {path.name}: {exc}")

    return title, artist, album, duration


def refresh_database() -> str:
    """Scan the configured music folder and update the SQLite index."""
    if not MUSIC_FOLDER.exists():
        return f"Music folder does not exist: {MUSIC_FOLDER}"

    indexed = 0
    removed = 0

    with get_connection() as conn:
        for path in MUSIC_FOLDER.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue

            try:
                title, artist, album, duration = read_metadata(path)

                conn.execute("""
                    INSERT INTO songs
                        (path, title, artist, album, duration)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        title = excluded.title,
                        artist = excluded.artist,
                        album = excluded.album,
                        duration = excluded.duration
                """, (
                    str(path.resolve()),
                    title,
                    artist,
                    album,
                    duration
                ))

                indexed += 1

            except (OSError, sqlite3.Error) as exc:
                print(f"Could not index {path}: {exc}")

        rows = conn.execute("SELECT id, path FROM songs").fetchall()

        for row in rows:
            if not Path(row["path"]).is_file():
                conn.execute(
                    "DELETE FROM songs WHERE id = ?", (row["id"],)
                )
                removed += 1

    return (
        f"Library refreshed. Scanned {indexed} audio file(s); "
        f"removed {removed} missing file record(s)."
    )


def row_to_song(row):
    return {
        "id": row["id"],
        "path": row["path"],
        "title": row["title"],
        "artist": row["artist"],
        "album": row["album"],
        "duration": row["duration"] or 0
    }


def get_all_songs():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, path, title, artist, album, duration
            FROM songs
            ORDER BY artist, title
        """).fetchall()

    return [row_to_song(row) for row in rows]


def format_songs(songs, limit=15):
    if not songs:
        return "I couldn't find any matching songs in your library."

    lines = [f"I found {len(songs)} matching song(s):"]

    for index, song in enumerate(songs[:limit], start=1):
        lines.append(
            f"{index}. {song['title']} | "
            f"{song['artist']} | {song['album']}"
        )

    if len(songs) > limit:
        lines.append(f"...and {len(songs) - limit} more.")

    return "\n".join(lines)


def set_search_context(
    ctx: RunContext[MusicContext],
    songs: list[dict],
    query: str | None = None,
    artist: str | None = None
):
    ctx.deps.last_results = list(songs)
    ctx.deps.last_query = query

    if artist is not None:
        ctx.deps.last_artist = artist


# ============================================================
# SEARCH TOOLS
# ============================================================

def search_song_records(query: str, limit: int = 30):
    query = query.strip()

    if not query:
        return []

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, path, title, artist, album, duration
            FROM songs
            WHERE title LIKE ?
               OR artist LIKE ?
               OR album LIKE ?
               OR path LIKE ?
            ORDER BY title
            LIMIT 300
        """, (
            f"%{query}%",
            f"%{query}%",
            f"%{query}%",
            f"%{query}%"
        )).fetchall()

    results = [row_to_song(row) for row in rows]

    if not results:
        library = get_all_songs()

        choices = {
            f"{s['title']} {s['artist']} {s['album']} {s['path']}": s
            for s in library
        }

        matches = process.extract(
            query,
            choices.keys(),
            scorer=fuzz.WRatio,
            limit=limit
        )

        results = [
            choices[text]
            for text, score, _ in matches
            if score >= 65
        ]

    return results[:limit]


# ============================================================
# PLAYBACK
# ============================================================

def play_file(song: dict) -> str:
    global mpv_process

    path = Path(song["path"])

    if not path.is_file():
        return f"The audio file no longer exists: {path}"

    try:
        mpv = shutil.which("mpv")

        if mpv:
            if mpv_process and mpv_process.poll() is None:
                mpv_process.terminate()
                try:
                    mpv_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    mpv_process.kill()

            mpv_process = subprocess.Popen([
                mpv, "--no-terminal", str(path)
            ])
        else:
            # Windows opens the file in the default associated player.
            os.startfile(str(path))

        return f"Started playback: {song['title']} by {song['artist']}."

    except Exception as exc:
        return f"Playback failed: {exc}"


def play_selected_song(
    ctx: RunContext[MusicContext],
    song: dict
) -> str:
    result = play_file(song)

    if result.startswith("Started playback:"):
        ctx.deps.current_song = song
        ctx.deps.played_song_ids.add(song["id"])

        for index, item in enumerate(ctx.deps.active_playlist):
            if item["id"] == song["id"]:
                ctx.deps.playlist_index = index
                break

    return result


# ============================================================
# AGENT TOOLS
# ============================================================

def build_agent():
    model = OpenAIChatModel(
        MODEL_NAME,
        provider=OpenAIProvider(
            base_url="https://api.groq.com/openai/v1",
            api_key=API_KEY
        )
    )

    agent = Agent(
        model,
        deps_type=MusicContext,
        instructions="""
You are a friendly AI music assistant for the user's local Windows library.

Understand ordinary conversational language, spelling mistakes, follow-up
questions, and implied meaning. Do not require exact commands.

Use the tools to perform real operations. Do not pretend a song was played,
found, or saved if the tool did not confirm it.

CONVERSATION BEHAVIOUR:
- "What songs do I have?" -> call list_songs.
- "Do I have songs by Meghan Trainor?" -> call search_artist.
- "Play one you like", "pick one", "play one of them" -> play_from_context.
- "Play another one" -> play_another_song.
- "Play any song", "surprise me", "anything" -> play_any_song.
- "Play number 3" -> play_result_number with number 3.
- "Next" -> next_song.
- "Stop the music" -> stop_music.
- "Create a playlist called favourites" -> create a playlist from the current
  search results. If there are no results, ask what songs should go in it.
- "Play my favourites playlist" -> play_playlist.
- "Refresh my library" -> refresh_library.

SEARCH:
When a user asks for an artist, search artist metadata and filenames, not only
song titles. If no tracks are found, say so honestly.
Do not search for literal phrases such as "play one you like".
Use the remembered search results for follow-up requests.
For broad questions, answer naturally and avoid overwhelming the user with
long lists. If useful, show a few song examples.

PERSONALITY:
Be warm and concise, like a helpful friend who enjoys music. Avoid repeating
the entire command menu after each reply.
""",
    )

    @agent.tool
    def list_songs(
        ctx: RunContext[MusicContext],
        limit: int = 15
    ) -> str:
        """Show a sample of songs available in the music library."""
        songs = get_all_songs()
        set_search_context(ctx, songs, query=None)
        return (
            f"Your library contains {len(songs)} indexed song(s).\n"
            + format_songs(songs, max(1, min(limit, 30)))
        )

    @agent.tool
    def search_songs(
        ctx: RunContext[MusicContext],
        query: str,
        limit: int = 20
    ) -> str:
        """Search for tracks by title, artist, album, or filename."""
        songs = search_song_records(query, max(1, min(limit, 100)))
        set_search_context(ctx, songs, query=query)
        return format_songs(songs)

    @agent.tool
    def search_artist(
        ctx: RunContext[MusicContext],
        artist: str,
        limit: int = 50
    ) -> str:
        """Find songs by a particular artist, including artist names in filenames."""
        artist = artist.strip()

        if not artist:
            return "Please tell me which artist to search for."

        library = get_all_songs()

        # Prefer metadata matches.
        exact_matches = [
            song for song in library
            if artist.casefold() in song["artist"].casefold()
        ]

        # If tags are missing, try matching the artist in the file path.
        filename_matches = [
            song for song in library
            if artist.casefold() in song["path"].casefold()
        ]

        # Fuzzy match artist tags if no exact match is found.
        if not exact_matches and not filename_matches:
            artist_names = sorted({
                song["artist"] for song in library
                if song["artist"] != "Unknown Artist"
            })

            matches = process.extract(
                artist, artist_names,
                scorer=fuzz.WRatio, limit=5
            )

            matched_names = {
                name for name, score, _ in matches if score >= 78
            }

            exact_matches = [
                song for song in library
                if song["artist"] in matched_names
            ]

        # Merge results without duplicates, preserving artist-tag matches first.
        merged = []
        seen = set()

        for song in exact_matches + filename_matches:
            if song["id"] not in seen:
                merged.append(song)
                seen.add(song["id"])

        merged = merged[:max(1, min(limit, 100))]
        set_search_context(
            ctx, merged, query=artist, artist=artist
        )

        return format_songs(merged)

    @agent.tool
    def play_song(
        ctx: RunContext[MusicContext],
        query: str
    ) -> str:
        """Find and play a song by title or filename."""
        songs = search_song_records(query, limit=30)

        if not songs:
            return f"I couldn't find a song matching '{query}'."

        exact = [
            song for song in songs
            if song["title"].casefold() == query.casefold()
        ]

        song = exact[0] if exact else songs[0]
        set_search_context(ctx, songs, query=query)
        return play_selected_song(ctx, song)

    @agent.tool
    def play_result_number(
        ctx: RunContext[MusicContext],
        number: int
    ) -> str:
        """Play a numbered track from the most recent search results."""
        results = ctx.deps.last_results

        if not results:
            return "We haven't searched for songs yet. What would you like to hear?"

        if number < 1 or number > len(results):
            return f"Choose a number from 1 to {len(results)}."

        return play_selected_song(ctx, results[number - 1])

    @agent.tool
    def play_from_context(ctx: RunContext[MusicContext]) -> str:
        """Play a random song from the most recent search results."""
        results = ctx.deps.last_results

        if not results:
            return play_any_song(ctx)

        choices = [
            song for song in results
            if not ctx.deps.current_song
            or song["id"] != ctx.deps.current_song["id"]
        ]

        if not choices:
            choices = results

        return play_selected_song(ctx, random.choice(choices))

    @agent.tool
    def play_another_song(ctx: RunContext[MusicContext]) -> str:
        """Play a different song from the current results if possible."""
        results = ctx.deps.last_results
        current = ctx.deps.current_song

        if not results:
            return play_any_song(ctx)

        choices = [
            song for song in results
            if (not current or song["id"] != current["id"])
            and song["id"] not in ctx.deps.played_song_ids
        ]

        if not choices:
            choices = [
                song for song in results
                if not current or song["id"] != current["id"]
            ]

        if not choices:
            return "That's the only song in those results. Want to explore another artist?"

        return play_selected_song(ctx, random.choice(choices))

    @agent.tool
    def play_any_song(ctx: RunContext[MusicContext]) -> str:
        """Play a random song from the entire library."""
        songs = get_all_songs()

        if not songs:
            return "Your music library is empty. Ask me to refresh it first."

        song = random.choice(songs)
        set_search_context(ctx, songs, query=None)
        ctx.deps.active_playlist = []
        ctx.deps.playlist_index = -1

        return play_selected_song(ctx, song)

    @agent.tool
    def next_song(ctx: RunContext[MusicContext]) -> str:
        """Play the next song in the current playlist or search results."""
        if ctx.deps.active_playlist:
            playlist = ctx.deps.active_playlist
            next_index = (ctx.deps.playlist_index + 1) % len(playlist)
            ctx.deps.playlist_index = next_index
            return play_selected_song(ctx, playlist[next_index])

        return play_another_song(ctx)

    @agent.tool
    def stop_music(ctx: RunContext[MusicContext]) -> str:
        """Stop music playback when using MPV."""
        global mpv_process

        if mpv_process and mpv_process.poll() is None:
            mpv_process.terminate()
            try:
                mpv_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                mpv_process.kill()

            mpv_process = None
            return "Playback stopped."

        return (
            "I can stop playback directly when the song is running in MPV. "
            "If Windows opened another media player, stop it in that player."
        )

    @agent.tool
    def create_playlist(
        ctx: RunContext[MusicContext],
        name: str
    ) -> str:
        """Save the most recent search results as a named playlist."""
        name = name.strip()
        songs = ctx.deps.last_results

        if not name:
            return "What would you like to name the playlist?"

        if not songs:
            return (
                "I don't have any recent search results to save. "
                "First search for an artist or some songs."
            )

        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO playlists (name) VALUES (?)",
                (name,)
            )

            row = conn.execute(
                "SELECT id FROM playlists WHERE name = ?", (name,)
            ).fetchone()

            playlist_id = row["id"]

            conn.execute(
                "DELETE FROM playlist_tracks WHERE playlist_id = ?",
                (playlist_id,)
            )

            for position, song in enumerate(songs):
                conn.execute("""
                    INSERT INTO playlist_tracks
                        (playlist_id, song_id, position)
                    VALUES (?, ?, ?)
                """, (playlist_id, song["id"], position))

        return f"Saved playlist '{name}' with {len(songs)} song(s)."

    @agent.tool
    def list_playlists(ctx: RunContext[MusicContext]) -> str:
        """List all saved playlists and their song counts."""
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT p.name, COUNT(pt.song_id) AS count
                FROM playlists p
                LEFT JOIN playlist_tracks pt ON pt.playlist_id = p.id
                GROUP BY p.id
                ORDER BY p.name
            """).fetchall()

        if not rows:
            return "You don't have any saved playlists yet."

        return "Your playlists:\n" + "\n".join(
            f"- {row['name']} ({row['count']} songs)" for row in rows
        )

    @agent.tool
    def play_playlist(
        ctx: RunContext[MusicContext],
        name: str
    ) -> str:
        """Load a saved playlist and play its first song."""
        with get_connection() as conn:
            row = conn.execute("""
                SELECT id, name FROM playlists
                WHERE name LIKE ? LIMIT 1
            """, (f"%{name.strip()}%",)).fetchone()

            if not row:
                return f"I couldn't find a playlist called '{name}'."

            tracks = conn.execute("""
                SELECT s.id, s.path, s.title, s.artist, s.album, s.duration
                FROM playlist_tracks pt
                JOIN songs s ON s.id = pt.song_id
                WHERE pt.playlist_id = ?
                ORDER BY pt.position
            """, (row["id"],)).fetchall()

        songs = [row_to_song(track) for track in tracks]

        if not songs:
            return f"Your playlist '{row['name']}' is empty."

        ctx.deps.active_playlist = songs
        ctx.deps.playlist_index = 0
        set_search_context(ctx, songs, query=row["name"])

        return play_selected_song(ctx, songs[0])

    @agent.tool
    def refresh_library(ctx: RunContext[MusicContext]) -> str:
        """Scan the music folder for new, changed, or deleted files."""
        result = refresh_database()
        ctx.deps.last_results = []
        ctx.deps.current_song = None
        ctx.deps.last_artist = None
        ctx.deps.last_query = None
        ctx.deps.active_playlist = []
        ctx.deps.playlist_index = -1
        ctx.deps.played_song_ids.clear()
        return result

    return agent


# ============================================================
# CHAT LOOP WITH PERSISTENT MESSAGE HISTORY
# ============================================================

async def main():
    initialise_database()

    print("\n🎵 CONVERSATIONAL MUSIC ASSISTANT")
    print(f"📁 Music folder: {MUSIC_FOLDER}")

    if not API_KEY:
        print("\nMissing LLM_MODEL_KEY in your .env file.")
        return

    if not MUSIC_FOLDER.exists():
        print("\nMusic folder not found. Check MUSIC_FOLDER in .env.")
        return

    with get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]

    if count == 0:
        print("\nScanning your music folder for the first time...")
        print(refresh_database())
    else:
        print(f"\nLoaded {count} indexed tracks.")



    agent = build_agent()
    deps = MusicContext()
    message_history: list[ModelMessage] = []

    while True:
        try:
            user_input = input("\n🎵 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.casefold() in {"exit", "quit", "bye", "goodbye"}:
            print("🎶 Enjoy your music. See you next time!")
            break

        try:
            result = await agent.run(
                user_input,
                deps=deps,
                message_history=message_history,
            )

            # Preserve the complete exchange for the next turn.
            message_history = result.all_messages()

            print(f"\n🎧 Assistant: {result.output}")

        except Exception as exc:
            print(f"\n❌ I couldn't complete that request: {exc}")
            print(
                "Check your API key, model name, internet connection, "
                "and installed Pydantic AI version."
            )


if __name__ == "__main__":
    asyncio.run(main())