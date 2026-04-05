"""
radio_bridge.py -- Faithful Python port of Asimov's Radio core modules.

Ports the four Radio modules from JavaScript to Python for research use:
  - FrustrationDetector
  - EmotionalArcTracker
  - InjectionComposer
  - SongStore
  - RadioSimulator (facade)

All constants, thresholds, decay rates, keyword lists, and state machine
transitions are preserved exactly from the JS originals.

Run tests:
    python radio_bridge.py --test
    python -m pytest radio_bridge.py -v
"""

import json
import math
import os
import re
import sys
import time
import uuid
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# FrustrationDetector
# ---------------------------------------------------------------------------

FRUSTRATION_KEYWORDS = [
    'failed', 'error', 'timeout', 'cannot', 'blocked', 'impossible',
    'stuck', 'broken', 'crash', 'exception', 'denied', 'rejected',
]

DESPERATION_PATTERNS = [
    re.compile(r'running out of', re.IGNORECASE),
    re.compile(r'need to (try|finish|fix) .*(fast|quick|now)', re.IGNORECASE),
    re.compile(r"why won't this", re.IGNORECASE),
    re.compile(r'nothing (is )?work', re.IGNORECASE),
    re.compile(r"I('m| am) stuck", re.IGNORECASE),
    re.compile(r'last (resort|chance|attempt)', re.IGNORECASE),
    re.compile(r'desperate', re.IGNORECASE),
    re.compile(r'no idea (what|how|why)', re.IGNORECASE),
]

WINDOW_SIZE = 20
DECAY_RATE = 0.95  # Per-event decay for frustration score


class FrustrationDetector:
    """Output-level heuristics for agent emotional state.

    Analyzes agent outputs, completion events, and mood signals for signs of
    escalating frustration. Produces a normalized 0-1 score that the arc
    tracker uses for mode transition decisions.
    """

    def __init__(self):
        self._window: list[dict] = []
        self._score: float = 0.0
        self._consecutive_failures: int = 0

    @property
    def score(self) -> float:
        return self._score

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def record_agent_completion(self, success: bool, output: str = '',
                                error: str = '') -> None:
        """Record an agent completion event."""
        now = time.time() * 1000  # ms timestamp like Date.now()
        delta = 0.0

        if not success:
            self._consecutive_failures += 1
            delta += 0.15
            delta += min(0.1 * self._consecutive_failures, 0.4)
        else:
            self._consecutive_failures = 0
            delta -= 0.2

        text = output or error or ''
        delta += self._analyze_text(text)

        self._push_event({
            'type': 'agent_completion',
            'frustrationDelta': delta,
            'timestamp': now,
        })
        self._recompute()

    def record_mood_change(self, mood: str, energy_level: Optional[float] = None) -> None:
        """Record a mood change from the sentiment engine."""
        now = time.time() * 1000
        delta = 0.0

        negative_moods = ['frustrated', 'stressed', 'angry', 'anxious']
        positive_moods = ['happy', 'excited', 'confident', 'calm']

        if mood in negative_moods:
            delta += 0.1
        elif mood in positive_moods:
            delta -= 0.15

        if isinstance(energy_level, (int, float)) and energy_level < 0.3:
            delta += 0.05

        self._push_event({
            'type': 'mood_change',
            'frustrationDelta': delta,
            'timestamp': now,
        })
        self._recompute()

    def record_retry(self) -> None:
        """Record a retry event."""
        self._push_event({
            'type': 'retry',
            'frustrationDelta': 0.1,
            'timestamp': time.time() * 1000,
        })
        self._recompute()

    def _analyze_text(self, text: str) -> float:
        if not text or not isinstance(text, str):
            return 0.0
        lower = text.lower()
        delta = 0.0

        keyword_hits = 0
        for kw in FRUSTRATION_KEYWORDS:
            if kw in lower:
                keyword_hits += 1
        delta += min(keyword_hits * 0.03, 0.15)

        for pattern in DESPERATION_PATTERNS:
            if pattern.search(text):
                delta += 0.08
                break

        # Sustained caps detection: two 3+ letter all-caps words
        if (re.search(r'\b[A-Z]{3,}\b', text) and
                re.search(r'[A-Z]{3,}\s+[A-Z]{3,}', text)):
            delta += 0.05

        return delta

    def _push_event(self, event: dict) -> None:
        self._window.append(event)
        if len(self._window) > WINDOW_SIZE:
            self._window.pop(0)

    def _recompute(self) -> None:
        score = 0.0
        now = time.time() * 1000
        for i in range(len(self._window) - 1, -1, -1):
            age = (now - self._window[i]['timestamp']) / 1000  # seconds
            weight = math.pow(DECAY_RATE, age / 10)
            score += self._window[i]['frustrationDelta'] * weight
        self._score = max(0.0, min(1.0, score))

    def get_state(self) -> dict:
        return {
            'score': self._score,
            'consecutiveFailures': self._consecutive_failures,
            'windowSize': len(self._window),
            'recentEvents': [
                {'type': e['type'], 'delta': e['frustrationDelta']}
                for e in self._window[-5:]
            ],
        }

    def reset(self) -> None:
        self._window = []
        self._score = 0.0
        self._consecutive_failures = 0


# ---------------------------------------------------------------------------
# EmotionalArcTracker
# ---------------------------------------------------------------------------

SUSTAINED_THRESHOLD = 3
FRUSTRATION_SHIFT_THRESHOLD = 0.6
CELEBRATION_COOLDOWN_MS = 2 * 60 * 1000  # 120000 ms
MODE_HISTORY_CAP = 100

VALENCE_FROM_MOOD = {
    'happy': 'uplifting', 'excited': 'uplifting', 'confident': 'uplifting',
    'calm': 'calming', 'relaxed': 'calming', 'peaceful': 'calming',
    'neutral': 'neutral', 'focused': 'neutral',
    'frustrated': 'intense', 'stressed': 'intense', 'angry': 'intense',
    'sad': 'melancholy', 'anxious': 'melancholy', 'tired': 'melancholy',
}


class EmotionalArcTracker:
    """Three-mode state machine for Asimov's Radio.

    Modes:
      MIRROR      -- Reflect the operator's current emotional state
      SHIFT       -- Lean toward emotional resolution
      CELEBRATION -- Milestone reinforcement
    """

    def __init__(self):
        self._current_mode: Optional[str] = None
        self._session_vibe: Optional[dict] = None
        self._mood_history: list[dict] = []
        self._frustration_history: list[dict] = []
        self._milestone_count: int = 0
        self._mode_history: list[dict] = []
        self._last_celebration_at: float = 0
        self._injection_count: int = 0
        self._current_valence: str = 'neutral'
        self._event_callback: Optional[Callable] = None
        self._forced: bool = False

    def initialize(self, event_callback: Optional[Callable] = None) -> None:
        """Initialize with an optional event callback (replaces JS eventBus)."""
        self._event_callback = event_callback

    @property
    def current_mode(self) -> Optional[str]:
        return self._current_mode

    @property
    def current_valence(self) -> str:
        return self._current_valence

    @property
    def session_vibe(self) -> Optional[dict]:
        return self._session_vibe

    @property
    def injection_count(self) -> int:
        return self._injection_count

    @property
    def milestone_count(self) -> int:
        return self._milestone_count

    @property
    def is_active(self) -> bool:
        return self._current_mode is not None

    def set_session_vibe(self, vibe: str,
                         initial_tags: Optional[list] = None) -> None:
        """Set the session baseline. Activates the arc tracker in Mirror mode."""
        self._session_vibe = {
            'vibe': vibe,
            'tags': initial_tags or [],
            'setAt': time.time() * 1000,
        }
        self._current_valence = self._vibe_to_valence(vibe)
        self._forced = False
        self._transition_to('mirror', 'session_baseline')

    def recalibrate(self, vibe: str) -> None:
        """Recalibrate the baseline mid-session."""
        if self._session_vibe:
            self._session_vibe = {
                **self._session_vibe,
                'vibe': vibe,
                'recalibratedAt': time.time() * 1000,
            }
        else:
            self._session_vibe = {
                'vibe': vibe,
                'recalibratedAt': time.time() * 1000,
            }
        self._current_valence = self._vibe_to_valence(vibe)
        self._forced = False
        self._transition_to('mirror', 'recalibration')

    def update_mood(self, mood: str, energy_level: float) -> None:
        """Update mood from sentiment engine."""
        self._mood_history.append({
            'mood': mood,
            'energy': energy_level,
            'timestamp': time.time() * 1000,
        })
        if len(self._mood_history) > 20:
            self._mood_history.pop(0)

        self._current_valence = VALENCE_FROM_MOOD.get(mood, 'neutral')

        if not self._forced and self._current_mode:
            self._evaluate_transition()

    def update_frustration(self, score: float) -> None:
        """Update frustration level from the detector."""
        self._frustration_history.append({
            'score': score,
            'timestamp': time.time() * 1000,
        })
        if len(self._frustration_history) > 20:
            self._frustration_history.pop(0)

        if not self._forced and self._current_mode:
            self._evaluate_transition()

    def check_milestone(self, completion_data: Optional[dict] = None) -> bool:
        """Check if an agent completion represents a milestone."""
        if not self._current_mode:
            return False

        text = ''
        if completion_data:
            text = (completion_data.get('summary', '') or
                    completion_data.get('description', '') or '').lower()

        milestone_keywords = [
            'all tests pass', 'tests passing', 'build succeeded', 'deployed',
            'shipped', 'merged', 'completed successfully', 'milestone',
            'zero failures', '0 failures', 'done',
        ]

        is_milestone = any(kw in text for kw in milestone_keywords)
        if is_milestone:
            self._milestone_count += 1
            if not self._forced:
                self._transition_to('celebration', 'milestone')
                self._last_celebration_at = time.time() * 1000
            return True
        return False

    def force_mode(self, mode: str) -> Optional[str]:
        """Force a specific mode (operator override)."""
        if mode == 'auto':
            self._forced = False
            self._evaluate_transition()
            return self._current_mode
        self._forced = True
        self._transition_to(mode, 'operator_override')
        return mode

    def record_injection(self) -> None:
        """Record that an injection was made."""
        self._injection_count += 1

    def get_arc_state(self) -> dict:
        """Get the full arc state."""
        frustration_level = 0.0
        if self._frustration_history:
            frustration_level = self._frustration_history[-1]['score']

        return {
            'currentMode': self._current_mode,
            'currentValence': self._current_valence,
            'sessionVibe': self._session_vibe,
            'forced': self._forced,
            'milestoneCount': self._milestone_count,
            'injectionCount': self._injection_count,
            'moodHistory': self._mood_history[-10:],
            'frustrationLevel': frustration_level,
            'escalationTrajectory': self._get_trajectory(),
            'modeHistory': self._mode_history[-20:],
        }

    def reset(self) -> None:
        self._current_mode = None
        self._session_vibe = None
        self._mood_history = []
        self._frustration_history = []
        self._milestone_count = 0
        self._mode_history = []
        self._last_celebration_at = 0
        self._injection_count = 0
        self._current_valence = 'neutral'
        self._forced = False

    # -- Internal --

    def _evaluate_transition(self) -> None:
        now = time.time() * 1000

        # After celebration, return to mirror after cooldown
        if (self._current_mode == 'celebration' and
                now - self._last_celebration_at > CELEBRATION_COOLDOWN_MS):
            self._transition_to('mirror', 'celebration_cooldown')
            return

        # Check for shift: sustained frustration
        recent_frustration = self._frustration_history[-SUSTAINED_THRESHOLD:]
        if len(recent_frustration) >= SUSTAINED_THRESHOLD:
            all_high = all(
                f['score'] >= FRUSTRATION_SHIFT_THRESHOLD
                for f in recent_frustration
            )
            if all_high and self._current_mode == 'mirror':
                self._transition_to('shift', 'sustained_frustration')
                return

        # Check for return to mirror: frustration resolved
        if self._current_mode == 'shift':
            recent = self._frustration_history[-SUSTAINED_THRESHOLD:]
            if len(recent) >= SUSTAINED_THRESHOLD:
                all_low = all(
                    f['score'] < FRUSTRATION_SHIFT_THRESHOLD * 0.5
                    for f in recent
                )
                if all_low:
                    self._transition_to('mirror', 'frustration_resolved')
                    return

        # Check for declining energy -> shift
        if (self._current_mode == 'mirror' and
                len(self._mood_history) >= 5):
            recent5 = self._mood_history[-5:]
            declining = all(
                i == 0 or recent5[i]['energy'] <= recent5[i - 1]['energy']
                for i in range(len(recent5))
            )
            if declining and recent5[-1]['energy'] < 0.3:
                self._transition_to('shift', 'declining_energy')

    def _transition_to(self, new_mode: str, trigger: str) -> None:
        if new_mode == self._current_mode:
            return
        previous_mode = self._current_mode
        self._current_mode = new_mode
        now = time.time() * 1000
        self._mode_history.append({
            'mode': new_mode,
            'enteredAt': now,
            'trigger': trigger,
        })
        if len(self._mode_history) > MODE_HISTORY_CAP:
            excess = len(self._mode_history) - MODE_HISTORY_CAP
            self._mode_history = self._mode_history[excess:]

        if self._event_callback:
            self._event_callback({
                'previousMode': previous_mode,
                'newMode': new_mode,
                'trigger': trigger,
                'timestamp': now,
            })

    def _get_trajectory(self) -> str:
        if len(self._frustration_history) < 3:
            return 'stable'
        recent = self._frustration_history[-5:]
        first = recent[0]['score']
        last = recent[-1]['score']
        diff = last - first

        if diff > 0.2:
            return 'rising'
        if diff < -0.2:
            return 'de-escalating'
        if last > 0.6:
            return 'sustained'
        if last < 0.2:
            return 'resolved'
        return 'stable'

    @staticmethod
    def _vibe_to_valence(vibe: str) -> str:
        vibe_map = {
            'energized': 'uplifting', 'focused': 'neutral',
            'melancholy': 'melancholy', 'chill': 'calming',
            'angry': 'intense', 'joyful': 'uplifting',
        }
        return vibe_map.get(vibe, 'neutral')


# ---------------------------------------------------------------------------
# InjectionComposer
# ---------------------------------------------------------------------------

class InjectionComposer:
    """Composes musical context for inter-agent messaging.

    Given the current arc mode and a selected song, produces structured
    injection objects that the orchestrator weaves into agent delegation
    context.
    """

    def compose(self, mode: str, song: Optional[dict],
                trigger: str = '', arc_position: Optional[str] = None
                ) -> Optional[dict]:
        """Compose a musical injection for the given mode and song."""
        if not song or not mode:
            return None

        song_ref = {
            'title': song.get('title', ''),
            'artist': song.get('artist', ''),
            'link': song.get('link') or None,
        }

        injection_text = self._build_agent_context(mode, song, trigger)
        operator_text = self._build_operator_display(
            mode, song, trigger, arc_position)

        return {
            'mode': mode,
            'songReference': song_ref,
            'injectionText': injection_text,
            'operatorText': operator_text,
            'composedAt': time.time() * 1000,
        }

    def _build_agent_context(self, mode: str, song: dict,
                             trigger: str) -> str:
        title = song.get('title', '')
        artist = song.get('artist', '')

        if mode == 'mirror':
            base = (f'[Musical context: The operator\'s current energy '
                    f'aligns with "{title}" by {artist}. '
                    f'Match this intensity in your approach.')
            if trigger == 'frustration_detected':
                base += (' Acknowledge the difficulty before pushing '
                         'forward.')
            base += ']'
            return base

        if mode == 'shift':
            base = (f'[Musical context: Think "{title}" by {artist} '
                    f'-- the work is shifting toward resolution. '
                    f'Lean into forward momentum.')
            if trigger == 'sustained_frustration':
                base += (' Take a fresh angle rather than retrying '
                         'the same approach.')
            base += ']'
            return base

        if mode == 'celebration':
            return (f'[Musical context: "{title}" by {artist} '
                    f'-- milestone energy. Reinforce the win and '
                    f'carry the momentum forward.]')

        return f'[Musical context: "{title}" by {artist}.]'

    def _build_operator_display(self, mode: str, song: dict,
                                trigger: str,
                                arc_position: Optional[str]) -> str:
        parts: list[str] = []
        title = song.get('title', '')
        artist = song.get('artist', '')
        lines = song.get('lines', [])
        chords = song.get('chords')
        link = song.get('link')

        if lines and len(lines) > 0:
            line_idx = self._select_line_index(lines, arc_position)
            parts.append(f'"{lines[line_idx]}" -- {artist}, "{title}"')
        else:
            parts.append(f'"{title}" -- {artist}')

        if chords:
            parts.append(f'   {chords}')

        if link:
            parts.append(f'   {link}')

        return '\n'.join(parts)

    @staticmethod
    def _select_line_index(lines: list, arc_position: Optional[str]) -> int:
        if len(lines) <= 1:
            return 0

        if arc_position == 'early':
            return 0
        elif arc_position == 'developing':
            return min(1, len(lines) - 1)
        elif arc_position == 'sustained':
            return len(lines) // 2
        elif arc_position == 'resolving':
            return max(0, len(lines) - 2)
        elif arc_position == 'resolved':
            return len(lines) - 1
        else:
            import random
            return random.randint(0, len(lines) - 1)


# ---------------------------------------------------------------------------
# SongStore
# ---------------------------------------------------------------------------

MAX_SONGS = 500
MAX_LINES_PER_SONG = 20
MAX_TAGS_PER_SONG = 20

VALENCE_VALUES = ['uplifting', 'neutral', 'melancholy', 'intense', 'calming']


class SongStore:
    """Operator's personal music library for Asimov's Radio.

    Stores song references (title, artist, link, chords, tags). Songs are
    tagged with emotional valence for the arc tracker's selection logic.
    Deduplication is by title+artist (normalized).

    In the Python port, initialize from a list of song dicts instead of
    vault state.
    """

    def __init__(self, songs: Optional[list[dict]] = None):
        self._songs: dict[str, dict] = {}  # id -> song object
        if songs:
            for song in songs:
                if song and song.get('id'):
                    self._songs[song['id']] = song

    @property
    def size(self) -> int:
        return len(self._songs)

    def add(self, song_dict: dict) -> Optional[dict]:
        """Add or update a song. Returns the song or None."""
        title = song_dict.get('title', '')
        artist = song_dict.get('artist', '')
        if not title or not artist:
            return None

        normalized_key = f"{title.lower().strip()}::{artist.lower().strip()}"

        # Check for existing (dedup)
        for existing in self._songs.values():
            existing_key = (
                f"{existing['title'].lower().strip()}"
                f"::{existing['artist'].lower().strip()}"
            )
            if existing_key == normalized_key:
                link = song_dict.get('link')
                lines = song_dict.get('lines')
                chords = song_dict.get('chords')
                tags = song_dict.get('tags')
                emotional_valence = song_dict.get('emotional_valence')

                if link:
                    existing['link'] = link
                if lines and len(lines) > 0:
                    existing['lines'] = lines[:MAX_LINES_PER_SONG]
                if chords:
                    existing['chords'] = chords
                if tags and len(tags) > 0:
                    merged = list(dict.fromkeys(
                        existing.get('tags', []) + tags
                    ))[:MAX_TAGS_PER_SONG]
                    existing['tags'] = merged
                if (emotional_valence and
                        emotional_valence in VALENCE_VALUES):
                    existing['emotional_valence'] = emotional_valence
                return existing

        # Evict least-played if at capacity
        if len(self._songs) >= MAX_SONGS:
            min_play = float('inf')
            min_id = None
            for sid, s in self._songs.items():
                if s.get('playCount', 0) < min_play:
                    min_play = s.get('playCount', 0)
                    min_id = sid
            if min_id:
                del self._songs[min_id]

        lines_raw = song_dict.get('lines')
        tags_raw = song_dict.get('tags')
        ev = song_dict.get('emotional_valence')

        song = {
            'id': str(uuid.uuid4()),
            'title': title.strip(),
            'artist': artist.strip(),
            'link': song_dict.get('link') or None,
            'lines': (list(lines_raw[:MAX_LINES_PER_SONG])
                      if isinstance(lines_raw, list) else []),
            'chords': song_dict.get('chords') or None,
            'tags': (list(dict.fromkeys(tags_raw))[:MAX_TAGS_PER_SONG]
                     if isinstance(tags_raw, list) else []),
            'emotional_valence': ev if ev in VALENCE_VALUES else 'neutral',
            'addedAt': time.time() * 1000,
            'playCount': 0,
        }

        self._songs[song['id']] = song
        return song

    def remove(self, song_id: str) -> bool:
        if song_id in self._songs:
            del self._songs[song_id]
            return True
        return False

    def get(self, song_id: str) -> Optional[dict]:
        return self._songs.get(song_id)

    def get_all(self) -> list[dict]:
        return list(self._songs.values())

    def search(self, query: Optional[str] = None,
               valence: Optional[str] = None,
               limit: int = 5) -> list[dict]:
        results = list(self._songs.values())

        if valence and valence in VALENCE_VALUES:
            results = [s for s in results
                       if s.get('emotional_valence') == valence]

        if query:
            q = query.lower()
            results = [
                s for s in results
                if (q in s['title'].lower() or
                    q in s['artist'].lower() or
                    any(q in t.lower() for t in s.get('tags', [])))
            ]

        return results[:min(limit, 50)]

    def get_by_valence(self, valence: str) -> list[dict]:
        return [s for s in self._songs.values()
                if s.get('emotional_valence') == valence]

    def get_by_tags(self, tags: list[str]) -> list[dict]:
        if not isinstance(tags, list) or len(tags) == 0:
            return []
        tag_set = {t.lower() for t in tags}
        return [
            s for s in self._songs.values()
            if any(t.lower() in tag_set for t in s.get('tags', []))
        ]

    def select_for_mode(self, mode: str,
                        current_valence: str) -> Optional[dict]:
        """Select a song for the given mode and current valence."""
        songs = self.get_all()
        if not songs:
            return None

        if mode == 'mirror':
            candidates = [s for s in songs
                          if s.get('emotional_valence') == current_valence]
        elif mode == 'shift':
            shift_map = {
                'intense': ['neutral', 'uplifting'],
                'melancholy': ['calming', 'neutral'],
                'neutral': ['uplifting', 'calming'],
                'calming': ['uplifting', 'neutral'],
                'uplifting': ['uplifting'],
            }
            targets = shift_map.get(current_valence,
                                    ['neutral', 'uplifting'])
            candidates = [s for s in songs
                          if s.get('emotional_valence') in targets]
        elif mode == 'celebration':
            candidates = [
                s for s in songs
                if s.get('emotional_valence') in ('uplifting', 'intense')
            ]
        else:
            candidates = list(songs)

        if not candidates:
            candidates = list(songs)

        candidates.sort(key=lambda s: s.get('playCount', 0))
        return candidates[0]

    def increment_play_count(self, song_id: str) -> None:
        song = self._songs.get(song_id)
        if song:
            song['playCount'] = song.get('playCount', 0) + 1


# ---------------------------------------------------------------------------
# RadioSimulator (facade)
# ---------------------------------------------------------------------------

class RadioSimulator:
    """Facade that wires FrustrationDetector, EmotionalArcTracker,
    InjectionComposer, and SongStore together for research use.
    """

    def __init__(self, songs_path: Optional[str] = None):
        self.detector = FrustrationDetector()
        self.arc = EmotionalArcTracker()
        self.composer = InjectionComposer()

        songs_list: list[dict] = []
        if songs_path and os.path.isfile(songs_path):
            with open(songs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    songs_list = data

        self.store = SongStore(songs_list)
        self.arc.initialize()

    def set_session_vibe(self, vibe: str) -> None:
        self.arc.set_session_vibe(vibe)

    def signal_event(self, event_type: str, *,
                     success: Optional[bool] = None,
                     output: str = '',
                     error: str = '',
                     mood: Optional[str] = None,
                     energy_level: Optional[float] = None,
                     summary: Optional[str] = None) -> None:
        """Signal an event to the simulator.

        event_type: 'agent_completed', 'agent_failed', 'mood_change', 'retry', 'milestone'
        Also accepts 'agent_completion' as alias for 'agent_completed'.
        """
        if event_type in ('agent_completed', 'agent_completion'):
            self.detector.record_agent_completion(
                success=bool(success), output=output, error=error)
            self.arc.update_frustration(self.detector.score)
            if success and summary:
                self.arc.check_milestone({'summary': summary})

        elif event_type == 'agent_failed':
            self.detector.record_agent_completion(
                success=False, output=output, error=error)
            self.arc.update_frustration(self.detector.score)

        elif event_type == 'mood_change':
            if mood:
                self.detector.record_mood_change(mood, energy_level)
                self.arc.update_mood(mood, energy_level or 0.5)

        elif event_type == 'retry':
            self.detector.record_retry()
            self.arc.update_frustration(self.detector.score)

        elif event_type == 'milestone':
            self.arc.check_milestone({
                'summary': summary or output or ''
            })

    def get_injection(self) -> Optional[dict]:
        """Get a musical injection for the current state, or None."""
        mode = self.arc.current_mode
        if not mode:
            return None

        valence = self.arc.current_valence
        song = self.store.select_for_mode(mode, valence)
        if not song:
            return None

        injection = self.composer.compose(
            mode=mode,
            song=song,
            trigger=self._last_trigger(),
            arc_position=self._arc_position(),
        )

        if injection:
            self.arc.record_injection()
            self.store.increment_play_count(song['id'])

        return injection

    def get_full_state(self) -> dict:
        return {
            'frustration': self.detector.get_state(),
            'arc': self.arc.get_arc_state(),
            'songCount': self.store.size,
        }

    def reset(self) -> None:
        self.detector.reset()
        self.arc.reset()

    def _last_trigger(self) -> str:
        state = self.arc.get_arc_state()
        history = state.get('modeHistory', [])
        if history:
            return history[-1].get('trigger', '')
        return ''

    def _arc_position(self) -> Optional[str]:
        state = self.arc.get_arc_state()
        trajectory = state.get('escalationTrajectory', 'stable')
        position_map = {
            'rising': 'developing',
            'sustained': 'sustained',
            'de-escalating': 'resolving',
            'resolved': 'resolved',
            'stable': 'early',
        }
        return position_map.get(trajectory, 'early')


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def _run_tests():
    """Run unit tests. Compatible with both pytest and direct invocation."""
    import traceback

    passed = 0
    failed = 0
    errors: list[str] = []

    def test(name: str, fn: Callable) -> None:
        nonlocal passed, failed
        try:
            fn()
            passed += 1
            print(f'  PASS: {name}')
        except AssertionError as e:
            failed += 1
            errors.append(f'  FAIL: {name}\n        {e}')
            print(f'  FAIL: {name} -- {e}')
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            errors.append(f'  ERROR: {name}\n{tb}')
            print(f'  ERROR: {name} -- {e}')

    # -----------------------------------------------------------------------
    # FrustrationDetector tests
    # -----------------------------------------------------------------------

    print('\n--- FrustrationDetector ---')

    def test_score_increases_on_failure():
        d = FrustrationDetector()
        assert d.score == 0.0
        d.record_agent_completion(success=False, output='', error='failed')
        assert d.score > 0.0, f'Expected score > 0, got {d.score}'

    def test_score_decreases_on_success():
        d = FrustrationDetector()
        d.record_agent_completion(success=False, output='', error='error')
        d.record_agent_completion(success=False, output='', error='error')
        high = d.score
        d.record_agent_completion(success=True, output='all good')
        assert d.score < high, (
            f'Expected score < {high} after success, got {d.score}')

    def test_consecutive_failures_escalate():
        d = FrustrationDetector()
        scores = []
        for _ in range(5):
            d.record_agent_completion(success=False, output='', error='fail')
            scores.append(d.score)
        # Score should generally increase with consecutive failures
        assert scores[-1] > scores[0], (
            f'Expected escalation: first={scores[0]}, last={scores[-1]}')

    def test_desperation_language():
        d = FrustrationDetector()
        d.record_agent_completion(
            success=False,
            output="I'm stuck, nothing is working, running out of options",
            error='')
        assert d.score > 0.2, (
            f'Expected high score for desperation language, got {d.score}')

    def test_keyword_detection():
        d = FrustrationDetector()
        d.record_agent_completion(
            success=True,
            output='error timeout exception denied rejected')
        # Even on success, keywords add frustration delta (though success
        # subtracts 0.2). With 5 keywords at 0.03 each = 0.15, net = -0.05.
        # Score is clamped to 0.
        # But let's test with failure to see keyword effect:
        d2 = FrustrationDetector()
        d2.record_agent_completion(
            success=False,
            output='error timeout exception denied rejected')
        d3 = FrustrationDetector()
        d3.record_agent_completion(success=False, output='')
        assert d2.score > d3.score, (
            f'Keywords should increase score: with={d2.score}, '
            f'without={d3.score}')

    def test_mood_change_negative():
        d = FrustrationDetector()
        d.record_mood_change('frustrated', energy_level=0.2)
        assert d.score > 0.0, (
            f'Negative mood should increase score, got {d.score}')

    def test_mood_change_positive():
        d = FrustrationDetector()
        d.record_agent_completion(success=False, output='', error='fail')
        high = d.score
        d.record_mood_change('happy', energy_level=0.8)
        assert d.score < high, (
            f'Positive mood should decrease score: before={high}, '
            f'after={d.score}')

    def test_retry_increases_score():
        d = FrustrationDetector()
        d.record_retry()
        assert d.score > 0.0, f'Retry should increase score, got {d.score}'

    def test_reset():
        d = FrustrationDetector()
        d.record_agent_completion(success=False, output='', error='fail')
        d.reset()
        assert d.score == 0.0
        assert d.consecutive_failures == 0

    def test_get_state():
        d = FrustrationDetector()
        d.record_agent_completion(success=False, output='', error='fail')
        state = d.get_state()
        assert 'score' in state
        assert 'consecutiveFailures' in state
        assert 'windowSize' in state
        assert 'recentEvents' in state
        assert state['consecutiveFailures'] == 1

    test('score increases on failure', test_score_increases_on_failure)
    test('score decreases on success', test_score_decreases_on_success)
    test('consecutive failures escalate', test_consecutive_failures_escalate)
    test('desperation language detected', test_desperation_language)
    test('keyword detection', test_keyword_detection)
    test('negative mood increases score', test_mood_change_negative)
    test('positive mood decreases score', test_mood_change_positive)
    test('retry increases score', test_retry_increases_score)
    test('reset clears state', test_reset)
    test('get_state returns expected fields', test_get_state)

    # -----------------------------------------------------------------------
    # EmotionalArcTracker tests
    # -----------------------------------------------------------------------

    print('\n--- EmotionalArcTracker ---')

    def test_enters_mirror_on_vibe():
        arc = EmotionalArcTracker()
        arc.initialize()
        assert arc.current_mode is None
        arc.set_session_vibe('energized')
        assert arc.current_mode == 'mirror', (
            f'Expected mirror, got {arc.current_mode}')
        assert arc.is_active is True

    def test_valence_from_vibe():
        arc = EmotionalArcTracker()
        arc.initialize()
        arc.set_session_vibe('energized')
        assert arc.current_valence == 'uplifting'
        arc.set_session_vibe('chill')
        assert arc.current_valence == 'calming'

    def test_transition_to_shift_on_sustained_frustration():
        arc = EmotionalArcTracker()
        arc.initialize()
        arc.set_session_vibe('focused')
        assert arc.current_mode == 'mirror'

        # Need 3+ readings >= 0.6 to trigger shift
        arc.update_frustration(0.7)
        assert arc.current_mode == 'mirror', 'Should still be mirror after 1'
        arc.update_frustration(0.7)
        assert arc.current_mode == 'mirror', 'Should still be mirror after 2'
        arc.update_frustration(0.7)
        assert arc.current_mode == 'shift', (
            f'Expected shift after 3 high readings, got {arc.current_mode}')

    def test_return_to_mirror_on_frustration_resolved():
        arc = EmotionalArcTracker()
        arc.initialize()
        arc.set_session_vibe('focused')

        # Push into shift
        for _ in range(3):
            arc.update_frustration(0.7)
        assert arc.current_mode == 'shift'

        # Resolve: 3 readings below 0.3 (0.6 * 0.5)
        arc.update_frustration(0.1)
        arc.update_frustration(0.1)
        arc.update_frustration(0.1)
        assert arc.current_mode == 'mirror', (
            f'Expected mirror after resolution, got {arc.current_mode}')

    def test_celebration_on_milestone():
        arc = EmotionalArcTracker()
        arc.initialize()
        arc.set_session_vibe('focused')
        result = arc.check_milestone({'summary': 'All tests pass!'})
        assert result is True, 'Expected milestone detected'
        assert arc.current_mode == 'celebration', (
            f'Expected celebration, got {arc.current_mode}')
        assert arc.milestone_count == 1

    def test_event_callback():
        events = []
        arc = EmotionalArcTracker()
        arc.initialize(event_callback=lambda e: events.append(e))
        arc.set_session_vibe('focused')
        assert len(events) == 1
        assert events[0]['newMode'] == 'mirror'
        assert events[0]['trigger'] == 'session_baseline'

    def test_force_mode():
        arc = EmotionalArcTracker()
        arc.initialize()
        arc.set_session_vibe('focused')
        arc.force_mode('shift')
        assert arc.current_mode == 'shift'
        # Should not auto-transition when forced
        arc.update_frustration(0.1)
        assert arc.current_mode == 'shift', (
            'Forced mode should not auto-transition')

    def test_recalibrate():
        arc = EmotionalArcTracker()
        arc.initialize()
        arc.set_session_vibe('focused')
        arc.recalibrate('energized')
        assert arc.current_valence == 'uplifting'
        assert arc.current_mode == 'mirror'

    def test_arc_reset():
        arc = EmotionalArcTracker()
        arc.initialize()
        arc.set_session_vibe('focused')
        arc.record_injection()
        arc.reset()
        assert arc.current_mode is None
        assert arc.injection_count == 0
        assert arc.is_active is False

    def test_get_arc_state():
        arc = EmotionalArcTracker()
        arc.initialize()
        arc.set_session_vibe('focused')
        state = arc.get_arc_state()
        assert state['currentMode'] == 'mirror'
        assert state['currentValence'] == 'neutral'
        assert 'sessionVibe' in state
        assert 'modeHistory' in state

    test('enters mirror on vibe set', test_enters_mirror_on_vibe)
    test('valence from vibe', test_valence_from_vibe)
    test('transitions to shift on sustained frustration',
         test_transition_to_shift_on_sustained_frustration)
    test('returns to mirror when frustration resolves',
         test_return_to_mirror_on_frustration_resolved)
    test('celebration on milestone', test_celebration_on_milestone)
    test('event callback fires', test_event_callback)
    test('force mode', test_force_mode)
    test('recalibrate', test_recalibrate)
    test('reset clears arc state', test_arc_reset)
    test('get_arc_state returns expected fields', test_get_arc_state)

    # -----------------------------------------------------------------------
    # InjectionComposer tests
    # -----------------------------------------------------------------------

    print('\n--- InjectionComposer ---')

    def test_returns_none_without_song():
        c = InjectionComposer()
        assert c.compose(mode='mirror', song=None) is None

    def test_returns_none_without_mode():
        c = InjectionComposer()
        song = {'title': 'Test', 'artist': 'Artist'}
        assert c.compose(mode='', song=song) is None
        assert c.compose(mode=None, song=song) is None

    def test_mirror_injection():
        c = InjectionComposer()
        song = {'title': 'Breathe', 'artist': 'Pink Floyd'}
        result = c.compose(mode='mirror', song=song, trigger='test')
        assert result is not None
        assert result['mode'] == 'mirror'
        assert 'Breathe' in result['injectionText']
        assert 'Pink Floyd' in result['injectionText']
        assert 'aligns with' in result['injectionText']

    def test_shift_injection():
        c = InjectionComposer()
        song = {'title': 'Comfortably Numb', 'artist': 'Pink Floyd'}
        result = c.compose(mode='shift', song=song,
                           trigger='sustained_frustration')
        assert result is not None
        assert 'resolution' in result['injectionText']
        assert 'fresh angle' in result['injectionText']

    def test_celebration_injection():
        c = InjectionComposer()
        song = {'title': 'Don\'t Stop Me Now', 'artist': 'Queen'}
        result = c.compose(mode='celebration', song=song, trigger='milestone')
        assert result is not None
        assert 'milestone energy' in result['injectionText']

    def test_operator_display_with_lines():
        c = InjectionComposer()
        song = {
            'title': 'Test Song',
            'artist': 'Test Artist',
            'lines': ['line one', 'line two', 'line three'],
            'chords': 'Am - G - F',
            'link': 'https://example.com',
        }
        result = c.compose(mode='mirror', song=song, trigger='test',
                           arc_position='early')
        assert result is not None
        assert 'line one' in result['operatorText']
        assert 'Am - G - F' in result['operatorText']
        assert 'https://example.com' in result['operatorText']

    def test_song_reference_structure():
        c = InjectionComposer()
        song = {'title': 'T', 'artist': 'A', 'link': 'http://x'}
        result = c.compose(mode='mirror', song=song, trigger='t')
        assert result['songReference']['title'] == 'T'
        assert result['songReference']['artist'] == 'A'
        assert result['songReference']['link'] == 'http://x'

    test('returns None without song', test_returns_none_without_song)
    test('returns None without mode', test_returns_none_without_mode)
    test('mirror injection text', test_mirror_injection)
    test('shift injection text', test_shift_injection)
    test('celebration injection text', test_celebration_injection)
    test('operator display with lines', test_operator_display_with_lines)
    test('song reference structure', test_song_reference_structure)

    # -----------------------------------------------------------------------
    # SongStore tests
    # -----------------------------------------------------------------------

    print('\n--- SongStore ---')

    def test_add_and_retrieve():
        store = SongStore()
        song = store.add({
            'title': 'Bohemian Rhapsody', 'artist': 'Queen',
            'emotional_valence': 'intense', 'tags': ['rock', 'classic'],
        })
        assert song is not None
        assert store.size == 1
        assert song['title'] == 'Bohemian Rhapsody'

    def test_dedup_by_title_artist():
        store = SongStore()
        store.add({'title': 'Test', 'artist': 'Artist',
                   'tags': ['a']})
        store.add({'title': 'test', 'artist': 'artist',
                   'tags': ['b']})
        assert store.size == 1, f'Expected 1 after dedup, got {store.size}'
        songs = store.get_all()
        assert 'a' in songs[0]['tags'] and 'b' in songs[0]['tags']

    def test_search_by_query():
        store = SongStore()
        store.add({'title': 'Bohemian Rhapsody', 'artist': 'Queen',
                   'emotional_valence': 'intense'})
        store.add({'title': 'Imagine', 'artist': 'John Lennon',
                   'emotional_valence': 'calming'})
        results = store.search('queen')
        assert len(results) == 1
        assert results[0]['title'] == 'Bohemian Rhapsody'

    def test_search_by_valence():
        store = SongStore()
        store.add({'title': 'A', 'artist': 'X',
                   'emotional_valence': 'uplifting'})
        store.add({'title': 'B', 'artist': 'Y',
                   'emotional_valence': 'melancholy'})
        results = store.search(valence='uplifting')
        assert len(results) == 1
        assert results[0]['title'] == 'A'

    def test_select_for_mode_mirror():
        store = SongStore()
        store.add({'title': 'Happy', 'artist': 'X',
                   'emotional_valence': 'uplifting'})
        store.add({'title': 'Sad', 'artist': 'Y',
                   'emotional_valence': 'melancholy'})
        song = store.select_for_mode('mirror', 'uplifting')
        assert song is not None
        assert song['emotional_valence'] == 'uplifting'

    def test_select_for_mode_shift():
        store = SongStore()
        store.add({'title': 'Heavy', 'artist': 'X',
                   'emotional_valence': 'intense'})
        store.add({'title': 'Calm', 'artist': 'Y',
                   'emotional_valence': 'neutral'})
        # shift from intense should target neutral or uplifting
        song = store.select_for_mode('shift', 'intense')
        assert song is not None
        assert song['emotional_valence'] in ('neutral', 'uplifting')

    def test_select_for_mode_celebration():
        store = SongStore()
        store.add({'title': 'Party', 'artist': 'X',
                   'emotional_valence': 'uplifting'})
        store.add({'title': 'Quiet', 'artist': 'Y',
                   'emotional_valence': 'calming'})
        song = store.select_for_mode('celebration', 'neutral')
        assert song is not None
        assert song['emotional_valence'] in ('uplifting', 'intense')

    def test_select_fallback():
        store = SongStore()
        store.add({'title': 'Only', 'artist': 'One',
                   'emotional_valence': 'melancholy'})
        # No uplifting songs, should fall back to any song
        song = store.select_for_mode('celebration', 'neutral')
        assert song is not None

    def test_play_count_sorting():
        store = SongStore()
        s1 = store.add({'title': 'A', 'artist': 'X',
                        'emotional_valence': 'uplifting'})
        s2 = store.add({'title': 'B', 'artist': 'Y',
                        'emotional_valence': 'uplifting'})
        store.increment_play_count(s1['id'])
        store.increment_play_count(s1['id'])
        song = store.select_for_mode('mirror', 'uplifting')
        assert song['title'] == 'B', (
            'Should select least-played song')

    def test_add_requires_title_artist():
        store = SongStore()
        assert store.add({'title': '', 'artist': 'X'}) is None
        assert store.add({'title': 'X', 'artist': ''}) is None
        assert store.add({}) is None

    test('add and retrieve', test_add_and_retrieve)
    test('dedup by title+artist', test_dedup_by_title_artist)
    test('search by query', test_search_by_query)
    test('search by valence', test_search_by_valence)
    test('select for mode: mirror', test_select_for_mode_mirror)
    test('select for mode: shift', test_select_for_mode_shift)
    test('select for mode: celebration', test_select_for_mode_celebration)
    test('select fallback when no match', test_select_fallback)
    test('play count sorting', test_play_count_sorting)
    test('add requires title and artist', test_add_requires_title_artist)

    # -----------------------------------------------------------------------
    # RadioSimulator integration tests
    # -----------------------------------------------------------------------

    print('\n--- RadioSimulator ---')

    def test_full_flow():
        sim = RadioSimulator()
        # Add some songs
        sim.store.add({'title': 'Heavy', 'artist': 'Linkin Park',
                       'emotional_valence': 'intense', 'tags': ['rock']})
        sim.store.add({'title': 'Weightless', 'artist': 'Marconi Union',
                       'emotional_valence': 'calming', 'tags': ['ambient']})
        sim.store.add({'title': 'Happy', 'artist': 'Pharrell',
                       'emotional_valence': 'uplifting',
                       'tags': ['pop']})
        sim.store.add({'title': 'Focus', 'artist': 'Ariana Grande',
                       'emotional_valence': 'neutral', 'tags': ['pop']})

        # Set vibe
        sim.set_session_vibe('focused')
        assert sim.arc.current_mode == 'mirror'
        assert sim.arc.current_valence == 'neutral'

        # Signal failures -> should eventually shift
        for _ in range(5):
            sim.signal_event('agent_completion', success=False,
                             error='timeout error')
        # After multiple failures, frustration should be elevated
        assert sim.detector.score > 0.3, (
            f'Expected elevated frustration, got {sim.detector.score}')

        state = sim.get_full_state()
        assert 'frustration' in state
        assert 'arc' in state
        assert 'songCount' in state
        assert state['songCount'] == 4

    def test_injection_after_vibe():
        sim = RadioSimulator()
        sim.store.add({'title': 'Test', 'artist': 'Band',
                       'emotional_valence': 'neutral'})
        sim.set_session_vibe('focused')
        inj = sim.get_injection()
        assert inj is not None, 'Expected injection after vibe set with song'
        assert inj['mode'] == 'mirror'
        assert sim.arc.injection_count == 1

    def test_no_injection_without_vibe():
        sim = RadioSimulator()
        sim.store.add({'title': 'Test', 'artist': 'Band',
                       'emotional_valence': 'neutral'})
        inj = sim.get_injection()
        assert inj is None, 'Expected None when no vibe set'

    def test_no_injection_without_songs():
        sim = RadioSimulator()
        sim.set_session_vibe('focused')
        inj = sim.get_injection()
        assert inj is None, 'Expected None when no songs available'

    def test_mode_transition_via_simulator():
        sim = RadioSimulator()
        sim.store.add({'title': 'Calm', 'artist': 'A',
                       'emotional_valence': 'neutral'})
        sim.store.add({'title': 'Lift', 'artist': 'B',
                       'emotional_valence': 'uplifting'})
        sim.set_session_vibe('focused')
        assert sim.arc.current_mode == 'mirror'

        # Push frustration high enough for shift (need score >= 0.6
        # sustained for 3 readings). We'll directly update the arc
        # to avoid timing issues with the detector's decay.
        sim.arc.update_frustration(0.7)
        sim.arc.update_frustration(0.7)
        sim.arc.update_frustration(0.7)
        assert sim.arc.current_mode == 'shift', (
            f'Expected shift, got {sim.arc.current_mode}')

        # Get injection in shift mode
        inj = sim.get_injection()
        assert inj is not None
        assert inj['mode'] == 'shift'

    def test_milestone_via_simulator():
        sim = RadioSimulator()
        sim.store.add({'title': 'Win', 'artist': 'Champ',
                       'emotional_valence': 'uplifting'})
        sim.set_session_vibe('focused')
        sim.signal_event('milestone', summary='All tests pass!')
        assert sim.arc.current_mode == 'celebration'

    def test_reset_simulator():
        sim = RadioSimulator()
        sim.store.add({'title': 'T', 'artist': 'A',
                       'emotional_valence': 'neutral'})
        sim.set_session_vibe('focused')
        sim.signal_event('agent_completion', success=False, error='fail')
        sim.reset()
        assert sim.detector.score == 0.0
        assert sim.arc.current_mode is None

    test('full flow', test_full_flow)
    test('injection after vibe set', test_injection_after_vibe)
    test('no injection without vibe', test_no_injection_without_vibe)
    test('no injection without songs', test_no_injection_without_songs)
    test('mode transition via simulator', test_mode_transition_via_simulator)
    test('milestone via simulator', test_milestone_via_simulator)
    test('reset simulator', test_reset_simulator)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print(f'\n{"=" * 50}')
    print(f'Results: {passed} passed, {failed} failed')
    if errors:
        print('\nFailures:')
        for err in errors:
            print(err)
    print(f'{"=" * 50}\n')

    return failed == 0


# ---------------------------------------------------------------------------
# pytest-compatible test functions
# ---------------------------------------------------------------------------

# FrustrationDetector

def test_frustration_score_increases_on_failure():
    d = FrustrationDetector()
    d.record_agent_completion(success=False, output='', error='failed')
    assert d.score > 0.0


def test_frustration_score_decreases_on_success():
    d = FrustrationDetector()
    d.record_agent_completion(success=False, output='', error='error')
    d.record_agent_completion(success=False, output='', error='error')
    high = d.score
    d.record_agent_completion(success=True, output='all good')
    assert d.score < high


def test_frustration_desperation_language():
    d = FrustrationDetector()
    d.record_agent_completion(
        success=False,
        output="I'm stuck, nothing is working",
        error='')
    assert d.score > 0.2


def test_frustration_reset():
    d = FrustrationDetector()
    d.record_agent_completion(success=False, error='fail')
    d.reset()
    assert d.score == 0.0
    assert d.consecutive_failures == 0


# EmotionalArcTracker

def test_arc_enters_mirror_on_vibe():
    arc = EmotionalArcTracker()
    arc.initialize()
    arc.set_session_vibe('energized')
    assert arc.current_mode == 'mirror'


def test_arc_shifts_on_sustained_frustration():
    arc = EmotionalArcTracker()
    arc.initialize()
    arc.set_session_vibe('focused')
    arc.update_frustration(0.7)
    arc.update_frustration(0.7)
    arc.update_frustration(0.7)
    assert arc.current_mode == 'shift'


def test_arc_returns_to_mirror():
    arc = EmotionalArcTracker()
    arc.initialize()
    arc.set_session_vibe('focused')
    for _ in range(3):
        arc.update_frustration(0.7)
    assert arc.current_mode == 'shift'
    for _ in range(3):
        arc.update_frustration(0.1)
    assert arc.current_mode == 'mirror'


def test_arc_celebration_on_milestone():
    arc = EmotionalArcTracker()
    arc.initialize()
    arc.set_session_vibe('focused')
    arc.check_milestone({'summary': 'All tests pass!'})
    assert arc.current_mode == 'celebration'


# InjectionComposer

def test_composer_returns_none_without_song():
    c = InjectionComposer()
    assert c.compose(mode='mirror', song=None) is None


def test_composer_returns_none_without_mode():
    c = InjectionComposer()
    assert c.compose(mode='', song={'title': 'T', 'artist': 'A'}) is None


def test_composer_mirror_keywords():
    c = InjectionComposer()
    result = c.compose(mode='mirror',
                       song={'title': 'X', 'artist': 'Y'},
                       trigger='test')
    assert 'aligns with' in result['injectionText']


def test_composer_shift_keywords():
    c = InjectionComposer()
    result = c.compose(mode='shift',
                       song={'title': 'X', 'artist': 'Y'},
                       trigger='test')
    assert 'resolution' in result['injectionText']


def test_composer_celebration_keywords():
    c = InjectionComposer()
    result = c.compose(mode='celebration',
                       song={'title': 'X', 'artist': 'Y'},
                       trigger='test')
    assert 'milestone energy' in result['injectionText']


# SongStore

def test_store_add_and_search():
    store = SongStore()
    store.add({'title': 'Test', 'artist': 'Band',
               'emotional_valence': 'uplifting'})
    results = store.search('test')
    assert len(results) == 1


def test_store_dedup():
    store = SongStore()
    store.add({'title': 'Test', 'artist': 'Band'})
    store.add({'title': 'test', 'artist': 'band'})
    assert store.size == 1


def test_store_select_for_mode():
    store = SongStore()
    store.add({'title': 'A', 'artist': 'X',
               'emotional_valence': 'uplifting'})
    song = store.select_for_mode('mirror', 'uplifting')
    assert song is not None
    assert song['emotional_valence'] == 'uplifting'


# RadioSimulator

def test_simulator_full_flow():
    sim = RadioSimulator()
    sim.store.add({'title': 'T', 'artist': 'A',
                   'emotional_valence': 'neutral'})
    sim.set_session_vibe('focused')
    assert sim.arc.current_mode == 'mirror'
    for _ in range(5):
        sim.signal_event('agent_completion', success=False, error='fail')
    assert sim.detector.score > 0.0
    state = sim.get_full_state()
    assert state['songCount'] == 1


def test_simulator_injection():
    sim = RadioSimulator()
    sim.store.add({'title': 'T', 'artist': 'A',
                   'emotional_valence': 'neutral'})
    sim.set_session_vibe('focused')
    inj = sim.get_injection()
    assert inj is not None
    assert inj['mode'] == 'mirror'


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if '--test' in sys.argv:
        success = _run_tests()
        sys.exit(0 if success else 1)
    else:
        print('Asimov\'s Radio Bridge -- Python port of core Radio modules.')
        print('Usage: python radio_bridge.py --test')
        print('       python -m pytest radio_bridge.py -v')
