import os
import sys
import re
import time
import asyncio
import tempfile
import logging

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
logger = logging.getLogger("WednesdayTTS")
logger.setLevel(logging.ERROR)

try:
    import pygame
    pygame.mixer.init()
except Exception:
    pygame = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

DEFAULT_VOICE = "en-US-AvaNeural"

class WednesdaySpeaker:
    def __init__(self, voice_name: str = DEFAULT_VOICE):
        self.voice_name = voice_name
        self.pyttsx3_engine = None
        if pyttsx3:
            try:
                self.pyttsx3_engine = pyttsx3.init()
            except Exception:
                pass
        print(f"[Wednesday TTS]: Fast Neural Engine Active (Voice: '{self.voice_name}')")

    async def _async_edge_speak(self, text: str, voice: str) -> tuple[float, float]:
        temp_mp3 = os.path.join(tempfile.gettempdir(), "wednesday_speech.mp3")
        gen_time = 0.0
        play_time = 0.0
        try:
            t0 = time.perf_counter()
            communicate = edge_tts.Communicate(text.strip(), voice)
            await communicate.save(temp_mp3)
            gen_time = time.perf_counter() - t0

            t_play_start = time.perf_counter()
            if pygame and pygame.mixer.get_init():
                pygame.mixer.music.load(temp_mp3)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(30)
                pygame.mixer.music.unload()
            play_time = time.perf_counter() - t_play_start

            try:
                if os.path.exists(temp_mp3):
                    os.remove(temp_mp3)
            except Exception:
                pass
        except Exception as err:
            logger.error(f"[Edge TTS Error]: {err}")
            play_time = self._fallback_speak(text)
        return gen_time, play_time

    def _fallback_speak(self, text: str) -> float:
        t0 = time.perf_counter()
        if self.pyttsx3_engine:
            try:
                self.pyttsx3_engine.say(text)
                self.pyttsx3_engine.runAndWait()
                return time.perf_counter() - t0
            except Exception:
                pass
        print(f"[Wednesday]: {text}")
        return time.perf_counter() - t0

    def speak(self, text: str, voice: str = None) -> tuple[float, float]:
        if not text or not text.strip():
            return 0.0, 0.0
        target_voice = voice or self.voice_name

        if edge_tts is not None:
            try:
                return asyncio.run(self._async_edge_speak(text, target_voice))
            except Exception as err:
                logger.error(f"[TTS Async Error]: {err}")

        play_t = self._fallback_speak(text)
        return 0.0, play_t

_speaker_instance = None

def get_speaker(voice_name: str = DEFAULT_VOICE) -> WednesdaySpeaker:
    global _speaker_instance
    if _speaker_instance is None:
        _speaker_instance = WednesdaySpeaker(voice_name=voice_name)
    return _speaker_instance

def speak(text: str, voice_name: str = DEFAULT_VOICE) -> tuple[float, float]:
    spk = get_speaker(voice_name=voice_name)
    return spk.speak(text, voice=voice_name)
