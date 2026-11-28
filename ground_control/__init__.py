"""Kiosk Satellite Ground Control: speaker recognition and other-voice trimming.

Kiosk Satellite calls this service on demand. It identifies who is speaking
and says when someone else starts talking, and Kiosk Satellite sends the
result to whichever backend it uses (Home Assistant or a realtime provider).
"""

__version__ = "2026.11.18"

SAMPLE_RATE = 16000
