"""Keep late STT frames from satisfying a newer acoustic turn's stop gate."""

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame, VADUserStartedSpeakingFrame
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy


class SpeechGenerationTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """The stock Smart Turn strategy, with an STT generation check.

    A VAD system frame can overtake a queued transcript data frame. The STT
    adapter's generation check at emission time is therefore not sufficient:
    validate it again here, where the turn decision is made. The user aggregator
    still keeps the older text in order; it just cannot finalize the newer speech.
    Untagged transcripts retain Pipecat's original behavior (including the
    configuration switch that restores the unmodified provider adapter).
    """

    def __init__(self, *, finalize_transcripts: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._finalize_transcripts = finalize_transcripts
        self._speech_generation = 0
        self._latest_interim_item = None

    async def process_frame(self, frame):
        if not self._finalize_transcripts:
            return await super().process_frame(frame)
        if isinstance(frame, InterimTranscriptionFrame) and frame.metadata.get("stt_item_id"):
            self._latest_interim_item = frame.metadata["stt_item_id"]
            # A newer live item can start before local VAD detects it. Neither
            # an old final flag nor old text should satisfy the STT timeout.
            self._text = ""
        return await super().process_frame(frame)

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        if self._finalize_transcripts:
            self._speech_generation += 1
            self._latest_interim_item = None
            # Native Pipecat keeps earlier segment text across a mid-turn VAD
            # resume. It must not satisfy the new segment's safety-net timer.
            self._text = ""
        await super()._handle_vad_user_started_speaking(frame)

    async def _handle_transcription(self, frame: TranscriptionFrame):
        if not self._finalize_transcripts:
            return await super()._handle_transcription(frame)
        generation = frame.metadata.get("stt_speech_generation")
        if generation is not None and generation != self._speech_generation:
            return
        item_ids = frame.metadata.get("stt_item_ids")
        if item_ids is not None and self._latest_interim_item is not None and self._latest_interim_item not in item_ids:
            return
        await super()._handle_transcription(frame)
