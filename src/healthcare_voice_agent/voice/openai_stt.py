"""Pipecat 1.11 compatibility mixin for ordered OpenAI STT finalization.

The actual provider class is imported and assembled only in providers.py. The
private receive dispatcher is version-pinned: Pipecat has no committed-item hook.
No model, audio format, VAD threshold, or STT safety timeout is changed here.
"""

import asyncio
import json
from contextvars import ContextVar

from pipecat.frames.frames import (
    CancelFrame, EndFrame, InterimTranscriptionFrame, TranscriptionFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.time import time_now_iso8601

from .transcript_commits import TranscriptCommits, TranscriptOrderError


class CommittedTranscriptSTTMixin:
    def __init__(self, *args, finalize_transcripts: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._finalize_transcripts = finalize_transcripts
        self._transcript_commits = TranscriptCommits()
        self._latest_interim_item: str | None = None
        self._send_lock = asyncio.Lock()
        self._commit_context: ContextVar[int | None] = ContextVar("stt_commit_sequence", default=None)

    async def process_frame(self, frame, direction):
        if not self._finalize_transcripts:
            return await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._transcript_commits.active:
                return
            self._transcript_commits.speech_started()
            self._latest_interim_item = None
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if not self._transcript_commits.active:
                return
            try:
                sequence = self._transcript_commits.reserve_commit()
            except TranscriptOrderError as error:
                await self._finalization_failure(str(error))
                return
            # Keep the stop's identity across awaits/resumed speech inside the
            # stock STT process_frame -> _commit_audio_buffer call chain.
            token = self._commit_context.set(sequence)
            try:
                await super().process_frame(frame, direction)
            finally:
                self._commit_context.reset(token)
            return
        await super().process_frame(frame, direction)

    async def _commit_audio_buffer(self):
        if not self._finalize_transcripts:
            return await super()._commit_audio_buffer()
        if not self._transcript_commits.active:
            return
        try:
            self._transcript_commits.mark_sent(self._commit_context.get())
        except TranscriptOrderError as error:
            await self._finalization_failure(str(error))
            return
        # Register before sending: the acknowledgement can arrive during await.
        await super()._commit_audio_buffer()

    def _model_compatible_message(self, message):
        """Adapt Pipecat's legacy realtime payload for model-specific API fields."""
        if self._settings.model != "gpt-live-transcribe" or message.get("type") != "session.update":
            return message
        session = message.get("session")
        audio = session.get("audio") if isinstance(session, dict) else None
        input_audio = audio.get("input") if isinstance(audio, dict) else None
        transcription = input_audio.get("transcription") if isinstance(input_audio, dict) else None
        if not isinstance(transcription, dict):
            return message

        # Pipecat 1.11 builds the legacy singular field. gpt-live-transcribe's
        # realtime API instead requires `languages` and rejects sending both.
        transcription = dict(transcription)
        transcription.pop("language", None)
        transcription["languages"] = ["en"]
        input_audio = {**input_audio, "transcription": transcription}
        audio = {**audio, "input": input_audio}
        session = {**session, "audio": audio}
        return {**message, "session": session}

    async def _ws_send(self, message):
        message = self._model_compatible_message(message)
        if not self._finalize_transcripts:
            return await super()._ws_send(message)
        # Preserve actual wire order when two stop handlers overlap while the
        # first send is suspended by network backpressure.
        async with self._send_lock:
            if not self._transcript_commits.active or self._disconnecting:
                return
            if self._websocket is None:
                await self._finalization_failure("Transcription socket is unavailable")
                return
            try:
                await self._websocket.send(json.dumps(message))
            except asyncio.CancelledError:
                raise
            except Exception:
                # The stock sender reports errors but swallows them. Do not keep
                # waiting for an ACK, or finalize audio whose append failed.
                await self._finalization_failure("A transcription message could not be sent")

    async def _handle_commit_acknowledged(self, event):
        if not self._transcript_commits.active:
            return
        try:
            self._transcript_commits.acknowledge(
                event.get("item_id"), event.get("previous_item_id"),
            )
        except TranscriptOrderError as error:
            await self._finalization_failure(str(error))
            return
        await self._release_completed_transcripts()

    async def _handle_transcription_delta(self, event):
        if not self._finalize_transcripts:
            return await super()._handle_transcription_delta(event)
        tracker = self._transcript_commits
        if not tracker.active:
            return
        item_id = event.get("item_id", "")
        if tracker.is_complete(item_id):
            return  # Late/duplicate partial text cannot interrupt a newer reply.
        generation = tracker.item_generation(item_id)
        if generation is not None and generation != tracker.generation:
            return
        delta = event.get("delta", "")
        if delta:
            if not isinstance(item_id, str) or not item_id or not isinstance(delta, str):
                await self._finalization_failure("Transcription delta is malformed")
                return
            self._latest_interim_item = item_id
            frame = InterimTranscriptionFrame(delta, self._user_id, time_now_iso8601(), result=event)
            frame.metadata["stt_item_id"] = item_id
            await self.push_frame(frame)

    async def _handle_transcription_completed(self, event):
        if not self._finalize_transcripts:
            return await super()._handle_transcription_completed(event)
        tracker = self._transcript_commits
        if not tracker.active:
            return
        try:
            if event.get("content_index", 0) != 0:
                raise TranscriptOrderError("Unexpected transcription content index")
            accepted = tracker.complete(
                event.get("item_id"), event.get("transcript"), time_now_iso8601(),
            )
        except TranscriptOrderError as error:
            await self._finalization_failure(str(error))
            return
        if accepted:
            await self.emit_stt_usage_metrics()
        await self._release_completed_transcripts()

    async def _release_completed_transcripts(self):
        tracker = self._transcript_commits
        batch = tracker.take_ready()
        if batch is None or not batch.text:
            return
        frame = TranscriptionFrame(
            batch.text, self._user_id, time_now_iso8601(),
            finalized=(
                tracker.is_current(batch)
                and (self._latest_interim_item is None or self._latest_interim_item in batch.item_ids)
            ),
        )
        frame.metadata.update({
            "stt_speech_generation": batch.generation,
            "stt_commit_sequences": list(batch.sequences),
            "stt_item_ids": list(batch.item_ids),
            "stt_completion_received_at": list(batch.received_at),
        })
        # One ordered frame avoids a timeout committing only the first piece.
        # A downstream generation check also covers VAD overtaking this queued
        # data frame after it leaves this service.
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)
        await self._handle_transcription_trace(batch.text, True)

    async def _finalization_failure(self, reason):
        if not self._transcript_commits.active:
            return
        self._transcript_commits.close()
        await self.push_error(
            error_msg=f"STT finalization stopped: {reason}. Reconnect to start a fresh session.",
            force_treat_as_permanent=True,
        )

    async def _handle_transcription_failed(self, event):
        if not self._finalize_transcripts:
            return await super()._handle_transcription_failed(event)
        # Do not skip a missing segment and answer from the remaining words.
        await self._finalization_failure("A committed audio segment could not be transcribed")

    async def _receive_messages(self):
        if not self._finalize_transcripts:
            return await super()._receive_messages()
        assert self._websocket is not None
        handlers = {
            "session.created": self._handle_session_created,
            "session.updated": self._handle_session_updated,
            "input_audio_buffer.committed": self._handle_commit_acknowledged,
            "conversation.item.input_audio_transcription.delta": self._handle_transcription_delta,
            "conversation.item.input_audio_transcription.completed": self._handle_transcription_completed,
            "conversation.item.input_audio_transcription.failed": self._handle_transcription_failed,
        }
        try:
            async for message in self._websocket:
                if not self._transcript_commits.active:
                    return
                try:
                    event = json.loads(message)
                    if not isinstance(event, dict):
                        raise ValueError
                except (ValueError, UnicodeError):
                    await self._finalization_failure("Malformed transcription event")
                    return
                event_type = event.get("type")
                if event_type == "error":
                    code = (event.get("error") or {}).get("code")
                    known_codes = {
                        "invalid_api_key", "insufficient_quota", "rate_limit_exceeded",
                        "input_audio_buffer_commit_empty",
                    }
                    detail = code if code in known_codes else "unspecified"
                    await self._finalization_failure(f"Transcription provider error ({detail})")
                    return
                if event_type in {"input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"}:
                    await self._finalization_failure("Unexpected server-side turn detection")
                    return
                handler = handlers.get(event_type)
                if handler is not None:
                    await handler(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._finalization_failure("Transcription stream failed")
        finally:
            if self._transcript_commits.active and not self._disconnecting:
                await self._finalization_failure("Transcription stream closed")
            self._transcript_commits.close()

    async def stop(self, frame: EndFrame):
        self._transcript_commits.close()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._transcript_commits.close()
        await super().cancel(frame)

    async def cleanup(self):
        self._transcript_commits.close()
        await super().cleanup()
