#
# Copyright (c) 2024–2025, Agora
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import resample_audio
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams

try:
    from agora_realtime_ai_api.rtc import RtcEngine, RtcOptions, Channel
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Agora, you need to `pip install pipecat-ai[agora]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class AgoraTransportMessageFrame(TransportMessageFrame):
    participant_id: str | None = None


@dataclass
class AgoraTransportMessageUrgentFrame(TransportMessageUrgentFrame):
    participant_id: str | None = None


class AgoraParams(TransportParams):
    app_id: str = ""
    app_cert: str = ""


class AgoraCallbacks(BaseModel):
    on_connected: Callable[[], Awaitable[None]]
    on_disconnected: Callable[[], Awaitable[None]]
    on_participant_connected: Callable[[str], Awaitable[None]]
    on_participant_disconnected: Callable[[str], Awaitable[None]]
    on_audio_track_subscribed: Callable[[str], Awaitable[None]]
    on_audio_track_unsubscribed: Callable[[str], Awaitable[None]]
    on_data_received: Callable[[bytes, str], Awaitable[None]]
    on_first_participant_joined: Callable[[str], Awaitable[None]]


class AgoraTransportClient:
    def __init__(
        self,
        params: AgoraParams,
        callbacks: AgoraCallbacks,
        loop: asyncio.AbstractEventLoop,
    ):
        self._params = params
        self._callbacks = callbacks
        self._loop = loop
        self._engine = RtcEngine(appid=params.app_id, appcert=params.app_cert)
        self._channel: Channel | None = None
        self._participant_id: str = ""
        self._connected = False
        self._disconnect_counter = 0
        self._audio_queue = asyncio.Queue()
        self._other_participant_has_joined = False
        self._remote_user_id: Optional[int] = None

    @property
    def participant_id(self) -> str:
        return self._participant_id

    async def connect(self, channel_name: str, uid: int):
        if self._connected:
            self._disconnect_counter += 1
            return

        logger.info(f"Connecting to {channel_name}")

        try:
            options = RtcOptions(
                channel_name=channel_name,
                uid=uid,
                sample_rate=self._params.audio_out_sample_rate,
                channels=self._params.audio_out_channels,
            )
            self._channel = self._engine.create_channel(options)
            await self._channel.connect()

            self._connected = True
            self._disconnect_counter += 1
            self._participant_id = str(uid)

            # Set up channel event handlers
            self._channel.on("user_joined")(self._on_user_joined_wrapper)
            self._channel.on("user_left")(self._on_user_left_wrapper)
            self._channel.on("stream_message")(self._on_stream_message_wrapper)

            await self._callbacks.on_connected()
            logger.info(f"Connected to {channel_name}")

            # Wait for and subscribe to remote user
            try:
                remote_users = list(self._channel.remote_users.keys())
                if remote_users:
                    self._remote_user_id = remote_users[0]
                    await self._channel.subscribe_audio(self._remote_user_id)
            except Exception as e:
                logger.error(f"Error subscribing to remote user: {e}")

        except Exception as e:
            logger.error(f"Error connecting to {channel_name}: {e}")
            raise

    async def disconnect(self):
        self._disconnect_counter -= 1

        if not self._connected or self._disconnect_counter > 0:
            return

        logger.info("Disconnecting from channel")
        if self._channel:
            await self._channel.disconnect()
            self._connected = False
            logger.info("Disconnected from channel")
            await self._callbacks.on_disconnected()

    async def send_data(self, data: bytes, participant_id: str | None = None):
        if not self._connected or not self._channel:
            return

        try:
            # Convert participant_id to int since Agora uses numeric UIDs
            target_uid = int(participant_id) if participant_id else None
            await self._channel.send_stream_message(data, target_uid)
        except Exception as e:
            logger.error(f"Error sending data: {e}")

    async def write_audio_frame(self, audio_frame: bytes):
        if not self._connected or not self._channel:
            return

        try:
            await self._channel.push_audio_frame(audio_frame)
        except Exception as e:
            logger.error(f"Error publishing audio: {e}")

    async def get_next_audio_frame(self) -> Optional[bytes]:
        if not self._remote_user_id or not self._channel:
            return None

        try:
            audio_frames = self._channel.get_audio_frames(self._remote_user_id)
            async for frame in audio_frames:
                return frame.data
        except Exception as e:
            logger.error(f"Error getting audio frame: {e}")
            return None

    def get_participants(self) -> List[str]:
        if not self._channel:
            return []
        return [str(uid) for uid in self._channel.remote_users.keys()]

    # Event handler wrappers
    def _on_user_joined_wrapper(self, _conn, user_id: int):
        asyncio.create_task(self._async_on_user_joined(user_id))

    def _on_user_left_wrapper(self, _conn, user_id: int, _reason: int):
        asyncio.create_task(self._async_on_user_left(user_id))

    def _on_stream_message_wrapper(self, _conn, user_id: int, _stream_id: int, data: bytes, length: int):
        asyncio.create_task(self._async_on_stream_message(user_id, data))

    # Async event handlers
    async def _async_on_user_joined(self, user_id: int):
        logger.info(f"User joined: {user_id}")
        self._remote_user_id = user_id
        participant_id = str(user_id)
        await self._callbacks.on_participant_connected(participant_id)
        
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._callbacks.on_first_participant_joined(participant_id)

        # Subscribe to audio when user joins
        if self._channel:
            await self._channel.subscribe_audio(user_id)
            await self._callbacks.on_audio_track_subscribed(participant_id)

    async def _async_on_user_left(self, user_id: int):
        logger.info(f"User left: {user_id}")
        if self._remote_user_id == user_id:
            self._remote_user_id = None
        participant_id = str(user_id)
        await self._callbacks.on_participant_disconnected(participant_id)
        await self._callbacks.on_audio_track_unsubscribed(participant_id)

    async def _async_on_stream_message(self, user_id: int, data: bytes):
        await self._callbacks.on_data_received(data, str(user_id))


class AgoraInputTransport(BaseInputTransport):
    def __init__(self, client: AgoraTransportClient, params: AgoraParams, **kwargs):
        super().__init__(params, **kwargs)
        self._client = client
        self._audio_in_task = None
        self._vad_analyzer: VADAnalyzer | None = params.vad_analyzer

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._params.audio_in_enabled or self._params.vad_enabled:
            self._audio_in_task = asyncio.create_task(self._audio_in_task_handler())
        logger.info("AgoraInputTransport started")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        if self._audio_in_task:
            self._audio_in_task.cancel()
            await self._audio_in_task
        logger.info("AgoraInputTransport stopped")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        if self._audio_in_task:
            self._audio_in_task.cancel()
            await self._audio_in_task

    def vad_analyzer(self) -> VADAnalyzer | None:
        return self._vad_analyzer

    async def push_app_message(self, message: Any, sender: str):
        frame = AgoraTransportMessageUrgentFrame(message=message, participant_id=sender)
        await self.push_frame(frame)

    async def _audio_in_task_handler(self):
        logger.info("Audio input task started")
        while True:
            try:
                if not self._remote_user_id or not self._channel:
                    await asyncio.sleep(0.1)
                    continue

                audio_frames = self._channel.get_audio_frames(self._remote_user_id)
                async for audio_frame in audio_frames:
                    input_audio_frame = InputAudioRawFrame(
                        audio=audio_frame.data,
                        sample_rate=self._params.audio_in_sample_rate,
                        num_channels=self._params.audio_in_channels,
                    )
                    await self.push_audio_frame(input_audio_frame)
                    
            except asyncio.CancelledError:
                logger.info("Audio input task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in audio input task: {e}")
                await asyncio.sleep(0.1)  # Avoid tight loop on errors


class AgoraOutputTransport(BaseOutputTransport):
    def __init__(self, client: AgoraTransportClient, params: AgoraParams, **kwargs):
        super().__init__(params, **kwargs)
        self._client = client

    async def start(self, frame: StartFrame):
        await super().start(frame)
        logger.info("AgoraOutputTransport started")

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        logger.info("AgoraOutputTransport stopped")

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        if isinstance(frame, (AgoraTransportMessageFrame, AgoraTransportMessageUrgentFrame)):
            await self._client.send_data(frame.message.encode(), frame.participant_id)
        else:
            await self._client.send_data(frame.message.encode())

    async def write_raw_audio_frames(self, frames: bytes):
        if not self._connected or not self._channel:
            return

        try:
            await self._channel.push_audio_frame(frames)
        except Exception as e:
            logger.error(f"Error publishing audio: {e}")


class AgoraTransport(BaseTransport):
    def __init__(
        self,
        channel_name: str,
        uid: int,
        params: AgoraParams,
        input_name: str | None = None,
        output_name: str | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name, loop=loop)

        callbacks = AgoraCallbacks(
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_participant_connected=self._on_participant_connected,
            on_participant_disconnected=self._on_participant_disconnected,
            on_audio_track_subscribed=self._on_audio_track_subscribed,
            on_audio_track_unsubscribed=self._on_audio_track_unsubscribed,
            on_data_received=self._on_data_received,
            on_first_participant_joined=self._on_first_participant_joined,
        )

        self._channel_name = channel_name
        self._uid = uid
        self._params = params
        self._client = AgoraTransportClient(params, callbacks, self._loop)
        self._input: AgoraInputTransport | None = None
        self._output: AgoraOutputTransport | None = None

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_participant_connected")
        self._register_event_handler("on_participant_disconnected")
        self._register_event_handler("on_audio_track_subscribed")
        self._register_event_handler("on_audio_track_unsubscribed")
        self._register_event_handler("on_data_received")
        self._register_event_handler("on_first_participant_joined")

    def input(self) -> AgoraInputTransport:
        if not self._input:
            self._input = AgoraInputTransport(self._client, self._params, name=self._input_name)
        return self._input

    def output(self) -> AgoraOutputTransport:
        if not self._output:
            self._output = AgoraOutputTransport(self._client, self._params, name=self._output_name)
        return self._output

    @property
    def participant_id(self) -> str:
        return self._client.participant_id

    async def send_audio(self, frame: OutputAudioRawFrame):
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    def get_participants(self) -> List[str]:
        return self._client.get_participants()

    async def start(self):
        await self._client.connect(self._channel_name, self._uid)

    async def stop(self):
        await self._client.disconnect()

    async def cleanup(self):
        await self.stop()
        if self._input:
            await self._input.cleanup()
        if self._output:
            await self._output.cleanup()

    # Event handlers
    async def _on_connected(self):
        await self._call_event_handler("on_connected")

    async def _on_disconnected(self):
        await self._call_event_handler("on_disconnected")

    async def _on_participant_connected(self, participant_id: str):
        await self._call_event_handler("on_participant_connected", participant_id)

    async def _on_participant_disconnected(self, participant_id: str):
        await self._call_event_handler("on_participant_disconnected", participant_id)

    async def _on_audio_track_subscribed(self, participant_id: str):
        await self._call_event_handler("on_audio_track_subscribed", participant_id)

    async def _on_audio_track_unsubscribed(self, participant_id: str):
        await self._call_event_handler("on_audio_track_unsubscribed", participant_id)

    async def _on_data_received(self, data: bytes, participant_id: str):
        if self._input:
            await self._input.push_app_message(data.decode(), participant_id)
        await self._call_event_handler("on_data_received", data, participant_id)

    async def _on_first_participant_joined(self, participant_id: str):
        await self._call_event_handler("on_first_participant_joined", participant_id)
