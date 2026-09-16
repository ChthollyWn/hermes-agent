"""Feishu topic mode: the ``/topic`` command is the switch, never a config key.

Design under test:
* ``/topic``            → turn the mode ON (persisted per chat)
* ``/topic status``     → report the current state
* ``/topic off``        → leave the mode
* ``/topic help``       → usage
* ``/topic <标题>``     → turn it on and open one titled topic right away
* mode off (default, and for any chat that never opted in) → stock Hermes behaviour, untouched.

Auto-opened topics carry no placeholder post: the answer's own reply to the question creates the
topic, and the adapter maps that thread back to the question so follow-ups share its session.
"""

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_topics import GatewayTopicThreadsMixin
from gateway.session import SessionSource, build_session_key
from plugins.platforms.feishu import adapter as feishu_adapter

FEISHU_CHAT = "oc_821766e4678dd2d41cf2d97b0ee048a2"
OTHER_CHAT = "oc_00000000000000000000000000000000"


class _FakeResp:
    def __init__(self, thread_id="omt_auto1", ok=True, message_id="om_bot1"):
        self.data = (
            SimpleNamespace(thread_id=thread_id, message_id=message_id) if ok else None
        )
        self.code = 0 if ok else 9499
        self.msg = "ok" if ok else "boom"

    def success(self):
        return self.code == 0


class _FakeMessageAPI:
    def __init__(self, resp_factory=None, raise_exc=None, update_resp_factory=None):
        self.calls = []
        self.updates = []
        self.creates = []
        self.resp_factory = resp_factory or _FakeResp
        self.update_resp_factory = update_resp_factory or self.resp_factory
        self.raise_exc = raise_exc

    def reply(self, request):
        self.calls.append(request)
        if self.raise_exc:
            raise self.raise_exc
        return self.resp_factory()

    def update(self, request):
        self.updates.append(request)
        if self.raise_exc:
            raise self.raise_exc
        return self.update_resp_factory()

    def create(self, request):
        self.creates.append(request)
        if self.raise_exc:
            raise self.raise_exc
        return self.resp_factory()


class _FakeClient:
    def __init__(self, **kw):
        self.im = SimpleNamespace(v1=SimpleNamespace(message=_FakeMessageAPI(**kw)))


class _SentRecorder:
    """Stands in for ``adapter.send`` and records every outbound reply."""

    def __init__(self):
        self.sent = []

    async def __call__(self, chat_id, content, reply_to=None, metadata=None, **kwargs):
        self.sent.append(
            {"chat_id": chat_id, "text": content, "reply_to": reply_to, "metadata": metadata})
        return True

    @property
    def last_text(self):
        return self.sent[-1]["text"] if self.sent else ""


class _AdapterHarness(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.state_path = Path(self._tmp.name) / "feishu_topic_mode.json"
        self.addCleanup(self._tmp.cleanup)

    def make_adapter(self, *, connected=True, extra=None, real_send=False, **client_kw):
        adapter = object.__new__(feishu_adapter.FeishuAdapter)
        adapter._client = _FakeClient(**client_kw) if connected else None
        adapter.config = SimpleNamespace(extra=extra if extra is not None else {})
        adapter._topic_state_lock = threading.RLock()
        adapter._topic_state_cache = None
        adapter._topic_state_path = lambda: self.state_path
        self.recorder = _SentRecorder()
        if not real_send:
            adapter.send = self.recorder  # type: ignore[method-assign]  # test double
        return adapter

    def run_cmd(self, adapter, command, *, chat_id=FEISHU_CHAT, thread_id=None, message_id="om_cmd"):
        """/topic <command> through the adapter's DM command handler."""
        parts = command.split(maxsplit=1)
        asyncio.run(adapter._handle_topic_command_message(
            chat_id=chat_id, message_id=message_id, thread_id=thread_id,
            arg=(parts[1].strip() if len(parts) > 1 else ""), user_id="ou_owner",
        ))
        return adapter


class TopicModeSwitchTests(_AdapterHarness):
    def test_default_off_for_chat_that_never_opted_in(self):
        adapter = self.make_adapter()
        self.assertFalse(adapter.topic_mode_enabled(FEISHU_CHAT))
        self.assertEqual(adapter.topic_mode_record(FEISHU_CHAT), {})

    def test_config_extra_cannot_switch_it_on(self):
        """The switch is command-driven: a leftover config key must have no effect."""
        adapter = self.make_adapter(extra={"topic_mode": True, "topic_command": True})
        self.assertFalse(adapter.topic_mode_enabled(FEISHU_CHAT))
        self.assertEqual(adapter.topic_mode_record(FEISHU_CHAT), {})

    def test_state_is_per_chat(self):
        adapter = self.make_adapter()
        adapter.set_topic_mode(FEISHU_CHAT, True, user_id="ou_owner")
        self.assertTrue(adapter.topic_mode_enabled(FEISHU_CHAT))
        self.assertFalse(adapter.topic_mode_enabled(OTHER_CHAT))

    def test_state_persists_across_adapter_instances(self):
        self.make_adapter().set_topic_mode(FEISHU_CHAT, True, user_id="ou_owner")
        reloaded = self.make_adapter()
        self.assertTrue(reloaded.topic_mode_enabled(FEISHU_CHAT))
        self.assertEqual(reloaded.topic_mode_record(FEISHU_CHAT)["user_id"], "ou_owner")
        self.assertIn("activated_at", reloaded.topic_mode_record(FEISHU_CHAT))

    def test_disable_keeps_record_but_reports_off(self):
        adapter = self.make_adapter()
        adapter.set_topic_mode(FEISHU_CHAT, True)
        adapter.set_topic_mode(FEISHU_CHAT, False)
        self.assertFalse(adapter.topic_mode_enabled(FEISHU_CHAT))
        self.assertFalse(adapter.topic_mode_record(FEISHU_CHAT)["enabled"])
        self.assertIn("activated_at", adapter.topic_mode_record(FEISHU_CHAT))

    def test_corrupt_state_file_reads_as_off(self):
        self.state_path.write_text("{not json", encoding="utf-8")
        adapter = self.make_adapter()
        self.assertFalse(adapter.topic_mode_enabled(FEISHU_CHAT))
        adapter.set_topic_mode(FEISHU_CHAT, True)  # must recover, not raise
        self.assertTrue(adapter.topic_mode_enabled(FEISHU_CHAT))
        self.assertTrue(json.loads(self.state_path.read_text(encoding="utf-8"))["chats"])

    def test_unwritable_state_path_reports_failure_and_keeps_state(self):
        adapter = self.make_adapter()
        adapter._topic_state_path = lambda: Path("/proc/nope/feishu_topic_mode.json")
        self.assertIsNone(adapter.set_topic_mode(FEISHU_CHAT, True))  # logged, not raised
        self.assertFalse(adapter.topic_mode_enabled(FEISHU_CHAT))

    def test_command_reports_write_failure_instead_of_pretending(self):
        adapter = self.make_adapter()
        adapter._topic_state_path = lambda: Path("/proc/nope/feishu_topic_mode.json")
        self.run_cmd(adapter, "/topic")
        self.assertIn("开启失败", self.recorder.last_text)
        self.assertFalse(adapter.topic_mode_enabled(FEISHU_CHAT))


class TopicCommandTests(_AdapterHarness):
    def test_bare_topic_enables_mode_without_opening_a_topic(self):
        adapter = self.run_cmd(self.make_adapter(), "/topic")
        self.assertTrue(adapter.topic_mode_enabled(FEISHU_CHAT))
        self.assertEqual(adapter._client.im.v1.message.calls, [])
        self.assertIn("已开启", self.recorder.last_text)

    def test_chinese_aliases_enable_mode(self):
        for command in ("/新话题", "/开个话题"):
            with self.subTest(command):
                adapter = self.run_cmd(self.make_adapter(), command)
                self.assertTrue(adapter.topic_mode_enabled(FEISHU_CHAT))

    def test_status_reports_off_then_on(self):
        adapter = self.make_adapter()
        self.run_cmd(adapter, "/topic status")
        self.assertIn("已关闭", self.recorder.last_text)
        self.run_cmd(adapter, "/topic")
        self.run_cmd(adapter, "/topic status")
        self.assertIn("已开启", self.recorder.last_text)

    def test_status_aliases_and_case(self):
        for command in ("/topic Status", "/topic 状态", "/topic 查看"):
            with self.subTest(command):
                adapter = self.make_adapter()
                self.run_cmd(adapter, command)
                self.assertIn("已关闭", self.recorder.last_text)

    def test_off_disables_mode(self):
        adapter = self.make_adapter()
        self.run_cmd(adapter, "/topic")
        self.run_cmd(adapter, "/topic off")
        self.assertFalse(adapter.topic_mode_enabled(FEISHU_CHAT))
        self.assertEqual(self.recorder.last_text, "🗂 topic 模式已关闭。")

    def test_off_aliases(self):
        for command in ("/topic disable", "/topic 关闭", "/topic 退出"):
            with self.subTest(command):
                adapter = self.make_adapter()
                adapter.set_topic_mode(FEISHU_CHAT, True)
                self.run_cmd(adapter, command)
                self.assertFalse(adapter.topic_mode_enabled(FEISHU_CHAT))

    def test_help_lists_subcommands(self):
        adapter = self.run_cmd(self.make_adapter(), "/topic help")
        self.assertFalse(adapter.topic_mode_enabled(FEISHU_CHAT))  # help must not enable
        for token in ("/topic off", "/topic status", "/topic <标题>"):
            self.assertIn(token, self.recorder.last_text)

    def test_title_argument_enables_and_opens_titled_topic(self):
        adapter = self.run_cmd(self.make_adapter(), "/topic 查机票")
        self.assertTrue(adapter.topic_mode_enabled(FEISHU_CHAT))
        call = adapter._client.im.v1.message.calls[0]
        self.assertEqual(call.message_id, "om_cmd")
        self.assertIs(call.request_body.reply_in_thread, True)
        self.assertIn("查机票", call.request_body.content)

    def test_inside_topic_enables_but_never_nests_a_topic(self):
        adapter = self.run_cmd(self.make_adapter(), "/topic", thread_id="omt_existing")
        self.assertTrue(adapter.topic_mode_enabled(FEISHU_CHAT))
        self.assertEqual(adapter._client.im.v1.message.calls, [])
        self.assertIn("已在话题内", self.recorder.last_text)
        self.assertEqual(self.recorder.sent[-1]["metadata"]["thread_id"], "omt_existing")

    def test_inside_topic_status_mentions_current_topic(self):
        adapter = self.make_adapter()
        adapter.set_topic_mode(FEISHU_CHAT, True)
        self.run_cmd(adapter, "/topic status", thread_id="omt_existing")
        self.assertIn("已开启", self.recorder.last_text)
        self.assertIn("已在话题内", self.recorder.last_text)

    def test_inside_topic_off_works(self):
        adapter = self.make_adapter()
        adapter.set_topic_mode(FEISHU_CHAT, True)
        self.run_cmd(adapter, "/topic off", thread_id="omt_existing")
        self.assertFalse(adapter.topic_mode_enabled(FEISHU_CHAT))

    def test_disconnected_adapter_ignores_command(self):
        adapter = self.run_cmd(self.make_adapter(connected=False), "/topic")
        self.assertEqual(self.recorder.sent, [])
        self.assertFalse(adapter.topic_mode_enabled(FEISHU_CHAT))


class OpenTopicForMessageTests(_AdapterHarness):
    def test_creates_thread_with_reply_in_thread(self):
        adapter = self.make_adapter()
        thread_id = asyncio.run(adapter.open_topic_for_message("om_anchor"))
        call = adapter._client.im.v1.message.calls[0]
        self.assertEqual(thread_id, "omt_auto1")
        self.assertEqual(call.message_id, "om_anchor")
        self.assertIs(call.request_body.reply_in_thread, True)
        # Only the explicit titled path uses this helper; its seed text is the topic's first post.
        self.assertEqual(json.loads(call.request_body.content)["text"], "🗂")

    def test_seed_and_title_overrides(self):
        adapter = self.make_adapter()
        asyncio.run(adapter.open_topic_for_message("om_anchor", seed_text="自定义种子", title="查机票"))
        self.assertIn("自定义种子：查机票", adapter._client.im.v1.message.calls[0].request_body.content)

    def test_api_failure_returns_none(self):
        adapter = self.make_adapter(resp_factory=lambda: _FakeResp(ok=False))
        self.assertIsNone(asyncio.run(adapter.open_topic_for_message("om_anchor")))

    def test_exception_returns_none(self):
        adapter = self.make_adapter(raise_exc=RuntimeError("network down"))
        self.assertIsNone(asyncio.run(adapter.open_topic_for_message("om_anchor")))

    def test_disconnected_or_anchorless_returns_none(self):
        self.assertIsNone(asyncio.run(
            self.make_adapter(connected=False).open_topic_for_message("om_a")))
        self.assertIsNone(asyncio.run(
            self.make_adapter().open_topic_for_message("")))


class TopicAnchorRoutingTests(_AdapterHarness):
    """The answer opens the topic — no placeholder post, no pre-created topic.

    The gateway stamps the question's message id as the session's thread key; the first reply to
    that anchor carries ``reply_in_thread`` and therefore creates the topic. The adapter then maps
    the real thread id back to the anchor, so follow-ups stay in the session the question opened.
    """

    def _reply_into(self, adapter, anchor="om_q1", text="答案", reply_to=None):
        return asyncio.run(adapter.send(
            FEISHU_CHAT, text, reply_to=reply_to, metadata={"thread_id": anchor},
        ))

    def test_reply_to_the_anchor_opens_the_topic_and_maps_it(self):
        adapter = self.make_adapter(real_send=True)
        self.assertTrue(self._reply_into(adapter).success)
        call = adapter._client.im.v1.message.calls[0]
        self.assertEqual(call.message_id, "om_q1")                # anchored on the question
        self.assertIs(call.request_body.reply_in_thread, True)    # …which opens the topic
        self.assertIn("答案", call.request_body.content)
        self.assertEqual(adapter._client.im.v1.message.creates, [])  # nothing pre-created
        self.assertEqual(adapter._client.im.v1.message.updates, [])  # nothing to overwrite
        self.assertEqual(adapter._topic_anchor_for_thread("omt_auto1"), "om_q1")

    def test_mapping_persists_across_instances(self):
        self._reply_into(self.make_adapter(real_send=True))
        reloaded = self.make_adapter()
        self.assertEqual(reloaded._topic_anchor_for_thread("omt_auto1"), "om_q1")
        self.assertIsNone(reloaded._topic_anchor_for_thread("omt_other"))
        self.assertIsNone(reloaded._topic_anchor_for_thread("om_q1"))  # anchors are not threads

    def test_follow_up_inside_the_topic_replies_in_place(self):
        adapter = self.make_adapter(real_send=True)
        self._reply_into(adapter)
        before = self.state_path.read_text(encoding="utf-8")
        self._reply_into(adapter, text="第二段", reply_to="om_follow_up")
        self.assertEqual(self.state_path.read_text(encoding="utf-8"), before)  # written once
        second = adapter._client.im.v1.message.calls[1]
        self.assertEqual(second.message_id, "om_follow_up")
        self.assertIs(second.request_body.reply_in_thread, True)

    def test_send_without_reply_target_still_anchors_on_the_question(self):
        """Media/card sends carry no reply_to: the anchor becomes the reply target instead of a
        receive_id thread that does not exist yet."""
        adapter = self.make_adapter(real_send=True)
        asyncio.run(adapter._send_raw_message(
            chat_id=FEISHU_CHAT, msg_type="text", payload='{"text":"hi"}',
            reply_to=None, metadata={"thread_id": "om_q1"},
        ))
        self.assertEqual(adapter._client.im.v1.message.calls[0].message_id, "om_q1")
        self.assertEqual(adapter._client.im.v1.message.creates, [])

    def test_main_dm_send_is_untouched(self):
        adapter = self.make_adapter(real_send=True)
        asyncio.run(adapter.send(FEISHU_CHAT, "主会话回答"))
        self.assertEqual(adapter._client.im.v1.message.calls, [])
        self.assertEqual(len(adapter._client.im.v1.message.creates), 1)

    def test_copy_stays_terse(self):
        adapter = self.make_adapter()
        self.run_cmd(adapter, "/topic")
        self.assertNotIn("从现在起", self.recorder.last_text)
        self.assertLessEqual(len(self.recorder.last_text.splitlines()), 2)
        self.run_cmd(adapter, "/topic help")
        self.assertLessEqual(len(self.recorder.last_text.splitlines()), 5)
        self.assertNotIn("•", self.recorder.last_text)
        self.run_cmd(adapter, "/topic status")
        self.assertLessEqual(len(self.recorder.last_text.splitlines()), 2)


class _FakeRunnerAdapter:
    """Minimal adapter surface the runner helper touches; mode is per chat_id."""

    def __init__(self, *, enabled_chats=(), thread_id="omt_auto1", raises=False):
        self.enabled_chats = set(enabled_chats)
        self.thread_id = thread_id
        self.raises = raises
        self.opened = []
        self.mode_calls = []

    def topic_mode_enabled(self, chat_id=""):
        self.mode_calls.append(chat_id)
        if self.raises:
            raise RuntimeError("state read blew up")
        return str(chat_id) in self.enabled_chats

    async def open_topic_for_message(self, message_id, *, seed_text="", title=""):
        self.opened.append(message_id)
        if self.raises:
            raise RuntimeError("api blew up")
        return self.thread_id

    def _resolve_channel_prompt(self, chat_id, thread_id):
        return f"prompt-for:{thread_id}"


class _DummyRunner(GatewayTopicThreadsMixin):
    def __init__(self, adapter):
        self._adapter = adapter

    def _adapter_for_source(self, source):
        return self._adapter


def _source(*, thread_id=None, chat_type="dm", chat_id=FEISHU_CHAT, platform=Platform.FEISHU):
    return SessionSource(
        platform=platform, chat_id=chat_id, chat_name="Home", chat_type=chat_type,
        user_id="ou_d2185d8b4b8f4ca2779765ea7fce1c06", user_name="owner", thread_id=thread_id,
    )


def _event(source, text="check tomorrow's flights", message_id="om_q1", internal=False):
    return MessageEvent(
        text=text, message_type=MessageType.TEXT, source=source,
        message_id=message_id, internal=internal,
    )


class RunnerAutoTopicTests(unittest.TestCase):
    def _run(self, adapter, source, event):
        return asyncio.run(_DummyRunner(adapter)._hm_maybe_auto_open_feishu_topic(event, source))

    def test_mode_off_keeps_native_behaviour(self):
        adapter = _FakeRunnerAdapter(enabled_chats=())
        source, event = _source(), _event(_source())
        _, out_source = self._run(adapter, source, event)
        self.assertEqual(adapter.opened, [])
        self.assertIsNone(out_source.thread_id)
        self.assertEqual(adapter.mode_calls, [FEISHU_CHAT])  # asked for THIS chat, not a global flag

    def test_mode_on_stamps_the_question_as_its_own_session(self):
        adapter = _FakeRunnerAdapter(enabled_chats={FEISHU_CHAT})
        source, event = _source(), _event(_source())
        out_event, out_source = self._run(adapter, source, event)
        self.assertEqual(adapter.opened, [])  # no API call: the answer itself opens the topic
        self.assertEqual(out_source.thread_id, "om_q1")
        self.assertEqual(out_event.source.thread_id, "om_q1")
        self.assertEqual(out_event.channel_prompt, "prompt-for:om_q1")

    def test_other_chat_is_unaffected(self):
        adapter = _FakeRunnerAdapter(enabled_chats={OTHER_CHAT})
        source, event = _source(chat_id=FEISHU_CHAT), _event(_source(chat_id=FEISHU_CHAT))
        _, out_source = self._run(adapter, source, event)
        self.assertEqual(adapter.opened, [])
        self.assertIsNone(out_source.thread_id)

    def test_session_key_splits_main_from_topic(self):
        main_key = build_session_key(_source())
        topic_key = build_session_key(_source(thread_id="om_q1"))
        self.assertNotEqual(main_key, topic_key)
        self.assertIn("om_q1", topic_key)
        self.assertNotIn("om_q1", main_key)

    def test_skips_topic_group_command_internal_and_foreign(self):
        cases = (
            ("already in a topic", {"thread_id": "omt_old"}, "continue", False),
            ("group chat", {"chat_type": "group"}, "check flights", False),
            ("slash command", {}, "/new", False),
            ("internal event", {}, "check flights", True),
        )
        for label, source_kw, text, internal in cases:
            with self.subTest(label):
                adapter = _FakeRunnerAdapter(enabled_chats={FEISHU_CHAT})
                source = _source(**source_kw)
                _, out_source = self._run(adapter, source, _event(source, text=text, internal=internal))
                self.assertEqual(adapter.opened, [])
                self.assertEqual(out_source.thread_id, source.thread_id)

        adapter = _FakeRunnerAdapter(enabled_chats={FEISHU_CHAT})
        source = _source(platform=Platform.WEIXIN)
        self._run(adapter, source, _event(source))
        self.assertEqual(adapter.opened, [])

    def test_anchorless_or_broken_adapter_stays_in_the_main_dm(self):
        source = _source()
        self._run(_FakeRunnerAdapter(enabled_chats={FEISHU_CHAT}), source, _event(source, message_id=None))
        self.assertIsNone(source.thread_id)

        source = _source()
        self._run(None, source, _event(source))
        self.assertIsNone(source.thread_id)

        adapter = _FakeRunnerAdapter(enabled_chats={FEISHU_CHAT}, raises=True)
        source = _source()
        self._run(adapter, source, _event(source))  # must not raise
        self.assertIsNone(source.thread_id)

        source = _source()
        self._run(_FakeRunnerAdapter(enabled_chats={FEISHU_CHAT}), source, _event(source, message_id=None))
        self.assertIsNone(source.thread_id)

        source = _source()
        self._run(None, source, _event(source))
        self.assertIsNone(source.thread_id)


class InboundInterceptionTests(unittest.TestCase):
    """``/topic …`` is swallowed by the adapter (zero model calls); anything else passes through."""

    def _drive(self, text, *, chat_type="p2p", message_id="om_d", thread_id=None, adapter=None):
        adapter = adapter or object.__new__(feishu_adapter.FeishuAdapter)
        adapter._client = _FakeClient()
        adapter.config = SimpleNamespace(extra={})  # type: ignore[assignment]  # test double
        seen, leaked = [], []

        async def fake_extract(message):
            return (text, MessageType.TEXT, [], [], [])

        async def record_handler(*, chat_id, message_id, thread_id, arg="", user_id=""):
            seen.append((chat_id, message_id, thread_id, arg))

        async def record_chat_info(chat_id):
            leaked.append(chat_id)
            raise RuntimeError("stop here — proves the message was not intercepted")

        adapter._extract_message_content = fake_extract
        adapter._handle_topic_command_message = record_handler
        adapter.get_chat_info = record_chat_info
        msg = SimpleNamespace(
            chat_id="oc_a", thread_id=thread_id, parent_id=None,
            upper_message_id=None, root_id=None)
        try:
            asyncio.run(adapter._process_inbound_message(
                data={}, message=msg, sender_id=SimpleNamespace(open_id="ou_u"),
                chat_type=chat_type, message_id=message_id))
        except RuntimeError:
            pass
        return seen, leaked

    def test_topic_command_is_intercepted_with_argument(self):
        seen, leaked = self._drive("/新话题 测试")
        self.assertEqual(seen, [("oc_a", "om_d", None, "测试")])
        self.assertEqual(leaked, [])

    def test_status_and_off_subcommands_are_intercepted(self):
        for text, arg in (("/topic status", "status"), ("/topic off", "off")):
            with self.subTest(text):
                seen, _ = self._drive(text)
                self.assertEqual(seen, [("oc_a", "om_d", None, arg)])

    def test_other_commands_pass_through(self):
        seen, leaked = self._drive("/status")
        self.assertEqual(seen, [])
        self.assertEqual(leaked, ["oc_a"])

    def test_group_chat_topic_passes_through(self):
        seen, leaked = self._drive("/topic", chat_type="group")
        self.assertEqual(seen, [])
        self.assertEqual(leaked, ["oc_a"])

    def test_follow_up_inside_a_mapped_topic_keeps_the_question_session(self):
        """A mapped thread id is rewritten to the anchor the question stamped, so the follow-up
        shares the first turn's session instead of splitting into a new one."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = object.__new__(feishu_adapter.FeishuAdapter)
            adapter._client = _FakeClient()
            adapter._topic_state_lock = threading.RLock()
            adapter._topic_state_cache = None
            adapter._topic_state_path = lambda: Path(tmp) / "feishu_topic_mode.json"
            asyncio.run(adapter._send_raw_message(
                chat_id="oc_a", msg_type="text", payload='{"text":"答案"}',
                reply_to=None, metadata={"thread_id": "om_q1"},
            ))
            self.assertEqual(adapter._topic_anchor_for_thread("omt_auto1"), "om_q1")
            seen, leaked = self._drive("/topic status", thread_id="omt_auto1", adapter=adapter)
            self.assertEqual(seen, [("oc_a", "om_d", "om_q1", "status")])
            self.assertEqual(leaked, [])



class EarlyTopicStampTests(_AdapterHarness):
    """Adapter must stamp topic thread_id before session claim so main-DM questions parallelize."""

    def _dm_event(self, *, message_id, text="check flights", thread_id=None, chat_type="dm",
                  chat_id=FEISHU_CHAT, internal=False):
        return MessageEvent(
            text=text,
            message_type=MessageType.TEXT if not text.startswith("/") else MessageType.COMMAND,
            source=SessionSource(
                platform=Platform.FEISHU, chat_id=chat_id, chat_name="Home",
                chat_type=chat_type, user_id="ou_owner", user_name="owner", thread_id=thread_id,
            ),
            message_id=message_id,
            internal=internal,
        )

    def test_mode_on_stamps_before_session_key(self):
        adapter = self.make_adapter()
        adapter.set_topic_mode(FEISHU_CHAT, True, user_id="ou_owner")
        event = self._dm_event(message_id="om_q_early")
        adapter._maybe_stamp_topic_mode_thread(event)
        self.assertEqual(event.source.thread_id, "om_q_early")
        key = adapter._event_session_key(event)
        self.assertIn("om_q_early", key)
        self.assertNotEqual(key, build_session_key(_source()))

    def test_mode_off_does_not_stamp(self):
        adapter = self.make_adapter()
        event = self._dm_event(message_id="om_q_off")
        adapter._maybe_stamp_topic_mode_thread(event)
        self.assertIsNone(event.source.thread_id)

    def test_skips_existing_thread_commands_group_internal(self):
        adapter = self.make_adapter()
        adapter.set_topic_mode(FEISHU_CHAT, True, user_id="ou_owner")
        cases = (
            self._dm_event(message_id="om_a", thread_id="omt_existing"),
            self._dm_event(message_id="om_b", text="/status"),
            self._dm_event(message_id="om_c", chat_type="group"),
            self._dm_event(message_id="om_d", internal=True),
            self._dm_event(message_id=None),
        )
        for event in cases:
            before = event.source.thread_id
            adapter._maybe_stamp_topic_mode_thread(event)
            self.assertEqual(event.source.thread_id, before)

    def test_gateway_hook_idempotent_after_early_stamp(self):
        adapter = self.make_adapter()
        adapter.set_topic_mode(FEISHU_CHAT, True, user_id="ou_owner")
        source = _source()
        event = _event(source, message_id="om_q_idem")
        adapter._maybe_stamp_topic_mode_thread(event)
        self.assertEqual(event.source.thread_id, "om_q_idem")

        class _Runner(GatewayTopicThreadsMixin):
            def _adapter_for_source(self, src):
                return adapter

        out_event, out_source = asyncio.run(_Runner()._hm_maybe_auto_open_feishu_topic(event, event.source))
        self.assertEqual(out_source.thread_id, "om_q_idem")
        self.assertIs(out_event.source, event.source)

    def test_two_main_dm_questions_claim_distinct_active_sessions(self):
        """Regression: without early stamp both claim the launcher key and serialize."""
        adapter = feishu_adapter.FeishuAdapter(PlatformConfig(enabled=True, token="fake"))
        adapter._client = _FakeClient()
        adapter._topic_state_path = lambda: self.state_path
        adapter._topic_state_cache = None
        adapter._topic_state_lock = threading.RLock()
        adapter.set_topic_mode(FEISHU_CHAT, True, user_id="ou_owner")

        started = []
        release = asyncio.Event()

        async def handler(ev):
            started.append(ev.message_id)
            await release.wait()
            return "ok"

        adapter.set_message_handler(handler)

        async def drive():
            e1 = self._dm_event(message_id="om_q1", text="question one")
            e2 = self._dm_event(message_id="om_q2", text="question two")
            await adapter._handle_message_with_guards(e1)
            await adapter._handle_message_with_guards(e2)
            self.assertEqual(e1.source.thread_id, "om_q1")
            self.assertEqual(e2.source.thread_id, "om_q2")
            k1 = adapter._event_session_key(e1)
            k2 = adapter._event_session_key(e2)
            self.assertNotEqual(k1, k2)
            self.assertIn(k1, adapter._active_sessions)
            self.assertIn(k2, adapter._active_sessions)
            for _ in range(100):
                if len(started) >= 2:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(sorted(started), ["om_q1", "om_q2"])
            release.set()
            pending = [t for t in list(adapter._session_tasks.values()) if not t.done()]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run(drive())


class _MetadataCapturingAdapter(BasePlatformAdapter):
    """Minimal adapter that records the metadata of the final send."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.FEISHU)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content,
                          "reply_to": reply_to, "metadata": metadata})
        return SendResult(success=True, message_id="om_bot1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class TopicDeliveryMetadataTest(unittest.TestCase):
    """The answer's own reply must carry the topic key stamped mid-turn.

    The base adapter snapshots the routing metadata BEFORE the message handler runs, and the
    Feishu topic hook stamps ``source.thread_id`` while the handler runs. Using the stale
    snapshot sent the answer as a plain reply in the main DM (no topic ever opened, empty
    thread map) even though the session key was already the anchored one.
    """

    def test_answer_carries_mid_turn_thread_anchor(self):
        adapter = _MetadataCapturingAdapter()
        event = MessageEvent(
            text="你好呀",
            message_type=MessageType.TEXT,
            source=SessionSource(platform=Platform.FEISHU, chat_id=FEISHU_CHAT, chat_type="dm"),
            message_id="om_q_meta",
        )

        async def handler(ev):
            # Mirrors _hm_maybe_auto_open_feishu_topic: anchor = the question's message id.
            ev.source.thread_id = str(ev.message_id)
            return "主人，你好。"

        adapter.set_message_handler(handler)
        asyncio.run(adapter._process_message_background(event, build_session_key(event.source)))

        self.assertEqual(len(adapter.sent), 1)
        meta = adapter.sent[0]["metadata"] or {}
        self.assertEqual(meta.get("thread_id"), "om_q_meta",
                         "answer lost the topic key: it would land in the main DM")
        self.assertEqual(adapter.sent[0]["reply_to"], "om_q_meta")


if __name__ == "__main__":
    unittest.main()
