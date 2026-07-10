from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import ToolCall

from cordon.agent import tool_info
from cordon.models.chat import hf_messages, hf_tools, parse_reply
from cordon.shop import TOOLS


def test_parse_reply_reads_calls_and_prose():
    text = (
        "I will look it up.\n<tool_call>\n"
        '{"name": "get_order", "arguments": {"order_id": "O-2001"}}\n</tool_call>'
    )
    prose, calls = parse_reply(text)
    assert prose == "I will look it up."
    assert [(c.function, c.arguments) for c in calls] == [("get_order", {"order_id": "O-2001"})]


def test_parse_reply_handles_several_calls_and_a_missing_close_tag():
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": "{\\"x\\": 1}"}'
    )
    _, calls = parse_reply(text)
    assert [(c.function, c.arguments) for c in calls] == [("a", {}), ("b", {"x": 1})]


def test_parse_reply_keeps_a_broken_call_with_its_error():
    _, calls = parse_reply("<tool_call>{not json}</tool_call>")
    assert len(calls) == 1 and calls[0].parse_error


def test_plain_text_has_no_calls():
    assert parse_reply("All done.") == ("All done.", [])


def test_messages_and_tools_convert_to_chat_template_form():
    call = ToolCall(id="c1", function="get_order", arguments={"order_id": "O-2001"})
    messages = [
        ChatMessageUser(content="hi"),
        ChatMessageAssistant(content="", tool_calls=[call]),
        ChatMessageTool(content='{"ok": true}', tool_call_id="c1", function="get_order"),
    ]
    rendered = hf_messages(messages)
    assert rendered[1]["tool_calls"][0]["function"] == {
        "name": "get_order",
        "arguments": {"order_id": "O-2001"},
    }
    assert rendered[2] == {"role": "tool", "content": '{"ok": true}', "name": "get_order"}

    (spec,) = hf_tools([tool_info(TOOLS["issue_refund"])])
    params = spec["function"]["parameters"]
    assert params["required"] == ["order_id", "amount"]
    assert params["properties"]["amount"]["type"] == "number"
