import json

from cordon.shop import TOOLS, World, call_tool, map_untrusted, seed_world, untrusted_strings


def test_snapshot_round_trips():
    world = seed_world()
    call_tool(world, "issue_refund", {"order_id": "O-2002", "amount": 10})
    snapshot = json.loads(json.dumps(world.snapshot()))
    assert World.from_snapshot(snapshot).snapshot() == world.snapshot()


def test_order_total_is_sum_of_lines():
    assert seed_world().orders["O-2004"].total == 128.00


def test_read_tool_returns_data_and_records_the_call():
    world = seed_world()
    result, ok = call_tool(world, "get_order", {"order_id": "O-2001"}, agent="support")
    assert ok and result["tracking"] == "TRK-48213"
    assert [(c.agent, c.tool, c.ok) for c in world.calls] == [("support", "get_order", True)]


def test_errors_are_returned_not_raised():
    world = seed_world()
    for name, args in [
        ("get_order", {"order_id": "O-0000"}),
        ("get_order", {}),
        ("get_order", {"order_id": "O-2001", "extra": 1}),
        ("no_such_tool", {}),
        ("issue_refund", {"order_id": "O-2002", "amount": "lots"}),
    ]:
        result, ok = call_tool(world, name, args)
        assert not ok and "error" in result
    assert all(not c.ok for c in world.calls)


def test_numbers_passed_as_strings_are_accepted():
    world = seed_world()
    _, ok = call_tool(world, "issue_refund", {"order_id": "O-2002", "amount": "129.00"})
    assert ok and world.refunds[0].amount == 129.00


def test_refunds_cannot_exceed_the_order_total():
    world = seed_world()
    assert call_tool(world, "issue_refund", {"order_id": "O-2002", "amount": 100})[1]
    assert not call_tool(world, "issue_refund", {"order_id": "O-2002", "amount": 100})[1]
    assert len(world.refunds) == 1


def test_address_is_fixed_once_an_order_ships():
    world = seed_world()
    assert not call_tool(world, "update_shipping_address", {"order_id": "O-2001", "address": "x"})[
        1
    ]
    assert call_tool(world, "update_shipping_address", {"order_id": "O-2003", "address": "x"})[1]


def test_every_tool_that_changes_money_or_sends_data_is_high_risk():
    high = {name for name, tool in TOOLS.items() if tool.high_risk}
    assert high == {
        "issue_refund",
        "send_email",
        "update_shipping_address",
        "create_discount_code",
        "update_product",
    }
    assert all(TOOLS[name].writes for name in high)


def test_untrusted_keys_name_keys_the_tool_returns():
    world = seed_world()
    samples = {
        "get_order": {"order_id": "O-2004"},
        "get_product": {"sku": "SKU-PACK"},
        "list_reviews": {"sku": "SKU-PACK"},
        "get_ticket": {"ticket_id": "T-1001"},
        "list_open_tickets": {},
        "list_inbox": {},
        "get_email": {"email_id": "E-301"},
    }
    for name, tool in TOOLS.items():
        if not tool.untrusted_keys:
            continue
        result, _ = call_tool(world, name, samples[name])
        rows = result if isinstance(result, list) else [result]
        for key in tool.untrusted_keys:
            assert all(key in row for row in rows), (name, key)


def test_map_untrusted_changes_only_the_named_keys():
    data = {"id": "T-1", "body": "hello", "replies": ["hello"], "nested": [{"body": "x"}]}
    marked = map_untrusted(data, ("body",), str.upper)
    assert marked == {"id": "T-1", "body": "HELLO", "replies": ["hello"], "nested": [{"body": "X"}]}
    assert untrusted_strings(data, ("body",)) == ["hello", "x"]
