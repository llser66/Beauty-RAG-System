from agent.tools import create_tools


def test_skin_tool_does_not_accept_an_arbitrary_local_path():
    tools = {item.name: item for item in create_tools()}

    assert "image_path" not in tools["analyze_skin"].args
    assert tools["analyze_skin"].args == {}
