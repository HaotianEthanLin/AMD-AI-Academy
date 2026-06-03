from aup_config import aup_setup
aup_setup()

import asyncio
import sys
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import HumanMessage, ToolMessage
model = ChatOpenAI(model="qwen3.5:9b", base_url="", api_key="")

server_params = StdioServerParameters(command="python", args=["math_server.py"])

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            # Convert MCP tools to LangChain tools
            tools = await load_mcp_tools(session)
            print(f"Available tools:")
            for tool in tools:
                print(f'{tool}\n')

    async def run_react_agent(query: str):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                agent = create_agent(model, tools)
                agent_response = await agent.ainvoke({"messages": query})
                return agent_response
            
    query = "what is the square root of 55?"
    agent_response = await run_react_agent(query)
    for m in agent_response['messages']:
        m.pretty_print()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            model_with_tools = model.bind_tools(tools)

            # Inspect the tool definitions that are sent to the LLM with every API request
            bound_tools = model_with_tools.kwargs['tools']
            for t in bound_tools:
                func = t['function']
                print(f"Tool: {func['name']}")
                print(f"  Description: {func['description']}")
                print(f"  Parameters: {func['parameters']}\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            model_with_tools = model.bind_tools(tools)

            messages = [HumanMessage(content="what is the square root of 55?")]
            ai_response = await model_with_tools.ainvoke(messages)

            print("Content:", ai_response.content)
            print("\nTool calls requested by the LLM:")
            for tc in ai_response.tool_calls:
                print(f"  Function: {tc['name']}, Arguments: {tc['args']}")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            model_with_tools = model.bind_tools(tools)

            messages = [HumanMessage(content="what is the square root of 55?")]
            ai_response = await model_with_tools.ainvoke(messages)
            messages.append(ai_response)

            for tool_call in ai_response.tool_calls:
                result = await session.call_tool(tool_call['name'], arguments=tool_call['args'])
                print(f"Tool executed: {tool_call['name']}({tool_call['args']}) → {result}")
                # Append the tool result as a ToolMessage
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

            # Send the full conversation (including tool results) back to the LLM
            print(f"Message sent to the LLM:")
            for message in messages:
                print(f"{message}\n")
            final_response = await model_with_tools.ainvoke(messages)
            print(f"\nFinal LLM answer: {final_response.content}")
    
if __name__ == "__main__":
    asyncio.run(main())