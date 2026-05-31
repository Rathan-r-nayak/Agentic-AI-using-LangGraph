from contextlib import AsyncExitStack
from typing import List, Optional
from langchain_core.tools import tool, BaseTool
from mcp import ClientSession
from mcp.client.sse import sse_client
from Utils.Logger import get_logger


logger = get_logger("MAIN")


class ChatbotMCPClient:
    def __init__(self, sse_url: str):
        """
        Initialize the client with the FastAPI SSE endpoint.
        Example: 'http://127.0.0.1:8000/mcp/sse'
        """
        self.sse_url = sse_url
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect(self):
        """Establish the SSE connection to the FastAPI MCP server."""
        logger.info(f"🔄 Connecting to MCP Server at {self.sse_url}...")

        # Connect to the SSE endpoint exposed by FastMCP in FastAPI
        sse_transport = await self.exit_stack.enter_async_context(sse_client(self.sse_url))
        self.read, self.write = sse_transport

        self.session = await self.exit_stack.enter_async_context(ClientSession(self.read, self.write))
        #read:for listening to data coming from the server
        #write:for sending tool requests to the server

        await self.session.initialize()
        logger.info("✅ Successfully connected to MCP Server!")


    async def disconnect(self):
        """Cleanly shut down the connection."""
        await self.exit_stack.aclose()
        logger.info("🛑 Disconnected from MCP Server.")


    def get_tools(self) -> List[BaseTool]:
        """
        Creates LangChain @tool wrappers that proxy requests to the active MCP session.
        This provides perfect type-hinting for your LLM while executing remotely.
        """
        
        @tool
        async def get_ticket_details(ticket_id: int) -> str:
            """Fetch details and current status for a single support ticket."""
            result = await self.session.call_tool("get_ticket_details", {"ticket_id": ticket_id})
            return result.content[0].text

        @tool
        async def get_employee_tickets(employee_id: str) -> str:
            """Fetch all tickets submitted by a specific employee."""
            result = await self.session.call_tool("get_employee_tickets", {"employee_id": employee_id})
            return result.content[0].text

        @tool
        async def create_ticket(
            employee_id: str,
            specific_facility: str,
            catalog_category: str,
            catalog_item: str,
            ticket_description: str,
            request_sub_type: Optional[str] = None,
        ) -> str:
            """Create a new IT support ticket to escalate an unresolved issue to a human agent."""
            result = await self.session.call_tool("create_ticket", {
                "employee_id": employee_id,
                "specific_facility": specific_facility,
                "catalog_category": catalog_category,
                "catalog_item": catalog_item,
                "ticket_description": ticket_description,
                "request_sub_type": request_sub_type
            })
            return result.content[0].text

        @tool
        async def search_historical_resolutions(
            query: str, 
            catalog_category: Optional[str] = None, 
            catalog_item: Optional[str] = None
        ) -> str:
            """Search the historical vector database for past resolved tickets."""
            result = await self.session.call_tool("search_historical_resolutions", {
                "query": query,
                "catalog_category": catalog_category,
                "catalog_item": catalog_item
            })
            return result.content[0].text

        # Return the list of LangChain tools ready to be bound to your LLM
        return [
            get_ticket_details, 
            get_employee_tickets, 
            create_ticket, 
            search_historical_resolutions
        ]