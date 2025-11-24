from app.modules.agent.repository import AgentRepository
from app.modules.agent.schemas import ChatResponse, Chat
from app.modules.agent.agent import agent
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Any
from datetime import datetime

class AgentService:
    def __init__(self, agent_repository: AgentRepository):
        self.agent_repository = agent_repository

    async def handle_chat(self, user_id: str, thread_id: str, message: str) -> Chat:
        # Prepare config for LangGraph - scope by both user_id and thread_id
        config = {
            "configurable": {
                "thread_id": f"{user_id}_{thread_id}"
            }
        }

        # Save the user message to database
        user_chat = await self.agent_repository.create_chat(
            user_id=user_id,
            thread_id=thread_id,
            message_type="user_message",
            message=message
        )

        # Run the agent with just the new message
        # LangGraph will automatically load history from checkpointer
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "llm_calls": 0,
            "user_id": user_id
        }

        result = await agent.ainvoke(initial_state, config)

        # Get the last AI message as response
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            last_ai_message = ai_messages[-1]
            response_content = last_ai_message.content

            # Save the AI response
            ai_chat = await self.agent_repository.create_chat(
                user_id=user_id,
                thread_id=thread_id,
                message_type="ai_message",
                message=response_content
            )

            return ai_chat

        # Fallback if no AI message found
        return user_chat

