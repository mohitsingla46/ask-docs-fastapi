from app.modules.agent.repository import AgentRepository
from app.modules.agent.schemas import ChatResponse, Chat
from app.modules.agent.agent import agent
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict, Any
from datetime import datetime
import groq
import logging

logger = logging.getLogger(__name__)

class AgentService:
    def __init__(self, agent_repository: AgentRepository):
        self.agent_repository = agent_repository

    async def handle_chat(self, user_id: str, thread_id: str, message: str) -> Chat:
        # Save the user message to database
        user_chat = await self.agent_repository.create_chat(
            user_id=user_id,
            thread_id=thread_id,
            message_type="user",
            message=message
        )

        # Get chat history from database to pass as context
        history = await self.agent_repository.get_chats_by_thread(user_id, thread_id)
        
        # Convert database history to LangChain messages
        message_history = []
        for chat in history[:-1]:  # Exclude the just-added user message
            if chat.type == "user":
                message_history.append(HumanMessage(content=chat.message))
            elif chat.type == "ai":
                message_history.append(AIMessage(content=chat.message))
        
        # Add the new user message
        message_history.append(HumanMessage(content=message))
        
        # Run the agent with the full conversation history
        initial_state = {
            "messages": message_history,
            "llm_calls": 0,
            "user_id": user_id
        }

        try:
            result = await agent.ainvoke(initial_state)

            # Get the last AI message as response
            ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                last_ai_message = ai_messages[-1]
                response_content = last_ai_message.content

                # Save the AI response
                ai_chat = await self.agent_repository.create_chat(
                    user_id=user_id,
                    thread_id=thread_id,
                    message_type="ai",
                    message=response_content
                )

                return ai_chat
                
        except groq.BadRequestError as e:
            # Handle Groq API errors (like malformed tool calls)
            logger.error(f"Groq API error: {str(e)}")
            error_message = "I apologize, but I encountered an error processing your request. This might be due to a long conversation. Try starting a new conversation or rephrase your question."
            
            # Save error response
            ai_chat = await self.agent_repository.create_chat(
                user_id=user_id,
                thread_id=thread_id,
                message_type="ai",
                message=error_message
            )
            return ai_chat
            
        except Exception as e:
            # Handle other errors
            logger.error(f"Unexpected error in agent: {str(e)}", exc_info=True)
            error_message = "I apologize, but I encountered an unexpected error. Please try again."
            
            # Save error response
            ai_chat = await self.agent_repository.create_chat(
                user_id=user_id,
                thread_id=thread_id,
                message_type="ai",
                message=error_message
            )
            return ai_chat

        # Fallback if no AI message found
        return user_chat
    
    async def get_chat_history(self, user_id: str, thread_id: str) -> List[Chat]:
        # Get chat history directly from database
        return await self.agent_repository.get_chats_by_thread(user_id, thread_id)
        
    async def delete_chat_history(self, user_id: str, thread_id: str) -> None:
        # Delete all chats for this thread from database
        await self.agent_repository.delete_chats_by_thread(user_id, thread_id)

