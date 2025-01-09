import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from fastapi import Request
from openai import AuthenticationError as OpenAIAuthError
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import convert_to_anthropic_tool
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools.render import format_tool_to_openai_function

from app.agent_services.tools import (
    get_invoices_details, # <COMPARTE INFORMACION FACTURAS>
    get_payments_details, # <COMPARTE INFORMACION PAGOS>
    get_closest_location, # <LOCALIZA UBICACION>
    get_client_id, # <VALIDA CLIENTE>
    validate_address, # <VALIDA UBICACIÓN> 
    create_booking, # <AGENDA CITA> 
    get_available_slots, # <EXTRAE SLOTS>
    calculate_rfc, #<CALCULA RFC>
    calculate_curp, #<CALCULA CURP>
    get_payment_links_with_products ## STRIPE TOOL
)
from app.utils.db_wrapper import (
    Database,
    EndUser,
    EndUserCreate,
    ConversationCreate,
    ConversationStatus,
    MessageCreate,
    MessagePayload
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType:
    GPT4 = "gpt-4"
    CLAUDE = "claude"
    GPT35 = "gpt-3.5"

class AgentService:
    def __init__(self, agent_id: UUID, external_user_id: Optional[str] = None, 
                 source: Optional[str] = 'web'):
        self.agent_id = agent_id
        self.external_user_id = external_user_id
        self.source = source
        self.agent = None
        self.conversation_id = None
        self.db = None
        self.end_user = None
        self.contextual_info = {}
        self.current_model = ModelType.GPT4

    def create_model(self, request: Request, model_type: str, tools=None):
        """Create language model based on type"""
        if model_type == ModelType.GPT4:
            model = ChatOpenAI(
                temperature=0,
                model="gpt-4o",
                openai_api_key=request.app.state.OPENAI_API_KEY
            )
            if tools:
                functions = [format_tool_to_openai_function(tool) for tool in tools]
                return model.bind(functions=functions)
        elif model_type == ModelType.CLAUDE:
            model = ChatAnthropic(
                model="claude-3-5-sonnet-latest",
                anthropic_api_key=request.app.state.ANTHROPIC_API_KEY,
                temperature=0
            )
            if tools:
                anthropic_tools = [convert_to_anthropic_tool(tool) for tool in tools]
                return model.bind_tools(anthropic_tools)
        elif model_type == ModelType.GPT35:
            model = ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo",
                openai_api_key=request.app.state.OPENAI_API_KEY
            )
            if tools:
                functions = [format_tool_to_openai_function(tool) for tool in tools]
                return model.bind(functions=functions)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return model
        
    async def get_or_create_user(self, phone_number: Optional[str] = None) -> Optional[EndUser]:
        """
        Get existing user or create a new one
        
        Args:
            phone_number: Optional phone number for lookup
            
        Returns:
            Optional[EndUser]: User object if found or created
        """
        try:
            if not phone_number and not self.external_user_id:
                return None

            if self.external_user_id:
                user = await self.db.get_end_user(self.external_user_id)
                if user:
                    return user

            metadata = {}
            if phone_number:
                metadata["phone_number"] = phone_number
                metadata["created_from"] = f"{self.source}_agent"

            user_data = EndUserCreate(
                external_user_id=self.external_user_id or str(phone_number),
                external_profile_name=None,
                source=self.source,
                user_metadata=metadata
            )
            return await self.db.create_end_user(user_data)

        except Exception as e:
            logger.error(f"Error in get_or_create_user: {str(e)}")
            return None
        
    async def initialize_agent(self, request: Request, phone_number: Optional[str] = None, 
                             model_type: Optional[str] = None):
        """Initialize agent with specified model type"""
        try:
            logger.info(f"Initializing agent with model: {model_type or self.current_model}")
            
            # Only initialize DB and user if first time
            if self.db is None:
                self.db = request.app.state.db
                self.end_user = await self.get_or_create_user(phone_number)
                if self.end_user:
                    self.external_user_id = self.end_user.external_user_id
                    logger.info(f"Using user ID: {self.end_user.id}")

                conversation_data = ConversationCreate(
                    external_user_id=self.external_user_id,
                    source=self.source,
                    agent_id=self.agent_id,
                    end_user_id=self.end_user.id if self.end_user else None
                )
                
                conversation = await self.db.create_conversation(conversation_data)
                self.conversation_id = conversation.id

            if model_type:
                self.current_model = model_type
            
            prompt_hub = hub.pull("pinsheinakipapa/semantiks-conversion")
            initial_profiling_message = self._prepare_initial_message(prompt_hub.template)

            tools = [
                get_invoices_details, 
                get_payments_details, 
                get_closest_location, 
                get_client_id,
                validate_address,
                create_booking,
                get_available_slots,
                calculate_rfc,
                calculate_curp,
                get_payment_links_with_products
            ]
            
            model = self.create_model(request, self.current_model, tools)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", initial_profiling_message),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            chain = RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
            ) | prompt | model | OpenAIFunctionsAgentOutputParser()
            
            # Create memory with fallback handling
            try:
                memory_model = ChatOpenAI(
                    temperature=0,
                    model="gpt-4-turbo",
                    openai_api_key=request.app.state.OPENAI_API_KEY
                )
            except OpenAIAuthError:
                logger.warning("GPT-4-turbo auth failed for memory, trying Claude")
                try:
                    memory_model = ChatAnthropic(
                        model="claude-3-5-sonnet-latest",
                        anthropic_api_key=request.app.state.ANTHROPIC_API_KEY,
                        temperature=0
                    )
                except Exception as claude_error:
                    logger.warning(f"Claude failed for memory, using GPT-3.5: {str(claude_error)}")
                    try:
                        memory_model = ChatOpenAI(
                            temperature=0,
                            model="gpt-3.5-turbo",
                            openai_api_key=request.app.state.OPENAI_API_KEY
                        )
                    except OpenAIAuthError:
                        logger.error("All models failed for memory")
                        raise
            
            memory = ConversationSummaryBufferMemory(
                llm=memory_model,
                max_tokens=650,
                return_messages=True,
                memory_key="chat_history",
            )
            
            self.agent = AgentExecutor(
                agent=chain,
                memory=memory,
                tools=tools,
                verbose=True,
                tags=self._create_agent_tags(request)
            )

            if not hasattr(request.app.state, 'agents'):
                request.app.state.agents = {}
            request.app.state.agents[self.conversation_id] = self
            
            logger.debug(f"Agent initialized successfully with model {self.current_model}")
            
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            raise
    
    async def process_message(
                self, 
                user_message: str, 
                request: Request, 
                message_metadata: Optional[Dict] = None,
                message_type: str = "text",
                trace_id = ""
            ) -> str:
            try:
                if self.agent is None:
                    try:
                        await self.initialize_agent(request)
                    except OpenAIAuthError:
                        logger.warning("OpenAI auth failed during initialization, trying Claude")
                        await self.initialize_agent(request, model_type=ModelType.CLAUDE)
                
                current_model = self.current_model
                while True:  # Keep trying models until one works or we run out of options
                    try:
                        response = await self.agent.ainvoke({"input": user_message}, {"run_id": trace_id})
                        output = response["output"]
                        
                        # Handle Anthropic response format
                        if current_model == ModelType.CLAUDE and isinstance(output, list):
                            for item in output:
                                if isinstance(item, dict) and 'text' in item:
                                    output = item['text']
                                    break
                        
                        logging.info(f"Agent response with {current_model}: {output}")
                        return output
                        
                    except OpenAIAuthError:
                        logger.warning(f"Authentication error with {current_model}")
                        if current_model == ModelType.GPT4:
                            logger.info("Switching to Claude")
                            await self.initialize_agent(request, model_type=ModelType.CLAUDE)
                            current_model = ModelType.CLAUDE
                        elif current_model == ModelType.GPT35:
                            logger.error("All OpenAI models failed authentication")
                            raise
                        
                    except Exception as e:
                        if current_model == ModelType.CLAUDE:
                            logger.warning(f"Claude failed, trying GPT-3.5: {str(e)}")
                            await self.initialize_agent(request, model_type=ModelType.GPT35)
                            current_model = ModelType.GPT35
                        else:
                            raise
                    
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                raise

    def _prepare_initial_message(self, template):
        """Prepare the initial message with contextual information"""
        if self.end_user:
            user_data = self.end_user.to_dict()
            self.contextual_info = {
                'user_id': user_data['id'],
                'user_profile_name': user_data['external_profile_name'],
                'conversation_channel': user_data['source'],
                'message_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'conversation_id': self.agent_id
            }
            
            contextual_info_str = "\n".join(
                [f"{key} = {value}" for key, value in self.contextual_info.items()]
            )
            return template.format(contextual_info=contextual_info_str)
        return template

    def _create_agent_tags(self, request):
        """Create tags for the agent"""
        user_data = self.end_user.to_dict()
        tags = [
            f"\"agent_type\":\"onboarding\"",
            f"\"external_user_id\":\"{self.external_user_id}\"",
            f"\"conversation_id\":\"{str(self.agent_id)}\"",
            f"\"channel\":\"{self.source}\"",
            f"\"organization_id\":\"{ request.app.state.COMPANY_ID}\"",
            f"\"user_name\":\"{ user_data['external_profile_name']}\""
        ]
        
        if self.end_user:
            tags.append(f"\"user_id\":\"{str(self.end_user.id)}\"")
        
        return tags

    @staticmethod
    async def cleanup_inactive_agents(request: Request) -> List[UUID]:
        """Cleanup inactive agents but preserve conversation history."""
        try:
            db = request.app.state.db
            inactive_ids = await db.get_inactive_conversation_ids(hours=1)
            
            for conv_id in inactive_ids:
                await db.end_conversation(conv_id)
            
            if hasattr(request.app.state, 'agents'):
                for conv_id in inactive_ids:
                    if conv_id in request.app.state.agents:
                        del request.app.state.agents[conv_id]
            
            logger.info(f"Cleaned up {len(inactive_ids)} inactive agents")
            return inactive_ids
            
        except Exception as e:
            logger.error(f"Error cleaning up inactive agents: {str(e)}")
            raise

    async def end_conversation(self, request: Request):
        """End the current conversation and cleanup the agent"""
        try:
            if self.db and self.conversation_id:
                await self.db.end_conversation(self.conversation_id)
                
                if hasattr(request.app.state, 'agents') and self.conversation_id in request.app.state.agents:
                    del request.app.state.agents[self.conversation_id]
                    
                logger.info(f"Ended conversation {self.conversation_id}")
                
        except Exception as e:
            logger.error(f"Error ending conversation: {str(e)}")
            raise
