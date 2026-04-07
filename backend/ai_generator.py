import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use **get_course_outline** for questions about a course's structure, lesson list, or overview (e.g. "what lessons does X have?", "outline of X", "what topics are covered in X?")
- Use **search_course_content** for questions about specific content within a course
- You may make up to 2 sequential tool calls when a question requires gathering information from multiple sources before answering; prefer a single call when it provides sufficient information
- Synthesize results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use get_course_outline, then present the course title, link, and numbered lesson list clearly
- **Course-specific content questions**: Use search_course_content, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle up to 2 sequential rounds of tool execution.

        Each round: append assistant tool-use content, execute tools, append
        results, then offer tools again (unless the round budget is exhausted).
        A synthesis call with no tools is always made after the loop.

        Args:
            initial_response: The first response containing tool use requests
            base_params: Base API parameters (may include tools/tool_choice)
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        MAX_ROUNDS = 2
        available_tools = base_params.get("tools")
        tool_choice = base_params.get("tool_choice")
        system = base_params["system"]

        messages = base_params["messages"].copy()
        current_response = initial_response

        for round_number in range(1, MAX_ROUNDS + 1):
            # Append this round's assistant message
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls in this round
            tool_results = []
            failed = False
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(
                            content_block.name,
                            **content_block.input
                        )
                    except Exception as e:
                        result = f"Tool execution failed: {e}"
                        failed = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # On exception, skip remaining rounds and go straight to synthesis
            if failed:
                break

            # If more rounds remain, offer tools so Claude can chain calls
            if round_number < MAX_ROUNDS:
                intermediate_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": system,
                    "tools": available_tools,
                    "tool_choice": tool_choice,
                }
                intermediate_response = self.client.messages.create(**intermediate_params)

                # Early exit: Claude chose to answer rather than call another tool
                if intermediate_response.stop_reason != "tool_use":
                    return intermediate_response.content[0].text

                current_response = intermediate_response

        # Final synthesis call — no tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system,
        }
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text