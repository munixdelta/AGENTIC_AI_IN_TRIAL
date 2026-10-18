
# DAY 17: ITERATIVE TOOL CALLING WITH TOOL REGISTRY

# STEP 1: IMPORT LIBRARIES

import os
import json

from dotenv import load_dotenv
from groq import Groq


# STEP 2: LOAD API KEY FROM .env

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please check your .env file."
    )


# STEP 3: INITIALIZE GROQ CLIENT

groq_client = Groq(api_key=api_key)

GROQ_MODEL = "openai/gpt-oss-120b"


# STEP 4: DEFINE PYTHON FUNCTIONS

def calculate_percentage(number, percentage):
    return number * percentage / 100


def calculate_sum(a, b):
    return a + b


# STEP 5: CREATE TOOL REGISTRY

tool_registry = {
    "calculate_percentage": calculate_percentage,
    "calculate_sum": calculate_sum
}


# STEP 6: DEFINE TOOL SCHEMAS

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_percentage",
            "description": "Calculate a percentage of a number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "number"},
                    "percentage": {"type": "number"}
                },
                "required": ["number", "percentage"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sum",
            "description": "Calculate the sum of two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]


# STEP 7: GET USER QUERY

user_query = "Calculate 10% of 200, then add 5 to the result."

messages = [
    {
        "role": "user",
        "content": user_query
    }
]


# STEP 8: ITERATIVE TOOL CALLING

max_iterations = 10
iteration = 0

while iteration < max_iterations:

    iteration += 1

    print(f"\n--- Iteration {iteration} ---")

    # Ask Groq for the next response

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    print("Finish reason:", response.choices[0].finish_reason)

    # CASE 1: No tool calls; model has responded directly

    if not message.tool_calls:

        print("\nFinal Answer:", message.content)
        break

    # CASE 2: Model has requested one or more tools

    print("Tool calls:", message.tool_calls)

    # Add the assistant's tool-call message to history

    messages.append({
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments
                }
            }
            for call in message.tool_calls
        ]
    })

    # Execute every tool requested in this response

    for tool_call in message.tool_calls:

        tool_name = tool_call.function.name

        print("\nTool name:", tool_name)

        try:
            arguments = json.loads(
                tool_call.function.arguments
            )

            function_to_call = tool_registry.get(tool_name)

            if function_to_call is None:
                result = f"Unknown tool: {tool_name}"

            else:
                result = function_to_call(**arguments)

        except (json.JSONDecodeError, TypeError, KeyError) as error:
            result = f"Tool execution error: {error}"

        print("Tool result:", result)

        # Add the tool result to conversation history

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

else:
    # This runs if the loop reaches its iteration limit

    print(
        "\nStopped: maximum iterations reached. "
        "No final answer was confirmed."
    )
