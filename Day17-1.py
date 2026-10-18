
# DAY 17: MULTIPLE TOOL CALLING WITH TOOL REGISTRY

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
                    "number": {
                        "type": "number"
                    },
                    "percentage": {
                        "type": "number"
                    }
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
                    "a": {
                        "type": "number"
                    },
                    "b": {
                        "type": "number"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]


# STEP 7: GET USER QUERY

# user_query = "Calculate the sum of 25 and 35."

# Try this query too:
user_query = "Calculate 10% of 200."


# STEP 8: SEND QUERY AND TOOLS TO GROQ

messages = [
    {
        "role": "user",
        "content": user_query
    }
]

response = groq_client.chat.completions.create(
    model=GROQ_MODEL,
    messages=messages,
    tools=tools,
    tool_choice="auto"
)


# STEP 9: CHECK MODEL RESPONSE

message = response.choices[0].message

print("Finish reason:", response.choices[0].finish_reason)
print("Tool calls:", message.tool_calls)



# STEP 10: PROCESS ALL TOOL CALLS

if message.tool_calls:

    # Add the assistant's tool-call requests to history

    messages.append({
        "role": "assistant",
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

    # Execute each requested tool

    for tool_call in message.tool_calls:

        tool_name = tool_call.function.name 

        print("\nTool name:", tool_name)

        try:
            arguments = json.loads(
                tool_call.function.arguments
            )

            # Find the function using the tool registry
            function_to_call = tool_registry.get(tool_name)

            if function_to_call is None:
                result = f"Unknown tool: {tool_name}"

            else:
                result = function_to_call(**arguments)

        except (json.JSONDecodeError, TypeError, KeyError) as error:
            result = f"Tool execution error: {error}"

        print("Tool result:", result)

        # Add each tool result to conversation history

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })


    # STEP 11: SEND TOOL RESULTS BACK TO GROQ

    final_response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages
    )

    final_answer = final_response.choices[0].message.content

    print("\nFinal Answer:", final_answer)

else:
    # If the model answers directly without calling a tool
    print("\nFinal Answer:", message.content)
