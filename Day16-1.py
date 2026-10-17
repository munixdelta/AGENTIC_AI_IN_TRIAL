
# DAY 15: INTRODUCTION TO TOOL CALLING

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


# STEP 4: DEFINE TOOL SCHEMA

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
    }
]


# STEP 5: SEND TOOL SCHEMA TO GROQ

response = groq_client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
        {
            "role": "user",
            "content": "Calculate 10% of 200."
        }
    ],
    tools=tools,
    tool_choice="auto"
)


# STEP 6: CHECK THE MODEL'S RESPONSE

message = response.choices[0].message

print("Finish reason:", response.choices[0].finish_reason)
print("Tool calls:", message.tool_calls)
print("Model content:", repr(message.content))

print("Tool calls:", message.tool_calls)


# STEP 7: DEFINE THE ACTUAL PYTHON FUNCTION

def calculate_percentage(number, percentage):
    return number * percentage / 100



# STEP 8: PROCESS ALL TOOL CALLS

if message.tool_calls:

    # Start with the original user message
    messages = [
        {
            "role": "user",
            "content": "Calculate 10% of 200."
        }
    ]

    # Add ALL tool calls made by the assistant
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

    # Execute each tool call separately
    for tool_call in message.tool_calls:

        print("\nTool name:", tool_call.function.name)

        arguments = json.loads(
            tool_call.function.arguments
        )

        if tool_call.function.name == "calculate_percentage":

            result = calculate_percentage(
                arguments["number"],
                arguments["percentage"]
            )

        else:
            result = "Unknown tool"

        print("Calculation result:", result)

        # Add this tool's result to the conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

    # STEP 9: SEND ALL TOOL RESULTS BACK TO GROQ

    final_response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages
    )

    final_answer = final_response.choices[0].message.content

    print("\nFinal Answer:", final_answer)

else:
    print("No tool call requested.")
