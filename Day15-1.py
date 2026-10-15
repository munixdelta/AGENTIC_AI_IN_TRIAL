
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

print("Tool calls:", message.tool_calls)


# STEP 7: DEFINE THE ACTUAL PYTHON FUNCTION

def calculate_percentage(number, percentage):
    return number * percentage / 100


# STEP 8: CHECK AND READ THE TOOL CALL

if message.tool_calls:
    tool_call = message.tool_calls[0]

    arguments = json.loads(tool_call.function.arguments)

    result = calculate_percentage(
        arguments["number"],
        arguments["percentage"]
    )

    print("Calculation result:", result)
