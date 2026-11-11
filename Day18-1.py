
# DAY 17: ITERATIVE TOOL CALLING WITH TOOL REGISTRY

# STEP 1: IMPORT LIBRARIES

import os
import json

from dotenv import load_dotenv
from groq import Groq


# STEP 2: LOAD API KEY FROM .env

from pathlib import Path

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        f"GROQ_API_KEY missing or empty. Check: {env_path}"
    )

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



# STEP 7: GET USER QUERY AND DEFINE EXPECTED OPERATIONS

user_query = (
    "Calculate 10% of 200, then add 5 to the result, "
    "then calculate 10% of the final result, "
    "then add 99 to that result."
)

# This plan is specific to the current test query.
# Later, we will learn how to generate plans dynamically.

pending_operations = [
    {
        "tool": "calculate_percentage",
        "arguments": {"number": 200, "percentage": 10}
    },
    {
        "tool": "calculate_sum",
        "arguments": {"a": 20, "b": 5}
    },
    {
        "tool": "calculate_percentage",
        "arguments": {"number": 25, "percentage": 10}
    },
    {
        "tool": "calculate_sum",
        "arguments": {"a": 2.5, "b": 99}
    }
]

messages = [
    {
        "role": "system",
        "content": (
            "You are a tool-using calculator assistant. "
            "Use the provided Python tools for calculations. "
            "Perform only the next pending operation requested "
            "by the user or application. Make one tool call "
            "at a time. Do not skip operations."
        )
    },
    {
        "role": "user",
        "content": user_query
    }
]


# STEP 8: ITERATIVE TOOL CALLING WITH VALIDATION

max_iterations = 10
iteration = 0
final_answer_received = False

while iteration < max_iterations:

    # Stop only when all planned operations are completed.
    if not pending_operations:
        break

    iteration += 1

    expected = pending_operations[0]

    print(f"\n--- Iteration {iteration} ---")
    print("Next pending operation:", expected)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    print("Finish reason:", response.choices[0].finish_reason)

    # CASE 1: Model tries to answer before completing the plan.

    if not message.tool_calls:

        print("Model skipped the next tool call.")

        messages.append({
            "role": "assistant",
            "content": message.content or ""
        })

        messages.append({
            "role": "user",
            "content": (
                "The next operation is still pending. "
                f"Call the tool '{expected['tool']}' with "
                f"these arguments: {json.dumps(expected['arguments'])}. "
                "Do not provide the final answer yet."
            )
        })

        continue

    # Keep the original assistant tool-call message in history.

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

    # Process each requested tool call.

    for tool_call in message.tool_calls:

        tool_name = tool_call.function.name
        result = None

        try:
            arguments = json.loads(
                tool_call.function.arguments
            )

            # Validate the call against the next expected operation.

            if not pending_operations:
                result = "Error: No operations remain pending."

            else:
                expected = pending_operations[0]

                if (
                    tool_name != expected["tool"]
                    or arguments != expected["arguments"]
                ):
                    result = (
                        "Validation error: This is not the next "
                        "expected operation. Try the next pending "
                        "operation using the exact tool and arguments."
                    )

                else:
                    function_to_call = tool_registry.get(tool_name)

                    if function_to_call is None:
                        result = f"Unknown tool: {tool_name}"

                    else:
                        result = function_to_call(**arguments)

                        # Remove the operation only after successful
                        # execution and validation.
                        pending_operations.pop(0)

        except (json.JSONDecodeError, TypeError, KeyError) as error:
            result = f"Tool execution error: {error}"

        print("Tool name:", tool_name)
        print("Tool result:", result)
        print("Operations remaining:", len(pending_operations))

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

        # Do not allow additional calls to execute against an
        # outdated expected step. The model is asked for one call
        # at a time, but we still validate every requested call.


# STEP 9: REQUEST FINAL ANSWER ONLY AFTER PLAN COMPLETION

if not pending_operations:

    print("\nAll planned operations completed!")

    final_response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages + [
            {
                "role": "user",
                "content": (
                    "All planned operations are complete. "
                    "Give the final answer using the tool results "
                    "in the conversation. Do not perform new calculations."
                )
            }
        ],
        tool_choice="none"
    )

    final_answer = final_response.choices[0].message.content

    if final_answer:
        print("\nFinal Answer:", final_answer)
        final_answer_received = True

    else:
        print("\nNo final answer was returned by the model.")

else:
    print("\nMaximum iterations reached.")
    print("Pending operations:", pending_operations)
    print("The task is incomplete; do not report success.")
