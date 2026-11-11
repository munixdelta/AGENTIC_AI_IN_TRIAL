
# DAY 19-2: MULTIPLE TOOL CALLING
# ERROR HANDLING, RETRIES, OPERATION ID AND IDEMPOTENCY


# STEP 1: IMPORT LIBRARIES

import json
import os

from pathlib import Path
from dotenv import load_dotenv
from groq import Groq, APIError, APIConnectionError, RateLimitError


# STEP 2: LOAD API KEY FROM .env

env_path = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        f"GROQ_API_KEY missing or empty. Check: {env_path}"
    )


# STEP 3: INITIALIZE GROQ CLIENT

groq_client = Groq(api_key=api_key)

GROQ_MODEL = "openai/gpt-oss-120b"


# STEP 4: DEFINE PYTHON FUNCTIONS

# Simulate one temporary failure.
failure_simulated = False


def calculate_percentage(number, percentage):

    global failure_simulated

    if not failure_simulated:

        failure_simulated = True

        raise ValueError(
            "Temporary tool execution failure. Please retry."
        )

    return number * percentage / 100


def calculate_sum(a, b):

    return a + b


# STEP 5: CREATE TOOL REGISTRY

tool_registry = {
    "calculate_percentage": calculate_percentage,
    "calculate_sum": calculate_sum,
}


# STEP 6: DEFINE TOOL SCHEMAS

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_percentage",
            "description": (
                "Calculate a percentage of a number. "
                "Generate a unique operation_id for each new logical "
                "calculation. Reuse the same ID when retrying it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation_id": {
                        "type": "string",
                        "description": (
                            "Unique ID for this logical operation. "
                            "Reuse this ID only when retrying the "
                            "same operation."
                        ),
                    },
                    "number": {"type": "number"},
                    "percentage": {"type": "number"},
                },
                "required": [
                    "operation_id",
                    "number",
                    "percentage",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sum",
            "description": (
                "Calculate the sum of two numbers. "
                "Generate a unique operation_id for each new logical "
                "calculation. Reuse the same ID when retrying it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation_id": {
                        "type": "string",
                        "description": (
                            "Unique ID for this logical operation. "
                            "Reuse this ID only when retrying the "
                            "same operation."
                        ),
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": [
                    "operation_id",
                    "a",
                    "b",
                ],
            },
        },
    },
]


# STEP 7: GET USER QUERY

user_query = (
    "Calculate 10% of 200 using calculate_percentage. "
    "Calculate 10% of 200 again using same operation id"
    "Calculate 50% of 100"
    "Add the three results using calculate_sum. "
    "Perform all calculations using the provided tools."
)


messages = [
    {
        "role": "system",
        "content": (
            "You are a calculator assistant. "
            "You MUST use Python tools for all calculations. "

            "IMPORTANT OPERATION ID RULES: "

            "1. Generate a unique operation_id for every new "
            "logical tool operation. "

            "2. If two operations are requested separately, "
            "they MUST have different operation_ids, even if "
            "their tool names and arguments are identical. "

            "3. If a tool execution fails, retry the SAME logical "
            "operation using its original operation_id. "

            "4. Never generate a new operation_id for a retry. "

            "5. If an operation already succeeded, do not execute "
            "it again. "

            "6. Do not reuse an operation_id for a different task. "

            "When multiple independent operations are requested, "
            "request their corresponding tool calls in the same "
            "response if possible. "

            "If a tool returns an execution error, understand "
            "the error and retry with corrected arguments."
        ),
    },
    {
        "role": "user",
        "content": user_query,
    },
]


# STEP 8: ITERATIVE TOOL CALLING

max_iterations = 10
max_tool_retries = 3

iteration = 0

# Maintain separate retry counts for each tool.
tool_retry_counts = {}

# Store completed operations using their operation IDs.
# Each entry contains the tool name, arguments fingerprint and result.
idempotency_store = {}

final_answer_received = False
stop_execution = False
stop_reason = None


while iteration < max_iterations:

    if stop_execution:
        break

    iteration += 1

    print(f"\n--- Iteration {iteration} ---")


    # API-LEVEL ERROR HANDLING

    try:

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True,
            temperature=0,
        )

    except RateLimitError as error:

        print("\nAPI rate limit exceeded:", error)

        stop_execution = True
        stop_reason = "api_error"

        break

    except APIConnectionError as error:

        print("\nAPI connection failed:", error)

        stop_execution = True
        stop_reason = "api_error"

        break

    except APIError as error:

        print("\nGroq API request failed:", error)

        stop_execution = True
        stop_reason = "api_error"

        break

    except Exception as error:

        print("\nUnexpected API error:", error)

        stop_execution = True
        stop_reason = "api_error"

        break


    message = response.choices[0].message

    print(
        "Finish reason:",
        response.choices[0].finish_reason
    )


    # CASE 1: NO TOOL CALLS

    if not message.tool_calls:

        final_answer = message.content

        print("\nFinal Answer:", final_answer)

        final_answer_received = True

        break


    # CASE 2: MODEL REQUESTED TOOL CALLS

    print(
        "\nNumber of tool calls:",
        len(message.tool_calls)
    )


    # ADD ASSISTANT TOOL-CALL MESSAGE TO HISTORY

    messages.append({
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ],
    })


    # EXECUTE EVERY REQUESTED TOOL

    for tool_call in message.tool_calls:

        tool_name = tool_call.function.name

        print("\nTool name:", tool_name)

        # Initialize retry counter for this tool.
        tool_retry_counts.setdefault(tool_name, 0)

        operation_id = None

        try:

            # PARSE TOOL ARGUMENTS

            arguments = json.loads(
                tool_call.function.arguments
            )


            # EXTRACT OPERATION ID

            operation_id = arguments.pop("operation_id", None)

            if not isinstance(operation_id, str) or not operation_id.strip():

                raise ValueError(
                    "Missing or invalid operation_id."
                )

            operation_id = operation_id.strip()

            print("Operation ID:", operation_id)


            # FIND THE CORRESPONDING PYTHON FUNCTION

            function_to_call = tool_registry.get(tool_name)

            if function_to_call is None:

                raise ValueError(
                    f"Unknown tool: {tool_name}"
                )


            # CREATE ARGUMENT FINGERPRINT

            arguments_fingerprint = json.dumps(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                },
                sort_keys=True,
                separators=(",", ":"),
            )


            # CHECK WHETHER THIS OPERATION ID ALREADY EXISTS

            if operation_id in idempotency_store:

                previous_operation = idempotency_store[operation_id]


                # SAME ID BUT DIFFERENT OPERATION

                if (
                    previous_operation["fingerprint"]
                    != arguments_fingerprint
                ):

                    raise ValueError(
                        "Operation ID already belongs to a "
                        "different tool operation."
                    )


                # SAME ID AND SAME OPERATION

                print(
                    "Duplicate operation detected. "
                    "Using previously stored result."
                )

                result = previous_operation["result"]

                tool_retry_counts[tool_name] = 0

                tool_response = {
                    "status": "success",
                    "operation_id": operation_id,
                    "result": result,
                    "duplicate": True,
                }


            else:

                # EXECUTE A NEW OPERATION

                print("Executing new operation...")

                result = function_to_call(**arguments)

                print("Tool result:", result)


                # STORE ONLY SUCCESSFUL OPERATIONS

                idempotency_store[operation_id] = {
                    "tool_name": tool_name,
                    "fingerprint": arguments_fingerprint,
                    "result": result,
                }


                # RESET RETRY COUNTER AFTER SUCCESS

                tool_retry_counts[tool_name] = 0

                tool_response = {
                    "status": "success",
                    "operation_id": operation_id,
                    "result": result,
                    "duplicate": False,
                }


        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
        ) as error:

            # HANDLE TOOL EXECUTION FAILURE

            tool_retry_counts[tool_name] += 1

            retry_count = tool_retry_counts[tool_name]

            print("Tool execution failed:", error)

            print(
                f"Retry attempts used: "
                f"{retry_count}/{max_tool_retries}"
            )

            tool_response = {
                "status": "error",
                "operation_id": operation_id,
                "error": str(error),
                "retry_count": retry_count,
                "instruction": (
                    "Correct the error and retry the same logical "
                    "operation. If an operation_id was provided, "
                    "reuse exactly that operation_id. "
                    "Do not create a new ID for this retry."
                ),
            }


        # SEND RESULT OR ERROR BACK TO THE MODEL

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(tool_response),
        })


        # STOP AFTER MAXIMUM FAILED ATTEMPTS FOR THIS TOOL

        if tool_retry_counts[tool_name] >= max_tool_retries:

            print(
                f"\nMaximum retry limit reached for {tool_name}."
            )

            stop_execution = True
            stop_reason = "tool_retry_limit"

            break


# STEP 9: HANDLE MAXIMUM ITERATIONS OR ERRORS

if not final_answer_received:

    if stop_reason == "api_error":

        print(
            "\nAgent stopped because an API-level error occurred."
        )

    elif stop_reason == "tool_retry_limit":

        print(
            "\nAgent stopped because maximum tool retries were reached."
        )

    else:

        print("\nMaximum iterations reached.")

        print("Requesting final response...")

        try:

            final_response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tool_choice="none",
            )

            final_answer = final_response.choices[0].message.content

            if final_answer:

                print("\nFinal Answer:", final_answer)

            else:

                print("\nNo final answer was returned.")

        except APIError as error:

            print("\nFinal API request failed:", error)


# STEP 10: DISPLAY COMPLETED OPERATION SUMMARY

print("\n--- OPERATION SUMMARY ---")

for operation_id, operation_data in idempotency_store.items():

    print(
        f"Operation ID: {operation_id} | "
        f"Tool: {operation_data['tool_name']} | "
        f"Result: {operation_data['result']}"
    )
