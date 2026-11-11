
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq, APIError, APIConnectionError, RateLimitError
from concurrent.futures import ThreadPoolExecutor, as_completed


# STEP 1: LOAD API KEY

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY missing. Check your .env file.")


# STEP 2: INITIALIZE GROQ CLIENT

groq_client = Groq(api_key=api_key)
GROQ_MODEL = "openai/gpt-oss-120b"


# STEP 3: DEFINE PYTHON FUNCTIONS

def calculate_percentage(number, percentage):

    print(f"Starting: {percentage}% of {number}")
    time.sleep(2)

    result = number * percentage / 100

    print(f"Completed percentage: {result}")
    return result


def calculate_sum(a, b):

    print(f"Starting sum: {a} + {b}")
    time.sleep(2)

    result = a + b

    print(f"Completed sum: {result}")
    return result


# STEP 4: TOOL REGISTRY

tool_registry = {
    "calculate_percentage": calculate_percentage,
    "calculate_sum": calculate_sum
}


# STEP 5: TOOL SCHEMAS

tools = [

    {
        "type": "function",
        "function": {
            "name": "calculate_percentage",
            "description": "Calculate percentage of a number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation_id": {"type": "string"},
                    "number": {"type": "number"},
                    "percentage": {"type": "number"}
                },
                "required": [
                    "operation_id",
                    "number",
                    "percentage"
                ]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "calculate_sum",
            "description": "Calculate sum of two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation_id": {"type": "string"},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": [
                    "operation_id",
                    "a",
                    "b"
                ]
            }
        }
    }

]


# STEP 6: USER QUERY

user_query = (
    "Perform these three independent calculations: "
    "1. Calculate 10% of 200. "
    "2. Calculate 20% of 500. "
    "3. Calculate 30% of 100. "
    "Use the provided Python tools. "
    "After all results are received, explain the calculations."
)


# STEP 7: CONVERSATION HISTORY

messages = [

    {
        "role": "system",
        "content": (
            "You are a calculator assistant. "
            "You MUST use Python tools for calculations. "

            "Rules: "

            "1. Use a unique operation_id for every "
            "new logical calculation. "

            "2. Use different IDs for independent calculations. "

            "3. Reuse the same operation_id when retrying "
            "a failed operation. "

            "4. Request independent tool calls in the same "
            "response whenever possible. "

            "5. Never assume a tool succeeded without its result. "

            "6. Retry failed operations only when instructed "
            "by the tool result. "

            "7. If an operation reaches its retry limit, "
            "do not retry it again. "

            "8. Continue other independent operations even "
            "if one operation fails. "

            "9. Do not execute dependent calculations until "
            "their required results are available. "

            "10. Clearly report operations that permanently failed."
        )
    },

    {
        "role": "user",
        "content": user_query
    }

]


# STEP 8: TOOL EXECUTION HELPER

def execute_tool_call(tool_call):

    operation_id = None
    tool_name = None

    try:

        tool_name = tool_call.function.name

        arguments = json.loads(
            tool_call.function.arguments
        )

        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")

        operation_id = arguments.pop("operation_id", None)

        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("Missing or invalid operation_id.")

        operation_id = operation_id.strip()

        function = tool_registry.get(tool_name)

        if function is None:
            raise ValueError(f"Unknown tool: {tool_name}")

        print(
            f"\nExecuting {tool_name} | "
            f"Operation ID: {operation_id}"
        )

        result = function(**arguments)

        return {
            "status": "success",
            "operation_id": operation_id,
            "tool_name": tool_name,
            "result": result
        }

    except Exception as error:

        return {
            "status": "error",
            "operation_id": operation_id,
            "tool_name": tool_name,
            "error": str(error)
        }


# STEP 9: EXECUTION SETTINGS

max_iterations = 10
max_tool_retries = 3
max_workers = 5

iteration = 0
retry_counts = {}

final_answer_received = False
stop_execution = False


# STEP 10: MAIN AGENT LOOP

while iteration < max_iterations:

    if stop_execution:
        break

    iteration += 1

    print(f"\n========== ITERATION {iteration} ==========")

    # GET RESPONSE FROM GROQ

    try:

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True,
            temperature=0
        )

    except RateLimitError as error:

        print("\nRate limit error:", error)
        stop_execution = True
        break

    except APIConnectionError as error:

        print("\nConnection error:", error)
        stop_execution = True
        break

    except APIError as error:

        print("\nGroq API error:", error)
        stop_execution = True
        break

    except Exception as error:

        print("\nUnexpected API error:", error)
        stop_execution = True
        break


    message = response.choices[0].message

    print(
        "Finish reason:",
        response.choices[0].finish_reason
    )


    # CASE 1: FINAL ANSWER

    if not message.tool_calls:

        print("\nFinal Answer:", message.content)
        final_answer_received = True
        break


    # CASE 2: TOOL CALLS

    print(
        "\nNumber of tool calls:",
        len(message.tool_calls)
    )


    # SAVE ASSISTANT TOOL CALL MESSAGE

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


    # STEP 11: PARALLEL EXECUTION

    print("\n--- PARALLEL EXECUTION STARTED ---")

    tool_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_to_call = {

            executor.submit(
                execute_tool_call,
                tool_call
            ): tool_call

            for tool_call in message.tool_calls

        }


        # COLLECT EACH RESULT INDEPENDENTLY

        for future in as_completed(future_to_call):

            tool_call = future_to_call[future]

            try:

                result = future.result()

            except Exception as error:

                # Unexpected Future-level failure

                result = {
                    "status": "error",
                    "operation_id": None,
                    "error": str(error)
                }


            # CASE 1: TOOL SUCCESS

            if result["status"] == "success":

                operation_id = result["operation_id"]

                retry_counts[operation_id] = 0

                tool_response = {
                    "status": "success",
                    "operation_id": operation_id,
                    "result": result["result"]
                }

                print(
                    f"\nSUCCESS: {result['tool_name']} "
                    f"= {result['result']}"
                )


            # CASE 2: TOOL FAILURE

            else:

                operation_id = result.get("operation_id")

                retry_key = operation_id or tool_call.id

                retry_counts[retry_key] = (
                    retry_counts.get(retry_key, 0) + 1
                )

                retry_count = retry_counts[retry_key]

                print(
                    f"\nFAILED: {result.get('tool_name')}"
                )

                print("Error:", result["error"])

                print(
                    f"Retry attempts: "
                    f"{retry_count}/{max_tool_retries}"
                )


                # RETRY LIMIT NOT REACHED

                if retry_count < max_tool_retries:

                    tool_response = {
                        "status": "error",
                        "operation_id": operation_id,
                        "error": result["error"],
                        "retry_count": retry_count,
                        "instruction": (
                            "Retry this failed operation "
                            "using the SAME operation_id. "
                            "Do not generate a new ID."
                        )
                    }


                # MAXIMUM RETRIES REACHED

                else:

                    print(
                        "\nMaximum retries reached for:",
                        retry_key
                    )

                    tool_response = {
                        "status": "permanent_failure",
                        "operation_id": operation_id,
                        "error": result["error"],
                        "retry_count": retry_count,
                        "instruction": (
                            "Do not retry this operation again. "
                            "Mark it as failed and continue "
                            "other independent operations."
                        )
                    }


            # PRESERVE ORIGINAL TOOL CALL ID

            tool_results.append({

                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_response)

            })


    # STEP 12: SEND ALL TOOL RESULTS TO MODEL

    messages.extend(tool_results)

    print("\nAll tool results sent back to the model.")

    # Do not stop the whole batch just because one tool failed.
    # The model receives each result and decides the next step.


# STEP 13: FINAL STATUS

if not final_answer_received:

    if stop_execution:

        print(
            "\nAgent stopped because of an API-level error."
        )

    else:

        print("\nMaximum iterations reached.")

        print("Requesting final response...")

        try:

            final_response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tool_choice="none",
                temperature=0
            )

            final_answer = (
                final_response.choices[0].message.content
            )

            print("\nFinal Answer:", final_answer)

        except APIError as error:

            print("\nFinal API request failed:", error)


print("\n========== AGENT EXECUTION FINISHED ==========")
