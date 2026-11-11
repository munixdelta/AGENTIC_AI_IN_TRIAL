
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# STEP 1: DEFINE FUNCTIONS

def calculate_percentage(number, percentage):

    print(f"Starting percentage: {number}, {percentage}")

    time.sleep(2)

    result = number * percentage / 100

    print(f"Completed percentage: {result}")

    return result


def calculate_sum(a, b):

    print(f"Starting sum: {a} + {b}")

    time.sleep(1)

    result = a + b

    print(f"Completed sum: {result}")

    return result


# STEP 2: TOOL REGISTRY

tool_registry = {
    "calculate_percentage": calculate_percentage,
    "calculate_sum": calculate_sum
}


# STEP 3: OPERATION PLAN

operation_plan = [

    {
        "id": "op1",
        "tool": "calculate_percentage",
        "arguments": {
            "number": 200,
            "percentage": 10
        },
        "depends_on": []
    },

    {
        "id": "op2",
        "tool": "calculate_percentage",
        "arguments": {
            "number": 500,
            "percentage": 20
        },
        "depends_on": []
    },

    {
        "id": "op3",
        "tool": "calculate_sum",
        "arguments": {
            "a": "op1",
            "b": "op2"
        },
        "depends_on": ["op1", "op2"]
    }

]


# STEP 4: STORE RESULTS

completed_operations = {}

pending_operations = operation_plan.copy()


# STEP 5: VALIDATE DEPENDENCIES

all_operation_ids = {
    operation["id"]
    for operation in operation_plan
}

if len(all_operation_ids) != len(operation_plan):
    raise ValueError("Duplicate operation IDs found!")


for operation in operation_plan:

    for dependency in operation["depends_on"]:

        if dependency not in all_operation_ids:

            raise ValueError(
                f"Unknown dependency: {dependency}"
            )


# STEP 6: CHECK DEPENDENCIES

def can_execute(operation):

    return all(
        dependency in completed_operations
        for dependency in operation["depends_on"]
    )


# STEP 7: RESOLVE ARGUMENTS

def resolve_arguments(arguments):

    resolved = {}

    for key, value in arguments.items():

        if isinstance(value, str) and value in completed_operations:

            resolved[key] = completed_operations[value]

        else:

            resolved[key] = value

    return resolved


# STEP 8: PARALLEL EXECUTION

print("\n--- PARALLEL EXECUTION STARTED ---")

with ThreadPoolExecutor(max_workers=3) as executor:

    while pending_operations:

        # Find all currently ready operations

        ready_operations = [
            operation
            for operation in pending_operations
            if can_execute(operation)
        ]

        # No ready operation means unresolved dependency or cycle

        if not ready_operations:

            raise RuntimeError(
                "No operations are ready. "
                "Check for circular dependencies."
            )

        print("\nReady operations:",
              [op["id"] for op in ready_operations])

        # Submit all ready operations concurrently

        future_to_operation = {}

        for operation in ready_operations:

            tool_name = operation["tool"]

            function = tool_registry.get(tool_name)

            if function is None:
                raise ValueError(f"Unknown tool: {tool_name}")

            arguments = resolve_arguments(
                operation["arguments"]
            )

            future = executor.submit(
                function,
                **arguments
            )

            future_to_operation[future] = operation


        # Collect results as tasks complete

        for future in as_completed(future_to_operation):

            operation = future_to_operation[future]

            operation_id = operation["id"]

            try:

                result = future.result()

                completed_operations[operation_id] = result

                print(
                    f"\n{operation_id} completed: {result}"
                )

            except Exception as error:

                print(
                    f"\n{operation_id} failed: {error}"
                )

                raise RuntimeError(
                    f"Operation {operation_id} failed."
                ) from error


        # Remove completed operations from pending list

        for operation in ready_operations:

            pending_operations.remove(operation)


# STEP 9: FINAL SUMMARY

print("\n--- ALL OPERATIONS COMPLETED ---")

for operation_id, result in completed_operations.items():

    print(f"{operation_id}: {result}")


print(
    "\nFinal Result:",
    completed_operations["op3"]
)
