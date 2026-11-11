
# DAY 19-3: DYNAMIC OPERATION PLANNING
# Dependency Management and Sequential Execution


# STEP 1: DEFINE PYTHON FUNCTIONS

def calculate_percentage(number, percentage):
    return number * percentage / 100


def calculate_sum(a, b):
    return a + b


# STEP 2: CREATE TOOL REGISTRY

tool_registry = {
    "calculate_percentage": calculate_percentage,
    "calculate_sum": calculate_sum
}


# STEP 3: DEFINE OPERATION PLAN

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
        "tool": "calculate_sum",
        "arguments": {
            "a": "op1",
            "b": 50
        },
        "depends_on": ["op1"]
    },

    {
        "id": "op3",
        "tool": "calculate_percentage",
        "arguments": {
            "number": "op2",
            "percentage": 20
        },
        "depends_on": ["op2"]
    }

]


# STEP 4: STORE OPERATION RESULTS

completed_operations = {}

pending_operations = operation_plan.copy()


# STEP 5: VALIDATE OPERATION PLAN

# Collect all operation IDs

all_operation_ids = {
    operation["id"]
    for operation in operation_plan
}


# Check duplicate operation IDs

if len(all_operation_ids) != len(operation_plan):
    raise ValueError("Duplicate operation IDs found!")


# Check whether all dependencies exist

for operation in operation_plan:

    for dependency in operation["depends_on"]:

        if dependency not in all_operation_ids:

            raise ValueError(
                f"{operation['id']} depends on "
                f"unknown operation: {dependency}"
            )


print("All dependencies are valid!")


# STEP 6: CHECK DEPENDENCIES

def can_execute(operation):

    for dependency in operation["depends_on"]:

        if dependency not in completed_operations:
            return False

    return True


# STEP 7: RESOLVE ARGUMENTS

def resolve_arguments(arguments):

    resolved = {}

    for key, value in arguments.items():

        # Replace operation ID with its actual result

        if isinstance(value, str) and value in completed_operations:

            resolved[key] = completed_operations[value]

        else:

            resolved[key] = value

    return resolved


# STEP 8: DYNAMIC OPERATION EXECUTION

print("\n--- EXECUTION STARTED ---")

while pending_operations:

    progress = False

    # Check every pending operation

    for operation in pending_operations.copy():

        # Execute only when dependencies are completed

        if can_execute(operation):

            operation_id = operation["id"]

            tool_name = operation["tool"]

            print(f"\nExecuting: {operation_id}")

            # Find corresponding Python function

            function = tool_registry.get(tool_name)

            if function is None:
                raise ValueError(f"Unknown tool: {tool_name}")

            # Resolve previous operation results

            arguments = resolve_arguments(
                operation["arguments"]
            )

            print("Arguments:", arguments)

            # Execute the function

            result = function(**arguments)

            # Store successful result

            completed_operations[operation_id] = result

            print("Result:", result)

            # Remove completed operation

            pending_operations.remove(operation)

            progress = True

    # Detect unresolved or circular dependencies

    if not progress:

        raise RuntimeError(
            "Unable to execute pending operations. "
            "Check for circular dependencies."
        )


# STEP 9: DISPLAY FINAL SUMMARY

print("\n--- ALL OPERATIONS COMPLETED ---")

for operation_id, result in completed_operations.items():

    print(f"{operation_id}: {result}")


# STEP 10: FINAL RESULT

print("\nFinal Result:", completed_operations["op3"])
