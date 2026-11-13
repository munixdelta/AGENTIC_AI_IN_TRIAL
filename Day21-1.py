
import json
import os

MEMORY_FILE = "memory.json"


# 1. Load existing memory
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as file:
            return json.load(file)

    return {}


# 2. Save memory
def save_memory(memory):
    with open(MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=4)


# 3. Add or update memory
def add_memory(memory, key, value):
    memory[key] = value
    save_memory(memory)
    print(f"Memory saved: {key} = {value}")


# 4. Retrieve memory
def get_memory(memory, key):
    return memory.get(key, "Information not found")


# 5. Main program
memory = load_memory()

add_memory(memory, "name", "Suraj")
add_memory(memory, "course", "BTech CSE AI")
add_memory(memory, "skill", "Python")

print("\nRetrieving Memory:")

print("Name:", get_memory(memory, "name"))
print("Course:", get_memory(memory, "course"))
print("Skill:", get_memory(memory, "skill"))
