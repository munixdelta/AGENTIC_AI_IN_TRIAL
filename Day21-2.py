
import json
import os

from groq import Groq
from dotenv import load_dotenv


# ==================================
# 1. INITIALIZATION
# ==================================

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MEMORY_FILE = "memory.json"


# ==================================
# 2. LOAD MEMORY
# ==================================

def load_memory():

    if os.path.exists(MEMORY_FILE):

        with open(MEMORY_FILE, "r") as file:
            return json.load(file)

    return {}


# ==================================
# 3. SAVE MEMORY
# ==================================

def save_memory(memory):

    with open(MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=4)


# ==================================
# 4. RETRIEVE MEMORY
# ==================================

def retrieve_memory(keys=None):

    memory = load_memory()

    # If no keys are provided,
    # retrieve all stored information

    if keys is None:
        return memory

    retrieved = {}

    for key in keys:

        if key in memory:
            retrieved[key] = memory[key]

    return retrieved


# ==================================
# 5. UPDATE MEMORY
# ==================================

def update_memory(key, value):

    memory = load_memory()

    memory[key] = value

    save_memory(memory)

    print(f"Memory updated: {key} = {value}")


# ==================================
# 6. CHAT WITH AGENT
# ==================================

conversation_history = []


def chat_with_agent(user_query):

    # Retrieve relevant memory
    relevant_memory = retrieve_memory()

    # Convert memory into text
    memory_context = json.dumps(
        relevant_memory,
        indent=4
    )

    messages = [

        {
            "role": "system",
            "content": f"""
            You are a helpful AI assistant.

            Use the following stored user memory
            when answering relevant questions.

            USER MEMORY:
            {memory_context}

            If the memory does not contain the answer,
            do not invent personal information.
            """
        }

    ]

    # Add previous conversation
    messages.extend(conversation_history)

    # Add current user query
    messages.append({
        "role": "user",
        "content": user_query
    })

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=messages,

        temperature=0.3
    )

    answer = response.choices[0].message.content

    # Maintain conversation history
    conversation_history.append({
        "role": "user",
        "content": user_query
    })

    conversation_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer


# ==================================
# 7. MAIN PROGRAM
# ==================================

if __name__ == "__main__":

    print("AI Agent with Long-Term Memory")

    while True:

        print("\n1. Chat with Agent")
        print("2. Update Memory")
        print("3. View Memory")
        print("4. Exit")

        choice = input("\nChoose option: ")

        if choice == "1":

            query = input("\nYou: ")

            response = chat_with_agent(query)

            print("\nAgent:", response)

        elif choice == "2":

            key = input("Enter memory key: ")
            value = input("Enter new value: ")

            update_memory(key, value)

        elif choice == "3":

            print("\nStored Memory:")
            print(json.dumps(
                retrieve_memory(),
                indent=4
            ))

        elif choice == "4":

            print("Agent stopped.")
            break

        else:

            print("Invalid choice!")
