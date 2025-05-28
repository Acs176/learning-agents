from ollama import AsyncClient

async def chat(query, context):
    system_message = {
        "role": "system",
        "content": f"here's some context to answer the user's question. You're connected to a web crawler and it gives you info on current events. Only answer based on the context data.\n {context}"
    }
    message = {
        "role": "user",
        "content": query
    }
    async for part in await AsyncClient().chat(
        model="llama3.2", messages=[system_message, message], stream=True
    ):
        print(part["message"]["content"], end="", flush=True)
    print()