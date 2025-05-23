from openai import OpenAI

client = OpenAI()

file = client.files.create(
    file=open("gl-report-24-october-budget-and-comp.csv", "rb"),
    purpose="user_data"
)

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": file.id,
                },
                {
                    "type": "input_text",
                    "text": "what is the total of staff expenses in the given file. Give me a breakdown on it",
                },
            ]
        }
    ]
)

print(response.output_text)