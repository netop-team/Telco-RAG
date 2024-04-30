import openai
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = "your-api-key"
os.environ["OPENAI_API_KEY"] = api_key

def generate(question):
    """Generate a response using the GPT model given a structured question object."""
    model_name = "gpt-3.5-turbo-1106"
    # model_name = "gpt-4-turbo-2024-04-09"
    try:
        client = openai.OpenAI()

        # Constructing the content context from the question object
        content = '\n'.join(question.context)
        prompt = f"""
Please provide the answer to the following question and provide the number of the retrievals you used in your judgment:
{question.query}

Considering the following context:
{content}

Please provide a detailed answer to the following question:
{question.question}
        """

        logging.info("Generated system prompt for OpenAI completion.")
        
        # Generating the model's response based on the final prompt
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content":'You are a telecom expert!'},
                {"role": "user", "content": prompt}
            ],
        )

        # Extracting and cleaning the model's response
        predicted_answers_str = response.choices[0].message.content.replace('"\n', '",\n')
        logging.info("Model response generated successfully.")

        context = f"The retrieved context provided to the LLM is:\n{content}"
        return predicted_answers_str, context, question.question

    except Exception as e:
        # Logging the error and returning a failure indicator
        logging.error(f"An error occurred: {e}")
        return None, None, None

