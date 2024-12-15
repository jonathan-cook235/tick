import asyncio
import os
import time
from typing import Any, List, Optional, Tuple, Union
import cohere
import openai
import together

async def call_api_single_prompt(provider: str, client: Any, model: str, prompt: str) -> str:
    """
    Gets response from single prompt API call.
    Params:
    provider (str): Model provider.
    client (Any): Chat client.
    model (str): Model name.
    prompt (str): User message.
    Returns:
    response (str): API response.
    """
    if provider == "cohere":
        try:
            response = await client.chat(model=model, message=prompt)
            response = response.text
        except:
            response = ""
    elif provider == "openai" or provider == "meta":
        while True:
            try:
                response = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}]
                )
                response = response.choices[0].message.content
                break
            except:
                continue
    else:
        raise NotImplementedError
    return response
  
async def call_api_multi_response(
    provider: str, client: Any, model: str, prompt: str, num_generations: int
) -> List[str]:
    """
    Gets multiple responses for the prompt in parallel.
    Params:
    provider (str): Model provider.
    client (Any): Chat client.
    model (str): Model name.
    prompt (str): User message.
    n (int): Number of responses.
    Returns:
    responses (list): API responses.
    """
    requests = [call_api_single_prompt(provider, client, model, prompt) for i in range(num_generations)]
    responses = await asyncio.gather(*requests)
    return responses
  
async def call_api(
    provider: str, client: Any, model: str, prompts: List[str], num_generations: Optional[int] = None
) -> List[Union[str, List[str]]]:
    """
    Gets responses from parallel API calls.
    Params:
    provider (str): Model provider.
    client (Any): Chat client.
    model (str): Model name.
    prompts (list of strings): User messages.
    n (optional; int): Number of responses per user message.
    Returns:
    responses (list): API responses.
    """
    assert isinstance(prompts, list)
    start = time.time()
    while True:
        try:
            print("Making requests.")
            if num_generations:
                requests = [
                    call_api_multi_response(provider, client, model, prompt, num_generations) for prompt in prompts
                ]
            else:
                requests = [call_api_single_prompt(provider, client, model, prompt) for prompt in prompts]
            print("Gathering.")
            # responses = await asyncio.wait_for(asyncio.gather(*requests), timeout=300)
            responses = await asyncio.gather(*requests)
            break
        except asyncio.TimeoutError:
            print("API calls timed out. Trying again.")
            continue
    print(f"Done in {time.time() - start:.2f}")
    return responses
  
def get_api_client(provider: str) -> Tuple:
    if provider == "cohere":
        api_key = os.environ["COHERE_API_KEY"]
        client = cohere.AsyncClient(api_key=api_key)
        model = "command-r-plus-08-2024"
    elif provider == "openai":
        api_key = os.environ["OPENAI_API_KEY"]
        client = openai.AsyncOpenAI(api_key=api_key)
        model = "gpt-4o"
    elif provider == "meta":
        api_key = os.environ["TOGETHERAI_API_KEY"]
        client = together.AsyncTogether(api_key=api_key)
        model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    else:
        raise NotImplementedError
    return client, model
