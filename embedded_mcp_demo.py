import os
import openai
from token_manager import get_token_manager
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

# Step 2: Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key in env

def main():
    # Configure TokenManager with env variables
    client_id = os.getenv("AIRBYTE_CLIENT_ID")
    client_secret = os.getenv("AIRBYTE_CLIENT_SECRET")
    workspace_id = os.getenv("AIRBYTE_WORKSPACE_ID")
    token_manager = get_token_manager()
    token_manager.configure(client_id, client_secret, workspace_id)
    AIRBYTE_BEARER_TOKEN = token_manager.get_token()

    # Step 3: Call OpenAI Responses API with MCP tool
    resp = openai.responses.create(
        model="gpt-5",
        tools=[
            {
                "type": "mcp",
                "server_label": "airbyte-embedded-mcp",
                "server_url": "https://mcp.airbyte.ai",
                "headers": {
                    "Authorization": f"Bearer {AIRBYTE_BEARER_TOKEN}"
                },
                "require_approval": "never",
            },
        ],
        input=(
            "You are an experienced financial planner and accountant. "
            "Call the apis_make_request tool to fetch Stripe invoices "
            "using the stripe connector with the id 9def5920-d4eb-41c7-aacd-0369800e4817 "
            "Then, analyze the results and prepare a plan for me to manage my invoices."
        ),
    )

    print(resp)

if __name__ == "__main__":
    main()
