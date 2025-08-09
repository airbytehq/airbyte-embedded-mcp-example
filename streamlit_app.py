import os
import streamlit as st
from token_manager import get_token_manager
from dotenv import load_dotenv
import openai
import pandas as pd

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-5"

st.title("Embedded MCP Demo: Stripe Invoice Planner")

if st.button("Fetch & Analyze Invoices"):
    client_id = os.getenv("AIRBYTE_CLIENT_ID")
    client_secret = os.getenv("AIRBYTE_CLIENT_SECRET")
    workspace_id = os.getenv("AIRBYTE_WORKSPACE_ID")
    token_manager = get_token_manager()
    token_manager.configure(client_id, client_secret, workspace_id)
    # Force a new token fetch
    token_manager.invalidate_token()
    AIRBYTE_BEARER_TOKEN = token_manager.get_token()

    st.info(f"Fetched new token: {AIRBYTE_BEARER_TOKEN[:8]}... (truncated)")

    prompt = (
       "You are an experienced financial planner and accountant. "
            "Call the apis_make_request tool to fetch Stripe invoices "
            "using the stripe connector with the id 9def5920-d4eb-41c7-aacd-0369800e4817 "
            "Then, analyze the results and prepare a plan for me to manage my invoices."
    )

    import time
    with st.spinner("Fetching customer data via Airbyte Embedded and asking GPT to analyze it..."):
        start_time = time.time()
        resp = openai.responses.create(
            model=MODEL_NAME,
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
            input=prompt,
        )
        duration = time.time() - start_time
    st.success(f"Request completed in {duration:.2f} seconds.")
    # Show assistant analysis (summary)
    analysis = None
    if hasattr(resp, 'output') and resp.output:
        for item in resp.output:
            if hasattr(item, 'content') and item.content:
                for content in item.content:
                    if hasattr(content, 'text'):
                        analysis = content.text
    import re
    if analysis:
        st.subheader("Analysis & Recommendations")
        st.markdown(analysis, unsafe_allow_html=True)

        # Extract Markdown table and show as DataFrame
        table_match = re.search(r'(\|.+\|\n)+', analysis)
        if table_match:
            table_md = table_match.group(0)
            # Parse Markdown table to DataFrame
            lines = [line.strip() for line in table_md.strip().split('\n') if line.strip()]
            if len(lines) > 2:
                headers = [h.strip() for h in lines[0].split('|') if h.strip()]
                rows = [
                    [cell.strip() for cell in row.split('|') if cell.strip()]
                    for row in lines[2:]
                ]
                import pandas as pd
                df_table = pd.DataFrame(rows, columns=headers)
                st.subheader("Invoice Table")
                st.dataframe(df_table)

                # Try to plot Amount Due as a bar chart
                if 'Invoice Number' in df_table.columns and 'Amount Due (USD)' in df_table.columns:
                    # Remove $ and commas, convert to float
                    df_table['Amount Due (USD)'] = df_table['Amount Due (USD)'].replace('[\$,]', '', regex=True).astype(float)
                    st.subheader("Outstanding Invoice Amounts")
                    st.bar_chart(df_table.set_index('Invoice Number')['Amount Due (USD)'])
    else:
        st.info("No analysis found in response.")

    # Parse invoice data from tool call output
    invoices = []
    if hasattr(resp, 'output') and resp.output:
        for item in resp.output:
            if hasattr(item, 'type') and item.type == 'mcp_call' and hasattr(item, 'output'):
                import json
                try:
                    output_dict = json.loads(item.output)
                    if 'response_body' in output_dict:
                        stripe_data = json.loads(output_dict['response_body'])
                        if 'data' in stripe_data and isinstance(stripe_data['data'], list):
                            invoices = stripe_data['data']
                except Exception as e:
                    st.warning(f"Error parsing invoice data: {e}")

    if invoices:
        df = pd.DataFrame(invoices)
        if 'amount_due' in df.columns and 'id' in df.columns:
            st.subheader("Outstanding Invoice Amounts")
            st.bar_chart(df.set_index('id')['amount_due'])
        else:
            st.info("Invoice data found, but missing 'amount_due' or 'id' fields for chart.")
    else:
        st.info("No outstanding invoices found.")
