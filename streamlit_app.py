import os
from pathlib import Path
import json
import time
import re
from typing import Any, Dict, List

import requests
import streamlit as st
from token_manager import get_token_manager
from dotenv import load_dotenv
import openai
import pandas as pd

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4.1"

# Constants for Airbyte Sonar API
AIRBYTE_SONAR_BASE_URL = "https://api.airbyte.ai"
AIRBYTE_SOURCE_ID = "9def5920-d4eb-41c7-aacd-0369800e4817"

# Precompiled regex for parsing Markdown tables from model analysis
MARKDOWN_TABLE_RE = re.compile(r"(\|.+\|\n)+")


def fetch_invoices_via_airbyte_sonar(
    bearer_token: str, source_id: str, stripe_url: str
) -> List[Dict[str, Any]]:
    """
    Fetch Stripe invoices by calling the Airbyte Sonar API proxy directly (outside OpenAI).
    POST https://api.airbyte.ai/api/v1/sonar/apis/{SOURCE_ID}/request
    Body: {"method":"GET","url":"https://api.stripe.com/v1/invoices"}
    Returns a list of invoice dicts (stripe-style "data" array) or [] on failure.
    """
    endpoint = f"{AIRBYTE_SONAR_BASE_URL}/api/v1/sonar/apis/{source_id}/request"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    body = {
        "method": "GET",
        "url": stripe_url,
    }
    try:
        resp = requests.post(endpoint, headers=headers, json=body, timeout=60)
        if resp.status_code != 200:
            st.warning(f"Airbyte API fetch failed with status {resp.status_code}")
            return []
        data = resp.json()

        # Common shapes:
        # 1) {"response_body": "...json string..."}
        # 2) {"body": "...json string..." or {...}}
        # 3) {"data": [...]}  (already parsed)
        stripe_data = None

        if isinstance(data, dict) and "response_body" in data:
            try:
                stripe_data = json.loads(data["response_body"])
            except Exception:
                pass

        if stripe_data is None and isinstance(data, dict) and "body" in data:
            body_field = data["body"]
            if isinstance(body_field, str):
                try:
                    stripe_data = json.loads(body_field)
                except Exception:
                    stripe_data = None
            elif isinstance(body_field, dict):
                stripe_data = body_field

        if stripe_data is None:
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                stripe_data = data
            elif isinstance(data, list):
                # Some proxies might return the array directly
                return data

        if isinstance(stripe_data, dict) and isinstance(stripe_data.get("data"), list):
            return stripe_data["data"]

    except requests.RequestException as e:
        st.warning(f"Airbyte API fetch error: {e}")

    # Fallback
    st.warning(
        "Unable to fetch invoices from Airbyte API. Proceeding without live data."
    )
    return []


def setup_authentication() -> str:
    """
    Configure and fetch Airbyte authentication token.
    """
    client_id = os.getenv("AIRBYTE_CLIENT_ID")
    client_secret = os.getenv("AIRBYTE_CLIENT_SECRET")
    workspace_id = os.getenv("AIRBYTE_WORKSPACE_ID")

    token_manager = get_token_manager()
    token_manager.configure(client_id, client_secret, workspace_id)
    token_manager.invalidate_token()
    bearer_token = token_manager.get_token()

    st.info(f"Fetched new token: {bearer_token[:8]}... (truncated)")
    return bearer_token


def fetch_invoices_via_airbyte(bearer_token: str) -> List[Dict[str, Any]]:
    """
    Fetch invoice data through Airbyte Sonar API proxy.
    """
    stripe_url = "https://api.stripe.com/v1/invoices"

    fetch_start = time.time()
    invoices = fetch_invoices_via_airbyte_sonar(
        bearer_token, AIRBYTE_SOURCE_ID, stripe_url
    )
    fetch_duration = time.time() - fetch_start

    st.info(f"Airbyte API fetch completed in {fetch_duration:.2f} seconds.")
    if invoices:
        st.success(f"Fetched {len(invoices)} invoices via Airbyte Embedded API.")
    else:
        st.info(
            "No invoices returned from Airbyte Embedded API; analysis will proceed with empty data."
        )

    return invoices


def build_analysis_prompt(invoices: List[Dict[str, Any]]) -> str:
    """
    Build the analysis prompt with simplified invoice data.
    """
    base_prompt = (
        "You are an experienced financial planner and accountant. "
        "Analyze the provided Stripe invoice data and prepare a plan for me to manage my invoices."
    )
    invoices_for_prompt = simplify_invoices(invoices, limit=50)
    prompt = (
        f"{base_prompt}\n\n"
        f"Here is a JSON array of invoices (subset of fields, up to 50 rows):\n"
        f"{json.dumps(invoices_for_prompt, ensure_ascii=False)}\n\n"
        "Please provide:\n"
        "- Key insights and trends\n"
        "- Risk assessment (e.g., overdue, large unpaid balances)\n"
        "- A prioritized plan of actions\n"
        "- If useful, include a concise Markdown table summary (e.g., Invoice Number, Amount Due (USD), Status, Due Date)"
    )

    # Save prompt to file for debugging
    try:
        filepath = os.path.join(os.getcwd(), "test2_prompt.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(prompt)
        file_uri = Path(filepath).as_uri()
        st.info(f"Saved prompt to: {filepath}")
        st.markdown(f"[Open test2_prompt.txt]({file_uri})")
    except Exception as e:
        st.warning(f"Failed to write test2_prompt.txt: {e}")

    return prompt


def analyze_with_openai(prompt: str) -> str:
    """
    Send prompt to OpenAI for analysis.
    """
    with st.spinner("Analyzing invoice data with the model..."):
        start_time = time.time()
        resp = openai.responses.create(
            model=MODEL_NAME,
            input=prompt,
        )
        duration = time.time() - start_time

    st.success(f"OpenAI Request completed in {duration:.2f} seconds.")

    # Extract analysis text from response
    analysis = None
    if hasattr(resp, 'output') and resp.output:
        for item in resp.output:
            if hasattr(item, 'content') and item.content:
                for content in item.content:
                    if hasattr(content, 'text'):
                        analysis = content.text

    return analysis or ""


def display_results(analysis: str, invoices: List[Dict[str, Any]]) -> None:
    """
    Display analysis results and visualizations.
    """
    if analysis:
        st.subheader("Analysis & Recommendations")
        st.markdown(analysis, unsafe_allow_html=True)

        # Extract Markdown table and show as DataFrame
        table_match = MARKDOWN_TABLE_RE.search(analysis)
        if table_match:
            table_md = table_match.group(0)
            # Parse Markdown table to DataFrame
            lines = [line.strip() for line in table_md.strip().split('\n') if line.strip()]
            if len(lines) > 2:
                headers = [h.strip() for h in lines[0].split('|') if h.strip()]
                rows = [
                    [cell.strip() for cell in row.split("|") if cell.strip()]
                    for row in lines[2:]
                ]
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

    # Render a bar chart from the fetched invoices directly (if available)
    if invoices:
        df = pd.DataFrame(invoices)
        if 'amount_due' in df.columns and 'id' in df.columns:
            st.subheader("Outstanding Invoice Amounts (from fetched data)")
            st.bar_chart(df.set_index('id')['amount_due'])
        else:
            st.info("Invoice data found, but missing 'amount_due' or 'id' fields for chart.")
    else:
        st.info("No outstanding invoices found.")


def simplify_invoices(
    invoices: List[Dict[str, Any]], limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Reduce invoice objects to a compact subset of fields to keep the prompt concise.
    """
    fields = [
        "id",
        "customer",
        "status",
        "amount_due",
        "currency",
        "due_date",
        "created",
        "paid",
        "amount_paid",
    ]
    slim: List[Dict[str, Any]] = []
    for inv in invoices[:limit]:
        slim.append({k: inv.get(k) for k in fields if k in inv})
    return slim


st.title("Embedded MCP Demo Test 2: Stripe Invoice Planner")
st.text(f"Currently using *{MODEL_NAME}* for analysis.")

if st.button("Fetch & Analyze Invoices"):
    # 1. Setup Authentication
    bearer_token = setup_authentication()

    # 2. Fetch Invoices via Airbyte
    invoices = fetch_invoices_via_airbyte(bearer_token)

    # 3. Build Analysis Prompt
    prompt = build_analysis_prompt(invoices)

    # 4. Analyze with OpenAI
    analysis = analyze_with_openai(prompt)

    # 5. Display Results
    display_results(analysis, invoices)
