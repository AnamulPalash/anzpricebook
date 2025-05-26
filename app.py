import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# --- Configuration ---
# Configure the Gemini API key.
# In Streamlit Cloud, this should be set as a secret named 'GEMINI_API_KEY'.
# For local testing, you can set it as an environment variable or directly here (not recommended for production).
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("Gemini API Key not found. Please set it as a Streamlit secret named 'GEMINI_API_KEY'.")
    st.stop() # Stop the app if API key is not found

# List of CSV files to load from your pricebook.
# Ensure these files are in the same directory as your app.py when deploying.
CSV_FILES = [
    "Trellix Helix.csv",
    "Trellix Email Security.csv",
    "Trellix Endpoint Security.csv",
    "Trellix Server Security.csv",
    "Trellix Data Security.csv",
    "Trellix EU_DE Data Center.csv",
    "Trellix JP_ AU Data Centers.csv",
    "Trellix SG Data Center.csv",
    "Trellix Network Security.csv",
    "Trellix IVX.csv",
    "Trellix SIEM.csv",
    "Trellix HW Accessories.csv",
    "Trellix Success Services.csv",
    "Trellix Legacy.csv",
    "Trellix Legacy (former FireEye.csv",
    "Skyhigh.csv",
    "Skyhigh Legacy.csv",
    # Exclude general sheets like Coversheet, Index, RawData, Additions, Changes for direct search
    # "Coversheet.csv",
    # "Coversheet_Japanese.csv",
    # "Index.csv",
    # "RawData.csv",
    # "Additions.csv",
    # "MSRP_Margin_Cost_Change.csv",
    # "Deletions.csv",
    # "ALL Changes.csv",
]

# --- Data Loading Function ---
@st.cache_data # Cache the data loading to improve performance
def load_and_combine_data(file_paths):
    """Loads multiple CSV files into pandas DataFrames and combines them."""
    all_data = []
    for file_path in file_paths:
        try:
            # Read CSV, assuming first row is header.
            # Some sheets might have extra rows before the actual header,
            # so we'll try to infer the header or skip rows.
            # For this example, we assume the first row is the header.
            df = pd.read_csv(file_path)
            all_data.append(df)
        except FileNotFoundError:
            st.warning(f"File not found: {file_path}. Please ensure all CSV files are in the same directory.")
            continue
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            continue

    if not all_data:
        st.error("No data loaded. Please check your CSV file paths and names.")
        return pd.DataFrame()

    # Concatenate all dataframes. This might result in many columns if headers differ.
    # We'll rely on Gemini to make sense of the combined data.
    combined_df = pd.concat(all_data, ignore_index=True)

    # Fill NaN values with empty strings for better search and Gemini processing
    combined_df = combined_df.fillna('')
    return combined_df

# --- Search Function ---
def search_dataframe(df, query):
    """Searches the DataFrame for the query in all string columns."""
    if query.strip() == "":
        return pd.DataFrame() # Return empty if query is empty

    query_lower = query.lower()
    results = df[df.apply(lambda row: row.astype(str).str.lower().str.contains(query_lower).any(), axis=1)]
    return results

# --- Gemini Interaction Function ---
def get_gemini_answer(query, search_results_df):
    """
    Uses Gemini to provide a natural language answer based on the search query and results.
    """
    if search_results_df.empty:
        prompt = f"The user searched for '{query}' but no relevant data was found in the pricebook. Please provide a helpful response indicating that no matching items were found."
    else:
        # Convert search results to a string format for Gemini
        # We'll limit the number of rows to avoid exceeding token limits for very large results
        max_rows_for_gemini = 10
        if len(search_results_df) > max_rows_for_gemini:
            results_str = search_results_df.head(max_rows_for_gemini).to_markdown(index=False)
            results_str += f"\n... (and {len(search_results_df) - max_rows_for_gemini} more results not shown due to length)"
        else:
            results_str = search_results_df.to_markdown(index=False)

        prompt = (
            f"The user searched for '{query}' in a pricebook spreadsheet. "
            f"Here are the relevant search results:\n\n{results_str}\n\n"
            f"Based on these results, please provide a concise and helpful answer to the user's query. "
            f"If the results contain pricing information, mention it clearly. "
            f"If the results are unclear or incomplete for the query, state that you are providing information based on the available data. "
            f"Focus on extracting key information like product names, descriptions, and prices if available."
        )

    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error communicating with Gemini API: {e}")
        return "Sorry, I couldn't get an answer from Gemini at this moment."

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Pricebook Search with Gemini")

st.title("📚 Pricebook Search Assistant")
st.markdown(
    """
    This app allows you to search through your uploaded pricebook data and get
    intelligent answers powered by Google Gemini.
    """
)

# Load data once
with st.spinner("Loading pricebook data... This might take a moment."):
    combined_pricebook_df = load_and_combine_data(CSV_FILES)

if not combined_pricebook_df.empty:
    st.success(f"Loaded {len(combined_pricebook_df)} rows from your pricebook.")
    st.info("Enter your query in the text box below to search the pricebook.")

    user_query = st.text_input("Enter your search query (e.g., 'Trellix Endpoint Security price', 'Helix features')", "")

    if st.button("Search Pricebook"):
        if user_query:
            with st.spinner("Searching and generating answer..."):
                # Perform search
                search_results = search_dataframe(combined_pricebook_df, user_query)

                st.subheader("🔍 Search Results from Pricebook")
                if not search_results.empty:
                    st.dataframe(search_results)
                else:
                    st.info("No direct matches found in the pricebook for your query.")

                # Get Gemini's answer
                st.subheader("✨ Gemini's Answer")
                gemini_answer = get_gemini_answer(user_query, search_results)
                st.write(gemini_answer)
        else:
            st.warning("Please enter a query to search.")
else:
    st.error("Could not load pricebook data. Please ensure CSV files are correctly named and present.")

st.markdown("---")
st.markdown("Developed with Streamlit and Google Gemini API.")

