import streamlit as st
import pandas as pd
import numpy as np
from google import genai


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Data Analyst")
st.caption(
    "Upload your CSV and ask questions about your data using Gemini AI."
)


# ============================================================
# GEMINI API KEY
# ============================================================

# API key is automatically loaded from:
#
# .streamlit/secrets.toml
#
# Example:
#
# GEMINI_API_KEY = "your-real-key"
#

try:
    api_key = st.secrets["GEMINI_API_KEY"]

except Exception:
    st.error(
        "❌ Gemini API key not found.\n\n"
        "Please check:\n"
        ".streamlit/secrets.toml\n\n"
        "The key name must be:\n"
        "GEMINI_API_KEY"
    )
    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

try:
    client = genai.Client(api_key=api_key)

except Exception as e:
    st.error(f"❌ Gemini client initialization failed:\n{e}")
    st.stop()


# ============================================================
# FIND AVAILABLE GEMINI MODEL
# ============================================================

@st.cache_resource
def get_available_model():

    preferred_models = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    try:

        available_models = []

        for model in client.models.list():

            name = getattr(model, "name", "")

            supported_actions = getattr(
                model,
                "supported_actions",
                []
            )

            if not name:
                continue

            if "generateContent" not in supported_actions:
                continue

            clean_name = name.replace("models/", "")

            available_models.append(clean_name)

        # First try preferred models
        for preferred in preferred_models:

            if preferred in available_models:
                return preferred

        # Fallback to any Gemini generation model
        for model_name in available_models:

            if model_name.startswith("gemini-"):
                return model_name

        return None

    except Exception:
        return None


MODEL_NAME = get_available_model()


# ============================================================
# MODEL STATUS
# ============================================================

if MODEL_NAME:

    st.sidebar.success(
        f"🟢 Gemini Ready\n\n{MODEL_NAME}"
    )

else:

    st.sidebar.error(
        "❌ No usable Gemini model found."
    )


# ============================================================
# CSV UPLOAD
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)


# ============================================================
# INITIAL INFORMATION
# ============================================================

if not uploaded_file:

    st.info(
        "👈 Upload a CSV file from the sidebar to start."
    )

    st.stop()


# ============================================================
# LOAD CSV
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"❌ Could not read CSV:\n{e}"
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DATASET SUCCESS MESSAGE
# ============================================================

st.success(
    f"✅ {uploaded_file.name} loaded successfully!"
)


# ============================================================
# DATASET METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Rows",
        df.shape[0]
    )


with col2:

    st.metric(
        "Columns",
        df.shape[1]
    )


with col3:

    missing_values = int(
        df.isnull().sum().sum()
    )

    st.metric(
        "Missing Values",
        missing_values
    )


with col4:

    duplicate_rows = int(
        df.duplicated().sum()
    )

    st.metric(
        "Duplicate Rows",
        duplicate_rows
    )


# ============================================================
# DATASET PREVIEW
# ============================================================

with st.expander("👀 View Dataset"):

    st.dataframe(
        df.head(100),
        use_container_width=True
    )


# ============================================================
# DATASET STRUCTURE
# ============================================================

with st.expander("📊 Dataset Structure"):

    structure_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing": df.isnull().sum().values,
        "Unique Values": [
            df[column].nunique()
            for column in df.columns
        ]
    })

    st.dataframe(
        structure_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

with st.expander("📈 Numerical Summary"):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_columns) > 0:

        summary = df[numeric_columns].describe().T

        st.dataframe(
            summary,
            use_container_width=True
        )

    else:

        st.info(
            "No numerical columns found."
        )


# ============================================================
# DATASET INFORMATION FOR GEMINI
# ============================================================

def build_dataset_context(dataframe):

    context = []

    context.append(
        f"Dataset rows: {dataframe.shape[0]}"
    )

    context.append(
        f"Dataset columns: {dataframe.shape[1]}"
    )

    context.append(
        f"Column names: {list(dataframe.columns)}"
    )

    context.append(
        "\nData types:\n"
        + dataframe.dtypes.to_string()
    )

    context.append(
        "\nMissing values:\n"
        + dataframe.isnull().sum().to_string()
    )

    # Numeric statistics
    numeric = dataframe.select_dtypes(
        include=np.number
    )

    if not numeric.empty:

        try:

            context.append(
                "\nNumerical statistics:\n"
                + numeric.describe().to_string()
            )

        except Exception:
            pass

    # Sample data
    context.append(
        "\nFirst 30 rows:\n"
        + dataframe.head(30).to_string(
            index=False
        )
    )

    return "\n".join(context)


dataset_context = build_dataset_context(df)


# ============================================================
# CHAT HISTORY DISPLAY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER QUESTION
# ============================================================

user_question = st.chat_input(
    "Ask anything about your CSV..."
)


# ============================================================
# GEMINI ANALYSIS
# ============================================================

if user_question:

    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_question
        )

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })


    # --------------------------------------------------------
    # Conversation History
    # --------------------------------------------------------

    history_parts = []

    for message in st.session_state.messages[-8:]:

        history_parts.append(
            f"{message['role'].upper()}:\n"
            f"{message['content']}"
        )

    conversation_history = "\n\n".join(
        history_parts
    )


    # --------------------------------------------------------
    # AI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an expert AI Data Analyst.

You are analyzing a CSV dataset uploaded by the user.

================ DATASET INFORMATION ================

{dataset_context}

================ CONVERSATION HISTORY ================

{conversation_history}

================ USER QUESTION ================

{user_question}

================ INSTRUCTIONS ================

1. Answer the user's question using the provided dataset.

2. Do NOT invent values that are not supported by the dataset.

3. If the dataset does not contain enough information,
   clearly explain that.

4. Perform calculations carefully when required.

5. Mention relevant column names when useful.

6. For numerical questions, provide the calculated result.

7. For statistical questions, explain the result simply.

8. For machine-learning questions, explain what can
   realistically be done using this dataset.

9. Do not claim that an ML model has been trained unless
   the user actually provides a trained model.

10. Keep the response practical and easy to understand.

11. Use Markdown formatting where useful.

12. If the user simply greets you, respond naturally.

13. Never reveal the Gemini API key.

14. Do not make up rows, columns, statistics, or predictions.

================ END =================
"""


    # --------------------------------------------------------
    # ASSISTANT RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🤖 Analyzing your dataset..."
        ):

            try:

                if MODEL_NAME is None:

                    raise RuntimeError(
                        "No Gemini model supporting "
                        "generateContent is available "
                        "for this API key."
                    )


                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt
                )


                answer = response.text


                if not answer:

                    answer = (
                        "⚠️ Gemini returned an empty response."
                    )


                st.markdown(answer)


            except Exception as e:

                answer = (
                    "❌ Gemini request failed.\n\n"
                    f"**Error:** `{e}`"
                )

                st.error(answer)


    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📊 Dataset Information"
)

st.sidebar.write(
    f"Rows: **{df.shape[0]}**"
)

st.sidebar.write(
    f"Columns: **{df.shape[1]}**"
)

st.sidebar.write(
    f"Missing values: **{int(df.isnull().sum().sum())}**"
)

st.sidebar.write(
    f"Duplicate rows: **{int(df.duplicated().sum())}**"
)


# ============================================================
# CLEAR CHAT
# ============================================================

if st.sidebar.button(
    "🗑️ Clear Chat"
):

    st.session_state.messages = []

    st.rerun()