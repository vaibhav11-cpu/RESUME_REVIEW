import streamlit as st
import io
import pdfplumber
import requests
import json
import base64
import time
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# --- API Configuration ---
# Your API key will be provided by the environment, so leave this as an empty string.
apiKey = st.secrets["api_keys"]["gemini"]
api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={apiKey}"

# --- User Interface Setup ---
st.set_page_config(layout="wide", page_title="Smart Resume Reviewer")

st.title("📄 Smart Resume Reviewer")
st.markdown(
    """
    **Your AI-Powered Career Coach:** Upload your resume and get tailored, constructive feedback to optimize it for your dream job.
    """
)
st.divider()

# --- Sidebar for Language Selection ---
st.sidebar.header("Settings")
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de"
}
selected_language_name = st.sidebar.selectbox(
    "Select Language for Feedback",
    options=list(languages.keys())
)
selected_language_code = languages[selected_language_name]


# --- Input Sections ---
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.header("Upload Your Resume")
        uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])
    
    with col2:
        st.header("Specify Job Details")
        job_role = st.text_input(
            "Target Job Role",
            placeholder="e.g., Software Engineer, Data Scientist, Product Manager"
        )
        job_description = st.text_area(
            "Paste Job Description (Optional)",
            placeholder="Paste the full job description here to get more tailored feedback."
        )

st.divider()
st.subheader("AI Feedback Report")
st.write("---")

# --- Core Logic ---

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
            return "\n".join(pages)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Use tenacity to handle API retries with exponential backoff
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=15))
def make_api_call_with_retry(payload):
    """Handles the actual API request with retry logic."""
    response = requests.post(
        api_url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    response.raise_for_status()
    return response

def get_llm_feedback(resume_text, job_role, job_description, lang_code):
    """Sends the resume and job details to the LLM for feedback."""
    
    if not resume_text:
        return None, "Please upload a resume to get feedback."
    
    # Define the system instruction for the LLM's persona and JSON output
    system_prompt = f"""
    You are an expert career coach and a highly experienced resume reviewer. 
    Your task is to provide a comprehensive, constructive, and actionable review of a resume.
    
    Provide all your responses in {selected_language_name}.
    
    Your final response must be a single JSON object with the following schema:

    {{
      "feedback_report": {{
        "areas_of_strength": ["What the candidate does well. Use clear, bulleted points."],
        "suggestions_for_improvement": [
          {{
            "heading": "Clarity & Brevity",
            "advice": "Suggest how to be more concise or clear."
          }},
          {{
            "heading": "Impactful Language",
            "advice": "Suggest using more action verbs and quantifying achievements."
          }},
          {{
            "heading": "Missing Keywords",
            "advice": "Identify relevant skills or terms missing from the resume."
          }},
          {{
            "heading": "Formatting & Readability",
            "advice": "Provide tips on layout, spacing, or font."
          }},
          {{
            "heading": "Tailoring to the Job",
            "advice": "Explain how to better align their experience with the job description."
          }}
        ]
      }},
      "final_score": {{
        "rating": "A score from 1 to 10. Do not include the '/10'.",
        "summary": "A brief paragraph summarizing the key takeaways."
      }},
      "improved_resume": "A fully rewritten, improved version of the resume in Markdown format. Ensure it follows all best practices and professional standards."
    }}
    
    The 'improved_resume' should be a complete, ready-to-use resume text.
    Do not add any text or explanation outside of this structured JSON format.
    """
    
    # Define the user query
    user_query = f"Here is the resume to be reviewed:\n\n---\n{resume_text}\n---\n\nTarget Job Role: {job_role}\n\n"
    if job_description:
        user_query += f"Here is the job description for a more tailored review:\n\n---\n{job_description}\n---"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "feedback_report": {
                        "type": "OBJECT",
                        "properties": {
                            "areas_of_strength": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "suggestions_for_improvement": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "heading": {"type": "STRING"},
                                        "advice": {"type": "STRING"}
                                    }
                                }
                            }
                        }
                    },
                    "final_score": {
                        "type": "OBJECT",
                        "properties": {
                            "rating": {"type": "INTEGER"},
                            "summary": {"type": "STRING"}
                        }
                    },
                    "improved_resume": {"type": "STRING"}
                }
            }
        }
    }

    try:
        response = make_api_call_with_retry(payload)
        data = response.json()
        
        # Check if the response contains the expected content
        if 'candidates' in data and len(data['candidates']) > 0:
            json_str = data['candidates'][0]['content']['parts'][0]['text']
            return json.loads(json_str), None
        else:
            return None, "Could not get a response from the AI. Please try again."
    except requests.exceptions.RequestException as e:
        # Log the full error to the terminal for debugging
        st.error(f"API request failed. Please check your API key and network connection. Details are logged to the console.")
        print("API request failed with error:", e)
        # Using a fallback in case response is not available
        response_content = response.content.decode() if 'response' in locals() and response.content else 'No response content available'
        print("Response content:", response_content)
        return None, "An API error occurred. Please check the console for details."
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return None, f"There was an issue processing the AI's response: {e}. The AI may have provided an unexpected output format."


# --- Button and Output ---
if st.button("Get Resume Feedback", type="primary"):
    if not job_role:
        st.warning("Please specify a target job role.")
    elif not uploaded_file:
        st.warning("Please upload a resume file.")
    else:
        # Check if API key is set
        if not st.secrets.get("api_keys", {}).get("gemini"):
            st.error("API key not found. Please set the Gemini API key in your .streamlit/secrets.toml file.")
        else:
            with st.spinner("Analyzing your resume with the AI..."):
                # Read resume content
                if uploaded_file.type == "application/pdf":
                    resume_content = extract_text_from_pdf(uploaded_file)
                else:
                    resume_content = uploaded_file.read().decode("utf-8")

                if resume_content:
                    st.subheader("AI Feedback Report")
                    st.write("---")
                    
                    try:
                        # Get feedback from LLM
                        feedback_data, error = get_llm_feedback(resume_content, job_role, job_description, selected_language_code)
                        
                        if error:
                            st.error(error)
                        elif feedback_data and "final_score" in feedback_data:
                            # Display the structured feedback
                            st.markdown(f"### Resume Review for {job_role}")
                            st.markdown(f"**Overall Rating:** {feedback_data['final_score']['rating']}/10")
                            st.markdown(f"**Summary:** {feedback_data['final_score']['summary']}")

                            st.markdown("### Areas of Strength")
                            for strength in feedback_data['feedback_report']['areas_of_strength']:
                                st.markdown(f"- {strength}")
                            
                            st.markdown("### Suggestions for Improvement")
                            # Safely loop through suggestions
                            for suggestion in feedback_data['feedback_report']['suggestions_for_improvement']:
                                # Use .get() to safely access keys
                                heading = suggestion.get('heading')
                                advice = suggestion.get('advice')

                                if heading and advice:
                                    st.markdown(f"**{heading}:** {advice}")
                                else:
                                    # Handle malformed or incomplete suggestions
                                    pass

                            st.markdown("### Improved Resume Version")
                            st.code(feedback_data['improved_resume'], language='markdown')
                            
                            st.write("---")
                            st.success("Analysis complete!")
                            
                            # Note: PDF export functionality has been removed to avoid errors.
                            # The code here is a placeholder for future implementation.
                            # pdf_bytes = create_pdf(feedback_data)
                            # if pdf_bytes:
                            #   st.download_button(
                            #       label="Download PDF Report",
                            #       data=pdf_bytes,
                            #       file_name="resume_report.pdf",
                            #       mime="application/pdf"
                            #   )
                        else:
                            st.error("The AI returned an invalid or incomplete response. Please try again.")
                    
                    except RetryError:
                        st.error("The API is currently unavailable. Please try again in a few moments.")
                        
                
# --- Instructions for the user ---
st.markdown("### How to Use")
st.markdown(
    """
    1.  **Select Language**: Choose the language for your feedback from the sidebar.
    2.  **Specify Job Role**: Enter the job title you are targeting (e.g., 'Data Analyst').
    3.  **Upload Resume**: Drag and drop your resume file. PDF and plain text are supported.
    4.  **Get Feedback**: Click the 'Get Resume Feedback' button. The AI will provide a detailed report.
    """
)
st.info("Your resume data is **not** stored. It is used temporarily for the review and then discarded.")
