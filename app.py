import streamlit as st
import json
from datetime import datetime
import pandas as pd
import time
import re
from system_integration import EnhancedRAGPipeline

# Set page configuration
st.set_page_config(
    page_title="Enhanced RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Remove sidebar brand and main menu
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# At the top of app.py
@st.cache_resource
def get_enhanced_rag_pipeline():
    """Create and initialize enhanced RAG pipeline with caching"""
    pipeline = EnhancedRAGPipeline()
    return pipeline

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #90CAF9;
    }
    .response-container {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .citation-container {
        background-color: #E3F2FD;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
    }
    .feedback-container {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .enhancement-container {
        background-color: #E8F5E9;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
    }
    .reliability-high {
        background-color: #C8E6C9;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        color: #1B5E20;
    }
    .reliability-medium {
        background-color: #FFF9C4;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        color: #F57F17;
    }
    .reliability-low {
        background-color: #FFCCBC;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        color: #BF360C;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedRAGInterface:
    def __init__(self):
        # Use cached pipeline instance instead of creating new one
        self.pipeline = get_enhanced_rag_pipeline()
        
        # Check if pipeline is initialized
        if 'pipeline_initialized' not in st.session_state:
            with st.spinner("Loading..."):
                # The initialization is done in the constructor
                st.session_state.pipeline_initialized = True
                
        # Ensure user_id exists in session state
        if 'user_id' not in st.session_state:
            st.session_state.user_id = "default_user"
            
        # Initialize conversation context
        if 'conversation_context' not in st.session_state:
            st.session_state.conversation_context = []
        
    def clean_response(self, response):
        """Remove introductory text from response"""
        if not response:
            return response
            
        # Remove "Based on the provided context information..." and similar phrases
        patterns = [
            r"^Based on the provided context information,\s*I\s*will\s*provide.*?\.\s*\[\d+\]\s*",
            r"^Based on the context information,\s*I\s*will\s*provide.*?\.\s*\[\d+\]\s*",
            r"^I apologize,\s*but\s*there\s*is\s*no.*?\.\s*\[\d+\]\s*",
            r"^Based on the provided context,\s*I\s*will\s*answer.*?\.\s*\[\d+\]\s*"
        ]
        
        clean_text = response
        for pattern in patterns:
            clean_text = re.sub(pattern, "", clean_text, flags=re.DOTALL | re.IGNORECASE)
            
        return clean_text
        
    def run(self):
        # Sidebar for search history only
        with st.sidebar:
            st.markdown("### Search History")
            if 'history' in st.session_state and st.session_state.history:
                history_df = pd.DataFrame(st.session_state.history)
                st.dataframe(history_df[['query', 'timestamp']], hide_index=True)
                
                if st.button("Clear History"):
                    st.session_state.history = []
                    st.session_state.conversation_context = []
        
        # Main content area
        st.markdown("<h1 class='main-header'>Enhanced Multi-Source RAG System</h1>", unsafe_allow_html=True)
        st.markdown("Ask questions with advanced personalization, conversation context, and source reliability.")
        
        # Initialize session state
        if 'response' not in st.session_state:
            st.session_state.response = None
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        # Create search form with Enter key support
        with st.form(key='query_form', clear_on_submit=False):
            col1, col2 = st.columns([5, 1])
            with col1:
                query = st.text_input("Enter your query:", key="query_input")
            with col2:
                search_button = st.form_submit_button("Search", use_container_width=True)
                
        # Process query when search is clicked
        if search_button and query:
            with st.spinner("Processing query..."):
                # Start timing
                start_time = time.time()
                
                # Process query with enhanced pipeline
                st.session_state.response = self.pipeline.enhanced_process_query(
                    query, 
                    user_id=st.session_state.user_id,
                    conversation_context=st.session_state.conversation_context
                )
                
                # Update conversation context for next query
                new_context = {
                    'user': query,
                    'assistant': st.session_state.response.get('response', '')
                }
                st.session_state.conversation_context.append(new_context)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Add to history
                st.session_state.history.append({
                    "query": query,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "response_id": st.session_state.response.get("metadata", {}).get("response_id", "unknown")
                })
                
            # Scroll back to top
            st.rerun()
        
        # Display response if available
        if st.session_state.response:
            response = st.session_state.response
            
            # Display query enhancement information
            if 'metadata' in response and 'query_enhancement' in response['metadata']:
                enhancement = response['metadata']['query_enhancement']
                st.markdown("<div class='enhancement-container'>", unsafe_allow_html=True)
                st.markdown("#### Query Enhancement")
                st.markdown(f"**Original query:** {enhancement.get('original_query', query)}")
                st.markdown(f"**Enhanced query:** {enhancement.get('enhanced_query', 'No enhancement')}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Check if query was corrected (from original pipeline)
            corrected_query = response.get("metadata", {}).get("corrected_query", None)
            if corrected_query:
                st.info(f"Did you mean: **{corrected_query.get('corrected')}**? (Originally: '{corrected_query.get('original')}')")
            
            # Display response - clean it first and remove the "Response" heading
            # Clean the response by removing intro text
            clean_response = self.clean_response(response.get("response", "No response generated"))
            st.markdown("<div class='response-container'>", unsafe_allow_html=True)
            st.markdown(clean_response)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display sources with reliability information
            st.markdown("<h2 class='sub-header'>Sources</h2>", unsafe_allow_html=True)
            
            # Check for enhanced documents with reliability scores
            enhanced_docs = response.get("metadata", {}).get("enhanced_documents", [])
            reliability_info = {}
            if enhanced_docs:
                for doc in enhanced_docs:
                    doc_id = doc.get('id', '')
                    reliability_score = doc.get('reliability_score', 0)
                    reliability_info[doc_id] = reliability_score
            
            # Display citations
            citations = response.get("metadata", {}).get("citations", {}).get("citations", [])
            if citations:
                for i, citation in enumerate(citations):
                    source_info = citation.get('source_info', {})
                    source_name = source_info.get('source', 'Unknown')
                    
                    # Add reliability badge if available
                    reliability_badge = ""
                    if source_name in reliability_info:
                        score = reliability_info[source_name]
                        badge_class = "reliability-high" if score > 0.7 else ("reliability-medium" if score > 0.4 else "reliability-low")
                        reliability_badge = f"<span class='{badge_class}'>Reliability: {score:.2f}</span>"
                    
                    # Create expander title with reliability badge
                    expander_title = f"{citation.get('text', '')} {reliability_badge}"
                    
                    with st.expander(expander_title, expanded=False):
                        st.markdown(f"**Source:** {source_name}")
                        st.markdown(f"**Document Type:** {source_info.get('document_type', 'Unknown')}")
                        
                        if 'chunk_number' in source_info:
                            st.markdown(f"**Chunk:** {source_info.get('chunk_number')} of {source_info.get('total_chunks', 'Unknown')}")
                        
                        if 'url' in source_info:
                            st.markdown(f"**URL:** {source_info.get('url')}")
            else:
                st.info("No citations available")
            
            # Feedback mechanism
            st.markdown("<h2 class='sub-header'>Feedback</h2>", unsafe_allow_html=True)
            st.markdown("<div class='feedback-container'>", unsafe_allow_html=True)
            
            feedback_col1, feedback_col2 = st.columns([2, 3])
            
            with feedback_col1:
                feedback = st.radio("Was this response helpful?", 
                                    ["Yes", "Somewhat", "No"])
            
            with feedback_col2:
                comments = st.text_area("Additional comments (optional)")
            
            if st.button("Submit Feedback"):
                self.save_feedback(query, response, feedback, comments)
                
            st.markdown("</div>", unsafe_allow_html=True)
    
    def save_feedback(self, query, response, feedback, comments):
        # Implement feedback saving logic
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "user_id": st.session_state.user_id,
            "response_id": response.get("metadata", {}).get("response_id", "unknown"),
            "feedback_rating": feedback,
            "comments": comments
        }
        
        # Save to file
        try:
            with open("feedback_log.jsonl", "a") as f:
                f.write(json.dumps(feedback_data) + "\n")
            st.success("Thank you for your feedback!")
            
            # Record feedback in history
            for item in st.session_state.history:
                if item["response_id"] == feedback_data["response_id"]:
                    item["feedback"] = feedback
                    
            # Add feedback to advanced features system
            try:
                profile_updates = {
                    'history': {
                        'feedback': {
                            response.get("metadata", {}).get("response_id", "unknown"): 
                            1 if feedback == "Yes" else (0.5 if feedback == "Somewhat" else 0)
                        }
                    }
                }
                self.pipeline.advanced_features.personalization.update_profile(
                    st.session_state.user_id, profile_updates
                )
            except Exception as e:
                pass
                
        except Exception as e:
            st.error(f"Error saving feedback: {e}")

def main():
    interface = EnhancedRAGInterface()
    interface.run()

if __name__ == "__main__":
    main()