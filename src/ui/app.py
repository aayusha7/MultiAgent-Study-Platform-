"""Streamlit UI for Multi-Agent Learning Platform"""

import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import json
import random
import html

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.manager_agent import ManagerAgent
from src.core.messages import ContentType, LearningMode
from src.core.logger import logger
from src.core.memory import load_state


# Page configuration
st.set_page_config(
    page_title="Multi-Agent Learning Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "manager" not in st.session_state:
    st.session_state.manager = ManagerAgent()
    st.session_state.session_id = f"session_{os.urandom(4).hex()}"
    st.session_state.uploaded_file = None
    st.session_state.extracted_chunks = None
    st.session_state.generated_content = None
    st.session_state.current_mode = None
    st.session_state.feedback_given = False
    st.session_state.flipped = {}  # For flashcard flip state
    st.session_state.quiz_answers = {}  # For storing quiz answers
    st.session_state.quiz_submitted = {}  # For tracking submitted quizzes
    st.session_state.checkpoint_responses = {}  # For storing interactive checkpoint responses

# Load theme CSS if available
theme_css_path = Path(__file__).parent / "theme.css"
if theme_css_path.exists():
    with open(theme_css_path, 'r') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_survey():
    """Render learning style survey"""
    st.title("🎓 Welcome to Your Personalized Learning Platform")
    st.markdown("### Let's find your preferred learning style")
    
    preference = st.radio(
        "What's your preferred learning style?",
        options=[
            LearningMode.QUIZ.value,
            LearningMode.FLASHCARD.value,
            LearningMode.INTERACTIVE.value,
            LearningMode.UNKNOWN.value
        ],
        format_func=lambda x: {
            LearningMode.QUIZ.value: "📝 Quizzes (Test your knowledge)",
            LearningMode.FLASHCARD.value: "🃏 Flashcards (Quick memorization)",
            LearningMode.INTERACTIVE.value: "🎯 Interactive (Step-by-step learning)",
            LearningMode.UNKNOWN.value: "❓ I don't know (Show me all options)"
        }[x]
    )
    
    if st.button("Submit Preference", type="primary"):
        result = st.session_state.manager.process_user_request(
            "survey",
            {"preference": preference},
            st.session_state.session_id
        )
        
        if result.get("success"):
            st.success("✅ Preference saved! Upload a file to get started.")
            st.rerun()
        else:
            st.error(f"Error: {result.get('error', 'Unknown error')}")


def render_quiz_content(data: Dict[str, Any]):
    """Render quiz content"""
    questions = data.get("questions", [])
    
    if not questions:
        st.warning("No questions generated.")
        return
    
    st.subheader("📝 Quiz")
    
    # Initialize quiz state if needed
    if "quiz_shuffled" not in st.session_state:
        st.session_state.quiz_shuffled = {}
    
    user_answers = {}
    for i, q in enumerate(questions):
        st.markdown(f"### Question {i + 1}")
        st.markdown(f"**{q.get('question', '')}**")
        
        options = q.get("options", [])
        correct_answer_idx = q.get("correct_answer", 0)
        
        if len(options) >= 4:
            # Shuffle options if not already shuffled for this question
            if i not in st.session_state.quiz_shuffled:
                # Create shuffled options with mapping
                shuffled_options = options.copy()
                random.shuffle(shuffled_options)
                
                # Find new index of correct answer after shuffling
                correct_answer_text = options[correct_answer_idx]
                new_correct_idx = shuffled_options.index(correct_answer_text)
                
                st.session_state.quiz_shuffled[i] = {
                    "options": shuffled_options,
                    "correct_idx": new_correct_idx,
                    "original_correct": correct_answer_idx
                }
            
            shuffled_data = st.session_state.quiz_shuffled[i]
            shuffled_options = shuffled_data["options"]
            
            # Get user's answer
            answer_key = f"quiz_q_{i}"
            if answer_key not in st.session_state.quiz_answers:
                st.session_state.quiz_answers[answer_key] = None
            
            # Determine default index
            default_idx = None
            if st.session_state.quiz_answers[answer_key] and st.session_state.quiz_answers[answer_key] in shuffled_options:
                try:
                    default_idx = shuffled_options.index(st.session_state.quiz_answers[answer_key])
                except ValueError:
                    default_idx = None
            
            selected_answer = st.radio(
                "Select your answer:",
                options=shuffled_options,
                key=answer_key,
                label_visibility="collapsed",
                index=default_idx
            )
            
            # Store answer
            if selected_answer:
                st.session_state.quiz_answers[answer_key] = selected_answer
            
            # Submit button for this question
            submit_key = f"submit_q_{i}"
            if st.button("Submit Answer", key=submit_key):
                st.session_state.quiz_submitted[i] = True
                st.rerun()
            
            # Show result if submitted
            if st.session_state.quiz_submitted.get(i, False):
                user_selected_idx = shuffled_options.index(st.session_state.quiz_answers[answer_key]) if st.session_state.quiz_answers[answer_key] in shuffled_options else -1
                correct_idx = shuffled_data["correct_idx"]
                
                if user_selected_idx == correct_idx:
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Incorrect. The correct answer is: **{shuffled_options[correct_idx]}**")
                
                if "explanation" in q:
                    with st.expander("💡 Explanation"):
                        st.markdown(q["explanation"])
                
                # Show source reference
                if "source_reference" in q:
                    st.info(f"📄 **Source:** {q['source_reference']}")
            else:
                # Show explanation expander even before submission (optional)
                if "explanation" in q:
                    with st.expander("💡 Show Explanation (after submitting)"):
                        st.info("Submit your answer first to see the explanation.")
        
        st.divider()
    
    return user_answers


def render_flashcard_content(data: Dict[str, Any]):
    """Render flashcard content with amazing 3D flip animation - clickable card"""
    cards = data.get("cards", [])
    
    if not cards:
        st.warning("No flashcards generated.")
        return
    
    st.subheader("🃏 Flashcards")
    st.caption("👆 Click on the card to flip it!")
    
    # Initialize current_card if not exists
    if "current_card" not in st.session_state:
        st.session_state.current_card = 0
    
    # Ensure flipped dict exists (already initialized in session state)
    if st.session_state.current_card < len(cards):
        card = cards[st.session_state.current_card]
        is_flipped = st.session_state.flipped.get(st.session_state.current_card, False)
        
        # Create a unique ID for this card
        card_id = f"flashcard-{st.session_state.current_card}"
        flip_key = f"flip_{st.session_state.current_card}"
        
        # Create the animated flashcard HTML (escape HTML for safety)
        flip_class = "flipped" if is_flipped else ""
        front_text = html.escape(card.get('front', ''))
        back_text = html.escape(card.get('back', ''))
        
        # Create clickable flashcard using form - card is clickable via form submit
        current_card_num = st.session_state.current_card
        
        # Use form to make entire card clickable
        with st.form(key=f"flashcard_form_{current_card_num}", clear_on_submit=False):
            # Card HTML with click handler
            flashcard_html = f"""
            <div class="flashcard-container" id="card-wrapper-{card_id}" style="position: relative; max-width: 600px; margin: 20px auto; cursor: pointer;" onclick="this.closest('form').requestSubmit();">
                <div class="flashcard {flip_class}" id="{card_id}">
                    <div class="flashcard-front">
                        <div class="flashcard-icon">🃏</div>
                        <h3>{front_text}</h3>
                        <p style="opacity: 0.8; font-size: 0.9em; margin-top: 20px;"></p>
                    </div>
                <div class="flashcard-back">
                    <div class="flashcard-icon">✨</div>
                    <h3>Answer</h3>
                    <p>{back_text}</p>
                    {f'<p style="opacity: 0.9; font-size: 0.85em; margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.25); border-radius: 5px;"><strong>📄 Source:</strong> {html.escape(card.get("source_reference", ""))}</p>' if card.get("source_reference") else ''}
                    <p style="opacity: 0.8; font-size: 0.9em; margin-top: 20px;"></p>
                </div>
                </div>
            </div>
            <style>
            /* Hide the submit button completely */
            form[data-testid*="flashcard_form_{current_card_num}"] button[type="submit"] {{
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
                width: 0 !important;
                padding: 0 !important;
                margin: 0 !important;
                opacity: 0 !important;
            }}
            </style>
            """
            
            st.markdown(flashcard_html, unsafe_allow_html=True)
            
            # Hidden submit button (form will submit when card is clicked)
            submitted = st.form_submit_button("Flip", key=flip_key)
            if submitted:
                st.session_state.flipped[st.session_state.current_card] = not is_flipped
                st.rerun()
        
        # Navigation buttons
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Previous", disabled=st.session_state.current_card == 0, 
                        key=f"prev_card_{st.session_state.current_card}", use_container_width=True):
                # Reset flip state when changing cards
                st.session_state.current_card = max(0, st.session_state.current_card - 1)
                st.rerun()
        
        with col_next:
            if st.button("Next ➡️", disabled=st.session_state.current_card >= len(cards) - 1, 
                        key=f"next_card_{st.session_state.current_card}", use_container_width=True):
                # Reset flip state when changing cards
                st.session_state.current_card = min(len(cards) - 1, st.session_state.current_card + 1)
                st.rerun()
        
        # Card counter with animation
        st.markdown(
            f'<div class="card-counter">📊 Card {st.session_state.current_card + 1} of {len(cards)}</div>',
            unsafe_allow_html=True
        )
    
    return len(cards)


def render_interactive_content(data: Dict[str, Any]):
    """Render interactive lesson content with engaging UI"""
    steps = data.get("steps", [])
    title = data.get("title", "Interactive Lesson")
    
    if not steps:
        st.warning("No interactive content generated.")
        return
    
    # Initialize current_step if not exists
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    
    # Initialize checkpoint responses if not exists
    if "checkpoint_responses" not in st.session_state:
        st.session_state.checkpoint_responses = {}
    
    # Progress bar
    progress = (st.session_state.current_step + 1) / len(steps)
    st.progress(progress, text=f"Progress: Step {st.session_state.current_step + 1} of {len(steps)}")
    
    # Title
    st.markdown(f"# 🎯 {title}")
    st.divider()
    
    if st.session_state.current_step < len(steps):
        step = steps[st.session_state.current_step]
        step_num = step.get('step_number', st.session_state.current_step + 1)
        step_title = step.get('title', f'Step {step_num}')
        step_content = step.get("content", "")
        
        # Step header with nice styling
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; 
                    border-radius: 10px; 
                    color: white; 
                    margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">Step {step_num}: {step_title}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Step content
        st.markdown("### 📚 Content")
        st.markdown(step_content)
        
        # Show source reference for step
        if "source_reference" in step and step.get("source_reference"):
            st.info(f"📄 **Source:** {step['source_reference']}")
        
        # Checkpoint section
        if "checkpoint" in step and step.get("checkpoint"):
            st.divider()
            st.markdown("### ✅ Checkpoint")
            st.info(step["checkpoint"])
            
            checkpoint_key = f"checkpoint_{st.session_state.current_step}"
            checkpoint_response = st.text_area(
                "📝 Your response:",
                key=checkpoint_key,
                height=150,
                placeholder="Type your thoughts, answer, or reflection here...",
                help="Take a moment to reflect on what you've learned"
            )
            
            # Save response
            if checkpoint_response:
                st.session_state.checkpoint_responses[checkpoint_key] = checkpoint_response
                st.success("✅ Response saved! Great thinking!")
            elif checkpoint_key in st.session_state.checkpoint_responses:
                st.info(f"💭 Your previous response: {st.session_state.checkpoint_responses[checkpoint_key][:100]}...")
            
            # Show answer/solution if available
            if "checkpoint_answer" in step and step.get("checkpoint_answer"):
                show_answer_key = f"show_answer_{st.session_state.current_step}"
                if show_answer_key not in st.session_state:
                    st.session_state[show_answer_key] = False
                
                # Toggle button
                button_text = "🙈 Hide Answer" if st.session_state[show_answer_key] else "💡 Show Answer"
                if st.button(button_text, key=f"toggle_answer_{st.session_state.current_step}", use_container_width=False):
                    st.session_state[show_answer_key] = not st.session_state[show_answer_key]
                    st.rerun()
                
                # Display answer if toggled on
                if st.session_state.get(show_answer_key, False):
                    st.markdown("---")
                    st.markdown("### 💡 Model Answer")
                    answer_text = html.escape(step["checkpoint_answer"])
                    st.markdown(f"""
                    <div style="background-color: #ffffff; 
                                color: #000000;
                                padding: 20px; 
                                border-radius: 8px; 
                                border-left: 4px solid #667eea;
                                margin-top: 10px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <p style="color: #000000; margin: 0;">{answer_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation buttons
        col_prev, col_spacer, col_next = st.columns([2, 1, 2])
        
        with col_prev:
            if st.button("⬅️ Previous Step", 
                        disabled=st.session_state.current_step == 0, 
                        key=f"prev_step_{st.session_state.current_step}",
                        use_container_width=True):
                st.session_state.current_step = max(0, st.session_state.current_step - 1)
                st.rerun()
        
        with col_next:
            if st.button("Next Step ➡️", 
                        disabled=st.session_state.current_step >= len(steps) - 1, 
                        key=f"next_step_{st.session_state.current_step}",
                        use_container_width=True):
                st.session_state.current_step = min(len(steps) - 1, st.session_state.current_step + 1)
                st.rerun()
        
        # Step indicator
        st.markdown(f"""
        <div style="text-align: center; margin-top: 20px; color: #666;">
            📊 Step {st.session_state.current_step + 1} of {len(steps)}
        </div>
        """, unsafe_allow_html=True)
        
        # Show completion message if on last step
        if st.session_state.current_step == len(steps) - 1:
            st.balloons()
            st.success("🎉 Congratulations! You've completed the interactive lesson!")
    
    return len(steps)


def render_feedback_buttons(mode: str):
    """Render like/dislike feedback buttons"""
    st.markdown("---")
    st.markdown("### How was this content?")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("👍 Like", key=f"like_{mode}", use_container_width=True):
            give_feedback(mode, 1.0)
    
    with col2:
        if st.button("👎 Dislike", key=f"dislike_{mode}", use_container_width=True):
            give_feedback(mode, 0.0)
    
    with col3:
        if st.button("😐 Neutral", key=f"neutral_{mode}", use_container_width=True):
            give_feedback(mode, 0.5)


def give_feedback(mode: str, feedback: float):
    """Send feedback to RL agent"""
    result = st.session_state.manager.process_user_request(
        "update_rl",
        {
            "mode": mode,
            "feedback": feedback
        },
        st.session_state.session_id
    )
    
    if result.get("success"):
        st.success("✅ Feedback recorded! The system is learning your preferences.")
        st.session_state.feedback_given = True
    else:
        st.error(f"Error recording feedback: {result.get('error')}")


def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.title("🧠 Learning Platform")
        
        # Check survey status
        state = load_state()
        survey_completed = state.survey_completed
        
        if not survey_completed:
            st.info("📋 Complete the survey to get started!")
        else:
            # Show current preference
            preference = state.initial_preference
            if preference:
                pref_display = {
                    "quiz": "📝 Quizzes",
                    "flashcard": "🃏 Flashcards",
                    "interactive": "🎯 Interactive",
                    "i_dont_know": "❓ Exploring"
                }.get(preference, preference)
                st.success(f"Current preference: {pref_display}")
            
            # Get RL recommendation
            rec_result = st.session_state.manager.process_user_request(
                "recommend",
                {},
                st.session_state.session_id
            )
            
            if rec_result.get("success"):
                recommended = rec_result.get("recommended_mode", "quiz")
                rec_display = {
                    "quiz": "📝 Quizzes",
                    "flashcard": "🃏 Flashcards",
                    "interactive": "🎯 Interactive"
                }.get(recommended, recommended)
                st.info(f"💡 Recommended: {rec_display}")
        
        st.divider()
        
        # File upload
        st.subheader("📄 Upload Material")
        uploaded_file = st.file_uploader(
            "Upload PDF or text file",
            type=["pdf", "txt", "md"],
            key="file_uploader"
        )
        
        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            
            if st.button("📥 Process File", type="primary"):
                with st.spinner("Extracting text..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    # Extract text
                    extract_result = st.session_state.manager.process_user_request(
                        "extract",
                        {
                            "file_path": tmp_path,
                            "file_type": Path(uploaded_file.name).suffix[1:]
                        },
                        st.session_state.session_id
                    )
                    
                    if extract_result.get("success"):
                        extracted_chunks = extract_result.get("chunks", [])
                        # Filter out empty chunks
                        extracted_chunks = [chunk for chunk in extracted_chunks if chunk and chunk.strip()]
                        st.session_state.extracted_chunks = extracted_chunks
                        st.success(f"✅ Extracted {len(st.session_state.extracted_chunks)} chunks")
                        
                        if not st.session_state.extracted_chunks:
                            st.error("⚠️ No valid content chunks extracted. Please try a different file.")
                        else:
                            # Auto-generate content based on preference
                            if survey_completed and preference != LearningMode.UNKNOWN.value:
                                generate_content_for_mode(preference)
                            else:
                                generate_mixed_bundle()
                    else:
                        st.error(f"Error: {extract_result.get('error')}")
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
        
        st.divider()
        
        # Reset preferences button
        if st.button("🔄 Reset Preferences", use_container_width=True):
            reset_result = st.session_state.manager.process_user_request(
                "reset_preferences",
                {},
                st.session_state.session_id
            )
            
            if reset_result.get("success"):
                st.success("✅ Preferences reset!")
                st.rerun()
        
        # Show logs
        with st.expander("📊 View Logs"):
            logs = logger.get_streamlit_logs()
            if logs:
                st.text("\n".join(logs[-20:]))  # Last 20 logs
            else:
                st.text("No logs yet")


    # Main area
    if not survey_completed:
        render_survey()
    else:
        if st.session_state.extracted_chunks is None:
            st.info("👆 Upload a file in the sidebar to get started!")
        elif st.session_state.generated_content is None:
            st.info("📝 Content will appear here after processing your file.")
        else:
            # Display generated content
            content = st.session_state.generated_content
            content_type = content.get("content_type", "")
            
            if content_type == ContentType.MIXED.value:
                # Show mixed bundle in tabs
                tab_quiz, tab_flash, tab_inter = st.tabs(["📝 Quiz", "🃏 Flashcards", "🎯 Interactive"])
                
                with tab_quiz:
                    quiz_data = content.get("data", {}).get("quiz", {})
                    if quiz_data:
                        render_quiz_content(quiz_data)
                        render_feedback_buttons("quiz")
                
                with tab_flash:
                    flashcard_data = content.get("data", {}).get("flashcards", {})
                    if flashcard_data:
                        render_flashcard_content(flashcard_data)
                        render_feedback_buttons("flashcard")
                
                with tab_inter:
                    interactive_data = content.get("data", {}).get("interactive", {})
                    if interactive_data:
                        render_interactive_content(interactive_data)
                        render_feedback_buttons("interactive")
            
            elif content_type == ContentType.QUIZ.value:
                render_quiz_content(content.get("data", {}))
                render_feedback_buttons("quiz")
            
            elif content_type == ContentType.FLASHCARD.value:
                render_flashcard_content(content.get("data", {}))
                render_feedback_buttons("flashcard")
            
            elif content_type == ContentType.INTERACTIVE.value:
                render_interactive_content(content.get("data", {}))
                render_feedback_buttons("interactive")


def generate_content_for_mode(mode: str):
    """Generate content for a specific mode"""
    with st.spinner(f"Generating {mode} content..."):
        # Pass chunks directly as fallback if available in session state
        params = {
            "content_type": mode,
            "session_id": st.session_state.session_id
        }
        # Add chunks from session state as fallback AND primary source
        if st.session_state.extracted_chunks:
            params["extracted_chunks"] = st.session_state.extracted_chunks
            params["chunks"] = st.session_state.extracted_chunks  # Also try the standard key
            logger.get_logger().info(f"Passing {len(st.session_state.extracted_chunks)} chunks to generate function")
        else:
            logger.get_logger().warning("No extracted_chunks in session state!")
        
        result = st.session_state.manager.process_user_request(
            "generate",
            params,
            st.session_state.session_id
        )
        
        if result.get("success"):
            # Reset quiz state when new content is generated
            if "quiz_shuffled" in st.session_state:
                st.session_state.quiz_shuffled = {}
            if "quiz_answers" in st.session_state:
                st.session_state.quiz_answers = {}
            if "quiz_submitted" in st.session_state:
                st.session_state.quiz_submitted = {}
            # Reset flashcard state
            if "current_card" in st.session_state:
                st.session_state.current_card = 0
            if "flipped" in st.session_state:
                st.session_state.flipped = {}
            # Reset interactive state
            if "current_step" in st.session_state:
                st.session_state.current_step = 0
            if "checkpoint_responses" in st.session_state:
                st.session_state.checkpoint_responses = {}
            
            st.session_state.generated_content = result
            st.session_state.current_mode = mode
            st.rerun()
        else:
            st.error(f"Error generating content: {result.get('error')}")


def generate_mixed_bundle():
    """Generate mixed bundle with all content types"""
    with st.spinner("Generating mixed content bundle..."):
        # Pass chunks directly as fallback if available in session state
        params = {
            "content_type": ContentType.MIXED.value,
            "session_id": st.session_state.session_id
        }
        # Add chunks from session state as fallback AND primary source
        if st.session_state.extracted_chunks:
            params["extracted_chunks"] = st.session_state.extracted_chunks
            params["chunks"] = st.session_state.extracted_chunks  # Also try the standard key
            logger.get_logger().info(f"Passing {len(st.session_state.extracted_chunks)} chunks to generate_mixed_bundle")
        else:
            logger.get_logger().warning("No extracted_chunks in session state for mixed bundle!")
        
        result = st.session_state.manager.process_user_request(
            "generate",
            params,
            st.session_state.session_id
        )
        
        if result.get("success"):
            # Reset quiz state when new content is generated
            if "quiz_shuffled" in st.session_state:
                st.session_state.quiz_shuffled = {}
            if "quiz_answers" in st.session_state:
                st.session_state.quiz_answers = {}
            if "quiz_submitted" in st.session_state:
                st.session_state.quiz_submitted = {}
            # Reset flashcard state
            if "current_card" in st.session_state:
                st.session_state.current_card = 0
            if "flipped" in st.session_state:
                st.session_state.flipped = {}
            # Reset interactive state
            if "current_step" in st.session_state:
                st.session_state.current_step = 0
            if "checkpoint_responses" in st.session_state:
                st.session_state.checkpoint_responses = {}
            
            st.session_state.generated_content = result
            st.session_state.current_mode = "mixed"
            st.rerun()
        else:
            st.error(f"Error generating content: {result.get('error')}")


if __name__ == "__main__":
    main()

