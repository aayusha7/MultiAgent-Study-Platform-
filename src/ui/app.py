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
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

if "manager" not in st.session_state or st.session_state.get("username") != st.session_state.get("_last_username"):
    # Reinitialize manager when username changes
    username = st.session_state.get("username")
    st.session_state.manager = ManagerAgent(username=username)
    st.session_state._last_username = username
    st.session_state.session_id = f"session_{os.urandom(4).hex()}"
    st.session_state.uploaded_file = None
    st.session_state.extracted_chunks = None
    st.session_state.current_filename = None  # Track current file name
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


def render_login():
    """Render login/registration page"""
    st.title("🧠 Learning Platform")
    st.markdown("### Welcome! Please login or create an account")
    
    tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab_login:
        st.markdown("### Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_login = st.form_submit_button("Login", type="primary", use_container_width=True)
            
            if submit_login:
                if username and password:
                    from src.core.auth import authenticate_user
                    success, user, message = authenticate_user(username, password)
                    
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter both username and password")
    
    with tab_register:
        st.markdown("### Create a New Account")
        with st.form("register_form"):
            new_username = st.text_input("Username", placeholder="Choose a username (min 3 characters)", key="reg_username")
            new_email = st.text_input("Email", placeholder="Enter your email", key="reg_email")
            new_password = st.text_input("Password", type="password", placeholder="Choose a password (min 6 characters)", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="reg_confirm")
            submit_register = st.form_submit_button("Create Account", type="primary", use_container_width=True)
            
            if submit_register:
                if new_username and new_email and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("Passwords do not match. Please try again.")
                    else:
                        from src.core.auth import register_user
                        success, message = register_user(new_username, new_email, new_password)
                        
                        if success:
                            st.success(message)
                            st.info("Please switch to the Login tab to sign in.")
                        else:
                            st.error(message)
                else:
                    st.warning("Please fill in all fields")


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
            tracking_key = f"quiz_tracked_{i}"
            
            if st.button("Submit Answer", key=submit_key):
                st.session_state.quiz_submitted[i] = True
                # Track performance immediately on submit (only once)
                if "source_reference" in q and not st.session_state.get(tracking_key, False):
                    from src.core.analytics import record_quiz_answer, extract_chunk_id_from_reference
                    user_selected_idx = shuffled_options.index(st.session_state.quiz_answers[answer_key]) if st.session_state.quiz_answers[answer_key] in shuffled_options else -1
                    correct_idx = shuffled_data["correct_idx"]
                    is_correct = user_selected_idx == correct_idx
                    
                    source_ref = q.get("source_reference", "")
                    filename = st.session_state.get("current_filename", "unknown_file")
                    chunk_id = extract_chunk_id_from_reference(source_ref, filename=filename)
                    record_quiz_answer(
                        chunk_id=chunk_id,
                        source_reference=source_ref,
                        is_correct=is_correct,
                        question_text=q.get("question", ""),
                        username=st.session_state.get("username"),
                        filename=filename
                    )
                    st.session_state[tracking_key] = True
                st.rerun()
            
            # Show result if submitted
            if st.session_state.quiz_submitted.get(i, False):
                user_selected_idx = shuffled_options.index(st.session_state.quiz_answers[answer_key]) if st.session_state.quiz_answers[answer_key] in shuffled_options else -1
                correct_idx = shuffled_data["correct_idx"]
                is_correct = user_selected_idx == correct_idx
                
                # Show result inline without popup notifications
                if is_correct:
                    st.markdown("**✅ Correct!**")
                else:
                    st.markdown(f"**❌ Incorrect.** The correct answer is: **{shuffled_options[correct_idx]}**")
                
                if "explanation" in q:
                    with st.expander("💡 Explanation"):
                        st.markdown(q["explanation"])
                
                # Show source reference
                if "source_reference" in q:
                    st.markdown(f"📄 **Source:** {q['source_reference']}")
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
    
    # Initialize checkpoint submission state
    if "checkpoint_submitted" not in st.session_state:
        st.session_state.checkpoint_submitted = {}
    
    # Initialize checkpoint correctness state
    if "checkpoint_correct" not in st.session_state:
        st.session_state.checkpoint_correct = {}
    
    # Initialize stars/rewards count
    if "interactive_stars" not in st.session_state:
        st.session_state.interactive_stars = 0
    
    # Progress bar
    progress = (st.session_state.current_step + 1) / len(steps)
    st.progress(progress, text=f"Progress: Step {st.session_state.current_step + 1} of {len(steps)}")
    
    # Title with stars display
    col_title, col_stars = st.columns([3, 1])
    with col_title:
        st.markdown(f"# 🎯 {title}")
    with col_stars:
        stars_display = "⭐" * min(5, st.session_state.interactive_stars)
        st.markdown(f"""
        <div style="text-align: right; padding: 10px;">
            <div style="font-size: 20px; color: #ffd700;">{stars_display}</div>
            <div style="font-size: 12px; color: #666;">Total: {st.session_state.interactive_stars} stars</div>
        </div>
        """, unsafe_allow_html=True)
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
            is_submitted = st.session_state.checkpoint_submitted.get(checkpoint_key, False)
            is_correct = st.session_state.checkpoint_correct.get(checkpoint_key, False)
            
            checkpoint_response = st.text_area(
                "📝 Your response:",
                key=checkpoint_key,
                height=150,
                placeholder="Type your thoughts, answer, or reflection here...",
                help="Take a moment to reflect on what you've learned",
                disabled=is_submitted  # Disable after submission
            )
            
            # Save response
            if checkpoint_response and not is_submitted:
                st.session_state.checkpoint_responses[checkpoint_key] = checkpoint_response
            
            # Submit button
            if not is_submitted:
                tracking_key = f"interactive_tracked_{st.session_state.current_step}"
                if st.button("📤 Submit Answer", key=f"submit_checkpoint_{st.session_state.current_step}", 
                           type="primary", use_container_width=True, disabled=not checkpoint_response.strip()):
                    # Check if answer is correct
                    user_answer = checkpoint_response.strip().lower()
                    correct_answer = step.get("checkpoint_answer", "").strip().lower()
                    
                    # Simple similarity check (can be improved with semantic similarity)
                    # Check if key concepts from correct answer are in user answer
                    correct_words = set(correct_answer.split())
                    user_words = set(user_answer.split())
                    
                    # Calculate similarity (at least 30% of key words should match)
                    if len(correct_words) > 0:
                        similarity = len(correct_words.intersection(user_words)) / len(correct_words)
                        is_correct = similarity >= 0.3 or user_answer in correct_answer or correct_answer in user_answer
                    else:
                        is_correct = False
                    
                    # Mark as submitted
                    st.session_state.checkpoint_submitted[checkpoint_key] = True
                    st.session_state.checkpoint_correct[checkpoint_key] = is_correct
                    
                    # Track performance for analytics (only once per submission)
                    if "source_reference" in step and not st.session_state.get(tracking_key, False):
                        from src.core.analytics import record_quiz_answer, extract_chunk_id_from_reference
                        source_ref = step.get("source_reference", "")
                        filename = st.session_state.get("current_filename", "unknown_file")
                        chunk_id = extract_chunk_id_from_reference(source_ref, filename=filename)
                        record_quiz_answer(
                            chunk_id=chunk_id,
                            source_reference=source_ref,
                            is_correct=is_correct,
                            question_text=f"Interactive Checkpoint: {step.get('checkpoint', '')[:50]}",
                            username=st.session_state.get("username"),
                            filename=filename
                        )
                        st.session_state[tracking_key] = True
                    
                    # Award stars if correct
                    if is_correct:
                        st.session_state.interactive_stars += 1
                    
                    st.rerun()
                
            # Show result after submission
            if is_submitted:
                if is_correct:
                    # Show rewards/stars for correct answer
                    stars_earned = "⭐" * min(3, st.session_state.interactive_stars)
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 20px; 
                                border-radius: 10px; 
                                color: white; 
                                text-align: center;
                                margin: 20px 0;">
                        <h2 style="color: white; margin: 0;">🎉 Correct! Well done!</h2>
                        <p style="color: white; font-size: 24px; margin: 10px 0;">{stars_earned}</p>
                        <p style="color: white; margin: 0;">You've earned a star! Total stars: {st.session_state.interactive_stars} ⭐</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("✅ Great job! Keep it up!")
                else:
                    # Show correct answer if wrong
                    st.error("❌ Not quite right. Here's the correct answer:")
                    answer_text = html.escape(step.get("checkpoint_answer", "No answer provided"))
                    st.markdown(f"""
                    <div style="background-color: #fff3cd; 
                                color: #856404;
                                padding: 20px; 
                                border-radius: 8px; 
                                border-left: 4px solid #ffc107;
                                margin-top: 10px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h4 style="color: #856404; margin-top: 0;">💡 Correct Answer:</h4>
                        <p style="color: #856404; margin: 0;">{answer_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show user's answer for comparison
                    user_answer_display = html.escape(st.session_state.checkpoint_responses.get(checkpoint_key, ""))
                    if user_answer_display:
                        st.info(f"📝 Your answer: {user_answer_display}")
            
            # Show previous response if exists and not submitted
            elif checkpoint_key in st.session_state.checkpoint_responses:
                st.info(f"💭 Your previous response: {st.session_state.checkpoint_responses[checkpoint_key][:100]}...")
        
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
            st.markdown("### 🎉 Congratulations! You've completed the interactive lesson!")
    
    return len(steps)


def render_performance_heatmap(all_perf: Dict[str, Dict[str, Any]]):
    """Render a heatmap showing performance per chunk"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Plotly not installed. Install with: pip install plotly")
        # Fallback to simple text display
        # Import format_topic_name for user-friendly labels
        from src.core.analytics import format_topic_name
        
        st.write("**Performance by Area:**")
        for chunk_id, perf in sorted(all_perf.items()):
            if perf["attempts"] > 0:
                accuracy_color = "🟢" if perf["accuracy"] >= 80 else "🟡" if perf["accuracy"] >= 60 else "🔴"
                ref = perf.get("source_reference", chunk_id)
                topic_name = format_topic_name(ref, max_length=50)
                st.write(f"{accuracy_color} {topic_name}: {perf['accuracy']:.1f}% ({perf['attempts']} questions)")
        return
    
    if not all_perf:
        st.info("No performance data available yet.")
        return
    
    # Import format_topic_name for user-friendly labels
    from src.core.analytics import format_topic_name
    
    # Prepare data for heatmap
    chunk_ids = []
    accuracies = []
    attempts = []
    labels = []
    
    for chunk_id, perf in sorted(all_perf.items()):
        if perf["attempts"] > 0:
            chunk_ids.append(chunk_id)
            accuracies.append(perf["accuracy"])
            attempts.append(perf["attempts"])
            # Create user-friendly label using format_topic_name
            ref = perf.get("source_reference", chunk_id)
            topic_name = format_topic_name(ref, max_length=30)
            labels.append(f"{topic_name}\n{perf['accuracy']:.0f}% ({perf['attempts']} questions)")
    
    if not chunk_ids:
        st.info("No performance data available yet.")
        return
    
    # Create heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=[accuracies],  # Single row heatmap
        x=chunk_ids,
        y=["Performance"],
        colorscale=[
            [0, '#d73027'],      # Red for low (0%)
            [0.5, '#fee08b'],    # Yellow for medium (50%)
            [1, '#1a9850']       # Green for high (100%)
        ],
        text=[[f"{acc:.1f}%" for acc in accuracies]],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Accuracy %"),
        hovertemplate='<b>%{x}</b><br>Accuracy: %{z:.1f}%<br>Attempts: %{customdata[0]}<extra></extra>',
        customdata=[[attempts]]
    ))
    
    fig.update_layout(
        title="Performance Heatmap by Content Area",
        xaxis_title="Content Areas",
        yaxis_title="",
        height=200,
        xaxis=dict(tickangle=-45, tickmode='array', tickvals=chunk_ids, ticktext=[l.split('\n')[0] for l in labels]),
        margin=dict(l=20, r=20, t=50, b=100)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Also show a bar chart for better readability
    st.subheader("📊 Performance by Area")
    fig2 = go.Figure(data=[
        go.Bar(
            x=[l.split('\n')[0] for l in labels],
            y=accuracies,
            marker=dict(
                color=accuracies,
                colorscale=[
                    [0, '#d73027'],      # Red
                    [0.5, '#fee08b'],    # Yellow
                    [1, '#1a9850']       # Green
                ],
                showscale=True,
                colorbar=dict(title="Accuracy %")
            ),
            text=[f"{acc:.1f}%" for acc in accuracies],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.1f}%<br>Attempts: %{customdata}<extra></extra>',
            customdata=attempts
        )
    ])
    
    fig2.update_layout(
        xaxis_title="Content Areas",
        yaxis_title="Accuracy (%)",
        height=400,
        xaxis=dict(tickangle=-45),
        margin=dict(b=100)
    )
    
    st.plotly_chart(fig2, use_container_width=True)


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
    
    # Check authentication
    if not st.session_state.authenticated:
        render_login()
        return
    
    # Sidebar
    with st.sidebar:
        # User info and logout
        st.markdown(f"### 👤 {st.session_state.username}")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            # Clear all session data including _last_username to force reinit on next login
            for key in list(st.session_state.keys()):
                if key not in ['authenticated', 'username']:
                    del st.session_state[key]
            # Explicitly clear _last_username to ensure ManagerAgent reinitializes
            if '_last_username' in st.session_state:
                del st.session_state['_last_username']
            st.rerun()
        st.divider()
        st.title("🧠 Learning Platform")
        
        # Check survey status
        state = load_state(st.session_state.username)
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
                        st.session_state.current_filename = uploaded_file.name  # Store filename
                        
                        # Register file for analytics (create file hash -> filename mapping)
                        from src.core.analytics import register_file
                        register_file(uploaded_file.name, username=st.session_state.get("username"))
                        
                        st.success(f"✅ Extracted {len(st.session_state.extracted_chunks)} chunks from {uploaded_file.name}")
                        
                        if not st.session_state.extracted_chunks:
                            st.error("⚠️ No valid content chunks extracted. Please try a different file.")
                        else:
                            # Auto-generate content based on preference
                            if survey_completed:
                                # If preference is "I don't know", always show mixed bundle
                                if preference == LearningMode.UNKNOWN.value:
                                    generate_mixed_bundle()
                                else:
                                    # For specific preferences, use that mode
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
        
        # Analytics Dashboard link in sidebar
        st.markdown("---")
        if st.button("📊 View Analytics Dashboard", use_container_width=True):
            st.session_state.show_analytics = True
            st.rerun()
        
        # Show logs
        with st.expander("📋 View Logs"):
            logs = logger.get_streamlit_logs()
            if logs:
                st.text("\n".join(logs[-20:]))  # Last 20 logs
            else:
                st.text("No logs yet")


    # Main area
    # Check if user wants to see analytics
    if st.session_state.get("show_analytics", False):
        # Analytics Dashboard - Full Page
        st.title("📊 Analytics & Progress Dashboard")
        
        from src.core.analytics import (
            get_performance_summary,
            get_weak_areas,
            get_strong_areas,
            get_all_chunk_performance,
            format_topic_name,
            group_chunks_by_file
        )
        
        # Ensure username is available (should always be set if authenticated)
        username = st.session_state.get("username")
        if not username:
            st.error("❌ Username not found. Please log out and log back in.")
            if st.button("← Back to Learning", type="primary"):
                st.session_state.show_analytics = False
                st.rerun()
            return
        
        # Verify state file exists and can be loaded
        from src.core.memory import get_state_path
        state_path = get_state_path(username)
        if not state_path.exists():
            st.warning(f"⚠️ No state file found for user '{username}'. Your analytics data will appear once you start answering questions.")
        else:
            # Try to load state to verify it's accessible
            try:
                test_state = load_state(username)
                if not hasattr(test_state, 'chunk_performance') or not test_state.chunk_performance:
                    st.info("ℹ️ No analytics data recorded yet. Start answering questions to see your progress!")
            except Exception as e:
                st.error(f"❌ Error loading state: {e}")
        
        summary = get_performance_summary(username)
        
        if st.button("← Back to Learning", type="primary"):
            st.session_state.show_analytics = False
            st.rerun()
        
        st.divider()
        
        if summary["total_attempts"] > 0:
            st.header("📈 Overall Performance")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Accuracy", f"{summary['overall_accuracy']:.1f}%")
            with col2:
                st.metric("Total Questions", summary["total_attempts"])
            with col3:
                st.metric("Correct", summary["total_correct"])
            with col4:
                st.metric("Areas Tested", summary["chunks_with_data"])
            
            st.divider()
            
            # Heatmap
            st.header("🔥 Performance Heatmap")
            all_perf = get_all_chunk_performance(username)
            if all_perf:
                render_performance_heatmap(all_perf)
            else:
                st.info("No performance data available yet.")
            
            st.divider()
            
            # Performance by File
            st.header("📊 Performance by File")
            if all_perf:
                # Group chunks by file
                files_data = group_chunks_by_file(all_perf, username=username)
                
                # Sort files by last attempt (most recent first) or by accuracy
                sorted_files = sorted(
                    files_data.items(),
                    key=lambda x: (
                        x[1].get("last_attempt") or "",
                        -x[1].get("accuracy", 0)
                    ),
                    reverse=True
                )
                
                for file_hash, file_data in sorted_files:
                    file_accuracy = file_data.get("accuracy", 0.0)
                    file_attempts = file_data.get("total_attempts", 0)
                    chunks_count = len(file_data.get("chunks", {}))
                    chunks_with_data = file_data.get("chunks_with_data", 0)
                    
                    # Color code based on file-level performance
                    if file_accuracy >= 80:
                        emoji = "🟢"
                        status = "Strong"
                    elif file_accuracy >= 60:
                        emoji = "🟡"
                        status = "Moderate"
                    else:
                        emoji = "🔴"
                        status = "Needs Practice"
                    
                    # Get filename for display (prefer stored filename, fallback to hash)
                    display_filename = file_data.get("filename")
                    if not display_filename or display_filename == "unknown_file":
                        # Try to extract meaningful name from chunks
                        from src.core.analytics import extract_file_name_from_chunks
                        chunks = file_data.get("chunks", {})
                        extracted_name = extract_file_name_from_chunks(file_hash, chunks)
                        if extracted_name:
                            display_filename = extracted_name
                        else:
                            display_filename = f"File {file_hash[:8]}..."
                    else:
                        # Show just the filename, truncate if too long
                        from pathlib import Path
                        display_filename = Path(display_filename).name
                        if len(display_filename) > 40:
                            display_filename = display_filename[:37] + "..."
                    
                    # Create file-level display text
                    file_display = f"{emoji} {display_filename} - {file_accuracy:.1f}% ({file_attempts} question{'s' if file_attempts != 1 else ''}, {chunks_with_data} section{'s' if chunks_with_data != 1 else ''})"
                    
                    with st.expander(file_display, expanded=False):
                        st.markdown(f"**File Status:** {status}")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("File Accuracy", f"{file_accuracy:.1f}%")
                        with col2:
                            st.metric("Total Questions", file_attempts)
                        with col3:
                            st.metric("Correct", file_data.get("total_correct", 0))
                        with col4:
                            st.metric("Sections Tested", chunks_with_data)
                        
                        st.write(f"**Incorrect:** {file_data.get('total_incorrect', 0)}")
                        if file_data.get("last_attempt"):
                            from datetime import datetime
                            try:
                                last_attempt_dt = datetime.fromisoformat(file_data['last_attempt'].replace('Z', '+00:00'))
                                st.write(f"**Last Attempt:** {last_attempt_dt.strftime('%Y-%m-%d %H:%M')}")
                            except:
                                st.write(f"**Last Attempt:** {file_data['last_attempt']}")
                        
                        st.divider()
                        
                        # Show chunks within this file
                        st.subheader(f"📑 Sections in this File ({chunks_count} total)")
                        chunks = file_data.get("chunks", {})
                        
                        if chunks:
                            # Sort chunks by chunk number if possible
                            sorted_chunks = sorted(
                                chunks.items(),
                                key=lambda x: (
                                    int(x[0].split("_chunk_")[1]) if "_chunk_" in x[0] and x[0].split("_chunk_")[1].isdigit() else 999,
                                    -x[1].get("accuracy", 0)
                                )
                            )
                            
                            for chunk_id, chunk_perf in sorted_chunks:
                                chunk_accuracy = chunk_perf.get("accuracy", 0.0)
                                chunk_attempts = chunk_perf.get("attempts", 0)
                                source_ref = chunk_perf.get("source_reference", chunk_id)
                                
                                # Format topic name for user-friendly display
                                topic_name = format_topic_name(source_ref, max_length=45)
                                
                                # Color code based on chunk performance
                                if chunk_accuracy >= 80:
                                    chunk_emoji = "🟢"
                                    chunk_status = "Strong"
                                elif chunk_accuracy >= 60:
                                    chunk_emoji = "🟡"
                                    chunk_status = "Moderate"
                                else:
                                    chunk_emoji = "🔴"
                                    chunk_status = "Needs Practice"
                                
                                # Only show chunks with attempts
                                if chunk_attempts > 0:
                                    chunk_display = f"{chunk_emoji} {topic_name} - {chunk_accuracy:.1f}% ({chunk_attempts} question{'s' if chunk_attempts != 1 else ''})"
                                    
                                    with st.expander(chunk_display, expanded=False):
                                        st.markdown(f"**Status:** {chunk_status}")
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Accuracy", f"{chunk_accuracy:.1f}%")
                                        with col2:
                                            st.metric("Questions", chunk_attempts)
                                        with col3:
                                            st.metric("Correct", chunk_perf.get("correct", 0))
                                        st.write(f"**Incorrect:** {chunk_perf.get('incorrect', 0)}")
                                        if chunk_perf.get("last_attempt"):
                                            from datetime import datetime
                                            try:
                                                last_attempt_dt = datetime.fromisoformat(chunk_perf['last_attempt'].replace('Z', '+00:00'))
                                                st.write(f"**Last Attempt:** {last_attempt_dt.strftime('%Y-%m-%d %H:%M')}")
                                            except:
                                                st.write(f"**Last Attempt:** {chunk_perf['last_attempt']}")
                                        with st.expander("📄 View Source Reference", expanded=False):
                                            st.text(source_ref)
                        else:
                            st.info("No sections tested in this file yet.")
            
            st.divider()
            
            # Weak Areas
            st.header("⚠️ Weak Areas (Need Improvement)")
            weak_areas = get_weak_areas(threshold=60.0, min_attempts=2, username=username)
            if weak_areas:
                for area in weak_areas[:10]:  # Show top 10 weakest
                    topic_name = format_topic_name(area['source_reference'], max_length=55)
                    with st.expander(f"🔴 {topic_name} - {area['accuracy']:.1f}% accuracy"):
                        st.markdown("**Status:** Needs Practice")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{area['accuracy']:.1f}%")
                        with col2:
                            st.metric("Questions", area['attempts'])
                        with col3:
                            st.metric("Correct", area['correct'])
                        st.write(f"**Incorrect:** {area['incorrect']}")
                        if area.get('last_attempt'):
                            from datetime import datetime
                            try:
                                last_attempt_dt = datetime.fromisoformat(area['last_attempt'].replace('Z', '+00:00'))
                                st.write(f"**Last Attempt:** {last_attempt_dt.strftime('%Y-%m-%d %H:%M')}")
                            except:
                                st.write(f"**Last Attempt:** {area['last_attempt']}")
                        with st.expander("📄 View Source Reference", expanded=False):
                            st.text(area['source_reference'])
            else:
                st.info("No weak areas identified yet. Keep practicing!")
            
            st.divider()
            
            # Strong Areas
            st.header("✅ Strong Areas (Mastered)")
            strong_areas = get_strong_areas(threshold=80.0, min_attempts=2, username=username)
            if strong_areas:
                for area in strong_areas[:10]:  # Show top 10 strongest
                    topic_name = format_topic_name(area['source_reference'], max_length=55)
                    with st.expander(f"🟢 {topic_name} - {area['accuracy']:.1f}% accuracy"):
                        st.markdown("**Status:** Strong")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{area['accuracy']:.1f}%")
                        with col2:
                            st.metric("Questions", area['attempts'])
                        with col3:
                            st.metric("Correct", area['correct'])
                        st.write(f"**Incorrect:** {area['incorrect']}")
                        if area.get('last_attempt'):
                            from datetime import datetime
                            try:
                                last_attempt_dt = datetime.fromisoformat(area['last_attempt'].replace('Z', '+00:00'))
                                st.write(f"**Last Attempt:** {last_attempt_dt.strftime('%Y-%m-%d %H:%M')}")
                            except:
                                st.write(f"**Last Attempt:** {area['last_attempt']}")
                        with st.expander("📄 View Source Reference", expanded=False):
                            st.text(area['source_reference'])
            else:
                st.info("No strong areas identified yet. Keep practicing!")
        else:
            st.info("📝 Complete some quizzes to see your progress and analytics!")
            if st.button("← Back to Learning"):
                st.session_state.show_analytics = False
                st.rerun()
    
    elif not survey_completed:
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
                    if quiz_data and quiz_data.get("questions"):
                        render_quiz_content(quiz_data)
                        render_feedback_buttons("quiz")
                    else:
                        st.info("📝 Quiz content will be generated here. Try generating quiz content specifically or wait for it to be included in the mixed bundle.")
                
                with tab_flash:
                    flashcard_data = content.get("data", {}).get("flashcards", {})
                    if flashcard_data and flashcard_data.get("cards"):
                        render_flashcard_content(flashcard_data)
                        render_feedback_buttons("flashcard")
                    else:
                        st.info("🃏 Flashcard content will be generated here. Try generating flashcard content specifically or wait for it to be included in the mixed bundle.")
                
                with tab_inter:
                    interactive_data = content.get("data", {}).get("interactive", {})
                    if interactive_data and interactive_data.get("steps"):
                        render_interactive_content(interactive_data)
                        render_feedback_buttons("interactive")
                    else:
                        st.info("🎯 Interactive content will be generated here. Try generating interactive content specifically or wait for it to be included in the mixed bundle.")
            
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
            if "checkpoint_submitted" in st.session_state:
                st.session_state.checkpoint_submitted = {}
            if "checkpoint_correct" in st.session_state:
                st.session_state.checkpoint_correct = {}
            # Reset stars when starting new content (optional - comment out if you want stars to persist)
            # if "interactive_stars" in st.session_state:
            #     st.session_state.interactive_stars = 0
            
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
            if "checkpoint_submitted" in st.session_state:
                st.session_state.checkpoint_submitted = {}
            if "checkpoint_correct" in st.session_state:
                st.session_state.checkpoint_correct = {}
            # Reset stars when starting new content (optional - comment out if you want stars to persist)
            # if "interactive_stars" in st.session_state:
            #     st.session_state.interactive_stars = 0
            
            st.session_state.generated_content = result
            st.session_state.current_mode = "mixed"
            st.rerun()
        else:
            st.error(f"Error generating content: {result.get('error')}")


if __name__ == "__main__":
    main()

