"""Supabase client for persistent data storage"""

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from .logger import logger

# Try to import Supabase, but make it optional
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.get_logger().warning("Supabase not installed. Install with: pip install supabase")


def get_supabase_client() -> Optional[Any]:
    """Get Supabase client if configured"""
    if not SUPABASE_AVAILABLE:
        return None
    
    try:
        supabase_url = None
        supabase_key = None
        
        # Try to get from Streamlit secrets first
        try:
            import streamlit as st
            secrets = st.secrets
            supabase_url = secrets.get("SUPABASE_URL")
            supabase_key = secrets.get("SUPABASE_KEY")
        except (ImportError, AttributeError, KeyError, FileNotFoundError, RuntimeError):
            # Streamlit not available or secrets not configured
            pass
        
        # Fall back to environment variables
        if not supabase_url:
            supabase_url = os.getenv("SUPABASE_URL")
        if not supabase_key:
            supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            return None
        
        client = create_client(supabase_url, supabase_key)
        return client
    except Exception as e:
        logger.get_logger().debug(f"Supabase not configured: {e}")
        return None


def init_supabase_tables(client: Any) -> bool:
    """Initialize Supabase tables (run SQL in Supabase dashboard)"""
    # Note: Tables need to be created manually in Supabase dashboard
    # This function just verifies connection
    try:
        # Test connection by querying (will fail if tables don't exist, which is expected)
        result = client.table("users").select("username").limit(1).execute()
        return True
    except Exception as e:
        logger.get_logger().debug(f"Supabase tables may not exist yet: {e}")
        return False


def save_user_supabase(username: str, password_hash: str, email: str = "") -> bool:
    """Save user to Supabase"""
    client = get_supabase_client()
    if not client:
        return False
    
    try:
        client.table("users").upsert({
            "username": username,
            "password_hash": password_hash,
            "email": email,
            "created_at": datetime.now().isoformat()
        }).execute()
        logger.get_logger().info(f"Saved user {username} to Supabase")
        return True
    except Exception as e:
        logger.get_logger().error(f"Error saving user to Supabase: {e}")
        return False


def get_user_supabase(username: str) -> Optional[Dict[str, Any]]:
    """Get user from Supabase"""
    client = get_supabase_client()
    if not client:
        return None
    
    try:
        result = client.table("users").select("*").eq("username", username).execute()
        if result.data and len(result.data) > 0:
            return result.data[0]
        return None
    except Exception as e:
        logger.get_logger().error(f"Error getting user from Supabase: {e}")
        return None


def get_all_users_supabase() -> Dict[str, Dict[str, Any]]:
    """Get all users from Supabase"""
    client = get_supabase_client()
    if not client:
        return {}
    
    try:
        result = client.table("users").select("*").execute()
        users = {}
        for user in result.data:
            users[user["username"]] = user
        return users
    except Exception as e:
        logger.get_logger().error(f"Error getting users from Supabase: {e}")
        return {}


def save_rl_state_supabase(username: str, state_data: Dict[str, Any]) -> bool:
    """Save RL state to Supabase"""
    client = get_supabase_client()
    if not client:
        return False
    
    try:
        now = datetime.now().isoformat()
        # Supabase JSONB columns accept native Python dicts/lists
        data = {
            "username": username,
            "mode_alpha": state_data.get("mode_alpha", {}),
            "mode_beta": state_data.get("mode_beta", {}),
            "mode_history": state_data.get("mode_history", []),
            "chunk_performance": state_data.get("chunk_performance", {}),
            "file_mapping": state_data.get("file_mapping", {}),
            "survey_completed": state_data.get("survey_completed", False),
            "initial_preference": state_data.get("initial_preference"),
            "total_sessions": state_data.get("total_sessions", 0),
            "last_updated": state_data.get("last_updated"),
            "updated_at": now
        }
        
        # Check if state exists
        existing = client.table("rl_state").select("username").eq("username", username).execute()
        
        if existing.data and len(existing.data) > 0:
            # Update
            client.table("rl_state").update(data).eq("username", username).execute()
        else:
            # Insert
            data["created_at"] = now
            client.table("rl_state").insert(data).execute()
        
        logger.get_logger().debug(f"Saved RL state for {username} to Supabase")
        return True
    except Exception as e:
        logger.get_logger().error(f"Error saving RL state to Supabase: {e}")
        return False


def load_rl_state_supabase(username: str) -> Optional[Dict[str, Any]]:
    """Load RL state from Supabase"""
    client = get_supabase_client()
    if not client:
        return None
    
    try:
        result = client.table("rl_state").select("*").eq("username", username).execute()
        
        if result.data and len(result.data) > 0:
            row = result.data[0]
            # Supabase JSONB columns are returned as native Python dicts/lists
            # Handle both cases: if it's already a dict/list, use it; if string, parse it
            def parse_jsonb(value, default):
                if isinstance(value, (dict, list)):
                    return value
                elif isinstance(value, str):
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        return default
                return default if value is None else value
            
            return {
                "mode_alpha": parse_jsonb(row.get("mode_alpha"), {}),
                "mode_beta": parse_jsonb(row.get("mode_beta"), {}),
                "mode_history": parse_jsonb(row.get("mode_history"), []),
                "chunk_performance": parse_jsonb(row.get("chunk_performance"), {}),
                "file_mapping": parse_jsonb(row.get("file_mapping"), {}),
                "survey_completed": bool(row.get("survey_completed", False)),
                "initial_preference": row.get("initial_preference"),
                "total_sessions": row.get("total_sessions", 0),
                "last_updated": row.get("last_updated")
            }
        return None
    except Exception as e:
        logger.get_logger().error(f"Error loading RL state from Supabase: {e}")
        return None
