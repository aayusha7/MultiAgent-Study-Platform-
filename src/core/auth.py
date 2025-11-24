"""User authentication and management"""

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .logger import logger


@dataclass
class User:
    """User data structure"""
    username: str
    email: str
    password_hash: str  # SHA-256 hash
    created_at: str
    last_login: Optional[str] = None


def get_users_db_path() -> Path:
    """Get path to users database file"""
    return Path(__file__).parent.parent.parent / "users.json"


def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> Dict[str, User]:
    """Load all users - tries Supabase first, falls back to JSON file"""
    # Try Supabase first
    try:
        from .supabase_client import get_all_users_supabase
        supabase_users = get_all_users_supabase()
        if supabase_users:
            users = {}
            for username, user_data in supabase_users.items():
                # Convert Supabase format to User dataclass
                users[username] = User(
                    username=user_data.get("username", username),
                    email=user_data.get("email", ""),
                    password_hash=user_data.get("password_hash", ""),
                    created_at=user_data.get("created_at", datetime.now().isoformat()),
                    last_login=user_data.get("last_login")
                )
            if users:
                return users
    except Exception as e:
        logger.get_logger().debug(f"Supabase not available, using file storage: {e}")
    
    # Fallback to JSON file
    users_path = get_users_db_path()
    
    if not users_path.exists():
        return {}
    
    try:
        # Check if file is empty
        if users_path.stat().st_size == 0:
            # Initialize empty file
            with open(users_path, 'w') as f:
                json.dump({}, f)
            return {}
        
        with open(users_path, 'r') as f:
            data = json.load(f)
        
        # Handle case where file contains empty dict or None
        if not data or not isinstance(data, dict):
            return {}
        
        users = {}
        for username, user_data in data.items():
            users[username] = User(**user_data)
        
        return users
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.get_logger().error(f"Error loading users: {e}")
        # Initialize corrupted file with empty dict
        try:
            with open(users_path, 'w') as f:
                json.dump({}, f)
        except Exception:
            pass
        return {}


def save_users(users: Dict[str, User]) -> bool:
    """Save users - tries Supabase first, falls back to JSON file"""
    # Try Supabase first (save each user)
    supabase_success = False
    try:
        from .supabase_client import save_user_supabase
        for username, user in users.items():
            if save_user_supabase(username, user.password_hash, user.email):
                supabase_success = True
    except Exception as e:
        logger.get_logger().debug(f"Supabase not available, using file storage: {e}")
    
    # Always save to file as backup
    users_path = get_users_db_path()
    
    try:
        # Convert to dict
        data = {}
        for username, user in users.items():
            data[username] = asdict(user)
        
        # Ensure directory exists
        users_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(users_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        logger.get_logger().error(f"Error saving users: {e}")
        return False


def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """
    Register a new user.
    
    Returns:
        (success: bool, message: str)
    """
    users = load_users()
    
    # Check if username already exists
    if username in users:
        return False, "Username already exists. Please choose a different one."
    
    # Check if email already exists
    for user in users.values():
        if user.email.lower() == email.lower():
            return False, "Email already registered. Please use a different email."
    
    # Validate input
    if len(username) < 3:
        return False, "Username must be at least 3 characters long."
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    
    if "@" not in email or "." not in email:
        return False, "Please enter a valid email address."
    
    # Create new user
    new_user = User(
        username=username,
        email=email.lower(),
        password_hash=hash_password(password),
        created_at=datetime.now().isoformat(),
        last_login=None
    )
    
    users[username] = new_user
    
    # Try Supabase first
    try:
        from .supabase_client import save_user_supabase
        if save_user_supabase(username, new_user.password_hash, new_user.email):
            logger.get_logger().info(f"User registered in Supabase: {username}")
    except Exception as e:
        logger.get_logger().debug(f"Supabase not available: {e}")
    
    if save_users(users):
        logger.get_logger().info(f"User registered: {username}")
        return True, f"Account created successfully! Welcome, {username}!"
    else:
        return False, "Error creating account. Please try again."


def authenticate_user(username: str, password: str) -> tuple[bool, Optional[User], str]:
    """
    Authenticate a user.
    
    Returns:
        (success: bool, user: Optional[User], message: str)
    """
    users = load_users()
    
    if username not in users:
        return False, None, "Username or password is incorrect."
    
    user = users[username]
    password_hash = hash_password(password)
    
    if user.password_hash != password_hash:
        return False, None, "Username or password is incorrect."
    
    # Update last login
    user.last_login = datetime.now().isoformat()
    users[username] = user
    save_users(users)
    
    logger.get_logger().info(f"User logged in: {username}")
    return True, user, f"Welcome back, {username}!"


def get_user_state_path(username: str) -> Path:
    """Get path to user-specific state file"""
    return Path(__file__).parent.parent.parent / f".ma_state_{username}.json"


def user_exists(username: str) -> bool:
    """Check if user exists"""
    users = load_users()
    return username in users

