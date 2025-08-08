#!/usr/bin/env python3
"""
Standalone Token Manager for Airbyte API Authentication
This module provides a singleton token manager that handles access token creation
and management without relying on environment variables.
"""

import requests
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import threading


class TokenManager:
    """
    Singleton class for managing Airbyte API access tokens.
    
    This class handles token creation, caching, and refresh logic without
    depending on environment variables. All credentials must be passed
    explicitly during initialization.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TokenManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the token manager (only runs once due to singleton)."""
        if self._initialized:
            return
            
        self._client_id: Optional[str] = None
        self._client_secret: Optional[str] = None
        self._workspace_id: Optional[str] = None
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._token_type: str = "Bearer"
        self._initialized = True
    
    def configure(self, client_id: str, client_secret: str, workspace_id: str) -> None:
        """
        Configure the token manager with credentials.
        
        Args:
            client_id: Airbyte client ID
            client_secret: Airbyte client secret
            workspace_id: Airbyte workspace ID
            
        Raises:
            ValueError: If any required parameter is missing or empty
        """
        if not all([client_id, client_secret, workspace_id]):
            raise ValueError("All parameters (client_id, client_secret, workspace_id) are required")
        
        with self._lock:
            self._client_id = client_id
            self._client_secret = client_secret
            self._workspace_id = workspace_id
            # Clear existing token when reconfiguring
            self._access_token = None
            self._token_expires_at = None
    
    def is_configured(self) -> bool:
        """
        Check if the token manager has been configured with credentials.
        
        Returns:
            True if configured, False otherwise
        """
        return all([self._client_id, self._client_secret, self._workspace_id])
    
    def _is_token_valid(self) -> bool:
        """
        Check if the current token is valid and not expired.
        
        Returns:
            True if token is valid, False otherwise
        """
        if not self._access_token:
            return False
        
        if not self._token_expires_at:
            # If we don't have expiration info, assume token is still valid
            # This is a fallback for tokens that don't provide expiry info
            return True
        
        # Add a 5-minute buffer before expiration
        buffer_time = timedelta(minutes=5)
        return datetime.now() < (self._token_expires_at - buffer_time)
    
    def _create_access_token(self) -> str:
        """
        Create a new access token using client credentials.
        
        Returns:
            Access token string
            
        Raises:
            Exception: If token creation fails
            ValueError: If not configured
        """
        if not self.is_configured():
            raise ValueError("Token manager not configured. Call configure() first.")
        
        print("Creating Airbyte access token...")
        
        # Try multiple endpoint variations
        endpoints_to_try = [
            "https://api.airbyte.com/api/public/v1/applications/token",
            "https://api.airbyte.com/v1/applications/token",
            "https://api.airbyte.ai/api/v1/applications/token"
        ]
        
        payload = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "grant_type": "client_credentials",
            "scope": f"workspace:{self._workspace_id}"
        }
        
        for i, url in enumerate(endpoints_to_try, 1):
            print(f"🔄 Attempt {i}: Trying endpoint {url}")
            
            # Try JSON format first
            headers = {
                "accept": "application/json",
                "content-type": "application/json"
            }
            
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    token_data = response.json()
                    access_token = token_data.get("access_token")
                    
                    if not access_token:
                        print(f"❌ No access token in response from {url}")
                        continue
                    
                    # Store token metadata
                    self._token_type = token_data.get("token_type", "Bearer")
                    expires_in = token_data.get("expires_in")
                    
                    if expires_in:
                        self._token_expires_at = datetime.now() + timedelta(seconds=int(expires_in))
                        print(f"✅ Access token created successfully")
                        print(f"Token type: {self._token_type}")
                        print(f"Expires in: {expires_in} seconds")
                    else:
                        print(f"✅ Access token created successfully (no expiration info)")
                        self._token_expires_at = None
                    
                    return access_token
                    
                elif response.status_code == 500:
                    print(f"❌ 500 error with {url}, trying form-encoded format...")
                    
                    # Try form-encoded format
                    form_headers = {
                        "accept": "application/json",
                        "content-type": "application/x-www-form-urlencoded"
                    }
                    
                    alt_response = requests.post(url, data=payload, headers=form_headers, timeout=30)
                    
                    if alt_response.status_code == 200:
                        token_data = alt_response.json()
                        access_token = token_data.get("access_token")
                        
                        if access_token:
                            self._token_type = token_data.get("token_type", "Bearer")
                            expires_in = token_data.get("expires_in")
                            
                            if expires_in:
                                self._token_expires_at = datetime.now() + timedelta(seconds=int(expires_in))
                            
                            print(f"✅ Access token created successfully with form-encoded format")
                            return access_token
                    else:
                        print(f"❌ Form-encoded also failed: {alt_response.status_code}")
                else:
                    print(f"❌ Failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed for {url}: {e}")
                continue
        
        raise Exception("Failed to create access token with all available endpoints")
    
    def get_token(self) -> str:
        """
        Get a valid access token, creating or refreshing as needed.
        
        Returns:
            Valid access token string
            
        Raises:
            ValueError: If not configured
            Exception: If token creation fails
        """
        if not self.is_configured():
            raise ValueError("Token manager not configured. Call configure() first.")
        
        with self._lock:
            if not self._is_token_valid():
                print("Token invalid or expired, creating new token...")
                self._access_token = self._create_access_token()
            
            return self._access_token
    
    def get_auth_header(self) -> Dict[str, str]:
        """
        Get the authorization header for API requests.
        
        Returns:
            Dictionary containing the Authorization header
            
        Raises:
            ValueError: If not configured
            Exception: If token creation fails
        """
        token = self.get_token()
        return {"Authorization": f"{self._token_type} {token}"}
    
    def invalidate_token(self) -> None:
        """
        Invalidate the current token, forcing a refresh on next use.
        This is useful when you know the token has been revoked or expired.
        """
        with self._lock:
            self._access_token = None
            self._token_expires_at = None
            print("Token invalidated")
    
    def get_token_info(self) -> Dict[str, Any]:
        """
        Get information about the current token.
        
        Returns:
            Dictionary containing token information
        """
        return {
            "has_token": bool(self._access_token),
            "token_type": self._token_type,
            "expires_at": self._token_expires_at.isoformat() if self._token_expires_at else None,
            "is_valid": self._is_token_valid(),
            "is_configured": self.is_configured()
        }


# Convenience function to get the singleton instance
def get_token_manager() -> TokenManager:
    """
    Get the singleton TokenManager instance.
    
    Returns:
        TokenManager singleton instance
    """
    return TokenManager()