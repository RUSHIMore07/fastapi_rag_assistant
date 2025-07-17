import requests
import streamlit as st
from typing import Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def get(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Make GET request to API"""
        try:
            response = self.session.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("🔴 Cannot connect to API. Make sure the FastAPI backend is running.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return None
    
    def post(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make POST request to API"""
        try:
            response = self.session.post(f"{self.base_url}{endpoint}", json=data)
            
            if response.status_code == 422:
                error_detail = response.json()
                st.error("❌ Validation Error:")
                st.json(error_detail)
                return None
            elif response.status_code == 500:
                st.error("❌ Server Error")
                try:
                    error_detail = response.json()
                    st.json(error_detail)
                except:
                    st.text(response.text)
                return None
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.ConnectionError:
            st.error("🔴 Cannot connect to API. Make sure the FastAPI backend is running.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"❌ API Error: {e}")
            return None
    
    def delete(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Make DELETE request to API"""
        try:
            response = self.session.delete(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            st.error("🔴 Cannot connect to API. Make sure the FastAPI backend is running.")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return None
