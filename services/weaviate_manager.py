import streamlit as st
from typing import Dict, List, Optional, Tuple, Any

class WeaviateManager:
    """Handles Weaviate operations for the FDA Consultant AI using client v4."""
    
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = url
        self.api_key = api_key
        self.client = None
        self.initialized = False
    
    def initialize(self, url: Optional[str] = None, api_key: Optional[str] = None) -> bool:
        """Initialize Weaviate client."""
        if url:
            self.url = url
        if api_key:
            self.api_key = api_key
            
        if not self.url:
            return False
            
        try:
            import weaviate
            from weaviate.classes.init import Auth
            
            # Remove protocol prefix if present for proper connection
            clean_url = self.url
            for prefix in ["http://", "https://"]:
                if clean_url.startswith(prefix):
                    clean_url = clean_url[len(prefix):]
                    break
            
            # Connect based on URL type
            if ".weaviate.network" in clean_url or ".weaviate.cloud" in clean_url:
                # Connect to Weaviate Cloud
                if self.api_key:
                    self.client = weaviate.connect_to_weaviate_cloud(
                        cluster_url=clean_url,
                        auth_credentials=Auth.api_key(self.api_key)
                    )
                else:
                    self.client = weaviate.connect_to_weaviate_cloud(
                        cluster_url=clean_url
                    )
            else:
                # Connect to self-hosted Weaviate
                if self.api_key:
                    self.client = weaviate.connect_to_local(
                        host=clean_url,
                        auth_credentials=Auth.api_key(self.api_key)
                    )
                else:
                    self.client = weaviate.connect_to_local(
                        host=clean_url
                    )
            
            # Test connection - just try to list collections instead of getting cluster status
            self.client.collections.list_all()
            
            self.initialized = True
            return True
            
        except ImportError:
            st.error("Weaviate client not installed. Run 'pip install weaviate-client'.")
            self.initialized = False
            return False
        except Exception as e:
            st.error(f"Failed to connect to Weaviate: {e}")
            self.initialized = False
            return False
    
    def __del__(self):
        """Ensure client is closed on garbage collection."""
        self.close()
    
    def close(self):
        """Close the Weaviate client connection."""
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass  # Ignore errors during closing
            self.client = None
            self.initialized = False
    
    def get_schema(self) -> Dict:
        """Get the Weaviate schema."""
        if not self.initialized and not self.initialize():
            return {}
            
        try:
            # Get all collections
            collections = self.client.collections.list_all()
            
            # Format as schema similar to v3 API
            schema = {
                "classes": [
                    {
                        "class": collection.name,
                        "description": getattr(collection, "description", ""),
                        "properties": [
                            {
                                "name": prop.name,
                                "dataType": [prop.data_type.value],
                                "description": getattr(prop, "description", "")
                            }
                            for prop in collection.properties
                        ]
                    }
                    for collection in collections
                ]
            }
            
            return schema
        except Exception as e:
            st.error(f"Failed to get Weaviate schema: {e}")
            return {}
    
    def get_class_names(self) -> List[str]:
        """Get a list of all class names (collections) in Weaviate."""
        if not self.initialized and not self.initialize():
            return []
            
        try:
            collections = self.client.collections.list_all()
            return [collection.name for collection in collections]
        except Exception as e:
            st.error(f"Failed to get collection names: {e}")
            return []
    
    def get_class_count(self, class_name: str) -> int:
        """Get the count of objects in a collection."""
        if not self.initialized and not self.initialize():
            return 0
            
        try:
            # Check if the collection exists
            try:
                collection = self.client.collections.get(class_name)
            except ValueError:
                # Collection doesn't exist
                return 0
                
            # Get the count using aggregate
            try:
                aggregate_result = collection.aggregate.over_all(total_count=True)
                return aggregate_result.total_count
            except Exception as e:
                st.error(f"Error getting collection count: {e}")
                return 0
                
        except Exception as e:
            st.error(f"Failed to get count for collection {class_name}: {e}")
            return 0