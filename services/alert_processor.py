"""
Alert Processor for handling user preferences, enhancing queries, and database storage.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from pymongo import MongoClient
from bson import ObjectId

# Import our query enhancer
from services.rag_news.query_enhancer import QueryEnhancer

logger = logging.getLogger(__name__)

class AlertProcessor:
    """Processes user alerts, enhances queries, and manages database storage."""
    
    def __init__(self, db_uri: str = "mongodb://localhost:27017/", db_name: str = "naarad"):
        """Initialize the alert processor with database connection."""
        self.client = MongoClient(db_uri)
        self.db = self.client[db_name]
        self.alerts = self.db.alerts
        self.query_enhancer = QueryEnhancer()
    
    def process_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single alert, enhance its query, and store in database.
        
        Args:
            alert_data: Alert data in the specified format
            
        Returns:
            Processed alert with enhanced query
        """
        try:
            # Extract preferences
            preferences = alert_data.get('preferences', [])
            if not preferences:
                raise ValueError("No preferences found in alert data")
            
            processed_alerts = []
            
            for pref in preferences:
                # Extract alert details
                alert_id = pref.get('alert_id')
                category = pref.get('category', '').lower()
                sub_categories = [s.lower() for s in pref.get('sub_categories', [])]
                custom_question = pref.get('custom_question', '')
                
                # Create context for query enhancement
                context = {
                    'category': category,
                    'sub_categories': sub_categories,
                    'followup_questions': pref.get('followup_questions', [])
                }
                
                # Enhance the query with event conditions and entity expansion
                enhanced = self.query_enhancer.enhance_query(custom_question, context)
                
                # Build the alert document
                alert_doc = {
                    '_id': ObjectId(alert_id) if alert_id else ObjectId(),
                    'category': category,
                    'sub_categories': sub_categories,
                    'followup_questions': pref.get('followup_questions', []),
                    'custom_question': custom_question,
                    'event_conditions': enhanced.get('event_conditions', []),
                    'keywords': enhanced.get('keywords', []),
                    'negative_keywords': enhanced.get('negative_keywords', []),
                    'contextual_query': ' '.join(enhanced['keywords']),
                    'status': 'active',
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }
                
                # Store in database
                self._store_alert(alert_doc)
                processed_alerts.append(alert_doc)
            
            return {
                'status': 'success',
                'processed_alerts': processed_alerts
            }
            
        except Exception as e:
            logger.error(f"Error processing alert: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _store_alert(self, alert_doc: Dict[str, Any]) -> str:
        """Store or update an alert in the database."""
        alert_id = alert_doc.pop('_id')
        
        # Update if exists, insert if not
        result = self.alerts.update_one(
            {'_id': alert_id},
            {'$set': {**alert_doc, 'updated_at': datetime.utcnow()}},
            upsert=True
        )
        
        return str(alert_id)
    
    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an alert by ID."""
        alert = self.alerts.find_one({'_id': ObjectId(alert_id)})
        if alert:
            alert['_id'] = str(alert['_id'])  # Convert ObjectId to string
        return alert
    
    def deactivate_alert(self, alert_id: str) -> bool:
        """Deactivate an alert."""
        result = self.alerts.update_one(
            {'_id': ObjectId(alert_id)},
            {'$set': {'status': 'inactive', 'updated_at': datetime.utcnow()}}
        )
        return result.modified_count > 0
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        cursor = self.alerts.find({'status': 'active'})
        return [self._format_alert(alert) for alert in cursor]
    
    def _format_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Format alert document for JSON serialization."""
        if '_id' in alert:
            alert['_id'] = str(alert['_id'])
        if 'created_at' in alert and isinstance(alert['created_at'], datetime):
            alert['created_at'] = alert['created_at'].isoformat()
        if 'updated_at' in alert and isinstance(alert['updated_at'], datetime):
            alert['updated_at'] = alert['updated_at'].isoformat()
        return alert


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = AlertProcessor()
    
    # Example alert data
    alert_data = {
        "preferences": [
            {
                "alert_id": "68ea3e598bc5b55c37af219d",
                "category": "sports",
                "sub_categories": ["cricket"],
                "followup_questions": ["team india", "worldcup", "test", "ODI"],
                "custom_question": "only give me my team win"
            }
        ]
    }
    
    # Process the alert
    result = processor.process_alert(alert_data)
    print(json.dumps(result, indent=2))
    
    # Retrieve the processed alert
    if result['status'] == 'success':
        alert_id = result['processed_alerts'][0]['_id']
        alert = processor.get_alert(alert_id)
        print("\nRetrieved alert:")
        print(json.dumps(alert, indent=2))
