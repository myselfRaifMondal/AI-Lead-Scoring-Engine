import pandas as pd
import numpy as np
from typing import List, Dict, Any
import openai
from sentence_transformers import SentenceTransformer
import logging
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMReranker:
    def __init__(self, openai_api_key: str = None, sentence_transformer_model: str = "all-MiniLM-L6-v2"):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Initialize sentence transformer for embeddings
        self.sentence_model = SentenceTransformer(sentence_transformer_model)
        
        # Intent signal patterns
        self.intent_patterns = {
            'urgency': [
                r'asap', r'urgent', r'immediately', r'as soon as possible',
                r'time sensitive', r'deadline', r'rush', r'quick'
            ],
            'decision_making': [
                r'ready to buy', r'ready to purchase', r'make an offer',
                r'pre-approved', r'financing secured', r'cash buyer'
            ],
            'engagement': [
                r'very interested', r'love this', r'perfect', r'exactly what',
                r'schedule viewing', r'see the property', r'visit today'
            ],
            'qualification': [
                r'budget', r'down payment', r'mortgage', r'loan',
                r'credit score', r'income', r'employment'
            ],
            'negative_signals': [
                r'just browsing', r'not ready', r'maybe later',
                r'too expensive', r'out of budget', r'not interested'
            ]
        }
        
    def extract_intent_signals(self, text: str) -> Dict[str, float]:
        """Extract intent signals from text using pattern matching"""
        if not text:
            return {}
        
        text_lower = text.lower()
        signals = {}
        
        for signal_type, patterns in self.intent_patterns.items():
            signal_score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                signal_score += matches
            
            # Normalize by text length
            text_length = len(text_lower.split())
            signals[signal_type] = signal_score / max(text_length, 1) if text_length > 0 else 0
        
        return signals
    
    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for text using sentence transformer"""
        if not texts:
            return np.array([])
        
        # Filter out empty texts
        texts = [text for text in texts if text and text.strip()]
        
        if not texts:
            return np.array([])
        
        embeddings = self.sentence_model.encode(texts)
        return embeddings
    
    def analyze_communication_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment and tone of communication"""
        if not text:
            return {'sentiment': 0.0, 'confidence': 0.0}
        
        # Use OpenAI for sentiment analysis if available
        if self.openai_api_key:
            try:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=f"""Analyze the sentiment and buying intent of this real estate inquiry:
                    
                    Text: "{text}"
                    
                    Return a JSON with:
                    - sentiment: score from -1 (negative) to 1 (positive)
                    - buying_intent: score from 0 (no intent) to 1 (strong intent)
                    - urgency: score from 0 (no urgency) to 1 (high urgency)
                    - confidence: confidence in assessment from 0 to 1
                    
                    JSON:""",
                    max_tokens=100,
                    temperature=0.1
                )
                
                result = response.choices[0].text.strip()
                # Parse JSON response (simplified)
                import json
                try:
                    return json.loads(result)
                except:
                    pass
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}")
        
        # Fallback to simple pattern-based analysis
        positive_words = ['great', 'excellent', 'perfect', 'love', 'interested', 'beautiful']
        negative_words = ['expensive', 'small', 'problem', 'issue', 'concerned', 'worried']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        sentiment = (positive_count - negative_count) / max(total_words, 1)
        
        return {
            'sentiment': max(-1, min(1, sentiment)),
            'buying_intent': min(1, positive_count / max(total_words, 1)),
            'urgency': 0.5,  # Default value
            'confidence': 0.7
        }
    
    def rerank_leads(self, leads_df: pd.DataFrame, base_scores: np.ndarray) -> np.ndarray:
        """Re-rank leads using LLM analysis of text data"""
        logger.info(f"Re-ranking {len(leads_df)} leads using LLM analysis")
        
        reranked_scores = base_scores.copy()
        
        for idx, row in leads_df.iterrows():
            # Collect all text data for this lead
            text_fields = []
            
            # Email content
            if 'email_content' in row and pd.notna(row['email_content']):
                text_fields.append(row['email_content'])
            
            # Chat messages
            if 'chat_messages' in row and pd.notna(row['chat_messages']):
                text_fields.append(row['chat_messages'])
            
            # Property inquiries
            if 'property_inquiries' in row and pd.notna(row['property_inquiries']):
                text_fields.append(row['property_inquiries'])
            
            # Notes from sales team
            if 'sales_notes' in row and pd.notna(row['sales_notes']):
                text_fields.append(row['sales_notes'])
            
            if not text_fields:
                continue
            
            # Combine all text
            combined_text = ' '.join(text_fields)
            
            # Extract intent signals
            intent_signals = self.extract_intent_signals(combined_text)
            
            # Analyze sentiment
            sentiment_analysis = self.analyze_communication_sentiment(combined_text)
            
            # Calculate text-based score adjustment
            text_score_adjustment = 0.0
            
            # Positive adjustments
            text_score_adjustment += intent_signals.get('urgency', 0) * 0.3
            text_score_adjustment += intent_signals.get('decision_making', 0) * 0.4
            text_score_adjustment += intent_signals.get('engagement', 0) * 0.2
            text_score_adjustment += intent_signals.get('qualification', 0) * 0.1
            
            # Negative adjustments
            text_score_adjustment -= intent_signals.get('negative_signals', 0) * 0.5
            
            # Sentiment adjustment
            text_score_adjustment += sentiment_analysis.get('buying_intent', 0) * 0.3
            text_score_adjustment += sentiment_analysis.get('sentiment', 0) * 0.1
            
            # Apply confidence weighting
            confidence = sentiment_analysis.get('confidence', 0.5)
            text_score_adjustment *= confidence
            
            # Limit adjustment to reasonable range
            text_score_adjustment = max(-0.3, min(0.3, text_score_adjustment))
            
            # Apply adjustment to base score
            reranked_scores[idx] = min(1.0, max(0.0, base_scores[idx] + text_score_adjustment))
        
        logger.info("LLM re-ranking completed")
        return reranked_scores
    
    def explain_text_adjustment(self, text: str, adjustment: float) -> str:
        """Explain why text analysis led to score adjustment"""
        if not text:
            return "No text data available for analysis"
        
        intent_signals = self.extract_intent_signals(text)
        sentiment_analysis = self.analyze_communication_sentiment(text)
        
        explanation_parts = []
        
        if intent_signals.get('urgency', 0) > 0:
            explanation_parts.append(f"Shows urgency signals (impact: +{intent_signals['urgency']*0.3:.2f})")
        
        if intent_signals.get('decision_making', 0) > 0:
            explanation_parts.append(f"Shows decision-making readiness (impact: +{intent_signals['decision_making']*0.4:.2f})")
        
        if intent_signals.get('engagement', 0) > 0:
            explanation_parts.append(f"Shows high engagement (impact: +{intent_signals['engagement']*0.2:.2f})")
        
        if intent_signals.get('negative_signals', 0) > 0:
            explanation_parts.append(f"Contains negative signals (impact: -{intent_signals['negative_signals']*0.5:.2f})")
        
        if sentiment_analysis.get('buying_intent', 0) > 0.5:
            explanation_parts.append(f"Positive buying intent detected (impact: +{sentiment_analysis['buying_intent']*0.3:.2f})")
        
        if not explanation_parts:
            explanation_parts.append("No significant text signals detected")
        
        return f"Text analysis adjustment: {adjustment:+.3f}. " + "; ".join(explanation_parts)
    
    def process_batch(self, leads_df: pd.DataFrame, base_scores: np.ndarray, 
                     batch_size: int = 100) -> Dict[str, Any]:
        """Process leads in batches for better performance"""
        logger.info(f"Processing {len(leads_df)} leads in batches of {batch_size}")
        
        results = []
        
        for i in range(0, len(leads_df), batch_size):
            batch_df = leads_df.iloc[i:i+batch_size]
            batch_scores = base_scores[i:i+batch_size]
            
            # Re-rank batch
            reranked_batch = self.rerank_leads(batch_df, batch_scores)
            
            # Store results
            for j, (idx, row) in enumerate(batch_df.iterrows()):
                lead_id = row.get('lead_id', f'lead_{idx}')
                original_score = batch_scores[j]
                new_score = reranked_batch[j]
                adjustment = new_score - original_score
                
                # Get text for explanation
                text_fields = []
                for field in ['email_content', 'chat_messages', 'property_inquiries', 'sales_notes']:
                    if field in row and pd.notna(row[field]):
                        text_fields.append(row[field])
                
                combined_text = ' '.join(text_fields)
                
                results.append({
                    'lead_id': lead_id,
                    'original_score': original_score,
                    'reranked_score': new_score,
                    'adjustment': adjustment,
                    'explanation': self.explain_text_adjustment(combined_text, adjustment)
                })
        
        return {
            'processed_count': len(results),
            'average_adjustment': np.mean([r['adjustment'] for r in results]),
            'results': results
        }

# Example usage
if __name__ == "__main__":
    # Initialize reranker
    reranker = LLMReranker()
    
    # Sample data (would come from your database)
    sample_data = pd.DataFrame({
        'lead_id': ['lead_1', 'lead_2', 'lead_3'],
        'email_content': [
            'Hi, I am very interested in the property and ready to make an offer ASAP!',
            'Just browsing properties, not ready to buy yet.',
            'Perfect house! Can we schedule a viewing today? I have pre-approval ready.'
        ],
        'chat_messages': [
            'This is exactly what I was looking for!',
            'Maybe I will consider it later.',
            'Love this property, how quickly can we move?'
        ]
    })
    
    # Base scores from traditional model
    base_scores = np.array([0.6, 0.3, 0.7])
    
    # Re-rank using LLM
    results = reranker.process_batch(sample_data, base_scores)
    
    print("Re-ranking Results:")
    for result in results['results']:
        print(f"Lead {result['lead_id']}: {result['original_score']:.3f} → {result['reranked_score']:.3f}")
        print(f"  {result['explanation']}")
        print()
