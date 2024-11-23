# services/generative_agent/application/analyzers/sentiment_analyzer.py

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import asyncio

from ...domain.models import AnalysisResponse, AnalysisType, ConfidenceLevel
from ...domain.interfaces import LLMProvider, AnalysisProvider
from ...domain.exceptions import AnalysisError

class SentimentLevel(Enum):
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"

class SentimentSource(Enum):
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    MARKET_DATA = "market_data"
    ANALYST_REPORTS = "analyst_reports"
    TECHNICAL_INDICATORS = "technical_indicators"

class SentimentAnalyzer:
    """
    Analyzes market sentiment using multiple data sources and LLM processing.
    Combines news, social media, market indicators, and analyst sentiment.
    """
    
    def __init__(
        self,
        llm_client: LLMProvider,
        confidence_threshold: float = 0.7,
        sentiment_window: int = 7  # days
    ):
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        self.sentiment_window = sentiment_window

    async def analyze(
        self,
        market_data: Dict[str, Any],
        news_data: List[Dict[str, Any]],
        social_data: List[Dict[str, Any]],
        analyst_reports: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> AnalysisResponse:
        """
        Process sentiment analysis using multiple sources
        """
        try:
            # Analyze different sentiment sources concurrently
            news_sentiment, social_sentiment, market_sentiment, analyst_sentiment = \
                await asyncio.gather(
                    self._analyze_news_sentiment(news_data),
                    self._analyze_social_sentiment(social_data),
                    self._analyze_market_sentiment(market_data),
                    self._analyze_analyst_sentiment(analyst_reports)
                )

            # Combine sentiment analyses
            combined_sentiment = await self._combine_sentiment_sources(
                news_sentiment,
                social_sentiment,
                market_sentiment,
                analyst_sentiment
            )

            # Generate detailed analysis
            detailed_analysis = self._generate_sentiment_analysis(
                combined_sentiment,
                news_sentiment,
                social_sentiment,
                market_sentiment,
                analyst_sentiment
            )

            # Generate recommendations
            recommendations = self._generate_sentiment_recommendations(
                combined_sentiment
            )

            return AnalysisResponse(
                request_id=metadata.get('request_id', ''),
                analysis_type=AnalysisType.SENTIMENT,
                summary=self._generate_sentiment_summary(combined_sentiment),
                detailed_analysis=detailed_analysis,
                recommendations=recommendations,
                confidence_score=combined_sentiment['confidence'],
                confidence_level=self._determine_confidence_level(
                    combined_sentiment['confidence']
                ),
                metadata={
                    **metadata,
                    'sentiment_level': combined_sentiment['level'].value,
                    'sentiment_sources': len(combined_sentiment['sources']),
                    'sentiment_window': self.sentiment_window
                },
                processing_time=metadata.get('processing_time', 0),
                timestamp=datetime.now()
            )

        except Exception as e:
            raise AnalysisError(f"Sentiment analysis failed: {str(e)}")

    async def _analyze_news_sentiment(
        self,
        news_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze sentiment from news articles"""
        
        # Prepare news content for LLM analysis
        news_content = []
        for article in news_data:
            news_content.append({
                'title': article.get('title', ''),
                'summary': article.get('summary', ''),
                'source': article.get('source', ''),
                'timestamp': article.get('timestamp', ''),
                'relevance': article.get('relevance', 0.0)
            })

        # Group articles by topic/impact
        grouped_articles = self._group_news_by_topic(news_content)
        
        # Analyze sentiment for each group
        sentiments = []
        for topic, articles in grouped_articles.items():
            prompt = self._create_news_sentiment_prompt(topic, articles)
            response = await self.llm_client.generate(prompt)
            sentiment = self._parse_sentiment_response(response)
            sentiments.append({
                'topic': topic,
                'sentiment': sentiment['level'],
                'confidence': sentiment['confidence'],
                'articles': len(articles)
            })

        return {
            'source': SentimentSource.NEWS,
            'sentiment_by_topic': sentiments,
            'overall_sentiment': self._aggregate_topic_sentiments(sentiments),
            'article_count': len(news_data),
            'confidence': sum(s['confidence'] for s in sentiments) / len(sentiments)
                if sentiments else 0.0
        }

    async def _analyze_social_sentiment(
        self,
        social_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze sentiment from social media data"""
        
        # Group social media content by platform and influence
        grouped_content = self._group_social_content(social_data)
        
        platform_sentiments = []
        for platform, content in grouped_content.items():
            prompt = self._create_social_sentiment_prompt(platform, content)
            response = await self.llm_client.generate(prompt)
            sentiment = self._parse_sentiment_response(response)
            
            platform_sentiments.append({
                'platform': platform,
                'sentiment': sentiment['level'],
                'confidence': sentiment['confidence'],
                'post_count': len(content),
                'engagement_metrics': self._calculate_engagement_metrics(content)
            })

        return {
            'source': SentimentSource.SOCIAL_MEDIA,
            'sentiment_by_platform': platform_sentiments,
            'overall_sentiment': self._aggregate_platform_sentiments(
                platform_sentiments
            ),
            'total_posts': len(social_data),
            'confidence': sum(s['confidence'] for s in platform_sentiments) 
                / len(platform_sentiments) if platform_sentiments else 0.0
        }

    async def _analyze_market_sentiment(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze sentiment from market indicators"""
        
        # Calculate technical sentiment indicators
        vix_sentiment = self._analyze_vix_sentiment(
            market_data.get('vix_data', [])
        )
        put_call_sentiment = self._analyze_put_call_sentiment(
            market_data.get('options_data', {})
        )
        momentum_sentiment = self._analyze_momentum_sentiment(
            market_data.get('price_data', [])
        )
        volume_sentiment = self._analyze_volume_sentiment(
            market_data.get('volume_data', [])
        )

        # Combine market-based sentiment indicators
        indicator_sentiments = [
            {'indicator': 'VIX', **vix_sentiment},
            {'indicator': 'Put/Call Ratio', **put_call_sentiment},
            {'indicator': 'Momentum', **momentum_sentiment},
            {'indicator': 'Volume', **volume_sentiment}
        ]

        return {
            'source': SentimentSource.MARKET_DATA,
            'sentiment_by_indicator': indicator_sentiments,
            'overall_sentiment': self._aggregate_indicator_sentiments(
                indicator_sentiments
            ),
            'confidence': sum(s['confidence'] for s in indicator_sentiments) 
                / len(indicator_sentiments) if indicator_sentiments else 0.0
        }

    async def _analyze_analyst_sentiment(
        self,
        analyst_reports: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze sentiment from analyst reports"""
        
        # Group reports by firm/analyst
        grouped_reports = self._group_analyst_reports(analyst_reports)
        
        firm_sentiments = []
        for firm, reports in grouped_reports.items():
            prompt = self._create_analyst_sentiment_prompt(firm, reports)
            response = await self.llm_client.generate(prompt)
            sentiment = self._parse_sentiment_response(response)
            
            firm_sentiments.append({
                'firm': firm,
                'sentiment': sentiment['level'],
                'confidence': sentiment['confidence'],
                'report_count': len(reports),
                'average_rating': self._calculate_average_rating(reports)
            })

        return {
            'source': SentimentSource.ANALYST_REPORTS,
            'sentiment_by_firm': firm_sentiments,
            'overall_sentiment': self._aggregate_firm_sentiments(firm_sentiments),
            'total_reports': len(analyst_reports),
            'confidence': sum(s['confidence'] for s in firm_sentiments) 
                / len(firm_sentiments) if firm_sentiments else 0.0
        }

    async def _combine_sentiment_sources(
        self,
        news_sentiment: Dict[str, Any],
        social_sentiment: Dict[str, Any],
        market_sentiment: Dict[str, Any],
        analyst_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine sentiment from all sources with weighting"""
        
        # Define source weights
        weights = {
            SentimentSource.NEWS: 0.3,
            SentimentSource.SOCIAL_MEDIA: 0.15,
            SentimentSource.MARKET_DATA: 0.35,
            SentimentSource.ANALYST_REPORTS: 0.2
        }

        # Calculate weighted sentiment scores
        sentiment_scores = {
            SentimentSource.NEWS: self._sentiment_to_score(
                news_sentiment['overall_sentiment']
            ) * weights[SentimentSource.NEWS] * news_sentiment['confidence'],
            
            SentimentSource.SOCIAL_MEDIA: self._sentiment_to_score(
                social_sentiment['overall_sentiment']
            ) * weights[SentimentSource.SOCIAL_MEDIA] * social_sentiment['confidence'],
            
            SentimentSource.MARKET_DATA: self._sentiment_to_score(
                market_sentiment['overall_sentiment']
            ) * weights[SentimentSource.MARKET_DATA] * market_sentiment['confidence'],
            
            SentimentSource.ANALYST_REPORTS: self._sentiment_to_score(
                analyst_sentiment['overall_sentiment']
            ) * weights[SentimentSource.ANALYST_REPORTS] * analyst_sentiment['confidence']
        }

        # Calculate overall weighted sentiment
        total_weighted_score = sum(sentiment_scores.values())
        total_weight = sum(weights[source] * sentiment['confidence']
            for source, sentiment in [
                (SentimentSource.NEWS, news_sentiment),
                (SentimentSource.SOCIAL_MEDIA, social_sentiment),
                (SentimentSource.MARKET_DATA, market_sentiment),
                (SentimentSource.ANALYST_REPORTS, analyst_sentiment)
            ])

        if total_weight > 0:
            normalized_score = total_weighted_score / total_weight
        else:
            normalized_score = 0

        return {
            'level': self._score_to_sentiment(normalized_score),
            'score': normalized_score,
            'confidence': total_weight,
            'sources': {
                'news': news_sentiment,
                'social': social_sentiment,
                'market': market_sentiment,
                'analyst': analyst_sentiment
            },
            'source_weights': weights,
            'source_scores': sentiment_scores
        }

    def _sentiment_to_score(self, sentiment: SentimentLevel) -> float:
        """Convert sentiment level to numerical score"""
        scores = {
            SentimentLevel.VERY_BEARISH: -1.0,
            SentimentLevel.BEARISH: -0.5,
            SentimentLevel.NEUTRAL: 0.0,
            SentimentLevel.BULLISH: 0.5,
            SentimentLevel.VERY_BULLISH: 1.0
        }
        return scores.get(sentiment, 0.0)

    def _score_to_sentiment(self, score: float) -> SentimentLevel:
        """Convert numerical score to sentiment level"""
        if score <= -0.7:
            return SentimentLevel.VERY_BEARISH
        elif score <= -0.3:
            return SentimentLevel.BEARISH
        elif score <= 0.3:
            return SentimentLevel.NEUTRAL
        elif score <= 0.7:
            return SentimentLevel.BULLISH
        else:
            return SentimentLevel.VERY_BULLISH

    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        if confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            return ConfidenceLevel.MODERATE
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _generate_sentiment_analysis(
        self,
        combined_sentiment: Dict[str, Any],
        news_sentiment: Dict[str, Any],
        social_sentiment: Dict[str, Any],
        market_sentiment: Dict[str, Any],
        analyst_sentiment: Dict[str, Any]
    ) -> str:
        """Generate detailed sentiment analysis"""
        analysis_parts = []
        
        # Overall sentiment
        analysis_parts.append("## Overall Sentiment Analysis\n")
        analysis_parts.append(
            f"Current Sentiment: {combined_sentiment['level'].value}\n"
            f"Confidence Score: {combined_sentiment['confidence']:.2f}\n"
            f"Sentiment Score: {combined_sentiment['score']:.2f}\n"
        )

        # News sentiment
        analysis_parts.append("## News Sentiment\n")
        for topic in news_sentiment['sentiment_by_topic']:
            analysis_parts.append(
                f"Topic: {topic['topic']}\n"
                f"- Sentiment: {topic['sentiment'].value}\n"
                f"- Confidence: {topic['confidence']:.2f}\n"
                f"- Articles: {topic['articles']}\n"
            )

        # Social media sentiment
        analysis_parts.append("## Social Media Sentiment\n")
        for platform in social_sentiment['sentiment_by_platform']:
            analysis_parts.append(
                f"Platform: {platform['platform']}\n"
                f"- Sentiment: {platform['sentiment'].value}\n"
                f"- Posts: {platform['post_count']}\n"
                f"- Engagement: {platform['engagement_metrics']}\n"
            )

        # Market sentiment
        analysis_parts.append("## Market Sentiment Indicators\n")
        for indicator in market_sentiment['sentiment_by_indicator']:
            analysis_parts.append(
                f"Indicator: {indicator['indicator']}\n"
                f"- Sentiment: {indicator['sentiment'].value}\n"
                f"- Confidence: {indicator['confidence']:.2f}\n"
            )

        # Analyst sentiment
        analysis_parts.append("## Analyst Sentiment\n")
        for firm in analyst_sentiment['sentiment_by_firm']:
            analysis_parts.append(
