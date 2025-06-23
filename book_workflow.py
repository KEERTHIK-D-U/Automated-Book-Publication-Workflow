import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from playwright.async_api import async_playwright
import chromadb
from chromadb.config import Settings
import numpy as np
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentStatus(Enum):
    SCRAPED = "scraped data!"
    AI_WRITTEN = "ai_written data!"
    AI_REVIEWED = "ai_reviewed data!"
    HUMAN_REVIEW_PENDING = "human_review_pending!"
    HUMAN_APPROVED = "human_approved data!"
    FINALIZED = "finalized data!"
    PUBLISHED = "published data!"


class AgentType(Enum):
    SCRAPER = "scraper!"
    AI_WRITER = "ai_writer1"
    AI_REVIEWER = "ai_reviewer1"
    HUMAN_WRITER = "human_writer!"
    HUMAN_REVIEWER = "human_reviewer!"
    HUMAN_EDITOR = "human_editor!"


@dataclass
class ContentVersion:
    id: str
    content: str
    status: ContentStatus
    agent_type: AgentType
    timestamp: datetime
    metadata: Dict[str, Any]
    parent_version_id: Optional[str] = None
    human_feedback: Optional[str] = None
    ai_feedback: Optional[str] = None


@dataclass
class Chapter:
    id: str
    title: str
    original_url: str
    versions: List[ContentVersion]
    current_version_id: str
    screenshot_path: Optional[str] = None


class WebScraper:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None

    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def scrape_content(self, url: str, screenshot_dir: str = "screenshots") -> Tuple[str, str]:
        try:
            page = await self.context.new_page()
            await page.goto(url)

            await page.wait_for_load_state('networkidle')

            content_selector = '.mw-parser-output'  # WikiSource specific
            content = await page.text_content(content_selector)

            os.makedirs(screenshot_dir, exist_ok=True)
            screenshot_filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            screenshot_path = os.path.join(screenshot_dir, screenshot_filename)
            await page.screenshot(path=screenshot_path, full_page=True)

            await page.close()

            logger.info(f"Successfully scraped content from {url}")
            return content.strip(), screenshot_path

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            raise


class AIAgent:

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    async def spin_content(self, original_content: str, style_instructions: str = "") -> str:
        prompt = f"""
        You are an expert AI writer tasked with creating an engaging, creative adaptation of the following content.

        Instructions:
        - Maintain the core narrative and key plot points
        - Enhance the prose with more vivid descriptions and modern language
        - Add depth to character development and dialogue
        - Improve pacing and narrative flow
        - Keep the overall story structure intact
        {style_instructions}

        Original Content:
        {original_content}

        Please provide your creative adaptation:
        """

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.model.generate_content, prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Error in AI content spinning: {str(e)}")
            raise

    async def review_content(self, content: str, original_content: str) -> Tuple[str, str]:
        prompt = f"""
        You're a very good AI reviewer and editor. So please review the following selected content against the actual one.

        Please Provide:
        ->A described review with specific feedback on imporvements needed 
        ->A changed version of the content incorporating your suggestions

        Og Content:
        {original_content}

        Content to be Review:
        {content}

        Please format your response as:
        REVIEW:
        [Your detailed review and feedback]

        REVISED_CONTENT:
        [Your improved version]
        """

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.model.generate_content, prompt
            )

            response_text = response.text
            if "REVISED_CONTENT:" in response_text:
                parts = response_text.split("REVISED_CONTENT:")
                review = parts[0].replace("REVIEW:", "").strip()
                revised_content = parts[1].strip()
                return review, revised_content
            else:
                return response_text, content

        except Exception as e:
            logger.error(f"Error in AI content review: {str(e)}")
            raise


class ReinforcementSearchEngine:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.content_vectors = None
        self.content_metadata = []
        self.learning_rate = 0.1
        self.search_weights = np.ones(1000)  # Initialize with equal weights

    def index_content(self, contents: List[str], metadata: List[Dict]):
        self.content_vectors = self.vectorizer.fit_transform(contents)
        self.content_metadata = metadata
        logger.info(f"Indexed {len(contents)} documents")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        if self.content_vectors is None:
            return []

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(query_vector, self.content_vectors)[0]

        weighted_similarities = similarities * self.search_weights[:len(similarities)]

        top_indices = np.argsort(weighted_similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((idx, weighted_similarities[idx], self.content_metadata[idx]))

        return results

    def update_weights(self, query: str, clicked_results: List[int], user_feedback: float):
        query_vector = self.vectorizer.transform([query])

        for result_idx in clicked_results:
            if result_idx < len(self.search_weights):
                self.search_weights[result_idx] += self.learning_rate * user_feedback

        self.search_weights = np.clip(self.search_weights, 0.1, 2.0)


class ContentVersionManager:

    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="book_content",
            metadata={"description": "Book publication workflow content"}
        )

    def save_version(self, version: ContentVersion) -> str:
        try:
            content_hash = hashlib.md5(version.content.encode()).hexdigest()

            self.collection.add(
                documents=[version.content],
                metadatas=[{
                    "id": version.id,
                    "status": version.status.value,
                    "agent_type": version.agent_type.value,
                    "timestamp": version.timestamp.isoformat(),
                    "parent_version_id": version.parent_version_id or "",
                    "human_feedback": version.human_feedback or "",
                    "ai_feedback": version.ai_feedback or "",
                    **version.metadata
                }],
                ids=[version.id]
            )

            logger.info(f"Saved version {version.id} with status {version.status.value}")
            return version.id

        except Exception as e:
            logger.error(f"Error saving version: {str(e)}")
            raise

    def get_version(self, version_id: str) -> Optional[ContentVersion]:
        try:
            results = self.collection.get(ids=[version_id])
            if not results['documents']:
                return None

            metadata = results['metadatas'][0]
            return ContentVersion(
                id=metadata['id'],
                content=results['documents'][0],
                status=ContentStatus(metadata['status']),
                agent_type=AgentType(metadata['agent_type']),
                timestamp=datetime.fromisoformat(metadata['timestamp']),
                metadata={k: v for k, v in metadata.items() if k not in [
                    'id', 'status', 'agent_type', 'timestamp', 'parent_version_id',
                    'human_feedback', 'ai_feedback'
                ]},
                parent_version_id=metadata['parent_version_id'] or None,
                human_feedback=metadata['human_feedback'] or None,
                ai_feedback=metadata['ai_feedback'] or None
            )

        except Exception as e:
            logger.error(f"Error retrieving version {version_id}: {str(e)}")
            return None

    def get_version_history(self, chapter_id: str) -> List[ContentVersion]:
        try:
            results = self.collection.query(
                query_texts=[""],
                where={"chapter_id": chapter_id},
                n_results=100
            )

            versions = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                versions.append(ContentVersion(
                    id=metadata['id'],
                    content=doc,
                    status=ContentStatus(metadata['status']),
                    agent_type=AgentType(metadata['agent_type']),
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    metadata={k: v for k, v in metadata.items() if k not in [
                        'id', 'status', 'agent_type', 'timestamp', 'parent_version_id',
                        'human_feedback', 'ai_feedback'
                    ]},
                    parent_version_id=metadata['parent_version_id'] or None,
                    human_feedback=metadata['human_feedback'] or None,
                    ai_feedback=metadata['ai_feedback'] or None
                ))

            versions.sort(key=lambda x: x.timestamp)
            return versions

        except Exception as e:
            logger.error(f"Error retrieving version history: {str(e)}")
            return []


class BookPublicationWorkflow:

    def __init__(self, gemini_api_key: str, db_path: str = "./chroma_db"):
        self.ai_agent = AIAgent(gemini_api_key)
        self.version_manager = ContentVersionManager(db_path)
        self.search_engine = ReinforcementSearchEngine()
        self.chapters: Dict[str, Chapter] = {}

    async def scrape_and_initialize_chapter(self, url: str, title: str) -> str:
        chapter_id = str(uuid.uuid4())

        async with WebScraper() as scraper:
            content, screenshot_path = await scraper.scrape_content(url)

        initial_version = ContentVersion(
            id=str(uuid.uuid4()),
            content=content,
            status=ContentStatus.SCRAPED,
            agent_type=AgentType.SCRAPER,
            timestamp=datetime.now(),
            metadata={
                "chapter_id": chapter_id,
                "source_url": url,
                "title": title
            }
        )

        self.version_manager.save_version(initial_version)

        chapter = Chapter(
            id=chapter_id,
            title=title,
            original_url=url,
            versions=[initial_version],
            current_version_id=initial_version.id,
            screenshot_path=screenshot_path
        )

        self.chapters[chapter_id] = chapter
        logger.info(f"Initialized chapter: {title}")
        return chapter_id

    async def ai_process_chapter(self, chapter_id: str, style_instructions: str = "") -> str:
        chapter = self.chapters[chapter_id]
        current_version = self.version_manager.get_version(chapter.current_version_id)

        spun_content = await self.ai_agent.spin_content(
            current_version.content,
            style_instructions
        )

        writer_version = ContentVersion(
            id=str(uuid.uuid4()),
            content=spun_content,
            status=ContentStatus.AI_WRITTEN,
            agent_type=AgentType.AI_WRITER,
            timestamp=datetime.now(),
            metadata={
                "chapter_id": chapter_id,
                "style_instructions": style_instructions
            },
            parent_version_id=current_version.id
        )

        self.version_manager.save_version(writer_version)

        ai_feedback, reviewed_content = await self.ai_agent.review_content(
            spun_content,
            current_version.content
        )

        reviewer_version = ContentVersion(
            id=str(uuid.uuid4()),
            content=reviewed_content,
            status=ContentStatus.AI_REVIEWED,
            agent_type=AgentType.AI_REVIEWER,
            timestamp=datetime.now(),
            metadata={"chapter_id": chapter_id},
            parent_version_id=writer_version.id,
            ai_feedback=ai_feedback
        )

        self.version_manager.save_version(reviewer_version)

        chapter.versions.extend([writer_version, reviewer_version])
        chapter.current_version_id = reviewer_version.id

        logger.info(f"AI processing completed for chapter: {chapter.title}")
        return reviewer_version.id

    def request_human_review(self, chapter_id: str, reviewer_type: AgentType) -> str:
        chapter = self.chapters[chapter_id]
        current_version = self.version_manager.get_version(chapter.current_version_id)

        review_version = ContentVersion(
            id=str(uuid.uuid4()),
            content=current_version.content,
            status=ContentStatus.HUMAN_REVIEW_PENDING,
            agent_type=reviewer_type,
            timestamp=datetime.now(),
            metadata={
                "chapter_id": chapter_id,
                "review_requested": True
            },
            parent_version_id=current_version.id
        )

        self.version_manager.save_version(review_version)
        chapter.versions.append(review_version)

        logger.info(f"Human review requested for chapter: {chapter.title}")
        return review_version.id

    def submit_human_feedback(self, version_id: str, feedback: str,
                              revised_content: Optional[str] = None) -> str:
        version = self.version_manager.get_version(version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")

        approved_version = ContentVersion(
            id=str(uuid.uuid4()),
            content=revised_content or version.content,
            status=ContentStatus.HUMAN_APPROVED,
            agent_type=version.agent_type,
            timestamp=datetime.now(),
            metadata=version.metadata,
            parent_version_id=version_id,
            human_feedback=feedback
        )

        self.version_manager.save_version(approved_version)

        chapter_id = version.metadata["chapter_id"]
        chapter = self.chapters[chapter_id]
        chapter.versions.append(approved_version)
        chapter.current_version_id = approved_version.id

        logger.info(f"Human feedback submitted for version!: {version_id}")
        return approved_version.id

    def finalize_chapter(self, chapter_id: str) -> str:
        chapter = self.chapters[chapter_id]
        current_version = self.version_manager.get_version(chapter.current_version_id)

        final_version = ContentVersion(
            id=str(uuid.uuid4()),
            content=current_version.content,
            status=ContentStatus.FINALIZED,
            agent_type=current_version.agent_type,
            timestamp=datetime.now(),
            metadata={**current_version.metadata, "finalized": True},
            parent_version_id=current_version.id
        )

        self.version_manager.save_version(final_version)
        chapter.versions.append(final_version)
        chapter.current_version_id = final_version.id

        # Index for search
        self.search_engine.index_content(
            [final_version.content],
            [{"chapter_id": chapter_id, "title": chapter.title}]
        )

        logger.info(f"Chapter finalized: {chapter.title}")
        return final_version.id

    def search_content(self, query: str, top_k: int = 5) -> List[Dict]:
        results = self.search_engine.search(query, top_k)

        formatted_results = []
        for idx, score, metadata in results:
            chapter = self.chapters.get(metadata["chapter_id"])
            if chapter:
                formatted_results.append({
                    "chapter_id": metadata["chapter_id"],
                    "title": metadata["title"],
                    "relevance_score": score,
                    "url": chapter.original_url
                })

        return formatted_results

    def get_chapter_status(self, chapter_id: str) -> Dict:
        chapter = self.chapters[chapter_id]
        current_version = self.version_manager.get_version(chapter.current_version_id)

        return {
            "chapter_id": chapter_id,
            "title": chapter.title,
            "current_status": current_version.status.value,
            "current_agent": current_version.agent_type.value,
            "version_count": len(chapter.versions),
            "last_updated": current_version.timestamp.isoformat(),
            "screenshot_available": chapter.screenshot_path is not None
        }


class WorkflowAPI:

    def __init__(self, workflow: BookPublicationWorkflow):
        self.workflow = workflow

    async def start_chapter_processing(self, url: str, title: str,
                                       style_instructions: str = "") -> Dict:
        try:
            chapter_id = await self.workflow.scrape_and_initialize_chapter(url, title)

            ai_version_id = await self.workflow.ai_process_chapter(
                chapter_id, style_instructions
            )

            return {
                "success": True,
                "chapter_id": chapter_id,
                "ai_version_id": ai_version_id,
                "status": "ai_review_completed",
                "next_step": "human_review"
            }

        except Exception as e:
            logger.error(f"Error in chapter processing: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


# Demo function
async def demo_workflow():

    GEMINI_API_KEY = "# Replace with actual key"

    workflow = BookPublicationWorkflow(GEMINI_API_KEY)
    api = WorkflowAPI(workflow)

    url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
    title = "The Gates of Morning - Book 1, Chapter 1"
    style_instructions = "Make the prose more modern and engaging while maintaining the original story"

    print("Starting automated book publication workflow...")

    result = await api.start_chapter_processing(url, title, style_instructions)

    if result["success"]:
        chapter_id = result["chapter_id"]
        print(f"✓ Chapter processed successfully: {chapter_id}")

        status = workflow.get_chapter_status(chapter_id)
        print(f"✓ Current status: {json.dumps(status, indent=2)}")

        review_id = workflow.request_human_review(chapter_id, AgentType.HUMAN_REVIEWER)
        print(f"✓ Human review requested: {review_id}")

        feedback = "The content looks good, just minor formatting improvements needed."
        approved_id = workflow.submit_human_feedback(review_id, feedback)
        print(f"✓ Human feedback submitted: {approved_id}")

        final_id = workflow.finalize_chapter(chapter_id)
        print(f"✓ Chapter finalized: {final_id}")

        search_results = workflow.search_content("morning gates")
        print(f"✓ Search results: {json.dumps(search_results, indent=2)}")

    else:
        print(f"✗ Error: {result['error']}")


if __name__ == "__main__":
    asyncio.run(demo_workflow())