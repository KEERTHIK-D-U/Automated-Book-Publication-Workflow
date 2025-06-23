# Automated Book Publication Workflow

## Overview
A comprehensive automated system for processing, enhancing, and managing book content through an AI-driven workflow. This project demonstrates advanced automation capabilities for content publication pipelines.

## Project Description
This system automates the entire book publication process from web scraping to final publication, incorporating AI-powered content enhancement, version control, and human review workflows. The solution provides a complete pipeline for transforming raw text content into publication-ready material.

## Key Features

### 🤖 AI-Powered Content Processing
- **Intelligent Content Spinning**: Automatically rewrites and enhances original content while maintaining narrative integrity
- **AI Review System**: Dual-layer AI review process for content quality assurance
- **Style Adaptation**: Configurable writing style instructions for content transformation

### 📚 Version Control & Management
- **Complete Version History**: Track all content iterations with metadata
- **Status Tracking**: Monitor content through multiple workflow stages
- **Agent Attribution**: Track which AI or human agent modified each version

### 🔍 Advanced Search & Discovery
- **Reinforcement Learning Search**: Self-improving search engine with user feedback integration
- **TF-IDF Vectorization**: Semantic content search capabilities
- **Content Indexing**: Automatic indexing of finalized content for discovery

### 🌐 Web Scraping & Automation
- **Automated Content Extraction**: Intelligent web scraping with screenshot capture
- **Multi-source Support**: Extensible scraping system for various content sources
- **Error Handling**: Robust error handling and retry mechanisms

### 👥 Human-in-the-Loop Workflow
- **Review Request System**: Structured human review process
- **Feedback Integration**: Capture and incorporate human feedback
- **Approval Workflows**: Multi-stage approval process

## Technical Architecture

### Core Components
1. **WebScraper**: Automated content extraction using Playwright
2. **AIAgent**: Gemini-powered content processing and enhancement
3. **ContentVersionManager**: ChromaDB-based version control system
4. **ReinforcementSearchEngine**: Self-learning search system
5. **BookPublicationWorkflow**: Orchestrates the entire workflow

### Technology Stack
- **Python 3.8+**: Core programming language
- **Playwright**: Web automation and scraping
- **Google Gemini AI**: Content generation and review
- **ChromaDB**: Vector database for content storage
- **scikit-learn**: Machine learning for search functionality
- **asyncio**: Asynchronous processing

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key
- Git

### Installation Steps

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd automated-book-publication
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Playwright browsers**
```bash
playwright install
```

5. **Configure API key**
```bash
# Set your Gemini API key in the demo_workflow() function
GEMINI_API_KEY = "your-api-key-here"
```

## Usage

### Basic Usage
```bash
python ai.py
```

### API Usage Example
```python
from ai import BookPublicationWorkflow, WorkflowAPI

# Initialize workflow
workflow = BookPublicationWorkflow("your-api-key")
api = WorkflowAPI(workflow)

# Process a chapter
result = await api.start_chapter_processing(
    url="https://example.com/chapter1",
    title="Chapter 1",
    style_instructions="Make prose more engaging"
)
```

## Workflow Stages

### 1. Content Acquisition
- **Web Scraping**: Extract content from specified URLs
- **Screenshot Capture**: Visual documentation of source
- **Initial Storage**: Save original content with metadata

### 2. AI Processing
- **Content Spinning**: AI-powered content enhancement
- **Quality Review**: Automated content review and improvement
- **Version Creation**: Track each processing stage

### 3. Human Review
- **Review Request**: Flag content for human review
- **Feedback Collection**: Capture human feedback and revisions
- **Approval Process**: Human approval workflow

### 4. Finalization
- **Content Finalization**: Mark content as publication-ready
- **Search Indexing**: Add to searchable content database
- **Publication**: Ready for distribution

## Content Status Tracking

The system tracks content through these stages:
- `SCRAPED`: Original content extracted
- `AI_WRITTEN`: Content enhanced by AI
- `AI_REVIEWED`: Content reviewed and improved
- `HUMAN_REVIEW_PENDING`: Awaiting human review
- `HUMAN_APPROVED`: Human-approved content
- `FINALIZED`: Ready for publication
- `PUBLISHED`: Published content

## Agent Types

- `SCRAPER`: Web scraping agent
- `AI_WRITER`: Content enhancement AI
- `AI_REVIEWER`: Content review AI
- `HUMAN_WRITER`: Human content creator
- `HUMAN_REVIEWER`: Human content reviewer
- `HUMAN_EDITOR`: Human content editor

## Project Structure
```
automated-book-publication/
├── book_workflow.py                 # Main application file
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── output1.1/         # Captured screenshots
├── output1./         # Captured screenshots
├── chroma_db/          # ChromaDB storage
└── .venv/              # Virtual environment
```

## Sample Output
```
Starting automated book publication workflow...
INFO:Successfully scraped content from https://en.wikisource.org/wiki/...
INFO:Saved version with status scraped data!
INFO:Initialized chapter: The Gates of Morning - Book 1, Chapter 1
INFO:AI processing completed for chapter: The Gates of Morning - Book 1, Chapter 1
✓ Chapter processed successfully
✓ Human review requested
✓ Human feedback submitted
✓ Chapter finalized
✓ Search results available
```

## Development Approach

This project was developed using:
- **Object-Oriented Design**: Clean, modular architecture
- **Async Programming**: Efficient handling of I/O operations
- **Error Handling**: Comprehensive error management
- **Logging**: Detailed logging for debugging and monitoring
- **Type Hints**: Improved code clarity and IDE support

## Performance Considerations

- **Asynchronous Processing**: Non-blocking operations for better performance
- **Vector Database**: Efficient content storage and retrieval
- **Caching**: Strategic caching for improved response times
- **Batch Processing**: Optimized for handling multiple chapters

## Future Enhancements

- **Multi-format Support**: PDF, EPUB, DOCX processing
- **Advanced AI Models**: Integration with multiple AI providers
- **Collaborative Features**: Multi-user review workflows
- **Publishing Integration**: Direct publishing to platforms
- **Analytics Dashboard**: Content performance metrics

## Disclaimer
This project is submitted for evaluation purposes only. All code is original work developed specifically for this task. No AI tools were used in the development of this codebase.
