from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY

pdf_path = 'submission_deliverable.pdf'
doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
styles = getSampleStyleSheet()
styles['Heading1'].fontName = 'Helvetica-Bold'
styles['Heading1'].fontSize = 16
styles['Heading1'].leading = 20
styles['Heading1'].spaceAfter = 12
styles['Heading1'].textColor = colors.HexColor('#1f4e79')
styles['Heading2'].fontName = 'Helvetica-Bold'
styles['Heading2'].fontSize = 12
styles['Heading2'].leading = 14
styles['Heading2'].spaceAfter = 8
styles['Heading2'].textColor = colors.HexColor('#2f4f4f')
styles['BodyText'].fontName = 'Helvetica'
styles['BodyText'].fontSize = 10.5
styles['BodyText'].leading = 13.5
styles['BodyText'].alignment = TA_JUSTIFY
styles['BodyText'].spaceAfter = 6
styles.add(ParagraphStyle(name='Caption', parent=styles['BodyText'], fontName='Helvetica-Oblique', fontSize=9.5, leading=12, textColor=colors.grey, spaceAfter=8))

story = []
story.append(Paragraph('Glance ML Internship Assignment Submission', styles['Heading1']))
story.append(Paragraph('Multimodal Fashion & Context Retrieval', styles['Caption']))
story.append(Spacer(1, 8))

story.append(Paragraph('1. Approaches: Possible ways to solve this problem, tradeoffs, and when they are useful', styles['Heading2']))
story.append(Paragraph('Fashion image retrieval can be approached in several ways, each with different strengths and costs.', styles['BodyText']))

bullets = [
    'Simple CLIP-only retrieval: encode the full natural language query and the image into a shared embedding space. Good for broad semantic matching, but often struggles with compositional queries such as color + garment + location.',
    'Attribute decomposition approach: split the query into sub-queries such as color, clothing, location, and style, and score each separately before fusing them. This improves precision for complex fashion prompts, but requires a more careful parsing strategy.',
    'Cross-encoder or re-ranker approach: use a stronger ranking model on top of a first-stage retrieval system. This can improve precision substantially, but adds cost and latency and is harder to run at scale.',
    'Hybrid retrieval with vector database: combine an efficient ANN/vector search step with a reranking stage. This is the best balance of speed and relevance for practical applications, especially when the image collection grows large.'
]
story.append(ListFlowable([ListItem(Paragraph(item, styles['Bullet']), bulletColor=colors.HexColor('#1f4e79')) for item in bullets], bulletType='bullet', bulletColor=colors.HexColor('#1f4e79')))
story.append(Spacer(1, 10))

story.append(Paragraph('2. Short Write-up on Chosen Approach', styles['Heading2']))
text = '''The chosen architecture uses CLIP embeddings for text and images, combined with a query parser that decomposes the incoming request into attribute-specific sub-queries. The system first encodes the full query and the extracted attributes, then retrieves candidate images using ChromaDB and re-ranks them with a weighted fusion strategy. This design handles fashion queries well because it preserves both the overall semantics of the request and the more discriminative fashion-specific signals such as color, garment type, location, and style.'''
story.append(Paragraph(text, styles['BodyText']))
story.append(Paragraph('How it handles fashion queries:', styles['BodyText']))
sub_bullets = [
    'The parser extracts structured attributes from natural language prompts.',
    'CLIP generates embeddings for the full query and for each attribute axis.',
    'The vector database provides a fast candidate pool for retrieval.',
    'A fusion score combines the global semantic match with attribute-focused matches to produce a more accurate ranking.',
    'The result is a system that is both interpretable and practical for fashion search use cases.'
]
story.append(ListFlowable([ListItem(Paragraph(item, styles['Bullet']), bulletColor=colors.HexColor('#1f4e79')) for item in sub_bullets], bulletType='bullet', bulletColor=colors.HexColor('#1f4e79')))
story.append(Spacer(1, 10))

story.append(Paragraph('3. Codebase (GitHub) Link', styles['Heading2']))
text = '''The implementation is organized into clearly separated indexing and retrieval modules. The indexing pipeline builds image embeddings and stores them in a persistent vector database, while the retrieval pipeline parses queries, encodes text, ranks candidates, and serves results through a Gradio demo. The codebase is available in the workspace and is ready to be pushed to GitHub. Please replace the placeholder link below with the public GitHub repository URL after publishing the project.'''
story.append(Paragraph(text, styles['BodyText']))
story.append(Paragraph('GitHub Link: https://github.com/rathodakashnayak', styles['BodyText']))
story.append(Spacer(1, 10))

story.append(Paragraph('4. Approaches for Future Work', styles['Heading2']))
story.append(Paragraph('a. Extending the solution for locations and weather', styles['BodyText']))
text = '''To support locations and weather, the system can be extended by adding metadata tags for city, place, season, and climate at index time. The parser can also be expanded to detect location terms and weather descriptors, and the retrieval stage can filter or re-rank candidates using these metadata fields. A stronger version could combine vision-based scene understanding with structured metadata for better context-aware retrieval.'''
story.append(Paragraph(text, styles['BodyText']))
story.append(Paragraph('b. Improving precision', styles['BodyText']))
text = '''Precision can be improved by fine-tuning CLIP on a fashion-specific dataset, adding a dedicated reranker for difficult queries, or using more detailed region-based representations for garments and accessories. Another useful improvement is to include richer metadata such as category, color, style, and occasion in the index, allowing more precise filtering before ranking.'''
story.append(Paragraph(text, styles['BodyText']))

story.append(PageBreak())
story.append(Paragraph('Submission Note', styles['Heading2']))
story.append(Paragraph('This document summarizes the approach, architecture, implementation structure, and future directions for the proposed fashion retrieval system.', styles['BodyText']))

doc.build(story)
print(f'Created {pdf_path}')
