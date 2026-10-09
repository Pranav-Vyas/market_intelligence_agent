SaaS Competitor Intelligence & Market Gap Scout
This is more focused on understanding an entire market rather than one company.
What it does
Suppose the user enters:
"AI meeting assistants"

The system discovers companies such as:
- Otter
- Fireflies
- Fathom
- Granola
- etc.
Then it investigates them.
The final output could look conceptually like:
AI Meeting Assistant Market

Competitors
────────────────────────
Company A
Company B
Company C
Company D

Feature comparison
────────────────────────
Transcription
Speaker identification
Meeting summaries
Action items
CRM integration
...

Pricing comparison
────────────────────────
Free
Pro
Business
Enterprise

Customer complaints
────────────────────────
1. Poor accuracy
2. Expensive plans
3. Weak integrations
4. Privacy concerns
...

Potential market gaps
────────────────────────
1. Affordable solution for small teams
2. Better calendar integration
3. Better multilingual support

Where does the data come from?
This project needs more diverse data.
Information	Source
Competitors	Search engines
Product features	Official websites
Pricing	Official pricing pages
Product descriptions	Official websites
Customer complaints	Reddit
Reviews	G2, Trustpilot, app stores, forums
Product comparisons	Public review/comparison sites
News	Search/news sources
Company information	Official websites


The official company website should generally be preferred for factual product information, while customer communities/review platforms are useful for discovering complaints.
Where does RAG fit?
This project is a very good place to use your Hybrid RAG idea.
For example, you collect hundreds/thousands of documents:
Competitor websites
Reviews
Reddit posts
Articles
Product documentation
        ↓
Document processing
        ↓
Chunking
        ↓
BM25 + Vector embeddings
        ↓
Hybrid retrieval
        ↓
Reranker
        ↓
Relevant evidence

Then the agent uses that evidence to reason about the market.
Where does the agent fit?
The agent controls the research process.
For example:
User:
"Analyze the AI calendar market"

          ↓

Agent
          ↓
Find competitors
          ↓
Are there enough competitors?
          ↓
Search pricing
          ↓
Search customer complaints
          ↓
Retrieve relevant reviews
          ↓
Identify recurring problems
          ↓
Verify important findings
          ↓
Generate market-gap report

So:
RAG = retrieves evidence
LLM = analyzes/reasons over evidence
Agent = decides what research action to take next
How do you identify a market gap?
This is important because you don't want the LLM simply inventing one.
A better approach is:
Customer reviews
       ↓
Extract complaints
       ↓
Normalize complaints
       ↓
Cluster similar complaints
       ↓
Calculate frequency
       ↓
Measure severity
       ↓
Check competitor coverage
       ↓
Generate potential gaps
       ↓
Verify with evidence
       ↓
Rank opportunities

For example:
Pain point:
"Poor integration with Google Calendar"

Frequency:
18% of relevant complaints

Competitor coverage:
2/7 competitors provide strong support

Severity:
High

Evidence:
12 supporting reviews

Opportunity score:
8.4 / 10

That makes the "market gap" much more defensible.
Difficulty: Medium–high
I'd rate it around 7.5/10.
The individual components are manageable, but there are more moving pieces:
- Web search
- Scraping
- Review collection
- Document processing
- Hybrid retrieval
- Reranking
- Clustering
- Agent orchestration
- Evidence verification
- Opportunity scoring
The advantage is that you can build it incrementally.
Resume value
Very high.
It demonstrates almost everything you want in a strong GenAI project:
Agentic AI + RAG + retrieval + reranking + tool calling + information extraction + reasoning + evaluation.