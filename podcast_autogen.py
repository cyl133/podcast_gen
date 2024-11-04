import os
import json
import time
import re
from typing import Union, Literal
import openai
from autogen import AssistantAgent, UserProxyAgent, Agent
import autogen
from sqlalchemy import create_engine, text
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Configuration Parameters
OPENAI_API_KEY = ''
DB_CONNECTION_STRING = ''

# Story Parameters
NUM_TOP_STORIES = 2

# Chat Parameters
MAX_CHAT_TURNS = 5
MAX_CHAT_STEPS = 10
CHAT_SUMMARY_FILE = "chat_summary.txt"
PODCAST_SCRIPT_FILE = "podcast_script.txt"

# Embedding Parameters
EMBEDDINGS_CACHE_FILE = 'embeddings_cache.json'
EMBEDDING_BATCH_SIZE = 20
EMBEDDING_MODEL = "text-embedding-ada-002"

# Clustering Parameters
MIN_CLUSTER_SIZE = 0
CLUSTERING_THRESHOLD = 0.95
TOP_CLUSTERS_COUNT = 30

# LLM Models
CHAT_MODEL = "gpt-4o"
HEADLINE_MODEL = "o1-preview"

# Required keys for personas
REQUIRED_PERSONA_KEYS = [
    "Name",
    "Role",
    "Background",
    "Areas of expertise relevant to the article",
    "Personality traits for engaging conversation",
    "Speaking style",
    "Potential biases or perspectives on the topics",
    "Typical emotional responses or reactions"
]

CLUSTER_SUMMARIES_CACHE_FILE = 'cluster_summaries_cache.json'

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Define required keys for personas
REQUIRED_PERSONA_KEYS = [
    "Name",
    "Role",
    "Background",
    "Areas of expertise relevant to the article",
    "Personality traits for engaging conversation",
    "Speaking style",
    "Potential biases or perspectives on the topics",
    "Typical emotional responses or reactions"
]

# Function to load the news article from a plain text file
def load_news_article(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

# Load the news article content from input_article.txt
print("Loading news article content from input_article.txt...")
news_article_content = load_news_article("input_article.txt")
print("News Article Content Loaded:\n", news_article_content[:200], "...") 

def parse_json(response_text):
    """Parse JSON with improved error handling and cleaning."""
    # Clean the response text
    if not response_text:
        print("Empty response text")
        return None
        
    # Remove code block markers if present
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```") and cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[3:-3].strip()
        # Remove language identifier if present (e.g., ```json)
        if ' ' in cleaned_text.split('\n')[0]:
            cleaned_text = '\n'.join(cleaned_text.split('\n')[1:])
        elif cleaned_text.split('\n')[0].lower() in ['json', 'javascript']:
            cleaned_text = '\n'.join(cleaned_text.split('\n')[1:])
    
    # Try to find JSON content within the text
    try:
        # Look for content between curly braces
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            cleaned_text = cleaned_text[start_idx:end_idx + 1]
        
        # Try to parse the JSON
        parsed_response = json.loads(cleaned_text)
        print("Parsed JSON successfully")
        return parsed_response
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        try:
            # Try parsing as array if the content starts with [
            if cleaned_text.strip().startswith('['):
                start_idx = cleaned_text.find('[')
                end_idx = cleaned_text.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    array_text = cleaned_text[start_idx:end_idx + 1]
                    parsed_response = json.loads(array_text)
                    print("Parsed JSON array successfully")
                    return parsed_response
        except:
            pass
        
        print("All JSON parsing attempts failed")
        return None

# Function to identify the top headlines based on the criteria
def identify_top_headlines(article_content, headlines_list, num_headlines=NUM_TOP_STORIES, title_to_summary=None):
    prompt = f"""
You are an expert news podcast writer.

Given the following [Article Content] and [Side Headlines and Summaries], identify {num_headlines} news articles FROM [Side Headlines and Summaries] that are:

- Most INSIGHFUL and NATURAL to discuss TOGETHER WITH THE main podcast article under [Article Content], i.e. it links naturally to the MAIN_ARTICLE; host can create a very INSIGHTFUL and ENGAGING podcast episode with supplements from them, BUT NOT THE SAME AS THE [Article Content]

- NATURAL RELEVANCE is the KEY

**Output the result as a valid JSON array of {num_headlines} strings. Ensure the response is in JSON format without code blocks, extra text, or explanations.**

[
  "Selected Headline 1",
  "Selected Headline 2"
]

Do not include any additional text.

[Side Headlines and Summaries]:
{json.dumps([{
    "headline": headline,
    "summary": title_to_summary.get(headline, "No summary available")
} for headline in headlines_list], indent=2)}

[Article Content]:
\"\"\"
{article_content}
\"\"\"
"""
    print("Identifying top headlines...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    response_text = response.choices[0].message.content.strip()
    print("GPT Response:")
    print(response_text)
    top_headlines = parse_json(response_text)
    if top_headlines and len(top_headlines) == num_headlines:
        print("Top Headlines Identified:\n", json.dumps(top_headlines, indent=2))
        return top_headlines
    else:
        print(f"Failed to parse {num_headlines} top headlines.")
    return []

# Function to summarize the aggregated summaries for each cluster
def summarize_cluster(titles, summaries, use_cache=True):
    """Generate a comprehensive summary for a cluster of related articles with optional caching."""
    if use_cache:
        cache_key = "|".join(sorted(titles[:3]))
        
        cache = {}
        if os.path.exists(CLUSTER_SUMMARIES_CACHE_FILE):
            try:
                with open(CLUSTER_SUMMARIES_CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            except json.JSONDecodeError:
                pass
        
        if cache_key in cache:
            return cache[cache_key]

    combined_text = f"""
Titles:
{json.dumps(titles, indent=2)}

Summaries:
{" ".join(summaries)}
"""
    prompt = f"""
You are an expert summarizer. Create a COMPREHENSIVE summary that:
1. Incorporates ALL unique points from the provided titles and summaries
2. Ignores any repeated information
3. Maintains specific details, numbers, and quotes when relevant
4. Uses the titles to understand the broader context and perspectives
5. Aims for 300-500 words in length

Text to summarize:
\"\"\"
{combined_text}
\"\"\"

Provide only the summary without any additional text or formatting.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert summarizer focused on comprehensive coverage while avoiding redundancy."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    summary = response.choices[0].message.content.strip()
    
    if use_cache:
        cache[cache_key] = summary
        try:
            with open(CLUSTER_SUMMARIES_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    return summary

# Updated function to extract top stories and cluster them semantically
def extract_top_story(article_content, num_headlines=2):
    print("Starting extract_top_story function")
    
    # Initialize database connection
    try:
        engine = create_engine(DB_CONNECTION_STRING)
    except Exception as e:
        print("Error creating database engine:", e)
        return []
    
    # Use the provided SQL query
    query = text("""
        SELECT title,
               STRING_AGG(summary, ' ') AS concatenated_summary,
               MAX(updated_at) AS latest_updated_at,
               COUNT(DISTINCT story_id) AS total_records
        FROM recommender.stories_info_wide
        WHERE summary IS NOT NULL 
          AND language = 'en'
          AND story_id NOT SIMILAR TO '[0-9]%'
          AND updated_at BETWEEN '2024-10-23' AND '2024-11-01'
        GROUP BY title
        ORDER BY total_records DESC, title ASC, latest_updated_at DESC;
    """)
    
    # Execute the query
    try:
        with engine.connect() as connection:
            result = connection.execute(query)
            headlines_data = result.fetchall()
        print(f"Query executed successfully. Number of rows fetched: {len(headlines_data)}")
    except Exception as e:
        print("Error executing query:", e)
        return []
    
    # Convert the result to a list of dictionaries
    headlines_list = []
    for row in headlines_data:
        headlines_list.append({
            'title': row[0],                   # title
            'concatenated_summary': row[1],    # concatenated_summary
            'latest_updated_at': row[2],       # latest_updated_at
            'total_records': row[3]            # total_records
        })
    
    # Load existing embeddings cache
    embeddings_cache = {}
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        with open(EMBEDDINGS_CACHE_FILE, 'r', encoding='utf-8') as f:
            embeddings_cache = json.load(f)

    # Perform semantic comparison
    titles = [item['title'] for item in headlines_list]

    titles_to_embed = []
    embeddings = []
    for title in titles:
        if title in embeddings_cache:
            embeddings.append(embeddings_cache[title])
        else:
            titles_to_embed.append(title)

    if titles_to_embed:
        for i in range(0, len(titles_to_embed), EMBEDDING_BATCH_SIZE):
            batch = titles_to_embed[i:i+EMBEDDING_BATCH_SIZE]
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [embedding_obj.embedding for embedding_obj in response.data]
            for title, embedding in zip(batch, batch_embeddings):
                embeddings_cache[title] = embedding
                embeddings.append(embedding)
        
        with open(EMBEDDINGS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(embeddings_cache, f)

    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)

    # Perform clustering
    print("Clustering titles based on embeddings...")
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=CLUSTERING_THRESHOLD, metric='euclidean') 
    clustering_model.fit(embeddings_array)
    labels = clustering_model.labels_

    # Add labels to headlines_list
    for i, item in enumerate(headlines_list):
        item['cluster'] = labels[i]

    print(f"Titles clustered. Number of clusters: {len(set(labels))}")

    # Aggregate data for each cluster
    clusters = {}
    for item in headlines_list:
        cluster_id = item['cluster']
        if cluster_id not in clusters:
            clusters[cluster_id] = {
                'titles': [item['title']],
                'summaries': [item['concatenated_summary']],
                'total_records': item['total_records']
            }
        else:
            clusters[cluster_id]['titles'].append(item['title'])
            clusters[cluster_id]['summaries'].append(item['concatenated_summary'])
            clusters[cluster_id]['total_records'] += item['total_records']
    
    # Filter out clusters with fewer than minimum size
    filtered_clusters = {k: v for k, v in clusters.items() if len(v['titles']) >= MIN_CLUSTER_SIZE}
    print(f"Filtered out {len(clusters) - len(filtered_clusters)} clusters with fewer than {MIN_CLUSTER_SIZE} records")

    # Sort filtered clusters by total_records in descending order
    sorted_clusters = sorted(filtered_clusters.items(), key=lambda x: x[1]['total_records'], reverse=True)[:TOP_CLUSTERS_COUNT]

    # Prepare the list of aggregated clusters
    aggregated_clusters = []
    for cluster_id, data in sorted_clusters:
        aggregated_clusters.append({
            'cluster_id': cluster_id,
            'titles': data['titles'],
            'concatenated_summary': data['summaries'],
            'total_records': data['total_records']
        })

    # Extract the representative title for each cluster and generate summaries
    cluster_titles = []
    cluster_summaries = []
    for data in aggregated_clusters:
        cluster_titles.append(data['titles'][0])
        cluster_summary = summarize_cluster(data['titles'], data['concatenated_summary'])
        cluster_summaries.append(cluster_summary)
        print(f"Generated summary for cluster with title: {data['titles'][0][:100]}...")

    # Create a mapping of titles to their summaries for later use
    title_to_summary = dict(zip(cluster_titles, cluster_summaries))

    # Use GPT to identify the top headlines based on the criteria
    relevant_headlines = identify_top_headlines(
        article_content, 
        cluster_titles, 
        num_headlines,
        title_to_summary
    )
    print("Relevant headlines identified:", relevant_headlines)

    # Retrieve the aggregated summaries for the selected headlines
    selected_summaries = []

    # Create a mapping from title to aggregated data for quick lookup
    title_to_cluster = {data['titles'][0]: data for data in aggregated_clusters}

    for headline in relevant_headlines:
        cluster_data = title_to_cluster.get(headline)
        if cluster_data:
            selected_summaries.append({
                'headline': headline,
                'summary': summarize_cluster(
                    cluster_data['titles'],
                    cluster_data['concatenated_summary'],
                    use_cache=True
                ),
                'total_records': cluster_data['total_records'],
                'titles_in_cluster': cluster_data['titles']
            })
        else:
            selected_summaries.append({
                'headline': headline,
                'summary': "Summary not found",
                'total_records': 0,
                'titles_in_cluster': []
            })

    print("Selected summaries with aggregated summaries:", selected_summaries)
    return selected_summaries

# Function to generate a persona for each identified headline
def generate_persona(article_content, headline, summary):
    prompt = f"""
Based on the following news article content, headline, and summary, generate a persona who is affected by BOTH events deeply.

The persona must be output strictly in JSON format with the following keys and exact names:

{{
  "Name": "Person's Name",
  "Role": "Their job or role (e.g., stock analyst, political commentator, health expert)",
  "Background": "Professional and personal background",
  "Areas of expertise relevant to the article": "Expertise areas",
  "Personality traits for engaging conversation": "Personality traits",
  "Speaking style": "Speaking style (e.g., formal, casual, technical)",
  "Potential biases or perspectives on the topics": "Any biases or perspectives",
  "Typical emotional responses or reactions": "Emotional responses or reactions"
}}

Do not include any additional text.

Article Content:
\"\"\"
{article_content}
\"\"\"

Headline:
\"\"\"
{headline}
\"\"\"

Summary:
\"\"\"
{summary}
\"\"\"
"""
    print(f"Generating persona for headline: '{headline}'")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are skilled at creating diverse and engaging personas with strict JSON formatting. Do not include any extra text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=700
    )
    response_text = response.choices[0].message.content.strip()
    persona = parse_json(response_text)
    if persona:
        # Validate persona structure and fill missing keys with placeholders
        missing_keys = [key for key in REQUIRED_PERSONA_KEYS if key not in persona]
        if missing_keys:
            print(f"Warning: Missing keys in generated persona: {missing_keys}. Adding placeholders.")
            for key in missing_keys:
                persona[key] = "Placeholder text for missing key"
        print("Generated Persona:\n", json.dumps(persona, indent=2))
        return persona
    else:
        print("Failed to parse persona.")
        print("Using placeholder persona.")
        placeholder_persona = {key: "Placeholder text" for key in REQUIRED_PERSONA_KEYS}
        return placeholder_persona

# Function to generate the persona prompt
def get_persona_prompt(name, persona_data, assigned_headline, assigned_summary):
    return f"""
You are {name}, a {persona_data['Role']}. {persona_data['Background']}
You discuss the main article and assigned news headline DIRECTLY and SPECIFICALLY, you hate flowery language and empty rhetoric; you focus solely on the practical aspects of things. 
Main Article:
\"\"\"
{news_article_content}
\"\"\"

Your Assigned News:
Headline: {assigned_headline}
Summary: {assigned_summary}

SPECIFICALLY DISCUSS THE DETAILS OF TRHE NEWS ONLY, DO NOT OVERGENERALIZE TO BROADER THEMES. QUOTE THE SPECIFIC ASPECT OF THE NEWS EACH TIME YOU SAY SOMETHING. BE VERY SPECIFIC.
Talk like an expert in a PODCAST.
Your areas of expertise include {persona_data['Areas of expertise relevant to the article']}.
You are known for being {persona_data['Personality traits for engaging conversation']}.
Your potential biases or perspectives on the topics are {persona_data['Potential biases or perspectives on the topics']}.

"""

# Define the CustomGroupChatManager and CustomGroupChat classes
class CustomGroupChatManager(autogen.GroupChatManager):
    def __init__(self, groupchat, **kwargs):
        super().__init__(groupchat, **kwargs)
        self.step_counter = 0
        self.max_steps = MAX_CHAT_STEPS

    def _process_received_message(self, message, sender, silent):
        self.step_counter += 1
        formatted_message = ""
        # Handle the case when message is a dictionary
        if isinstance(message, dict):
            if 'content' in message and message['content'].strip():
                content = message['content']
                formatted_message = f"[{sender.name}]: {content}"
            else:
                return super()._process_received_message(message, sender, silent)
        # Handle the case when message is a string
        elif isinstance(message, str) and message.strip():
            content = message
            formatted_message = f"[{sender.name}]: {content}"
        else:
            return super()._process_received_message(message, sender, silent)
        
        # Print the message
        print(formatted_message + "\n")
        time.sleep(1)
        
        # Save the conversation to a file
        with open(CHAT_SUMMARY_FILE, 'a', encoding='utf-8') as f:
            f.write(formatted_message + "\n")

        # Check if we've reached the maximum number of steps
        if self.step_counter >= self.max_steps:
            return "TERMINATE"

        return super()._process_received_message(message, sender, silent)

class CustomGroupChat(autogen.GroupChat):
    @staticmethod
    def custom_speaker_selection_func(
        last_speaker: Agent, groupchat: autogen.GroupChat
    ) -> Union[Agent, Literal['auto', 'manual', 'random', 'round_robin'], None]:
        # Return 'auto' to use the LLM-based selection
        return 'auto'

    select_speaker_message_template = """You are in a focus group. The following roles are available:
                    {roles}.
                    Read the following conversation.
                    Then select the next role from {agentlist} to play. Only return the role."""

# Generate the podcast script based on the discussion and the articles
def generate_podcast_script(article_content, discussion, assigned_headlines_summaries):
    # Add error handling and cleanup for JSON responses
    def clean_and_parse_json(response_text):
        try:
            # Remove any potential markdown code block markers
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```") and cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[3:-3].strip()
                if cleaned_text.startswith("json"):
                    cleaned_text = cleaned_text[4:].strip()
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return {"new_podcast": []}

    assigned_news_content = "\n".join([f"Headline: {item['headline']}\nSummary: {item['summary']}" for item in assigned_headlines_summaries])

    # First part using your original prompt
    part1_prompt = f"""
AIM FOR 1500 WORDS FOR THIS PART

FOR THIS PART, FOCUS MORE ON THE MAIN ARTICLE; as it ends, slowly and NATURALLY prepares for the second half, which discusses the [Main Article] TOGETHER with [Side News Articles].

Create THE FIRST HALF of a podcast script on the [Main Article]. Do not end it, we will concatenate it with the second half.

Do not jump around topics abruptly, transition naturally like a podcast host will do. BUT DO NOT DRAW LINKS THAT SEEM WEIRD, such as "It's like walking a tightrope. And speaking of tightropes..."

At all times, DO NOT STRAY FROM THE MAIN TOPIC. When discussing a point, drill into it deeply.

Do not repeat points already covered.

The second half of the podcast script will be generated by the next prompt - so do not end the podcast in the first half.

Return ONLY A VALID JSON object in this exact format, with no markdown formatting or additional text:
{{
  "new_podcast": [
    {{
      "role": "Speaker 1",
      "content": "First line of dialogue"
    }},
    {{
      "role": "Speaker 2",
      "content": "Response to first line"
    }}
  ]
}}

STRICT rules:
1. Speaker 1 is the HOST, Speaker 2 is the DOMAIN EXPERT, but don't explitcitly mention this.
1. Start and end the podcast as just two persons conversing, no mentioning of things like "episodes" or "podcasts" or "show" etc.
1. Host talk VERY ENGAGINGLY and COLLOQUIALLY, act exactly like a podcast host.
1. The response must start with {{ and end with }} - no other characters before or after
2. Each speaker turn must be a separate object in the array
3. Roles must alternate between "Speaker 1" and "Speaker 2"
4. All quotes and special characters must be properly escaped in JSON
5. Focus primarily on the [Main Article] content
6. Side articles' content should be introduced when talking about them, but do not mention 
7. Keep each speaker's content relatively short (2-4 sentences)
8. Content should focus on highly specific issues, avoiding generic statements
9. Use 2-space indentation
10. Ensure proper commas between array elements

[Main Article]:
\"\"\"
{article_content}
\"\"\"

[Side News Articles]:
\"\"\"
{assigned_news_content}
\"\"\"

"""

    print("Generating first part of podcast script...")
    response1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an EXPERT PODCST WRITER who creates VERY ENGAGING AND COLLOQUIAL podcast scripts in valid JSON format. Your output must be pure JSON with no markdown or additional formatting."},
            {"role": "user", "content": part1_prompt}
        ],
        temperature=0.6
    )

    print("First part OpenAI response:")
    print(response1)  # Debug print
    print("\nFirst part content:")
    print(response1.choices[0].message.content)  # Debug print
    
    try:
        first_part = clean_and_parse_json(response1.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"Error parsing first part: {e}")
        print("Raw content that failed to parse:")
        print(response1.choices[0].message.content)
        return "{}"  # Return empty JSON object on failure

    # Second part using the same prompt structure but with full context
    part2_prompt = f"""
Continue the podcast script by seamlessly extending the following conversation, with NATURAL LINKS to the [Side News Articles] . The continuation should feel natural and maintain the same tone, style, and depth of discussion. Do not repeat points already covered.

AIM FOR 1500 WORDS FOR THIS PART

Do not jump around topics abruptly, transition naturally like a podcast host will do. BUT DO NOT DRAW LINKS THAT SEEM WEIRD, such as "It's like walking a tightrope. And speaking of tightropes..."

Do not repeat points already covered.

At all times, DO NOT STRAY FROM THE MAIN TOPIC. When discussing a point, drill into it deeply.

Previous conversation for context:
{json.dumps(first_part["new_podcast"], indent=2)}

Return ONLY A VALID JSON object in this exact format, with no markdown formatting or additional text:
{{
  "new_podcast": [
    {{
      "role": "Speaker 1",
      "content": "First line of dialogue"
    }},
    {{
      "role": "Speaker 2",
      "content": "Response to first line"
    }}
  ]
}}

STRICT rules:
1. Speaker 1 is the HOST, Speaker 2 is the DOMAIN EXPERT, but don't explitcitly mention this.
1. Start and end the podcast as just two persons conversing, no mentioning of things like "episodes" or "podcasts" or "show" etc.
1. Host talk VERY ENGAGINGLY and COLLOQUIALLY, act exactly like a podcast host.
1. The response must start with {{ and end with }} - no other characters before or after
2. Each speaker turn must be a separate object in the array
3. Roles must alternate between "Speaker 1" and "Speaker 2"
4. All quotes and special characters must be properly escaped in JSON
5. Focus primarily on the [Main Article] content
6. Side articles should be introduced when talking about them.
7. Keep each speaker's content relatively short (2-4 sentences)
8. Content should focus on highly specific issues, avoiding generic statements
9. Use 2-space indentation
10. Ensure proper commas between array elements

[Main Article]:
\"\"\"
{article_content}
\"\"\"

[Side News Articles]:
\"\"\"
{assigned_news_content}
\"\"\"
"""
    print("Generating second part of podcast script...")
    response2 = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an EXPERT PODCAST WRITER who creates VERY ENGAGING AND COLLOQUIAL podcast scripts in valid JSON format. Your output must be pure JSON with no markdown or additional formatting."},
            {"role": "user", "content": part2_prompt}
        ],
        temperature=0.8
    )

    print("Second part OpenAI response:")
    print(response2)  # Debug print
    print("\nSecond part content:")
    print(response2.choices[0].message.content)  # Debug print
    
    try:
        second_part = clean_and_parse_json(response2.choices[0].message.content)
    except json.JSONDecodeError as e:
        print(f"Error parsing second part: {e}")
        print("Raw content that failed to parse:")
        print(response2.choices[0].message.content)
        return "{}"  # Return empty JSON object on failure

    # Combine both parts into one seamless script
    combined_script = {
        "new_podcast": (first_part.get("new_podcast", []) + 
                       second_part.get("new_podcast", []))
    }
    
    return json.dumps(combined_script, indent=2)

if __name__ == '__main__':
    # Extract top stories from the database
    print("Extracting top stories from the database...")
    top_stories = extract_top_story(news_article_content, num_headlines=NUM_TOP_STORIES)

    # Extract the headlines and summaries
    relevant_headlines = [story['headline'] for story in top_stories]
    summaries = [story['summary'] for story in top_stories]

    print("Relevant Headlines Identified:\n", json.dumps(relevant_headlines, indent=2))

    # Generate personas for each relevant headline
    personas = []
    for idx, story in enumerate(top_stories):
        persona = generate_persona(news_article_content, story['headline'], story['summary'])
        personas.append({'persona': persona, 'headline': story['headline'], 'summary': story['summary']})

    print("\nAll Generated Personas:\n", json.dumps(personas, indent=2))

    # Create AssistantAgent instances for each persona
    personas_agents = []
    for idx, persona_info in enumerate(personas):
        persona = persona_info['persona']
        assigned_headline = persona_info['headline']
        assigned_summary = persona_info['summary']
        sanitized_name = ''.join(e for e in persona['Name'] if e.isalnum() or e in ['_', '-'])
        if not sanitized_name:
            sanitized_name = f"Participant_{idx+1}"
        persona_prompt = get_persona_prompt(sanitized_name, persona, assigned_headline, assigned_summary)
        persona_agent = AssistantAgent(
            name=sanitized_name,
            system_message=f"{persona_prompt}\n\nImportant: Be direct, blunt, and opinionated in your responses. Do not sugarcoat or use generic language. Express your true views on the subjects without hesitation. Keep your responses concise (1-3 sentences) and insightful.",
            llm_config={"config_list": [{"model": "gpt-4o"}]},
            human_input_mode="NEVER"
        )
        personas_agents.append(persona_agent)
        print(f"Created AssistantAgent for persona: {sanitized_name}")

    # Create the moderator agent
    predefined_topic = f"""Discuss the the main article and your assigned news headlines in a pod-cast style (but with LONGER TEXTS).

    Main Article:
    \"\"\"
    {news_article_content}
    \"\"\"

    Assigned News Headlines and Summaries:
    {json.dumps([{ 'headline': story['headline'], 'summary': story['summary']} for story in top_stories], indent=2)}

    Each participant should focus on how the main article and their assigned news headline."""

    moderator_agent = AssistantAgent(
        name="Moderator",
        system_message=f''' 
        You are moderating a focus group discussion about the main article and assigned news headlines. KEEP TO LESS THAN 100 WORDS.

        Your role is to:
        1. Encourage participants to focus on how the main article and their assigned news headlines affect each other. Make sure all assigned news are covered.
        2. NOT ALLOW PARTICIPANTS TO COMEP UP WITH FORCED AND UNNATURAL CONNECTIONS
        Facilitate dynamic discussions where participants respond to each other's insights on these inter-impacts.
        3. Gently steer the conversation to ensure everyone participates and interacts.
        4. Highlight interesting points made by participants to stimulate further debate.

        Remember:
        - Your interventions should be minimal and aimed at enhancing participant interactions.
        - Encourage participants to explore differing viewpoints and challenge each other respectfully.
        ''',
        default_auto_reply="Reply `TERMINATE` if the task is done.",
        llm_config={"config_list": [{"model": "gpt-4o"}]},
        description="A Focus Group moderator encouraging participant interaction on inter-impacts between news articles.",
        is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
        human_input_mode="NEVER"
    )
    print("Moderator Agent created")

    # Create the user proxy agent (Admin)
    user_proxy = UserProxyAgent(
        name="Admin",
        human_input_mode="NEVER",
        system_message="Human Admin for the Focus Group.",
        max_consecutive_auto_reply=5,
        default_auto_reply="Reply `TERMINATE` if the task is done.",
        is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
        code_execution_config={"use_docker": False}
    )
    print("Admin User Proxy created")

    # Create the group chat
    groupchat = CustomGroupChat(
        agents=[user_proxy, moderator_agent] + personas_agents,
        messages=[],
        speaker_selection_method=CustomGroupChat.custom_speaker_selection_func,
        select_speaker_message_template=CustomGroupChat.select_speaker_message_template,
        max_round = 3
    )
    print("Custom Group Chat created")

    # Create the manager
    manager = CustomGroupChatManager(groupchat=groupchat, llm_config={"config_list": [{"model": CHAT_MODEL}]})
    print("Custom Group Chat Manager created")

    # Clear previous chat summary if exists
    if os.path.exists(CHAT_SUMMARY_FILE):
        os.remove(CHAT_SUMMARY_FILE)
    
    # Initiate the chat
    print("Starting group chat discussion...")
    moderator_agent.initiate_chat(
        manager,
        max_turns=MAX_CHAT_TURNS,
        message=f"Welcome everyone! Let's begin our focus group discussion. {predefined_topic}",
    )

    print(f"Focus group discussion completed after {manager.step_counter} steps.")

    # Read the discussion from the saved file
    with open(CHAT_SUMMARY_FILE, 'r', encoding='utf-8') as f:
        discussion = f.read()

    podcast_script = generate_podcast_script(news_article_content, discussion, top_stories)

    # Save the podcast script to a file
    with open(PODCAST_SCRIPT_FILE, 'w', encoding='utf-8') as f:
        f.write(podcast_script)

    print(f"\nPodcast script saved to '{PODCAST_SCRIPT_FILE}'")