import os
import json
from sqlalchemy import create_engine, text
from podcast_autogen import (
    extract_top_story, 
    generate_podcast_script, 
    CustomGroupChat,
    CustomGroupChatManager,
    AssistantAgent,
    UserProxyAgent,
    generate_persona,
    get_persona_prompt
)

def get_stories_with_podcast():
    try:
        engine = create_engine('')
        query = text("""
            SELECT title, story_id, podcast, updated_at, summary 
            FROM recommender.stories_info_wide 
            WHERE story_id in (
                'CAAqNggKIjBDQklTSGpvSmMzUnZjbmt0TXpZd1NoRUtEd2pWeU9EUERCRU5KVTVuWEVOUEJDZ0FQAQ',
                'CAAqNggKIjBDQklTSGpvSmMzUnZjbmt0TXpZd1NoRUtEd2lNdzZmUkRCRjNtOTVxaFlDYW1TZ0FQAQ',
                'CAAqNggKIjBDQklTSGpvSmMzUnZjbmt0TXpZd1NoRUtEd2o0d19UUURCRVJraXFyYUZWQzZ5Z0FQAQ'
            )
            AND podcast != '{}'
            AND language = 'en'
            ORDER BY updated_at DESC, language DESC 
            LIMIT 10
        """)
        
        with engine.connect() as connection:
            result = connection.execute(query)
            stories = []
            for row in result:
                stories.append({
                    'title': row.title,
                    'story_id': row.story_id,
                    'podcast': row.podcast,
                    'updated_at': str(row.updated_at),
                    'summary': row.summary
                })
            return stories
    except Exception as e:
        print(f"Error executing query: {e}")
        return []

def process_stories_and_generate_podcasts():
    stories = get_stories_with_podcast()
    if not stories:
        print("No stories found")
        return
    
    # Process each story
    for idx, story in enumerate(stories):
        print(f"\nProcessing story {idx + 1}/{len(stories)}: {story['title']}")
        try:
            # Use the story's summary as the main article content
            news_article_content = story['summary']
            
            # Get related stories
            top_stories = extract_top_story(news_article_content)
            
            # Generate new podcast script directly without discussion
            new_podcast = generate_podcast_script(news_article_content, "", top_stories)
            
            # Add new podcast to story dict
            story['new_podcast'] = new_podcast
            
            print(f"Successfully generated new podcast for story: {story['title']}")
            
            # Save intermediate results after each successful processing
            intermediate_file = 'stories_with_new_podcasts_intermediate.json'
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(stories, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"Error processing story {story['title']}: {str(e)}")
            story['new_podcast'] = f"Error generating podcast: {str(e)}"
            continue
    
    # Save final results
    output_file = 'stories_with_new_podcasts.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stories, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(stories)} stories. Results saved to {output_file}")
    print("Stories with errors:", [s['title'] for s in stories if 'Error generating podcast' in s.get('new_podcast', '')])

if __name__ == '__main__':
    process_stories_and_generate_podcasts()