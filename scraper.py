
import os
import json
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


def scrape_blog(blog_url):
    """
    Scrapes all articles from a Blogspot blog.
    """
    print("Scraping blog...")
    all_urls = []
    sitemap_url = f"{blog_url}/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'lxml') # Parse as XML
        for loc in soup.find_all('loc'):
            url = loc.get_text()
            # Filter for blog post URLs (typically contain year/month in Blogspot)
            print(f"Processing URL: {url}")
            if re.search(r'/\d{4}/\d{2}/[a-zA-Z0-9-]+\.html$', url):
                print(f"  - MATCH: {url}")
                all_urls.append(url)
            else:
                print(f"  - NO MATCH: {url}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
        # Fallback to archive scraping if sitemap fails (optional, but good for robustness)
        print("Falling back to archive scraping...")
        for year in range(2015, 2025): # Adjust the range as needed
            for month in range(1, 13):
                archive_url = f"{blog_url}/{year}/{month:02d}/"
                try:
                    response = requests.get(archive_url)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        article_links = soup.select('h3.post-title a')
                        for link in article_links:
                            all_urls.append(link['href'])
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching {archive_url}: {e}")
    
    # Remove duplicate URLs
    all_urls = list(set(all_urls))
    
    print(f"Found {len(all_urls)} article URLs.")
    
    # Scrape each article
    articles = []
    for url in all_urls:
        article_data = extract_article_data(url)
        if article_data:
            articles.append(article_data)
            
    return articles

def extract_article_data(article_url):
    """
    Extracts title, content, and other metadata from a single article.
    """
    try:
        response = requests.get(article_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title_element = soup.select_one('h3.post-title')
            title = title_element.get_text(strip=True) if title_element else "No Title"
            
            content_element = soup.select_one('.post-body')
            content = content_element.get_text(strip=True) if content_element else ""
            
            return {
                "url": article_url,
                "title": title,
                "content": content,
                "processed_text": process_text(content)
            }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching article {article_url}: {e}")
        return None

def process_text(text):
    """
    Cleans and preprocesses text for NLP tasks.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Tokenize
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove punctuation and stop words, and lemmatize
    processed_words = [lemmatizer.lemmatize(w) for w in words if w.isalpha() and w not in stop_words]
    
    return " ".join(processed_words)

def get_topics(articles, num_clusters=10):
    """
    Identifies topics from a collection of articles using K-Means clustering.
    """
    print("Clustering articles to identify topics...")
    
    if not articles:
        return []
        
    vectorizer = TfidfVectorizer(max_features=1000, max_df=0.8, min_df=5)
    tfidf_matrix = vectorizer.fit_transform([article['processed_text'] for article in articles])
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    
    # Assign cluster labels to articles
    for i, article in enumerate(articles):
        article['cluster'] = kmeans.labels_[i]
        
    # Get top terms for each cluster
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    topics = []
    for i in range(num_clusters):
        topic_terms = [terms[ind] for ind in order_centroids[i, :10]]
        topics.append({"id": i, "name": f"Topic {i+1}", "keywords": topic_terms})
        
    return topics

def create_graph_data(articles, topics):
    """
    Creates the graph-data.json file from the scraped articles and topics.
    """
    print("Creating graph data...")
    
    nodes = []
    links = []
    
    # Create a color palette for the clusters
    colors = ["#f94144", "#f3722c", "#f8961e", "#f9c74f", "#90be6d", "#43aa8b", "#4d908e", "#577590", "#277da1", "#f94144"]

    # Add article nodes
    for i, article in enumerate(articles):
        nodes.append({
            "id": f"article_{i}",
            "label": article['title'],
            "url": article['url'],
            "cluster": f"Topic {article['cluster'] + 1}",
            "color": colors[article['cluster'] % len(colors)]
        })
        
    # Create links based on shared cluster
    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            if articles[i]['cluster'] == articles[j]['cluster']:
                links.append({
                    "source": f"article_{i}",
                    "target": f"article_{j}"
                })

    with open('graph-data.json', 'w') as f:
        json.dump({"nodes": nodes, "links": links}, f, indent=2)


if __name__ == '__main__':
    BLOG_URL = "https://organizing-india.blogspot.com"
    
    # Scrape all articles from the blog
    articles = scrape_blog(BLOG_URL)
    
    # Extract topics from the articles
    topics = get_topics(articles)
    
    # Create the graph-data.json file
    create_graph_data(articles, topics)
    
    print("Scraping and data generation complete!")
