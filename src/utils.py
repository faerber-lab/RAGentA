import os
import json
from datetime import datetime

def create_sample_documents(output_dir="data/documents", count=20):
    """Create sample documents for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample topics and content for testing
    topics = [
        "Paris", "France", "Eiffel Tower", 
        "Machine Learning", "Neural Networks", "Deep Learning",
        "Solar System", "Mars", "Jupiter",
        "Climate Change", "Global Warming", "Renewable Energy"
    ]
    
    for i in range(count):
        topic = topics[i % len(topics)]
        filename = f"document_{i+1}.txt"
        
        # Create sample content
        content = f"Document about {topic}.\n\n"
        content += f"This is an example document about {topic} for testing the MAIN-RAG system.\n"
        content += f"It contains information related to {topic} that may be useful for answering queries.\n"
        
        # Add some additional content based on topic
        if "Paris" in topic or "France" in topic or "Eiffel" in topic:
            content += "Paris is the capital of France. The Eiffel Tower is located in Paris and was completed in 1889.\n"
        elif "Machine" in topic or "Neural" in topic or "Deep" in topic:
            content += "Machine Learning is a subfield of AI. Neural Networks are a popular approach in Deep Learning.\n"
        elif "Solar" in topic or "Mars" in topic or "Jupiter" in topic:
            content += "The Solar System consists of the Sun and objects that orbit it. Mars and Jupiter are planets in our Solar System.\n"
        elif "Climate" in topic or "Global" in topic or "Renewable" in topic:
            content += "Climate Change refers to long-term shifts in temperatures and weather patterns. Renewable Energy sources include solar and wind power.\n"
        
        # Write to file
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Created {count} sample documents in {output_dir}")

def save_results(query, answer, debug_info, output_dir="results"):
    """Save query results to a file for analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result_{timestamp}.json"
    
    result = {
        "query": query,
        "answer": answer,
        "debug_info": {
            "scores": debug_info["scores"],
            "tau_q": debug_info["tau_q"],
            "adjusted_tau_q": debug_info["adjusted_tau_q"],
            "filtered_count": len(debug_info["filtered_docs"])
        }
    }
    
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {os.path.join(output_dir, filename)}")
