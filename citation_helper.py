"""
Helper class for extracting and formatting citations in RAG-generated answers.
"""

import re
import nltk

# Explicitly download the required NLTK data
nltk.download("punkt")

from nltk.tokenize import sent_tokenize


class CitationHelper:
    def __init__(self):
        """Initialize the citation helper."""
        pass

    def extract_sentences_with_citations(self, answer, documents):
        """
        Extract sentences from the answer and match them with source documents.

        Args:
            answer: The generated answer text
            documents: List of source documents

        Returns:
            Dictionary containing the answer text and a list of citation tuples
            (sentence, doc_index, document_content)
        """
        # Tokenize the answer into sentences
        try:
            sentences = sent_tokenize(answer)
        except LookupError:
            # Fallback if NLTK tokenization fails
            print(
                "Warning: NLTK tokenization failed. Using simple split-based tokenization."
            )
            sentences = self._simple_sentence_tokenize(answer)

        citations = []
        for i, sentence in enumerate(sentences):
            # Skip very short sentences (likely not substantive)
            if len(sentence.split()) < 3:
                continue

            # Find the best matching document for this sentence
            doc_index, similarity_score = self._find_best_match(sentence, documents)

            # Only include citations if the match is reasonable
            if similarity_score > 0.3:  # You may need to adjust this threshold
                citations.append((sentence, doc_index, documents[doc_index]))

        return {"text": answer, "citations": citations}

    def _simple_sentence_tokenize(self, text):
        """Simple sentence tokenization as a fallback."""
        # Split on common sentence-ending punctuation followed by space and capital letter
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return sentences

    def _find_best_match(self, sentence, documents):
        """
        Find the document that best matches a given sentence.

        Args:
            sentence: A sentence from the answer
            documents: List of source documents

        Returns:
            Tuple of (best_match_index, similarity_score)
        """
        best_match_index = -1
        best_similarity = -1

        # Simple word overlap similarity
        sentence_words = set(self._normalize_text(sentence).split())

        for i, doc in enumerate(documents):
            doc_words = set(self._normalize_text(doc).split())

            # Skip empty documents
            if not doc_words:
                continue

            # Calculate Jaccard similarity
            intersection = len(sentence_words.intersection(doc_words))
            union = len(sentence_words.union(doc_words))

            similarity = intersection / union if union > 0 else 0

            # Check if sentence is a subset of document (indicating potential quote)
            doc_text = self._normalize_text(doc)
            sentence_text = self._normalize_text(sentence)

            # Boost similarity if all sentence words appear in the document
            if all(word in doc_text for word in sentence_text.split()):
                similarity += 0.2

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_index = i

        return best_match_index, best_similarity

    def _normalize_text(self, text):
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        return text

    def format_answer_with_citations(self, answer_with_citations):
        """
        Format an answer with its citations for presentation.

        Args:
            answer_with_citations: Dictionary from extract_sentences_with_citations

        Returns:
            Formatted answer string with citation markers
        """
        # Create a mapping of sentences to citation numbers
        citation_map = {}
        for i, (sentence, doc_index, _) in enumerate(
            answer_with_citations["citations"]
        ):
            citation_map[sentence] = i + 1

        # Add citation markers to the original text
        formatted_answer = answer_with_citations["text"]
        for sentence, citation_num in sorted(
            citation_map.items(),
            key=lambda x: len(x[0]),
            reverse=True,  # Process longer sentences first to avoid partial matches
        ):
            # Replace the sentence with the sentence plus citation marker
            # Only if the sentence is not already cited
            if f"[{citation_num}]" not in formatted_answer:
                formatted_answer = formatted_answer.replace(
                    sentence, f"{sentence} [{citation_num}]"
                )

        # Add the citation list at the end
        if answer_with_citations["citations"]:
            formatted_answer += "\n\nSources:"
            for i, (_, doc_index, _) in enumerate(answer_with_citations["citations"]):
                formatted_answer += f"\n[{i+1}] Document {doc_index+1}"

        return formatted_answer
