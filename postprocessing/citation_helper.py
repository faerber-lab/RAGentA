"""
Helper class for extracting and formatting citations in RAG-generated answers.
Pure regex-based implementation with no external dependencies.
"""

import re


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
        # Tokenize the answer into sentences using regex
        sentences = self._regex_sentence_tokenize(answer)

        citations = []
        for i, sentence in enumerate(sentences):
            # Skip very short sentences (likely not substantive)
            if len(sentence.split()) < 3:
                continue

            # Find the best matching document for this sentence
            doc_index, similarity_score = self._find_best_match(sentence, documents)

            # Only include citations if the match is reasonable
            if similarity_score > 0.15:  # Lowered threshold for better coverage
                citations.append((sentence, doc_index, documents[doc_index]))

        return {"text": answer, "citations": citations}

    def _regex_sentence_tokenize(self, text):
        """Split text into sentences using regex patterns."""
        # Handle common abbreviations
        text = re.sub(r"([A-Za-z]\.[A-Za-z]\.)", r"\1<POINT>", text)  # e.g., i.e.
        text = re.sub(
            r"([A-Za-z][rsd]\.)(\s+[A-Z])", r"\1<POINT>\2", text
        )  # Mr., Mrs., Dr.

        # Split on sentence-ending punctuation followed by space and capital letter
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)

        # Process each part
        sentences = []
        for part in parts:
            if not part.strip():
                continue

            # Handle quoted sentences within a sentence
            if '"' in part and part.count('"') % 2 == 0:
                sentences.append(part)
            else:
                # Handle edge cases where we might have multiple sentences
                subparts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", part)
                sentences.extend([sp for sp in subparts if sp.strip()])

        # Restore special cases
        sentences = [s.replace("<POINT>", "") for s in sentences]

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
        best_match_index = 0  # Default to first document
        best_similarity = 0.01  # Minimal default similarity

        # Clean sentence for comparison
        sentence_clean = self._normalize_text(sentence)
        sentence_words = set(sentence_clean.split())

        # Find rare/significant words in the sentence
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "of",
            "to",
            "in",
            "for",
            "on",
            "with",
            "by",
            "at",
            "from",
            "as",
            "is",
            "are",
            "be",
            "been",
            "was",
            "were",
            "has",
            "have",
            "had",
            "can",
            "could",
            "will",
            "would",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
        }
        significant_words = [
            w for w in sentence_words if w not in common_words and len(w) > 3
        ]

        for i, doc in enumerate(documents):
            doc_clean = self._normalize_text(doc)
            doc_words = set(doc_clean.split())

            # Skip empty documents
            if not doc_words:
                continue

            # Calculate standard Jaccard similarity
            intersection = len(sentence_words.intersection(doc_words))
            union = len(sentence_words.union(doc_words))
            sim_jaccard = intersection / union if union > 0 else 0

            # Calculate ratio of significant words found
            if significant_words:
                sig_found = sum(1 for w in significant_words if w in doc_clean)
                sig_ratio = sig_found / len(significant_words)
            else:
                sig_ratio = 0

            # Check for exact phrases (3+ words in sequence)
            exact_match_bonus = 0
            if len(sentence_words) >= 3:
                # Look for exact matches of 3+ word sequences
                for j in range(len(sentence_clean.split()) - 2):
                    phrase = " ".join(sentence_clean.split()[j : j + 3])
                    if (
                        phrase in doc_clean and len(phrase) > 10
                    ):  # Only substantial phrases
                        exact_match_bonus = 0.3
                        break

            # Combine scores with weights
            similarity = (0.4 * sim_jaccard) + (0.5 * sig_ratio) + exact_match_bonus

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_index = i

        return best_match_index, best_similarity

    def _normalize_text(self, text):
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)  # Replace punctuation with spaces
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        return text.strip()

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

        # If we don't have any citations, return the original text with a note
        if not citation_map:
            return (
                answer_with_citations["text"]
                + "\n\nNote: No specific source documents could be reliably cited for this information."
            )

        # Add citation markers to the original text
        formatted_answer = answer_with_citations["text"]

        # Sort sentences by length (descending) to avoid partial matches
        for sentence, citation_num in sorted(
            citation_map.items(), key=lambda x: len(x[0]), reverse=True
        ):
            # Make sure we're not adding citations inside citations
            if f"[{citation_num}]" not in formatted_answer:
                # Only replace exact sentence matches
                formatted_answer = re.sub(
                    r"(" + re.escape(sentence) + r")(?!\[\d+\])",
                    r"\1 [" + str(citation_num) + "]",
                    formatted_answer,
                    count=1,
                )

        # Add the citation list at the end
        formatted_answer += "\n\nSources:"
        for i, (_, doc_index, _) in enumerate(answer_with_citations["citations"]):
            formatted_answer += f"\n[{i+1}] Document {doc_index+1}"

        return formatted_answer
