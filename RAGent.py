import numpy as np
from tqdm import tqdm
import re
import copy
import logging
from typing import List, Dict, Tuple, Any
import os
import datetime
import random
import string

# Generate a unique ID for log filename
def get_unique_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"logs/ragent_{timestamp}_{random_str}.log"

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(get_unique_log_filename()), logging.StreamHandler()],
)
logger = logging.getLogger("RAGENT")

class EnhancedAgent4:
    """
    Enhanced Agent 4 implementation for claim analysis and improved
    follow-up question processing.
    """

    def __init__(self, agent_model, retriever):
        """
        Initialize the enhanced Agent 4.

        Args:
            agent_model: The LLM agent to use for generation
            retriever: Document retriever instance
        """
        self.agent = agent_model
        self.retriever = retriever

    def _create_claim_analysis_prompt(
        self, query, answer_with_citations, claims, filtered_documents
    ):
        """
        Create a prompt for analyzing claims against the question.
        """
        # Format documents with index
        docs_text = "\n\n".join(
            [
                f"Document {i+1}: {doc_text}"
                for i, (doc_text, _) in enumerate(filtered_documents)
            ]
        )

        # Format claims with their citations
        claims_text = "\n\n".join(
            [
                f"Claim {i+1}: {claim['text']} [Citations: {', '.join(map(str, claim['citations']))}]"
                for i, claim in enumerate(claims)
            ]
        )

        return f"""You are a meticulous judge evaluating whether a question has been fully answered by analyzing each claim in the answer.

TASK:
1. STRUCTURE ANALYSIS: First, determine if the original question naturally contains multiple distinct sub-questions or is a single question. DO NOT artificially break a single question into parts.
2. QUESTION PARSING: If the question naturally contains multiple sub-questions, identify each one. If it's a single question, treat it as one component.
3. CLAIM ANALYSIS: For each claim in the answer, determine which component(s) of the question it addresses, if any.
4. COVERAGE ASSESSMENT: Identify which components of the question are fully answered, partially answered, or not answered at all.
5. CLAIM RELEVANCE: Evaluate if each claim directly contributes to answering the question. Identify any irrelevant claims.

Original Question: {query}

Answer with Citations:
{answer_with_citations}

Individual Claims:
{claims_text}

Available Documents:
{docs_text}

Please provide your analysis in the following format:

QUESTION STRUCTURE: [SINGLE/MULTIPLE]

QUESTION COMPONENTS:
- Component 1: [First sub-question or the single question itself]
- Component 2: [Second sub-question, if applicable]
...

CLAIM ANALYSIS:
- Claim 1: [Whether this claim addresses any components, and which ones]
- Claim 2: [Whether this claim addresses any components, and which ones]
...

COVERAGE ASSESSMENT:
- Component 1: [FULLY ANSWERED / PARTIALLY ANSWERED / NOT ANSWERED]
- Component 2: [FULLY ANSWERED / PARTIALLY ANSWERED / NOT ANSWERED]
...

CLAIMS TO REMOVE:
[List the numbers of any claims that don't directly answer the question]

UNANSWERED COMPONENTS:
- [List components that are partially or not answered]

FOLLOW-UP QUESTIONS:
- [Rewrite each unanswered component as a complete, standalone question that preserves context]

Your analysis:"""

    def _create_follow_up_answer_prompt(
        self, original_query, follow_up_question, filtered_documents, previous_answer
    ):
        """
        Create prompt for generating an answer to a follow-up question that integrates well with previous answers.
        """
        # Format documents with index
        docs_text = "\n\n".join(
            [
                f"Document {i+1}: {doc_text}"
                for i, (doc_text, _) in enumerate(filtered_documents)
            ]
        )

        return f"""You are an accurate and reliable AI assistant. Your task is to answer a follow-up question based on the provided documents.

Original Question: {original_query}
Previous Answer: {previous_answer}
Follow-up Question: {follow_up_question}

Documents:
{docs_text}

IMPORTANT:
1. Answer the follow-up question ONLY if the documents contain the information.
2. If the documents don't contain the necessary information, explicitly state that you cannot answer this part based on the available information.
3. For each claim in your answer, add a citation in square brackets [X] where X is the document number.
4. NEVER mention documents in your text (like "Document 1 states...").
5. ONLY include information that directly addresses the follow-up question.
6. Be specific about which documents support each claim - do NOT cite all documents for every claim.

Follow-up Answer (with citations):"""

    def _create_answer_integration_prompt(
        self, original_query, previous_answer, new_answer_with_citations
    ):
        """
        Create a prompt for integrating a new answer with the previous answer.
        """
        return f"""You are an expert editor combining information to create a cohesive, comprehensive answer.

Original Question: {original_query}

Previous Answer:
{previous_answer}

New Answer with Citations:
{new_answer_with_citations}

Your task:
1. Create a unified answer that integrates both the previous answer and the new information.
2. The new answer should appear as a natural extension of the previous answer.
3. Ensure logical flow and coherence throughout the combined answer.
4. Remove any redundant information.
5. Do NOT mention that you're combining answers or adding new information.
6. Keep all citation markers [X] in the integrated answer.

Combined Answer (with citations):"""

    def extract_question_structure(self, claim_analysis):
        """Extract whether the question is naturally single or multiple."""
        structure_section = re.search(
            r"QUESTION STRUCTURE:\s*(SINGLE|MULTIPLE)", claim_analysis, re.DOTALL
        )
        if structure_section and structure_section.group(1):
            return structure_section.group(1)
        return "SINGLE"  # Default to single if not found

    def extract_question_components(self, claim_analysis):
        """Extract question components from the claim analysis."""
        components = []

        # Extract components using regex
        component_section = re.search(
            r"QUESTION COMPONENTS:(.*?)(?=CLAIM ANALYSIS:|$)", claim_analysis, re.DOTALL
        )

        if component_section:
            component_text = component_section.group(1).strip()
            component_lines = [
                line.strip() for line in component_text.split("\n") if line.strip()
            ]

            for line in component_lines:
                # Extract component text from lines like "- Component 1: What is X?"
                match = re.search(r"-\s*(?:Component \d+:)?\s*(.*)", line)
                if match:
                    component = match.group(1).strip()
                    if component:
                        components.append(component)

        return components

    def determine_if_completely_answered(self, claim_analysis, unanswered_components):
        """
        Determine if the question is completely answered with better handling of edge cases.

        Args:
            claim_analysis: The full claim analysis text
            unanswered_components: The already extracted unanswered components list

        Returns:
            Boolean indicating if question is completely answered
        """
        # First check the explicit COMPLETELY_ANSWERED field if available
        completely_answered_match = re.search(
            r"COMPLETELY_ANSWERED:\s*(Yes|No)", claim_analysis, re.IGNORECASE
        )

        if completely_answered_match:
            explicit_answer = completely_answered_match.group(1).strip().lower()
            if explicit_answer == "yes":
                logger.info("Question explicitly marked as completely answered")
                return True
            elif explicit_answer == "no":
                logger.info("Question explicitly marked as not completely answered")
                return False

        # If no explicit answer, check the unanswered components
        if not unanswered_components:
            return True

        # Check if the only "unanswered component" is actually "None"
        if (
            len(unanswered_components) == 1
            and unanswered_components[0].lower() == "none"
        ):
            return True

        # Check if all components are variations of "None"
        if all(comp.lower() == "none" for comp in unanswered_components):
            return True

        # If we got here, there are genuine unanswered components
        return False

    def extract_unanswered_components(self, claim_analysis):
        """Extract unanswered components from the claim analysis."""
        unanswered = []

        # Extract unanswered components using regex
        unanswered_section = re.search(
            r"UNANSWERED COMPONENTS:(.*?)(?=FOLLOW-UP QUESTIONS:|$)",
            claim_analysis,
            re.DOTALL,
        )

        if unanswered_section:
            unanswered_text = unanswered_section.group(1).strip()
            unanswered_lines = [
                line.strip() for line in unanswered_text.split("\n") if line.strip()
            ]

            for line in unanswered_lines:
                # Extract component text from lines like "- What is X?"
                match = re.search(r"-\s*(.*)", line)
                if match:
                    component = match.group(1).strip()
                    if component:
                        unanswered.append(component)

        return unanswered

    def extract_follow_up_questions(self, claim_analysis):
        """Extract genuine follow-up questions only."""
        follow_ups = []
        follow_up_section = re.search(
            r"FOLLOW-UP QUESTIONS:(.*?)(?=$)", claim_analysis, re.DOTALL
        )

        if follow_up_section:
            follow_up_text = follow_up_section.group(1).strip()

            # Only process if it's not "None"
            if follow_up_text.strip() and follow_up_text.strip().lower() != "none":
                for line in follow_up_text.split("\n"):
                    if line.strip() and line.strip().lower() != "none":
                        # Only match lines that start with a bullet or dash
                        match = re.search(r"^-\s*(.*)", line.strip())
                        if match:
                            question = match.group(1).strip()
                            # Additional validation - must be an actual question
                            if question and (
                                "?" in question
                                or question.lower().startswith("how")
                                or question.lower().startswith("what")
                                or question.lower().startswith("why")
                            ):
                                follow_ups.append(question)

        return follow_ups

    def extract_coverage_assessment(self, claim_analysis):
        """Extract coverage assessment from the claim analysis."""
        coverage = {}

        # Extract coverage assessment using regex
        coverage_section = re.search(
            r"COVERAGE ASSESSMENT:(.*?)(?=CLAIMS TO REMOVE:|UNANSWERED COMPONENTS:|$)",
            claim_analysis,
            re.DOTALL,
        )

        if coverage_section:
            coverage_text = coverage_section.group(1).strip()
            coverage_lines = [
                line.strip() for line in coverage_text.split("\n") if line.strip()
            ]

            for line in coverage_lines:
                # Extract component and status from lines like "- Component 1: FULLY ANSWERED"
                match = re.search(
                    r"-\s*(?:Component \d+:)?\s*(.*?):\s*(FULLY ANSWERED|PARTIALLY ANSWERED|NOT ANSWERED)",
                    line,
                )
                if match:
                    component = match.group(1).strip()
                    status = match.group(2).strip()
                    if component:
                        coverage[component] = status

        return coverage

    def extract_claims_to_remove(self, claim_analysis):
        """Extract claims to remove with better validation."""
        to_remove = []

        section = re.search(
            r"CLAIMS TO REMOVE:(.*?)(?=UNANSWERED COMPONENTS:|$)",
            claim_analysis,
            re.DOTALL,
        )
        if section:
            text = section.group(1).strip()

            # Skip if text says "None" or is empty
            if text and text.lower() != "none":
                # Extract claim numbers, not citation numbers
                # Look for explicit claim numbers or bullet points
                for line in text.split("\n"):
                    if line.strip():
                        claim_match = re.search(
                            r"(?:^|\s)(?:Claim\s+)?(\d+)(?:\s|:|$)", line
                        )
                        if claim_match:
                            to_remove.append(int(claim_match.group(1)))

        return to_remove

    def process_follow_up_questions(
        self,
        query,
        filtered_answer,
        follow_up_questions,
        unanswered_components,
        completely_answered,
        filtered_documents,
    ):
        """
        Process valid follow-up questions in a unified way.

        Args:
            query: The original question
            filtered_answer: The filtered answer with citations
            follow_up_questions: List of extracted follow-up questions
            unanswered_components: List of unanswered components
            completely_answered: Boolean indicating if question is completely answered
            filtered_documents: The filtered documents

        Returns:
            Tuple of (final_answer, follow_up_answers)
        """
        # If the question is completely answered, no need for follow-ups
        if completely_answered:
            logger.info(
                "Question is completely answered - skipping follow-up processing"
            )
            return self.remove_citations(filtered_answer), []

        # Initialize variables
        current_answer = self.remove_citations(filtered_answer)
        follow_up_answers = []
        excluded_ids = {doc_id for _, doc_id in filtered_documents}

        # Filter and validate follow-up questions
        valid_follow_ups = []
        for q in follow_up_questions:
            # Skip if None or empty
            if not q or q.lower() == "none":
                continue

            # Check if it looks like a question (has question mark or starts with question words)
            question_words = [
                "how",
                "what",
                "why",
                "when",
                "where",
                "which",
                "can",
                "should",
                "do",
            ]
            if "?" in q or any(q.lower().startswith(w) for w in question_words):
                valid_follow_ups.append(q)
            else:
                logger.info(f"Skipping invalid follow-up: '{q}'")

        # Only process if we have genuine unanswered parts and valid follow-ups
        has_unanswered_parts = unanswered_components and not (
            len(unanswered_components) == 1
            and unanswered_components[0].lower() == "none"
        )

        if has_unanswered_parts and valid_follow_ups:
            logger.info(f"Processing {len(valid_follow_ups)} valid follow-up questions")

            for i, follow_up_q in enumerate(valid_follow_ups):
                logger.info(f"Processing follow-up question {i+1}: {follow_up_q}")

                try:
                    # Retrieve new documents for follow-up question
                    new_docs = self.retriever.retrieve(
                        follow_up_q, top_k=20, exclude_ids=excluded_ids
                    )

                    # Update excluded IDs
                    for _, doc_id in new_docs:
                        excluded_ids.add(doc_id)

                    if new_docs:
                        # Generate answer for follow-up question
                        follow_up_prompt = self._create_follow_up_answer_prompt(
                            query, follow_up_q, new_docs, current_answer
                        )
                        follow_up_answer_with_citations = self.agent.generate(
                            follow_up_prompt
                        )
                        logger.info(
                            f"Follow-up answer with citations: {follow_up_answer_with_citations}"
                        )

                        # Skip integration if follow-up couldn't be answered
                        if (
                            "cannot answer" in follow_up_answer_with_citations.lower()
                            and "available information"
                            in follow_up_answer_with_citations.lower()
                        ):
                            logger.info(
                                "Follow-up question couldn't be answered - skipping integration"
                            )
                            follow_up_answers.append(
                                {
                                    "question": follow_up_q,
                                    "answer_with_citations": follow_up_answer_with_citations,
                                }
                            )
                            continue

                        # Integrate with previous answer
                        integration_prompt = self._create_answer_integration_prompt(
                            query, current_answer, follow_up_answer_with_citations
                        )
                        integrated_answer = self.agent.generate(integration_prompt)
                        logger.info(f"Integrated answer: {integrated_answer}")

                        # Update current answer for next iteration
                        current_answer = integrated_answer
                        follow_up_answers.append(
                            {
                                "question": follow_up_q,
                                "answer_with_citations": follow_up_answer_with_citations,
                            }
                        )
                except Exception as e:
                    logger.error(f"Error processing follow-up question: {e}")

        # Final answer is the last integrated answer
        final_answer = current_answer

        return final_answer, follow_up_answers

    def filter_claims(self, claims, claims_to_remove):
        """Remove irrelevant claims from the list."""
        if not claims_to_remove:
            return claims

        return [claim for i, claim in enumerate(claims, 1) if i not in claims_to_remove]

    def rebuild_answer_from_claims(self, filtered_claims):
        """Rebuild the answer text from the filtered claims with citations."""
        if not filtered_claims:
            return "Based on the available information, I cannot provide a relevant answer to this question."

        answer_parts = []
        for claim in filtered_claims:
            citation_str = ",".join(map(str, claim["citations"]))
            answer_parts.append(f"{claim['text']} [{citation_str}]")

        return " ".join(answer_parts)

    def remove_citations(self, text):
        """Remove citation brackets from the text."""
        return re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", text)

    def process_answer(self, query, answer_with_citations, claims, filtered_documents):
        """
        Process an answer to check if it fully addresses the question and handle follow-up questions.

        Args:
            query: The original question
            answer_with_citations: The answer with citations
            claims: List of extracted claims with their citations
            filtered_documents: The filtered documents used to generate the answer

        Returns:
            Tuple of (final_answer, debug_info)
        """
        # Handle empty claims case
        if not claims and answer_with_citations.strip():
            logger.info("No claims extracted - using fallback processing")
            # Create a single claim from the entire answer
            claims = [
                {
                    "text": re.sub(
                        r"Document\s+\d+,?\s*", "", answer_with_citations
                    ).strip(),
                    "citations": list(range(1, len(filtered_documents) + 1)),
                }
            ]

        logger.info(f"Processing {len(claims)} claims...")

        # Step 1: Perform claim analysis
        logger.info("Agent 4 performing claim analysis...")
        claim_analysis_prompt = self._create_claim_analysis_prompt(
            query, answer_with_citations, claims, filtered_documents
        )
        claim_analysis = self.agent.generate(claim_analysis_prompt)
        logger.info(f"Claim analysis response: {claim_analysis}")

        # Step 2: Extract analysis results
        question_structure = self.extract_question_structure(claim_analysis)
        question_components = self.extract_question_components(claim_analysis)
        unanswered_components = self.extract_unanswered_components(claim_analysis)
        follow_up_questions = self.extract_follow_up_questions(claim_analysis)
        coverage_assessment = self.extract_coverage_assessment(claim_analysis)
        claims_to_remove = self.extract_claims_to_remove(claim_analysis)

        logger.info(f"Question structure: {question_structure}")
        logger.info(f"Question components: {question_components}")
        logger.info(f"Unanswered components: {unanswered_components}")
        logger.info(f"Follow-up questions: {follow_up_questions}")
        logger.info(f"Coverage assessment: {coverage_assessment}")
        logger.info(f"Claims to remove: {claims_to_remove}")

        # Step 3: Filter out irrelevant claims
        filtered_claims = self.filter_claims(claims, claims_to_remove)
        filtered_answer = self.rebuild_answer_from_claims(filtered_claims)
        logger.info(f"Filtered answer: {filtered_answer}")

        # Step 4: Determine if the question is completely answered
        completely_answered = self.determine_if_completely_answered(
            claim_analysis, unanswered_components
        )
        logger.info(f"Question completely answered: {completely_answered}")

        # Step 5: If completely answered, return filtered answer
        if completely_answered:
            logger.info("Question is completely answered - returning filtered answer")

            # Prepare debug info with required keys
            debug_info = {
                "claim_analysis": claim_analysis,
                "question_structure": question_structure,
                "question_components": question_components,
                "unanswered_components": unanswered_components,
                "coverage_assessment": coverage_assessment,
                "follow_up_questions": follow_up_questions,
                "completely_answered": completely_answered,
                "follow_up_answers": [],
            }

            return filtered_answer, debug_info

        # Step 6: Process follow-up questions if needed
        final_answer, follow_up_answers_list = self.process_follow_up_questions(
            query,
            filtered_answer,
            follow_up_questions,
            unanswered_components,
            completely_answered,
            filtered_documents,
        )

        # Prepare debug info with required keys
        debug_info = {
            "claim_analysis": claim_analysis,
            "question_structure": question_structure,
            "question_components": question_components,
            "unanswered_components": unanswered_components,
            "coverage_assessment": coverage_assessment,
            "follow_up_questions": follow_up_questions,
            "completely_answered": completely_answered,
            "follow_up_answers": follow_up_answers_list,
        }

        # Perform final sanity check
        if "cannot provide" in final_answer.lower() and claims:
            # We have claims but the answer says we can't answer - use original filtered answer
            logger.info("Sanity check failed - reverting to filtered answer")
            final_answer = self.remove_citations(filtered_answer)

        return final_answer, debug_info


class RAGENT:
    def __init__(
        self,
        retriever,
        agent_model=None,
        n=0.0,
        falcon_api_key=None,
        pinecone_api_key=None,
    ):
        """
        Enhanced RAGent framework implementation with citation tracking and judgment.

        Args:
            retriever: Document retriever instance (Pinecone or other)
            agent_model: Model name or pre-initialized agents
            n: Hyperparameter for adaptive judge bar adjustment (default 0.0)
            falcon_api_key: API key for Falcon model (if using Falcon)
            pinecone_api_key: API key for Pinecone (if not using a pre-initialized retriever)
        """
        self.retriever = retriever
        self.n = n  # Hyperparameter for adaptive judge bar adjustment

        # Save these for potential reuse in Agent4
        self.agent_model = agent_model
        self.falcon_api_key = falcon_api_key

        # Initialize agents based on provided parameters
        if isinstance(agent_model, str):
            if "falcon" in agent_model.lower() and falcon_api_key:
                # Initialize Falcon agents
                from api_agent import FalconAgent

                self.agent1 = FalconAgent(falcon_api_key)  # Predictor
                self.agent2 = (
                    self.agent1
                )  # Judge (reuse the same instance to save resources)
                self.agent3 = self.agent1  # Final-Predictor
                self.agent4 = self.agent1  # Claim Judge
                logger.info(f"Using Falcon agents with API for all four agent roles")
            else:
                # Initialize local LLM agents
                from local_agent import LLMAgent

                self.agent1 = LLMAgent(agent_model)  # Predictor
                self.agent2 = self.agent1  # Judge
                self.agent3 = self.agent1  # Final-Predictor
                self.agent4 = self.agent1  # Claim Judge
                logger.info(f"Using local LLM agents with model {agent_model}")
        else:
            # Use pre-initialized agent
            self.agent1 = agent_model  # Predictor
            self.agent2 = self.agent1  # Judge
            self.agent3 = self.agent1  # Final-Predictor
            self.agent4 = self.agent1  # Claim Judge
            logger.info("Using pre-initialized agent for all four agent roles")

    def _create_agent1_prompt(self, query, document_text):
        """Create prompt for Agent-1 (Predictor)."""
        return f"""You are an accurate and reliable AI assistant that can answer questions with the help of external documents. You should only provide the correct answer without repeating the question and instruction.

Document: {document_text}

Question: {query}

Answer:"""

    def _create_agent2_prompt(self, query, document_text, answer):
        """Create prompt for Agent-2 (Judge)."""
        return f"""You are a noisy document evaluator that can judge if the external document is noisy for the query with unrelated or misleading information. Given a retrieved Document, a Question, and an Answer generated by an LLM (LLM Answer), you should judge whether both the following two conditions are reached: (1) the Document provides specific information for answering the Question; (2) the LLM Answer directly answers the question based on the retrieved Document. Please note that external documents may contain noisy or factually incorrect information. If the information in the document does not contain the answer, you should point it out with evidence. You should answer with "Yes" or "No" with evidence of your judgment, where "No" means one of the conditions (1) and (2) are unreached and indicates it is a noisy document.

Document: {document_text}

Question: {query}

LLM Answer: {answer}

Is this document relevant and supportive for answering the question?"""

    def _create_agent3_prompt(self, query, filtered_documents):
        """Create prompt for Agent-3 with improved citation instructions."""
        # Format documents with index
        docs_text = "\n\n".join(
            [
                f"Document {i+1}: {doc_text}"
                for i, (doc_text, _) in enumerate(filtered_documents)
            ]
        )

        return f"""You are an accurate and reliable AI assistant. Answer questions based ONLY on the provided documents. If the documents don't contain sufficient information, state that clearly.

IMPORTANT ANSWER GUIDELINES:
1. Begin with a direct, concise answer to the question.
2. PUT YOUR MOST IMPORTANT INFORMATION IN THE FIRST 300 WORDS - this part will be evaluated most heavily.
3. Organize your answer in well-structured paragraphs.
4. Every claim must directly contribute to answering the question.
5. Be comprehensive but prioritize relevance over length.

CITATION INSTRUCTIONS - CRITICAL:
1. EVERY claim MUST have a citation using ONLY this format: [X] where X is the document number.
2. Place citations immediately after each sentence or claim like this: "Authentic Inuit sculptures don't always need signatures. [1,3]"
3. NEVER use phrases like "Document 1 states" or "according to Document 2" or "(Document 3, Document 4)" or ""As Document 5 explains""
4. ONLY cite specific relevant documents - don't cite all documents for every claim.
5. If no documents contain information to answer the question, say so clearly.

EXAMPLES OF CORRECT CITATIONS:
✓ "Nematodes should be stored in the refrigerator until ready to use. [2]"
✓ "The soil should be moist but not saturated before and after application. [3,5,7]"

EXAMPLES OF INCORRECT CITATIONS:
✗ "According to Document 2, nematodes should be stored in the refrigerator."
✗ "Nematodes should be stored in the refrigerator (Document 2)."
✗ "As Document 3 explains, the soil should be moist."
✗ "The soil should be moist. [Document 3]"

Documents:
{docs_text}

Question: {query}

Answer (with citations using ONLY [X] format):"""

    def _create_agent4_prompt(self, query, answer_with_citations, filtered_documents):
        """
        Create prompt for Agent-4 (Claim Judge).
        This agent judges if the cited claims properly answer the question and identifies gaps.
        """
        # Format documents with index
        docs_text = "\n\n".join(
            [
                f"Document {i+1}: {doc_text}"
                for i, (doc_text, _) in enumerate(filtered_documents)
            ]
        )

        return f"""You are a meticulous judge evaluating if a question has been fully answered. Your task has three parts:

1. CLAIM ANALYSIS: Analyze the answer with citations and identify each claim made.
2. QUESTION COVERAGE: Determine if the original question has been completely answered.
3. GAP IDENTIFICATION: If parts of the question remain unanswered, identify these gaps precisely.

Original Question: {query}

Answer with Citations:
{answer_with_citations}

Available Documents:
{docs_text}

First, analyze each claim with its citation and evaluate if it correctly addresses part of the question.
Then, determine if any parts of the question remain unanswered.

Finally, provide your analysis in this format:
COMPLETELY_ANSWERED: Yes/No
UNANSWERED_ASPECTS: [List any aspects of the question that remain unanswered]
FOLLOW_UP_QUESTIONS: [If needed, rewrite a specific single question for each unanswered aspect(keep the context of original question)]

Your analysis:"""

    def _create_follow_up_prompt(
        self, original_query, follow_up_question, filtered_documents, previous_answer
    ):
        """Create prompt for generating an answer to a follow-up question."""
        # Format documents with index
        docs_text = "\n\n".join(
            [
                f"Document {i+1}: {doc_text}"
                for i, (doc_text, _) in enumerate(filtered_documents)
            ]
        )

        return f"""You are an accurate and reliable AI assistant. Your task is to answer a follow-up question based on the provided documents.

Original Question: {original_query}
Previous Answer: {previous_answer}

Follow-up Question: {follow_up_question}

Documents:
{docs_text}

Answer the follow-up question ONLY based on the provided documents. Use the format [X] to cite which document(s) support each claim, where X is the document number. If the documents don't contain the necessary information, state that clearly.

Follow-up Answer (with citations):"""

    def _extract_claims_with_citations(self, answer_with_citations):
        """Extract individual claims and their citations from the answer with improved robustness."""
        # Original pattern looking for [X] citations
        bracket_pattern = r"(.*?)\s*\[(\d+(?:,\s*\d+)*)\]"

        # Additional pattern for "Document X" style references
        doc_pattern = r"(.*?)(?:Document\s+(\d+(?:,\s*\d+)*))"

        claims = []

        # First try to extract standard [X] citations
        bracket_matches = list(re.finditer(bracket_pattern, answer_with_citations))

        # If no standard citations found, check for "Document X" style
        if not bracket_matches:
            # Convert "Document X" to [X] format for easier processing
            normalized_answer = re.sub(
                r"Document\s+(\d+)", lambda m: f"[{m.group(1)}]", answer_with_citations
            )

            # Try extraction again with normalized format
            bracket_matches = list(re.finditer(bracket_pattern, normalized_answer))

        # Process matches
        for match in bracket_matches:
            claim_text = match.group(1).strip()
            citation_str = match.group(2)
            citations = [int(c.strip()) for c in citation_str.split(",")]

            if claim_text:
                claims.append({"text": claim_text, "citations": citations})

        # Fallback: If still no claims extracted, treat entire answer as one claim
        if not claims and answer_with_citations.strip():
            # Extract all document numbers mentioned
            doc_nums = re.findall(r"Document\s+(\d+)", answer_with_citations)
            if doc_nums:
                citations = [int(num) for num in doc_nums]

                # Remove explicit document references for the claim text
                clean_text = re.sub(r"Document\s+\d+,?\s*", "", answer_with_citations)
                clean_text = re.sub(
                    r"such as Document \d+(, Document \d+)*", "", clean_text
                )

                claims.append(
                    {"text": clean_text.strip(), "citations": sorted(set(citations))}
                )

        return claims

    def _combine_answers(self, original_answer, follow_up_answers):
        """Combine the original answer with follow-up answers."""
        if not follow_up_answers:
            return original_answer

        combined = original_answer + "\n\nAdditional information:\n"
        combined += "\n\n".join(follow_up_answers)

        return combined

    def _remove_citations(self, answer_with_citations):
        """Remove citation brackets from the answer for final output."""
        # Remove [X] or [X,Y,Z] patterns
        return re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", answer_with_citations)

    def clean_answer(self, answer):
        """Simple cleaning function to handle empty or problematic answers."""
        if not answer or answer.strip() == "":
            return "no answer provided"

        # Remove common problematic patterns
        patterns_to_remove = [
            r"^>.*$",  # Blog formatting
            r"^Post your responses.*$",  # Instructions
            r"^Source\(s\):.*$",  # Sources
            r"^Question:.*$",  # Follow-up questions
        ]

        for pattern in patterns_to_remove:
            answer = re.sub(pattern, "", answer, flags=re.MULTILINE)

        # Split by newlines and take all non-empty lines
        lines = [line.strip() for line in answer.split("\n") if line.strip()]
        if not lines:
            return "no answer provided"

        return "\n".join(lines)  # Return all lines to preserve structure

    def format_supporting_passages(self, filtered_docs, claims=None):
        """
        Format supporting passages in decreasing order of importance.
        If claims are provided, sort passages by how many claims they support.
        Otherwise, sort by the original document scores.
        """
        if claims:
            # Count how many claims each document supports
            doc_citation_counts = {}
            for claim in claims:
                for doc_idx in claim["citations"]:
                    if doc_idx <= len(filtered_docs):
                        doc_id = filtered_docs[doc_idx - 1][1]  # Get doc ID
                        doc_citation_counts[doc_id] = (
                            doc_citation_counts.get(doc_id, 0) + 1
                        )

            # Sort doc IDs by citation count in decreasing order
            sorted_doc_ids = sorted(
                doc_citation_counts.keys(),
                key=lambda x: doc_citation_counts[x],
                reverse=True,
            )

            # Map IDs back to (text, id) tuples
            doc_id_map = {
                doc_id: (doc_text, doc_id) for doc_text, doc_id in filtered_docs
            }
            supporting_passages = [
                doc_id_map[doc_id] for doc_id in sorted_doc_ids if doc_id in doc_id_map
            ]

            # Add any remaining documents that weren't cited
            for doc in filtered_docs:
                if doc[1] not in doc_citation_counts and doc not in supporting_passages:
                    supporting_passages.append(doc)
        else:
            # Just use the original filtered docs order
            supporting_passages = filtered_docs

        return supporting_passages

    def answer_query(self, query, choices=None):
        """Process a query using the enhanced RAGent framework with improved claim analysis and follow-up handling."""
        logger.info(f"Processing query: {query}")

        # Step 1: Retrieve documents
        logger.info("Retrieving documents...")
        retrieved_docs = self.retriever.retrieve(
            query, top_k=20
        )  # Returns [(text, id), ...]
        logger.info(f"Retrieved {len(retrieved_docs)} documents")

        # Step 2: Agent-1 generates answers for each document
        logger.info("Agent-1 generating answers for each document...")
        doc_answers = []
        for doc_text, doc_id in tqdm(retrieved_docs):
            prompt = self._create_agent1_prompt(query, doc_text)
            answer = self.agent1.generate(prompt)
            doc_answers.append((doc_text, doc_id, answer))

        # Step 3: Agent-2 evaluates and scores each document
        logger.info("Agent-2 evaluating and scoring documents...")
        scores = []
        for doc_text, doc_id, answer in tqdm(doc_answers):
            prompt = self._create_agent2_prompt(query, doc_text, answer)
            log_probs = self.agent2.get_log_probs(prompt, ["Yes", "No"])
            score = log_probs["Yes"] - log_probs["No"]  # Key scoring mechanism
            scores.append(score)

        # Step 4: Calculate adaptive judge bar (τq)
        tau_q = np.mean(scores)
        sigma = np.std(scores)
        adjusted_tau_q = tau_q - self.n * sigma  # Use the n hyperparameter here
        logger.info(
            f"Adaptive judge bar: τq={tau_q:.4f}, adjusted: {adjusted_tau_q:.4f}"
        )

        # Step 5: Filter and rank documents
        filtered_docs = []
        excluded_ids = set()

        for i, (doc_text, doc_id, _) in enumerate(doc_answers):
            if scores[i] >= adjusted_tau_q:
                filtered_docs.append((doc_text, doc_id, scores[i]))
                excluded_ids.add(doc_id)

        # Sort by score in descending order (crucial for performance)
        filtered_docs.sort(key=lambda x: x[2], reverse=True)
        filtered_docs_no_score = [
            (doc_text, doc_id) for doc_text, doc_id, _ in filtered_docs
        ]
        logger.info(f"Filtered to {len(filtered_docs)} documents")

        # Step 6: Agent-3 generates answer with citations
        logger.info("Agent-3 generating answer with citations...")
        if filtered_docs:
            prompt = self._create_agent3_prompt(query, filtered_docs_no_score)
            raw_answer = self.agent3.generate(prompt)
            answer_with_citations = self.clean_answer(raw_answer)
        else:
            # Fall back to using all documents if none pass the filter
            logger.warn("No documents passed the filter, using all documents")
            all_docs_no_score = [
                (doc_text, doc_id) for doc_text, doc_id, _ in doc_answers
            ]
            prompt = self._create_agent3_prompt(query, all_docs_no_score)
            raw_answer = self.agent3.generate(prompt)
            answer_with_citations = self.clean_answer(raw_answer)
            filtered_docs_no_score = all_docs_no_score

        # Extract claims with citations for analysis
        claims = self._extract_claims_with_citations(answer_with_citations)
        logger.info(f"Extracted {len(claims)} claims with citations")

        # Log each claim with its citations
        for i, claim in enumerate(claims):
            logger.info(
                f"Claim {i+1}: {claim['text']} - cited docs: {claim['citations']}"
            )

        # Special handling for when no claims are extracted
        if not claims:
            logger.warn("No claims extracted, attempting to normalize citations")
            # Try to normalize citation format
            normalized_answer = re.sub(
                r"Document\s+(\d+)", lambda m: f"[{m.group(1)}]", answer_with_citations
            )
            claims = self._extract_claims_with_citations(normalized_answer)
            if claims:
                logger.info(f"After normalization: Extracted {len(claims)} claims")
                # Update answer_with_citations with normalized format
                answer_with_citations = normalized_answer

        # Step 7: Enhanced Agent-4 processes the answer
        logger.info("Enhanced Agent-4 processing answer...")

        # Initialize the enhanced Agent 4
        enhanced_agent4 = EnhancedAgent4(self.agent4, self.retriever)

        # Process the answer using the enhanced Agent 4
        final_answer_with_citations, judge_debug_info = enhanced_agent4.process_answer(
            query, answer_with_citations, claims, filtered_docs_no_score
        )

        final_answer_for_user = self._remove_citations(final_answer_with_citations)

        # Format supporting passages in order of importance
        supporting_passages = self.format_supporting_passages(
            filtered_docs_no_score, claims
        )

        # Log the final answer
        logger.info(f"Answer with citations: {answer_with_citations}")
        logger.info(
            f"Final answer (after follow-up processing): {final_answer_for_user}"
        )
        logger.info(f"Supporting passages: {supporting_passages}")

        # Return the answer and debug information
        debug_info = {
            "tau_q": tau_q,
            "adjusted_tau_q": adjusted_tau_q,
            "sigma": sigma,
            "scores": scores,
            "filtered_docs": [
                (doc_text, doc_id, score) for doc_text, doc_id, score in filtered_docs
            ],
            "supporting_passages": supporting_passages,
            "raw_answer": raw_answer,
            "answer_with_citations": answer_with_citations,
            "claims": claims,
            "agent3_prompt": prompt,
            # Add the enhanced Agent 4 debug info
            "claim_analysis": judge_debug_info["claim_analysis"],
            "question_structure": judge_debug_info["question_structure"],
            "question_components": judge_debug_info["question_components"],
            "unanswered_components": judge_debug_info["unanswered_components"],
            "coverage_assessment": judge_debug_info.get("coverage_assessment", {}),
            "follow_up_questions": judge_debug_info["follow_up_questions"],
            "completely_answered": judge_debug_info["completely_answered"],
            "follow_up_answers": judge_debug_info["follow_up_answers"],
        }

        return final_answer_for_user, debug_info
