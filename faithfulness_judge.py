"""
Faithfulness Judge Agent to evaluate the integrity of generated answers.
This agent checks if answers are faithful to the source documents and properly attributed.
"""


class FaithfulnessJudge:
    def __init__(self, agent_model):
        """
        Initialize the Faithfulness Judge.

        Args:
            agent_model: The LLM agent to use for evaluation
        """
        self.agent = agent_model

    def evaluate_answer_with_citations(
        self, query, answer_with_citations, verbose_logging=False
    ):
        """
        Evaluates an answer with citations for faithfulness to the source documents.

        Args:
            query: The original query
            answer_with_citations: Dictionary with 'text' (the answer) and 'citations'
                                  (list of tuples with citation text and document index)
            verbose_logging: Whether to print detailed evaluation logs

        Returns:
            Dictionary with evaluation results and suggested actions
        """
        results = {
            "valid_citations": [],
            "invalid_citations": [],
            "hallucinations": [],
            "action": "keep",  # Options: keep, regenerate, use_next_docs
            "feedback": "",
        }

        citation_evaluations = []

        # Make sure we have citations to evaluate
        if (
            not answer_with_citations
            or "citations" not in answer_with_citations
            or not answer_with_citations["citations"]
        ):
            results["action"] = "use_next_docs"
            results["feedback"] = "No citations found. Try using different documents."
            return results, citation_evaluations

        # Store total citation count
        total_citations = len(answer_with_citations["citations"])

        if verbose_logging:
            print("\nEVALUATING CITATIONS:")
            print(f"Found {total_citations} citations to evaluate")
            print("-" * 50)

        # Evaluate each citation
        for i, citation_tuple in enumerate(answer_with_citations["citations"]):
            # Make sure citation tuple has the expected structure
            if len(citation_tuple) < 3:
                if verbose_logging:
                    print(f"Citation {i+1}: Invalid structure, skipping")
                continue

            citation_text, doc_index, doc_content = citation_tuple

            # Skip very short or empty citations
            if not citation_text or len(citation_text.split()) < 3:
                if verbose_logging:
                    print(f"Citation {i+1}: Too short, skipping")
                continue

            # Create smaller prompts to stay within token limits
            if len(doc_content) > 1500:
                doc_content = doc_content[:1500] + "..."

            if verbose_logging:
                print(f"\nCITATION {i+1}:")
                print(
                    f"Text: {citation_text[:100]}..."
                    if len(citation_text) > 100
                    else f"Text: {citation_text}"
                )
                print(
                    f"Document {doc_index+1}: {doc_content[:100]}..."
                    if len(doc_content) > 100
                    else f"Document {doc_index+1}: {doc_content}"
                )

            try:
                # QC evaluation - Does document address question?
                qc_evaluation = self._evaluate_question_citation(query, doc_content)

                # CP evaluation - Is citation supported by document?
                cp_evaluation = self._evaluate_citation_prediction(
                    citation_text, doc_content
                )

                if verbose_logging:
                    print(
                        f"QC Score: {qc_evaluation['score']:.2f} - {qc_evaluation['reason'][:100]}..."
                        if len(qc_evaluation["reason"]) > 100
                        else f"QC Score: {qc_evaluation['score']:.2f} - {qc_evaluation['reason']}"
                    )
                    print(
                        f"CP Score: {cp_evaluation['score']:.2f} - {cp_evaluation['reason'][:100]}..."
                        if len(cp_evaluation["reason"]) > 100
                        else f"CP Score: {cp_evaluation['score']:.2f} - {cp_evaluation['reason']}"
                    )

                citation_evaluations.append(
                    {
                        "citation_text": citation_text,
                        "doc_index": doc_index,
                        "qc_score": qc_evaluation["score"],
                        "qc_reason": qc_evaluation["reason"],
                        "cp_score": cp_evaluation["score"],
                        "cp_reason": cp_evaluation["reason"],
                    }
                )

                # Use more lenient thresholds (0.4 instead of 0.6/0.7)
                if qc_evaluation["score"] > 0.4 and cp_evaluation["score"] > 0.4:
                    results["valid_citations"].append((citation_text, doc_index))
                    if verbose_logging:
                        print("VERDICT: VALID ✓")
                elif qc_evaluation["score"] > 0.4 and cp_evaluation["score"] <= 0.4:
                    # Document is relevant but answer isn't faithful to it
                    results["hallucinations"].append(
                        (citation_text, doc_index, cp_evaluation["reason"])
                    )
                    if verbose_logging:
                        print("VERDICT: HALLUCINATION ⚠️")
                else:
                    # Document isn't relevant enough
                    results["invalid_citations"].append(
                        (citation_text, doc_index, qc_evaluation["reason"])
                    )
                    if verbose_logging:
                        print("VERDICT: INVALID ✗")
            except Exception as e:
                if verbose_logging:
                    print(f"Error evaluating citation: {str(e)}")
                # If evaluation fails, consider it a low-confidence valid citation
                # (more lenient error handling)
                results["valid_citations"].append((citation_text, doc_index))

        # If we have no evaluations at all, default to keeping the answer
        if not citation_evaluations:
            results["action"] = "keep"
            results["feedback"] = (
                "Unable to properly evaluate citations. Keeping original answer."
            )
            return results, citation_evaluations

        # Determine the overall action to take with more balanced logic
        if verbose_logging:
            print("\nDECISION PROCESS:")
            print(f"Valid citations: {len(results['valid_citations'])}")
            print(f"Invalid citations: {len(results['invalid_citations'])}")
            print(f"Hallucinations: {len(results['hallucinations'])}")

        # Keep the answer if we have ANY valid citations
        if len(results["valid_citations"]) > 0:
            # Only regenerate if hallucinations significantly outnumber valid citations
            if len(results["hallucinations"]) > 1.5 * len(results["valid_citations"]):
                results["action"] = "regenerate"
                results["feedback"] = (
                    "Some valid content, but too many hallucinations. Regenerating."
                )
                if verbose_logging:
                    print(
                        "ACTION: REGENERATE - Some valid content, but too many hallucinations"
                    )
            else:
                results["action"] = "keep"
                results["feedback"] = (
                    "Answer has valid citations. Minor issues can be addressed with notes."
                )
                if verbose_logging:
                    print("ACTION: KEEP - Answer has sufficient valid citations")
        else:
            # Only use next docs if we have NO valid citations
            results["action"] = "use_next_docs"
            results["feedback"] = (
                "No valid citations found. Trying alternative documents."
            )
            if verbose_logging:
                print("ACTION: USE_NEXT_DOCS - No valid citations found")

        return results, citation_evaluations

    def _evaluate_question_citation(self, query, doc_content):
        """
        Evaluates if a document addresses the query (Question-Citation relevance).

        Args:
            query: The original query
            doc_content: The content of the document

        Returns:
            Evaluation with score and reason
        """
        prompt = f"""You are an expert document evaluator tasked with determining if a document contains information 
that addresses a given question. Evaluate how relevant this document is to answering the question.

Question: {query}

Document: {doc_content}

First, identify the key information needs in the question.
Then, determine if the document contains information that addresses these needs.

On a scale of 0 to 1, where:
- 0 means the document is completely irrelevant to the question
- 0.3 means the document has very minimal relevance to the question
- 0.5 means the document partially addresses the question
- 0.7 means the document is quite relevant to the question
- 1 means the document fully addresses the question

Provide your score and reasoning in the following format:
Score: [your score between 0 and 1]
Reasoning: [your reasoning]
"""
        response = self.agent.generate(prompt)

        # Parse the response to extract score and reasoning
        try:
            score_line = [
                line for line in response.split("\n") if line.startswith("Score:")
            ][0]
            score = float(score_line.split(":")[1].strip())

            reasoning_lines = [
                line for line in response.split("\n") if line.startswith("Reasoning:")
            ]
            reasoning = (
                reasoning_lines[0].split(":")[1].strip()
                if reasoning_lines
                else "No reasoning provided"
            )

            return {"score": score, "reason": reasoning}
        except:
            # Fallback if parsing fails - more lenient default (0.5 instead of 0)
            return {
                "score": 0.5,
                "reason": "Failed to parse evaluation. Original response: " + response,
            }

    def _evaluate_citation_prediction(self, citation_text, doc_content):
        """
        Evaluates if an answer is entailed by the cited document (Citation-Prediction faithfulness).

        Args:
            citation_text: Text from the answer that is attributed to this citation
            doc_content: The content of the document

        Returns:
            Evaluation with score and reason
        """
        prompt = f"""You are an expert evaluator tasked with determining if a statement is supported by a given document.
Evaluate if the following statement is supported by information in the document.

Statement: {citation_text}

Document: {doc_content}

On a scale of 0 to 1, where:
- 0 means the statement directly contradicts the document
- 0.3 means the statement contains claims not found in the document (hallucination)
- 0.5 means the statement is a reasonable inference from the document though not explicitly stated
- 0.7 means the statement is mostly supported by the document with minor extrapolation
- 1 means the statement is fully supported by the document

Be generous in your evaluation - if the statement could reasonably be inferred from the document even if not explicitly stated, give it a higher score.

Provide your score and reasoning in the following format:
Score: [your score between 0 and 1]
Reasoning: [your reasoning]
"""
        response = self.agent.generate(prompt)

        # Parse the response to extract score and reasoning
        try:
            score_line = [
                line for line in response.split("\n") if line.startswith("Score:")
            ][0]
            score = float(score_line.split(":")[1].strip())

            reasoning_lines = [
                line for line in response.split("\n") if line.startswith("Reasoning:")
            ]
            reasoning = (
                reasoning_lines[0].split(":")[1].strip()
                if reasoning_lines
                else "No reasoning provided"
            )

            return {"score": score, "reason": reasoning}
        except:
            # Fallback if parsing fails - more lenient default (0.5 instead of 0)
            return {
                "score": 0.5,
                "reason": "Failed to parse evaluation. Original response: " + response,
            }
