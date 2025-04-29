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

    def evaluate_answer_with_citations(self, query, answer_with_citations):
        """
        Evaluates an answer with citations for faithfulness to the source documents.

        Args:
            query: The original query
            answer_with_citations: Dictionary with 'text' (the answer) and 'citations'
                                  (list of tuples with citation text and document index)

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

        # Evaluate each citation
        for citation_text, doc_index, doc_content in answer_with_citations["citations"]:
            qc_evaluation = self._evaluate_question_citation(query, doc_content)
            cp_evaluation = self._evaluate_citation_prediction(
                citation_text, doc_content
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

            # Categorize the citation based on evaluations
            if qc_evaluation["score"] > 0.7 and cp_evaluation["score"] > 0.7:
                results["valid_citations"].append((citation_text, doc_index))
            elif qc_evaluation["score"] > 0.7 and cp_evaluation["score"] <= 0.7:
                # Document is relevant but answer isn't faithful to it
                results["hallucinations"].append(
                    (citation_text, doc_index, cp_evaluation["reason"])
                )
            else:
                # Document isn't relevant enough
                results["invalid_citations"].append(
                    (citation_text, doc_index, qc_evaluation["reason"])
                )

        # Determine the overall action to take
        if len(results["valid_citations"]) > 0:
            if len(results["hallucinations"]) > len(results["valid_citations"]):
                results["action"] = "regenerate"
                results["feedback"] = (
                    "Significant hallucinations detected. Regenerate answer using only valid citations."
                )
            else:
                results["action"] = "keep"
                results["feedback"] = (
                    "Answer is mostly faithful to the documents. Keep with minor adjustments."
                )
        else:
            results["action"] = "use_next_docs"
            results["feedback"] = (
                "No valid citations found. Try using the next set of documents."
            )

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
- 0.5 means the document partially addresses the question
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
            # Fallback if parsing fails
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
        prompt = f"""You are an expert evaluator tasked with determining if a statement is entailed by a given document.
Evaluate if the following statement is supported by information in the document.

Statement: {citation_text}

Document: {doc_content}

On a scale of 0 to 1, where:
- 0 means the statement contradicts the document
- 0.3 means the statement contains claims not found in the document (hallucination)
- 0.5 means the statement is neither contradicted nor supported by the document
- 0.7 means the statement is partially supported by the document with some extrapolation
- 1 means the statement is fully supported by the document

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
            # Fallback if parsing fails
            return {
                "score": 0.5,
                "reason": "Failed to parse evaluation. Original response: " + response,
            }
