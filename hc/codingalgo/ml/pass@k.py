import torch
from collections import Counter
import math

def pass_at_1(responses_correct: torch.Tensor) -> float:
	"""
	Compute pass@1 using PyTorch.

	Args:
		responses_correct: Boolean tensor where True indicates a correct response.

	Returns:
		pass@1 score
	"""
	# pass@1 is simply the fraction of problems with at least one correct response
	return responses_correct.float().mean().item()


def majority_voting(responses: list[str]) -> str:
	"""
	Return most common response.
	"""
	# Use Counter to find the most common string
	data = Counter(responses)
	return data.most_common(1)[0][0]


def pass_at_k(n: int, c: int, k: int) -> float:
	"""
	Compute unbiased pass@k estimator.

	Args:
		n: Total number of generated responses for a problem.
		c: Number of correct responses among the 'n' generated responses.
		k: Number of responses to consider (subset size).

	Returns:
		The unbiased pass@k score.
	"""
	# If there are no correct responses, pass@k is 0.0
	if c == 0:
		return 0.0
	# If all responses are correct, or if it's impossible to pick k incorrect ones,
	# pass@k must be 1.0
	if n - c < k: 
		return 1.0

	# The unbiased estimator is 1 - P(no correct responses in k samples)
	# which is 1 - ( (n - c choose k) / (n choose k) )
	# This can be calculated as 1.0 - product( (n - c - i) / (n - i) for i in range(k) )
	score = 1.0
	for i in range(k):
		score *= (n - c - i) / (n - i)

	return 1.0 - score