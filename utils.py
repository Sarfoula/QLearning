import matplotlib.pyplot as plt
import numpy as np

def show_result(scores, num_games):
	smooth = np.convolve(scores, np.ones(10) / 10, mode='valid')
	plt.xlabel("Épisode")
	plt.ylabel("Récompenses")
	plt.plot(range(1, num_games + 1), scores, label="Scores", alpha=0.3, color="blue")
	plt.plot(range(10, num_games + 1), smooth, label="Average", color="red")
	plt.axhline(0, color="black", linestyle="dotted")
	plt.legend()
	plt.show()
