import matplotlib.pyplot as plt
import numpy as np

def show_result(scores):
	avg_rewards = [np.mean(scores[max(0, i-100):(i+1)]) for i in range(len(scores))]
	# plt.plot(scores, label="Récompense brute")
	plt.plot(avg_rewards, label=f"Moyenne ({100} épisodes)", linestyle='dashed', linewidth=2)
	plt.xlabel("Épisodes")
	plt.ylabel("Récompense")
	plt.legend()
	plt.show()
