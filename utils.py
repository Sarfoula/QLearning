import matplotlib.pyplot as plt
import numpy as np

def show_result(scores, hit, error, epsilon, save_path):
	avg_rewards = np.convolve(scores, np.ones(100)/100, mode='valid')
	avg_hit = np.convolve(hit, np.ones(100)/100, mode='valid')
	avg_error = np.convolve(error, np.ones(100)/100, mode='valid')

	x = np.arange(len(scores))
	coeffs = np.polyfit(x, scores, 2)
	poly = np.poly1d(coeffs)
	y_tendance = poly(x)

	plt.figure(figsize=(10, 6))
	plt.plot(avg_rewards, label=f"mean reward", linestyle='-', color='blue', linewidth=2)
	plt.plot(avg_hit, label=f"mean mean hit", linestyle='-', color='black', linewidth=2)
	plt.plot(avg_error, label=f"mean error", linestyle='-', color='green', linewidth=2)
	plt.plot(x, y_tendance, color='red', linestyle='--', linewidth=1, label='Tendance')

	if epsilon is not None:
		plt.axvline(x=epsilon, color='orange', linestyle='--', linewidth=1)

	plt.xlabel("Épisodes")
	plt.ylabel("Valeurs")
	plt.title("Évolution des performances de l'agent au fil des épisodes")
	plt.legend()

	plt.grid(True)
	plt.savefig(save_path, dpi='figure')
