import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

def moving_average(data, window_size):
	"""Calcule la moyenne mobile sur une fenêtre donnée"""
	if window_size > len(data):
		window_size = len(data)

	cumsum = np.cumsum(np.insert(data, 0, 0))
	return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def show_result(rewards, losses, path="agent_stats", window_size=20):
	os.makedirs("graphs", exist_ok=True)

	episodes = np.arange(len(rewards))

	if len(rewards) > window_size:
		ma_rewards = moving_average(rewards, window_size)
		ma_episodes = episodes[window_size-1:]
	else:
		ma_rewards = rewards
		ma_episodes = episodes

	if len(losses) > window_size:
		ma_losses = moving_average(losses, window_size)
	else:
		ma_losses = losses

	slope, intercept, _, _, _ = stats.linregress(ma_episodes, ma_rewards)
	trend_line = slope * ma_episodes + intercept

	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

	ax1.plot(episodes, rewards, 'b-', alpha=0.3, label='Récompenses brutes')
	ax1.plot(ma_episodes, ma_rewards, 'b-', linewidth=2, label=f'Moyenne mobile (fenêtre: {window_size})')
	ax1.plot(ma_episodes, trend_line, 'r--', label=f'Tendance (pente: {slope:.4f})')
	ax1.set_ylabel('Récompense', color='b')
	ax1.set_title('Évolution des récompenses avec tendance et moyenne mobile')
	ax1.grid(True)
	ax1.legend()

	if slope > 0:
		trend_text = f"Tendance positive: +{slope:.4f} par épisode"
		text_color = 'green'
	elif slope < 0:
		trend_text = f"Tendance négative: {slope:.4f} par épisode"
		text_color = 'red'
	else:
		trend_text = "Tendance stable"
		text_color = 'black'

	ax1.text(0.02, 0.95, trend_text, transform=ax1.transAxes,
			 fontsize=10, color=text_color, verticalalignment='top')

	if len(losses) > window_size:
		ax2.plot(episodes, losses, 'g-', alpha=0.3, label='Loss brute')
		ax2.plot(ma_episodes, ma_losses, 'g-', linewidth=2, label=f'Moyenne mobile (fenêtre: {window_size})')
	else:
		ax2.plot(episodes, losses, 'g-', label='Loss')

	ax2.set_xlabel('Épisodes')
	ax2.set_ylabel('Loss', color='g')
	ax2.set_title('Évolution de la fonction de perte (loss)')
	ax2.grid(True)
	ax2.legend()

	stats_text = (
		f"Récompense moyenne: {np.mean(rewards):.4f}\n"
		f"Récompense finale (moy. 10 derniers): {np.mean(rewards[-10:]):.4f}\n"
		f"Loss moyenne: {np.mean(losses):.4f}\n"
	)
	ax2.text(0.02, 0.15, stats_text, transform=ax2.transAxes,
			 fontsize=9, bbox=dict(facecolor='white', alpha=0.5))

	plt.tight_layout()

	file_path = f"graphs/{path}.png"
	plt.savefig(file_path)
	plt.close()

	print(f"Graphique sauvegardé dans {file_path}")
	print(f"Statistiques finales:")
	print(f"  - Récompense moyenne: {np.mean(rewards):.4f}")
	print(f"  - Récompense moyenne (10 derniers épisodes): {np.mean(rewards[-10:]):.4f}")
	print(f"  - Loss moyenne: {np.mean(losses):.4f}")

	return file_path
