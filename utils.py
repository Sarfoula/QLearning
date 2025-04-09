import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

def moving_average(data, window_size):
	if window_size > len(data):
		window_size = len(data)

	cumsum = np.cumsum(np.insert(data, 0, 0))
	return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def show_loss(losses, window_size=100, path="predictor_stats"):
	os.makedirs("graphs", exist_ok=True)

	epochs = np.arange(len(losses))
	ma_losses = moving_average(losses, window_size) if len(losses) > window_size else losses
	ma_epochs = epochs[window_size-1:] if len(losses) > window_size else epochs

	slope, intercept, _, _, _ = stats.linregress(ma_epochs, ma_losses)
	trend_line = slope * ma_epochs + intercept

	plt.figure(figsize=(10, 5))
	plt.plot(epochs, losses, 'b-', alpha=0.3, label='Loss brute')
	plt.plot(ma_epochs, ma_losses, 'b-', linewidth=2, label=f'Moyenne mobile ({window_size})')
	plt.plot(ma_epochs, trend_line, 'r--', label=f'Tendance ({slope:.6f})')

	plt.xlabel('Époques'), plt.ylabel('Loss (MSE)')
	plt.title('Évolution de la perte'), plt.grid(True), plt.legend()

	avg_loss = np.mean(losses)
	final_loss = np.mean(losses[-100:])
	stats_text = f"Loss moy: {avg_loss:.6f}\nLoss finale: {final_loss:.6f}\nTendance: {slope:.6f}"
	plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=9,
			 bbox=dict(facecolor='white', alpha=0.5))

	file_path = f"graphs/{path}.png"
	plt.savefig(file_path)
	plt.close()

	print(f"Graphique sauvegardé: {file_path}")
	print(f"Stats: Loss moy={avg_loss:.6f}, Loss finale={final_loss:.6f}, Tendance={slope:.6f}")

	return file_path

def show_result(rewards, losses, path="agent_stats", window_size=20):
	os.makedirs("graphs", exist_ok=True)

	episodes = np.arange(len(rewards))
	ma_rewards = moving_average(rewards, window_size) if len(rewards) > window_size else rewards
	ma_episodes = episodes[window_size-1:] if len(rewards) > window_size else episodes
	ma_losses = moving_average(losses, window_size) if len(losses) > window_size else losses

	slope, intercept, _, _, _ = stats.linregress(ma_episodes, ma_rewards)
	trend_line = slope * ma_episodes + intercept

	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

	ax1.plot(episodes, rewards, 'b-', alpha=0.3, label='Récompenses brutes')
	ax1.plot(ma_episodes, ma_rewards, 'b-', linewidth=2, label=f'Moyenne mobile ({window_size})')
	ax1.plot(ma_episodes, trend_line, 'r--', label=f'Tendance ({slope:.4f})')
	ax1.set_ylabel('Récompense'), ax1.set_title('Évolution des récompenses')
	ax1.grid(True), ax1.legend()

	trend_text = f"Tendance: {slope:.4f} par épisode"
	ax1.text(0.02, 0.95, trend_text, transform=ax1.transAxes, fontsize=10)

	ax2.plot(episodes, losses, 'g-', alpha=0.3, label='Loss brute')
	if len(losses) > window_size:
		ax2.plot(ma_episodes, ma_losses, 'g-', linewidth=2, label=f'Moyenne mobile ({window_size})')

	ax2.set_xlabel('Épisodes'), ax2.set_ylabel('Loss')
	ax2.set_title('Évolution de la loss'), ax2.grid(True), ax2.legend()

	avg_reward = np.mean(rewards)
	final_reward = np.mean(rewards[-10:])
	avg_loss = np.mean(losses)
	stats_text = f"Récomp. moy: {avg_reward:.4f}\nRécomp. finale: {final_reward:.4f}\nLoss moy: {avg_loss:.4f}"
	ax2.text(0.02, 0.15, stats_text, transform=ax2.transAxes, fontsize=9,
			 bbox=dict(facecolor='white', alpha=0.5))

	plt.tight_layout()

	file_path = f"graphs/{path}.png"
	plt.savefig(file_path)
	plt.close()

	print(f"Graphique sauvegardé: {file_path}")
	print(f"Stats: Récomp. moy={avg_reward:.4f}, Récomp. finale={final_reward:.4f}, Loss moy={avg_loss:.4f}")

	return file_path

def plot_performance(episode_rewards, agent_loss, predictor_loss):
	os.makedirs("graphs", exist_ok=True)

	plt.figure(figsize=(10, 5))
	plt.plot(episode_rewards)
	plt.xlabel('Épisode'), plt.ylabel('Récompense')
	plt.title('Récompenses par épisode'), plt.grid(True)
	plt.savefig("graphs/rewards.png")
	plt.close()

	show_loss(predictor_loss, window_size=20, path="predictor_loss")
	show_result(episode_rewards, agent_loss, path="training_summary")
