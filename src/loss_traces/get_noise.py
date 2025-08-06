from opacus.accountants.utils import get_noise_multiplier


if __name__ == "__main__":
    sample_rate = float(25000/256)
    noise_multiplier = get_noise_multiplier(
        target_epsilon=50.0,
        target_delta=1e-5,
        sample_rate=sample_rate,
        epochs=100,
        accountant='rdp',  # or 'gdp'
    )
    print(f"Noise Multiplier: {noise_multiplier}")