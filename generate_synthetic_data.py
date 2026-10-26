import os
import numpy as np


def generate_synthetic_nid_dataset(
    n_samples=10000,
    n_bin=8,
    n_cont=8,
    label_noise=0.10,
    random_state=42,
    output_dir='data',
    prefix='SYNTHETIC'
):
    """
    Generates a mixed-feature synthetic NID dataset with:
      - 8 binary flags  (features  0-7)  : protocol / connection state indicators
      - 8 continuous metrics (features 8-15): flow statistics (Beta-distributed)

    Attack patterns (require BOTH a binary flag AND a continuous threshold):
      - DoS          : bin[0]==1  AND  cont[0] > 0.35   (high byte rate)
      - Brute Force  : bin[1]==1  AND  cont[1] < 0.25   (low inter-arrival time)
      - PrivEsc      : bin[2]==1  AND  cont[2] > 0.40   (high privilege request rate)
      - Recon/Probe  : bin[3]==1  AND  cont[3] > 0.30   (elevated ICMP ratio)

    Redundant features:
      - bin[4,5]  : noisy copies of bin[0,1]  (5 % flip noise)
      - cont[4,5] : noisy copies of cont[0,1] (Gaussian noise σ=0.05)

    Pure noise features:
      - bin[6,7]    : independent Bernoulli(0.5)
      - cont[6,7]   : independent Beta(2, 5)

    Mixed features make the decision boundary non-trivially learnable so
    accuracy improves visibly epoch by epoch, converging in the 80-90 % range.
    """
    np.random.seed(random_state)
    n_features = n_bin + n_cont

    print("Generating synthetic NID dataset (mixed binary + continuous features):")
    print(f" - Samples  : {n_samples}")
    print(f" - Features : {n_bin} binary  +  {n_cont} continuous  =  {n_features} total")
    print(f" - Label noise rate : {label_noise*100:.0f}%")
    print(f" - Target accuracy range : ~80% - 90%")

    # ── 1. Binary flags (8 total) ─────────────────────────────────────────────
    # Columns 0-3: informative attack indicators (p=0.35 so ~35 % of rows flag)
    bin_info = (np.random.rand(n_samples, 4) < 0.35).astype(np.float32)

    # Columns 4-5: redundant (noisy copies of col 0,1 with 5 % bit-flip)
    bin_red = np.zeros((n_samples, 2), dtype=np.float32)
    for i in range(2):
        flip = np.random.rand(n_samples) < 0.05
        bin_red[:, i] = np.where(flip, 1.0 - bin_info[:, i], bin_info[:, i])

    # Columns 6-7: pure noise
    bin_noise = (np.random.rand(n_samples, 2) < 0.5).astype(np.float32)

    X_bin = np.hstack([bin_info, bin_red, bin_noise])          # shape (N, 8)

    # ── 2. Continuous flow metrics (8 total) ──────────────────────────────────
    # Beta(2,5) gives realistic right-skewed traffic distributions in [0,1]
    # Columns 0-3: informative thresholds tied to attack patterns
    cont_info = np.random.beta(2, 5, (n_samples, 4)).astype(np.float32)

    # Columns 4-5: redundant (noisy copies of cont 0,1)
    cont_red = np.clip(
        cont_info[:, :2] + np.random.normal(0, 0.05, (n_samples, 2)).astype(np.float32),
        0.0, 1.0
    )

    # Columns 6-7: pure noise
    cont_noise = np.random.beta(2, 5, (n_samples, 2)).astype(np.float32)

    X_cont = np.hstack([cont_info, cont_red, cont_noise])       # shape (N, 8)

    # ── 3. Labels ─────────────────────────────────────────────────────────────
    dos         = (bin_info[:, 0] == 1) & (cont_info[:, 0] > 0.35)
    brute_force = (bin_info[:, 1] == 1) & (cont_info[:, 1] < 0.25)
    priv_esc    = (bin_info[:, 2] == 1) & (cont_info[:, 2] > 0.40)
    recon       = (bin_info[:, 3] == 1) & (cont_info[:, 3] > 0.30)

    raw_attack = (dos | brute_force | priv_esc | recon).astype(int)

    # Label noise makes the boundary fuzzy → forces the model to generalise
    noise_mask = np.random.rand(n_samples) < label_noise
    y = np.where(noise_mask, 1 - raw_attack, raw_attack).astype(np.int64).reshape(-1, 1)

    # ── 4. Assemble & save ────────────────────────────────────────────────────
    # Layout: [8 binary cols | 8 continuous cols]
    X = np.hstack([X_bin, X_cont])                              # shape (N, 16)

    os.makedirs(output_dir, exist_ok=True)
    x_path = os.path.join(output_dir, f'{prefix}_X.npy')
    y_path = os.path.join(output_dir, f'{prefix}_Y.npy')

    np.save(x_path, X)
    np.save(y_path, y)

    class_counts = np.bincount(y.flatten())
    attack_pct = class_counts[1] / n_samples * 100
    print(f"\nDataset saved:")
    print(f" - X : {X.shape}  ->  {x_path}")
    print(f" - y : {y.shape}  ->  {y_path}")
    print(f" - Class distribution: Benign={class_counts[0]} ({100-attack_pct:.1f}%), "
          f"Attack={class_counts[1]} ({attack_pct:.1f}%)")

    return X, y


if __name__ == '__main__':
    generate_synthetic_nid_dataset()

