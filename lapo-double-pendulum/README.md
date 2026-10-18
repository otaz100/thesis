# LAPO on the inverted double pendulum

A from-scratch rebuild of LAPO ([Schmidt & Jiang, ICLR 2024](https://arxiv.org/abs/2312.10812)) on Gymnasium's
`InvertedDoublePendulum-v5`, with the torque discretized into 15 levels. The agent learns latent actions from state-only
expert demonstrations, imitates them, translates them into torques, and is then fine-tuned online.

Notebook: [`Double_Pendulum_With_LAPO.ipynb`](Double_Pendulum_With_LAPO.ipynb), saved with the outputs of the run below.

## Sections

| Section | Contents |
|---|---|
| I | Environment and evaluation harness |
| II | PPO expert, demonstrations, behavior cloning baseline |
| III | Dynamics data from the expert sampling its own policy |
| IV | Stage 1: inverse and forward dynamics models with a vector-quantized latent, and a UMAP of the latent actions |
| V | Stage 2: latent behavior cloning, without actions |
| VI | Stage 3: decoding the latents into torques from small labelled sets |
| VI B | Decoding from the agent's own play, without expert labels |
| VII | Online fine-tuning with PPO on equal step budgets |
| VIII | The same pipeline on CartPole |

## Results from the saved run

- **Stage 1:** the quantized latent identifies the expert's torque 88.5% of the time when stage 1 trains on the sampled
  expert, against 76.1% on the deterministic demos (31.2% by always guessing the most common torque).
- **Offline decoding:** with 128 labels, LAPO decodes 93.2% of torques against 74.5% for BC trained on the same labels.
  In the environment, BC overtakes from 2,048 labels.
- **Without expert labels:** decoding from 3,072 steps of the agent's own play reached 56.9% accuracy offline (no
  successful episodes), but after online fine-tuning it finished level with LAPO from 512 labels.
- **Online, 300,000 steps:** final mean return 6,150 for LAPO from 512 labels and 7,204 for LAPO without expert labels
  (both medians about 9,355, a full 1,000-step episode), against 2,962 for PPO from scratch and 917 for BC from 512 labels.
- **CartPole:** with 4 labels, LAPO matches the expert (98% success) against 18% for BC.

## Running it

Python 3.11 with the packages in `requirements.txt`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Open the notebook from this folder and run all cells. The two experts in `models/` are loaded instead of retrained, so the
demonstrations match this run. The datasets and the stage 1 and stage 2 models are rebuilt into `data_discrete/` on the
first run.
