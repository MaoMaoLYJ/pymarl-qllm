# QLLM: Do We Really Need a Mixing Network for Credit Assignment in Multi-Agent Reinforcement Learning?

This repository contains the implementation of **QLLM**, a novel value decomposition framework that leverages Large Language Models (LLMs) to construct training-free credit assignment functions (**TFCAF**).

---

## 1. Installing Dependencies

To install the core dependencies for the codebase, clone this repo and run:
```sh
pip install -r requirements.txt
```

To install the supported environments, follow the instructions below based on the environment you wish to test:

### Level-Based Foraging (LBF)
```sh
cd gymma/lb-foraging-master
pip install -e .
```

### Multi-agent Particle Environment (MPE)
```sh
pip install pettingzoo
```

### Google Research Football (GRF)
```sh
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc
pip install --upgrade psutil wheel pytest
pip install gfootball==2.10.2 gym==0.11
```

### StarCraft Multi-Agent Challenge (SMAC)
Please follow the [official SMAC installation guide](https://github.com/oxwhirl/smac) to install StarCraft II and the SMAC maps.

---

## 2. Supported Environments List

- **Level Based Foraging (LBF)**: Run in `gymma` folder.
- **PettingZoo (MPE)**: Run in `gymma` folder.
- **StarCraft Multi-Agent Challenge (SMAC)**: Run in `SMAC` file.
- **Google Research Football (GRF)**: Run in `G-football` file.

---

## 3. Run Instructions

To run the QLLM algorithm on different environments, refer to the examples and specific parameters below.

### Level-Based Foraging (LBF)
**Run in `gymma` folder**
- **Example Command:**
  ```sh
  python src/qllm_main.py --config=qllm --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-2s-8x8-2p-2f-coop-v3"
  ```
- **Specific Maps (All with `--env-config=gymma with env_args.time_limit=50`):**
  - 8x8-2p-2f-coop: `env_args.key="lbforaging:Foraging-2s-8x8-2p-2f-coop-v3"`
  - 10x10-3p-3f-2s: `env_args.key="lbforaging:Foraging-2s-10x10-3p-3f-v3"`
  - 15x15-4p-3f-2s: `env_args.key="lbforaging:Foraging-15x15-4p-3f-v3"`

### Multi-agent Particle Environment (MPE)
**Run in `gymma` folder**
- **Example Command:**
  ```sh
  python src/qllm_main.py --config=qllm --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3"
  ```
- **Specific Maps (All with `--env-config=gymma with env_args.time_limit=25`):**
  - Simple Spread: `env_args.key="pz-mpe-simple-spread-v3"`
  - Simple Adversary: `env_args.key="pz-mpe-simple-adversary-v3" env_args.pretrained_wrapper="PretrainedAdversary"`
  - Simple Tag: `env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag"`

### StarCraft Multi-Agent Challenge (SMAC)
**Run in `SMAC` file**
- **Example Command:**
  ```sh
  python src/qllm_main.py --config=qllm --env-config=sc2 with env_args.map_name="3s_vs_5z"
  ```
- **Specific Maps (All with `--env-config=sc2`):**
  - 3s_vs_5z: `env_args.map_name="3s_vs_5z"`
  - 2c_vs_64zg: `env_args.map_name="2c_vs_64zg"`

### Google Research Football (GRF)
**Run in `G-football` file**
- **Example Command:**
  ```sh
  python src/qllm_main.py --config=qllm --env-config=gfootball with env_args.time_limit=150 env_args.map_name="academy_3_vs_1_with_keeper"
  ```
- **Specific Maps (All with `--env-config=gfootball with env_args.time_limit=150`):**
  - Academy 3 vs 1: `env_args.map_name="academy_3_vs_1_with_keeper"`
  - Academy Counterattack Easy: `env_args.map_name="academy_counterattack_easy"`
  - Academy Pass and Shoot: `env_args.map_name="academy_pass_and_shoot_with_keeper"`

---

## 4. Running QLLM Algorithm (Detailed Logic)

QLLM replaces traditional mixing networks with an LLM-generated Python function (**TFCAF**). Running the algorithm involves two primary steps depending on the availability of the TFCAF file.

### Step 1: Generate TFCAF (Pretrain Phase)
If you have **not** yet generated a TFCAF file (e.g., `TFCAF_<map_name>.txt`) for a specific map:
1. **Ensure the task prompt file `src/setup_prompt_<map_name>.txt` exists.** This file is required for the LLM to understand the environment specifications and credit assignment logic.
2. Open `src/config/algs/qllm.yaml`.
3. Set **`LLM_pretrain: True`**.
4. Run the environment-specific command (as listed in Section 3). 
5. The system will consult the LLM to synthesize the function and save it as `src/TFCAF_<map_name>.txt`.

### Step 2: Train using TFCAF (Training Phase)
If you have already generated the `TFCAF_<map_name>.txt` file and wish to perform Reinforcement Learning training:
1. Open `src/config/algs/qllm.yaml`.
2. Set **`LLM_pretrain: False`**.
3. Run the same environment command again.
4. The algorithm will directly read the existing TFCAF from the `src` folder and start training without calling the LLM again.

---

## 5. API Configuration

**Important:** Before running Step 1, you must configure your API keys in the code.

1. Open `src/LLM_helper.py`.
2. Fill in the `api_key`, `base_url`, and `model_name` at the corresponding locations for either **DeepSeek** or **ChatGPT**.

- **DeepSeek API:** [https://platform.deepseek.com/](https://platform.deepseek.com/)
- **OpenAI API:** [https://platform.openai.com/](https://platform.openai.com/)

---
## 6. QLLM Config Parameters (`qllm.yaml`)
- `LLM_pretrain`: `True` to generate new TFCAF, `False` to use existing ones.
- `LLM_episode`: Number of iteration rounds for LLM refinement.
- `maker_num`: Number of candidate functions generated per round.
- `message_length`: Maximum context history for the LLM.
