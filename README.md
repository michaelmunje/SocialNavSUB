# SocialNav-SUB (Social Navigation Scene Understanding Benchmark)

This repository contains the code and resources for benchmarking VLMs for scene understanding of challenging social navigation scenarios. For more information, please see [the paper](link) (will add link later).

## Getting Started

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt (will add actual file soon)
   ```

2. **Download the Dataset**

    Please download our dataset from [HuggingFace](link) (will add real link later). This dataset contains the prompt data and human labels for the same social navigation scenarios that the VLMs are evaluated in, which allows quantitative evaluation against human data. Please provide place the human data in the `human_dataset` folder or make a custom folder and replace the `dataset_folder` entry in the config file, and add the prompt data in to `prompts/`.


3. **Benchmark a VLM**

    Make a config file and specify the VLM under the `baseline_model` parameter and parameters for the experiments (such as prompt representation).

   ```bash
   python socialnavsub/evaluate_vlm.py --cfg_path <cfg_path>
   ```

4. **View Results**

   Results will be saved in the directory specified in the config file under the `evaluation_folder` entry. To postprocess the results, please run:

   ```bash
   python socialnavsub/aggregate_eval_data.py --cfg_path <cfg_path>
   ```

   The results will be viewable in the csv whose filepath is specified in the `full_results_csv` entry in the config file.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

### How to add support for additional VLMs
You can make a new class file that contains a subclass of `APIBaseline` (`api_baseline.py`). For examples, please see `gemini.py`, `llava.py`, or `gpt4o.py`.

## Contact

For questions or support, please open an issue or email [michaelmunje@utexas.edu](mailto:michaelmunje@utexas.edu).
