# IDFlow
Energy-Based Flow Matching for Generating 3D Molecular Structure (ICML2025)
# Inference
The checkpoint for this project can be downloaded [here](https://drive.google.com/drive/folders/1MIiy-KPiKs8CjB8_qnRiYvLOYvUe7_F4).
The inference can be run with the command
```
python -W ignore experiments/inference_se3_flows.py -cn inference_unconditional inference.num_gpus=2 inference.ckpt_path=./ckpt/epoch\\=599-step\\=293400.ckpt 
```
# Acknowledgement
The majority of the code is from the [original work](https://github.com/microsoft/protein-frame-flow). We sincerely appreciate the authors' contribution. For additional details regarding the environment and setup, please refer to the original repository.
