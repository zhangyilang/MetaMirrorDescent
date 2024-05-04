# [ICASSP 2024] MetaMirrorDescent
Implementation of ICASSP 2024 paper “[Meta-Learning with Versatile Loss Geometries for Fast Adaptation Using Mirror Descent](https://arxiv.org/abs/2312.13486)”. 

Codes tested under the following environment:

---

- PyTorch 1.9.1
- CuDNN 7.6.5
- CUDAToolkit 11.2
- Torchvision 0.10.1
- Torch-utils 0.1.2
- [Torchmeta](https://github.com/tristandeleu/pytorch-meta) 1.8.0
- Pillow 9.2.0

---

To build up the environment with `Anaconda`, please activate the target environment, and then run

```shell
sh env_setup.sh
```

To prepare the datasets, follow the instructions of [Torchmeta](https://github.com/tristandeleu/pytorch-meta). 

Default experimental setups can be found in `main.py`. To carry out the numerical test, use the commands

```shell
python main.py "--arguments" "values"
```

where `arguments` and `values` are the algorithm parameters that you want to alter. 



# Citation

> Y. Zhang, B. Li, and G. B. Giannakis, "Meta-Learning with Versatile Loss Geometries for Fast Adaptation Using Mirror Descent," *ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Seoul, Korea, April 14-19, 2024, pp. 5220-5224, doi: 10.1109/ICASSP48485.2024.10448144. 

```tex
@inproceedings{MetaMirrorDescent, 
  author={Zhang, Yilang and Li, Bingcong and Giannakis, Georgios B.}, 
  title={Meta-Learning with Versatile Loss Geometries for Fast Adaptation Using Mirror Descent}, 
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  year={2024}, 
  volume={},
  number={},
  pages={5220-5224},
  doi={10.1109/ICASSP48485.2024.10448144}
}
```
