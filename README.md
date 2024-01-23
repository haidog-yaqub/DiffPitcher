# Diff-Pitcher: Diffusion-based Singing Voice Pitch Correction

Official Pytorch Implementation of [Diff-Pitcher: Diffusion-based Singing Voice Pitch Correction](https://engineering.jhu.edu/lcap/data/uploads/pdfs/waspaa2023_hai.pdf)

--------------------

<img src="img\diffpitcher.jpg" width="300px">

Diff-Pitcher

- [Demo](#demo)
- [Examples](#examples)
- [Todo](#todo)
- [References](#references)
- [Acknowledgement](#acknowledgement)

## Demo

🎵 Listen to [examples](https://jhu-lcap.github.io/Diff-Pitcher/)

## Todo
- [x] Update codes and demo
- [x] Support 🤗 [Diffusers](https://github.com/huggingface/diffusers)
- [x] Upload checkpoints
- [x] Pipeline tutorial
- [ ] Merge to [Your-Stable-Audio](https://github.com/haidog-yaqub/Your-Stable-Audio)
## Examples

- [x] template-based automatic pitch correction: [template_based_apc.py](template_based_apc.py)
- [x] score-based automatic pitch correction: [score_based_apc.py](score_based_apc.py)

## References

If you find the code useful for your research, please consider citing:

```bibtex
@inproceedings{hai2023diff,
  title={Diff-Pitcher: Diffusion-Based Singing Voice Pitch Correction},
  author={Hai, Jiarui and Elhilali, Mounya},
  booktitle={2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

This repo is inspired by:

```bibtex
@article{popov2021diffusion,
  title={Diffusion-based voice conversion with fast maximum likelihood sampling scheme},
  author={Popov, Vadim and Vovk, Ivan and Gogoryan, Vladimir and Sadekova, Tasnima and Kudinov, Mikhail and Wei, Jiansheng},
  journal={arXiv preprint arXiv:2109.13821},
  year={2021}
}
```
```bibtex
@inproceedings{liu2022diffsinger,
  title={Diffsinger: Singing voice synthesis via shallow diffusion mechanism},
  author={Liu, Jinglin and Li, Chengxi and Ren, Yi and Chen, Feiyang and Zhao, Zhou},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={36},
  number={10},
  pages={11020--11028},
  year={2022}
}
```

## Acknowledgement

We borrow code from following repos:

 - `Diffusion Schedulers` are based on 🤗 [Diffusers](https://github.com/huggingface/diffusers)
 - `2D UNet` is based on [DiffVC](https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC)