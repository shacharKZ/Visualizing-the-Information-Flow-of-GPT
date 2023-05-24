# Visualizing the Information Flow of GPT

This repository provides an implementation of the flow-graph modeling of our paper: Interpreting Transformer's Attention Dynamic Memory and Visualizing the Semantic Information Flow of GPT ([arxiv])(https://arxiv.org/abs/2305.13417)


Please try our demo: [![Colab ROME Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lsdlesXaEsVYwvcJWJac6jcxSr69gKcM?usp=sharing) 

This notebook enables to create dynamic plots that reflect the forward passes of GPTs from semantic perspective. These plots effectively illustrate the information flow within the models and provide insights into the impact of each component on the semantic information flow.


We also peovide examples of the modeling plots in the folder "dynamic example". Those are HTML files, please download them and open them in your browser.

Our implementation works with [GPT-2](https://github.com/shacharKZ/Visualizing-the-Information-Flow-of-GPT/blob/main/visual_nets_GPT_2.ipynb) and [GPT-J](https://github.com/shacharKZ/Visualizing-the-Information-Flow-of-GPT/blob/main/visual_nets_guided_with_GPT_J.ipynb), providing the latter as a guided notebook for adjusting the code to any GPT-like model.

Feel free to open an issue if you find any problems or contact us to discuss any related topics.


## Citation
```bibtex
@misc{katz2023interpreting,
      title={Interpreting Transformer's Attention Dynamic Memory and Visualizing the Semantic Information Flow of GPT}, 
      author={Shahar Katz and Yonatan Belinkov},
      year={2023},
      eprint={2305.13417},
      archivePrefix={arXiv},
}
```
