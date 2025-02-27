<h1 style="text-align: center;">
Memories of Forgotten Concepts [CVPR 2025]
</h1>

<a href="https://matanr.github.io/Memories_of_Forgotten_Concepts/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/abs/2412.00782"><img src="https://img.shields.io/badge/arXiv-2311.17891-b31b1b.svg"></a>





<img src="images/teaser.png">


> [Memories of Forgotten Concepts](https://matanr.github.io/Memories_of_Forgotten_Concepts/)
>
>
> [Matan Rusanovsky<sup>*</sup>](https://scholar.google.com/citations?user=5TS4vucAAAAJ&hl=en&oi=ao), [Shimon Malnick<sup>*</sup>](https://www.malnick.net/in/shimon-malnick-1b8404125/), [Amir Jevnisek<sup>*</sup>](https://scholar.google.com/citations?hl=en&user=czm6bkUAAAAJ), [Ohad Fried](https://www.ohadf.com/), [Shai Avidan](http://www.eng.tau.ac.il/~avidan/)
>
>
> <sup>*</sup>Equal Contribution
>
> ❓ Many studies aim to erase concepts from diffusion models. While these models may no longer generate images tied to the erased concept when prompted, we ask: **Is the concept truly erased? Could the model still reproduce it through other means?**
>
> ↪️Instead of analyzing a model that erased some concept by generating many images and analyzing them, we propose a method that analyzes it using latents that generate the erased concept.

<br>

</div>

# Getting Started
## Requirments:
1. Download [mscoco17](https://cocodataset.org/#download).
2. Download the ablated models (Links for [[Object](https://drive.google.com/file/d/1e5aX8gkC34YaHGR0S1-EQwBmUXiAPvpE/view), [Others](https://drive.google.com/file/d/1yeZNJ8MoHsisdZmt5lbnG_kSgl5xned0/view)]).
3. Generate the datasets of the erased concepts using the appropriate csv file. For example, to generate the Nudity dataset:
```shell
cd Memories_of_Forgotten_Concepts/src
export CONCEPT=nudity
export PROMPT_FILE=../prompts/${CONCEPT}.csv
export SAVE_PATH=../datasets
mkdir -p $SAVE_PATH
python generate_dataset.py --prompts_path ${PROMPT_FILE} --concept ${CONCEPT} --save_path ${SAVE_PATH} --device cuda:0
```
For objects use the ```generate_object_dataset.py```, for example to generate the Parachute object dataset:
```shell
cd Memories_of_Forgotten_Concepts/src
export CONCEPT=parachute
export SAVE_PATH=../datasets
mkdir -p $SAVE_PATH
python generate_object_dataset.py --concept ${CONCEPT} --save_path ${SAVE_PATH}
```
4. Download the [style classifier](https://drive.google.com/file/d/1me_MOrXip1Xa-XaUrPZZY7i49pgFe1po/view) for detection of the Van Gogh concept. Add an environemnt variable for the classifier:
```shell
export STYLE_CLASSIFIER_DIR=/path/to/cls/dir
```

## Setup Environment
### Conda + pip
create an environemnt using the supplied requirements.txt file:
```shell
git clone https://github.com/matanr/Memories_of_Forgotten_Concepts
cd Memories_of_Forgotten_Concepts/src
conda create -n mem python=3.10
conda activate mem
pip install -r requirements.txt
```

### [Docker Setup Information](docker/DOCKER-INFO.md)

## Running
#### Output directory
Make sure the outputs directory contains the ``concept`` name in either configuration.
In addition, it is recommended to specify the model name and the experimental configuration.
For example, when running on ``ESD`` that erased the concept ``nudity``, in the ``concept-level`` configuration, set: 

```
<out directory>=memories_of_ESD_nudity
```
Similarly, in the "image-level" configuration, set:
```
<out directory>=many_memories_of_ESD_nudity
```
#### Ablated Model
For all models except for AdvUnlearn, set:
```
--ablated_model <path to the ablated model>
```
Instead, for AdvUnlearn, include: 
```
--ablated_text_encoder OPTML-Group/AdvUnlearn
```
💡 Only one of these two options (--ablated_model or --ablated_text_encoder) should be used at a time, according to the model that is being analyzed.

### Concept-Level
Perform a concept-level analysis:

```shell
python memory_of_an_ablated_concept.py
--reference_dataset_root <path to mscoco17>
--out_dir <out directory>
--ablated_concept_name <nudity/vangogh/church/garbage_truck/tench/parachute>
--dataset_root <path to the dataset of images of ablated_concept_name>
--diffusion_inversion_method <renoise/nti>
--num_diffusion_inversion_steps 50
[--ablated_model <path to the ablated model> | --ablated_text_encoder OPTML-Group/AdvUnlearn]
```

### Image-Level
Perform an image-level analysis:
```shell
python many_memories_of_an_ablated_image.py 
--reference_dataset_root <path to mscoco17>
--out_dir <out directory>
--ablated_concept_name <nudity/vangogh/church/garbage_truck/tench/parachute>
--dataset_root <path to the dataset of images of ablated_concept_name>
--num_vae_inversion_steps 3000
--diffusion_inversion_method <renoise/nti>
--num_diffusion_inversion_steps 50
[--ablated_model <path to the ablated model> | --ablated_text_encoder OPTML-Group/AdvUnlearn]
```

# BibTex
```bib
@misc{rusanovsky2024memoriesforgottenconcepts,
      title={Memories of Forgotten Concepts}, 
      author={Matan Rusanovsky and Shimon Malnick and Amir Jevnisek and Ohad Fried and Shai Avidan},
      year={2024},
      eprint={2412.00782},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.00782}, 
}
```

# Acknowlegments
This repository is built upon and incorporates code from [Diffusion-MU-Attack](https://github.com/OPTML-Group/Diffusion-MU-Attack), [AdvUnlearn](https://github.com/OPTML-Group/AdvUnlearn) and [Renoise](https://github.com/garibida/ReNoise-Inversion).
