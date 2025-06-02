import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

TRAINING_DATASET = {
    "annotation_path": "/fs/scratch/PAS1576/drogozhang/lm-extend-analysis/data/datasets/training/train.json",
    "data_path": "/fs/scratch/PAS1576/drogozhang/lm-extend-analysis",
}

TRAINING_TEXT_DATASET = {
    "annotation_path": "/fs/scratch/PAS1576/drogozhang/lm-extend-analysis/data/datasets/training/text.json",
    "data_path": "/fs/scratch/PAS1576/drogozhang/lm-extend-analysis",
}

TRAINING_IMAGE_DATASET = {
    "annotation_path": "/fs/scratch/PAS1576/drogozhang/lm-extend-analysis/data/datasets/training/image.json",
    "data_path": "/fs/scratch/PAS1576/drogozhang/lm-extend-analysis",
}

TRAINING_VIDEO_DATASET = {
    "annotation_path": "/fs/scratch/PAS1576/drogozhang/lm-extend-analysis/data/datasets/training/video.json",
    "data_path": "/fs/scratch/PAS1576/drogozhang/lm-extend-analysis",
}

TRAINING_MM_DATASET = {
    "annotation_path": "/fs/scratch/PAS1576/drogozhang/lm-extend-analysis/data/datasets/training/train_multi.json",
    "data_path": "/fs/scratch/PAS1576/drogozhang/lm-extend-analysis",
}

TRAINING_DATASET

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "training_dataset": TRAINING_DATASET,
    "training_text_dataset": TRAINING_TEXT_DATASET,
    "training_image_dataset": TRAINING_IMAGE_DATASET,
    "training_video_dataset": TRAINING_VIDEO_DATASET,
    "training_mm_dataset": TRAINING_MM_DATASET,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
