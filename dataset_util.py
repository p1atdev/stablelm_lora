from datasets import load_dataset


def create_dataset(
    dataset_name_or_path: str,
    map_action: callable,
    val_split: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
):
    data = load_dataset(dataset_name_or_path)
    train_val = data["train"].train_test_split(
        test_size=val_split, shuffle=shuffle, seed=seed
    )
    train_data = train_val["train"].shuffle(seed=seed).map(map_action)
    val_data = train_val["test"].shuffle(seed=seed).map(map_action)

    return train_data, val_data
