import argparse
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.episode_utils import remove_episodes

def remove(repo_id, episode_indices, root=None):
    dataset = LeRobotDataset(repo_id, root=root, episodes=None)
    dataset = remove_episodes(dataset, episode_indices)
    
    # Save the modified dataset
    dataset.push_to_hub()
    print(f"Removed episodes {episode_indices} from dataset {repo_id}.")

def str_to_int_list(str):
    return [int(i) for i in str.split(',')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove episodes from a LeRobotDataset.")
    parser.add_argument("action", type=str, choices=["remove"], help="Action to perform on the dataset.")
    parser.add_argument("--repo-id", type=str, required=True, help="Repository ID of the dataset.")
    parser.add_argument("--episode-indices", type=str_to_int_list, help="List of episode indices to remove. e.g. 0, 1, 2")
    parser.add_argument("--root", type=str, default=None, help="Root directory of the dataset. If not provided, the default location will be used.")
    args = parser.parse_args()
    if args.action == "remove":
        remove(args.repo_id, args.episode_indices, root=args.root)
    else:
        raise ValueError(f"Invalid action {args.action}. Available actions are: remove.")

