from v21.dataset_creator.merged_dataset_creator import MergedDatasetCreator

# dataset_ids = ["standard/pika_4paper_blackbox", "standard/pika_4paper_tanbox"]
# new_repo_id = "standard/pika_place_pocket_pack_tissues_in_box"
# dataset_ids = ["standard/pika_4paper_blackbox_diff", "standard/pika_4paper_tanbox_diff"]
# new_repo_id = "standard/pika_4paper_blackbox_tanbox_diff"
# dataset_ids = ["standard/umi_4paper_blackbox", "standard/umi_4paper_tanbox"]
# new_repo_id = "standard/pika_umi_place_pocket_pack_tissues_in_box"
# dataset_ids = ["standard/umi_place_pocket_pack_tissues_in_box", "standard/pika_place_pocket_pack_tissues_in_box", "standard/pika_4paper_blackbox_tanbox_diff"]
# new_repo_id = "standard/pika_mix_place_pocket_pack_tissues_in_box"
# dataset_ids = ["standard/umi_fold_towel_blue", "standard/umi_fold_towel_white", "standard/umi_fold_towel_yellow"]
# new_repo_id = "standard/umi_fold_towel"
# dataset_ids = ["standard/pika_ctrl_fold_towel_blue", "standard/pika_ctrl_fold_towel_gray_fix", "standard/pika_ctrl_fold_towel_yellow_fix"]
# new_repo_id = "standard/pika_ctrl_fold_towel"
# dataset_ids = ["standard/pika_umi_fold_towel", "standard/pika_ctrl_fold_towel"]
# new_repo_id = "standard/pika_mix_fold_towel"
# dataset_ids = ["standard/pika_mix_fold_towel", "standard/pika_ctrl_fold_towel_gray_add160", "standard/pika_umi_flatten_towel_yellow"]
# new_repo_id = "standard/pika_mix_fold_towel_2"
# dataset_ids = ["standard/pika_ctrl_fold_towel_blue", "standard/pika_umi_fold_towel"]
# new_repo_id = "standard/pika_mix_fold_towel_small"
# dataset_ids = ["standard/darwin02_0401_2_lerobo", "standard/darwin02_0401_2_lerobot_2", "standard/darwin02_0401_lerobot"]
# new_repo_id = "standard/darwin02_0401"
# dataset_ids = ["standard/darwin02_0401", "standard/darwin02_0408"]
# new_repo_id = "standard/darwin02_0410"
dataset_ids = ["standard/darwin02_0501_2", "standard/darwin02_0502", "standard/darwin02_0502_error"]
new_repo_id = "standard/darwin02_0503"
tolerance_s = 1e-4

MergedDatasetCreator().create(dataset_ids, new_repo_id, tolerance_s)
