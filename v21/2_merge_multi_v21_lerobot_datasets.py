from dataset_creator.merged_dataset_creator import MergedDatasetCreator

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
# dataset_ids = ["standard/darwin02_0501_2", "standard/darwin02_0502", "standard/darwin02_0502_error"]
# new_repo_id = "standard/darwin02_0503"
# dataset_ids = ["/home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_gray_small_init_nonidle", "/home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_gray_small_nonidle", "/home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_green_small_nonidle"]
# new_repo_id = "/home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt"
# dataset_ids = ["/home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_gray_small_init_nonidle", "/home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_gray_small_nonidle", "/home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_green_small_nonidle"]
# new_repo_id = "/home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt"
# dataset_ids = ["/home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec", "/home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec_t"]
# new_repo_id = "/home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec_b_t"
# dataset_ids = ["/home/standard/workspace/test/kai0/data/my_Task_A/advantage_kai0_repred_b_bt_reest_merged", "/home/standard/workspace/test/kai0/data/Task_A/dagger", "/home/standard/workspace/test/kai0/data/Task_A/dagger_t", "/home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec_b_t"]
# new_repo_id = "/home/standard/workspace/test/kai0/data/my_Task_A/merge_kai0_advantage_b_t_kai0_dagger_b_t_std_dagger_b_t"
# dataset_ids = ["/home/standard/workspace/test/kai0/data/my_Task_A/advantage_kai0_repred_b_bt_reest_merged_delete", "/home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec_delete"]
# new_repo_id = "/home/standard/workspace/test/kai0/data/my_Task_A/merge_kai0_advantage_b_t_std_dagger_b"
# dataset_ids = ["/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_0526", "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_0527", "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_0528", "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_0529", "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_0601", "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_0602", "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_0603"]
# new_repo_id = "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_0526_0603"

# dataset_ids = ["/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_yellow_0603", 
#                "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_yellow_0604", 
#                "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_yellow_0605", 
#                "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_yellow_0608", 
#                "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_yellow_0609"]
# new_repo_id = "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_yellow_0603_0609"

# dataset_ids = ["/home/standard/workspace/gitlab/openpi/.cache/huggingface/lerobot/standard/piperx/piperx_grab_bigbox_0526_0603_nonidle_delete", "/home/standard/workspace/gitlab/openpi/.cache/huggingface/lerobot/standard/piperx/piperx_grab_bigbox_yellow_0603_0609_nonidle"]
# new_repo_id = "/home/standard/workspace/gitlab/openpi/.cache/huggingface/lerobot/standard/piperx/piperx_grab_bigbox_0526_0609_nonidle"

# 全流程（部分排除了夹爪空张的数据）+ 短dagger，缺点是容易停在半路
# dataset_ids = ["/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0617", 
#                "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0618", 
#                "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0622", 
#                "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0623", 
#                "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0624", 
#                "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_0529_merged", 
#                "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_yellow_0603_0609_nonidle"]
# new_repo_id = "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0624"

# 全流程（部分排除了夹爪空张的数据）
# dataset_ids = ["/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_data_0529_merged_nonidle", 
#                "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_yellow_0603_0609_nonidle"]
# new_repo_id = "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_yellow_0529_0609_nonidle"

# 0703
# 长dagger
# dataset_ids = ["/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0629", 
#                "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0630", 
#                "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0701", 
#                "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0702", 
#                "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0703"]
# new_repo_id = "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0629_0703"
# # 全流程 + 长dagger
# dataset_ids = ["/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0629_0703_nonidle_delete",
#                "/home/standard/agilex/lerobot/piperx/piperx_grab_bigbox_yellow_0529_0609_nonidle"]
# new_repo_id = "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0529_0703_nonidle"

# # piper peg insertion
# dataset_ids = ["/home/standard/workspace/gitlab/RLinf/temp/dataset/piper_peg_insertion_rot6d/lerobot/id_0",
#                "/home/standard/workspace/gitlab/RLinf/temp/dataset/piper_peg_insertion_rot6d/lerobot/id_1",
#                "/home/standard/workspace/gitlab/RLinf/temp/dataset/piper_peg_insertion_rot6d/lerobot/id_2",
#                "/home/standard/workspace/gitlab/RLinf/temp/dataset/piper_peg_insertion_rot6d/lerobot/id_3",
#                "/home/standard/workspace/gitlab/RLinf/temp/dataset/piper_peg_insertion_rot6d/lerobot/id_4",
#                "/home/standard/workspace/gitlab/RLinf/temp/dataset/piper_peg_insertion_rot6d/lerobot/id_5"]
# new_repo_id = "/home/standard/workspace/gitlab/RLinf/temp/dataset/piper_peg_insertion_rot6d/lerobot/piper_peg_insertion_rot6d"

# 0713
dataset_ids = ["/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0706",
                "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0707",
                "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0708",
                "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0709",
                "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_data_yellow_0713"]
new_repo_id = "/home/standard/agilex/lerobot/piperx/dagger/piperx_grab_bigbox_yellow_0706_0713"

# /home/standard/workspace/test/kai0/data/my_Task_A/advantage_kai0_repred_b_bt_reest_merged # kai0, advantage + advantage_t
# /home/standard/workspace/test/kai0/data/Task_A/dagger
# /home/standard/workspace/test/kai0/data/Task_A/dagger_t
# /home/standard/workspace/test/kai0/data/standard_Task_A/dagger/piper_fold_tshirt_task_a_aligned_recodec_b_t # standard, dagger + dagger_t
tolerance_s = 1e-4

MergedDatasetCreator().create(dataset_ids, new_repo_id, tolerance_s)
