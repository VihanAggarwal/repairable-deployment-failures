"""Local checks of every piece that does not need a GPU, a simulator or a checkpoint."""
import json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import plus_common as pc
import lerobot_tools as lt
import simpler_eval as se
import droid_pi05 as dp

# real LIBERO-Plus names, copied from the suite
B = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
assert pc.categorize(B + "_view_0_0_100_0_0_initstate_0_noise_1") == "sensor_noise"
assert pc.categorize(B + "_language_1_view_0_0_100_0_0_initstate_0") == "language"
assert pc.categorize("KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_light_10") == "light"
assert pc.categorize("KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_1") == "background"
assert pc.categorize("KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_tb_12") == "background"
assert pc.categorize("KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_add_14") == "layout"
assert pc.categorize(B + "_view_0_0_100_2_8_initstate_0") == "camera"
assert pc.categorize("KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_195_0_0_initstate_0") == "camera"
assert pc.categorize("KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_52_15_100_0_0_initstate_0") == "camera"
assert pc.categorize("LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_0_0_100_0_0_initstate_2") == "robot_init"
assert pc.categorize("LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_0_0_initstate_428") == "robot_init"
assert pc.categorize(B + "_view_0_0_100_0_0_initstate_0") == "other"
rows = [dict(category="camera", success=True), dict(category="camera", success=False), dict(category="light", success=True)]
s = pc.summarize(rows, "test")
assert s["ALL"]["n"] == 3 and abs(s["camera"]["success"] - 0.5) < 1e-9

with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "eval_info.json")
    json.dump({"per_task": [{"task_group": "libero_10", "task_id": 3, "metrics": {"pc_success": 100.0, "n_episodes": 1}},
                            {"task_group": "libero_10", "task_id": 5, "metrics": {"successes": [False]}}],
               "per_group": {"libero_10": {"pc_success": 50.0}}, "overall": {"pc_success": 50.0}}, open(p, "w"))
    assert lt.parse_eval_info(p) == {3: True, 5: False}, lt.parse_eval_info(p)

img = (np.random.default_rng(0).random((256, 320, 3)) * 255).astype(np.uint8)
for c in se.CONDS:
    out = se.perturb(img, c)
    assert out.shape == img.shape and out.dtype == np.uint8, c
assert np.array_equal(se.perturb(img, "clean"), img)
assert np.abs(se.perturb(img, "cam3").astype(int) - img).mean() > 1

ident = dp.apply(img, np.zeros(6))
assert np.abs(ident.astype(int) - img.astype(int)).max() <= 1, "identity correction must be a no-op"
for name, p_true in dp.NUIS.items():
    assert np.abs(dp.apply(img, p_true).astype(int) - img).mean() > 2, name
assert dp.pad224(img[:180, :320]).shape == (224, 224, 3)
assert abs(dp.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9
print("ALL LOCAL TESTS PASS")
