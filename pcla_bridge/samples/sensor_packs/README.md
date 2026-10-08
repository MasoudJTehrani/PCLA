# Sensor packs — what `input_data` each agent expects

Each pack is a real capture of an agent's per-frame sensor input: the sensor list, array shapes/dtypes, unit conventions, example camera frames, and the actual captured values. Use it to see exactly what to feed the bridge.

Agents with **identical sensor rigs share one pack** (to save space); each pack folder is named to list every variant it covers (seed numbers dropped). Find your agent below:

> **Declared vs. used.** A pack lists every sensor the agent *declares* (its `sensors()`), which for a few variants includes sensors the loaded checkpoint ignores. In particular `tfv6_noradar` and `tfv6_visiononly` still declare 4 radars (TransFuser-v6 hardcodes `radar=True` in `sensors()`), but their configs set `use_radar_detection=False`, so the model drops the radar branch. The bridge auto-fills those radars with empty “no detections” frames, so a radar-less source can still drive them; radar-using variants keep radar required.

| agent | sensor pack | sensors |
|---|---|---|
| `if_if` | [if_if/](if_if/) | rgb, rgb_left, rgb_right, lidar, imu, gps, speed |
| `lav_fast` | [lav_lav+fast/](lav_lav+fast/) _(shares `lav_lav+fast`)_ | EGO, GPS, IMU, LIDAR, RGB_0, RGB_1, RGB_2, TEL_RGB |
| `lav_lav` | [lav_lav+fast/](lav_lav+fast/) _(shares `lav_lav+fast`)_ | EGO, GPS, IMU, LIDAR, RGB_0, RGB_1, RGB_2, TEL_RGB |
| `lbc_lb` | [lbc_lb+nc/](lbc_lb+nc/) _(shares `lbc_lb+nc`)_ | EGO, GPS, RGB_0, RGB_1, RGB_2 |
| `lbc_nc` | [lbc_lb+nc/](lbc_lb+nc/) _(shares `lbc_lb+nc`)_ | EGO, GPS, RGB_0, RGB_1, RGB_2 |
| `lmdrive_llama` | [lmdrive_llava+vicuna+llama/](lmdrive_llava+vicuna+llama/) _(shares `lmdrive_llava+vicuna+llama`)_ | rgb_front, rgb_left, rgb_right, rgb_rear, lidar, imu, gps, speed |
| `lmdrive_llava` | [lmdrive_llava+vicuna+llama/](lmdrive_llava+vicuna+llama/) _(shares `lmdrive_llava+vicuna+llama`)_ | rgb_front, rgb_left, rgb_right, rgb_rear, lidar, imu, gps, speed |
| `lmdrive_vicuna` | [lmdrive_llava+vicuna+llama/](lmdrive_llava+vicuna+llama/) _(shares `lmdrive_llava+vicuna+llama`)_ | rgb_front, rgb_left, rgb_right, rgb_rear, lidar, imu, gps, speed |
| `minddrive_05b` | [orion_base+minddrive_05b+minddrive_3b/](orion_base+minddrive_05b+minddrive_3b/) _(shares `orion_base+minddrive_05b+minddrive_3b`)_ | CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT, IMU, GPS, SPEED |
| `minddrive_3b` | [orion_base+minddrive_05b+minddrive_3b/](orion_base+minddrive_05b+minddrive_3b/) _(shares `orion_base+minddrive_05b+minddrive_3b`)_ | CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT, IMU, GPS, SPEED |
| `neat_aim2ddepth` | [neat_aimbev+aim2dsem+aim2ddepth/](neat_aimbev+aim2dsem+aim2ddepth/) _(shares `neat_aimbev+aim2dsem+aim2ddepth`)_ | rgb, rgb_front, imu, gps, speed |
| `neat_aim2dsem` | [neat_aimbev+aim2dsem+aim2ddepth/](neat_aimbev+aim2dsem+aim2ddepth/) _(shares `neat_aimbev+aim2dsem+aim2ddepth`)_ | rgb, rgb_front, imu, gps, speed |
| `neat_aimbev` | [neat_aimbev+aim2dsem+aim2ddepth/](neat_aimbev+aim2dsem+aim2ddepth/) _(shares `neat_aimbev+aim2dsem+aim2ddepth`)_ | rgb, rgb_front, imu, gps, speed |
| `neat_neat` | [neat_neat/](neat_neat/) | rgb, rgb_left, rgb_right, rgb_front, bev, imu, gps, speed |
| `orion_base` | [orion_base+minddrive_05b+minddrive_3b/](orion_base+minddrive_05b+minddrive_3b/) _(shares `orion_base+minddrive_05b+minddrive_3b`)_ | CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT, IMU, GPS, SPEED |
| `simlingo_simlingo` | [simlingo_simlingo/](simlingo_simlingo/) | rgb_0, imu, gps, speed |
| `tfv3_gf` | [tfv3_tf+lf+gf/](tfv3_tf+lf+gf/) _(shares `tfv3_tf+lf+gf`)_ | rgb_front, rgb_left, rgb_right, imu, gps, speed, lidar |
| `tfv3_lf` | [tfv3_tf+lf+gf/](tfv3_tf+lf+gf/) _(shares `tfv3_tf+lf+gf`)_ | rgb_front, rgb_left, rgb_right, imu, gps, speed, lidar |
| `tfv3_ltf` | [tfv3_ltf/](tfv3_ltf/) | rgb_front, rgb_left, rgb_right, imu, gps, speed |
| `tfv3_tf` | [tfv3_tf+lf+gf/](tfv3_tf+lf+gf/) _(shares `tfv3_tf+lf+gf`)_ | rgb_front, rgb_left, rgb_right, imu, gps, speed, lidar |
| `tfv4_aim_0` | [tfv4_aim/](tfv4_aim/) _(shares `tfv4_aim`)_ | rgb_front, imu, gps, speed |
| `tfv4_l6_0` | [tfv4_lav+wp+l6/](tfv4_lav+wp+l6/) _(shares `tfv4_lav+wp+l6`)_ | rgb_front, imu, gps, speed, lidar |
| `tfv4_lav_0` | [tfv4_lav+wp+l6/](tfv4_lav+wp+l6/) _(shares `tfv4_lav+wp+l6`)_ | rgb_front, imu, gps, speed, lidar |
| `tfv4_wp_0` | [tfv4_lav+wp+l6/](tfv4_lav+wp+l6/) _(shares `tfv4_lav+wp+l6`)_ | rgb_front, imu, gps, speed, lidar |
| `tfv5_alltowns` | [tfv5_alltowns+notown13/](tfv5_alltowns+notown13/) _(shares `tfv5_alltowns+notown13`)_ | rgb_front, imu, gps, speed, lidar |
| `tfv5_notown13` | [tfv5_alltowns+notown13/](tfv5_alltowns+notown13/) _(shares `tfv5_alltowns+notown13`)_ | rgb_front, imu, gps, speed, lidar |
| `tfv6_4cameras` | [tfv6_4cameras+noradar+visiononly/](tfv6_4cameras+noradar+visiononly/) _(shares `tfv6_4cameras+noradar+visiononly`)_ | rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_6, lidar1, lidar2, radar1, radar2, radar3, radar4, imu, gps, speed |
| `tfv6_noradar` | [tfv6_4cameras+noradar+visiononly/](tfv6_4cameras+noradar+visiononly/) _(shares `tfv6_4cameras+noradar+visiononly`)_ | rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_6, lidar1, lidar2, radar1, radar2, radar3, radar4, imu, gps, speed |
| `tfv6_notown13` | [tfv6_regnet+resnet+notown13/](tfv6_regnet+resnet+notown13/) _(shares `tfv6_regnet+resnet+notown13`)_ | rgb_1, rgb_2, rgb_3, lidar1, lidar2, radar1, radar2, radar3, radar4, imu, gps, speed |
| `tfv6_regnet` | [tfv6_regnet+resnet+notown13/](tfv6_regnet+resnet+notown13/) _(shares `tfv6_regnet+resnet+notown13`)_ | rgb_1, rgb_2, rgb_3, lidar1, lidar2, radar1, radar2, radar3, radar4, imu, gps, speed |
| `tfv6_resnet` | [tfv6_regnet+resnet+notown13/](tfv6_regnet+resnet+notown13/) _(shares `tfv6_regnet+resnet+notown13`)_ | rgb_1, rgb_2, rgb_3, lidar1, lidar2, radar1, radar2, radar3, radar4, imu, gps, speed |
| `tfv6_visiononly` | [tfv6_4cameras+noradar+visiononly/](tfv6_4cameras+noradar+visiononly/) _(shares `tfv6_4cameras+noradar+visiononly`)_ | rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_6, lidar1, lidar2, radar1, radar2, radar3, radar4, imu, gps, speed |
| `tt_tt` | [tt_tt/](tt_tt/) | rgb_front, rgb_left, rgb_right, rgb_back, lidar, imu, gps, speed, topdown |
| `wor_lb` | [wor_lb+nc/](wor_lb+nc/) _(shares `wor_lb+nc`)_ | EGO, GPS, Wide_RGB_0, Wide_RGB_1, Wide_RGB_2, Narrow_RGB |
| `wor_nc` | [wor_lb+nc/](wor_lb+nc/) _(shares `wor_lb+nc`)_ | EGO, GPS, Wide_RGB_0, Wide_RGB_1, Wide_RGB_2, Narrow_RGB |

## Not portable — no sensor pack

These agents cannot be driven from a real sensor source (the bridge refuses them too), so there is no pack to capture:

| agent | why |
|---|---|
| `autoware_v1` | not a run_step model (external ROS 2 stack) |
| `carl_carl_0` | privileged: reads simulator ground-truth (ego pose / BEV map) |
| `carl_carlv11` | privileged: reads simulator ground-truth (ego pose / BEV map) |
| `carl_plant_0` | privileged: reads simulator ground-truth (ego pose / BEV map) |
| `carl_roach_0` | privileged: reads simulator ground-truth (ego pose / BEV map) |
| `plant2_plant2_0` | privileged: reads simulator ground-truth (ego pose / BEV map) |
