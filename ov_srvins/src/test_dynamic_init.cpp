#include "core/VioManagerOptions.h"
#include <cmath>
#include <csignal>
#include <deque>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <vector>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>

#if ROS_AVAILABLE == 1
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#endif

#include "initializer/dynamic/DynamicInitializer.h"
#include "initializer/InertialInitializerOptions.h"
#include "sim/SimulatorInit.h"

#include "track/TrackSIM.h"
#include "types/IMU.h"
#include "types/Landmark.h"
#include "types/PoseJPL.h"
#include "utils/colors.h"
#include "utils/sensor_data.h"

// Define the function to be called when ctrl-c (SIGINT) is sent to process
void signal_callback_handler(int signum) { std::exit(signum); }

// taken from ov_eval/src/alignment/AlignUtils.h
static inline double get_best_yaw(const Mat3& C) {
  double A = C(0, 1) - C(1, 0);
  double B = C(0, 0) + C(1, 1);
  // return M_PI_2 - atan2(B, A);
  return atan2(A, B);
}

// taken from ov_eval/src/alignment/AlignUtils.h
void align_posyaw_single(const Vec4 &q_es_0, const Vec3 &p_es_0, const Vec4 &q_gt_0,
                         const Vec3 &p_gt_0, Mat3 &R, Vec3 &t) {
  Mat3 g_rot = ov_core::quat_2_Rot(q_gt_0).transpose();
  Mat3 est_rot = ov_core::quat_2_Rot(q_es_0).transpose();
  Mat3 C_R = est_rot * g_rot.transpose();
  float theta = get_best_yaw(C_R);
  R = ov_core::rot_z(theta);
  t.noalias() = p_gt_0 - R * p_es_0;
}

// Main function
int main(int argc, char **argv) {
    // Ensure we have a path, if the user passes it then we should use it
    std::string config_path = "unset_path_to_config.yaml";
    if (argc > 1) {
        config_path = argv[1];
    }

#if ROS_AVAILABLE == 1
    // Launch our ros node
    ros::init(argc, argv, "test_dynamic_init");
    auto nh = std::make_shared<ros::NodeHandle>("~");
    nh->param<std::string>("config_path", config_path, config_path);

    // Topics to publish
    auto pub_pathimu = nh->advertise<nav_msgs::Path>("/ov_srvins/pathimu", 2);
    PRINT_DEBUG("Publishing: %s\n", pub_pathimu.getTopic().c_str());
    auto pub_pathgt = nh->advertise<nav_msgs::Path>("/ov_srvins/pathgt", 2);
    PRINT_DEBUG("Publishing: %s\n", pub_pathgt.getTopic().c_str());
    auto pub_loop_point = nh->advertise<sensor_msgs::PointCloud>("/ov_srvins/loop_feats", 2);
    PRINT_DEBUG("Publishing: %s\n", pub_loop_point.getTopic().c_str());
    auto pub_points_sim = nh->advertise<sensor_msgs::PointCloud2>("/ov_srvins/points_sim", 2);
    PRINT_DEBUG("Publishing: %s\n", pub_points_sim.getTopic().c_str());
#endif

    // Load the config
    auto parser = std::make_shared<ov_core::YamlParser>(config_path);
#if ROS_AVAILABLE == 1
    parser->set_node_handler(nh);
#endif

    // Verbosity
    std::string verbosity = "INFO";
    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(verbosity);

    // Create the simulator
    ov_srvins::VioManagerOptions params;
    params.print_and_load(parser);
    params.print_and_load_simulation(parser);
    params.init_options.print_and_load_simulation(parser);
    if (!parser->successful()) {
        PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
        std::exit(EXIT_FAILURE);
    }
    ov_srvins::SimulatorInit sim(params.init_options);

    // Our initialization class objects
    auto tracker = std::make_shared<ov_core::TrackSIM>(params.init_options.camera_intrinsics, 0);
    
    // Create our VIO system
    // ov_srvins::VioManagerOptions vio_params;
    // vio_params.print_and_load(parser);
    params.num_opencv_threads = 0;       // uncomment if you want repeatability
    params.use_multi_threading_pubs = 0; // uncomment if you want repeatability
    params.use_multi_threading_subs = false;
    // Initialize our state propagator:
    auto propagator = std::make_shared<ov_srvins::Propagator>(params.imu_noises, params.gravity_mag);

    // Make the updater!
    auto updaterMSCKF = std::make_shared<ov_srvins::UpdaterMSCKF>(params.msckf_options,
                                                params.featinit_options);
    auto updaterSLAM = std::make_shared<ov_srvins::UpdaterSLAM>(
        params.slam_options, params.aruco_options, params.featinit_options);

    auto initializer = std::make_shared<ov_srvins::DynamicInitializer>(params.init_options, tracker->get_feature_database(), propagator, updaterMSCKF, updaterSLAM);

    // Create the state
    std::shared_ptr<ov_srvins::State> state = std::make_shared<ov_srvins::State>(params.state_options, params.init_options);

    // Timeoffset from camera to IMU
    VecX temp_camimu_dt;
    temp_camimu_dt.resize(1);
    temp_camimu_dt(0) = params.calib_camimu_dt;
    state->calib_dt_CAMtoIMU->set_value(temp_camimu_dt);
    state->calib_dt_CAMtoIMU->set_fej(temp_camimu_dt);

    // Loop through and load each of the cameras
    state->cam_intrinsics_cameras = params.camera_intrinsics;
    for (int i = 0; i < state->options.num_cameras; i++) {
      state->cam_intrinsics.at(i)->set_value(
          params.camera_intrinsics.at(i)->get_value());
      state->cam_intrinsics.at(i)->set_fej(
          params.camera_intrinsics.at(i)->get_value());
      state->calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
      state->calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
    }

    //===================================================================================
    //===================================================================================
    //===================================================================================

    // Buffer our camera image
    double buffer_timecam = -1;
    std::vector<int> buffer_camids;
    std::vector<std::vector<std::pair<size_t, Eigen::Vector2f>>> buffer_feats;

    // Continue to simulate until we have processed all the measurements
    signal(SIGINT, signal_callback_handler);
    while (sim.ok()) {
        // IMU: get the next simulated IMU measurement if we have it
        ov_core::ImuData message_imu;
        bool hasimu = sim.get_next_imu(message_imu.timestamp, message_imu.wm, message_imu.am);
        if (hasimu) {
            // imu_readings->push_back(message_imu);
            propagator->feed_imu(message_imu, -1);
        }

        // CAM: get the next simulated camera uv measurements if we have them
        double time_cam;
        std::vector<int> camids;
        std::vector<std::vector<std::pair<size_t, Eigen::Vector2f>>> feats;
        bool hascam = sim.get_next_cam(time_cam, camids, feats);
        if (hascam) {

            // Pass to our feature database / tracker
            if (buffer_timecam != -1) {

                // Feed it
                tracker->feed_measurement_simulation(buffer_timecam, buffer_camids, buffer_feats);

                // Display the resulting tracks
                // cv::Mat img_history;
                // tracker->display_history(img_history, 255, 255, 0, 255, 255, 255);
                // cv::imshow("Track History", img_history);
                // cv::waitKey(1);
            }
            buffer_timecam = time_cam;
            buffer_camids = camids;
            buffer_feats = feats;

            // Return states of our initializer
            double timestamp = -1;
            Eigen::MatrixXd covariance;

            // First we will try to make sure we have all the data required for our initialization
            boost::posix_time::ptime rT1 = boost::posix_time::microsec_clock::local_time();
            bool success = initializer->initialize(state);
            boost::posix_time::ptime rT2 = boost::posix_time::microsec_clock::local_time();
            double time = (rT2 - rT1).total_microseconds() * 1e-6;
            if (success) {
                PRINT_INFO(GREEN "[TEST]: successfully initialized with dynamic initializer in %.4f seconds!!\n" RESET, time);

                timestamp = state->timestamp;
                // First lets align the groundtruth state with the IMU state
                // NOTE: imu biases do not have to be corrected with the pos yaw alignment here...
                Eigen::Matrix<DataType, 17, 1> gt_imu;
                bool success1 = sim.get_state(timestamp + sim.get_true_parameters().calib_camimu_dt, gt_imu);
                assert(success1);
                Mat3 R_ESTtoGT_imu;
                Vec3 t_ESTtoGT_imu;
                align_posyaw_single(state->imu->quat(), state->imu->pos(), gt_imu.block(1, 0, 4, 1), gt_imu.block(5, 0, 3, 1), R_ESTtoGT_imu, t_ESTtoGT_imu);
                gt_imu.block(1, 0, 4, 1) = ov_core::quat_multiply(gt_imu.block(1, 0, 4, 1), ov_core::rot_2_quat(R_ESTtoGT_imu));
                gt_imu.block(5, 0, 3, 1) = R_ESTtoGT_imu.transpose() * (gt_imu.block(5, 0, 3, 1) - t_ESTtoGT_imu);
                gt_imu.block(8, 0, 3, 1) = R_ESTtoGT_imu.transpose() * gt_imu.block(8, 0, 3, 1);

                // Finally compute the error
                Eigen::Matrix<double, 15, 1> err = Eigen::Matrix<double, 15, 1>::Zero();
                Eigen::Matrix3d R_GtoI_gt = ov_core::quat_2_Rot(gt_imu.block(1, 0, 4, 1)).cast<double>();
                Eigen::Matrix3d R_GtoI_hat = state->imu->Rot().cast<double>();
                err.block(0, 0, 3, 1) = -ov_core::log_so3(R_GtoI_gt * R_GtoI_hat.transpose()).cast<double>();
                err.block(3, 0, 3, 1) = gt_imu.block(5, 0, 3, 1).cast<double>() - state->imu->pos().cast<double>();
                err.block(6, 0, 3, 1) = gt_imu.block(8, 0, 3, 1).cast<double>() - state->imu->vel().cast<double>();
                err.block(9, 0, 3, 1) = gt_imu.block(11, 0, 3, 1).cast<double>() - state->imu->bias_g().cast<double>();
                err.block(12, 0, 3, 1) = gt_imu.block(14, 0, 3, 1).cast<double>() - state->imu->bias_a().cast<double>();

                // debug print the error of the recovered IMU state
                PRINT_INFO(REDPURPLE "e_ori = %.3f,%.3f,%.3f | %.3f,%.3f,%.3f,%.3f (true) | %.3f,%.3f,%.3f,%.3f (est)\n" RESET,
                           180.0 / M_PI * err(0 + 0), 180.0 / M_PI * err(0 + 1), 180.0 / M_PI * err(0 + 2), gt_imu(1), gt_imu(2), gt_imu(3),
                           gt_imu(4), state->imu->quat()(0), state->imu->quat()(1), state->imu->quat()(2), state->imu->quat()(3));
                PRINT_INFO(REDPURPLE "e_pos = %.3f,%.3f,%.3f | %.3f,%.3f,%.3f (true) | %.3f,%.3f,%.3f (est)\n" RESET, err(3 + 0), err(3 + 1),
                           err(3 + 2), gt_imu(5), gt_imu(6), gt_imu(7), state->imu->pos()(0), state->imu->pos()(1), state->imu->pos()(2));
                PRINT_INFO(REDPURPLE "e_vel = %.3f,%.3f,%.3f | %.3f,%.3f,%.3f (true) | %.3f,%.3f,%.3f (est)\n" RESET, err(6 + 0), err(6 + 1),
                           err(6 + 2), gt_imu(8), gt_imu(9), gt_imu(10), state->imu->vel()(0), state->imu->vel()(1), state->imu->vel()(2));
                PRINT_INFO(REDPURPLE "e_bias_g = %.3f,%.3f,%.3f | %.3f,%.3f,%.3f (true) | %.3f,%.3f,%.3f (est)\n" RESET, err(9 + 0), err(9 + 1),
                           err(9 + 2), gt_imu(11), gt_imu(12), gt_imu(13), state->imu->bias_g()(0), state->imu->bias_g()(1), state->imu->bias_g()(2));
                PRINT_INFO(REDPURPLE "e_bias_a = %.3f,%.3f,%.3f | %.3f,%.3f,%.3f (true) | %.3f,%.3f,%.3f (est)\n" RESET, err(12 + 0), err(12 + 1),
                           err(12 + 2), gt_imu(14), gt_imu(15), gt_imu(16), state->imu->bias_a()(0), state->imu->bias_a()(1), state->imu->bias_a()(2));
                
                // calculate normalized estimation error squared
                // the recovered error should be on the order of the state size (15 or 3 for marginals)
                Eigen::MatrixXd information = state->covariance().cast<double>().inverse();
                double nees_total = (err.transpose() * information * err)(0, 0);
                double nees_ori = (err.block(0, 0, 3, 1).transpose() * information.block(0, 0, 3, 3) * err.block(0, 0, 3, 1))(0, 0);
                double nees_pos = (err.block(3, 0, 3, 1).transpose() * information.block(3, 3, 3, 3) * err.block(3, 0, 3, 1))(0, 0);
                double nees_vel = (err.block(6, 0, 3, 1).transpose() * information.block(6, 6, 3, 3) * err.block(6, 0, 3, 1))(0, 0);
                double nees_bg = (err.block(9, 0, 3, 1).transpose() * information.block(9, 9, 3, 3) * err.block(9, 0, 3, 1))(0, 0);
                double nees_ba = (err.block(12, 0, 3, 1).transpose() * information.block(12, 12, 3, 3) * err.block(12, 0, 3, 1))(0, 0);
                PRINT_INFO(REDPURPLE "nees total = %.3f | ori = %.3f | pos = %.3f (ideal is 15 and 3)\n" RESET, nees_total, nees_ori, nees_pos);
                PRINT_INFO(REDPURPLE "nees vel = %.3f | bg = %.3f | ba = %.3f (ideal 3)\n" RESET, nees_vel, nees_bg, nees_ba);
                
#if ROS_AVAILABLE == 1
                // Align the groundtruth to the current estimate yaw
                // Only use the first frame so we can see the drift
                double oldestpose_time = -1;
                std::shared_ptr<ov_type::PoseJPL> oldestpose = nullptr;
                for (const auto& _pose : state->clones_IMU) {
                    if (oldestpose == nullptr || _pose.first < oldestpose_time) {
                        oldestpose_time = _pose.first;
                        oldestpose = _pose.second;
                    }
                }
                assert(oldestpose != nullptr);
            
                Vec4 q_es_0, q_gt_0;
                Vec3 p_es_0, p_gt_0;
                q_es_0 = oldestpose->quat();
                p_es_0 = oldestpose->pos();
                Eigen::Matrix<DataType, 17, 1> gt_imustate_0;
                bool success2 = sim.get_state(oldestpose_time + sim.get_true_parameters().calib_camimu_dt, gt_imustate_0);
                assert(success2);
                q_gt_0 = gt_imustate_0.block(1, 0, 4, 1);
                p_gt_0 = gt_imustate_0.block(5, 0, 3, 1);
                Mat3 R_ESTtoGT;
                Vec3 t_ESTinGT;
                align_posyaw_single(q_es_0, p_es_0, q_gt_0, p_gt_0, R_ESTtoGT, t_ESTinGT);

                // Pose states
                nav_msgs::Path arrEST, arrGT;
                arrEST.header.stamp = ros::Time::now();
                arrEST.header.frame_id = "global";
                arrGT.header.stamp = ros::Time::now();
                arrGT.header.frame_id = "global";
                for (const auto& _pose : state->clones_IMU) {
                    geometry_msgs::PoseStamped poseEST, poseGT;
                    poseEST.header.stamp = ros::Time(_pose.first);
                    poseEST.header.frame_id = "global";
                    poseEST.pose.orientation.x = _pose.second->quat()(0, 0);
                    poseEST.pose.orientation.y = _pose.second->quat()(1, 0);
                    poseEST.pose.orientation.z = _pose.second->quat()(2, 0);
                    poseEST.pose.orientation.w = _pose.second->quat()(3, 0);
                    poseEST.pose.position.x = _pose.second->pos()(0, 0);
                    poseEST.pose.position.y = _pose.second->pos()(1, 0);
                    poseEST.pose.position.z = _pose.second->pos()(2, 0);
                    
                    Eigen::Matrix<DataType, 17, 1> gt_imustate;
                    bool success3 = sim.get_state(_pose.first + sim.get_true_parameters().calib_camimu_dt, gt_imustate);
                    assert(success3);
                    gt_imustate.block(1, 0, 4, 1) = ov_core::quat_multiply(gt_imustate.block(1, 0, 4, 1), ov_core::rot_2_quat(R_ESTtoGT));
                    gt_imustate.block(5, 0, 3, 1) = R_ESTtoGT.transpose() * (gt_imustate.block(5, 0, 3, 1) - t_ESTinGT);
                    
                    poseGT.header.stamp = ros::Time(_pose.first);
                    poseGT.header.frame_id = "global";
                    poseGT.pose.orientation.x = gt_imustate(1);
                    poseGT.pose.orientation.y = gt_imustate(2);
                    poseGT.pose.orientation.z = gt_imustate(3);
                    poseGT.pose.orientation.w = gt_imustate(4);
                    poseGT.pose.position.x = gt_imustate(5);
                    poseGT.pose.position.y = gt_imustate(6);
                    poseGT.pose.position.z = gt_imustate(7);

                    arrEST.poses.push_back(poseEST);
                    arrGT.poses.push_back(poseGT);
                }
                pub_pathimu.publish(arrEST);
                pub_pathgt.publish(arrGT);

                // Features ESTIMATES pointcloud
                sensor_msgs::PointCloud point_cloud;
                point_cloud.header.frame_id = "global";
                point_cloud.header.stamp = ros::Time::now();
                for (const auto& feat : state->features_SLAM) {
                    geometry_msgs::Point32 pt;
                    pt.x = feat.second->get_xyz(false)(0, 0);
                    pt.y = feat.second->get_xyz(false)(1, 0);
                    pt.z = feat.second->get_xyz(false)(2, 0);
                    point_cloud.points.push_back(pt);
                }
                pub_loop_point.publish(point_cloud);

                // Features GROUNDTRUTH pointcloud
                sensor_msgs::PointCloud2 cloud;
                cloud.header.frame_id = "global";
                cloud.header.stamp = ros::Time::now();
                cloud.width = state->features_SLAM.size();
                cloud.height = 1;
                cloud.is_bigendian = false;
                cloud.is_dense = false; // there may be invalid points
                sensor_msgs::PointCloud2Modifier modifier(cloud);
                modifier.setPointCloud2FieldsByString(1, "xyz");
                modifier.resize(state->features_SLAM.size());
                sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
                sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
                sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");
                for (const auto& featpair : state->features_SLAM) {
                    // TrackSIM adds 1 to sim id if num_aruco is zero
                    Eigen::Vector3d feat = sim.get_map().at(featpair.first - 1);
                    feat = R_ESTtoGT.cast<double>().transpose() * (feat - t_ESTinGT.cast<double>());
                    *out_x = (float)feat(0);
                    ++out_x;
                    *out_y = (float)feat(1);
                    ++out_y;
                    *out_z = (float)feat(2);
                    ++out_z;
                }
                pub_points_sim.publish(cloud);
#endif  

                // Wait for user approval
                do {
                    std::cout << '\n' << "Press a key to continue...";
                } while (std::cin.get() != '\n');

                // Reset our tracker and simulator so we can try to init again
                if (params.init_options.sim_do_perturbation) {
                    sim.perturb_parameters(params.init_options);
                }

                tracker = std::make_shared<ov_core::TrackSIM>(params.init_options.camera_intrinsics, 0);
                propagator = std::make_shared<ov_srvins::Propagator>(params.imu_noises, params.gravity_mag);

                // Make the updater!
                updaterMSCKF = std::make_shared<ov_srvins::UpdaterMSCKF>(params.msckf_options,
                                                            params.featinit_options);
                updaterSLAM = std::make_shared<ov_srvins::UpdaterSLAM>(
                    params.slam_options, params.aruco_options, params.featinit_options);
                initializer = std::make_shared<ov_srvins::DynamicInitializer>(params.init_options, tracker->get_feature_database(), propagator, updaterMSCKF, updaterSLAM);
            } else {
                PRINT_ERROR(RED "[TEST]: failed to initialize with dynamic initializer!!\n" RESET);
            }
        }
    }
    return 0;
}