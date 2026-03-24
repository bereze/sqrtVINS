/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Copyright (C) 2025-2026 Yuxiang Peng
 * Copyright (C) 2025-2026 Chuchu Chen
 * Copyright (C) 2025-2026 Kejian Wu
 * Copyright (C) 2018-2026 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program. If not, see
 * <https://www.gnu.org/licenses/>.
 */





#include "Feature.h"

using namespace ov_core;

void Feature::clean_old_measurements(const std::vector<double> &valid_times) {

  // Loop through each of the cameras we have
  for (auto const &pair : timestamps) {

    // Assert that we have all the parts of a measurement
    assert(timestamps[pair.first].size() == uvs[pair.first].size());
    assert(timestamps[pair.first].size() == uvs_norm[pair.first].size());

    // Our iterators
    auto it1 = timestamps[pair.first].begin();
    auto it2 = uvs[pair.first].begin();
    auto it3 = uvs_norm[pair.first].begin();

    // Loop through measurement times, remove ones that are not in our
    // timestamps
    while (it1 != timestamps[pair.first].end()) {
      if (std::find(valid_times.begin(), valid_times.end(), *it1) ==
          valid_times.end()) {
        it1 = timestamps[pair.first].erase(it1);
        it2 = uvs[pair.first].erase(it2);
        it3 = uvs_norm[pair.first].erase(it3);
      } else {
        ++it1;
        ++it2;
        ++it3;
      }
    }
  }
}

void Feature::clean_invalid_measurements(
    const std::vector<double> &invalid_times) {

  // Loop through each of the cameras we have
  for (auto const &pair : timestamps) {

    // Assert that we have all the parts of a measurement
    assert(timestamps[pair.first].size() == uvs[pair.first].size());
    assert(timestamps[pair.first].size() == uvs_norm[pair.first].size());

    // Our iterators
    auto it1 = timestamps[pair.first].begin();
    auto it2 = uvs[pair.first].begin();
    auto it3 = uvs_norm[pair.first].begin();

    // Loop through measurement times, remove ones that are in our timestamps
    while (it1 != timestamps[pair.first].end()) {
      if (std::find(invalid_times.begin(), invalid_times.end(), *it1) !=
          invalid_times.end()) {
        it1 = timestamps[pair.first].erase(it1);
        it2 = uvs[pair.first].erase(it2);
        it3 = uvs_norm[pair.first].erase(it3);
      } else {
        ++it1;
        ++it2;
        ++it3;
      }
    }
  }
}

void Feature::clean_older_measurements(double timestamp) {

  // Loop through each of the cameras we have
  for (auto const &pair : timestamps) {

    // Assert that we have all the parts of a measurement
    assert(timestamps[pair.first].size() == uvs[pair.first].size());
    assert(timestamps[pair.first].size() == uvs_norm[pair.first].size());

    // Our iterators
    auto it1 = timestamps[pair.first].begin();
    auto it2 = uvs[pair.first].begin();
    auto it3 = uvs_norm[pair.first].begin();

    // Loop through measurement times, remove ones that are older then the
    // specified one
    while (it1 != timestamps[pair.first].end()) {
      if (*it1 <= timestamp) {
        it1 = timestamps[pair.first].erase(it1);
        it2 = uvs[pair.first].erase(it2);
        it3 = uvs_norm[pair.first].erase(it3);
      } else {
        ++it1;
        ++it2;
        ++it3;
      }
    }
  }
}

float Feature::calc_parallax(const std::vector<Mat3> &cam_orients_history) {
  float parallax = -1.f;
  auto orient = [](const Vec2& pt, Vec3& e) {
    float phi = std::atan2(pt[1], std::sqrt(std::pow(pt[0], 2) + 1));
    float psi = std::atan2(pt[0], 1);
    e << std::cos(phi) * std::sin(psi), std::sin(phi), std::cos(phi) * std::cos(psi);
  };
  assert(!timestamps.empty());
  for (const auto& campair : timestamps) {
    assert(campair.second.size() >= 2);
    int idk = cam_orients_history.size() - 1;
    int id0 = std::max((int)(cam_orients_history.size() - uvs_norm[campair.first].size()), 0); // some feat has meas more than max clones while not update yet
    Vec3 e0, ek;
    orient(uvs_norm[campair.first].front(), e0);
    orient(uvs_norm[campair.first].back(), ek);
    Mat3 mRk0 = cam_orients_history[idk].transpose() * cam_orients_history[id0];
    float theta = std::fabs(std::acos(ek.dot(mRk0 * e0)));
    parallax = std::max(parallax, (float)(40.f * std::sin(theta) > 1 ? theta * 180 / M_PI : 0));
  }
  return parallax;
}