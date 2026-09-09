-- 全新业务库：不修改 siglip_v2，也不使用 MySQL 系统库 sys。
CREATE DATABASE IF NOT EXISTS medical_audit_v3
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE medical_audit_v3;

CREATE TABLE IF NOT EXISTS cameras (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  camera_code VARCHAR(100) NOT NULL COMMENT '前端或设备侧使用的稳定编号',
  name VARCHAR(255) NOT NULL,
  channel_no VARCHAR(100) NULL COMMENT 'ISAPI/RTSP 通道号',
  location VARCHAR(500) NULL,
  source_type ENUM('isapi', 'rtsp', 'other') NOT NULL DEFAULT 'other',
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
    ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uq_cameras_code (camera_code)
) ENGINE=InnoDB COMMENT='物理摄像头或固定视频通道';

CREATE TABLE IF NOT EXISTS videos (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  source_key CHAR(64) NOT NULL COMMENT '来源路径的 SHA-256，用于避免重复登记',
  camera_id BIGINT UNSIGNED NULL COMMENT '本地未知来源视频允许为空',
  file_name VARCHAR(255) NOT NULL,
  source_type ENUM('upload', 'isapi', 'rtsp', 'other') NOT NULL DEFAULT 'upload',
  source_path VARCHAR(1024) NULL,
  captured_at DATETIME(3) NULL COMMENT '视频实际拍摄开始时间',
  fps DECIMAL(10,4) NULL,
  duration_sec DECIMAL(14,3) NULL,
  width INT UNSIGNED NULL,
  height INT UNSIGNED NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uq_videos_source_key (source_key),
  KEY idx_videos_camera_time (camera_id, captured_at),
  CONSTRAINT fk_videos_camera
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
    ON DELETE SET NULL
) ENGINE=InnoDB COMMENT='原始视频及其真实来源';

CREATE TABLE IF NOT EXISTS processing_runs (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  run_key CHAR(64) NOT NULL COMMENT '输入结果文件的 SHA-256，保证幂等导入',
  video_id BIGINT UNSIGNED NOT NULL,
  detector_model VARCHAR(255) NOT NULL,
  tracker_name VARCHAR(100) NOT NULL,
  reid_model VARCHAR(255) NOT NULL,
  process_fps DECIMAL(10,4) NULL,
  parameters_json JSON NULL,
  status ENUM('pending', 'processing', 'completed', 'failed') NOT NULL
    DEFAULT 'pending',
  started_at DATETIME(3) NULL,
  completed_at DATETIME(3) NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uq_processing_runs_key (run_key),
  KEY idx_processing_runs_video (video_id, created_at),
  CONSTRAINT fk_processing_runs_video
    FOREIGN KEY (video_id) REFERENCES videos(id)
    ON DELETE CASCADE
) ENGINE=InnoDB COMMENT='同一视频每执行一次算法产生一条记录';

CREATE TABLE IF NOT EXISTS global_people (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  status ENUM('candidate', 'confirmed', 'rejected') NOT NULL DEFAULT 'candidate',
  representative_embedding BLOB NULL COMMENT '跨视频聚合后的 float32 OSNet 特征',
  embedding_model VARCHAR(255) NULL,
  embedding_dim SMALLINT UNSIGNED NULL,
  sample_count INT UNSIGNED NOT NULL DEFAULT 0,
  first_seen_at DATETIME(3) NULL,
  last_seen_at DATETIME(3) NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
    ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  KEY idx_global_people_seen (first_seen_at, last_seen_at)
) ENGINE=InnoDB COMMENT='跨视频汇总后的候选人物 G';

CREATE TABLE IF NOT EXISTS video_people (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  processing_run_id BIGINT UNSIGNED NOT NULL,
  global_person_id BIGINT UNSIGNED NULL COMMENT '尚未跨视频关联时允许为空',
  local_person_no INT UNSIGNED NOT NULL COMMENT '当前视频结果中的 P 编号',
  start_sec DECIMAL(14,3) NOT NULL,
  end_sec DECIMAL(14,3) NOT NULL,
  representative_embedding BLOB NULL COMMENT '当前视频内成熟人物模板',
  embedding_model VARCHAR(255) NULL,
  embedding_dim SMALLINT UNSIGNED NULL,
  profile_updates INT UNSIGNED NOT NULL DEFAULT 0,
  assignment_method ENUM('new', 'auto', 'manual') NOT NULL DEFAULT 'new',
  assignment_score DECIMAL(7,6) NULL,
  is_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uq_video_people_run_local (processing_run_id, local_person_no),
  KEY idx_video_people_global (global_person_id),
  CONSTRAINT fk_video_people_run
    FOREIGN KEY (processing_run_id) REFERENCES processing_runs(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_video_people_global
    FOREIGN KEY (global_person_id) REFERENCES global_people(id)
    ON DELETE SET NULL,
  CONSTRAINT chk_video_people_time CHECK (end_sec >= start_sec)
) ENGINE=InnoDB COMMENT='单次视频分析中的人物 P';

CREATE TABLE IF NOT EXISTS person_identity_assignments (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  video_person_id BIGINT UNSIGNED NOT NULL,
  global_person_id BIGINT UNSIGNED NOT NULL,
  method ENUM('new', 'auto', 'manual', 'merge', 'split') NOT NULL,
  score DECIMAL(7,6) NULL,
  note VARCHAR(1000) NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  KEY idx_identity_assignments_video_person (video_person_id, created_at),
  KEY idx_identity_assignments_global (global_person_id, created_at),
  CONSTRAINT fk_identity_assignments_video_person
    FOREIGN KEY (video_person_id) REFERENCES video_people(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_identity_assignments_global
    FOREIGN KEY (global_person_id) REFERENCES global_people(id)
    ON DELETE RESTRICT
) ENGINE=InnoDB COMMENT='跨视频人物归属的可审计历史';

CREATE TABLE IF NOT EXISTS tracks (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  processing_run_id BIGINT UNSIGNED NOT NULL,
  video_person_id BIGINT UNSIGNED NOT NULL,
  local_track_no INT UNSIGNED NOT NULL COMMENT 'BoT-SORT 的 T 编号',
  start_sec DECIMAL(14,3) NOT NULL,
  end_sec DECIMAL(14,3) NOT NULL,
  observation_count INT UNSIGNED NOT NULL,
  representative_embedding BLOB NULL,
  embedding_model VARCHAR(255) NULL,
  embedding_dim SMALLINT UNSIGNED NULL,
  online_match_score DECIMAL(7,6) NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uq_tracks_run_local (processing_run_id, local_track_no),
  KEY idx_tracks_video_person_time (video_person_id, start_sec, end_sec),
  CONSTRAINT fk_tracks_run
    FOREIGN KEY (processing_run_id) REFERENCES processing_runs(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_tracks_video_person
    FOREIGN KEY (video_person_id) REFERENCES video_people(id)
    ON DELETE CASCADE,
  CONSTRAINT chk_tracks_time CHECK (end_sec >= start_sec)
) ENGINE=InnoDB COMMENT='单段连续本地轨迹 T';

CREATE TABLE IF NOT EXISTS rooms (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  camera_id BIGINT UNSIGNED NULL,
  video_id BIGINT UNSIGNED NULL COMMENT '仅对某个上传视频有效时使用',
  name VARCHAR(255) NOT NULL,
  polygon_json JSON NOT NULL COMMENT '建议保存 0~1 归一化坐标',
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
    ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  KEY idx_rooms_camera (camera_id),
  KEY idx_rooms_video (video_id),
  CONSTRAINT fk_rooms_camera
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_rooms_video
    FOREIGN KEY (video_id) REFERENCES videos(id)
    ON DELETE CASCADE,
  CONSTRAINT chk_rooms_scope CHECK (camera_id IS NOT NULL OR video_id IS NOT NULL)
) ENGINE=InnoDB COMMENT='摄像头或视频画面中的房间区域';

CREATE TABLE IF NOT EXISTS track_points (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  track_id BIGINT UNSIGNED NOT NULL,
  room_id BIGINT UNSIGNED NULL,
  frame_index BIGINT UNSIGNED NOT NULL,
  timestamp_sec DECIMAL(14,3) NOT NULL,
  occurred_at DATETIME(3) NULL COMMENT 'captured_at + timestamp_sec',
  bbox_x1 DECIMAL(10,2) NOT NULL,
  bbox_y1 DECIMAL(10,2) NOT NULL,
  bbox_x2 DECIMAL(10,2) NOT NULL,
  bbox_y2 DECIMAL(10,2) NOT NULL,
  foot_x DECIMAL(10,2) NULL,
  foot_y DECIMAL(10,2) NULL,
  confidence DECIMAL(7,6) NOT NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uq_track_points_frame (track_id, frame_index),
  KEY idx_track_points_time (track_id, timestamp_sec),
  KEY idx_track_points_room_time (room_id, occurred_at),
  CONSTRAINT fk_track_points_track
    FOREIGN KEY (track_id) REFERENCES tracks(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_track_points_room
    FOREIGN KEY (room_id) REFERENCES rooms(id)
    ON DELETE SET NULL,
  CONSTRAINT chk_track_points_bbox CHECK (
    bbox_x2 >= bbox_x1 AND bbox_y2 >= bbox_y1
  )
) ENGINE=InnoDB COMMENT='每条轨迹在每个采样时刻的位置事实';

CREATE TABLE IF NOT EXISTS stay_segments (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  video_person_id BIGINT UNSIGNED NOT NULL,
  room_id BIGINT UNSIGNED NULL,
  start_sec DECIMAL(14,3) NOT NULL,
  end_sec DECIMAL(14,3) NOT NULL,
  duration_sec DECIMAL(14,3) NOT NULL,
  entered_at DATETIME(3) NULL,
  exited_at DATETIME(3) NULL,
  point_count INT UNSIGNED NOT NULL DEFAULT 0,
  algorithm_version VARCHAR(100) NOT NULL DEFAULT 'v1',
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  KEY idx_stay_person_time (video_person_id, start_sec, end_sec),
  KEY idx_stay_room_time (room_id, entered_at, exited_at),
  CONSTRAINT fk_stay_video_person
    FOREIGN KEY (video_person_id) REFERENCES video_people(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_stay_room
    FOREIGN KEY (room_id) REFERENCES rooms(id)
    ON DELETE SET NULL,
  CONSTRAINT chk_stay_time CHECK (
    end_sec >= start_sec AND duration_sec >= 0
  )
) ENGINE=InnoDB COMMENT='由轨迹点计算得到的房间停留片段';
