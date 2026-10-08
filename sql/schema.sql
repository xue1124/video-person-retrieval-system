-- 医保稽查系统完整数据库结构。
-- 新环境只需执行本文件；系统运行时只使用 medical_audit_v3 一个业务库。
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
  origin_label VARCHAR(50) NOT NULL DEFAULT '手动上传'
    COMMENT '前端展示用来源：手动上传/NVR流/NVR_ISAPI/RTSP实时',
  task_id VARCHAR(100) NULL COMMENT 'Celery 任务 ID',
  status VARCHAR(32) NOT NULL DEFAULT 'pending'
    COMMENT 'pending/processing/transcoding/completed/failed/streaming/stopped',
  failure_reason VARCHAR(512) NULL,
  progress INT NOT NULL DEFAULT 0,
  source_path VARCHAR(1024) NULL,
  is_media_source TINYINT(1) NOT NULL DEFAULT 0
    COMMENT '1=媒体源列表中的任务视频；0=仅分析结果记录',
  captured_at DATETIME(3) NULL COMMENT '视频实际拍摄开始时间',
  fps DECIMAL(10,4) NULL,
  duration_sec DECIMAL(14,3) NULL,
  duration VARCHAR(32) NULL COMMENT '展示用时长 HH:MM:SS 或直播文案',
  target_count INT NOT NULL DEFAULT 0,
  completed_at DATETIME(3) NULL,
  processing_started_at DATETIME(3) NULL,
  width INT UNSIGNED NULL,
  height INT UNSIGNED NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
    ON UPDATE CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uq_videos_source_key (source_key),
  UNIQUE KEY uq_videos_file_name (file_name),
  KEY idx_videos_camera_time (camera_id, captured_at),
  KEY idx_videos_media (is_media_source, created_at),
  KEY idx_videos_task_id (task_id),
  KEY idx_videos_status (status),
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
  local_track_no INT UNSIGNED NOT NULL COMMENT '当前分析结果中的局部轨迹编号',
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
  UNIQUE KEY uq_rooms_video_name (video_id, name),
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
  pre_merge_id INT NULL COMMENT '在线分配 ID（二次合并前）',
  assign_score DECIMAL(7,6) NULL COMMENT '检测框并入当前簇的相似度',
  merge_score DECIMAL(7,6) NULL COMMENT '二次聚类合并时的相似度',
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

CREATE TABLE IF NOT EXISTS users (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  username VARCHAR(64) NOT NULL COMMENT '登录名',
  password_hash VARCHAR(255) NOT NULL COMMENT 'bcrypt 哈希',
  role VARCHAR(20) NOT NULL DEFAULT 'user' COMMENT 'admin / user',
  is_active TINYINT(1) NOT NULL DEFAULT 1,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='系统用户';

CREATE TABLE IF NOT EXISTS search_logs (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  user_id INT UNSIGNED NULL COMMENT '发起检索的用户',
  query_image_paths TEXT NULL COMMENT '查询图片路径，多个文件名以逗号分隔',
  search_query VARCHAR(500) NULL COMMENT '文字检索描述',
  algorithm VARCHAR(50) NOT NULL COMMENT 'OSNet / SigLIP',
  threshold FLOAT NOT NULL DEFAULT 0.82,
  results_json LONGTEXT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_search_logs_created (created_at),
  KEY idx_search_logs_algorithm (algorithm),
  KEY idx_search_logs_user (user_id),
  CONSTRAINT fk_search_logs_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='轨迹检索历史日志';

CREATE TABLE IF NOT EXISTS gallery_meta (
  id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  video_id BIGINT UNSIGNED NULL,
  video_name VARCHAR(255) NOT NULL,
  timestamp FLOAT NOT NULL COMMENT '视频时间戳（秒）',
  image_path VARCHAR(500) NOT NULL,
  bbox_x1 INT NULL,
  bbox_y1 INT NULL,
  bbox_x2 INT NULL,
  bbox_y2 INT NULL,
  feature_vector LONGBLOB NULL COMMENT 'RTSP 实时流产生的 OSNet 特征',
  clip_feature LONGBLOB NULL COMMENT 'RTSP 实时流产生的 SigLIP 特征',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_gallery_video_name (video_name),
  KEY idx_gallery_timestamp (timestamp),
  KEY idx_gallery_video_id (video_id),
  CONSTRAINT fk_gallery_meta_video
    FOREIGN KEY (video_id) REFERENCES videos(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='RTSP 实时分析的临时截图和向量；历史视频检索使用 observations';

CREATE TABLE IF NOT EXISTS person_snapshots (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  global_person_id BIGINT UNSIGNED NULL,
  video_person_id BIGINT UNSIGNED NOT NULL,
  track_id BIGINT UNSIGNED NULL,
  track_point_id BIGINT UNSIGNED NULL,
  stay_segment_id BIGINT UNSIGNED NULL,
  video_id BIGINT UNSIGNED NOT NULL,
  image_path VARCHAR(1024) NOT NULL COMMENT '相对 tracking_snapshots/ 的路径',
  timestamp_sec DECIMAL(14,3) NOT NULL,
  quality_score DECIMAL(12,6) NOT NULL DEFAULT 0,
  snapshot_type ENUM('candidate', 'stay') NOT NULL,
  is_primary BOOLEAN NOT NULL DEFAULT FALSE COMMENT '是否作为该人物的主头像',
  bbox_x1 DECIMAL(10,2) NULL,
  bbox_y1 DECIMAL(10,2) NULL,
  bbox_x2 DECIMAL(10,2) NULL,
  bbox_y2 DECIMAL(10,2) NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  KEY idx_snapshots_global_primary (global_person_id, is_primary),
  KEY idx_snapshots_video_person (video_person_id, snapshot_type),
  KEY idx_snapshots_stay (stay_segment_id),
  KEY idx_snapshots_video (video_id),
  CONSTRAINT fk_snapshots_global
    FOREIGN KEY (global_person_id) REFERENCES global_people(id)
    ON DELETE SET NULL,
  CONSTRAINT fk_snapshots_video_person
    FOREIGN KEY (video_person_id) REFERENCES video_people(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_snapshots_track
    FOREIGN KEY (track_id) REFERENCES tracks(id)
    ON DELETE SET NULL,
  CONSTRAINT fk_snapshots_track_point
    FOREIGN KEY (track_point_id) REFERENCES track_points(id)
    ON DELETE SET NULL,
  CONSTRAINT fk_snapshots_stay
    FOREIGN KEY (stay_segment_id) REFERENCES stay_segments(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_snapshots_video
    FOREIGN KEY (video_id) REFERENCES videos(id)
    ON DELETE CASCADE
) ENGINE=InnoDB COMMENT='人物候选图与停留段现场图';

CREATE TABLE IF NOT EXISTS observations (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  video_id BIGINT UNSIGNED NOT NULL,
  processing_run_id BIGINT UNSIGNED NOT NULL,
  video_person_id BIGINT UNSIGNED NOT NULL,
  global_person_id BIGINT UNSIGNED NULL COMMENT '跨视频匹配后回写',
  track_id BIGINT UNSIGNED NOT NULL,
  track_point_id BIGINT UNSIGNED NULL,
  timestamp_sec DECIMAL(14,3) NOT NULL,
  frame_index BIGINT UNSIGNED NOT NULL,
  bbox_x1 DECIMAL(10,2) NOT NULL,
  bbox_y1 DECIMAL(10,2) NOT NULL,
  bbox_x2 DECIMAL(10,2) NOT NULL,
  bbox_y2 DECIMAL(10,2) NOT NULL,
  room_id BIGINT UNSIGNED NULL,
  crop_path VARCHAR(1024) NOT NULL COMMENT '相对 TRACKING_SNAPSHOT_ROOT 的路径',
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uq_obs_run_track_frame (processing_run_id, track_id, frame_index),
  KEY idx_obs_video_time (video_id, timestamp_sec),
  KEY idx_obs_global (global_person_id),
  KEY idx_obs_video_person (video_person_id),
  KEY idx_obs_track (track_id),
  KEY idx_obs_room_time (room_id, timestamp_sec),
  KEY idx_obs_track_point (track_point_id),
  CONSTRAINT fk_obs_video
    FOREIGN KEY (video_id) REFERENCES videos(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_obs_run
    FOREIGN KEY (processing_run_id) REFERENCES processing_runs(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_obs_video_person
    FOREIGN KEY (video_person_id) REFERENCES video_people(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_obs_global
    FOREIGN KEY (global_person_id) REFERENCES global_people(id)
    ON DELETE SET NULL,
  CONSTRAINT fk_obs_track
    FOREIGN KEY (track_id) REFERENCES tracks(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_obs_track_point
    FOREIGN KEY (track_point_id) REFERENCES track_points(id)
    ON DELETE SET NULL,
  CONSTRAINT fk_obs_room
    FOREIGN KEY (room_id) REFERENCES rooms(id)
    ON DELETE SET NULL,
  CONSTRAINT chk_obs_bbox CHECK (bbox_x2 >= bbox_x1 AND bbox_y2 >= bbox_y1)
) ENGINE=InnoDB COMMENT='行人检测框观测事实；历史视频检索以本表为准';

CREATE TABLE IF NOT EXISTS observation_embeddings (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  observation_id BIGINT UNSIGNED NOT NULL,
  modality ENUM('osnet', 'siglip_image') NOT NULL,
  embedding BLOB NOT NULL COMMENT 'float32 L2 归一化向量',
  model_name VARCHAR(255) NOT NULL,
  model_version VARCHAR(100) NULL,
  embedding_dim SMALLINT UNSIGNED NOT NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  PRIMARY KEY (id),
  UNIQUE KEY uq_obs_emb_modality (observation_id, modality),
  KEY idx_obs_emb_modality_dim (modality, embedding_dim),
  CONSTRAINT fk_obs_emb_observation
    FOREIGN KEY (observation_id) REFERENCES observations(id)
    ON DELETE CASCADE
) ENGINE=InnoDB COMMENT='观测对应的 OSNet 与 SigLIP 特征向量';

CREATE TABLE IF NOT EXISTS analysis_reports (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  created_by INT UNSIGNED NOT NULL,
  report_type VARCHAR(64) NOT NULL DEFAULT 'daily_activity_report',
  report_date DATE NULL,
  timezone VARCHAR(64) NOT NULL DEFAULT 'Asia/Shanghai',
  scope_json JSON NOT NULL,
  report_data_json LONGTEXT NOT NULL,
  title VARCHAR(500) NULL,
  report_markdown LONGTEXT NULL,
  warnings_json JSON NULL,
  dify_workflow_run_id VARCHAR(128) NULL,
  dify_output_json JSON NULL,
  status ENUM('generating', 'completed', 'failed') NOT NULL DEFAULT 'generating',
  error_message VARCHAR(1000) NULL,
  created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
    ON UPDATE CURRENT_TIMESTAMP(3),
  deleted_at DATETIME(3) NULL COMMENT '软删除时间（UTC naive）',
  deleted_by INT UNSIGNED NULL COMMENT '删除人 users.id',
  PRIMARY KEY (id),
  KEY idx_analysis_reports_user_created (created_by, created_at),
  KEY idx_analysis_reports_status (status),
  KEY idx_analysis_reports_date (report_date),
  KEY idx_analysis_reports_created (created_at),
  KEY idx_analysis_reports_deleted (deleted_at),
  CONSTRAINT fk_analysis_reports_user
    FOREIGN KEY (created_by) REFERENCES users(id)
    ON DELETE RESTRICT,
  CONSTRAINT fk_analysis_reports_deleted_by
    FOREIGN KEY (deleted_by) REFERENCES users(id)
    ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='AI 分析报告';
