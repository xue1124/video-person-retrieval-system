-- 第二阶段：行人观测事实 + 双特征（OSNet / SigLIP）
-- 幂等：CREATE TABLE IF NOT EXISTS。身份合并仍只使用 OSNet，本表不参与聚类。
USE medical_audit_v3;

CREATE TABLE IF NOT EXISTS observations (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  video_id BIGINT UNSIGNED NOT NULL,
  processing_run_id BIGINT UNSIGNED NOT NULL,
  video_person_id BIGINT UNSIGNED NOT NULL,
  global_person_id BIGINT UNSIGNED NULL COMMENT '跨视频匹配后回写；检索时以本列为准',
  track_id BIGINT UNSIGNED NOT NULL,
  track_point_id BIGINT UNSIGNED NULL,
  timestamp_sec DECIMAL(14,3) NOT NULL,
  frame_index BIGINT UNSIGNED NOT NULL,
  bbox_x1 DECIMAL(10,2) NOT NULL,
  bbox_y1 DECIMAL(10,2) NOT NULL,
  bbox_x2 DECIMAL(10,2) NOT NULL,
  bbox_y2 DECIMAL(10,2) NOT NULL,
  room_id BIGINT UNSIGNED NULL,
  crop_path VARCHAR(1024) NOT NULL COMMENT '相对 TRACKING_SNAPSHOT_ROOT',
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
) ENGINE=InnoDB COMMENT='行人检测框观测事实；检索底库以本表为准';

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
) ENGINE=InnoDB COMMENT='观测向量。OSNet 仅用于检索图搜人，不回写身份合并';
