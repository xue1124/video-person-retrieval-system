-- 人物代表图 / 停留现场图（磁盘路径 + 元数据）
USE medical_audit_v3;

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
  is_primary BOOLEAN NOT NULL DEFAULT FALSE COMMENT '是否作为该 G 的主头像',
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
